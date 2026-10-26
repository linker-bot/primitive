#!/usr/bin/env python3
"""Build and verify a standalone Jazzy installation in a new external directory.

Uses Ubuntu's Python and numeric libraries, creates an isolated Web environment,
and starts only a dual-hand Mock panel. No ROS node or hardware driver is started.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

SYSTEM_PYTHON = "/usr/bin/python3"
NUMERIC_PROBE = """
import json, pathlib, sys
import numpy, scipy, yaml, filelock
result = {'python': sys.version_info[:2], 'numeric': {}}
for module in (numpy, scipy):
    path = pathlib.Path(module.__file__).resolve()
    if not path.is_relative_to('/usr/lib/python3/dist-packages'):
        raise RuntimeError(f'{module.__name__} must come from Ubuntu apt, got {path}')
    result['numeric'][module.__name__] = {'version': module.__version__, 'path': str(path)}
print(json.dumps(result))
"""
WEB_REQUIREMENTS = [
    "setuptools>=80,<81",
    "wheel",
    "pydantic>=2,<3",
    "fastapi<1",
    "uvicorn[standard]",
    "websockets>=12",
]
PACKAGES = [
    "hand_force_msg",
    "robot_perception_msgs",
    "hand_gesture_primitives_msgs",
    "hand_gesture_primitives",
]


def capture(command, environment):
    return subprocess.check_output(command, env=environment, text=True).strip()


def sourced_environment(setup, environment):
    return json.loads(
        capture(
            [
                "bash",
                "--noprofile",
                "--norc",
                "-c",
                'set -e; source "$1" >&2; exec /usr/bin/python3 -c "import json,os; print(json.dumps(dict(os.environ)))"',
                "source-ros-environment",
                str(setup),
            ],
            environment,
        )
    )


def run_logged(command, environment, log):
    print("Running:", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(
            command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True
        )


def mock_probe(environment, output):
    environment = {
        **environment,
        "HAND_GESTURE_RUNTIME_STATUS_PATH": str(output / "mock-runtime.json"),
        "ROS_DOMAIN_ID": "187",
        "ROS_LOCALHOST_ONLY": "1",
    }
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        "/opt/ros/jazzy/bin/ros2",
        "run",
        "hand_gesture_primitives",
        "teach_panel",
        "--mock",
        "--pose-dir",
        str(output / "mock-poses"),
        "--hand",
        "left:o6",
        "--hand",
        "right:o6",
        "--port",
        str(port),
    ]
    with (output / "mock.log").open("w") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            deadline = time.monotonic() + 30
            while time.monotonic() < deadline:
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Mock panel exited {process.returncode}; see {output / 'mock.log'}"
                    )
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/api/workbench/state", timeout=1
                    ) as response:
                        state = json.load(response)
                    if state.get("backend") != "mock" or set(state.get("hands", {})) != {
                        "left",
                        "right",
                    }:
                        raise RuntimeError(f"Unexpected Mock state: {state}")
                    return {"backend": state["backend"], "hands": sorted(state["hands"])}
                except (urllib.error.URLError, TimeoutError):
                    time.sleep(0.1)
            raise RuntimeError(f"Mock panel did not become ready; see {output / 'mock.log'}")
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)


def snapshot_sources(source, snapshot):
    """Copy independent build inputs, including the repository-owned FK models."""
    ignore = shutil.ignore_patterns(
        ".git",
        "__pycache__",
        "*.egg-info",
        ".pytest_cache",
        ".ruff_cache",
        "build",
        "install",
        "log",
        "dist",
    )
    for name in ("hand_gesture_primitives", "messages", "data/urdf"):
        shutil.copytree(source / name, snapshot / name, ignore=ignore)
    return {
        path.relative_to(snapshot).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(snapshot.rglob("*"))
        if path.is_file()
    }


def install(args):
    source = args.source.resolve()
    output = args.output.absolute()
    if output.exists():
        raise FileExistsError(f"Use a new output directory: {output}")
    if output.resolve().is_relative_to(source):
        raise ValueError("Output must be outside the source checkout")
    for package in PACKAGES:
        path = source / (
            package if package == "hand_gesture_primitives" else "messages/" + package
        )
        if not (path / "package.xml").is_file():
            raise ValueError(f"Missing source package: {path}")
    if not (source / "data/urdf/o6/right/linkerhand_o6_right.urdf").is_file():
        raise ValueError("Source checkout is missing the bundled O6 right URDF")
    environment = dict(os.environ)
    for name in (
        "PYTHONPATH",
        "PYTHONHOME",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "CMAKE_PREFIX_PATH",
        "LD_LIBRARY_PATH",
        "PIP_TARGET",
        "PIP_PYTHON",
        "PIP_ROOT",
        "PIP_PREFIX",
        "PIP_USER",
        "PIP_IGNORE_INSTALLED",
    ):
        environment.pop(name, None)
    environment.update(PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    environment["PATH"] = "/usr/bin:/bin:/usr/sbin:/sbin"
    environment = sourced_environment(Path("/opt/ros/jazzy/setup.bash"), environment)
    numeric = json.loads(capture([SYSTEM_PYTHON, "-c", NUMERIC_PROBE], environment))
    if numeric["python"] != [3, 12]:
        raise ValueError(
            "This installation requires Ubuntu 24.04 / ROS Jazzy / system Python 3.12"
        )
    output.mkdir(parents=True)
    constraints = output / "system-numeric-constraints.txt"
    constraints.write_text(
        "".join(f"{name}=={item['version']}\n" for name, item in numeric["numeric"].items())
    )
    venv = output / "venv"
    run_logged(
        [SYSTEM_PYTHON, "-m", "venv", "--system-site-packages", str(venv)],
        environment,
        output / "venv.log",
    )
    python = str(venv / "bin/python")
    environment["PATH"] = str(venv / "bin") + os.pathsep + environment["PATH"]
    environment["VIRTUAL_ENV"] = str(venv)
    environment["PIP_CONFIG_FILE"] = os.devnull
    environment["PIP_REQUIRE_VIRTUALENV"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    pip = [
        python,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--upgrade-strategy",
        "only-if-needed",
        "--constraint",
        str(constraints),
    ]
    if args.wheelhouse:
        pip += ["--no-index", "--find-links", str(args.wheelhouse.resolve())]
    run_logged(pip + WEB_REQUIREMENTS, environment, output / "pip.log")
    after = json.loads(capture([python, "-c", NUMERIC_PROBE], environment))
    if after != numeric:
        raise RuntimeError("The isolated installation changed Ubuntu's numeric module selection")

    snapshot = output / "source"
    source_hashes = snapshot_sources(source, snapshot)
    workspace = output / "workspace"
    command = [
        python,
        "/usr/bin/colcon",
        "--log-base",
        str(workspace / "log"),
        "build",
        "--base-paths",
        str(snapshot / "hand_gesture_primitives"),
        str(snapshot / "messages"),
        "--build-base",
        str(workspace / "build"),
        "--install-base",
        str(workspace / "install"),
        "--packages-select",
        *PACKAGES,
        "--cmake-args",
        f"-DPython3_EXECUTABLE={python}",
        f"-DPYTHON_EXECUTABLE={python}",
    ]
    run_logged(command, environment, output / "colcon.log")
    environment = sourced_environment(workspace / "install/setup.bash", environment)
    prefix = Path(
        capture(
            ["/opt/ros/jazzy/bin/ros2", "pkg", "prefix", "hand_gesture_primitives"], environment
        )
    )
    expected_prefix = workspace / "install/hand_gesture_primitives"
    if prefix != expected_prefix:
        raise RuntimeError(f"The overlay selects an unexpected package: {prefix}")
    scripts = {}
    for name in ("teach_panel", "gesture_node"):
        executable = prefix / "lib/hand_gesture_primitives" / name
        shebang = executable.read_text().splitlines()[0]
        if shebang != "#!" + python:
            raise RuntimeError(f"Wrong interpreter in {executable}: {shebang}")
        scripts[name] = {"path": str(executable), "shebang": shebang}
    versions = json.loads(
        capture(
            [
                python,
                "-c",
                """
import json, pydantic, fastapi, rclpy
from hand_gesture_primitives.teach_panel.dependencies import check_web_dependencies
from hand_gesture_primitives_msgs.srv import PlanGesture
from robot_perception_msgs.msg import LabeledBBox3DArray
from hand_force_msg.msg import HandForce
check_web_dependencies()
print(json.dumps({'pydantic': pydantic.__version__, 'fastapi': fastapi.__version__}))
""",
            ],
            environment,
        )
    )
    report = {
        "ok": True,
        "interpreter": python,
        "numeric_modules": numeric["numeric"],
        "web_versions": versions,
        "entry_points": scripts,
        "source_files_sha256": source_hashes,
        "mock_http": mock_probe(environment, output),
    }
    (output / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Validated installation: {output / 'acceptance.json'}")
    print(f"source {venv / 'bin/activate'}")
    print(f"source {workspace / 'install/setup.bash'}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory outside the source checkout"
    )
    parser.add_argument(
        "--wheelhouse",
        type=Path,
        help="Install Web/build dependencies from local wheels, without an index",
    )
    args = parser.parse_args()
    try:
        install(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"Standalone ROS installation failed: {exc}\n")


if __name__ == "__main__":
    main()
