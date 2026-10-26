"""Exercise pip redirection using a disposable venv and a local, dependency-free wheel."""

import importlib.util
import json
import subprocess
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.skipif(not Path("/usr/bin/python3").is_file(), reason="Ubuntu system Python required")
def test_inherited_pip_routing_cannot_install_into_another_environment(monkeypatch, tmp_path):
    bootstrap = subprocess.run(
        ["/usr/bin/python3", "-c", "import ensurepip"],
        capture_output=True,
    )
    if bootstrap.returncode:
        pytest.skip("System python3-venv is required for the real pip isolation regression")
    root = Path(__file__).resolve().parents[2]
    script = root / "hand_gesture_primitives/scripts/validate_ros_install.py"
    spec = importlib.util.spec_from_file_location("gesture_install_probe", script)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)

    redirected = tmp_path / "redirected"
    interpreter = tmp_path / "foreign-python"
    interpreter.write_text(f'#!/bin/sh\ntouch "{redirected}"\nexit 97\n')
    interpreter.chmod(0o755)
    foreign_root, foreign_target = tmp_path / "foreign-root", tmp_path / "foreign-target"
    monkeypatch.setenv("PIP_PYTHON", str(interpreter))
    monkeypatch.setenv("PIP_ROOT", str(foreign_root))
    monkeypatch.setenv("PIP_TARGET", str(foreign_target))

    # Apt/ROS discovery is outside this test. pip and both interpreter boundaries are real.
    numeric = {
        "python": [3, 12],
        "numeric": {
            "numpy": {
                "version": "1.26.4",
                "path": "/usr/lib/python3/dist-packages/numpy/__init__.py",
            },
            "scipy": {
                "version": "1.11.4",
                "path": "/usr/lib/python3/dist-packages/scipy/__init__.py",
            },
        },
    }
    monkeypatch.setattr(helper, "sourced_environment", lambda setup, env: env)
    monkeypatch.setattr(helper, "capture", lambda command, env: json.dumps(numeric))

    wheel = tmp_path / "gesture_install_routing_probe-1.0-py3-none-any.whl"
    dist_info = "gesture_install_routing_probe-1.0.dist-info"
    members = {
        "gesture_install_routing_probe/__init__.py": "INSTALLED = True\n",
        f"{dist_info}/METADATA": "Metadata-Version: 2.1\nName: gesture-install-routing-probe\nVersion: 1.0\n",
        f"{dist_info}/WHEEL": "Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
    }
    members[f"{dist_info}/RECORD"] = "".join(
        f"{name},,\n" for name in [*members, f"{dist_info}/RECORD"]
    )
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, text in members.items():
            archive.writestr(name, text)

    class ProbeFinished(Exception):
        pass

    output = tmp_path / "installation"
    run = helper.run_logged

    def install_only_probe(command, environment, log):
        if command[1:4] == ["-m", "pip", "install"]:
            # Keep the installer's actual interpreter and environment, substituting only
            # a local package so the regression never accesses a package index.
            run(
                [command[0], "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
                environment,
                log,
            )
            installed = subprocess.check_output(
                [
                    command[0],
                    "-I",
                    "-c",
                    "import gesture_install_routing_probe as p; print(p.__file__)",
                ],
                env=environment,
                text=True,
            ).strip()
            assert Path(installed).is_relative_to(output / "venv")
            raise ProbeFinished
        run(command, environment, log)

    monkeypatch.setattr(helper, "run_logged", install_only_probe)
    with pytest.raises(ProbeFinished):
        helper.install(SimpleNamespace(source=root, output=output, wheelhouse=None))
    assert not redirected.exists()
    assert not foreign_root.exists()
    assert not foreign_target.exists()
