import os
import shutil
from glob import glob
from pathlib import Path

from setuptools import find_namespace_packages, setup
from setuptools.command.sdist import sdist

package_name = "hand_gesture_primitives"
HERE = os.path.abspath(os.path.dirname(__file__))


def _rel(path: str) -> str:
    """colcon ament_python requires data_files paths relative to setup.py."""
    return os.path.relpath(path, HERE)


def _share_launch_files():
    """Install launch + rviz from package-root launch/ (robust to colcon cwd)."""
    launch_dir = os.path.join(HERE, "launch")
    py_files = sorted(glob(os.path.join(launch_dir, "*.launch.py")))
    rviz_files = sorted(glob(os.path.join(launch_dir, "*.rviz")))
    if not py_files:
        raise RuntimeError(
            f"No launch/*.launch.py under {launch_dir} — "
            "check package layout before building hand_gesture_primitives"
        )
    return [_rel(f) for f in py_files + rviz_files]


def _share_config_files():
    config_dir = os.path.join(HERE, "hand_gesture_primitives", "config")
    return [_rel(f) for f in sorted(glob(os.path.join(config_dir, "*.yaml")))]


def _urdf_root():
    # The checkout owns one canonical model tree. A source distribution stages
    # that tree inside its own project root so no archive entry escapes it.
    staged = Path(HERE) / "data" / "urdf"
    return staged if staged.is_dir() else Path(HERE).parent / "data" / "urdf"


def _share_urdf_files():
    root = _urdf_root()
    if not (root / "o6/right/linkerhand_o6_right.urdf").is_file():
        raise RuntimeError(f"Bundled O6 right model missing: {root}")
    groups = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"Bundled model must not be a symlink: {path}")
        if path.is_file():
            target = Path("share") / package_name / "urdf" / path.relative_to(root).parent
            groups.setdefault(str(target), []).append(_rel(str(path)))
    return list(groups.items())


class ModelSdist(sdist):
    """Include repository-level models without parent paths in the sdist."""

    def make_release_tree(self, base_dir, files):
        safe = [name for name in files
                if not Path(name).is_absolute() and ".." not in Path(name).parts]
        super().make_release_tree(base_dir, safe)
        target = Path(base_dir) / "data" / "urdf"
        shutil.copytree(_urdf_root(), target, dirs_exist_ok=True)
        # The released manifest must describe paths within the released project,
        # including when a wheel (or another sdist) is built from this archive.
        sources = Path(base_dir) / self.get_finalized_command("egg_info").egg_info / "SOURCES.txt"
        entries = set(safe)
        entries.update(str(path.relative_to(base_dir)) for path in target.rglob("*") if path.is_file())
        # setuptools may hardlink release-tree files back into the checkout.
        if sources.exists():
            sources.unlink()
        sources.write_text("\n".join(sorted(entries)) + "\n", encoding="utf-8")


setup(
    name=package_name,
    version="0.2.2",
    # config/、teach_panel/static/ 无 __init__.py，属 PEP 420 命名空间包；
    # find_packages 会漏掉它们并触发 setuptools “Package would be ignored” 警告。
    packages=find_namespace_packages(
        include=["hand_gesture_primitives", "hand_gesture_primitives.*"],
        exclude=["test", "test.*"],
    ),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), _share_launch_files()),
        (os.path.join("share", package_name, "config"), _share_config_files()),
        *_share_urdf_files(),
    ],
    # FK 后端为 Pinocchio（fk_solver 惰性导入，缺失时 create_fk_solver 返回 None）
    install_requires=[
        "setuptools",
        "numpy",
        "scipy",
        "PyYAML>=6",
        "filelock>=3.12",
        "pydantic>=2,<3",
    ],
    cmdclass={"sdist": ModelSdist},
    zip_safe=True,
    maintainer="LinkerHand Team",
    maintainer_email="opensource@linkerbot.ai",
    description="LinkerHand 手势原语（O20/L25/O30/O6/L6/G20；内部 semantic_30）",
    # The implementation is MIT; bundled O6 model resources retain Apache-2.0.
    license="MIT AND Apache-2.0",
    license_files=["LICENSE", "LICENSE.apache-2.0", "NOTICE"],
    extras_require={
        "test": ["pytest", "httpx", "ruff"],
        "teach": ["fastapi<1", "uvicorn[standard]", "websockets>=12"],
        "fk": ["pin"],
    },
    package_data={
        "hand_gesture_primitives.config": ["*.yaml"],
        "hand_gesture_primitives.teach_panel": ["static/*"],
        "hand_gesture_primitives.teach_panel.static": ["*"],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "gesture_node = hand_gesture_primitives.gesture_node:main",
            "mock_perception = hand_gesture_primitives.mock_perception_node:main",
            "joint_state_bridge = hand_gesture_primitives.joint_state_bridge:main",
            "grasp_gate = hand_gesture_primitives.grasp_gate:main",
            "save_pose = hand_gesture_primitives.save_pose:main",
            "teach_panel = hand_gesture_primitives.teach_panel.main:main",
            "pose_library_server = hand_gesture_primitives.pose_library_server:main",
            "import_gui_presets = hand_gesture_primitives.import_gui_presets:main",
            "pub_gesture_cmd = hand_gesture_primitives.pub_gesture_cmd:main",
            "pose_bundle_export = hand_gesture_primitives.pose_bundle_cli:main_export",
            "pose_bundle_import = hand_gesture_primitives.pose_bundle_cli:main_import",
            "export_primitive_poses = hand_gesture_primitives.export_primitive_poses:main",
            "run_gesture_cmd_suite = hand_gesture_primitives.run_gesture_cmd_suite:main",
        ],
    },
)
