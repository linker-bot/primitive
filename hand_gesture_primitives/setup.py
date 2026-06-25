import os
from glob import glob
from setuptools import find_packages, setup

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
            "check package layout before building hand_gesture_primitives")
    return [_rel(f) for f in py_files + rviz_files]


def _share_config_files():
    config_dir = os.path.join(HERE, "hand_gesture_primitives", "config")
    return [_rel(f) for f in sorted(glob(os.path.join(config_dir, "*.yaml")))]


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), _share_launch_files()),
        (os.path.join("share", package_name, "config"), _share_config_files()),
    ],
    install_requires=["setuptools", "numpy", "scipy", "yourdfpy"],
    zip_safe=True,
    maintainer="LinkerHand Team",
    maintainer_email="opensource@linkerbot.ai",
    description="LinkerHand O20 手势原语控制节点",
    license="MIT",
    extras_require={
        "test": ["pytest"],
    },
    entry_points={
        "console_scripts": [
            "gesture_node = hand_gesture_primitives.gesture_node:main",
            "mock_perception = hand_gesture_primitives.mock_perception_node:main",
            "joint_state_bridge = hand_gesture_primitives.joint_state_bridge:main",
            "grasp_gate = hand_gesture_primitives.grasp_gate:main",
        ],
    },
)
