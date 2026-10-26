"""Launch left + right hand_gesture_primitives nodes (optional gates).

Example:
  ros2 launch hand_gesture_primitives bimanual_gesture.launch.py \\
    hand_type:=l25 launch_gate:=false

  # L6（需自备手控 / 配合 launch_l6_hand）
  ros2 launch hand_gesture_primitives bimanual_gesture.launch.py \\
    hand_type:=l6 launch_gate:=false

  # With gates (publish to /{side}_hand/gesture_cmd):
  ros2 launch hand_gesture_primitives bimanual_gesture.launch.py \\
    hand_type:=l25 launch_gate:=true

Per-side cmd topics are always auto
  (/left_hand/gesture_cmd[_exec] and /right_hand/...).
Override a single side via gesture_node.launch.py instead.
"""

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    LogInfo,
    SetEnvironmentVariable,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def _side_include(side: str):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("hand_gesture_primitives"),
                    "launch",
                    "gesture_node.launch.py",
                ]
            )
        ),
        launch_arguments={
            "hand_side": side,
            "hand_type": LaunchConfiguration("hand_type"),
            "hand_joint": LaunchConfiguration("hand_joint"),
            "state_topic": LaunchConfiguration("state_topic"),
            "cmd_backend": LaunchConfiguration("cmd_backend"),
            "ros2_control_command_units": LaunchConfiguration("ros2_control_command_units"),
            "robot_description": LaunchConfiguration("robot_description"),
            "g20_joint_order": LaunchConfiguration("g20_joint_order"),
            "control_rate": LaunchConfiguration("control_rate"),
            "primitive_speed": LaunchConfiguration("primitive_speed"),
            "pose_dir": LaunchConfiguration("pose_dir"),
            "launch_gate": LaunchConfiguration("launch_gate"),
            "bboxes_3d_topic": LaunchConfiguration("bboxes_3d_topic"),
            "labeled_pose_topic": LaunchConfiguration("labeled_pose_topic"),
            # Force empty so left/right never share one override topic.
            "cmd_topic": "",
            "tcp_pose_topic": "",
            "object_pose_topic": "",
        }.items(),
    )


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pose_dir",
                default_value=EnvironmentVariable("HAND_GESTURE_POSE_DIR", default_value=""),
                description="Explicit shared pose directory; no implicit HOME fallback",
            ),
            SetEnvironmentVariable("HAND_GESTURE_POSE_DIR", LaunchConfiguration("pose_dir")),
            DeclareLaunchArgument(
                "hand_type",
                default_value="l25",
                description="Hand model for both sides: o20 / l25 / o6 / l6",
            ),
            DeclareLaunchArgument(
                "hand_joint", default_value="", description="Optional alias overriding hand_type"
            ),
            DeclareLaunchArgument("primitive_speed", default_value="1.0"),
            DeclareLaunchArgument("state_topic", default_value="/joint_states"),
            DeclareLaunchArgument("cmd_backend", default_value="ros2_control"),
            DeclareLaunchArgument(
                "ros2_control_command_units",
                default_value="radians",
                description="radians=URDF弧度（mock/sim/real 默认）；hardware=遗留SDK 0~255",
            ),
            DeclareLaunchArgument("robot_description", default_value=""),
            DeclareLaunchArgument(
                "g20_joint_order",
                default_value="",
                description="G20：l25_urdf（整机）| native（手-only）| 空=探测",
            ),
            DeclareLaunchArgument(
                "control_rate",
                default_value="10.0",
                description="原语 commands 下发频率 Hz（默认 10，可设 100）",
            ),
            DeclareLaunchArgument(
                "launch_gate",
                default_value="false",
                description="Start per-side GraspGate (cmd -> cmd_exec)",
            ),
            DeclareLaunchArgument(
                "bboxes_3d_topic", default_value="/camera_head/detection_bbox/bboxes_3d"
            ),
            DeclareLaunchArgument(
                "labeled_pose_topic", default_value="/camera_head/perception/labeled_poses"
            ),
            LogInfo(
                msg=[
                    "bimanual_gesture: left+right with side topics "
                    "/{side}_hand/gesture_cmd[_exec], tcp_pose, object_pose. "
                    "Upstream must publish /{side}_hand/tcp_pose for vision/gate.",
                ]
            ),
            _side_include("left"),
            _side_include("right"),
        ]
    )
