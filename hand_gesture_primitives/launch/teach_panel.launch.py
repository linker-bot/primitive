from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.conditions import IfCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _setup(context, *args, **kwargs):
    hand_side = LaunchConfiguration("hand_side").perform(context).strip().lower()
    suffix = f"_{hand_side}"
    gesture_name = f"{hand_side}_hand_gesture_node"
    panel_name = f"teach_panel{suffix}"
    server_name = f"pose_library_server{suffix}"

    nodes = []
    nodes.append(
        Node(
            package="hand_gesture_primitives",
            executable="gesture_node",
            name=gesture_name,
            output="screen",
            condition=IfCondition(LaunchConfiguration("launch_gesture_node")),
            additional_env={"PYTHONNOUSERSITE": "1"},
            parameters=[
                {
                    "hand_side": LaunchConfiguration("hand_side"),
                    "primitive_speed": ParameterValue(
                        LaunchConfiguration("primitive_speed"), value_type=float
                    ),
                    "hand_type": LaunchConfiguration("hand_type"),
                    "hand_joint": LaunchConfiguration("hand_joint"),
                    "cmd_topic": LaunchConfiguration("cmd_topic"),
                    "status_topic": LaunchConfiguration("status_topic"),
                    "result_topic": LaunchConfiguration("result_topic"),
                    "state_topic": LaunchConfiguration("state_topic"),
                    "cmd_backend": LaunchConfiguration("cmd_backend"),
                    "hand_cmd_topic": LaunchConfiguration("hand_cmd_topic"),
                    "ros2_control_command_units": LaunchConfiguration(
                        "ros2_control_command_units"
                    ),
                    "robot_description": LaunchConfiguration("robot_description"),
                    "g20_joint_order": LaunchConfiguration("g20_joint_order"),
                    "control_rate": ParameterValue(
                        LaunchConfiguration("control_rate"), value_type=float
                    ),
                    "stream_target_topic": LaunchConfiguration("stream_target_topic"),
                    "stream_watchdog_sec": ParameterValue(
                        LaunchConfiguration("stream_watchdog_sec"), value_type=float
                    ),
                }
            ],
        )
    )
    nodes.append(
        Node(
            package="hand_gesture_primitives",
            executable="teach_panel",
            name=panel_name,
            output="screen",
            parameters=[
                {
                    "hand_side": LaunchConfiguration("hand_side"),
                    "primitive_speed": ParameterValue(
                        LaunchConfiguration("primitive_speed"), value_type=float
                    ),
                    "hand_type": LaunchConfiguration("hand_type"),
                    "hand_joint": LaunchConfiguration("hand_joint"),
                    "cmd_topic": LaunchConfiguration("cmd_topic"),
                    "status_topic": LaunchConfiguration("status_topic"),
                    "result_topic": LaunchConfiguration("result_topic"),
                    "state_topic": LaunchConfiguration("state_topic"),
                    "cmd_backend": LaunchConfiguration("cmd_backend"),
                    "hand_cmd_topic": LaunchConfiguration("hand_cmd_topic"),
                    "ros2_control_command_units": LaunchConfiguration(
                        "ros2_control_command_units"
                    ),
                    "robot_description": LaunchConfiguration("robot_description"),
                    "g20_joint_order": LaunchConfiguration("g20_joint_order"),
                    "http_host": LaunchConfiguration("http_host"),
                    "http_port": ParameterValue(LaunchConfiguration("http_port"), value_type=int),
                    "control_rate": ParameterValue(
                        LaunchConfiguration("control_rate"), value_type=float
                    ),
                    "observe_only": ParameterValue(
                        LaunchConfiguration("observe_only"), value_type=bool
                    ),
                    "publish_rate": ParameterValue(
                        LaunchConfiguration("publish_rate"), value_type=float
                    ),
                }
            ],
        )
    )
    nodes.append(
        Node(
            package="hand_gesture_primitives",
            executable="pose_library_server",
            name=server_name,
            output="screen",
            condition=IfCondition(LaunchConfiguration("launch_pose_server")),
            parameters=[
                {
                    "hand_side": LaunchConfiguration("hand_side"),
                    "hand_type": LaunchConfiguration("hand_type"),
                    "hand_joint": LaunchConfiguration("hand_joint"),
                    "state_topic": LaunchConfiguration("state_topic"),
                }
            ],
        )
    )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "pose_dir",
                default_value=EnvironmentVariable("HAND_GESTURE_POSE_DIR", default_value=""),
                description="Explicit shared pose directory; no implicit HOME fallback",
            ),
            SetEnvironmentVariable("HAND_GESTURE_POSE_DIR", LaunchConfiguration("pose_dir")),
            DeclareLaunchArgument("primitive_speed", default_value="1.0"),
            DeclareLaunchArgument("control_rate", default_value="10.0"),
            DeclareLaunchArgument("hand_side", default_value="left"),
            DeclareLaunchArgument("hand_type", default_value="l25"),
            DeclareLaunchArgument("hand_joint", default_value=""),
            DeclareLaunchArgument(
                "cmd_topic",
                default_value="",
                description="空=/{side}_hand/gesture_cmd",
            ),
            DeclareLaunchArgument("status_topic", default_value=""),
            DeclareLaunchArgument("result_topic", default_value=""),
            DeclareLaunchArgument("state_topic", default_value="/joint_states"),
            DeclareLaunchArgument(
                "cmd_backend",
                default_value="",
                description="空=按手型默认 ros2_control；legacy_cb 走 /cb_*",
            ),
            DeclareLaunchArgument("hand_cmd_topic", default_value=""),
            DeclareLaunchArgument(
                "ros2_control_command_units",
                default_value="radians",
                description="radians=URDF弧度（机载默认）；hardware=遗留SDK 0~255",
            ),
            DeclareLaunchArgument(
                "robot_description",
                default_value="",
                description="radians 模式必填（可与 gesture 共用 URDF XML）",
            ),
            DeclareLaunchArgument(
                "g20_joint_order",
                default_value="",
                description="整机 G20 用 l25_urdf；手-only 用 native；空则探测 URDF",
            ),
            DeclareLaunchArgument("stream_target_topic", default_value=""),
            DeclareLaunchArgument("stream_watchdog_sec", default_value="0.5"),
            DeclareLaunchArgument("http_host", default_value="127.0.0.1"),
            DeclareLaunchArgument("http_port", default_value="8001"),
            DeclareLaunchArgument("publish_rate", default_value="15.0"),
            DeclareLaunchArgument("observe_only", default_value="false"),
            DeclareLaunchArgument("launch_gesture_node", default_value="false"),
            DeclareLaunchArgument("launch_pose_server", default_value="true"),
            OpaqueFunction(function=_setup),
        ]
    )
