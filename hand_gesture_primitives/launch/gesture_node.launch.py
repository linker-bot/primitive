from hand_gesture_primitives.topic_defaults import side_hand_topics
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import EnvironmentVariable, LaunchConfiguration
from launch_ros.actions import Node


def _truthy(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes", "on")


def _launch_setup(context, *args, **kwargs):
    side = LaunchConfiguration("hand_side").perform(context).strip().lower()
    if side not in ("left", "right"):
        raise RuntimeError(f"hand_side must be left|right, got {side!r}")

    launch_gate = _truthy(LaunchConfiguration("launch_gate").perform(context))
    topics = side_hand_topics(side)

    cmd_topic = LaunchConfiguration("cmd_topic").perform(context).strip()
    if not cmd_topic:
        # With gate: listen to exec; without gate: listen to raw cmd.
        cmd_topic = topics["gesture_cmd_exec"] if launch_gate else topics["gesture_cmd"]

    tcp_pose_topic = LaunchConfiguration("tcp_pose_topic").perform(context).strip()
    object_pose_topic = LaunchConfiguration("object_pose_topic").perform(context).strip()

    nodes = [
        Node(
            package="hand_gesture_primitives",
            executable="gesture_node",
            name=f"{side}_hand_gesture_node",
            output="screen",
            additional_env={"PYTHONNOUSERSITE": "1"},
            parameters=[
                {
                    "hand_side": side,
                    "hand_type": LaunchConfiguration("hand_type").perform(context),
                    "hand_joint": LaunchConfiguration("hand_joint").perform(context),
                    "cmd_topic": cmd_topic,
                    "state_topic": LaunchConfiguration("state_topic").perform(context),
                    "cmd_backend": LaunchConfiguration("cmd_backend").perform(context),
                    "hand_cmd_topic": LaunchConfiguration("hand_cmd_topic").perform(context),
                    "ros2_control_command_units": LaunchConfiguration(
                        "ros2_control_command_units"
                    ).perform(context),
                    "robot_description": LaunchConfiguration("robot_description").perform(context),
                    "g20_joint_order": LaunchConfiguration("g20_joint_order").perform(context),
                    "control_rate": float(LaunchConfiguration("control_rate").perform(context)),
                    "primitive_speed": float(
                        LaunchConfiguration("primitive_speed").perform(context)
                    ),
                    "tcp_pose_topic": tcp_pose_topic,
                    "object_pose_topic": object_pose_topic,
                    "bboxes_3d_topic": LaunchConfiguration("bboxes_3d_topic").perform(context),
                    "labeled_pose_topic": LaunchConfiguration("labeled_pose_topic").perform(
                        context
                    ),
                    "target_object_label": LaunchConfiguration("target_object_label").perform(
                        context
                    ),
                    "reach_check_strict": _truthy(
                        LaunchConfiguration("reach_check_strict").perform(context)
                    ),
                    "reach_max_distance": float(
                        LaunchConfiguration("reach_max_distance").perform(context)
                    ),
                    "palm_offset_x": float(LaunchConfiguration("palm_offset_x").perform(context)),
                    "palm_offset_y": float(LaunchConfiguration("palm_offset_y").perform(context)),
                    "palm_offset_z": float(LaunchConfiguration("palm_offset_z").perform(context)),
                    "publish_fingertips": _truthy(
                        LaunchConfiguration("publish_fingertips").perform(context)
                    ),
                    "fingertip_frame": LaunchConfiguration("fingertip_frame").perform(context),
                    "urdf_path": LaunchConfiguration("urdf_path").perform(context),
                    "contact_config_path": LaunchConfiguration("contact_config_path").perform(
                        context
                    ),
                    "contact_current_delta": float(
                        LaunchConfiguration("contact_current_delta").perform(context)
                    ),
                    "contact_current_delta_narrow": float(
                        LaunchConfiguration("contact_current_delta_narrow").perform(context)
                    ),
                    "hold_safe_current": float(
                        LaunchConfiguration("hold_safe_current").perform(context)
                    ),
                    "overload_current_threshold": float(
                        LaunchConfiguration("overload_current_threshold").perform(context)
                    ),
                    "overload_duration_sec": float(
                        LaunchConfiguration("overload_duration_sec").perform(context)
                    ),
                }
            ],
        ),
    ]

    if launch_gate:
        nodes.append(
            Node(
                package="hand_gesture_primitives",
                executable="grasp_gate",
                name=f"{side}_grasp_gate",
                output="screen",
                additional_env={"PYTHONNOUSERSITE": "1"},
                parameters=[
                    {
                        "hand_side": side,
                        "tcp_pose_topic": tcp_pose_topic,
                        "bbox_topic": LaunchConfiguration("bboxes_3d_topic").perform(context),
                        "labeled_pose_topic": LaunchConfiguration("labeled_pose_topic").perform(
                            context
                        ),
                        "input_cmd_topic": "",
                        "output_cmd_topic": "",
                        "selected_pose_topic": object_pose_topic,
                        "palm_z": 0.06,
                        "tip_z": 0.18,
                        "half_w": 0.20,
                        "half_h": 0.15,
                        "height_tol": 0.05,
                        "de_bounce": 3,
                    }
                ],
            ),
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
            DeclareLaunchArgument(
                "primitive_speed", default_value="1.0", description="Playback speed, 0.25..3.0"
            ),
            DeclareLaunchArgument("hand_side", default_value="left", description="left 或 right"),
            DeclareLaunchArgument(
                "hand_type", default_value="o20", description="手型: o20 / l25 / o6 / l6"
            ),
            DeclareLaunchArgument(
                "hand_joint",
                default_value="",
                description="hand_type 别名，如 O20 / L25 / O6 / L6",
            ),
            DeclareLaunchArgument(
                "cmd_topic",
                default_value="",
                description="指令输入；空=无门控用 /{side}_hand/gesture_cmd，"
                "有门控用 /{side}_hand/gesture_cmd_exec",
            ),
            DeclareLaunchArgument("state_topic", default_value="/joint_states"),
            DeclareLaunchArgument("cmd_backend", default_value=""),
            DeclareLaunchArgument("hand_cmd_topic", default_value=""),
            DeclareLaunchArgument(
                "ros2_control_command_units",
                default_value="radians",
                description="radians=URDF弧度（mock/sim/real 默认）；hardware=遗留SDK 0~255",
            ),
            DeclareLaunchArgument(
                "robot_description",
                default_value="",
                description="radians 模式用于读取精确关节限位的 URDF XML",
            ),
            DeclareLaunchArgument(
                "g20_joint_order",
                default_value="",
                description="G20 命令序：l25_urdf（整机）| native（手-only）| 空=探测",
            ),
            DeclareLaunchArgument(
                "control_rate",
                default_value="10.0",
                description="原语 commands 下发频率 Hz（默认 10，可设 100）",
            ),
            DeclareLaunchArgument(
                "tcp_pose_topic", default_value="", description="空=/{side}_hand/tcp_pose"
            ),
            DeclareLaunchArgument(
                "object_pose_topic", default_value="", description="空=/{side}_hand/object_pose"
            ),
            DeclareLaunchArgument(
                "bboxes_3d_topic", default_value="/camera_head/detection_bbox/bboxes_3d"
            ),
            DeclareLaunchArgument(
                "labeled_pose_topic", default_value="/camera_head/perception/labeled_poses"
            ),
            DeclareLaunchArgument("target_object_label", default_value=""),
            DeclareLaunchArgument("reach_check_strict", default_value="false"),
            DeclareLaunchArgument("reach_max_distance", default_value="0.15"),
            DeclareLaunchArgument("palm_offset_x", default_value="0.0"),
            DeclareLaunchArgument("palm_offset_y", default_value="0.0"),
            DeclareLaunchArgument("palm_offset_z", default_value="-0.05"),
            DeclareLaunchArgument(
                "publish_fingertips",
                default_value="true",
                description="是否发布 FK 指尖 Marker/数组（需 pinocchio + URDF）",
            ),
            DeclareLaunchArgument("fingertip_frame", default_value=""),
            DeclareLaunchArgument(
                "urdf_path",
                default_value="",
                description="FK URDF 绝对路径；空=自动搜索 evo_work / linkerhand-urdf（含 l6v3.1）",
            ),
            DeclareLaunchArgument("launch_gate", default_value="false"),
            DeclareLaunchArgument("contact_config_path", default_value=""),
            DeclareLaunchArgument(
                "contact_current_delta",
                default_value="270.0",
                description="O20/L25 电流接触增量 (mA)；O6 用力矩阈值，L6 以触觉为主",
            ),
            DeclareLaunchArgument("contact_current_delta_narrow", default_value="250.0"),
            DeclareLaunchArgument("hold_safe_current", default_value="800.0"),
            DeclareLaunchArgument("overload_current_threshold", default_value="1000.0"),
            DeclareLaunchArgument("overload_duration_sec", default_value="2.0"),
            OpaqueFunction(function=_launch_setup),
        ]
    )
