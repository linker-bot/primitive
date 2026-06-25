from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hand_side", default_value="left",
                              description="left 或 right"),
        DeclareLaunchArgument("hand_type", default_value="o20",
                              description="手型: o20 或 l25 (别名 hand_joint:=O20 也支持)"),
        DeclareLaunchArgument("hand_joint", default_value="",
                              description="hand_type 兼容别名，如 O20 / L25 (非空时覆盖 hand_type)"),
        DeclareLaunchArgument("cmd_topic", default_value="/hand_gesture_cmd_exec",
                              description="手势指令输入 topic"),
        DeclareLaunchArgument("tcp_pose_topic", default_value="/tcp_pose",
                              description="手腕 TCP 位姿 topic"),
        DeclareLaunchArgument("object_pose_topic", default_value="/object_pose",
                              description="物体位姿 topic (相对手腕)"),
        DeclareLaunchArgument("bboxes_3d_topic",
                              default_value="/camera_head/detection_bbox/bboxes_3d",
                              description="LabeledBBox3DArray topic (感知物体尺寸/方向)"),
        DeclareLaunchArgument("labeled_pose_topic",
                              default_value="/camera_head/perception/labeled_poses",
                              description="感知 pipeline LabeledPoseArray topic"),
        DeclareLaunchArgument("target_object_label", default_value="",
                              description="待抓取物体 label（空=自动选最高 score）"),
        DeclareLaunchArgument("reach_check_strict", default_value="false",
                              description="可达性严格检查: true=不可达时拒绝执行"),
        DeclareLaunchArgument("reach_max_distance", default_value="0.15",
                              description="TCP→目标最大允许距离 (m)"),
        DeclareLaunchArgument("palm_offset_x", default_value="0.0",
                              description="手掌中心相对 TCP 偏移 X (m)"),
        DeclareLaunchArgument("palm_offset_y", default_value="0.0",
                              description="手掌中心相对 TCP 偏移 Y (m)"),
        DeclareLaunchArgument("palm_offset_z", default_value="-0.05",
                              description="手掌中心相对 TCP 偏移 Z (m)"),
        DeclareLaunchArgument("publish_fingertips", default_value="true",
                              description="是否发布 FK 指尖位置 topic"),
        DeclareLaunchArgument("fingertip_frame", default_value="",
                              description="指尖 marker 坐标系 (空=自动 l_/r_hand_base_link)"),
        DeclareLaunchArgument("urdf_path", default_value="",
                              description="手部 URDF 路径 (空=自动搜索)"),
        DeclareLaunchArgument("launch_gate", default_value="false",
                              description="是否同时启动 GraspGate 门控节点"),
        DeclareLaunchArgument(
            "contact_config_path", default_value="",
            description="接触/电流阈值 YAML 路径 (空=包内 contact_thresholds.yaml)"),
        DeclareLaunchArgument("contact_current_delta", default_value="270.0",
                              description="无触觉时电流增量接触阈值 (mA)"),
        DeclareLaunchArgument("contact_current_delta_narrow", default_value="250.0",
                              description="窄物体原语电流增量阈值 (mA)"),
        DeclareLaunchArgument("hold_safe_current", default_value="800.0",
                              description="力控持握安全电流上限 (mA)"),
        DeclareLaunchArgument("overload_current_threshold", default_value="1000.0",
                              description="全局过载保护电流阈值 (mA)"),
        DeclareLaunchArgument("overload_duration_sec", default_value="2.0",
                              description="过载保护持续时间 (s)"),

        # ── 手势节点 ──
        Node(
            package="hand_gesture_primitives",
            executable="gesture_node",
            name="hand_gesture_node",
            output="screen",
            parameters=[{
                "hand_side": LaunchConfiguration("hand_side"),
                "hand_type": LaunchConfiguration("hand_type"),
                "hand_joint": LaunchConfiguration("hand_joint"),
                "cmd_topic": LaunchConfiguration("cmd_topic"),
                "tcp_pose_topic": LaunchConfiguration("tcp_pose_topic"),
                "object_pose_topic": LaunchConfiguration("object_pose_topic"),
                "bboxes_3d_topic": LaunchConfiguration("bboxes_3d_topic"),
                "labeled_pose_topic": LaunchConfiguration("labeled_pose_topic"),
                "target_object_label": LaunchConfiguration("target_object_label"),
                "reach_check_strict": LaunchConfiguration("reach_check_strict"),
                "reach_max_distance": LaunchConfiguration("reach_max_distance"),
                "palm_offset_x": LaunchConfiguration("palm_offset_x"),
                "palm_offset_y": LaunchConfiguration("palm_offset_y"),
                "palm_offset_z": LaunchConfiguration("palm_offset_z"),
                "publish_fingertips": LaunchConfiguration("publish_fingertips"),
                "fingertip_frame": LaunchConfiguration("fingertip_frame"),
                "urdf_path": LaunchConfiguration("urdf_path"),
                "contact_config_path": LaunchConfiguration("contact_config_path"),
                "contact_current_delta": ParameterValue(
                    LaunchConfiguration("contact_current_delta"), value_type=float),
                "contact_current_delta_narrow": ParameterValue(
                    LaunchConfiguration("contact_current_delta_narrow"),
                    value_type=float),
                "hold_safe_current": ParameterValue(
                    LaunchConfiguration("hold_safe_current"), value_type=float),
                "overload_current_threshold": ParameterValue(
                    LaunchConfiguration("overload_current_threshold"),
                    value_type=float),
                "overload_duration_sec": ParameterValue(
                    LaunchConfiguration("overload_duration_sec"), value_type=float),
            }],
        ),

        # ── 门控节点 (可选) ──
        Node(
            package="hand_gesture_primitives",
            executable="grasp_gate",
            name="grasp_gate",
            output="screen",
            parameters=[{
                "tcp_pose_topic": LaunchConfiguration("tcp_pose_topic"),
                "bbox_topic": LaunchConfiguration("bboxes_3d_topic"),
                "labeled_pose_topic": LaunchConfiguration("labeled_pose_topic"),
                "input_cmd_topic": "/hand_gesture_cmd",
                "output_cmd_topic": "/hand_gesture_cmd_exec",
                "selected_pose_topic": "/object_pose",
                # 手几何 — 三判定参数
                "palm_z": 0.06,
                "tip_z": 0.18,
                "half_w": 0.20,
                "half_h": 0.15,
                "height_tol": 0.05,
                "de_bounce": 3,
            }],
            condition=IfCondition(LaunchConfiguration("launch_gate")),
        ),
    ])
