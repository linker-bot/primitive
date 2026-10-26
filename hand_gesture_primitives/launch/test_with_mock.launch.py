"""测试 launch: 启动模拟感知与原语节点。

说明:
    本 launch 只模拟感知目标，不提供硬件驱动或关节反馈。
    独立网页 Mock 和 ROS 接入步骤见仓库根目录 docs/getting-started.md。

用法:
    # 默认: 圆柱形物体 (screwdriver)，左手，O20，domain 42
    ROS_DOMAIN_ID=42 ros2 launch hand_gesture_primitives test_with_mock.launch.py

    # L25 左手
    ROS_DOMAIN_ID=42 ros2 launch hand_gesture_primitives test_with_mock.launch.py \\
      hand_side:=left hand_joint:=L25

    # L6 左手（感知 mock；力控仍依赖触觉/配置，无真实电流）
    ROS_DOMAIN_ID=42 ros2 launch hand_gesture_primitives test_with_mock.launch.py \\
      hand_side:=left hand_joint:=L6

    # 方形物体 + 右手
    ros2 launch hand_gesture_primitives test_with_mock.launch.py shape:=box hand_side:=right

    # 多物体模式 (同时发布 4 种预设)
    ros2 launch hand_gesture_primitives test_with_mock.launch.py multi:=true

测试步骤:
    1. 启动本 launch
    2. 发送手势指令（按 hand_side，默认 left）:
       ROS_DOMAIN_ID=42 ros2 topic pub --once /left_hand/gesture_cmd \\
         std_msgs/msg/String "data: 'index_ring_by_vision'"
       ROS_DOMAIN_ID=42 ros2 topic pub --once /left_hand/gesture_cmd \\
         std_msgs/msg/String "data: 'large_wrap_by_vision'"
    3. 观察 /left_hand_controller/commands 或 /cb_left_hand_control_cmd 输出
       （本 launch 未起 controller 时仅看节点日志/无订阅警告）
    4. 切换物体:
       ROS_DOMAIN_ID=42 ros2 topic pub --once /left_hand/gesture_cmd \\
         std_msgs/msg/String "data: 'target large_motor'"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    side = LaunchConfiguration("hand_side").perform(context).strip().lower() or "left"
    return [
        Node(
            package="hand_gesture_primitives",
            executable="mock_perception",
            name="mock_perception",
            output="screen",
            parameters=[
                {
                    "shape": LaunchConfiguration("shape").perform(context),
                    "label": LaunchConfiguration("label").perform(context),
                    "multi": LaunchConfiguration("multi").perform(context).lower()
                    in ("1", "true", "yes", "on"),
                    "topic": LaunchConfiguration("bboxes_3d_topic").perform(context),
                    "publish_rate": 5.0,
                }
            ],
        ),
        Node(
            package="hand_gesture_primitives",
            executable="gesture_node",
            name=f"{side}_hand_gesture_node",
            output="screen",
            additional_env={"PYTHONNOUSERSITE": "1"},
            parameters=[
                {
                    "hand_side": side,
                    "hand_joint": LaunchConfiguration("hand_joint").perform(context),
                    "bboxes_3d_topic": LaunchConfiguration("bboxes_3d_topic").perform(context),
                    "fingertip_frame": LaunchConfiguration("fingertip_frame").perform(context),
                    # empty cmd/tcp/object => /{side}_hand/...
                }
            ],
        ),
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("hand_side", default_value="left"),
            DeclareLaunchArgument(
                "hand_joint",
                default_value="O20",
                description="手部型号: O20 / L25 / O6 / L6（默认 O20）",
            ),
            DeclareLaunchArgument(
                "shape", default_value="cylinder", description="box / cylinder / small / large"
            ),
            DeclareLaunchArgument(
                "label", default_value="", description="物体 label (空=使用 shape 预设)"
            ),
            DeclareLaunchArgument(
                "multi", default_value="false", description="true=同时发布所有预设物体"
            ),
            DeclareLaunchArgument(
                "bboxes_3d_topic", default_value="/camera_head/detection_bbox/bboxes_3d"
            ),
            DeclareLaunchArgument(
                "fingertip_frame", default_value="", description="指尖 marker 坐标系 (空=自动)"
            ),
            OpaqueFunction(function=_setup),
        ]
    )
