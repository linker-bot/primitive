"""测试 launch: 启动 mock 感知 + 手势节点，无需外部设备即可验证自适应手势。

用法:
    # 默认: 圆柱形物体 (screwdriver)，左手，L25，domain 42
    ROS_DOMAIN_ID=42 ros2 launch hand_gesture_primitives test_with_mock.launch.py

    # L25 左手
    ROS_DOMAIN_ID=42 ros2 launch hand_gesture_primitives test_with_mock.launch.py hand_side:=left hand_joint:=L25

    # 方形物体 + 右手
    ros2 launch hand_gesture_primitives test_with_mock.launch.py shape:=box hand_side:=right

    # 多物体模式 (同时发布 4 种预设)
    ros2 launch hand_gesture_primitives test_with_mock.launch.py multi:=true

测试步骤:
    1. 启动本 launch
    2. 发送手势指令:
       ROS_DOMAIN_ID=42 ros2 topic pub --once /hand_gesture_cmd std_msgs/msg/String "data: 'index_ring_by_vision'"
       ROS_DOMAIN_ID=42 ros2 topic pub --once /hand_gesture_cmd std_msgs/msg/String "data: 'large_wrap_by_vision'"
    3. 观察 /cb_left_hand_control_cmd 输出角度是否随物体尺寸自适应
    4. 切换物体:
       ROS_DOMAIN_ID=42 ros2 topic pub --once /hand_gesture_cmd std_msgs/msg/String "data: 'target large_motor'"
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('hand_side', default_value='left'),
        DeclareLaunchArgument('hand_joint', default_value='O20',
                             description='手部型号: O20 / L25'),
        DeclareLaunchArgument('shape', default_value='cylinder',
                             description='box / cylinder / small / large'),
        DeclareLaunchArgument('label', default_value='',
                             description='物体 label (空=使用 shape 预设)'),
        DeclareLaunchArgument('multi', default_value='false',
                             description='true=同时发布所有预设物体'),
        DeclareLaunchArgument('bboxes_3d_topic',
                             default_value='/camera_head/detection_bbox/bboxes_3d'),
        DeclareLaunchArgument('fingertip_frame', default_value='',
                             description='指尖 marker 坐标系 (空=自动)'),

        # Mock 感知节点
        Node(
            package='hand_gesture_primitives',
            executable='mock_perception',
            name='mock_perception',
            output='screen',
            parameters=[{
                'shape': LaunchConfiguration('shape'),
                'label': LaunchConfiguration('label'),
                'multi': LaunchConfiguration('multi'),
                'topic': LaunchConfiguration('bboxes_3d_topic'),
                'publish_rate': 5.0,
            }],
        ),

        # 手势控制节点
        Node(
            package='hand_gesture_primitives',
            executable='gesture_node',
            name='hand_gesture_node',
            output='screen',
            parameters=[{
                'hand_side': LaunchConfiguration('hand_side'),
                'hand_joint': LaunchConfiguration('hand_joint'),
                'bboxes_3d_topic': LaunchConfiguration('bboxes_3d_topic'),
                'fingertip_frame': LaunchConfiguration('fingertip_frame'),
            }],
        ),
    ])
