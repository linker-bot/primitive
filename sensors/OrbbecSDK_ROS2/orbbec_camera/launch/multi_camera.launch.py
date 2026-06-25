from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import PushRosNamespace
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    package_dir = get_package_share_directory('orbbec_camera')
    launch_file_dir = os.path.join(package_dir, 'launch')

    args = [
        DeclareLaunchArgument('cam_head_port', default_value='2-9'),
        DeclareLaunchArgument('cam_waist_port', default_value='2-6'),
        DeclareLaunchArgument('enable_cam_head', default_value='true'),
        DeclareLaunchArgument('enable_cam_waist', default_value='true'),
    ]

    launch_head = GroupAction(
        condition=IfCondition(LaunchConfiguration('enable_cam_head')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
                ),
                launch_arguments={
                    'camera_name': 'camera_head',
                    'usb_port': LaunchConfiguration('cam_head_port'),
                    'device_num': '2',
                    'sync_mode': 'standalone',
                    'depth_registration': 'true',
                    'enable_left_ir': 'true',
                    'enable_right_ir': 'true',
                    'connection_delay': '12',
                    'log_level': 'none',
                    'log_file_name': 'camera_head.log',
                }.items()
            ),
        ],
    )

    launch_waist = GroupAction(
        condition=IfCondition(LaunchConfiguration('enable_cam_waist')),
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(launch_file_dir, 'gemini_330_series.launch.py')
                ),
                launch_arguments={
                    'camera_name': 'camera_waist',
                    'usb_port': LaunchConfiguration('cam_waist_port'),
                    'device_num': '2',
                    'sync_mode': 'standalone',
                    'depth_registration': 'true',
                    'enable_left_ir': 'true',
                    'enable_right_ir': 'true',
                    'connection_delay': '10',
                    'log_level': 'none',
                    'log_file_name': 'camera_waist.log',
                }.items()
            ),
        ],
    )

    return LaunchDescription(args + [launch_waist, launch_head])
