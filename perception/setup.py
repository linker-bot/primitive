from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'robot_perception'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*')),
        (os.path.join('share', package_name, 'config', 'calib_results'),
         glob('config/calib_results/*.txt')),
        (os.path.join('share', package_name, 'config', 'pixel3d'),
         glob('config/pixel3d/*')),
        (os.path.join('share', package_name, 'config'),
         glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'detection_bbox = robot_perception.detection_bbox_node:main',
        ],
    },
)
