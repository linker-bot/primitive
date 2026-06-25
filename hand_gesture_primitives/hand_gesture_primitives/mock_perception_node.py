"""Mock 感知节点 — 发布模拟物体 3D 包围盒用于手势原语测试。

无需外部感知硬件，即可验证 index_ring_by_vision / large_wrap_by_vision 等自适应手势原语
对感知物体尺寸的响应。

发布两种预设物体（可通过参数选择）：
  - box:      方形物体 40x40x60mm (如小电机)
  - cylinder: 圆柱形物体 ø25x120mm (如螺丝刀柄)
  - small:    小圆形物体 ø12x15mm (如螺丝)
  - large:    大方形物体 80x80x50mm (如大电机)

用法:
    ros2 run hand_gesture_primitives mock_perception \
        --ros-args -p shape:=cylinder -p label:=screwdriver

    # 配合手势测试:
    ros2 run hand_gesture_primitives gesture_node --ros-args -p hand_side:=left
    ros2 topic pub --once /hand_gesture_cmd std_msgs/msg/String "data: 'index_ring_by_vision screwdriver'"
"""

import math

import rclpy
from rclpy.node import Node

from robot_perception_msgs.msg import LabeledBBox3D, LabeledBBox3DArray


# 预设物体尺寸 (meters): [sx, sy, sz]
SHAPE_PRESETS = {
    'box': {
        'size': [0.040, 0.040, 0.060],
        'label': 'small_motor',
        'grasp_type': 'power',
    },
    'cylinder': {
        'size': [0.025, 0.025, 0.120],
        'label': 'screwdriver',
        'grasp_type': 'precision',
    },
    'small': {
        'size': [0.012, 0.012, 0.015],
        'label': 'screw',
        'grasp_type': 'precision',
    },
    'large': {
        'size': [0.080, 0.080, 0.050],
        'label': 'large_motor',
        'grasp_type': 'power',
    },
}

# 模拟工作台上物体位置 (world frame, meters)
DEFAULT_CENTER = [0.35, 0.0, -0.97]


class MockPerceptionNode(Node):
    """以固定频率发布模拟 LabeledBBox3DArray 消息。"""

    def __init__(self):
        super().__init__('mock_perception')

        self.declare_parameter('shape', 'cylinder')
        self.declare_parameter('label', '')
        self.declare_parameter('grasp_type', '')
        self.declare_parameter('size_x', 0.0)
        self.declare_parameter('size_y', 0.0)
        self.declare_parameter('size_z', 0.0)
        self.declare_parameter('center_x', DEFAULT_CENTER[0])
        self.declare_parameter('center_y', DEFAULT_CENTER[1])
        self.declare_parameter('center_z', DEFAULT_CENTER[2])
        self.declare_parameter('orientation_yaw_deg', 0.0)
        self.declare_parameter('score', 0.92)
        self.declare_parameter('publish_rate', 5.0)
        self.declare_parameter('topic',
                               '/camera_head/detection_bbox/bboxes_3d')
        self.declare_parameter('multi', False)

        shape_name = self.get_parameter('shape').value
        preset = SHAPE_PRESETS.get(shape_name, SHAPE_PRESETS['cylinder'])

        sx = self.get_parameter('size_x').value
        sy = self.get_parameter('size_y').value
        sz = self.get_parameter('size_z').value
        if sx > 0 and sy > 0 and sz > 0:
            self._size = [sx, sy, sz]
        else:
            self._size = list(preset['size'])

        self._label = self.get_parameter('label').value or preset['label']
        self._grasp_type = self.get_parameter('grasp_type').value or preset['grasp_type']
        self._center = [
            self.get_parameter('center_x').value,
            self.get_parameter('center_y').value,
            self.get_parameter('center_z').value,
        ]
        self._yaw_deg = self.get_parameter('orientation_yaw_deg').value
        self._score = self.get_parameter('score').value
        self._multi = self.get_parameter('multi').value

        topic = self.get_parameter('topic').value
        rate = self.get_parameter('publish_rate').value

        self._pub = self.create_publisher(LabeledBBox3DArray, topic, 10)
        self._timer = self.create_timer(1.0 / rate, self._publish)
        self._instance_id = 1

        self.get_logger().info(
            f'Mock 感知已启动: shape={shape_name}, label={self._label}, '
            f'size={self._size}, grasp_type={self._grasp_type}, '
            f'rate={rate}Hz, topic={topic}'
        )
        if self._multi:
            self.get_logger().info(
                '多物体模式: 同时发布所有预设物体 (box + cylinder + small + large)')

    def _make_bbox3d(self, label, size, center, grasp_type, score, instance_id):
        """构造单个 LabeledBBox3D 消息。"""
        msg = LabeledBBox3D()
        msg.label = label
        msg.prompt = label
        msg.score = float(score)
        msg.instance_id = instance_id
        msg.frame_id = 'world'
        msg.center = [float(v) for v in center]
        msg.size = [float(v) for v in size]
        half = [s / 2.0 for s in size]
        msg.min = [c - h for c, h in zip(center, half)]
        msg.max = [c + h for c, h in zip(center, half)]

        yaw_rad = math.radians(self._yaw_deg)
        msg.orientation = [0.0, 0.0, math.sin(yaw_rad / 2), math.cos(yaw_rad / 2)]
        msg.top_normal = [0.0, 0.0, 1.0]
        msg.grasp_type = grasp_type
        msg.track_mode = 'refined'
        return msg

    def _publish(self):
        array_msg = LabeledBBox3DArray()
        array_msg.header.stamp = self.get_clock().now().to_msg()
        array_msg.header.frame_id = 'world'

        if self._multi:
            y_offsets = [-0.12, -0.04, 0.04, 0.12]
            for i, (name, preset) in enumerate(SHAPE_PRESETS.items()):
                center = [
                    self._center[0],
                    self._center[1] + y_offsets[i],
                    self._center[2],
                ]
                b = self._make_bbox3d(
                    preset['label'], preset['size'], center,
                    preset['grasp_type'], 0.90 - i * 0.02, i + 1)
                array_msg.boxes.append(b)
        else:
            b = self._make_bbox3d(
                self._label, self._size, self._center,
                self._grasp_type, self._score, self._instance_id)
            array_msg.boxes.append(b)

        self._pub.publish(array_msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockPerceptionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
