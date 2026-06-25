"""关节状态桥接节点 — 将手部驱动反馈转换为 URDF 关节弧度发布到 /joint_states。

用途: 在 RViz 中用真实硬件反馈驱动 URDF 模型，验证 FK 计算与模型是否吻合。

订阅: /cb_{side}_hand_state (JointState, 硬件 O20/L25 格式)
发布: /joint_states (JointState, URDF 弧度) → robot_state_publisher → TF → RViz
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .fk_solver import HandFKSolver, create_fk_solver
from .hand_config import HandConfig


class JointStateBridge(Node):
    """将手部硬件反馈转为 URDF 关节弧度，供 robot_state_publisher 使用。"""

    def __init__(self):
        super().__init__("joint_state_bridge")

        self.declare_parameter("hand_side", "left")
        self.declare_parameter("hand_joint", "L25")
        self.declare_parameter("urdf_path", "")

        hand_side = self.get_parameter("hand_side").value
        hand_joint = self.get_parameter("hand_joint").value
        urdf_path = self.get_parameter("urdf_path").value or ""

        self._config = HandConfig(hand_joint)
        self._fk: HandFKSolver = None
        self._last_o20 = None

        fk = create_fk_solver(hand_joint, hand_side, urdf_path, self.get_logger())
        if fk is None:
            self.get_logger().error("FK 求解器初始化失败，节点无法工作")
            raise RuntimeError("FK solver unavailable")
        self._fk = fk

        self._pub = self.create_publisher(JointState, "/joint_states", 10)

        state_topic = f"/cb_{hand_side}_hand_state"
        self.create_subscription(JointState, state_topic, self._on_state, 10)

        self.get_logger().info(
            f"JointStateBridge 就绪: {state_topic} → /joint_states "
            f"(model={hand_joint}, joints={len(fk._active_mapping)})"
        )

    def _on_state(self, msg: JointState) -> None:
        """接收硬件反馈, 转换并发布 URDF 关节角度。"""
        if len(msg.position) == self._config.num_joints:
            o20 = self._config.from_hardware(list(msg.position))
        elif len(msg.position) == 20:
            o20 = list(msg.position)
        else:
            return

        cfg = self._fk.o20_to_radians(o20)
        if not cfg:
            return

        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name = list(cfg.keys())
        out.position = list(cfg.values())
        self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    try:
        node = JointStateBridge()
    except RuntimeError:
        rclpy.shutdown()
        return
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
