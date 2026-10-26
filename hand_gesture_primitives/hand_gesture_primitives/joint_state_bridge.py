"""关节状态桥接节点 — 将手部驱动反馈转换为 URDF 关节弧度发布到 /joint_states。

用途: 在 RViz 中用真实硬件反馈驱动 **mesh URDF** 模型，验证 FK 与几何。
      通常由 `view_hand_urdf.launch.py mode:=live` 拉起。

订阅（由 state_source 决定）:
  - legacy_cb: /cb_{side}_hand_state（L25/O20 旧链路）
  - joint_states: /joint_states（ros2_control 手-only / 机载，HOP 关节名）

发布: 默认 **/mesh_joint_states**（mesh URDF 解剖关节名 + 弧度）
      → mesh robot_state_publisher remapping → TF → RViz
      勿回写 /joint_states，以免与 stub/机载 RSP 冲突。

手型: hand_joint = L25 / O20 / O6 / L6 / O30 / G20（须与加载的 mesh URDF 一致）。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

from .fk_solver import HandFKSolver, create_fk_solver
from .hand_config import HandConfig

# ros2_control 主路径手型：默认从 /joint_states 读 HOP/控制器关节名
_ROS2_CONTROL_HANDS = frozenset({"o30", "o6", "l6", "l25", "o20", "g20"})


def _looks_like_radians(values: List[float]) -> bool:
    if not values:
        return False
    return max(abs(float(v)) for v in values) <= 6.5


def hardware_vector_from_joint_state(
    msg: "JointState",
    *,
    hand_joint: str,
    hand_side: str,
    radian_converter=None,
) -> Optional[List[float]]:
    """从 JointState 提取产品 hardware 向量（0~255），必要时自弧度换算。"""
    from .joint_state_mapping import (
        extract_o6_from_joint_state,
        extract_o20_from_joint_state,
        extract_o30_from_joint_state,
        joint_state_to_g20_hardware,
        joint_state_to_l25_native,
    )

    key = (hand_joint or "").strip().lower()
    side = (hand_side or "left").strip().lower()
    positions = [float(v) for v in msg.position]

    hw: Optional[List[float]] = None
    if msg.name:
        if key == "l25":
            native = joint_state_to_l25_native(msg.name, positions, side)
            hw = list(native) if native is not None else None
        elif key in ("o6", "l6"):
            hw = extract_o6_from_joint_state(msg.name, positions, side)
        elif key == "o20":
            hw = extract_o20_from_joint_state(msg.name, positions, side)
        elif key == "o30":
            hw = extract_o30_from_joint_state(msg.name, positions, side)
        elif key == "g20":
            ghw, _uses = joint_state_to_g20_hardware(msg.name, positions, side)
            hw = list(ghw) if ghw is not None else None
    elif positions:
        cfg = HandConfig(hand_joint)
        if len(positions) == cfg.num_joints:
            hw = positions

    if hw is None:
        return None

    if _looks_like_radians(positions) and radian_converter is not None:
        ctrl = list(hw)
        if len(ctrl) == len(radian_converter.limits):
            hw = radian_converter.radians_to_hardware_uint8(ctrl)
    return [max(0.0, min(255.0, float(v))) for v in hw]


def semantic_from_hardware(hand_joint: str, hw: List[float]) -> List[float]:
    """hardware → S30（O30 等 from_semantic 手型走 to_semantic）。"""
    cfg = HandConfig(hand_joint)
    if cfg.has_from_semantic:
        return cfg.to_semantic(hw)
    return cfg.from_hardware(hw)


def mesh_urdf_joint_state_from_feedback(
    msg: "JointState",
    *,
    hand_joint: str,
    hand_side: str,
    fk: HandFKSolver,
    state_source: str,
    config: HandConfig,
    radian_converter=None,
) -> Optional[tuple[list[str], list[float]]]:
    """硬件/JointState 反馈 → mesh URDF 关节名 + 弧度。"""
    if state_source == "joint_states":
        hw = hardware_vector_from_joint_state(
            msg,
            hand_joint=hand_joint,
            hand_side=hand_side,
            radian_converter=radian_converter,
        )
        if hw is None:
            return None
        semantic = semantic_from_hardware(hand_joint, hw)
    else:
        positions = list(msg.position)
        if len(positions) == config.num_joints:
            semantic = semantic_from_hardware(hand_joint, [float(v) for v in positions])
        elif len(positions) == 20:
            semantic = list(positions)
        else:
            return None

    cfg = fk.o20_to_radians(semantic)
    if not cfg:
        return None
    return list(cfg.keys()), list(cfg.values())


try:
    from rclpy.node import Node
    from sensor_msgs.msg import JointState

    class JointStateBridge(Node):
        """将手部硬件反馈转为 mesh URDF 关节弧度，供 robot_state_publisher 使用。"""

        def __init__(self):
            super().__init__("joint_state_bridge")

            self.declare_parameter("hand_side", "left")
            self.declare_parameter("hand_joint", "L25")
            self.declare_parameter("urdf_path", "")
            self.declare_parameter("state_topic", "")
            self.declare_parameter(
                "state_source",
                "auto",
                description="auto | legacy_cb | joint_states",
            )
            self.declare_parameter("robot_description", "")
            self.declare_parameter(
                "output_topic",
                "/mesh_joint_states",
                description="mesh URDF 关节弧度输出（勿与控制 /joint_states 混用）",
            )

            hand_side = str(self.get_parameter("hand_side").value).strip().lower()
            hand_joint = str(self.get_parameter("hand_joint").value).strip()
            urdf_path = str(self.get_parameter("urdf_path").value or "").strip()
            state_topic = str(self.get_parameter("state_topic").value or "").strip()
            state_source = str(self.get_parameter("state_source").value or "auto").strip().lower()
            robot_description = str(self.get_parameter("robot_description").value or "").strip()
            output_topic = (
                str(self.get_parameter("output_topic").value or "/mesh_joint_states").strip()
                or "/mesh_joint_states"
            )

            self._hand_joint = hand_joint
            self._hand_side = hand_side
            self._config = HandConfig(hand_joint)
            key = hand_joint.strip().lower()

            if state_source == "auto":
                state_source = "joint_states" if key in _ROS2_CONTROL_HANDS else "legacy_cb"

            fk = create_fk_solver(hand_joint, hand_side, urdf_path, self.get_logger())
            if fk is None:
                self.get_logger().error("FK solver unavailable")
                raise RuntimeError("FK solver unavailable")
            self._fk: HandFKSolver = fk

            self._rad_conv = None
            if state_source == "joint_states" and key in _ROS2_CONTROL_HANDS:
                from .joint_state_mapping import (
                    Ros2ControlRadianConverter,
                    o30_single_hand_stub_urdf,
                    resolve_radian_hand_type,
                )

                stub = robot_description
                if not stub.strip() and key == "o30":
                    stub = o30_single_hand_stub_urdf(hand_side)
                if stub.strip():
                    try:
                        rad_type = resolve_radian_hand_type(key)
                        self._rad_conv = Ros2ControlRadianConverter(stub, rad_type, hand_side)
                    except Exception as exc:
                        self.get_logger().warn(f"Radian converter init skipped: {exc}")

            if not state_topic:
                state_topic = (
                    "/joint_states"
                    if state_source == "joint_states"
                    else f"/cb_{hand_side}_hand_state"
                )

            self._state_source = state_source
            self._pub = self.create_publisher(JointState, output_topic, 10)
            self.create_subscription(JointState, state_topic, self._on_state, 10)

            self.get_logger().info(
                f"JointStateBridge: subscribe {state_topic} ({state_source}) "
                f"→ {output_topic} mesh URDF "
                f"(model={hand_joint}, mapped_joints={len(fk._active_mapping)})"
            )

        def _on_state(self, msg: JointState) -> None:
            parsed = mesh_urdf_joint_state_from_feedback(
                msg,
                hand_joint=self._hand_joint,
                hand_side=self._hand_side,
                fk=self._fk,
                state_source=self._state_source,
                config=self._config,
                radian_converter=self._rad_conv,
            )
            if parsed is None:
                return
            names, positions = parsed
            out = JointState()
            out.header.stamp = self.get_clock().now().to_msg()
            out.name = names
            out.position = positions
            self._pub.publish(out)

except ImportError:  # pragma: no cover - 非 ROS 环境无 rclpy
    JointStateBridge = None  # type: ignore[misc, assignment]


def main(args=None):
    if JointStateBridge is None:
        raise RuntimeError("rclpy unavailable; joint_state_bridge requires ROS 2")
    import rclpy

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
