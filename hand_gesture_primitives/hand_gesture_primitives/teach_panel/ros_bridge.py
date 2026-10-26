"""ROS 桥：侧化 gesture 指令 + ros2_control / legacy 关节指令。

面板内部使用 layout 手型的 hardware 向量（如整机 G20 → L25 native25）。
对外：radians 经 O20 语义 + URDF 换算；hardware 经产品手型向量（G20×16）再按需 remap。
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray, Header, String

from ..hand_config import HandConfig, semantic_legacy255_to_hardware
from ..joint_state_mapping import (
    Ros2ControlRadianConverter,
    extract_o6_from_joint_state,
    extract_o20_from_joint_state,
    extract_o30_from_joint_state,
    joint_state_to_g20_hardware,
    joint_state_to_l25_native,
    native25_to_active16,
    resolve_g20_l25_urdf,
    resolve_radian_hand_type,
    semantic_to_controller_hardware,
)
from .layout_convert import product_hardware_to_ui


class RosBridge:
    def __init__(
        self,
        node: Node,
        *,
        hand_side: str,
        product_hand_key: str,
        layout_hand_key: str,
        layout_hand: HandConfig,
        cmd_topic: str,
        status_topic: str,
        result_topic: str,
        state_topic: str,
        cmd_backend: str,
        hand_cmd_topic: str,
        ros2_control_command_units: str,
        robot_description: str,
        on_status: Callable[[str], None],
        on_state: Callable[[List[float]], None],
        on_result: Optional[Callable[[str], None]] = None,
        g20_joint_order: str = "",
        motion_progress_topic: str = "",
        feedback_only: bool = False,
        on_controller_state: Optional[Callable[[List[float]], None]] = None,
    ) -> None:
        self._node = node
        self._feedback_only = feedback_only
        self._on_controller_state = on_controller_state
        self._layout_hand = layout_hand
        self._product_key = (product_hand_key or "").strip().lower()
        self._layout_key = (layout_hand_key or "").strip().lower()
        self._side = hand_side.strip().lower()
        self._num_joints = int(layout_hand.num_joints)
        self._joint_names = list(layout_hand.joint_names)
        self._on_status = on_status
        self._on_state = on_state
        self._on_result = on_result
        self._product_hand = HandConfig(self._product_key)
        self._g20_l25_urdf = False
        if self._product_key == "g20":
            self._g20_l25_urdf = resolve_g20_l25_urdf(
                g20_joint_order, robot_description or "", self._side
            )

        backend = (cmd_backend or "").strip().lower()
        if not backend:
            backend = (
                "ros2_control"
                if self._product_key in ("l25", "o6", "l6", "o20", "o30", "g20")
                else "legacy_cb"
            )
        if backend not in ("ros2_control", "legacy_cb"):
            raise ValueError(f"未知 cmd_backend: {cmd_backend}")
        self._cmd_backend = backend

        units = (ros2_control_command_units or "radians").strip().lower()
        if units not in ("hardware", "radians"):
            raise ValueError(f"ros2_control_command_units 无效: {ros2_control_command_units}")
        if backend != "ros2_control" and units != "hardware":
            raise ValueError("radians 仅适用于 cmd_backend=ros2_control")
        self._units = units
        self._radian_converter: Optional[Ros2ControlRadianConverter] = None
        if units == "radians":
            radian_hand_type = resolve_radian_hand_type(
                self._product_key,
                g20_joint_order=g20_joint_order,
                robot_description=robot_description or "",
                hand_side=self._side,
            )
            # 整机 G20 与 L25 共用同一 URDF 16 序 / 语义索引
            if radian_hand_type == "g20_l25_urdf":
                radian_hand_type = "l25"
            self._radian_converter = Ros2ControlRadianConverter(
                robot_description, radian_hand_type, self._side
            )

        self._cmd_pub = self._progress_pub = self._joint_pub = None
        self._hand_cmd_topic = hand_cmd_topic
        self._motion_progress_topic = motion_progress_topic
        if not feedback_only:
            self._cmd_pub = node.create_publisher(String, cmd_topic, 10)
            prog_topic = (motion_progress_topic or "").strip() or f"/{self._side}_hand/motion_progress"
            self._progress_pub = node.create_publisher(Float32, prog_topic, 10)
            self._motion_progress_topic = prog_topic

            if backend == "ros2_control":
                topic = (
                    hand_cmd_topic.strip()
                    if hand_cmd_topic.strip()
                    else f"/{self._side}_hand_controller/commands"
                )
                self._joint_pub = node.create_publisher(Float64MultiArray, topic, 10)
                self._hand_cmd_topic = topic
            else:
                topic = (
                    hand_cmd_topic.strip()
                    if hand_cmd_topic.strip()
                    else f"/cb_{self._side}_hand_control_cmd"
                )
                self._joint_pub = node.create_publisher(JointState, topic, 10)
                self._hand_cmd_topic = topic

        node.create_subscription(String, status_topic, self._status_cb, 10)
        node.create_subscription(JointState, state_topic, self._state_cb, 10)
        if state_topic == "/joint_states":
            node.create_subscription(
                JointState,
                f"/cb_{self._side}_hand_state",
                self._legacy_state_cb,
                10,
            )
        if on_result is not None:
            node.create_subscription(String, result_topic, self._result_cb, 10)

        note = ""
        if self._product_key != self._layout_key:
            note = f" product={self._product_key} layout={self._layout_key}"
        elif self._product_key == "g20":
            note = f" g20_order={'l25_urdf' if self._g20_l25_urdf else 'native'}"
        node.get_logger().info(
            f"[teach_panel] bridge backend={backend}/{units} "
            f"cmd={cmd_topic} hand_cmd={self._hand_cmd_topic} state={state_topic}"
            f" progress={self._motion_progress_topic}{note}"
        )

    @property
    def radian_converter(self):
        return self._radian_converter

    def send_cmd(self, text: str) -> None:
        if self._feedback_only:
            raise RuntimeError("workbench motion must use the shared gesture_node client")
        msg = String()
        msg.data = text
        self._cmd_pub.publish(msg)
        self._node.get_logger().info(f"[teach_panel] cmd: {text}")

    def publish_progress(self, progress: float) -> None:
        if self._feedback_only:
            raise RuntimeError("workbench motion must use the shared gesture_node client")
        msg = Float32()
        msg.data = float(max(0.0, min(1.0, progress)))
        self._progress_pub.publish(msg)

    def publish_joints(self, positions: Sequence[float]) -> None:
        if self._feedback_only:
            raise RuntimeError("workbench motion must use the shared gesture_node client")
        if len(positions) != self._num_joints:
            return
        ui = [max(0.0, min(255.0, float(v))) for v in positions]

        if self._cmd_backend == "ros2_control":
            # 统一走布局→语义→控制器序，避免 L25 abd[6-9] 被 native25_to_active16 丢掉
            semantic = self._layout_hand.from_hardware(ui)
            if self._units == "radians":
                if self._radian_converter is None:
                    return
                msg = Float64MultiArray()
                msg.data = self._radian_converter.semantic_to_radians(semantic)
                self._joint_pub.publish(msg)
                return

            # hardware：按产品控制器序打包（L25/整机 G20 用 l25 语义槽，含 abd）
            if self._product_key == "g20" and self._g20_l25_urdf:
                pack_key, pack_cfg = "l25", HandConfig("l25")
            elif self._product_key == "l25":
                pack_key, pack_cfg = "l25", HandConfig("l25")
            else:
                pack_key, pack_cfg = self._product_key, self._product_hand
            data = semantic_to_controller_hardware(pack_key, semantic, pack_cfg)
            msg = Float64MultiArray()
            msg.data = [max(0.0, min(255.0, float(a))) for a in data]
            self._joint_pub.publish(msg)
            return

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(self._joint_names)
        msg.position = ui
        self._joint_pub.publish(msg)

    def _status_cb(self, msg: String) -> None:
        self._on_status(msg.data)

    def _legacy_state_cb(self, msg: JointState) -> None:
        # CB feedback is hardware-valued even when the command backend uses radians.
        if self._feedback_only and self._units == "radians":
            product_hw = self._product_hw_from_msg(msg)
            if product_hw is not None:
                try:
                    ui = product_hardware_to_ui(product_hw, product_key=self._product_key,
                                                layout_key=self._layout_key)
                    self._on_state(ui)
                except ValueError:
                    pass
            return
        self._dispatch_state(msg)

    def _state_cb(self, msg: JointState) -> None:
        self._dispatch_state(msg)

    def _dispatch_state(self, msg: JointState) -> None:
        if self._on_controller_state is not None and self._units == "radians":
            positions = self._controller_state(msg)
            if positions is not None:
                self._on_controller_state(positions)
            return
        ui = self._extract_ui(msg)
        if ui is None:
            return
        self._on_state(ui)

    def _product_hw_from_msg(self, msg: JointState) -> Optional[List[float]]:
        """从 /joint_states 抽产品 hardware 向量。"""
        if not msg.name:
            if len(msg.position) == self._product_hand.num_joints:
                return [float(v) for v in msg.position]
            return None

        if self._product_key == "l25":
            native = joint_state_to_l25_native(msg.name, msg.position, self._side)
            return list(native) if native is not None else None
        if self._product_key in ("o6", "l6"):
            return extract_o6_from_joint_state(msg.name, msg.position, self._side)
        if self._product_key == "o20":
            hw = extract_o20_from_joint_state(msg.name, msg.position, self._side)
            return list(hw) if hw is not None else None
        if self._product_key == "g20":
            hw, _uses = joint_state_to_g20_hardware(msg.name, msg.position, self._side)
            return list(hw) if hw is not None else None
        return None

    def _controller_state(self, msg: JointState) -> Optional[List[float]]:
        """Preserve radians exactly; editor capture must not round through uint8."""
        if not msg.name or self._radian_converter is None:
            return None
        if self._product_key in ("l25", "g20") and (self._product_key == "l25" or self._g20_l25_urdf):
            native = joint_state_to_l25_native(msg.name, msg.position, self._side)
            return native25_to_active16(native) if native is not None else None
        if self._product_key == "g20":
            positions, uses_l25 = joint_state_to_g20_hardware(msg.name, msg.position, self._side)
            return list(positions) if positions is not None and not uses_l25 else None
        if self._product_key in ("o6", "l6"):
            return extract_o6_from_joint_state(msg.name, msg.position, self._side)
        if self._product_key == "o20":
            return extract_o20_from_joint_state(msg.name, msg.position, self._side)
        if self._product_key == "o30":
            return extract_o30_from_joint_state(msg.name, msg.position, self._side)
        return None

    def _extract_ui(self, msg: JointState) -> Optional[List[float]]:
        if self._cmd_backend == "ros2_control" and self._units == "radians":
            controller_state = self._controller_state(msg)
            if controller_state is None:
                return None
            semantic = self._radian_converter.radians_to_semantic(controller_state)
            return semantic_legacy255_to_hardware(self._layout_hand, semantic)

        product_hw = self._product_hw_from_msg(msg)
        if product_hw is None:
            if len(msg.position) == self._num_joints:
                return [float(v) for v in msg.position]
            return None
        try:
            return product_hardware_to_ui(
                product_hw,
                product_key=self._product_key,
                layout_key=self._layout_key,
            )
        except ValueError:
            return None

    def _result_cb(self, msg: String) -> None:
        if self._on_result is not None:
            self._on_result(msg.data)
