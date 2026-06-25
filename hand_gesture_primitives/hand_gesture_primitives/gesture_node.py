"""手势原语 ROS2 控制节点。"""

import time
from typing import Union

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterType
from rclpy.node import Node
from std_msgs.msg import String

from .contact_config import ContactThresholds, load_contact_thresholds
from .executor import GestureExecutor
from .grasp_mapping import GRASP_TYPE_TO_PRIMITIVE, primitive_for_grasp_type
from .label_utils import normalize_label
from .primitive_base import GRASP_PHASES, PHASED_GRASP_PRIMITIVES
from .grasp_gate import GATED_PRIMITIVES
from .primitives import PRIMITIVE_REGISTRY
from .topic_defaults import bboxes_3d_topic, labeled_poses_topic


class GestureNode(Node):
    """接收手势指令并调度对应原语执行。"""

    _STATUS_OVERRIDE_SEC = 3.0

    @staticmethod
    def _coerce_numeric(value: Union[int, float, str], kind: str):
        if kind == "int":
            return int(value)
        return float(value)

    def _declare_numeric_parameter(
        self, name: str, default: Union[int, float], kind: str = "float",
    ) -> None:
        """Declare float/int param; coerce launch INTEGER overrides to float."""
        if self.has_parameter(name):
            existing = self.get_parameter(name)
            if existing.type_ == ParameterType.PARAMETER_NOT_SET:
                pass
            elif kind == "float" and existing.type_ == ParameterType.PARAMETER_INTEGER:
                coerced = float(existing.value)
                self.undeclare_parameter(name)
                self.declare_parameter(name, coerced)
                return
            elif existing.type_ in (
                ParameterType.PARAMETER_DOUBLE,
                ParameterType.PARAMETER_INTEGER,
            ):
                self.undeclare_parameter(name)
                self.declare_parameter(
                    name, self._coerce_numeric(existing.value, kind))
                return
        self.declare_parameter(name, default)

    def __init__(self):
        super().__init__("hand_gesture_node")

        self.declare_parameter("hand_side", "left")
        self.declare_parameter("hand_type", "o20")
        self.declare_parameter("hand_joint", "")
        self.declare_parameter("cmd_topic", "/hand_gesture_cmd_exec")
        self.declare_parameter("tcp_pose_topic", "/tcp_pose")
        self.declare_parameter("object_pose_topic", "/object_pose")
        self.declare_parameter("bboxes_3d_topic", bboxes_3d_topic())
        self.declare_parameter("labeled_pose_topic", labeled_poses_topic())
        self.declare_parameter("target_object_label", "")
        self.declare_parameter("result_topic", "/hand_gesture/result")
        self.declare_parameter("selected_label_topic", "/grasp_gate/selected_label")
        self.declare_parameter("reach_check_strict", False)
        self.declare_parameter("reach_max_distance", 0.15)
        self.declare_parameter("palm_offset_x", 0.0)
        self.declare_parameter("palm_offset_y", 0.0)
        self.declare_parameter("palm_offset_z", -0.05)
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("publish_fingertips", True)
        self.declare_parameter("fingertip_frame", "")
        self.declare_parameter("contact_config_path", "")
        config_path = self.get_parameter("contact_config_path").value.strip()
        loaded_thresholds = load_contact_thresholds(config_path or None)
        self._declare_numeric_parameter(
            "contact_pressure_threshold", loaded_thresholds.pressure_threshold)
        self._declare_numeric_parameter(
            "contact_mass_threshold", loaded_thresholds.mass_threshold)
        self._declare_numeric_parameter(
            "contact_current_delta", loaded_thresholds.current_delta)
        self._declare_numeric_parameter(
            "contact_current_delta_narrow", loaded_thresholds.current_delta_narrow)
        self._declare_numeric_parameter(
            "contact_current_settle_frames",
            loaded_thresholds.current_settle_frames,
            kind="int",
        )
        self._declare_numeric_parameter(
            "hold_safe_current", loaded_thresholds.hold_safe_current)
        self._declare_numeric_parameter(
            "overload_current_threshold", loaded_thresholds.overload_threshold)
        self._declare_numeric_parameter(
            "overload_duration_sec", loaded_thresholds.overload_duration_sec)

        self._side = self.get_parameter("hand_side").value
        hand_type = self.get_parameter("hand_type").value
        hand_joint = self.get_parameter("hand_joint").value.strip()
        if hand_joint:
            hand_type = hand_joint.lower()
        self._hand_type = hand_type
        cmd_topic = self.get_parameter("cmd_topic").value
        tcp_topic = self.get_parameter("tcp_pose_topic").value
        obj_topic = self.get_parameter("object_pose_topic").value
        bboxes_topic = self.get_parameter("bboxes_3d_topic").value
        labeled_topic = self.get_parameter("labeled_pose_topic").value
        target_label = self.get_parameter("target_object_label").value
        result_topic = self.get_parameter("result_topic").value
        sel_label_topic = self.get_parameter("selected_label_topic").value
        reach_strict = self.get_parameter("reach_check_strict").value
        reach_max_dist = self.get_parameter("reach_max_distance").value
        palm_offset = np.array([
            self.get_parameter("palm_offset_x").value,
            self.get_parameter("palm_offset_y").value,
            self.get_parameter("palm_offset_z").value,
        ])
        urdf_path = self.get_parameter("urdf_path").value
        publish_fingertips = self.get_parameter("publish_fingertips").value
        fingertip_frame = self.get_parameter("fingertip_frame").value

        contact_thresholds = ContactThresholds(
            pressure_threshold=float(
                self.get_parameter("contact_pressure_threshold").value),
            mass_threshold=float(
                self.get_parameter("contact_mass_threshold").value),
            current_delta=float(
                self.get_parameter("contact_current_delta").value),
            current_delta_narrow=float(
                self.get_parameter("contact_current_delta_narrow").value),
            current_settle_frames=int(
                self.get_parameter("contact_current_settle_frames").value),
            hold_safe_current=float(
                self.get_parameter("hold_safe_current").value),
            overload_threshold=float(
                self.get_parameter("overload_current_threshold").value),
            overload_duration_sec=float(
                self.get_parameter("overload_duration_sec").value),
        )

        if self._side not in ("left", "right"):
            self.get_logger().error(f"hand_side 参数无效: '{self._side}'，必须为 left 或 right")
            raise ValueError(f"Invalid hand_side: {self._side}")

        if self._hand_type not in ("o20", "l25"):
            self.get_logger().error(f"hand_type 参数无效: '{self._hand_type}'，必须为 o20 或 l25")
            raise ValueError(f"Invalid hand_type: {self._hand_type}")

        self.get_logger().info(f"手势节点启动: side={self._side}, hand_type={self._hand_type}")
        n_prims = len(PRIMITIVE_REGISTRY)
        self.get_logger().info(
            f"已注册原语 ({n_prims}): {list(PRIMITIVE_REGISTRY.keys())}"
        )
        self.get_logger().info(
            f"包络门控原语 ({len(GATED_PRIMITIVES)}): {sorted(GATED_PRIMITIVES)}"
        )
        self.get_logger().info(
            f"grasp_type 自动映射: {GRASP_TYPE_TO_PRIMITIVE} (命令: auto [label])"
        )

        self.get_logger().info(
            f"接触/电流阈值: delta={contact_thresholds.current_delta}mA, "
            f"narrow={contact_thresholds.current_delta_narrow}mA, "
            f"hold_safe={contact_thresholds.hold_safe_current}mA, "
            f"overload={contact_thresholds.overload_threshold}mA"
        )

        self._executor = GestureExecutor(
            self,
            self._side,
            hand_type=self._hand_type,
            tcp_pose_topic=tcp_topic,
            object_pose_topic=obj_topic,
            bboxes_3d_topic=bboxes_topic,
            labeled_pose_topic=labeled_topic,
            target_object_label=target_label,
            result_topic=result_topic,
            selected_label_topic=sel_label_topic,
            reach_strict=reach_strict,
            reach_max_dist=reach_max_dist,
            palm_offset=palm_offset,
            urdf_path=urdf_path,
            publish_fingertips=publish_fingertips,
            fingertip_frame=fingertip_frame,
            contact_thresholds=contact_thresholds,
        )

        self.get_logger().info(
            f"订阅指令 topic: {cmd_topic}, 感知 topic: {labeled_topic}")

        self.create_subscription(String, cmd_topic, self._cmd_callback, 10)

        self._status_pub = self.create_publisher(String, "/hand_gesture_status", 10)
        self._status_override = ""
        self._status_override_until = 0.0
        self.create_timer(0.5, self._publish_status)

    def _set_status_override(self, message: str) -> None:
        self._status_override = message
        self._status_override_until = time.monotonic() + self._STATUS_OVERRIDE_SEC

    def _cmd_callback(self, msg: String) -> None:
        parts = msg.data.strip().lower().split()
        if not parts:
            return
        cmd = parts[0]
        args = parts[1:]
        self.get_logger().info(f"收到指令: '{msg.data.strip()}' (topic: /hand_gesture_cmd_exec)")

        if cmd == "stop":
            self._executor.stop()
            return

        if cmd == "target":
            label = normalize_label(args[0]) if args else ''
            self._executor.set_target_label(label)
            self.get_logger().info(f"目标物体过滤: '{label}' (空=最高score)")
            return

        if cmd == "target_id":
            if not args:
                self._executor.set_target_instance_id(0)
                self.get_logger().info("清除 instance_id 过滤")
                return
            try:
                instance_id = int(args[0])
            except ValueError:
                self.get_logger().warn(f"target_id 需要整数: '{args[0]}'")
                return
            self._executor.set_target_instance_id(instance_id)
            self.get_logger().info(f"目标 instance_id: {instance_id}")
            return

        if cmd == "auto":
            phase = "full"
            label_args = []
            for a in args:
                if a.lower() in GRASP_PHASES:
                    phase = a.lower()
                else:
                    label_args.append(a)
            label = normalize_label(label_args[0]) if label_args else ''
            if label:
                self._executor.set_target_label(label)
            prim_name = self._executor.suggested_primitive_name()
            if not prim_name:
                grasp = self._executor.grasp_type or "(无)"
                self.get_logger().warn(
                    f"auto 失败: 无可用 grasp_type={grasp}，"
                    f"请先确保 bboxes_3d 已发布或指定 label"
                )
                self._set_status_override(f"rejected:no_grasp_type:{grasp}")
                return
            primitive_cls = PRIMITIVE_REGISTRY.get(prim_name)
            if primitive_cls is None:
                self.get_logger().error(f"映射原语未注册: {prim_name}")
                return
            phase_info = f", phase={phase}" if prim_name in PHASED_GRASP_PRIMITIVES else ""
            self.get_logger().info(
                f"auto: grasp_type={self._executor.grasp_type} → {prim_name}{phase_info}"
                + (f", label={label}" if label else "")
            )
            if prim_name in PHASED_GRASP_PRIMITIVES:
                primitive = primitive_cls(phase=phase)
            else:
                if phase != "full":
                    self.get_logger().warn(
                        f"原语 {prim_name} 不支持 prep/close，忽略 phase={phase}"
                    )
                primitive = primitive_cls()
            reject_reason = self._executor.set_primitive(primitive)
            if reject_reason:
                self.get_logger().warn(f"原语被拒绝: {reject_reason}")
                self._set_status_override(reject_reason)
            return

        primitive_cls = PRIMITIVE_REGISTRY.get(cmd)
        if primitive_cls is None:
            hint = self._unknown_primitive_hint(cmd)
            self.get_logger().warn(
                f"未知原语指令: '{cmd}'，可用: {list(PRIMITIVE_REGISTRY.keys()) + ['auto']}"
                + (f"；{hint}" if hint else "")
            )
            return

        phase, target_label, prim_args = self._parse_primitive_args(cmd, args)

        try:
            if cmd in PHASED_GRASP_PRIMITIVES:
                if prim_args:
                    raise ValueError(
                        f"{cmd} 仅支持 phase 参数 (prep/close/full)，多余参数: {prim_args}"
                    )
                primitive = primitive_cls(phase=phase)
            else:
                if phase != "full":
                    self.get_logger().warn(
                        f"原语 {cmd} 不支持 prep/close，忽略 phase={phase}"
                    )
                primitive = primitive_cls(*prim_args)
        except (TypeError, ValueError) as e:
            self.get_logger().warn(f"原语参数错误: '{msg.data}' → {e}")
            return

        if target_label:
            self._executor.set_target_label(target_label)
        reject_reason = self._executor.set_primitive(primitive)
        if reject_reason:
            self.get_logger().warn(f"原语被拒绝: {reject_reason}")
            self._set_status_override(reject_reason)

    def _parse_primitive_args(self, cmd: str, args: list) -> tuple:
        """解析原语命令参数，返回 (phase, target_label, prim_args)。"""
        phase = "full"
        target_label = ''
        prim_args = []
        phased = cmd in PHASED_GRASP_PRIMITIVES

        for a in args:
            al = a.lower()
            if phased and al in GRASP_PHASES:
                phase = al
            elif (
                not target_label
                and a not in PRIMITIVE_REGISTRY
                and al not in GRASP_PHASES
                and not a.lstrip('-').replace('.', '', 1).isdigit()
            ):
                target_label = normalize_label(a)
            else:
                prim_args.append(a)
        return phase, target_label, prim_args

    @staticmethod
    def _unknown_primitive_hint(cmd: str) -> str:
        """原语名不匹配时给出可操作的提示。"""
        if cmd.endswith("_by_vision"):
            short = cmd[: -len("_by_vision")]
            if short in PRIMITIVE_REGISTRY:
                return f"可尝试短名 '{short}'（不经门控/旧版别名）"
            return "当前版本应含 *_by_vision，请 colcon build 并 source install"
        vision_name = f"{cmd}_by_vision"
        if vision_name in PRIMITIVE_REGISTRY:
            return f"产线推荐 '{vision_name}'（经 GraspGate 门控）"
        return ""

    def _publish_status(self) -> None:
        msg = String()
        if time.monotonic() < self._status_override_until:
            msg.data = self._status_override
        else:
            active = self._executor.active_primitive_name
            grasp_state = self._executor.grasp_state
            grasp = self._executor.grasp_type
            suggested = primitive_for_grasp_type(grasp) if grasp else ''
            parts = [active]
            if grasp_state:
                parts.append(f"grasp={grasp_state}")
            elif suggested and active in ('hold', 'init'):
                parts.append(f"type={grasp}")
                parts.append(f"suggest={suggested}")
            msg.data = "|".join(parts)
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
