"""10Hz 手势执行器：调度原语并发布关节指令。"""

import json
import time
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import Point, PoseStamped as PoseStampedMsg
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import ColorRGBA, Float32MultiArray, Header, MultiArrayDimension, String
from visualization_msgs.msg import Marker, MarkerArray

from .primitive_base import (
    HandGesturePrimitive,
    JOINT_NAMES,
    NUM_JOINTS,
    PHASED_GRASP_PRIMITIVES,
    RESERVED_INDICES,
    PoseStamped,
    PrimitiveContext,
    PrimitiveResult,
    HAND_CONFIGS,
)
from .contact_config import ContactThresholds, load_contact_thresholds
from .contact_detection import uses_torque_feedback
from .contact_resolver import (
    normalize_motor_torque_values,
    parse_hand_info_currents,
    parse_hand_info_torque,
)
from .fk_solver import create_fk_solver
from .grasp_mapping import primitive_for_grasp_type
from .hand_profile import HandProfile
from .label_utils import normalize_label, select_best_bbox
from .primitives.init_hand import InitHand
from .topic_defaults import bboxes_3d_topic, labeled_poses_topic


# 感知数据超时 (帧数, @5Hz 约 3s)
_BBOX_STALE_FRAMES = 15


class GestureExecutor:
    """以 10Hz 频率持续发布 20-DOF 目标角度。

    无论是否有活跃原语，始终 10Hz 下发控制信号。
    原语返回 infeasible 时保持上帧目标值继续发送。
    内置电流过载保护：任一关节电流超阈值持续过久自动 fallback 到 init。
    """

    _SAFE_PRIMITIVES = {'init', 'open', 'release', 'relax_grip'}

    def __init__(
        self,
        node: Node,
        hand_side: str,
        hand_type: str = "o20",
        hand_profile: Optional[HandProfile] = None,
        tcp_pose_topic: str = "/tcp_pose",
        object_pose_topic: str = "/object_pose",
        bboxes_3d_topic: str = bboxes_3d_topic(),
        labeled_pose_topic: str = labeled_poses_topic(),
        target_object_label: str = "",
        result_topic: str = "/hand_gesture/result",
        selected_label_topic: str = "/grasp_gate/selected_label",
        reach_strict: bool = False,
        reach_max_dist: float = 0.15,
        palm_offset: Optional[np.ndarray] = None,
        urdf_path: str = "",
        publish_fingertips: bool = True,
        fingertip_frame: str = "",
        contact_thresholds: Optional[ContactThresholds] = None,
    ):
        self._node = node
        self._side = hand_side
        self._hand_type = hand_type
        self._profile = hand_profile
        if hand_type not in HAND_CONFIGS:
            raise ValueError(f"未知 hand_type: {hand_type}")
        self._hand_config = HAND_CONFIGS[hand_type]
        self._primitive_space_dim = (
            20 if hand_type == "o6" else self._hand_config.num_joints
        )
        self._active_primitive: Optional[HandGesturePrimitive] = None
        self._primitive_start_time: float = 0.0
        self._current_angles: List[float] = [0.0] * self._primitive_space_dim
        self._last_target: List[float] = [0.0] * self._hand_config.num_joints
        self._contact_thresholds = contact_thresholds or load_contact_thresholds()

        # 可达性检查参数
        self._reach_strict: bool = reach_strict
        self._reach_max_dist: float = reach_max_dist
        self._palm_offset: np.ndarray = palm_offset if palm_offset is not None else np.zeros(3)

        # 电流过载保护状态
        self._joint_overload_start: List[Optional[float]] = [None] * self._hand_config.num_joints
        # 电流反馈缓存 (mA, position order)
        self._joint_currents: List[float] = [0.0] * self._hand_config.num_joints
        self._hand_state_received: bool = False

        # 外部位姿缓存
        self._tcp_pose: Optional[PoseStamped] = None
        self._object_pose: Optional[PoseStamped] = None
        self._object_label: str = ""
        self._target_object_label = target_object_label

        # 感知输出: 目标物体几何缓存
        self._object_size: Optional[np.ndarray] = None
        self._object_orientation: Optional[np.ndarray] = None
        self._object_center: Optional[np.ndarray] = None
        self._grasp_type: str = ''
        self._target_label: str = ''
        self._target_instance_id: Optional[int] = None
        self._object_center_frame: str = ''
        self._bbox_miss_count: int = 0

        # 发布关节指令
        self._cmd_pub = node.create_publisher(
            JointState,
            f"/cb_{hand_side}_hand_control_cmd",
            10,
        )

        # 订阅手部状态反馈
        node.create_subscription(
            JointState,
            f"/cb_{hand_side}_hand_state",
            self._state_callback,
            10,
        )

        # 订阅手部信息（电流/温度/力矩）
        node.create_subscription(
            String,
            f"/cb_{hand_side}_hand_info",
            self._info_callback,
            10,
        )

        self._tcp_pose_topic = tcp_pose_topic
        self._object_pose_topic = object_pose_topic

        # 订阅 TCP 位姿
        node.create_subscription(
            PoseStampedMsg,
            tcp_pose_topic,
            self._tcp_pose_callback,
            10,
        )

        # 订阅物体位姿 (相对手腕)
        node.create_subscription(
            PoseStampedMsg,
            object_pose_topic,
            self._object_pose_callback,
            10,
        )

        # 订阅感知 3D 包围盒 (获取物体尺寸/方向/抓取类型)
        try:
            from robot_perception_msgs.msg import LabeledBBox3DArray
            node.create_subscription(
                LabeledBBox3DArray,
                bboxes_3d_topic,
                self._bboxes_3d_callback,
                10,
            )
        except ImportError:
            self._node.get_logger().warn(
                "robot_perception_msgs.LabeledBBox3DArray 不可用，禁用 3D bbox 订阅")

        # 订阅感知 pipeline 的 LabeledPoseArray (world frame)
        try:
            from robot_perception_msgs.msg import LabeledPoseArray
            node.create_subscription(
                LabeledPoseArray,
                labeled_pose_topic,
                self._labeled_pose_callback,
                10,
            )
            self._node.get_logger().info(
                f"已订阅感知 topic: {labeled_pose_topic}, 目标物体: {target_object_label}")
        except ImportError:
            self._node.get_logger().warn(
                "robot_perception_msgs.LabeledPoseArray 不可用，仅使用 /object_pose 输入")

        # 触觉反馈 (兼容 O20 hand_force 和 SDK normal_force)
        self._tactile_pressure: Optional[np.ndarray] = None
        self._tactile_mode: str = "none"
        self._contact_detected: bool = False
        self._force_updated: bool = False
        node.create_subscription(
            Float32MultiArray,
            f"/cb_{hand_side}_hand_force",
            self._force_callback,
            10,
        )
        node.create_subscription(
            Float32MultiArray,
            f"/cb_{hand_side}_hand_normal_force",
            self._normal_force_callback,
            10,
        )
        node.create_subscription(
            String,
            f"/cb_{hand_side}_hand_matrix_touch_mass",
            self._matrix_touch_mass_callback,
            10,
        )

        # 力矩反馈 (SDK 独立 topic)
        self._joint_torque: Optional[np.ndarray] = None
        node.create_subscription(
            Float32MultiArray,
            f"/cb_{hand_side}_hand_motor_torque",
            self._torque_callback,
            10,
        )

        # FK 正运动学求解器
        hand_joint = hand_type.upper()
        self._fk_solver = create_fk_solver(hand_joint, hand_side, urdf_path,
                                           logger=node.get_logger())
        self._publish_fingertips = publish_fingertips and self._fk_solver is not None
        if self._publish_fingertips:
            self._fingertip_pub = node.create_publisher(
                Float32MultiArray,
                f"/cb_{hand_side}_hand_fingertip_positions",
                10,
            )
            self._fingertip_marker_pub = node.create_publisher(
                MarkerArray,
                f"/cb_{hand_side}_hand_fingertip_markers",
                10,
            )
            if fingertip_frame:
                self._fingertip_frame = fingertip_frame
            elif self._fk_solver:
                self._fingertip_frame = self._fk_solver.base_link
            else:
                side_prefix = hand_side[0]
                self._fingertip_frame = f"{side_prefix}_hand_base_link"

        # 10Hz 定时器 — 始终运行
        self._timer = node.create_timer(0.1, self._tick)

        # 触觉日志: 仅状态变化时输出
        self._last_grasp_state: str = ""
        self._infeasible_hold_logged: bool = False
        self._motion_logged: bool = False

        # 执行结果反馈 (10Hz 节流到 ~1Hz 实际发布)
        self._result_pub = node.create_publisher(String, result_topic, 10)
        self._last_result_msg = ""
        self._result_msg_count = 0

        # 从门控接收选中的物体 label
        node.create_subscription(
            String,
            selected_label_topic,
            self._selected_label_callback,
            10,
        )

        self._node.get_logger().info(
            f"GestureExecutor 已启动: side={hand_side}, 10Hz 持续输出, "
            f"电流保护: >{self._contact_thresholds.overload_threshold}mA"
            f"持续>{self._contact_thresholds.overload_duration_sec}s"
        )
        self._node.get_logger().info(
            f"已订阅 tcp_pose: {self._tcp_pose_topic}, "
            f"object_pose: {self._object_pose_topic}, "
            f"hand_state: /cb_{hand_side}_hand_state, "
            f"hand_cmd: /cb_{hand_side}_hand_control_cmd"
        )

        # 启动时自动执行 init 原语
        self.set_primitive(InitHand())

    @property
    def active_primitive_name(self) -> str:
        if self._active_primitive is None:
            return "hold"
        return self._active_primitive.name

    @property
    def grasp_state(self) -> str:
        if self._active_primitive is None:
            return ""
        return getattr(self._active_primitive, 'grasp_state', "") or ""

    def set_target_label(self, label: str) -> None:
        """设置感知目标过滤 label (空=最高 score)。支持去标点与子串匹配。"""
        normalized = normalize_label(label)
        if normalized != self._target_label:
            self._target_label = normalized
            self._clear_object_cache()

    def set_target_instance_id(self, instance_id: int) -> None:
        """按 instance_id 过滤 (0 或负数=清除)。"""
        new_id = instance_id if instance_id > 0 else None
        if new_id != self._target_instance_id:
            self._target_instance_id = new_id
            self._clear_object_cache()

    def _clear_object_cache(self) -> None:
        self._object_size = None
        self._object_orientation = None
        self._object_center = None
        self._object_center_frame = ''
        self._grasp_type = ''

    @property
    def grasp_type(self) -> str:
        return self._grasp_type

    def suggested_primitive_name(self) -> str:
        """根据当前 grasp_type 推荐自适应原语名。"""
        if self._profile is not None:
            prim = self._profile.primitive_for_grasp_type(self._grasp_type)
            if prim:
                return prim
        prim = primitive_for_grasp_type(self._grasp_type)
        if prim and self._profile is not None and not self._profile.is_primitive_supported(prim):
            return ''
        return prim or ''

    def _check_reachability(self) -> str:
        """检查手掌→目标可达性。返回空=通过, 非空=原因。"""
        if self._object_pose is not None:
            dist = float(np.linalg.norm(self._object_pose.position - self._palm_offset))
            if dist > self._reach_max_dist:
                return f"too_far:{dist:.3f}m>(max:{self._reach_max_dist:.3f}m)"
            return ''

        if self._tcp_pose is None:
            return "no_tcp"
        if self._object_center is None:
            return "no_target"

        tcp_frame = self._tcp_pose.frame_id
        bbox_frame = self._object_center_frame
        if tcp_frame and bbox_frame and tcp_frame != bbox_frame:
            return f"frame_mismatch:tcp={tcp_frame},bbox={bbox_frame}"

        palm_pos = self._tcp_pose.position + self._palm_offset
        dist = float(np.linalg.norm(palm_pos - self._object_center))
        if dist > self._reach_max_dist:
            return f"too_far:{dist:.3f}m>(max:{self._reach_max_dist:.3f}m)"
        return ""

    def set_primitive(self, primitive: HandGesturePrimitive) -> str:
        """切换到新原语。返回空=成功, 非空=拒绝原因。"""
        new_phase = getattr(primitive, 'phase', 'full')
        if new_phase == 'close' and primitive.name in PHASED_GRASP_PRIMITIVES:
            prev = self._active_primitive
            if prev is None or prev.name != primitive.name:
                self._node.get_logger().warn(
                    f"'{primitive.name} close' 未检测到同原语 prep，从当前关节姿态闭合"
                )
            elif getattr(prev, 'grasp_state', '') != 'ready':
                self._node.get_logger().warn(
                    f"'{primitive.name} close' 时 grasp_state="
                    f"{getattr(prev, 'grasp_state', '')} (非 ready)，从当前姿态继续"
                )

        # 安全原语不做可达性检查
        if primitive.name not in self._SAFE_PRIMITIVES:
            reason = self._check_reachability()
            if reason:
                if self._reach_strict:
                    self._node.get_logger().error(
                        f"可达性检查失败, 拒绝执行 '{primitive.name}': {reason}"
                    )
                    return f"rejected:{reason}"
                else:
                    self._node.get_logger().warn(
                        f"可达性警告: {reason} (非严格模式, 继续执行 '{primitive.name}')"
                    )

        if self._active_primitive is not None:
            self._node.get_logger().info(
                f"原语切换: {self._active_primitive.name} → {primitive.name}"
            )
            self._active_primitive.on_exit()
        else:
            self._node.get_logger().info(f"激活原语: {primitive.name}")

        self._active_primitive = primitive
        self._primitive_start_time = time.monotonic()
        self._last_grasp_state = ""
        self._infeasible_hold_logged = False
        self._motion_logged = False
        primitive.on_enter(self._current_angles)
        self._log_primitive_context(primitive.name)
        return ""

    def stop(self) -> None:
        """停止原语，保持当前姿态继续发送。"""
        if self._active_primitive is not None:
            self._node.get_logger().info(f"停止原语: {self._active_primitive.name}")
            self._active_primitive.on_exit()
            self._active_primitive = None

    def _state_callback(self, msg: JointState) -> None:
        n = len(msg.position)
        if self._hand_type == "o6" and self._profile is not None:
            if n == self._profile.hardware.num_joints:
                self._current_angles = self._profile.hardware.from_hardware(
                    list(msg.position))
                self._hand_state_received = True
            return
        if n == self._hand_config.num_joints:
            self._current_angles = list(msg.position)
            self._hand_state_received = True

    def _semantic_to_hardware(self, semantic: List[float]) -> List[float]:
        """原语关节空间 → 驱动发布向量。"""
        o20 = list(semantic[:20])
        while len(o20) < 20:
            o20.append(0.0)
        for i in RESERVED_INDICES:
            o20[i] = 0.0
        if self._hand_type == "o6" and self._profile is not None:
            hw = self._profile.hardware.to_hardware(o20)
            return [max(0.0, min(255.0, v)) for v in hw]
        n = self._hand_config.num_joints
        target = list(o20)
        if len(target) < n:
            target.extend([0.0] * (n - len(target)))
        target = target[:n]
        for i in self._hand_config.reserved_indices:
            if i < len(target):
                target[i] = 0.0
        if self._hand_config.invert_angles:
            target = [255.0 - v for v in target]
        return [max(0.0, min(255.0, v)) for v in target]

    def _log_primitive_context(self, primitive_name: str) -> None:
        """原语切换时输出感知/驱动上下文，便于排查静默不执行。"""
        tcp = "有" if self._tcp_pose is not None else "无"
        obj = "有" if self._object_pose is not None else "无"
        bbox = "有" if self._object_center is not None else "无"
        label = self._object_label or self._target_object_label or "(未指定)"
        hand_state = "已收到" if self._hand_state_received else "未收到(仍用零位/上帧)"
        subs = self._cmd_pub.get_subscription_count()
        self._node.get_logger().info(
            f"原语 '{primitive_name}' 上下文: tcp={tcp}, object_pose={obj}, "
            f"bbox={bbox}, label={label}, hand_state={hand_state}, "
            f"cmd订阅数={subs} (/cb_{self._side}_hand_control_cmd)"
        )
        if subs == 0:
            self._node.get_logger().warn(
                f"无节点订阅 /cb_{self._side}_hand_control_cmd，"
                "关节指令不会被手驱动执行"
            )

    def _log_infeasible_hold(
        self, primitive_name: str, result: PrimitiveResult, ctx: PrimitiveContext,
    ) -> None:
        if self._infeasible_hold_logged:
            return
        reason = result.hold_reason or self._infer_hold_reason(ctx)
        self._node.get_logger().warn(
            f"原语 '{primitive_name}' 返回 hold (infeasible): {reason}；"
            "保持上帧目标角度，手部可能看起来无动作"
        )
        self._infeasible_hold_logged = True

    @staticmethod
    def _infer_hold_reason(ctx: PrimitiveContext) -> str:
        if ctx.tcp_pose is None:
            return "缺少 /tcp_pose（常见：未发布或 topic 不一致）"
        if ctx.object_pose is None:
            return "缺少 /object_pose（GraspGate 未放行或未发布选中位姿）"
        return "原语内部条件未满足（如前向距离 2–15cm 等，详见原语实现）"

    def _log_motion_started(self, primitive_name: str, target: List[float]) -> None:
        if self._motion_logged:
            return
        if self._hand_type == "o6" and self._profile is not None:
            hw_current = self._profile.hardware.to_hardware(
                list(self._current_angles[:20]))
            compare = hw_current
        else:
            compare = self._current_angles
        n = min(len(target), len(compare))
        if n == 0:
            return
        delta = max(abs(target[i] - compare[i]) for i in range(n))
        if delta < 1.0:
            return
        self._node.get_logger().info(
            f"原语 '{primitive_name}' 开始输出目标角度，"
            f"最大关节变化={delta:.1f} (0-255)"
        )
        self._motion_logged = True

    def _info_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, ValueError):
            return
        n = self._hand_config.num_joints
        currents = parse_hand_info_currents(data, n)
        torques = (
            parse_hand_info_torque(data, n)
            if uses_torque_feedback(self._hand_type) else None
        )
        if currents is None and torques is None:
            return

        if currents is not None:
            # SDK 未收到 CAN 0x36 前为 -1；仅更新有效采样，保留上帧值
            for i, c in enumerate(currents):
                if c >= 0:
                    self._joint_currents[i] = c

        if torques is not None:
            if self._joint_torque is None:
                self._joint_torque = np.full(n, -1.0, dtype=np.float64)
            for i, t in enumerate(torques):
                if t >= 0:
                    self._joint_torque[i] = t
        now = time.monotonic()
        SKIP_OVERLOAD = {self._hand_config.thumb_rot}
        reserved = set(self._hand_config.reserved_indices)
        overload = self._contact_thresholds.overload_threshold
        overload_duration = self._contact_thresholds.overload_duration_sec
        for i, c in enumerate(self._joint_currents):
            if c < 0:
                continue
            if i in reserved or i in SKIP_OVERLOAD:
                continue
            if c > overload:
                if self._joint_overload_start[i] is None:
                    self._joint_overload_start[i] = now
                elif now - self._joint_overload_start[i] > overload_duration:
                    self._trigger_protection(i, c)
                    return
            else:
                self._joint_overload_start[i] = None

    def _trigger_protection(self, joint_idx: int, current_value: int) -> None:
        joint_name = JOINT_NAMES[joint_idx] if (self._hand_type == "o20" and joint_idx < len(JOINT_NAMES)) else f"[{joint_idx}]"
        t = self._contact_thresholds
        self._node.get_logger().warn(
            f"电流过载保护触发: {joint_name}[{joint_idx}] 电流={current_value}mA, "
            f"超过{t.overload_threshold}mA持续>{t.overload_duration_sec}s → fallback到init"
        )
        self._joint_overload_start = [None] * self._hand_config.num_joints
        self.set_primitive(InitHand())

    def _tcp_pose_callback(self, msg: PoseStampedMsg) -> None:
        self._tcp_pose = PoseStamped(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            orientation=np.array([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w,
            ]),
            frame_id=msg.header.frame_id or '',
            stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        )

    def _object_pose_callback(self, msg: PoseStampedMsg) -> None:
        self._object_pose = PoseStamped(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            orientation=np.array([
                msg.pose.orientation.x, msg.pose.orientation.y,
                msg.pose.orientation.z, msg.pose.orientation.w,
            ]),
            frame_id=msg.header.frame_id or '',
            stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        )
        if not self._object_label:
            self._object_label = self._target_object_label

    def _bboxes_3d_callback(self, msg) -> None:
        """缓存目标物体几何。按 instance_id / label 过滤，否则取最高 score。"""
        if not msg.boxes:
            self._bbox_miss_count += 1
            if self._bbox_miss_count >= _BBOX_STALE_FRAMES:
                self._clear_object_cache()
            return

        best = select_best_bbox(
            msg.boxes,
            target_label=self._target_label,
            target_instance_id=self._target_instance_id,
        )
        if best is None:
            self._bbox_miss_count += 1
            if self._bbox_miss_count >= _BBOX_STALE_FRAMES:
                self._clear_object_cache()
            return

        size = np.array(best.size, dtype=np.float64)
        if np.any(size > 0.0):
            prev_size = self._object_size
            self._object_size = size
            self._object_center = np.array(best.center, dtype=np.float64)
            self._object_center_frame = best.frame_id or msg.header.frame_id or ''
            self._object_orientation = np.array(best.orientation, dtype=np.float64)
            self._grasp_type = best.grasp_type
            self._bbox_miss_count = 0
            if prev_size is None:
                sz_mm = size * 1000
                all_labels = [f"{b.label}({b.score:.2f})" for b in msg.boxes]
                filter_desc = self._target_label or '(best)'
                if self._target_instance_id:
                    filter_desc += f",id={self._target_instance_id}"
                self._node.get_logger().info(
                    f"感知就绪: 收到 {len(msg.boxes)} 个物体 {all_labels}, "
                    f"选中 '{best.label}' filter={filter_desc} "
                    f"[{sz_mm[0]:.1f}x{sz_mm[1]:.1f}x{sz_mm[2]:.1f}]mm "
                    f"frame={self._object_center_frame} grasp={best.grasp_type}"
                )

    def _labeled_pose_callback(self, msg) -> None:
        """从感知 pipeline 的 LabeledPoseArray 中提取目标物体位姿 (world frame)。"""
        for lp in msg.poses:
            if lp.label == self._target_object_label:
                pose = lp.pose
                self._object_pose = PoseStamped(
                    position=np.array([pose.position.x, pose.position.y, pose.position.z]),
                    orientation=np.array([
                        pose.orientation.x, pose.orientation.y,
                        pose.orientation.z, pose.orientation.w,
                    ]),
                    stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                )
                self._object_label = lp.label
                return

    def _selected_label_callback(self, msg: String) -> None:
        """接收门控选中的物体 label，覆盖默认值。"""
        if msg.data and msg.data != self._object_label:
            self._object_label = msg.data
            self._node.get_logger().info(
                f"门控选中物体: {msg.data}")

    # ------------------------------------------------------------------
    # 触觉 / 力矩回调
    # ------------------------------------------------------------------

    def _force_callback(self, msg: Float32MultiArray) -> None:
        """O20/SDK 单点压感 hand_force: 20维, 前5个为各指法向力 0-255。"""
        if len(msg.data) >= 5:
            self._tactile_pressure = np.array(msg.data[:5], dtype=np.float64)
            self._tactile_mode = "pressure"
            threshold = self._contact_thresholds.pressure_threshold
            self._contact_detected = bool(np.any(self._tactile_pressure > threshold))
            self._force_updated = True

    def _normal_force_callback(self, msg: Float32MultiArray) -> None:
        """SDK normal_force: 每指一个法向力值。"""
        if len(msg.data) >= 5:
            self._tactile_pressure = np.array(msg.data[:5], dtype=np.float64)
            self._tactile_mode = "pressure"
            threshold = self._contact_thresholds.pressure_threshold
            self._contact_detected = bool(np.any(self._tactile_pressure > threshold))
            self._force_updated = True

    def _matrix_touch_mass_callback(self, msg: String) -> None:
        """O20/SDK 矩阵压感合值 (JSON): 每指总压力质量 (g) 作为触觉 fallback。"""
        if self._force_updated:
            return
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, ValueError):
            return
        keys = ("thumb_mass", "index_mass", "middle_mass", "ring_mass", "little_mass")
        values = [float(data.get(k, 0.0)) for k in keys]
        pressure = np.array(values, dtype=np.float64)
        self._tactile_pressure = pressure
        self._tactile_mode = "mass"
        threshold = self._contact_thresholds.mass_threshold
        self._contact_detected = bool(np.any(pressure > threshold))

    def _torque_callback(self, msg: Float32MultiArray) -> None:
        """motor_torque topic：O6 官方 0~100%；hand_info 已有有效值时不覆盖。"""
        n = self._hand_config.num_joints
        normalized = normalize_motor_torque_values(list(msg.data), n)
        if normalized is None:
            return
        if uses_torque_feedback(self._hand_type):
            if self._joint_torque is not None and np.any(self._joint_torque >= 0):
                return
        self._joint_torque = np.array(normalized, dtype=np.float64)

    def _tick(self) -> None:
        # FK 指尖位置计算
        fingertip_positions = None
        if self._fk_solver is not None:
            try:
                fingertip_positions = self._fk_solver.compute_fingertips(
                    self._current_angles)
            except Exception as e:
                if not hasattr(self, '_fk_err_logged'):
                    self._node.get_logger().warn(f"FK 计算异常 (后续不再重复): {e}")
                    self._fk_err_logged = True

        if self._active_primitive is not None:
            if self._active_primitive.done:
                self._node.get_logger().info(
                    f"原语完成: {self._active_primitive.name}，保持最终姿态"
                )
                self._active_primitive.on_exit()
                self._active_primitive = None
            else:
                elapsed = time.monotonic() - self._primitive_start_time
                ctx = PrimitiveContext(
                    tcp_pose=self._tcp_pose,
                    object_pose=self._object_pose,
                    object_label=self._object_label,
                    joint_currents=list(self._joint_currents),
                    hand_type=self._hand_type,
                    object_size=self._object_size,
                    object_orientation=self._object_orientation,
                    grasp_type=self._grasp_type,
                    tactile_pressure=(
                        self._tactile_pressure.copy()
                        if self._tactile_pressure is not None else None
                    ),
                    tactile_mode=self._tactile_mode,
                    contact_detected=self._contact_detected,
                    joint_torque=self._joint_torque,
                    fingertip_positions=fingertip_positions,
                    contact_thresholds=self._contact_thresholds,
                )
                result = self._active_primitive.compute(
                    self._current_angles, elapsed, ctx
                )
                if result.feasible:
                    self._last_target = self._semantic_to_hardware(result.target_angles)
                    self._log_motion_started(self._active_primitive.name, self._last_target)
                else:
                    self._log_infeasible_hold(
                        self._active_primitive.name, result, ctx)

                # 触觉/抓取状态日志
                self._log_grasp_state(ctx)

        # 始终发送（有原语发计算结果，无原语/infeasible 发上次的目标值）
        self._publish_cmd(self._last_target)

        # 发布指尖位置
        if self._publish_fingertips and fingertip_positions is not None:
            stamp = self._node.get_clock().now().to_msg()
            tip_msg = Float32MultiArray()
            tip_msg.layout.dim = [
                MultiArrayDimension(label="finger", size=5, stride=15),
                MultiArrayDimension(label="xyz", size=3, stride=3),
            ]
            tip_msg.data = fingertip_positions.flatten().tolist()
            self._fingertip_pub.publish(tip_msg)
            self._publish_fingertip_markers(fingertip_positions, stamp)

        # 执行结果反馈 (10Hz 节流到 ~0.5Hz 实际发布)
        self._publish_result()

    def _log_grasp_state(self, ctx: PrimitiveContext) -> None:
        """输出触觉 + 抓取状态日志 (仅状态变化时输出)。"""
        prim = self._active_primitive
        grasp_state = getattr(prim, 'grasp_state', None)
        if grasp_state is None:
            return

        if grasp_state != self._last_grasp_state:
            self._last_grasp_state = grasp_state
            pressure = self._tactile_pressure
            if pressure is not None and len(pressure) >= 5:
                p_str = (
                    f"[{pressure[0]:.0f},{pressure[1]:.0f},"
                    f"{pressure[2]:.0f},{pressure[3]:.0f},{pressure[4]:.0f}]"
                )
            else:
                p_str = "none(current-fallback)"
            feedback = (
                "tactile" if (
                    self._tactile_mode != "none" and self._tactile_pressure is not None
                ) else "current"
            )
            stopped = getattr(getattr(prim, "_engine", None), "_stopped", None)
            if stopped is None:
                stopped = getattr(prim, "_stopped", None)
            stopped_str = (
                f" stopped={sorted(k for k, v in stopped.items() if v)}"
                if isinstance(stopped, dict) and any(stopped.values()) else ""
            )
            self._node.get_logger().info(
                f"[{prim.name}] grasp={grasp_state} | "
                f"feedback={feedback} tactile={p_str} mode={self._tactile_mode} "
                f"contact={self._contact_detected}{stopped_str}"
            )

    _FINGERTIP_COLORS = [
        (1.0, 0.2, 0.2),  # thumb — red
        (0.2, 1.0, 0.2),  # index — green
        (0.2, 0.5, 1.0),  # middle — blue
        (1.0, 0.8, 0.0),  # ring — yellow
        (0.8, 0.2, 1.0),  # pinky — purple
    ]

    def _publish_fingertip_markers(self, positions: np.ndarray, stamp) -> None:
        """发布 5 个球体 marker 表示指尖位置，供 RViz 显示。"""
        ma = MarkerArray()
        for i in range(5):
            m = Marker()
            m.header.frame_id = self._fingertip_frame
            m.header.stamp = stamp
            m.ns = "fingertips"
            m.id = i
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position = Point(
                x=float(positions[i, 0]),
                y=float(positions[i, 1]),
                z=float(positions[i, 2]),
            )
            m.pose.orientation.w = 1.0
            m.scale.x = 0.012
            m.scale.y = 0.012
            m.scale.z = 0.012
            r, g, b = self._FINGERTIP_COLORS[i]
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            m.lifetime.sec = 0
            m.lifetime.nanosec = 200_000_000  # 200ms, 略大于 tick 周期
            ma.markers.append(m)
        self._fingertip_marker_pub.publish(ma)

    def _publish_cmd(self, angles: List[float]) -> None:
        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(JOINT_NAMES) if self._hand_type == "o20" else [f"j{i}" for i in range(self._hand_config.num_joints)]
        msg.position = [float(a) for a in angles]
        # 勿填 effort/velocity：linker_hand_sdk 会把 effort 当作 set_torque，
        # 全 0 会覆盖 teach 力矩导致手无法运动。
        self._cmd_pub.publish(msg)

    def _publish_result(self) -> None:
        """每 10 tick (~1Hz) 发布执行状态到 /hand_gesture/result。"""
        self._result_msg_count += 1
        if self._result_msg_count % 10 != 0:
            return

        if self._active_primitive is None:
            msg_text = "IDLE"
        elif self._active_primitive.done:
            msg_text = f"DONE: {self._active_primitive.name}"
        elif self._object_label:
            msg_text = (f"EXECUTING: {self._active_primitive.name}, "
                        f"obj={self._object_label}")
        else:
            msg_text = f"EXECUTING: {self._active_primitive.name}"

        if msg_text != self._last_result_msg:
            result_msg = String()
            result_msg.data = msg_text
            self._result_pub.publish(result_msg)
            self._last_result_msg = msg_text
