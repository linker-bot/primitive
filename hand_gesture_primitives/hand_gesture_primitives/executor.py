"""可配置频率的手势执行器：调度原语并发布关节指令。"""

import json
import time
from enum import Enum
from typing import List, Optional

import numpy as np
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseStamped as PoseStampedMsg
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState
from std_msgs.msg import (
    ColorRGBA,
    Float32MultiArray,
    Float64MultiArray,
    Header,
    MultiArrayDimension,
    String,
)
from visualization_msgs.msg import Marker, MarkerArray

from .contact_config import (
    DEFAULT_CONTROL_RATE_HZ,
    ContactThresholds,
    clamp_control_rate_hz,
    load_contact_thresholds,
)
from .contact_detection import uses_torque_feedback
from .contact_resolver import (
    normalize_motor_torque_values,
    parse_hand_info_currents,
    parse_hand_info_torque,
)
from .fk_solver import create_fk_solver
from .grasp_mapping import primitive_for_grasp_type
from .hand_profile import HandProfile
from .joint_state_mapping import (
    O30_ACTIVE_SUFFIXES,
    Ros2ControlRadianConverter,
    extract_o6_from_joint_state,
    extract_o20_from_joint_state,
    extract_o30_from_joint_state,
    g20_native_to_l25_urdf16,
    joint_state_to_g20_hardware,
    joint_state_to_l25_native,
    native25_to_active16,
    resolve_g20_l25_urdf,
    resolve_radian_hand_type,
)
from .label_utils import normalize_label, select_best_bbox
from .primitive_base import (
    HAND_CONFIGS,
    JOINT_NAMES,
    PHASED_GRASP_PRIMITIVES,
    RESERVED_INDICES,
    HandGesturePrimitive,
    PoseStamped,
    PrimitiveContext,
    PrimitiveResult,
)
from .primitives import PRIMITIVE_REGISTRY
from .semantic_30 import (
    SEMANTIC_DIM,
    as_semantic_30,
    project_o20_to_s30,
    project_s30_to_o20,
)
from .topic_defaults import (
    bboxes_3d_topic,
    labeled_poses_topic,
    resolve_side_topic,
    side_hand_topics,
)

try:
    from hand_force_msg.msg import HandForce
except ImportError:  # 可选依赖：缺失时禁用 hand_force 触觉订阅
    HandForce = None

# HandForce / normal_force 优先于 matrix_touch；超过此间隔无主源更新则允许 fallback
_FORCE_PRIMARY_STALE_SEC = 0.5

# 感知数据超时 (帧数, @5Hz 约 3s)
_BBOX_STALE_FRAMES = 15

# HandForce.finger_names 固定指序（与驱动 kFingerNames 一致）
_TACTILE_FINGER_NAMES = ("Thumb", "Index", "Middle", "Ring", "Little")


class ControlMode(str, Enum):
    """控制权：谁允许写手命令。"""

    PRIMITIVE = "primitive"
    TEACH = "teach"
    STREAM = "stream"


class GestureExecutor:
    """按 control_rate 持续发布目标角度（默认 10Hz，可配至 100Hz）。

    无论是否有活跃原语，PRIMITIVE/STREAM 模式下始终按设定频率下发控制信号。
    TEACH 模式停发，把控制权让给示教 GUI / save_pose。
    STREAM 模式下目标来自 ``stream_target`` topic；断流 watchdog 后 hold。
    原语返回 infeasible 时保持上帧目标值继续发送。
    内置电流过载保护：任一关节电流超阈值持续过久自动 fallback 到 init。
    """

    _SAFE_PRIMITIVES = {"init", "open", "release", "relax_grip"}
    # __new__ で生成されたインスタンス（テスト用）が __init__ をスキップしても
    # _radian_converter へのアクセスが AttributeError にならないようクラス属性で宣言する。
    # __init__ が呼ばれた場合はインスタンス属性で上書きされるため実行時の挙動は変わらない。
    _radian_converter = None

    @staticmethod
    def _is_safe_primitive(name: str) -> bool:
        return name in GestureExecutor._SAFE_PRIMITIVES or name.startswith("named_pose:")

    def __init__(
        self,
        node: Node,
        hand_side: str,
        hand_type: str = "o20",
        hand_profile: Optional[HandProfile] = None,
        tcp_pose_topic: str = "",
        object_pose_topic: str = "",
        bboxes_3d_topic: str = bboxes_3d_topic(),
        labeled_pose_topic: str = labeled_poses_topic(),
        target_object_label: str = "",
        result_topic: str = "",
        selected_label_topic: str = "",
        state_topic: str = "/joint_states",
        cmd_backend: str = "",
        hand_cmd_topic: str = "",
        ros2_control_command_units: str = "radians",
        robot_description: str = "",
        g20_joint_order: str = "",
        reach_strict: bool = False,
        reach_max_dist: float = 0.15,
        palm_offset: Optional[np.ndarray] = None,
        urdf_path: str = "",
        publish_fingertips: bool = True,
        fingertip_frame: str = "",
        contact_thresholds: Optional[ContactThresholds] = None,
        stream_target_topic: str = "",
        stream_watchdog_sec: float = 0.5,
        control_rate: float = DEFAULT_CONTROL_RATE_HZ,
    ):
        self._node = node
        self._side = hand_side
        side_topics = side_hand_topics(hand_side)
        self._hand_type = hand_type
        self._profile = hand_profile
        if hand_type not in HAND_CONFIGS:
            raise ValueError(f"未知 hand_type: {hand_type}")
        self._hand_config = HAND_CONFIGS[hand_type]
        # 所有原语统一在 semantic_30 30-DOF 语义空间运行。手型差异只允许出现在
        # HandConfig 的硬件边界；不能把 L25 native25 反馈直接喂给原语。
        self._primitive_space_dim = SEMANTIC_DIM
        self._control_epoch = 0
        self._active_primitive: Optional[HandGesturePrimitive] = None
        self._primitive_start_time: float = 0.0
        self._playback_clock = None
        self._motion_deadline = None
        self._motion_expired = False
        self._live_active = False
        self._configured_speed = 1.0
        self._primitive_speed = 1.0
        self._settle_since = None
        self._settle_frames = 0
        self._settle_status = "idle"
        self._trajectory_complete = False
        self._feedback_received_at = 0.0
        self._settle_feedback_at = 0.0
        self._feedback_completion_reported = False
        self._feedback_completion_diagnostic_logged = False
        self._current_angles: List[float] = [0.0] * self._primitive_space_dim
        # T1：与 _current_angles(legacy 255) 并行的 semantic_rad 缓存（radians 模式维护）
        self._current_semantic_rad: List[float] = [0.0] * self._primitive_space_dim
        self._current_hw_angles: List[float] = []
        self._last_target: List[float] = [0.0] * self._hand_config.num_joints
        # radians 模式下优先用于发布的 semantic_rad 目标（避免 hw→255→rad 二次误差）
        self._last_target_semantic_rad: Optional[List[float]] = None
        # 最近一次产品 hardware 命令（示教 mirror / 本节点发布）；回放起点优先于此
        self._last_commanded_hw: List[float] = []
        self._control_mode: ControlMode = ControlMode.PRIMITIVE
        self._stream_target_topic = resolve_side_topic(
            stream_target_topic, side_topics["stream_target"]
        )
        self._stream_watchdog_sec = float(stream_watchdog_sec)
        self._last_stream_t: float = 0.0
        self._stream_watchdog_logged: bool = False
        self._control_rate = clamp_control_rate_hz(control_rate)
        self._control_period = 1.0 / self._control_rate
        self._result_period_ticks = max(1, int(round(self._control_rate)))
        self._contact_thresholds = contact_thresholds or load_contact_thresholds()

        # 可达性检查参数
        self._reach_strict: bool = reach_strict
        self._reach_max_dist: float = reach_max_dist
        self._palm_offset: np.ndarray = palm_offset if palm_offset is not None else np.zeros(3)

        # 电流过载保护状态
        self._joint_overload_start: List[Optional[float]] = [None] * self._hand_config.num_joints
        # 电流反馈缓存 (mA, position order)
        self._joint_currents: List[float] = [0.0] * self._hand_config.num_joints
        self._joint_currents_valid: bool = False
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
        self._grasp_type: str = ""
        self._target_label: str = ""
        self._target_instance_id: Optional[int] = None
        self._object_center_frame: str = ""
        self._bbox_miss_count: int = 0

        # 发布关节指令：
        # L25/O6/L6/O20 默认 ros2_control FCC；cmd_backend:=legacy_cb 走旧 /cb_*。
        backend = (cmd_backend or "").strip().lower()
        if not backend:
            backend = (
                "ros2_control"
                if hand_type in ("l25", "o6", "l6", "o20", "o30", "g20")
                else "legacy_cb"
            )
        if backend not in ("ros2_control", "legacy_cb"):
            raise ValueError(f"未知 cmd_backend: {cmd_backend} (期望 ros2_control|legacy_cb)")
        self._cmd_backend = backend
        command_units = (ros2_control_command_units or "radians").strip().lower()
        if command_units not in ("hardware", "radians"):
            raise ValueError(
                "ros2_control_command_units 只能是 hardware 或 radians，"
                f"当前为: {ros2_control_command_units!r}"
            )
        if backend != "ros2_control" and command_units != "hardware":
            raise ValueError("radians 命令量纲仅适用于 cmd_backend=ros2_control")
        self._ros2_control_command_units = command_units
        self._g20_l25_urdf = False
        self._g20_l25_urdf_locked = False
        if hand_type == "g20":
            order = (g20_joint_order or "").strip().lower()
            self._g20_l25_urdf = resolve_g20_l25_urdf(
                g20_joint_order, robot_description or "", hand_side
            )
            # 显式 l25_urdf|native 时不允许被 /joint_states 推断覆盖
            self._g20_l25_urdf_locked = bool(order)
        radian_hand_type = resolve_radian_hand_type(
            hand_type,
            g20_joint_order=g20_joint_order,
            robot_description=robot_description or "",
            hand_side=hand_side,
        )
        # 有 URDF 即建 Converter：原语内部统一 semantic_rad；command_units 只影响发布边界
        self._radian_converter = None
        if (robot_description or "").strip():
            try:
                self._radian_converter = Ros2ControlRadianConverter(
                    robot_description, radian_hand_type, hand_side
                )
            except ValueError as err:
                if command_units == "radians":
                    raise
                node.get_logger().warn(f"无法构建 radian converter，原语退化为 legacy255: {err}")
        if backend == "ros2_control":
            self._cmd_topic = (
                hand_cmd_topic.strip()
                if hand_cmd_topic.strip()
                else f"/{hand_side}_hand_controller/commands"
            )
            self._cmd_pub = node.create_publisher(Float64MultiArray, self._cmd_topic, 10)
            self._cmd_no_sub_warned = False
            # TEACH 期间 panel 独占下发：镜像同 topic，记下 commanded 供退出示教/回放起点
            node.create_subscription(
                Float64MultiArray,
                self._cmd_topic,
                self._external_cmd_callback,
                10,
            )
        else:
            self._cmd_topic = (
                hand_cmd_topic.strip()
                if hand_cmd_topic.strip()
                else f"/cb_{hand_side}_hand_control_cmd"
            )
            self._cmd_pub = node.create_publisher(JointState, self._cmd_topic, 10)
            node.create_subscription(
                JointState,
                self._cmd_topic,
                self._external_legacy_cmd_callback,
                10,
            )

        # 订阅手部状态反馈（机载：/joint_states；也可设为 /cb_{side}_hand_state）
        self._state_topic = state_topic
        node.create_subscription(
            JointState,
            state_topic,
            self._state_callback,
            10,
        )

        # STREAM：连续目标；长度依协议——hardware=产品 DOF，radians=controller 主动序
        node.create_subscription(
            JointState,
            self._stream_target_topic,
            self._stream_target_callback,
            10,
        )

        # 订阅手部信息（电流/温度/力矩）
        # linkerhand_hardware 用 SensorDataQoS(best_effort) 发布 /cb_*，须同 QoS 才能匹配
        node.create_subscription(
            String,
            f"/cb_{hand_side}_hand_info",
            self._info_callback,
            qos_profile_sensor_data,
        )

        self._tcp_pose_topic = resolve_side_topic(tcp_pose_topic, side_topics["tcp_pose"])
        self._object_pose_topic = resolve_side_topic(object_pose_topic, side_topics["object_pose"])
        result_topic = resolve_side_topic(result_topic, side_topics["gesture_result"])
        selected_label_topic = resolve_side_topic(
            selected_label_topic, side_topics["selected_label"]
        )

        # 订阅 TCP 位姿
        node.create_subscription(
            PoseStampedMsg,
            self._tcp_pose_topic,
            self._tcp_pose_callback,
            10,
        )

        # 订阅物体位姿 (相对手腕 / world，由上游发布)
        node.create_subscription(
            PoseStampedMsg,
            self._object_pose_topic,
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
                "robot_perception_msgs.LabeledBBox3DArray 不可用，禁用 3D bbox 订阅"
            )

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
                f"已订阅感知 topic: {labeled_pose_topic}, 目标物体: {target_object_label}"
            )
        except ImportError:
            self._node.get_logger().warn(
                f"robot_perception_msgs.LabeledPoseArray 不可用，"
                f"仅使用 {self._object_pose_topic} 输入"
            )

        # 触觉反馈 (linkerhand_hardware HandForce 矩阵压感 + SDK normal_force)
        self._tactile_pressure: Optional[np.ndarray] = None
        self._palm_pressure: Optional[float] = None
        self._tactile_mode: str = "none"
        self._contact_detected: bool = False
        # 主触觉源（HandForce / normal_force）最近更新时刻；0=从未收到
        self._force_primary_stamp: float = 0.0
        if HandForce is not None:
            node.create_subscription(
                HandForce,
                f"/cb_{hand_side}_hand_force",
                self._force_callback,
                qos_profile_sensor_data,
            )
        else:
            node.get_logger().warn(
                "hand_force_msg 不可用，禁用 /cb_"
                f"{hand_side}_hand_force 触觉订阅（降级 normal_force/mass）"
            )
        node.create_subscription(
            Float32MultiArray,
            f"/cb_{hand_side}_hand_normal_force",
            self._normal_force_callback,
            qos_profile_sensor_data,
        )
        node.create_subscription(
            String,
            f"/cb_{hand_side}_hand_matrix_touch_mass",
            self._matrix_touch_mass_callback,
            qos_profile_sensor_data,
        )

        # 力矩反馈 (SDK 独立 topic)
        self._joint_torque: Optional[np.ndarray] = None
        self._torque_samples = {}
        self._torque_source = None
        self._workbench_torque_at = 0.0
        node.create_subscription(
            Float32MultiArray,
            f"/cb_{hand_side}_hand_motor_torque",
            self._torque_callback,
            qos_profile_sensor_data,
        )

        # FK 正运动学求解器
        hand_joint = hand_type.upper()
        self._fk_solver = create_fk_solver(
            hand_joint, hand_side, urdf_path, logger=node.get_logger()
        )
        self._publish_fingertips = publish_fingertips and self._fk_solver is not None
        self._fk_err_logged = False
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

        # 按 control_rate 定时 — 始终运行（默认 10Hz，可配 100Hz）
        self._timer = node.create_timer(self._control_period, self._tick)

        # 触觉日志: 仅状态变化时输出
        self._last_grasp_state: str = ""
        self._infeasible_hold_logged: bool = False
        self._motion_logged: bool = False

        # 执行结果反馈 (节流到约 1Hz)
        self._result_pub = node.create_publisher(String, result_topic, 10)
        self._last_result_msg = ""
        self._terminal_result_until = 0.0
        self._result_msg_count = 0

        # 从门控接收选中的物体 label
        node.create_subscription(
            String,
            selected_label_topic,
            self._selected_label_callback,
            10,
        )

        self._node.get_logger().info(
            f"GestureExecutor 已启动: side={hand_side}, "
            f"{self._control_rate:g}Hz 持续输出 (period={self._control_period:.4f}s), "
            f"电流保护: >{self._contact_thresholds.overload_threshold}mA"
            f"持续>{self._contact_thresholds.overload_duration_sec}s"
        )
        self._node.get_logger().info(
            f"已订阅 tcp_pose: {self._tcp_pose_topic}, "
            f"object_pose: {self._object_pose_topic}, "
            f"hand_state: {self._state_topic}, "
            f"hand_cmd[{self._cmd_backend}/{self._ros2_control_command_units}]: "
            f"{self._cmd_topic}"
        )

        # 启动时自动执行 init 原语
        self.set_primitive(PRIMITIVE_REGISTRY["init"]())

    @property
    def control_epoch(self) -> int:
        """Generation of committed control changes, including legacy commands."""
        return getattr(self, "_control_epoch", 0)

    def advance_control_epoch(self) -> None:
        self._control_epoch = self.control_epoch + 1

    def _reset_completion(self) -> None:
        self._settle_since = None
        self._settle_frames = 0
        self._settle_feedback_at = getattr(self, "_feedback_received_at", 0.0)
        self._settle_status = "running"
        self._trajectory_complete = False
        if getattr(self, "_playback_clock", None):
            self._playback_clock.finished = False
        self._feedback_completion_reported = False
        self._last_result_msg = ""
        self._terminal_result_until = 0.0

    @property
    def can_reload_primitives(self) -> bool:
        return self._control_mode == ControlMode.PRIMITIVE and self._active_primitive is None

    @property
    def active_primitive_name(self) -> str:
        if self._active_primitive is None:
            return "hold"
        return self._active_primitive.name

    @property
    def grasp_state(self) -> str:
        if self._active_primitive is None:
            return ""
        return getattr(self._active_primitive, "grasp_state", "") or ""

    def apply_motion_progress(self, progress: float) -> Optional[str]:
        """热更新当前原语段内 progress；成功返回 None，否则错误说明。

        适用于 NamedPose 与支持 ``set_progress`` 的定位相位原语（如 tripod prep）。
        同段不重锚（见 KeyframeRunner / 原语内 override）。
        与 gesture_cmd 共用 ``progress_policy``，避免 topic 绕过力控拒绝矩阵。
        """
        import math

        if self._control_mode != ControlMode.PRIMITIVE:
            return f"rejected:{self._control_mode.value}_mode"
        prim = self._active_primitive
        if prim is None:
            return "rejected:no_active_pose"
        if not prim.supports_progress():
            return "rejected:no_set_progress"
        try:
            p = float(progress)
        except (TypeError, ValueError):
            return "rejected:bad_progress"
        if not math.isfinite(p):
            return "rejected:bad_progress"
        p = max(0.0, min(1.0, p))

        prim_name = str(getattr(prim, "name", "") or "")
        if not isinstance(prim, PRIMITIVE_REGISTRY["named_pose"]):
            from .progress_policy import validate_primitive_progress

            phase = str(getattr(prim, "phase", "") or "auto")
            policy_err = validate_primitive_progress(prim_name, phase, p)
            if policy_err:
                return f"rejected:{policy_err}"

        prim.set_progress(p)
        self._reset_completion()
        self.advance_control_epoch()
        self._motion_logged = False
        return None

    def set_named_pose(
        self,
        pose_name: str,
        *,
        keyframe: Optional[str] = None,
        progress: Optional[float] = None,
        auto_segment: bool = False,
    ) -> str:
        """激活或热更新 NamedPose。

        - ``pose <name>``（无 kf/progress）：始终整段 AUTO 重进。
        - ``pose <name> <kf> auto``：仅 AUTO 该关键帧段，段末 done。
        - 同名已激活且 runner 就绪：
          - 同段改 progress / 仅 kf→p=0 → **不重锚**热更新；
          - 换 kf → ``switch_keyframe``（重锚，符合换段语义）。
        - 其余情况 ``set_primitive`` 新建。

        返回空串成功，否则拒绝原因。
        """
        import math
        import time as _time

        name = (pose_name or "").strip().lower()
        kf = (keyframe or "").strip().lower() or None
        prog: Optional[float] = None
        if progress is not None:
            try:
                prog = float(progress)
            except (TypeError, ValueError):
                return "rejected:bad_progress"
            if not math.isfinite(prog):
                return "rejected:bad_progress"
            prog = max(0.0, min(1.0, prog))

        want_auto_seg = bool(auto_segment) and kf is not None and prog is None
        if want_auto_seg:
            return self.set_primitive(
                PRIMITIVE_REGISTRY["named_pose"](
                    name, keyframe=kf, progress=None, auto_segment=True
                )
            )

        prim = self._active_primitive
        bare_auto = kf is None and prog is None
        can_hot = (
            isinstance(prim, PRIMITIVE_REGISTRY["named_pose"])
            and prim.pose_name == name
            and prim.runner_ready
            and not bare_auto
        )
        if can_hot:
            elapsed = self._playback_clock.advance(_time.monotonic()) if getattr(self, "_playback_clock", None) else (_time.monotonic() - self._primitive_start_time) * getattr(self, "_primitive_speed", 1.0)
            cur_kf = prim.active_keyframe
            if kf is not None and cur_kf is not None and kf != cur_kf:
                try:
                    prim.switch_keyframe(
                        kf,
                        self._anchor_angles_for_keyframe_switch(),
                        progress=0.0 if prog is None else prog,
                        elapsed=elapsed,
                    )
                except KeyError:
                    return f"rejected:keyframe_missing:{kf}"
                self._reset_completion()
                self.advance_control_epoch()
                self._motion_logged = False
                self._last_result_msg = ""
                self._node.get_logger().info(f"named_pose 换段（重锚）: {prim.motion_status}")
                return ""
            # 同段：显式 progress，或仅 kf → hold p=0
            target_p = 0.0 if prog is None else prog
            prim.set_progress(target_p)
            self._reset_completion()
            self.advance_control_epoch()
            self._motion_logged = False
            self._last_result_msg = ""
            self._node.get_logger().info(
                f"named_pose 热更新 progress（不重锚）: {prim.motion_status}"
            )
            return ""

        return self.set_primitive(
            PRIMITIVE_REGISTRY["named_pose"](name, keyframe=kf, progress=prog)
        )

    @property
    def active_motion_status(self) -> str:
        prim = self._active_primitive
        if prim is None:
            return ""
        return str(getattr(prim, "motion_status", "") or "")

    @property
    def control_mode(self) -> ControlMode:
        return self._control_mode

    def set_control_mode(self, mode: ControlMode) -> None:
        """切换控制权。TEACH 停发 cmd；PRIMITIVE/STREAM 恢复发布。"""
        if mode == self._control_mode:
            return
        prev = self._control_mode
        if mode == ControlMode.TEACH:
            self.stop()
            self._control_mode = mode
            self._node.get_logger().info(
                f"控制模式: {prev.value} → {mode.value}（停发 hand cmd，示教让权）"
            )
            return
        if mode == ControlMode.STREAM:
            self.stop()
            self._last_stream_t = 0.0
            self._stream_watchdog_logged = False
            self._last_target = self._hold_hw_for_resume()
            self._sync_last_target_rad_from_hw(self._last_target)
            self._control_mode = mode
            self._node.get_logger().info(
                f"控制模式: {prev.value} → {mode.value} "
                f"（订阅 {self._stream_target_topic}，watchdog={self._stream_watchdog_sec}s）"
            )
            return
        # 退出 TEACH/恢复原语：优先 hold 最近下发命令，避免反馈重建偏开导致先张开
        self._last_target = self._hold_hw_for_resume()
        self._sync_last_target_rad_from_hw(self._last_target)
        self._control_mode = mode
        self.advance_control_epoch()
        self._node.get_logger().info(f"控制模式: {prev.value} → {mode.value}（恢复发布）")

    def _stream_expected_dof(self) -> int:
        """STREAM 输入维数：按手型 + 命令协议决定。

        - ``ros2_control`` + ``radians``：ForwardCommandController 主动序
          （L25/G20→16，O20/O30→20，O6/L6→6）
        - ``hardware`` / ``legacy_cb``：产品硬件 DOF（L25→25，其余同 num_joints）
        """
        if self._ros2_control_command_units == "radians" and self._radian_converter is not None:
            return len(self._radian_converter.joint_names)
        return int(self._hand_config.num_joints)

    def _stream_target_callback(self, msg: JointState) -> None:
        if self._control_mode != ControlMode.STREAM:
            return
        n = self._stream_expected_dof()
        if len(msg.position) < n:
            if not hasattr(self, "_stream_len_warned"):
                self._node.get_logger().warn(
                    f"stream_target 长度不足: {len(msg.position)} < {n} "
                    f"(hand={self._hand_type}, units={self._ros2_control_command_units})"
                )
                self._stream_len_warned = True
            return
        self._stream_len_warned = False
        if self._ros2_control_command_units == "radians" and self._radian_converter is not None:
            ctrl = [float(v) for v in msg.position[:n]]
            semantic_255 = self._radian_converter.radians_to_semantic(ctrl)
            self._last_target_semantic_rad = self._radian_converter.legacy255_to_semantic_rad(
                semantic_255
            )
            self._last_target = self._semantic_to_hardware(semantic_255)
        else:
            self._last_target = [max(0.0, min(255.0, float(v))) for v in msg.position[:n]]
            self._sync_last_target_rad_from_hw(self._last_target)
        self._note_commanded_hw(self._last_target)
        self._last_stream_t = time.monotonic()
        self._stream_watchdog_logged = False
        # JointState has no request-id field; Framework reserves this frame_id
        # prefix for an opaque receipt token on the final stream sample.
        token = msg.header.frame_id
        if token.startswith("luban-stream-"):
            self.publish_user_result(f"STREAM_ACCEPTED: {token}")

    def set_target_label(self, label: str, *, invalidate: bool = True) -> None:
        """设置感知目标过滤 label (空=最高 score)。支持去标点与子串匹配。"""
        normalized = normalize_label(label)
        if normalized != self._target_label:
            self._target_label = normalized
            self._clear_object_cache()
            if invalidate:
                self.advance_control_epoch()

    def set_target_instance_id(self, instance_id: int) -> None:
        """按 instance_id 过滤 (0 或负数=清除)。"""
        new_id = instance_id if instance_id > 0 else None
        if new_id != self._target_instance_id:
            self._target_instance_id = new_id
            self._clear_object_cache()
            self.advance_control_epoch()

    def _clear_object_cache(self) -> None:
        self._object_size = None
        self._object_orientation = None
        self._object_center = None
        self._object_center_frame = ""
        self._grasp_type = ""

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
            return ""
        return prim or ""

    def _palm_world_position(self) -> Optional[np.ndarray]:
        """将 TCP 系 palm_offset 旋到世界系，返回掌心世界坐标；无 TCP 则 None。"""
        if self._tcp_pose is None:
            return None
        offset = np.asarray(self._palm_offset, dtype=np.float64).reshape(3)
        quat = np.asarray(self._tcp_pose.orientation, dtype=np.float64).reshape(4)
        if np.linalg.norm(quat) < 1e-9 or np.allclose(quat, [0.0, 0.0, 0.0, 1.0]):
            rotated = offset
        else:
            rotated = Rotation.from_quat(quat).as_matrix() @ offset
        return np.asarray(self._tcp_pose.position, dtype=np.float64).reshape(3) + rotated

    def _check_reachability(self) -> str:
        """检查手掌→目标可达性。返回空=通过, 非空=原因。"""
        palm_pos = self._palm_world_position()
        if palm_pos is None:
            return "no_tcp"

        tcp_frame = self._tcp_pose.frame_id if self._tcp_pose is not None else ""

        if self._object_pose is not None:
            obj_frame = self._object_pose.frame_id
            if tcp_frame and obj_frame and tcp_frame != obj_frame:
                return f"frame_mismatch:tcp={tcp_frame},obj={obj_frame}"
            dist = float(
                np.linalg.norm(np.asarray(self._object_pose.position, dtype=np.float64) - palm_pos)
            )
            if dist > self._reach_max_dist:
                return f"too_far:{dist:.3f}m>(max:{self._reach_max_dist:.3f}m)"
            return ""

        if self._object_center is None:
            return "no_target"

        bbox_frame = self._object_center_frame
        if tcp_frame and bbox_frame and tcp_frame != bbox_frame:
            return f"frame_mismatch:tcp={tcp_frame},bbox={bbox_frame}"

        dist = float(np.linalg.norm(palm_pos - self._object_center))
        if dist > self._reach_max_dist:
            return f"too_far:{dist:.3f}m>(max:{self._reach_max_dist:.3f}m)"
        return ""

    def target_selection_snapshot(self):
        return {name: getattr(self, name, 0 if name.endswith("_at") or name == "_bbox_miss_count" else None) for name in (
            "_target_label", "_object_size", "_object_orientation", "_object_center",
            "_object_center_frame", "_grasp_type", "_object_pose", "_object_label",
            "_workbench_bbox_at", "_workbench_object_at", "_bbox_miss_count",
        )}

    def restore_target_selection(self, snapshot):
        for name, value in snapshot.items():
            setattr(self, name, value)

    def set_primitive(self, primitive: HandGesturePrimitive, *, target_label=None) -> str:
        snapshot = self.target_selection_snapshot() if target_label is not None else None
        accepted = False
        try:
            if target_label is not None:
                self.set_target_label(target_label, invalidate=False)
                self.refresh_target_from_cached_bbox()
            reason = self._install_primitive(primitive)
            accepted = not reason
            return reason
        finally:
            if snapshot is not None and not accepted:
                self.restore_target_selection(snapshot)

    def _install_primitive(self, primitive: HandGesturePrimitive) -> str:
        """切换到新原语。返回空=成功, 非空=拒绝原因。"""
        new_phase = getattr(primitive, "phase", "full")
        if new_phase == "close" and primitive.name in PHASED_GRASP_PRIMITIVES:
            prev = self._active_primitive
            if prev is None or prev.name != primitive.name:
                self._node.get_logger().warn(
                    f"'{primitive.name} close' 未检测到同原语 prep，从当前关节姿态闭合"
                )
            elif getattr(prev, "grasp_state", "") != "ready":
                self._node.get_logger().warn(
                    f"'{primitive.name} close' 时 grasp_state="
                    f"{getattr(prev, 'grasp_state', '')} (非 ready)，从当前姿态继续"
                )

        # 安全原语不做可达性检查
        if not self._is_safe_primitive(primitive.name):
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

        try:
            primitive.on_enter(self._primitive_current_angles())
            from .workbench_motion import PlaybackClock, trajectory_duration
            entry = getattr(primitive, "_entry_snapshot", None)
            # Legacy pose commands load their entry lazily; leave duration unknown.
            duration = trajectory_duration(
                self._hand_type, primitive.name, entry=entry,
                start=self._primitive_current_angles(), converter=self._radian_converter,
            )
            candidate_clock = PlaybackClock(time.monotonic(), getattr(self, "_configured_speed", 1.0), duration)
        except Exception as exc:
            return f"rejected:primitive_init_failed:{exc}"

        if self._active_primitive is not None:
            self._node.get_logger().info(
                f"原语切换: {self._active_primitive.name} → {primitive.name}"
            )
            try:
                self._active_primitive.on_exit()
            except Exception as exc:
                self._node.get_logger().warning(f"原语清理失败: {exc}")
        else:
            self._node.get_logger().info(f"激活原语: {primitive.name}")

        self._playback_clock = candidate_clock
        self._motion_deadline = None
        self._motion_expired = False
        self._live_active = False
        self._active_primitive = primitive
        self._primitive_start_time = time.monotonic()
        self._primitive_speed = getattr(self, "_configured_speed", 1.0)
        self._settle_since = None
        self._settle_frames = 0
        self._settle_status = "running"
        self._trajectory_complete = False
        self._feedback_completion_reported = False
        self._feedback_completion_diagnostic_logged = False
        # A repeated command (for example init during system initialization
        # and again before grasp) is a new operation and must emit a fresh
        # result sequence even when the hand is already at the same target.
        self._last_result_msg = ""
        self._terminal_result_until = 0.0
        self._last_grasp_state = ""
        self._infeasible_hold_logged = False
        self._motion_logged = False
        self.advance_control_epoch()
        self._log_primitive_context(primitive.name)
        return ""

    def _primitive_current_angles(self) -> List[float]:
        """原语 compute/on_enter 入参：有 Converter 时为 semantic_rad。"""
        if self._radian_converter is not None:
            return list(self._current_semantic_rad)
        return list(self._current_angles)

    def _anchor_angles_for_keyframe_switch(self) -> List[float]:
        """换段重锚：优先最近命令角，避免真机反馈滞后导致跳变。"""
        if self._radian_converter is not None:
            if self._last_target_semantic_rad is not None:
                return list(self._last_target_semantic_rad)
            cmd_hw = self._cmd_hw_for_context()
            if cmd_hw is not None:
                # hardware → semantic_rad（与命令路径一致）
                try:
                    legacy = self._hardware_to_semantic(cmd_hw)
                    return list(self._radian_converter.legacy255_to_semantic_rad(legacy))
                except Exception:
                    pass
        else:
            cmd_hw = self._cmd_hw_for_context()
            if cmd_hw is not None:
                return self._hardware_to_semantic(cmd_hw)
        return self._primitive_current_angles()

    def _primitive_angle_unit(self) -> str:
        from .angle_bridge import SPACE_LEGACY_SEMANTIC255, SPACE_SEMANTIC_RAD

        if self._radian_converter is not None:
            return SPACE_SEMANTIC_RAD
        return SPACE_LEGACY_SEMANTIC255

    def set_motion_speed(self, speed):
        from .workbench_motion import validate_speed
        self._configured_speed = validate_speed(speed)
        self._primitive_speed = float(speed)
        if getattr(self, "_playback_clock", None):
            self._playback_clock.set_speed(speed, time.monotonic())

    def playback_status(self):
        clock = getattr(self, "_playback_clock", None)
        state = clock.snapshot() if clock else {"playback_state": "live" if getattr(self, "_live_active", False) else "idle", "overall_progress": -1.0}
        return {**state, "keyframe": getattr(self._active_primitive, "active_keyframe", "") or "",
                "primitive_speed": getattr(self, "_configured_speed", 1.0)}

    def hold_playback(self, running):
        clock = getattr(self, "_playback_clock", None)
        if clock is None or not clock.held:
            raise ValueError("当前没有按住推进的动作")
        if running and getattr(self, "_motion_expired", False):
            raise ValueError("控制心跳已过期，请松开后重新按住")
        if not running:
            self._motion_expired = False
        if clock.running != running:
            clock.set_running(running, time.monotonic())
            self.advance_control_epoch()
        self._motion_deadline = time.monotonic() + 0.35 if running else None

    def live_target(self, positions):
        conv = self._radian_converter
        if conv is None or len(positions) != len(conv.limits):
            raise ValueError("实时跟随需要匹配的 URDF 关节换算器")
        if not all(np.isfinite(v) and lo - 1e-7 <= v <= hi + 1e-7 for v, (lo, hi) in zip(positions, conv.limits)):
            raise ValueError("实时跟随目标超出限位或不是有限数值")
        semantic = conv.controller_to_semantic_rad(positions)
        if not getattr(self, "_live_active", False):
            self.stop()
        self._apply_feasible_result(PrimitiveResult(feasible=True, target_angles=semantic, space="semantic_rad"))
        self._live_active = True
        self.advance_control_epoch()
        self._motion_deadline = time.monotonic() + 0.5

    def stop(self) -> None:
        """Stop and invalidate pending authorizations, including when already idle."""
        self.advance_control_epoch()
        self._playback_clock = None
        self._live_active = False
        self._motion_deadline = None
        """Stop this primitive and hold fresh measured feedback, never an unfinished target."""
        if self._active_primitive is not None:
            try:
                self._active_primitive.on_exit()
            except Exception as exc:
                self._node.get_logger().warning(f"原语清理失败: {exc}")
        self._active_primitive = None
        q = getattr(self, "_current_hw_angles", None)
        if q is not None and len(q) == self._hand_config.num_joints and np.isfinite(q).all():
            self._last_target = list(q)
            self._sync_last_target_rad_from_hw(self._last_target)

    def plan_static_primitive(
        self,
        primitive: str,
        *,
        sample_period: float = 0.04,
    ) -> dict[str, object]:
        """Plan a nominal static gesture without changing live executor state."""

        if not self._hand_state_received:
            raise RuntimeError("hand joint state is unavailable")
        if self._radian_converter is None:
            raise RuntimeError("hand trajectory preview requires ros2_control radians mode")
        from .trajectory_planner import plan_static_gesture

        planned = plan_static_gesture(
            hand_type=self._hand_type,
            primitive=primitive,
            current_semantic=list(self._current_angles),
            semantic_to_positions=self._radian_converter.semantic_to_radians,
            sample_period=sample_period,
            radian_converter=self._radian_converter,
        )
        planned["joint_names"] = list(self._radian_converter.joint_names)
        return planned

    def plan_pose_trajectory(
        self,
        pose_name: str,
        *,
        sample_period: float = 0.04,
    ) -> dict[str, object]:
        """Plan a multi-segment pose trajectory without changing live executor state."""
        if not self._hand_state_received:
            raise RuntimeError("hand joint state is unavailable")
        if self._radian_converter is None:
            raise RuntimeError("hand trajectory preview requires ros2_control radians mode")

        from .pose_library import load_pose, motion_program_from_entry
        from .trajectory_planner import plan_motion_program

        entry = load_pose(self._hand_type, pose_name, hand_side=self._side)
        if entry is None:
            raise ValueError(f"pose not found: {pose_name}")
        if entry.space == "hardware":
            raise ValueError(
                f"pose '{pose_name}' is space=hardware; "
                "PlanGesture requires semantic_v3/semantic_o20 (prefer unit=rad)"
            )
        if entry.space not in ("semantic_v3", "semantic_o20"):
            raise ValueError(
                f"pose '{pose_name}' has unsupported space='{entry.space}'; "
                "PlanGesture requires semantic_v3 or semantic_o20"
            )

        program = motion_program_from_entry(entry)

        start_q = list(self._current_semantic_rad)
        conv = self._radian_converter
        converted_segs = []
        from .keyframe_runner import KeyframeSegment, MotionProgram
        from .semantic_30 import SEMANTIC_DIM, project_o20_to_s30

        for seg in program.segments:
            positions = list(seg.target)
            if entry.space == "semantic_v3":
                semantic = positions[:SEMANTIC_DIM]
                while len(semantic) < SEMANTIC_DIM:
                    semantic.append(0.0)
            else:
                o20 = positions[:20]
                while len(o20) < 20:
                    o20.append(0.0)
                semantic = project_o20_to_s30(o20)
            unit = str(getattr(entry, "unit", "legacy255") or "legacy255").lower()
            if unit == "rad":
                tgt = semantic
            else:
                tgt = conv.legacy255_to_semantic_rad(semantic)
            converted_segs.append(
                KeyframeSegment(name=seg.name, target=tuple(tgt), duration=seg.duration)
            )
        program = MotionProgram(
            ref=program.ref, space=entry.space, unit=entry.unit, segments=converted_segs
        )
        to_positions = conv.semantic_rad_to_controller

        planned = plan_motion_program(
            program=program,
            start_q=start_q,
            to_positions=to_positions,
            sample_period=sample_period,
        )
        planned["joint_names"] = list(self._radian_converter.joint_names)
        return planned

    def _sync_semantic_rad_from_legacy_angles(self) -> None:
        """hardware/legacy 反馈写入 ``_current_angles`` 后同步 semantic_rad。"""
        if self._radian_converter is None:
            return
        self._current_semantic_rad = self._radian_converter.legacy255_to_semantic_rad(
            self._current_angles
        )

    def _state_callback(self, msg: JointState) -> None:
        # GenericSystem 的 position command/state 遵循 URDF 弧度；按控制器顺序
        # 抽取后先还原为统一 S30 语义空间，供原语插值与 FK 使用。
        # 同时缓存完整硬件向量，供 NamedPose hardware 直出与模式切换 hold。
        if (
            self._cmd_backend == "ros2_control"
            and self._ros2_control_command_units == "radians"
            and msg.name
        ):
            controller_state = None
            if self._hand_type == "l25":
                native = joint_state_to_l25_native(msg.name, msg.position, self._side)
                if native is not None:
                    controller_state = native25_to_active16(native)
            elif self._hand_type in ("o6", "l6"):
                controller_state = extract_o6_from_joint_state(msg.name, msg.position, self._side)
            elif self._hand_type == "o20":
                controller_state = extract_o20_from_joint_state(msg.name, msg.position, self._side)
            elif self._hand_type == "o30":
                controller_state = extract_o30_from_joint_state(msg.name, msg.position, self._side)
            elif self._hand_type == "g20":
                # 整机 l25_urdf：强制 L25 主动序，避免 SDK 名抢先导致 converter 序错乱
                if self._g20_l25_urdf:
                    native = joint_state_to_l25_native(msg.name, msg.position, self._side)
                    if native is not None:
                        controller_state = native25_to_active16(native)
                    elif not self._g20_l25_urdf_locked:
                        hw, uses_l25 = joint_state_to_g20_hardware(
                            msg.name, msg.position, self._side
                        )
                        if hw is not None and uses_l25:
                            controller_state = g20_native_to_l25_urdf16(hw)
                else:
                    hw, uses_l25 = joint_state_to_g20_hardware(msg.name, msg.position, self._side)
                    if hw is not None:
                        if not self._g20_l25_urdf_locked:
                            self._g20_l25_urdf = uses_l25
                        controller_state = g20_native_to_l25_urdf16(hw) if uses_l25 else hw

            if controller_state is not None and self._radian_converter is not None:
                self._current_semantic_rad = self._radian_converter.controller_to_semantic_rad(
                    controller_state
                )
                self._current_angles = self._radian_converter.semantic_rad_to_legacy255(
                    self._current_semantic_rad
                )
                self._current_hw_angles = self._semantic_to_hardware(list(self._current_angles))
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
            # radians 模式下不得把 URDF 弧度误当 0~255 硬件向量
            return

        # 机载 /joint_states：按名抽取，再转到原语空间
        if msg.name and self._hand_type == "l25":
            native = joint_state_to_l25_native(msg.name, msg.position, self._side)
            if native is not None:
                self._current_hw_angles = list(native)
                self._current_angles = self._hardware_to_semantic(native)
                self._sync_semantic_rad_from_legacy_angles()
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
                return

        if msg.name and self._hand_type in ("o6", "l6") and self._profile is not None:
            hw = extract_o6_from_joint_state(msg.name, msg.position, self._side)
            if hw is not None:
                self._current_hw_angles = list(hw)
                self._current_angles = self._hardware_to_semantic(hw)
                self._sync_semantic_rad_from_legacy_angles()
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
                return

        if msg.name and self._hand_type == "o20":
            hw = extract_o20_from_joint_state(msg.name, msg.position, self._side)
            if hw is not None:
                self._current_hw_angles = list(hw)
                self._current_angles = self._hardware_to_semantic(hw)
                self._sync_semantic_rad_from_legacy_angles()
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
                return

        if msg.name and self._hand_type == "o30" and self._profile is not None:
            hw = extract_o30_from_joint_state(msg.name, msg.position, self._side)
            if hw is not None:
                self._current_hw_angles = list(hw)
                self._current_angles = self._hardware_to_semantic(hw)
                self._sync_semantic_rad_from_legacy_angles()
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
                return

        if msg.name and self._hand_type == "g20" and self._profile is not None:
            hw, uses_l25 = joint_state_to_g20_hardware(msg.name, msg.position, self._side)
            if hw is not None:
                if not self._g20_l25_urdf_locked:
                    self._g20_l25_urdf = uses_l25
                self._current_hw_angles = list(hw)
                self._current_angles = self._hardware_to_semantic(hw)
                self._sync_semantic_rad_from_legacy_angles()
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
                return

        n = len(msg.position)
        if self._hand_type in ("o6", "l6") and self._profile is not None:
            if n == self._profile.hardware.num_joints:
                self._current_hw_angles = list(msg.position)
                self._current_angles = self._hardware_to_semantic(list(msg.position))
                self._sync_semantic_rad_from_legacy_angles()
                self._hand_state_received = True
                self._feedback_received_at = time.monotonic()
            return
        # 兼容旧 /cb_*_hand_state：纯 position 向量、长度=num_joints
        if n == self._hand_config.num_joints:
            self._current_hw_angles = list(msg.position)
            self._current_angles = self._hardware_to_semantic(list(msg.position))
            self._sync_semantic_rad_from_legacy_angles()
            self._hand_state_received = True
            self._feedback_received_at = time.monotonic()

    def _note_commanded_hw(self, angles: List[float]) -> None:
        """记录产品 hardware 命令向量（与 NamedPose / _last_target 同空间）。"""
        n = self._hand_config.num_joints
        hw = [max(0.0, min(255.0, float(v))) for v in angles[:n]]
        if len(hw) < n:
            hw.extend([0.0] * (n - len(hw)))
        self._last_commanded_hw = hw[:n]

    def _hold_hw_for_resume(self) -> List[float]:
        """退出 TEACH/切模式时的 hold：优先最近命令，其次反馈。"""
        n = self._hand_config.num_joints
        if len(self._last_commanded_hw) == n:
            return list(self._last_commanded_hw)
        if self._hand_state_received:
            return self._hw_feedback_target()
        if len(self._last_target) == n:
            return list(self._last_target)
        return [0.0] * n

    def _cmd_hw_for_context(self) -> Optional[List[float]]:
        n = self._hand_config.num_joints
        if len(self._last_commanded_hw) == n:
            return list(self._last_commanded_hw)
        if len(self._last_target) == n and any(abs(v) > 1e-6 for v in self._last_target):
            return list(self._last_target)
        return None

    def _external_cmd_callback(self, msg: Float64MultiArray) -> None:
        """镜像 hand_controller/commands（示教 panel 下发）。仅 TEACH 时吸收，避免自回环干扰。"""
        if self._control_mode != ControlMode.TEACH:
            return
        data = list(msg.data)
        if self._ros2_control_command_units == "radians":
            if self._radian_converter is None or not data:
                return
            semantic = self._radian_converter.radians_to_semantic(data)
            self._note_commanded_hw(self._semantic_to_hardware(list(semantic)))
            return
        self._note_commanded_hw(data)

    def _external_legacy_cmd_callback(self, msg: JointState) -> None:
        if self._control_mode != ControlMode.TEACH:
            return
        if msg.position:
            self._note_commanded_hw(list(msg.position))

    def _hw_feedback_target(self) -> List[float]:
        """用最新硬件反馈重建 _last_target，保留全部关节（含指尖）。"""
        if self._current_hw_angles:
            return [float(v) for v in self._current_hw_angles]
        return self._semantic_to_hardware(self._current_angles)

    def _hardware_to_semantic(self, hw: List[float]) -> List[float]:
        """硬件 0~255 → S30 legacy（优先 to_semantic，与 from_semantic 命令路径对称）。"""
        if self._profile is not None and self._profile.hardware.has_from_semantic:
            return self._profile.hardware.to_semantic(list(hw))
        if self._profile is not None:
            return project_o20_to_s30(self._profile.hardware.from_hardware(list(hw)))
        return project_o20_to_s30(list(hw[:20]))

    def _semantic_to_hardware(self, semantic: List[float]) -> List[float]:
        """原语关节空间（S30 或兼容 O20）→ 驱动发布向量。"""
        # P6：有 from_semantic（YAML 或由 from_o20 合成）时直接通道名映射
        if (
            self._profile is not None
            and len(semantic) >= SEMANTIC_DIM
            and self._profile.hardware.has_from_semantic
        ):
            hw = self._profile.hardware.from_semantic(semantic)
            return [max(0.0, min(255.0, v)) for v in hw]

        if len(semantic) >= SEMANTIC_DIM:
            o20 = project_s30_to_o20(semantic)
        else:
            o20 = list(semantic[:20])
            while len(o20) < 20:
                o20.append(0.0)
            for i in RESERVED_INDICES:
                o20[i] = 0.0
        # 无通道映射时的兜底：YAML from_o20 逐关节映射
        if self._profile is not None:
            hw = self._profile.hardware.to_hardware(o20)
            return [max(0.0, min(255.0, v)) for v in hw]
        # 无 profile 兜底（O20 等身份映射或旧 hand_type）
        n = self._hand_config.num_joints
        target = list(o20)
        if len(target) < n:
            target.extend([0.0] * (n - len(target)))
        target = target[:n]
        for i in self._hand_config.reserved_indices:
            if i < len(target):
                target[i] = 0.0
        if getattr(self._hand_config, "invert_angles", False):
            target = [255.0 - v for v in target]
        return [max(0.0, min(255.0, v)) for v in target]

    def _sync_last_target_rad_from_hw(self, hw: List[float]) -> None:
        """由 hardware 目标刷新 ``_last_target_semantic_rad``。"""
        if self._radian_converter is None:
            self._last_target_semantic_rad = None
            return
        semantic_255 = self._hardware_to_semantic(list(hw))
        self._last_target_semantic_rad = self._radian_converter.legacy255_to_semantic_rad(
            semantic_255
        )

    def _apply_feasible_result(self, result: PrimitiveResult) -> None:
        """根据 PrimitiveResult.space 更新 hardware / semantic_rad 目标缓存。"""
        space = getattr(result, "space", "semantic")
        if space == "hardware":
            self._last_target = [float(v) for v in result.target_angles]
            self._sync_last_target_rad_from_hw(self._last_target)
            return
        if space == "semantic_rad":
            # semantic_rad：30 维弧度；同步 legacy255→hardware 供 hold/收敛
            self._last_target_semantic_rad = as_semantic_30(result.target_angles)
            if self._radian_converter is not None:
                semantic_255 = self._radian_converter.semantic_rad_to_legacy255(
                    self._last_target_semantic_rad
                )
                self._last_target = self._semantic_to_hardware(semantic_255)
            else:
                self._last_target = self._semantic_to_hardware(
                    as_semantic_30(result.target_angles)
                )
            return
        # legacy semantic 0～255（S30 或兼容 O20）
        semantic_255 = as_semantic_30(result.target_angles)
        self._last_target = self._semantic_to_hardware(semantic_255)
        if self._radian_converter is not None:
            self._last_target_semantic_rad = self._radian_converter.legacy255_to_semantic_rad(
                semantic_255
            )
        else:
            self._last_target_semantic_rad = None

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
            f"cmd订阅数={subs} ({self._cmd_topic})"
        )
        if subs == 0:
            self._node.get_logger().info(
                f"启动瞬间未见 {self._cmd_topic} 订阅者（DDS 发现竞态，通常稍后恢复）"
            )

    def _log_infeasible_hold(
        self,
        primitive_name: str,
        result: PrimitiveResult,
        ctx: PrimitiveContext,
    ) -> None:
        if self._infeasible_hold_logged:
            return
        reason = result.hold_reason or self._infer_hold_reason(ctx)
        self._node.get_logger().warn(
            f"原语 '{primitive_name}' 返回 hold (infeasible): {reason}；"
            "保持上帧目标角度，手部可能看起来无动作"
        )
        self._infeasible_hold_logged = True

    def _infer_hold_reason(self, ctx: PrimitiveContext) -> str:
        if ctx.tcp_pose is None:
            return f"缺少 {self._tcp_pose_topic}（常见：未发布或 topic 不一致）"
        if ctx.object_pose is None:
            return f"缺少 {self._object_pose_topic}（GraspGate 未放行或未发布选中位姿）"
        return "原语内部条件未满足（如前向距离 2–15cm 等，详见原语实现）"

    def _log_motion_started(self, primitive_name: str, target: List[float]) -> None:
        if self._motion_logged:
            return
        # 与命令路径一致：优先当前硬件反馈；O30 等勿误用 project+to_hardware
        if self._current_hw_angles and len(self._current_hw_angles) >= len(target):
            compare = list(self._current_hw_angles)
        elif self._profile is not None:
            compare = self._semantic_to_hardware(as_semantic_30(self._current_angles))
        else:
            compare = self._current_angles
        n = min(len(target), len(compare))
        if n == 0:
            return
        worst_i = max(range(n), key=lambda i: abs(target[i] - compare[i]))
        delta = abs(target[worst_i] - compare[worst_i])
        if delta < 1.0:
            return
        joint = ""
        if self._profile is not None and worst_i < len(self._profile.hardware.joint_names):
            joint = f", 最大通道={self._profile.hardware.joint_names[worst_i]}"
        self._node.get_logger().info(
            f"原语 '{primitive_name}' 开始输出目标角度，最大关节变化={delta:.1f} (hw255){joint}"
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
            parse_hand_info_torque(data, n) if uses_torque_feedback(self._hand_type) else None
        )
        if currents is None and torques is None:
            return

        if currents is not None:
            # SDK 未收到 CAN 0x36 前为 -1；仅更新有效采样，保留上帧值
            for i, c in enumerate(currents):
                if c >= 0:
                    self._joint_currents[i] = c
            # linkerhand_hardware 的 L6 current 当前可能是全 0 占位；
            # legacy SDK 无 source 字段，且已实测 CAN 0x36 会返回实时值。
            placeholder = (
                self._hand_type == "l6"
                and data.get("source") == "linkerhand_hardware"
                and not any(c > 0 for c in currents)
            )
            if not placeholder and any(c >= 0 for c in currents):
                self._joint_currents_valid = True
                sensed = [v for i, v in enumerate(currents) if i not in self._hand_config.reserved_indices]
                if sensed and np.isfinite(sensed).all() and np.all(np.asarray(sensed) >= 0):
                    self._workbench_current_at = time.monotonic()

        if torques is not None:
            self._record_torque_feedback("hand_info", torques)
        now = time.monotonic()
        # joint_currents 为硬件序：跳过列表必须用 hardware 下标，不能用 S30/机型表混用下标
        skip_overload: set[int] = set()
        if self._profile is not None:
            skip_overload.update(int(i) for i in self._profile.hardware.reserved_indices)
            from .contact_resolver import current_index_for_s30
            from .semantic_30 import SEMANTIC_30_IDX

            # 固定用 S30 thumb_rot 槽再映射；勿用 HAND_CONFIGS[g20].thumb_rot（硬件序）
            hi = current_index_for_s30(self._profile.hardware, SEMANTIC_30_IDX["thumb_rot"])
            if hi is not None and hi >= 0:
                skip_overload.add(int(hi))
        overload = self._contact_thresholds.overload_threshold
        overload_duration = self._contact_thresholds.overload_duration_sec
        for i, c in enumerate(self._joint_currents):
            if c < 0:
                continue
            if i in skip_overload:
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
        joint_name = (
            JOINT_NAMES[joint_idx]
            if (self._hand_type == "o20" and joint_idx < len(JOINT_NAMES))
            else f"[{joint_idx}]"
        )
        t = self._contact_thresholds
        self._node.get_logger().warn(
            f"电流过载保护触发: {joint_name}[{joint_idx}] 电流={current_value}mA, "
            f"超过{t.overload_threshold}mA持续>{t.overload_duration_sec}s → fallback到init"
        )
        self._joint_overload_start = [None] * self._hand_config.num_joints
        self.set_primitive(PRIMITIVE_REGISTRY["init"]())

    def _tcp_pose_callback(self, msg: PoseStampedMsg) -> None:
        self._workbench_tcp_at = time.monotonic()
        self._tcp_pose = PoseStamped(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            orientation=np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            ),
            frame_id=msg.header.frame_id or "",
            stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        )

    def _object_pose_callback(self, msg: PoseStampedMsg) -> None:
        self._workbench_object_at = time.monotonic()
        self._object_pose = PoseStamped(
            position=np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            orientation=np.array(
                [
                    msg.pose.orientation.x,
                    msg.pose.orientation.y,
                    msg.pose.orientation.z,
                    msg.pose.orientation.w,
                ]
            ),
            frame_id=msg.header.frame_id or "",
            stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        )
        if not self._object_label:
            self._object_label = self._target_object_label

    def _bboxes_3d_callback(self, msg) -> None:
        """Keep the full observation so a later auto(label) can select another target."""
        self._last_bbox_message = msg
        self._last_bbox_received_at = time.monotonic()
        self._apply_bbox_observation(msg, self._last_bbox_received_at)

    def refresh_target_from_cached_bbox(self) -> None:
        from .workbench_support import FEEDBACK_MAX_AGE_SEC

        msg = getattr(self, "_last_bbox_message", None)
        stamp = getattr(self, "_last_bbox_received_at", 0.0)
        if msg is None or not 0 <= time.monotonic() - stamp <= FEEDBACK_MAX_AGE_SEC:
            return
        best = select_best_bbox(msg.boxes, target_label=self._target_label,
                                target_instance_id=self._target_instance_id)
        if best is None:
            return
        size, center, orientation = (np.asarray(value, dtype=float) for value in
                                     (best.size, best.center, best.orientation))
        if (size.shape != (3,) or center.shape != (3,) or orientation.shape != (4,)
                or not all(np.isfinite(value).all() for value in (size, center, orientation))
                or np.any(size < 0) or not np.any(size > 0)):
            return
        self._apply_bbox_observation(msg, stamp)
        if self._object_center is not None:
            self._object_pose = PoseStamped(
                position=self._object_center.copy(), orientation=self._object_orientation.copy(),
                frame_id=self._object_center_frame,
            )
            self._object_label = best.label
            self._workbench_object_at = stamp

    def _apply_bbox_observation(self, msg, received_at) -> None:
        """Apply a selected observation without renewing a cached sample's freshness."""
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
            self._workbench_bbox_at = received_at
            self._object_size = size
            self._object_center = np.array(best.center, dtype=np.float64)
            self._object_center_frame = best.frame_id or msg.header.frame_id or ""
            self._object_orientation = np.array(best.orientation, dtype=np.float64)
            self._grasp_type = best.grasp_type
            self._bbox_miss_count = 0
            if prev_size is None:
                sz_mm = size * 1000
                all_labels = [f"{b.label}({b.score:.2f})" for b in msg.boxes]
                filter_desc = self._target_label or "(best)"
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
        # 与 bbox 回调一致：优先动态 _target_label，回退构造参数 _target_object_label
        want = (self._target_label or self._target_object_label or "").strip()
        want_n = normalize_label(want) if want else ""
        for lp in msg.poses:
            if want_n:
                label_n = normalize_label(lp.label)
                if label_n != want_n and lp.label != want:
                    continue
            pose = lp.pose
            self._object_pose = PoseStamped(
                position=np.array([pose.position.x, pose.position.y, pose.position.z]),
                orientation=np.array(
                    [
                        pose.orientation.x,
                        pose.orientation.y,
                        pose.orientation.z,
                        pose.orientation.w,
                    ]
                ),
                stamp_sec=msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            )
            self._workbench_object_at = time.monotonic()
            self._object_label = lp.label
            return

    def _selected_label_callback(self, msg: String) -> None:
        """接收门控选中的物体 label，覆盖默认值。"""
        if msg.data and msg.data != self._object_label:
            self._object_label = msg.data
            self._node.get_logger().info(f"门控选中物体: {msg.data}")

    # ------------------------------------------------------------------
    # 触觉 / 力矩回调
    # ------------------------------------------------------------------

    def _force_callback(self, msg) -> None:
        """linkerhand_hardware HandForce：finger_force 矩阵 [finger][row][col]，
        每指取压感最大值（0-255）；palm_force [row][col] 取掌心压感峰值。
        指尖与掌心独立解析（驱动任一可用即发布，缺失侧为空数组）。"""
        per_finger = self._extract_finger_pressure(msg)
        palm_peak = self._extract_palm_pressure(msg)
        if per_finger is None and palm_peak is None:
            return
        if per_finger is not None:
            self._workbench_touch_at = time.monotonic()
            self._tactile_pressure = per_finger
            self._tactile_mode = "pressure"
        self._palm_pressure = palm_peak
        threshold = self._contact_thresholds.pressure_threshold
        finger_hit = per_finger is not None and bool(np.any(per_finger > threshold))
        palm_hit = palm_peak is not None and palm_peak > threshold
        self._contact_detected = finger_hit or palm_hit
        self._force_primary_stamp = time.monotonic()

    def _force_primary_fresh(self) -> bool:
        """主触觉源是否仍在有效窗口内（过期后允许 matrix_touch fallback）。"""
        if self._force_primary_stamp <= 0.0:
            return False
        return (time.monotonic() - self._force_primary_stamp) < _FORCE_PRIMARY_STALE_SEC

    @staticmethod
    def _extract_finger_pressure(msg) -> Optional[np.ndarray]:
        """从 HandForce.finger_force（行主序展平矩阵）提取各指压力峰值。

        layout.dim = [finger, row, col]；每指连续占据 row*col 个元素。
        finger_names 携带指序时按名称对齐到固定指序，否则按矩阵行序。
        """
        dim = msg.finger_force.layout.dim
        if len(dim) < 3:
            return None
        n_finger = int(dim[0].size)
        per_finger_cells = int(dim[1].size) * int(dim[2].size)
        if n_finger <= 0 or per_finger_cells <= 0:
            return None
        data = np.asarray(msg.finger_force.data, dtype=np.float64)

        order = list(range(n_finger))
        names = [str(n) for n in getattr(msg, "finger_names", []) or []]
        if names and len(names) == n_finger:
            try:
                order = [names.index(want) for want in _TACTILE_FINGER_NAMES]
            except ValueError:
                order = list(range(n_finger))

        peaks = []
        for idx in order[: len(_TACTILE_FINGER_NAMES)]:
            seg = data[idx * per_finger_cells : (idx + 1) * per_finger_cells]
            peaks.append(float(seg.max()) if seg.size else 0.0)
        if not peaks:
            return None
        return np.array(peaks, dtype=np.float64)

    @staticmethod
    def _extract_palm_pressure(msg) -> Optional[float]:
        """从 HandForce.palm_force（[row][col] 行主序展平）提取掌心压感峰值。

        空矩阵（无掌心传感器数据）返回 None；不依赖 dim 声明，直接取全量
        数据最大值，锯齿矩阵安全。
        """
        data = msg.palm_force.data
        if not data:
            return None
        return float(np.asarray(data, dtype=np.float64).max())

    def _normal_force_callback(self, msg: Float32MultiArray) -> None:
        """法向力反馈：每指一个牛顿值。"""
        if len(msg.data) >= 5:
            self._workbench_touch_at = time.monotonic()
            self._tactile_pressure = np.array(msg.data[:5], dtype=np.float64)
            self._tactile_mode = "normal_force_n"
            threshold = self._contact_thresholds.normal_force_threshold_n
            self._contact_detected = bool(np.any(self._tactile_pressure > threshold))
            self._force_primary_stamp = time.monotonic()

    def _matrix_touch_mass_callback(self, msg: String) -> None:
        """O20/SDK 矩阵压感合值 (JSON): 每指总压力质量 (g) 作为触觉 fallback。"""
        if self._force_primary_fresh():
            return
        try:
            data = json.loads(msg.data)
        except (json.JSONDecodeError, ValueError):
            return
        keys = ("thumb_mass", "index_mass", "middle_mass", "ring_mass", "little_mass")
        values = [float(data.get(k, 0.0)) for k in keys]
        pressure = np.array(values, dtype=np.float64)
        self._workbench_touch_at = time.monotonic()
        self._tactile_pressure = pressure
        self._tactile_mode = "mass"
        threshold = self._contact_thresholds.mass_threshold
        self._contact_detected = bool(np.any(pressure > threshold))

    def _record_torque_feedback(self, source, values):
        """Cache coherent samples independently; invalid messages never renew freshness."""
        n = self._hand_config.num_joints
        sample = np.asarray(values, dtype=float)
        reserved = self._hand_config.reserved_indices
        sensed = [v for i, v in enumerate(sample) if i not in reserved]
        if sample.shape == (n,) and sensed and np.isfinite(sensed).all() and np.all(np.asarray(sensed) >= 0):
            if not hasattr(self, "_torque_samples"):
                self._torque_samples = {}
            self._torque_samples[source] = (sample.copy(), time.monotonic())
        self._select_torque_feedback()

    def _select_torque_feedback(self):
        from .workbench_support import FEEDBACK_MAX_AGE_SEC

        now = time.monotonic()
        samples = getattr(self, "_torque_samples", {})
        sources = ("hand_info", "motor_torque") if uses_torque_feedback(self._hand_type) else ("motor_torque",)
        for source in sources:
            sample = samples.get(source)
            if sample is not None and 0 <= now - sample[1] <= FEEDBACK_MAX_AGE_SEC:
                self._joint_torque = sample[0].copy()
                self._workbench_torque_at = sample[1]
                self._torque_source = source
                return
        self._joint_torque = None
        self._workbench_torque_at = 0.0
        self._torque_source = None

    def _torque_callback(self, msg: Float32MultiArray) -> None:
        """Retain the topic backup even while a fresh hand_info sample has priority."""
        normalized = normalize_motor_torque_values(list(msg.data), self._hand_config.num_joints)
        if normalized is not None:
            self._record_torque_feedback("motor_torque", normalized)

    def _tick(self) -> None:
        self._select_torque_feedback()
        deadline = getattr(self, "_motion_deadline", None)
        clock = getattr(self, "_playback_clock", None)
        if deadline is not None and time.monotonic() >= deadline:
            if clock and clock.held:
                clock.set_running(False, time.monotonic())
                self._motion_expired = True
                self._motion_deadline = None
            elif getattr(self, "_live_active", False):
                self.stop()
        # FK 指尖位置计算
        fingertip_positions = self._compute_fingertip_positions()

        if self._control_mode == ControlMode.STREAM:
            now = time.monotonic()
            if (
                self._last_stream_t > 0.0
                and (now - self._last_stream_t) > self._stream_watchdog_sec
            ):
                if not self._stream_watchdog_logged:
                    self._node.get_logger().warn(
                        f"STREAM watchdog：>{self._stream_watchdog_sec}s 无目标，hold 当前"
                    )
                    self._stream_watchdog_logged = True
            # STREAM 不跑原语 compute；目标由 _stream_target_callback 更新
        elif self._active_primitive is not None and not (clock and not clock.running):
            if self._active_primitive.done:
                if self._settle_completed_motion():
                    self.publish_user_result(f"DONE: {self._active_primitive.name}")
                    if not (clock and clock.held):
                        self._active_primitive.on_exit()
                        self._active_primitive = None
            else:
                elapsed = clock.advance(time.monotonic()) if clock else (time.monotonic() - self._primitive_start_time) * getattr(self, "_primitive_speed", 1.0)
                ctx = PrimitiveContext(
                    tcp_pose=self._tcp_pose,
                    object_pose=self._object_pose,
                    object_label=self._object_label,
                    joint_currents=list(self._joint_currents),
                    joint_currents_valid=self._joint_currents_valid,
                    hand_type=self._hand_type,
                    hand_side=self._side,
                    hand_hw_angles=(
                        list(self._current_hw_angles) if self._current_hw_angles else None
                    ),
                    hand_cmd_angles=self._cmd_hw_for_context(),
                    object_size=self._object_size,
                    object_orientation=self._object_orientation,
                    grasp_type=self._grasp_type,
                    tactile_pressure=(
                        self._tactile_pressure.copy()
                        if self._tactile_pressure is not None
                        else None
                    ),
                    tactile_mode=self._tactile_mode,
                    contact_detected=self._contact_detected,
                    joint_torque=self._joint_torque,
                    fingertip_positions=fingertip_positions,
                    contact_thresholds=self._contact_thresholds,
                    control_rate=self._control_rate,
                    motion_speed=getattr(self, "_primitive_speed", 1.0),
                    angle_unit=self._primitive_angle_unit(),
                    radian_converter=self._radian_converter,
                )
                try:
                    result = self._active_primitive.compute(
                        self._primitive_current_angles(), elapsed, ctx
                    )
                except Exception as exc:
                    self.stop()
                    self.publish_user_result(f"rejected:primitive_compute_failed:{exc}")
                    self._node.get_logger().error(
                        f"原语 compute 异常: {exc}；保持当前反馈位置"
                    )
                else:
                    try:
                        if result.feasible:
                            self._apply_feasible_result(result)
                            self._trajectory_complete = self._active_primitive.trajectory_complete(result, elapsed)
                            if clock:
                                clock.finished = self._trajectory_complete
                            if self._trajectory_complete:
                                self._settle_completed_motion()
                            self._log_motion_started(
                                self._active_primitive.name, self._last_target
                            )
                        else:
                            self._log_infeasible_hold(self._active_primitive.name, result, ctx)
                        self._log_grasp_state(ctx)
                    except Exception as exc:
                        self.stop()
                        self.publish_user_result(f"rejected:primitive_result_failed:{exc}")
                        self._node.get_logger().error(
                            f"apply_result 异常: {exc}；"
                            f"保持上一目标"
                        )

        # 始终发送（有原语发计算结果，无原语/infeasible 发上次的目标值）
        # TEACH 模式让权给示教端，禁止抢发 cmd
        if self._control_mode != ControlMode.TEACH:
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

        # 执行结果反馈 (节流到约 1Hz)
        self._publish_result()

    def _compute_fingertip_positions(self):
        if self._fk_solver is None:
            return None
        try:
            # 有 Converter 则 FK 跟内部 semantic_rad（与 command_units 边界解耦）
            if self._radian_converter is not None:
                return self._fk_solver.compute_fingertips_rad(self._current_semantic_rad)
            return self._fk_solver.compute_fingertips(self._current_angles)
        except Exception as error:
            if not self._fk_err_logged:
                self._node.get_logger().warn(f"FK 计算异常 (后续不再重复): {error}")
                self._fk_err_logged = True
            return None

    def _log_grasp_state(self, ctx: PrimitiveContext) -> None:
        """输出触觉 + 抓取状态日志 (仅状态变化时输出)。"""
        prim = self._active_primitive
        grasp_state = getattr(prim, "grasp_state", None)
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
            palm_str = (
                f" palm={self._palm_pressure:.0f}" if self._palm_pressure is not None else ""
            )
            feedback = (
                "tactile"
                if (self._tactile_mode != "none" and self._tactile_pressure is not None)
                else "current"
            )
            stopped = getattr(getattr(prim, "_engine", None), "_stopped", None)
            if stopped is None:
                stopped = getattr(prim, "_stopped", None)
            stopped_str = (
                f" stopped={sorted(k for k, v in stopped.items() if v)}"
                if isinstance(stopped, dict) and any(stopped.values())
                else ""
            )
            self._node.get_logger().info(
                f"[{prim.name}] grasp={grasp_state} | "
                f"feedback={feedback} tactile={p_str} mode={self._tactile_mode}"
                f"{palm_str} "
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
        self._note_commanded_hw(angles)
        if self._cmd_backend == "ros2_control":
            if self._cmd_pub.get_subscription_count() == 0:
                if not self._cmd_no_sub_warned:
                    self._node.get_logger().warn(
                        f"无节点订阅 {self._cmd_topic}，关节指令不会被执行"
                    )
                    self._cmd_no_sub_warned = True
            else:
                self._cmd_no_sub_warned = False
            if self._ros2_control_command_units == "radians":
                if self._radian_converter is None:
                    return
                if self._last_target_semantic_rad is not None:
                    semantic_rad = as_semantic_30(self._last_target_semantic_rad)
                else:
                    semantic_255 = self._hardware_to_semantic(list(angles))
                    semantic_rad = self._radian_converter.legacy255_to_semantic_rad(semantic_255)
                msg = Float64MultiArray()
                msg.data = self._radian_converter.semantic_rad_to_controller(semantic_rad)
                self._cmd_pub.publish(msg)
                return

            # angles 已是硬件向量：L25 native25/active16；O6/L6×6；O20×20；G20×16。
            if self._hand_type in ("o6", "l6"):
                n = 6
                data = [float(a) for a in angles[:n]]
                if len(data) < n:
                    data.extend([0.0] * (n - len(data)))
            elif self._hand_type in ("o20", "o30"):
                n = 20
                data = [float(a) for a in angles[:n]]
                if len(data) < n:
                    data.extend([0.0] * (n - len(data)))
            elif self._hand_type == "g20":
                n = 16
                data = [float(a) for a in angles[:n]]
                if len(data) < n:
                    data.extend([0.0] * (n - len(data)))
                if self._g20_l25_urdf:
                    data = g20_native_to_l25_urdf16(data)
            elif len(angles) >= 25:
                data = native25_to_active16(angles)
            elif len(angles) >= 16:
                data = [float(a) for a in angles[:16]]
            else:
                data = [float(a) for a in angles] + [0.0] * (16 - len(angles))
            msg = Float64MultiArray()
            msg.data = [max(0.0, min(255.0, float(a))) for a in data]
            self._cmd_pub.publish(msg)
            return

        msg = JointState()
        msg.header = Header()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = (
            list(JOINT_NAMES)
            if self._hand_type == "o20"
            else (
                list(O30_ACTIVE_SUFFIXES)
                if self._hand_type == "o30"
                else [f"j{i}" for i in range(self._hand_config.num_joints)]
            )
        )
        msg.position = [float(a) for a in angles]
        # 勿填 effort/velocity：linker_hand_sdk 会把 effort 当作 set_torque，
        # 全 0 会覆盖 teach 力矩导致手无法运动。
        self._cmd_pub.publish(msg)

    def publish_user_result(self, text: str) -> None:
        """立刻发布一条用户可见结果（如 POSES:a,b），不节流。"""
        msg = String()
        msg.data = str(text or "")
        self._result_pub.publish(msg)
        self._last_result_msg = msg.data
        if msg.data.lower().startswith("done:"):
            self._terminal_result_until = time.monotonic() + 1.5

    def _publish_result(self) -> None:
        """约每秒发布一次执行状态到 result_topic（默认 /{side}_hand/gesture_result）。"""
        if self._active_primitive is None and time.monotonic() < self._terminal_result_until:
            return
        self._result_msg_count += 1
        if self._result_msg_count % self._result_period_ticks != 0:
            return

        if self._active_primitive is None:
            msg_text = "IDLE"
        elif self._active_primitive.done or self._settle_status == "settle_timeout":
            msg_text = f"{self._settle_status.upper()}: {self._active_primitive.name}"
        elif self._progress_driven_active():
            # progress scrub / 外部细控：不以反馈收敛冒充 DONE
            if self._object_label:
                msg_text = f"EXECUTING: {self._active_primitive.name}, obj={self._object_label}"
            else:
                msg_text = f"EXECUTING: {self._active_primitive.name}"
        elif self._feedback_command_converged():
            self._feedback_completion_reported = True
            msg_text = f"DONE: {self._active_primitive.name}"
        elif self._object_label:
            msg_text = f"EXECUTING: {self._active_primitive.name}, obj={self._object_label}"
        else:
            msg_text = f"EXECUTING: {self._active_primitive.name}"

        motion = getattr(self._active_primitive, "motion_status", "") or ""
        if motion:
            msg_text = f"{msg_text} | {motion}"

        if msg_text != self._last_result_msg:
            result_msg = String()
            result_msg.data = msg_text
            self._result_pub.publish(result_msg)
            self._last_result_msg = msg_text

    def _settle_completed_motion(self):
        now = time.monotonic()
        if self._settle_since is None:
            self._settle_since = now
        if self._settle_status in {"settled", "settle_timeout"}:
            return self._settle_status == "settled"
        measured, target = np.asarray(self._current_hw_angles), np.asarray(self._last_target)
        fresh = 0 <= now - self._feedback_received_at <= 0.5
        valid = (measured.shape == target.shape and measured.size > 0
                 and np.isfinite(measured).all() and np.isfinite(target).all())
        close = False
        if fresh and valid:
            profile = getattr(self, "_profile", None)
            if profile is not None:
                # The profile identifies genuinely measured channels. L25 tips
                # and reserved/coupled joints must not prevent completion, but
                # freshness and three distinct feedback messages still apply.
                indices = profile.hardware.convergence_channel_indices()
                measured = np.asarray(self._hardware_to_semantic(measured.tolist()))
                target = np.asarray(self._hardware_to_semantic(target.tolist()))
                close = bool(indices) and np.max(np.abs(measured[indices] - target[indices])) <= 3.0
            else:
                close = np.max(np.abs(measured - target)) <= 3.0
        if self._feedback_received_at != self._settle_feedback_at:
            self._settle_feedback_at = self._feedback_received_at
            self._settle_frames = self._settle_frames + 1 if close else 0
        self._settle_status = "settling"
        if self._settle_frames >= 3:
            self._settle_status = "settled"
            return True
        if now - self._settle_since >= 1.0:
            self._settle_status = "settle_timeout"
            self.publish_user_result(f"SETTLE_TIMEOUT: {self._active_primitive.name}")
        return False

    def _progress_driven_active(self) -> bool:
        """当前原语是否处于外部 progress 细控（禁止反馈假 DONE）。"""
        prim = self._active_primitive
        if prim is None:
            return False
        return bool(getattr(prim, "progress_driven", False))

    def _feedback_command_converged(self) -> bool:
        """Never treat an intermediate commanded pose as trajectory completion."""
        return self._trajectory_complete and self._settle_status == "settled"
