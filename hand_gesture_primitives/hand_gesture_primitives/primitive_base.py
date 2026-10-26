"""手势原语基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .contact_config import ContactThresholds

# O20 20-DOF 位置命令顺序 (与 linker_hand_o20_ros2 JointState.name 一致)
# 值域: 0-255 (uint8)，驱动内部通过 uint8_to_angle 映射到各电机角度范围
#
# 各位置含义:
#   [0]  thumb_base  → 拇指指根弯曲 (0=伸直, 255=弯曲120°)
#   [1]  index_base  → 食指指根弯曲 (0=伸直, 255=弯曲180°)
#   [2]  middle_base → 中指指根弯曲 (0=伸直, 255=弯曲180°)
#   [3]  ring_base   → 无名指指根弯曲 (0=伸直, 255=弯曲180°)
#   [4]  pinky_base  → 小指指根弯曲 (0=伸直, 255=弯曲180°)
#   [5]  thumb_abd   → 拇指侧摆 (0=0°, 255=180°)
#   [6]  index_abd   → 食指侧摆 (0=-30°, 128≈0°中立, 255=+30°)
#   [7]  middle_abd  → 中指侧摆 (0=-30°, 128≈0°中立, 255=+30°)
#   [8]  ring_abd    → 无名指侧摆 (0=-20°, 128≈0°中立, 255=+20°)
#   [9]  pinky_abd   → 小指侧摆 (0=-20°, 128≈0°中立, 255=+20°)
#   [10] thumb_rot   → 拇指旋转 (0=0°, 255=130°)
#   [11-14] rsv      → 预留位 (恒为 0)
#   [15] thumb_tip   → 拇指指尖弯曲 (0=伸直, 255=弯曲150°)
#   [16] index_tip   → 食指指尖弯曲 (0=伸直, 255=弯曲180°)
#   [17] middle_tip  → 中指指尖弯曲 (0=伸直, 255=弯曲180°)
#   [18] ring_tip    → 无名指指尖弯曲 (0=伸直, 255=弯曲180°)
#   [19] pinky_tip   → 小指指尖弯曲 (0=伸直, 255=弯曲180°)

JOINT_NAMES = [
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "pinky_base",
    "thumb_abd",
    "index_abd",
    "middle_abd",
    "ring_abd",
    "pinky_abd",
    "thumb_rot",
    "rsv_11",
    "rsv_12",
    "rsv_13",
    "rsv_14",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]

# 预留位索引 (这些位恒为 0)
RESERVED_INDICES = [11, 12, 13, 14]

# 侧摆关节索引 (这些关节 128 = 中立位)
ABD_INDICES = [6, 7, 8, 9]
ABD_NEUTRAL = 128

NUM_JOINTS = 20

# semantic_30 全链路语义维度（P2 起，executor 内部统一使用 30 维）
SEMANTIC_DIM = 30

# 侧向夹持原语分阶段控制: prep → (arm 就位) → close
GRASP_PHASES = frozenset({"full", "prep", "close"})
# L6 large_wrap 额外支持 auto（full 为兼容别名）
LARGE_WRAP_PHASES = frozenset({"full", "prep", "close", "auto"})
# tripod / tripod_by_force：默认 auto（裸发同效）；prep 才等待 close
TRIPOD_PHASES = frozenset({"full", "prep", "close", "auto"})


def _command_phased_primitives() -> frozenset:
    """gesture_cmd prep/close 集合 — 派生自 ``primitive_catalog.command_phased``。"""
    from .primitive_catalog import command_phased_primitive_names

    return command_phased_primitive_names()


# 历史名保留；真相源为 catalog ``command_phased=True``
PHASED_GRASP_PRIMITIVES = _command_phased_primitives()
TRIPOD_PHASED_PRIMITIVES = frozenset({"tripod", "tripod_by_force"})


def parse_grasp_phase(phase: str) -> str:
    """解析 grasp phase 参数，非法值抛出 ValueError。"""
    p = (phase or "full").lower().strip()
    if p not in GRASP_PHASES:
        raise ValueError(f"grasp phase 必须为 full/prep/close，收到: {phase!r}")
    return p


def parse_large_wrap_phase(phase: str) -> str:
    """解析 O6/L6 large_wrap phase；full 是 auto 的兼容别名。"""
    p = (phase or "auto").lower().strip()
    if p not in LARGE_WRAP_PHASES:
        raise ValueError(f"large_wrap phase 必须为 prep/close/auto，收到: {phase!r}")
    return "auto" if p == "full" else p


def parse_tripod_phase(phase: str) -> str:
    """解析 tripod / tripod_by_force phase。

    默认 ``auto``（裸发 ``tripod`` 同效：prep → 停顿 → 自动 close）；
    ``full`` 为 ``auto`` 别名；``prep`` 到位后等待 close 指令。
    """
    p = (phase or "auto").lower().strip()
    if p not in TRIPOD_PHASES:
        raise ValueError(f"tripod phase 必须为 prep/close/auto/full，收到: {phase!r}")
    return "auto" if p == "full" else p


# ---------------------------------------------------------------------------
# 手型配置 (语义姿态索引 / 硬件元数据)
# ---------------------------------------------------------------------------


def semantic_pose_config(hand_type: str) -> "HandConfig":
    """搭语义姿态时用的索引表（``SemanticPoseConfig``）。

    灵巧手（O20/O30/L25/G20）与欠驱动 O6/L6 均返回 **semantic_30 下标**
    （O6/L6 仅 mcp_flex/thumb_abd 有效，tip/rot/pip = -1）。
    与 ``hand_config.HandConfig``（硬件映射）分离。
    """
    from dataclasses import replace

    key = (hand_type or "").strip().lower()
    if key in ("o6", "l6"):
        # num_joints 保留硬件 6（电流数组元数据）；姿态下标已是 S30
        return HAND_CONFIGS[key]
    if key == "o30":
        # HAND_CONFIGS['o30'].num_joints 保留为硬件 20；构姿表强制 30
        return replace(HAND_CONFIGS["o30"], num_joints=SEMANTIC_DIM)
    return HAND_CONFIGS["s30"]


def to_executor_semantic(angles: List[float]) -> List[float]:
    """原语输出规范为 executor 内部 S30（30 维）。

    - 已是 ≥30：截断
    - 20 / 带 O20 头的 25：``project_o20_to_s30``
    - 更短向量：不在此函数处理（应由 hardware ``space`` 直出）
    """
    from .semantic_30 import SEMANTIC_DIM, project_o20_to_s30

    if len(angles) >= SEMANTIC_DIM:
        return [float(v) for v in angles[:SEMANTIC_DIM]]
    if len(angles) >= 20:
        return project_o20_to_s30(angles[:20])
    return [float(v) for v in angles]


@dataclass
class HandConfig:
    """语义姿态索引表（S30 通道下标）。

    **注意**：与 ``hand_config.HandConfig``（YAML 硬件映射 / from_semantic）是两类。
    新代码请用 ``semantic_pose_config(hand_type)`` 获取本表；对外更清晰的别名是
    ``SemanticPoseConfig``（本模块末尾）。勿与硬件 ``HandConfig`` /
    ``HardwareHandConfig`` 混用。解剖别名：``thumb_base``≡mcp_flex，
    ``thumb_tip``≡dip。``num_joints``：executor 侧硬件 DOF（电流数组长度）；
    姿态构姿用 ``semantic_pose_config().num_joints``（灵巧手为 30）。
    """

    num_joints: int
    # 拇指（S30: flex/abd/pip/dip/rot）
    thumb_base: int  # thumb_mcp_flex
    thumb_abd: int  # thumb_mcp_abd
    thumb_rot: int  # thumb_rot
    thumb_tip: int  # thumb_dip
    thumb_root2: int  # thumb_pip；无独立轴为 -1
    # 食指
    index_base: int  # index_mcp_flex
    index_tip: int  # index_dip
    index_abd: int  # index_mcp_abd
    index_pip: int  # index_pip（S30）；无独立轴为 -1
    # 中指
    middle_base: int
    middle_tip: int
    middle_root2: int  # middle_pip；O30 等启用
    # 无名指
    ring_base: int
    ring_tip: int
    ring_pip: int  # 无独立轴为 -1
    # 小指
    pinky_base: int
    pinky_tip: int
    pinky_pip: int  # 无独立轴为 -1
    # 侧摆中立值
    abd_neutral: int
    # 侧摆关节列表（不含拇指）
    abd_indices: List[int] = field(default_factory=list)
    # 预留位（硬件空间；S30 构姿表为空）
    reserved_indices: List[int] = field(default_factory=list)
    # L25 角度方向与 O20 相反: 0=弯曲, 255=张开（硬件元数据）
    invert_angles: bool = False

    # --- A/B 解剖别名（与字段并存；新代码优先用 mcp_flex / dip）---
    @property
    def thumb_mcp_flex(self) -> int:
        return self.thumb_base

    @property
    def thumb_mcp_abd(self) -> int:
        return self.thumb_abd

    @property
    def thumb_pip(self) -> int:
        return self.thumb_root2

    @property
    def thumb_dip(self) -> int:
        return self.thumb_tip

    @property
    def index_mcp_flex(self) -> int:
        return self.index_base

    @property
    def index_mcp_abd(self) -> int:
        return self.index_abd

    @property
    def index_dip(self) -> int:
        return self.index_tip

    @property
    def middle_mcp_flex(self) -> int:
        return self.middle_base

    @property
    def middle_dip(self) -> int:
        return self.middle_tip

    @property
    def middle_pip(self) -> int:
        return self.middle_root2

    @property
    def ring_mcp_flex(self) -> int:
        return self.ring_base

    @property
    def ring_pip_idx(self) -> int:
        return self.ring_pip

    @property
    def ring_dip(self) -> int:
        return self.ring_tip

    @property
    def pinky_mcp_flex(self) -> int:
        return self.pinky_base

    @property
    def pinky_pip_idx(self) -> int:
        return self.pinky_pip

    @property
    def pinky_dip(self) -> int:
        return self.pinky_tip

    @property
    def index_pip_idx(self) -> int:
        return self.index_pip


# 推荐别名：与 hand_config.HandConfig（硬件映射）区分
SemanticPoseConfig = HandConfig


# semantic_30 权威姿态索引（P3）；thumb_base/tip 等为解剖别名
_S30_POSE = dict(
    num_joints=SEMANTIC_DIM,
    thumb_base=0,
    thumb_abd=1,
    thumb_root2=-1,  # thumb_pip 无电机
    thumb_tip=3,
    thumb_rot=4,
    index_base=5,
    index_abd=6,
    index_pip=7,
    index_tip=8,
    middle_base=10,
    middle_tip=13,
    middle_root2=12,
    ring_base=15,
    ring_pip=17,
    ring_tip=18,
    pinky_base=20,
    pinky_pip=22,
    pinky_tip=23,
    abd_neutral=128,
    abd_indices=[6, 11, 16, 21],
    reserved_indices=[],
)

HAND_CONFIGS: Dict[str, HandConfig] = {
    # 通用灵巧手语义表（O20/L25/G20 姿态构姿）
    "s30": HandConfig(**_S30_POSE),
    "o20": HandConfig(**{**_S30_POSE, "num_joints": 20}),
    "l25": HandConfig(**{**_S30_POSE, "num_joints": 25, "invert_angles": True}),
    "o30": HandConfig(
        **{
            **_S30_POSE,
            "num_joints": 20,
            # 四指 pip 在 O30 有独立 root2 电机
        }
    ),
    "o6": HandConfig(
        # 欠驱动：仅 mcp_flex + thumb_abd；下标为 S30（写入 30 维姿态）
        num_joints=6,
        thumb_base=0,
        thumb_abd=1,
        thumb_rot=-1,
        thumb_tip=-1,
        thumb_root2=-1,
        index_base=5,
        index_tip=-1,
        index_abd=-1,
        index_pip=-1,
        middle_base=10,
        middle_tip=-1,
        middle_root2=-1,
        ring_base=15,
        ring_tip=-1,
        ring_pip=-1,
        pinky_base=20,
        pinky_tip=-1,
        pinky_pip=-1,
        abd_neutral=128,
        abd_indices=[],
        reserved_indices=[],
        invert_angles=False,
    ),
    # L6：与 O6 同为 6-DOF 欠驱动布局；无独立 tip/rot；力控以触觉为主
    "l6": HandConfig(
        num_joints=6,
        thumb_base=0,
        thumb_abd=1,
        thumb_rot=-1,
        thumb_tip=-1,
        thumb_root2=-1,
        index_base=5,
        index_tip=-1,
        index_abd=-1,
        index_pip=-1,
        middle_base=10,
        middle_tip=-1,
        middle_root2=-1,
        ring_base=15,
        ring_tip=-1,
        ring_pip=-1,
        pinky_base=20,
        pinky_tip=-1,
        pinky_pip=-1,
        abd_neutral=128,
        abd_indices=[],
        reserved_indices=[],
        invert_angles=False,
    ),
    # G20：硬件 16 维元数据（姿态构姿走 s30 / semantic_pose_config）
    "g20": HandConfig(
        num_joints=16,
        thumb_base=0,
        thumb_abd=5,
        thumb_rot=10,
        thumb_tip=11,
        thumb_root2=-1,
        index_base=1,
        index_tip=12,
        index_abd=6,
        index_pip=-1,
        middle_base=2,
        middle_tip=13,
        middle_root2=-1,
        ring_base=3,
        ring_tip=14,
        ring_pip=-1,
        pinky_base=4,
        pinky_tip=15,
        pinky_pip=-1,
        abd_neutral=128,
        abd_indices=[6, 7, 8, 9],
        reserved_indices=[],
        invert_angles=True,
    ),
}


# ---------------------------------------------------------------------------
# 外部输入数据结构
# ---------------------------------------------------------------------------


@dataclass
class PoseStamped:
    """简化的位姿表示 (从 geometry_msgs/PoseStamped 缓存)。"""

    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    frame_id: str = ""
    stamp_sec: float = 0.0


def palm_world_position(
    tcp_position: np.ndarray,
    tcp_orientation_xyzw: np.ndarray,
    palm_offset: np.ndarray,
) -> np.ndarray:
    """TCP 系 palm_offset 旋到世界系后的掌心位置。"""
    from scipy.spatial.transform import Rotation

    offset = np.asarray(palm_offset, dtype=np.float64).reshape(3)
    quat = np.asarray(tcp_orientation_xyzw, dtype=np.float64).reshape(4)
    if np.linalg.norm(quat) < 1e-9 or np.allclose(quat, [0.0, 0.0, 0.0, 1.0]):
        rotated = offset
    else:
        rotated = Rotation.from_quat(quat).as_matrix() @ offset
    return np.asarray(tcp_position, dtype=np.float64).reshape(3) + rotated


@dataclass
class PrimitiveContext:
    """每帧传入原语的外部上下文。

    tcp_pose 和 object_pose 均为 world frame 下的位姿。
    原语自行计算两者的相对关系来判断可行性。
    joint_currents: 各关节当前电流 (mA)，按 position order 排列。
    hand_type: 手型标识 ("o20" / "l25" / "o6" / "l6")。
    """

    tcp_pose: Optional[PoseStamped] = None
    object_pose: Optional[PoseStamped] = None
    object_label: str = ""
    joint_currents: List[float] = field(default_factory=lambda: [0.0] * 20)
    joint_currents_valid: bool = False
    hand_type: str = "o20"
    hand_side: str = "left"
    # 最新硬件关节反馈（完整 DOF）；供 NamedPose hardware 直出插值起点
    hand_hw_angles: Optional[List[float]] = None
    # 最近一次下发的产品 hardware 命令（示教/原语）；优先于反馈作为回放起点
    hand_cmd_angles: Optional[List[float]] = None
    # 感知输出: 目标物体几何信息
    object_size: Optional[np.ndarray] = None  # [sx, sy, sz] meters
    object_orientation: Optional[np.ndarray] = None  # quaternion [x, y, z, w]
    grasp_type: str = ""  # "precision"/"lateral"/"power"/""
    # 触觉反馈
    tactile_pressure: Optional[np.ndarray] = None  # [thumb, index, middle, ring, pinky]
    tactile_mode: str = "none"  # none | pressure | mass | normal_force_n
    contact_detected: bool = False  # 任一指检测到接触
    grasp_stable: bool = False  # 抓取稳定 (压力平稳无滑移)
    # 力矩反馈
    joint_torque: Optional[np.ndarray] = None  # 20-DOF 各关节力矩 mNm
    external_force: Optional[np.ndarray] = None  # 末端外力估计 [fx, fy, fz] N
    torque_overload: bool = False  # 力矩超限标志
    # FK 指尖位置
    fingertip_positions: Optional[np.ndarray] = None  # [5,3] xyz (hand_base_link 系, 米)
    contact_thresholds: ContactThresholds = field(default_factory=ContactThresholds)
    # Executor 下发频率 (Hz)。YAML 中按 10Hz 标定的 tick 计数需配合 scale_tick_count。
    control_rate: float = 10.0
    motion_speed: float = 1.0

    @property
    def motion_control_rate(self) -> float:
        """Virtual rate for motion increments only; contact timing uses control_rate."""
        return self.control_rate / self.motion_speed

    # T1：角度单位与换算器（有 URDF 时为 semantic_rad）
    angle_unit: str = "semantic_rad"
    radian_converter: Optional[Any] = None


@dataclass
class PrimitiveResult:
    """原语单帧计算结果。

    ``space``：
    - ``semantic_rad``：S30 槽 + URDF 弧度（执行核默认）
    - ``semantic``：S30 legacy 0～255（无 Converter 退化）
    - ``hardware``：产品 DOF 0～255 直出（驱动/示教边界）
    """

    feasible: bool
    target_angles: List[float] = field(default_factory=list)
    hold_reason: str = ""
    space: str = "semantic_rad"
    keyframe: str = ""
    progress: float = -1.0
    drive_mode: str = ""
    # None retains compatibility with primitives implementing done/engine completion.
    trajectory_finished: Optional[bool] = None


# ---------------------------------------------------------------------------
# 原语基类
# ---------------------------------------------------------------------------


class HandGesturePrimitive(ABC):
    """手势原语基类。

    子类实现 compute() 返回 PrimitiveResult，由 GestureExecutor 按 control_rate 调用。

    有 Converter 时：``current_angles`` / 默认 ``target_angles`` 为 S30 **semantic_rad**；
    0～255 仅出现在 hardware 边界（NamedPose / STREAM / Pose 落盘）。
    详见仓库根目录的 ``docs/workbench.md`` 与 ``angle_bridge.AngleUnits``。
    """

    def on_enter(self, current_angles: List[float]) -> None:
        """原语被激活时调用。记录起始角度用于插值。"""
        self._start_angles = list(current_angles)

    def on_exit(self) -> None:
        """原语被替换前调用。可用于资源清理。"""
        pass

    @abstractmethod
    def compute(
        self,
        current_angles: List[float],
        elapsed: float,
        ctx: PrimitiveContext,
    ) -> PrimitiveResult:
        """计算当前时刻的 20-DOF 目标角度。

        Args:
            current_angles: 手部当前状态反馈 (20-DOF, 0-255)
            elapsed: 自本原语激活以来经过的秒数
            ctx: 外部上下文 (TCP 位姿、物体位姿)

        Returns:
            PrimitiveResult — feasible=True 时 target_angles 有效
        """
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """原语名称，用于指令匹配和日志。"""
        ...

    @property
    def duration(self) -> Optional[float]:
        """原语持续时间 (秒)。None 表示持续执行直到被新指令替代。"""
        return None

    @property
    def done(self) -> bool:
        """原语是否已执行完成。"""
        return False

    def supports_progress(self) -> bool:
        """该原语是否支持 motion_progress 热更新。"""
        return False

    def set_progress(self, p: float) -> None:
        """段内 progress 热更新 [0,1]。默认无操作；支持 progress 的子类须 override。"""
        pass

    @property
    def progress_driven(self) -> bool:
        """是否处于外部 progress 细控（Executor 不以反馈收敛报 DONE）。"""
        return False

    @property
    def motion_status(self) -> str:
        """结构化运动状态，供 gesture_status 发布。默认空串。"""
        return ""

    # ------ 便利方法 ------

    def _hold(self, reason: str = "") -> PrimitiveResult:
        """返回 infeasible — executor 保持上帧目标值。"""
        return PrimitiveResult(feasible=False, hold_reason=reason)

    def trajectory_complete(self, result: PrimitiveResult, elapsed: float) -> bool:
        """Trajectory completion only; the caller must still confirm fresh feedback."""
        if not result.feasible or getattr(self, "progress_driven", False):
            return False
        if result.trajectory_finished is not None:
            return bool(result.trajectory_finished)
        if self.done or result.progress >= 1.0:
            return True
        from .gesture_engine import PhasedPoseEngine, RelaxGripEngine, StaticPoseEngine

        engine = getattr(self, "_engine", None)
        if isinstance(engine, PhasedPoseEngine):
            p = engine._params
            return elapsed >= p.phase1 + p.phase2 + max(p.phase3, 0) + p.phase4
        if isinstance(engine, RelaxGripEngine):
            return engine.done
        return isinstance(engine, StaticPoseEngine) and (
            elapsed >= engine._params.duration or getattr(self, "_frozen_pose", None) is not None
        )

    def _move(self, angles: List[float], *, space: str = "semantic_rad",
              trajectory_finished: Optional[bool] = None) -> PrimitiveResult:
        """返回 feasible + 目标角度（默认 semantic_rad；hardware/legacy 须显式声明）。

        ``semantic`` / ``semantic_rad`` 时自动规范到 S30；``hardware`` 直出不做投影。
        """
        if space == "hardware":
            return PrimitiveResult(
                feasible=True,
                target_angles=[float(v) for v in angles],
                space="hardware",
                trajectory_finished=trajectory_finished,
            )
        out_space = space if space in ("semantic", "semantic_rad") else "semantic_rad"
        return PrimitiveResult(
            feasible=True,
            target_angles=to_executor_semantic(angles),
            space=out_space,
            trajectory_finished=trajectory_finished,
        )


def lerp_angles(
    start: List[float],
    target: List[float],
    t: float,
) -> List[float]:
    """线性插值角度向量（单位无关）。t in [0, 1]。"""
    if len(start) != len(target):
        raise ValueError(f"lerp_angles length mismatch: start={len(start)}, target={len(target)}")
    t = max(0.0, min(1.0, t))
    result = [s + (g - s) * t for s, g in zip(start, target)]
    if len(result) == 20:
        for i in RESERVED_INDICES:
            result[i] = 0.0
    return result


# ---------------------------------------------------------------------------
# 欠驱动关节守卫（O6/L6：tip/rot 等为 -1，禁止 angles[-1] 误写）
# ---------------------------------------------------------------------------


def joint_present(idx: int) -> bool:
    """关节索引有效（O6/L6 的 tip/rot 等为 -1）。"""
    return idx is not None and int(idx) >= 0


def set_joint(angles: List[float], idx: int, value: float) -> None:
    """仅当关节存在时写入，避免 angles[-1] 误改末关节。"""
    if joint_present(idx) and idx < len(angles):
        angles[idx] = float(value)


def joint_delta(
    values: List[float],
    baseline: List[float],
    idx: int,
) -> float:
    """相对基线增量；关节不存在时返回 0（不单独触发接触）。"""
    if not joint_present(idx) or idx >= len(values) or idx >= len(baseline):
        return 0.0
    return float(values[idx]) - float(baseline[idx])


def bump_joint(
    angles: List[float],
    idx: int,
    step: float,
    max_val: float,
) -> bool:
    """存在则步进；返回 True 表示已达上限或不存在（可视为该关节闭合完成）。

    ``max_val`` 须与执行核量纲一致（semantic_rad 或 legacy255），由调用方传入
    ``AngleUnits.close_max(...)``，避免 rad 路径误用 255 上限。
    """
    if not joint_present(idx) or idx >= len(angles):
        return True
    if angles[idx] < max_val:
        angles[idx] = min(angles[idx] + step, max_val)
        return angles[idx] >= max_val
    return True


# 四指屈伸链字段名（mcp_flex → pip → dip）
_DIGIT_FLEX_CHAIN: Dict[str, tuple] = {
    "index": ("index_base", "index_pip", "index_tip"),
    "middle": ("middle_base", "middle_root2", "middle_tip"),
    "ring": ("ring_base", "ring_pip", "ring_tip"),
    "pinky": ("pinky_base", "pinky_pip", "pinky_tip"),
}


def bump_digit_flex_chain(
    angles: List[float],
    cfg: HandConfig,
    digit: str,
    step: float,
    max_val: float = 255.0,
) -> None:
    """步进单指屈伸链（含 pip，字段为 -1 时跳过）。"""
    fields = _DIGIT_FLEX_CHAIN.get(digit)
    if not fields:
        raise KeyError(f"未知手指: {digit!r}")
    for name in fields:
        bump_joint(angles, getattr(cfg, name), step, max_val)


def flex_chain_indices(cfg: HandConfig, digit: str) -> List[int]:
    """单指屈伸链 S30 下标（用于监测/冻结）。"""
    fields = _DIGIT_FLEX_CHAIN.get(digit)
    if not fields:
        raise KeyError(f"未知手指: {digit!r}")
    return present_joints(*(getattr(cfg, name) for name in fields))


def apply_four_finger_pip(angles: List[float], cfg: HandConfig, value: float) -> None:
    """预成型：四指 pip（有独立轴时）写入同一预备值。"""
    for attr in ("index_pip", "middle_root2", "ring_pip", "pinky_pip"):
        idx = getattr(cfg, attr, -1)
        if idx is not None and idx >= 0:
            angles[idx] = float(value)


def sync_pip_from_dip(angles: List[float]) -> List[float]:
    """将四指 dip 拷到 pip（O30 root2 跟随指尖弯曲；无独立轴的手由 mask 消化）。"""
    from .semantic_30 import SEMANTIC_30_IDX

    out = list(angles)
    for finger in ("index", "middle", "ring", "pinky"):
        out[SEMANTIC_30_IDX[f"{finger}_pip"]] = out[SEMANTIC_30_IDX[f"{finger}_dip"]]
    return out


def digit_flex_close_step(
    angles: List[float],
    cfg: HandConfig,
    digit: str,
    currents,
    baseline,
    hw_cfg,
    contact_delta: float,
    step: float,
    max_val: float = 255.0,
) -> str:
    """屈伸链力控一步。

    返回 ``\"contact\"``（电流超阈）、``\"done\"``（达上限）、``\"continue\"``（已 bump）。
    ``currents`` / ``baseline`` 为硬件序；经 ``hw_current_delta`` 按 S30 通道监测。
    """
    from .contact_resolver import hw_current_delta

    chain = flex_chain_indices(cfg, digit)
    if not chain:
        return "done"
    if any(hw_current_delta(currents, baseline, hw_cfg, idx) > contact_delta for idx in chain):
        return "contact"
    bump_digit_flex_chain(angles, cfg, digit, step, max_val)
    if all(angles[idx] >= max_val for idx in chain):
        return "done"
    return "continue"


def digit_flex_should_stop(
    angles: List[float],
    cfg: HandConfig,
    digit: str,
    currents,
    baseline,
    hw_cfg,
    contact_delta: float,
    step: float,
    max_val: float = 255.0,
) -> bool:
    """``digit_flex_close_step`` 的布尔封装（contact/done → True）。"""
    return (
        digit_flex_close_step(
            angles, cfg, digit, currents, baseline, hw_cfg, contact_delta, step, max_val
        )
        != "continue"
    )


def tip_or_base(tip_idx: int, base_idx: int) -> int:
    """优先 tip；无 tip 的手型（O6/L6）回退到 base。"""
    return tip_idx if joint_present(tip_idx) else base_idx


def present_joints(*idxs: int) -> List[int]:
    """过滤掉 -1 / 无效关节索引。"""
    return [int(i) for i in idxs if joint_present(i)]


# 清晰别名：避免与 ``hand_config.HandConfig``（硬件映射）混淆。
SemanticPoseConfig = HandConfig
