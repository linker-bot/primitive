"""手势原语基类。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .contact_config import ContactThresholds

import numpy as np


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
    "thumb_base", "index_base", "middle_base", "ring_base", "pinky_base",
    "thumb_abd", "index_abd", "middle_abd", "ring_abd", "pinky_abd",
    "thumb_rot", "rsv_11", "rsv_12", "rsv_13", "rsv_14",
    "thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip",
]

# 预留位索引 (这些位恒为 0)
RESERVED_INDICES = [11, 12, 13, 14]

# 侧摆关节索引 (这些关节 128 = 中立位)
ABD_INDICES = [6, 7, 8, 9]
ABD_NEUTRAL = 128

NUM_JOINTS = 20

# 侧向夹持原语分阶段控制: prep → (arm 就位) → close
GRASP_PHASES = frozenset({"full", "prep", "close"})
PHASED_GRASP_PRIMITIVES = frozenset({
    "thumb_adduction_grip",
    "index_middle_adduction_grip",
})


def parse_grasp_phase(phase: str) -> str:
    """解析 grasp phase 参数，非法值抛出 ValueError。"""
    p = (phase or "full").lower().strip()
    if p not in GRASP_PHASES:
        raise ValueError(f"grasp phase 必须为 full/prep/close，收到: {phase!r}")
    return p


# ---------------------------------------------------------------------------
# 手型配置 (O20 / L25)
# ---------------------------------------------------------------------------

@dataclass
class HandConfig:
    """手型关节映射配置。

    每个字段为该逻辑关节在位置命令数组中的索引。
    num_joints: 该手型的总 DOF 数。
    """
    num_joints: int
    # 拇指
    thumb_base: int       # 拇指根部弯曲
    thumb_abd: int        # 拇指侧摆/偏航
    thumb_rot: int        # 拇指旋转/侧滚
    thumb_tip: int        # 拇指指尖弯曲
    thumb_root2: int      # 拇指第二根部弯曲 (L25独有, O20=-1表示无)
    # 食指
    index_base: int       # 食指根部弯曲
    index_tip: int        # 食指指尖弯曲
    index_abd: int        # 食指侧摆
    # 中指
    middle_base: int      # 中指根部弯曲
    middle_tip: int       # 中指指尖弯曲
    middle_root2: int     # 中指第二根部弯曲 (L25独有, O20=-1表示无)
    # 无名指
    ring_base: int        # 无名指根部弯曲
    ring_tip: int         # 无名指指尖弯曲
    # 小指
    pinky_base: int       # 小指根部弯曲
    pinky_tip: int        # 小指指尖弯曲
    # 侧摆中立值
    abd_neutral: int
    # 侧摆关节列表
    abd_indices: List[int] = field(default_factory=list)
    # 预留位 (恒为0)
    reserved_indices: List[int] = field(default_factory=list)
    # L25 角度方向与 O20 相反: 0=弯曲, 255=张开
    invert_angles: bool = False


HAND_CONFIGS: Dict[str, HandConfig] = {
    "o20": HandConfig(
        num_joints=20,
        thumb_base=0,     thumb_abd=5,     thumb_rot=10,   thumb_tip=15,
        thumb_root2=-1,
        index_base=1,     index_tip=16,    index_abd=6,
        middle_base=2,    middle_tip=17,   middle_root2=-1,
        ring_base=3,      ring_tip=18,
        pinky_base=4,     pinky_tip=19,
        abd_neutral=128,
        abd_indices=[6, 7, 8, 9],
        reserved_indices=[11, 12, 13, 14],
    ),
    "l25": HandConfig(
        num_joints=25,
        thumb_base=0,     thumb_abd=5,     thumb_rot=10,   thumb_tip=20,
        thumb_root2=15,
        index_base=1,     index_tip=21,    index_abd=6,
        middle_base=2,    middle_tip=22,   middle_root2=17,
        ring_base=3,      ring_tip=23,
        pinky_base=4,     pinky_tip=24,
        abd_neutral=128,
        abd_indices=[5, 6, 7, 8, 9],
        reserved_indices=[],
        invert_angles=True,
    ),
    "o6": HandConfig(
        num_joints=6,
        thumb_base=0,     thumb_abd=1,     thumb_rot=-1,   thumb_tip=-1,
        thumb_root2=-1,
        index_base=2,     index_tip=-1,    index_abd=-1,
        middle_base=3,    middle_tip=-1,   middle_root2=-1,
        ring_base=4,      ring_tip=-1,
        pinky_base=5,     pinky_tip=-1,
        abd_neutral=128,
        abd_indices=[],
        reserved_indices=[],
        invert_angles=False,
    ),
}


# ---------------------------------------------------------------------------
# 外部输入数据结构
# ---------------------------------------------------------------------------

@dataclass
class PoseStamped:
    """简化的位姿表示 (从 geometry_msgs/PoseStamped 缓存)。"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0., 1.]))
    frame_id: str = ''
    stamp_sec: float = 0.0


@dataclass
class PrimitiveContext:
    """每帧传入原语的外部上下文。

    tcp_pose 和 object_pose 均为 world frame 下的位姿。
    原语自行计算两者的相对关系来判断可行性。
    joint_currents: 各关节当前电流 (mA)，按 position order 排列。
    hand_type: 手型标识 ("o20" 或 "l25")。
    """
    tcp_pose: Optional[PoseStamped] = None
    object_pose: Optional[PoseStamped] = None
    object_label: str = ""
    joint_currents: List[float] = field(default_factory=lambda: [0.0] * 20)
    hand_type: str = "o20"
    # 感知输出: 目标物体几何信息
    object_size: Optional[np.ndarray] = None          # [sx, sy, sz] meters
    object_orientation: Optional[np.ndarray] = None   # quaternion [x, y, z, w]
    grasp_type: str = ''                              # "precision"/"lateral"/"power"/""
    # 触觉反馈
    tactile_pressure: Optional[np.ndarray] = None     # [thumb, index, middle, ring, pinky]
    tactile_mode: str = "none"                        # none | pressure | mass
    contact_detected: bool = False                    # 任一指检测到接触
    grasp_stable: bool = False                        # 抓取稳定 (压力平稳无滑移)
    # 力矩反馈
    joint_torque: Optional[np.ndarray] = None         # 20-DOF 各关节力矩 mNm
    external_force: Optional[np.ndarray] = None       # 末端外力估计 [fx, fy, fz] N
    torque_overload: bool = False                     # 力矩超限标志
    # FK 指尖位置
    fingertip_positions: Optional[np.ndarray] = None  # [5,3] xyz (hand_base_link 系, 米)
    contact_thresholds: ContactThresholds = field(default_factory=ContactThresholds)


@dataclass
class PrimitiveResult:
    """原语单帧计算结果。"""
    feasible: bool
    target_angles: List[float] = field(default_factory=list)
    hold_reason: str = ""


# ---------------------------------------------------------------------------
# 原语基类
# ---------------------------------------------------------------------------

class HandGesturePrimitive(ABC):
    """手势原语基类。

    子类实现 compute() 返回 PrimitiveResult，由 GestureExecutor 以 10Hz 调用。
    角度值范围 0-255 (uint8 映射到各电机角度范围)。
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

    # ------ 便利方法 ------

    def _hold(self, reason: str = "") -> PrimitiveResult:
        """返回 infeasible — executor 保持上帧目标值。"""
        return PrimitiveResult(feasible=False, hold_reason=reason)

    def _move(self, angles: List[float]) -> PrimitiveResult:
        """返回 feasible + 目标角度。"""
        return PrimitiveResult(feasible=True, target_angles=angles)


def lerp_angles(
    start: List[float],
    target: List[float],
    t: float,
) -> List[float]:
    """线性插值 0-255 范围的角度。t in [0, 1]。"""
    t = max(0.0, min(1.0, t))
    result = [s + (g - s) * t for s, g in zip(start, target)]
    # O20 预留位归零
    if len(result) == 20:
        for i in RESERVED_INDICES:
            result[i] = 0.0
    return result
