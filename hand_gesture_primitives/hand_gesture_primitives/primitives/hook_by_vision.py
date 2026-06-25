"""四指钩握原语 — 食指+中指+无名指+小指弯曲钩握，拇指不参与。

抓取类型: 0 vs 2-5 (无拇指，食指+中指+无名指+小指钩握)

适用场景: 提拎把手、挂钩、环形物体等，
手指弯曲形成钩状插入物体开口处，依靠手指自身弯曲力承托。

两阶段算法:
1. 预成型 (Pre-shape): 四指弯曲到中间位准备钩入，拇指伸直不参与
2. 力控闭合: 四指同时收紧钩住，各自独立检测接触后停止
"""

import logging
from enum import Enum
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL, HAND_CONFIGS, HandConfig,
)

_logger = logging.getLogger(__name__)

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# --- 预成型: 四指中等弯曲准备钩入 ---
_PRESHAPE = {
    "index_base": 160,
    "index_tip": 140,
    "middle_base": 160,
    "middle_tip": 140,
    "ring_base": 160,
    "ring_tip": 140,
    "pinky_base": 160,
    "pinky_tip": 140,
}

# --- 力控闭合参数 ---
FINGER_STEP = 8
PRESHAPE_DURATION = 0.5

def _build_finger_joints(cfg):
    return [
        (cfg.index_base, cfg.index_tip),
        (cfg.middle_base, cfg.middle_tip),
        (cfg.ring_base, cfg.ring_tip),
        (cfg.pinky_base, cfg.pinky_tip),
    ]


class _Phase(Enum):
    PRESHAPE = 0
    CLOSE = 1
    HOLD = 2


class HookByVision(HandGesturePrimitive):
    """四指钩握 (0 vs 2-5)。

    拇指不参与，食指+中指+无名指+小指弯曲形成钩状，
    适合提拎把手、挂钩、环形物体。
    四指同时力控闭合，各自独立检测接触。

    支持 O20 (20-DOF) 和 L25 (25-DOF) 两种手型。
    """

    def __init__(self):
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._finger_stopped: List[bool] = [False, False, False, False]
        self._cfg: Optional[HandConfig] = None

    @property
    def name(self) -> str:
        return "hook_by_vision"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._finger_stopped = [False, False, False, False]
        self._cfg = None

    def _get_cfg(self, ctx: PrimitiveContext) -> HandConfig:
        if self._cfg is None:
            self._cfg = HAND_CONFIGS[ctx.hand_type]
        return self._cfg

    def _build_preshape_target(self, cfg: HandConfig) -> List[float]:
        angles = [0.0] * cfg.num_joints
        # 拇指保持伸直，不参与抓取
        angles[cfg.thumb_base] = 0
        angles[cfg.thumb_abd] = ABD_NEUTRAL
        angles[cfg.thumb_rot] = 0
        angles[cfg.thumb_tip] = 0
        # 四指预成型弯曲
        angles[cfg.index_base] = _PRESHAPE["index_base"]
        angles[cfg.index_tip] = _PRESHAPE["index_tip"]
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        angles[cfg.middle_tip] = _PRESHAPE["middle_tip"]
        angles[cfg.ring_base] = _PRESHAPE["ring_base"]
        angles[cfg.ring_tip] = _PRESHAPE["ring_tip"]
        angles[cfg.pinky_base] = _PRESHAPE["pinky_base"]
        angles[cfg.pinky_tip] = _PRESHAPE["pinky_tip"]
        # L25 root2
        if cfg.middle_root2 >= 0:
            angles[cfg.middle_root2] = 60
        # 侧摆归中（收拢手指使钩更紧凑）
        for idx in cfg.abd_indices:
            if idx != cfg.thumb_abd:
                angles[idx] = cfg.abd_neutral
        return angles

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        cfg = self._get_cfg(ctx)
        finger_joints = _build_finger_joints(cfg)

        if ctx.tcp_pose is None:
            return self._hold("缺少 tcp_pose")

        # 无 object_pose 时用默认值，跳过可达性检查
        # (auto_grasp_node 等外部节点已完成抓取判定)
        if ctx.object_pose is not None:
            tcp_to_obj = ctx.object_pose.position - ctx.tcp_pose.position
            tcp_dist = float(np.linalg.norm(tcp_to_obj))

            tcp_quat = ctx.tcp_pose.orientation
            if not np.allclose(tcp_quat, [0, 0, 0, 1]):
                R_tcp = Rotation.from_quat(tcp_quat).as_matrix()
                palm_forward = R_tcp[:, 2]
                forward_dist = float(np.dot(tcp_to_obj, palm_forward))
                if forward_dist < PALM_FORWARD_MIN or forward_dist > PALM_FORWARD_MAX:
                    return self._hold("物体不在掌心前方 2–15cm")
            else:
                if tcp_dist > REACH_THRESHOLD:
                    return self._hold("TCP到物体距离过远")

        # === Phase 1: 预成型 ===
        if self._phase == _Phase.PRESHAPE:
            target = self._build_preshape_target(cfg)
            t = elapsed / PRESHAPE_DURATION
            if t >= 1.0:
                self._phase = _Phase.CLOSE
                self._close_angles = list(target)
                self._baseline_currents = list(ctx.joint_currents)
                _logger.warning("hook: 进入钩握闭合阶段")
                return self._move(list(target))
            return self._move(lerp_angles(self._start_angles, target, t))

        # === Phase 2: 力控闭合 ===
        if self._phase == _Phase.CLOSE:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles))

            base = self._baseline_currents

            # 四指各自独立力控
            for i, (base_idx, tip_idx) in enumerate(finger_joints):
                if self._finger_stopped[i]:
                    continue
                base_delta = currents[base_idx] - base[base_idx]
                tip_delta = currents[tip_idx] - base[tip_idx]
                if base_delta > ctx.contact_thresholds.current_delta_narrow or tip_delta > ctx.contact_thresholds.current_delta_narrow:
                    self._finger_stopped[i] = True
                    finger_names = ["食指", "中指", "无名指", "小指"]
                    _logger.warning("hook: %s接触!", finger_names[i])
                else:
                    if self._close_angles[base_idx] < 255:
                        self._close_angles[base_idx] = min(
                            self._close_angles[base_idx] + FINGER_STEP, 255)
                    else:
                        self._finger_stopped[i] = True
                    if self._close_angles[tip_idx] < 255:
                        self._close_angles[tip_idx] = min(
                            self._close_angles[tip_idx] + FINGER_STEP, 255)

            # L25: 中指 root2 同步
            if cfg.middle_root2 >= 0 and not self._finger_stopped[1]:
                if self._close_angles[cfg.middle_root2] < 255:
                    self._close_angles[cfg.middle_root2] = min(
                        self._close_angles[cfg.middle_root2] + FINGER_STEP, 255)

            # 所有手指都停 → HOLD
            if all(self._finger_stopped):
                _logger.warning("hook: 四指钩握完成 → HOLD")
                self._phase = _Phase.HOLD

            return self._move(list(self._close_angles))

        # === Phase 3: 保持 ===
        currents = ctx.joint_currents
        monitored = []
        for base_idx, tip_idx in finger_joints:
            monitored.extend([base_idx, tip_idx])
        if cfg.middle_root2 >= 0:
            monitored.append(cfg.middle_root2)
        for idx in monitored:
            if currents[idx] > ctx.contact_thresholds.hold_safe_current and self._close_angles[idx] > 0:
                over = currents[idx] - ctx.contact_thresholds.hold_safe_current
                step = max(1, int(over / 30))
                self._close_angles[idx] = max(0, self._close_angles[idx] - step)
        return self._move(list(self._close_angles))

    @property
    def done(self) -> bool:
        return False
