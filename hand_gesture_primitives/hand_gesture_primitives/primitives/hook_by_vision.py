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

from ..angle_bridge import AngleUnits
from ..contact_config import scale_per_tick_delta, scaled_hold_retreat_step
from ..contact_resolver import current_index_for_s30
from ..gesture_engine import (
    PhasedLerpEngine,
    SequentialForceCloseEngine,
    make_hook_by_vision_engine,
    make_hook_by_vision_sfce_engine,
)
from ..gesture_params import load_hook_by_vision_params
from ..hand_config import HandConfig as HardwareHandConfig
from ..primitive_base import (
    ABD_NEUTRAL,
    SEMANTIC_DIM,
    HandConfig,
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    apply_four_finger_pip,
    digit_flex_close_step,
    flex_chain_indices,
    lerp_angles,
    semantic_pose_config,
    set_joint,
)

_logger = logging.getLogger(__name__)

_CLOSE_DIGITS = ("index", "middle", "ring", "pinky")
_CLOSE_DIGIT_LABELS = ("食指", "中指", "无名指", "小指")

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
        self._engine: Optional[PhasedLerpEngine] = None
        self._sfce_engine: Optional[SequentialForceCloseEngine] = None
        self._engine_hand_type: str = ""
        self._phase = _Phase.PRESHAPE
        self._close_angles: List[float] = []
        self._baseline_currents: List[float] = []
        self._settle_count = 0
        self._finger_stopped: List[bool] = [False, False, False, False]
        self._cfg: Optional[HandConfig] = None
        self._grasp_state: str = "approaching"

    @property
    def name(self) -> str:
        return "hook_by_vision"

    @property
    def grasp_state(self) -> str:
        if self._engine is not None:
            return self._engine.grasp_state
        if self._sfce_engine is not None:
            return self._sfce_engine.grasp_state
        return self._grasp_state

    def _ensure_engine(self, ctx: PrimitiveContext) -> PhasedLerpEngine:
        if self._engine is None or self._engine_hand_type != ctx.hand_type:
            self._engine = make_hook_by_vision_engine(
                ctx.hand_type, radian_converter=ctx.radian_converter
            )
            if self._engine is None:
                raise RuntimeError(f"hook_by_vision config engine missing for {ctx.hand_type}")
            self._engine.reset(self._start_angles)
            self._engine_hand_type = ctx.hand_type
        return self._engine

    def _ensure_sfce_engine(self, ctx: PrimitiveContext) -> Optional[SequentialForceCloseEngine]:
        if self._sfce_engine is None or self._engine_hand_type != ctx.hand_type:
            self._sfce_engine = make_hook_by_vision_sfce_engine(
                ctx.hand_type, radian_converter=ctx.radian_converter
            )
            if self._sfce_engine is not None:
                self._sfce_engine.reset(self._start_angles)
                self._engine_hand_type = ctx.hand_type
        return self._sfce_engine

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._sfce_engine = None
        self._engine_hand_type = ""
        self._phase = _Phase.PRESHAPE
        self._baseline_currents = [0.0] * len(current_angles)
        self._settle_count = 0
        self._finger_stopped = [False, False, False, False]
        self._cfg = None
        self._hw_cfg = None
        self._grasp_state = "approaching"

    def _get_cfg(self, ctx: PrimitiveContext) -> HandConfig:
        if self._cfg is None:
            self._cfg = semantic_pose_config(ctx.hand_type)
            self._hw_cfg = HardwareHandConfig(ctx.hand_type)
        return self._cfg

    def _build_preshape_target(self, cfg: HandConfig) -> List[float]:
        angles = [0.0] * SEMANTIC_DIM
        # 拇指保持伸直，不参与抓取
        angles[cfg.thumb_base] = 0
        angles[cfg.thumb_abd] = ABD_NEUTRAL
        set_joint(angles, cfg.thumb_rot, 0)
        set_joint(angles, cfg.thumb_tip, 0)
        # 四指预成型弯曲
        angles[cfg.index_base] = _PRESHAPE["index_base"]
        set_joint(angles, cfg.index_tip, _PRESHAPE["index_tip"])
        angles[cfg.middle_base] = _PRESHAPE["middle_base"]
        set_joint(angles, cfg.middle_tip, _PRESHAPE["middle_tip"])
        angles[cfg.ring_base] = _PRESHAPE["ring_base"]
        set_joint(angles, cfg.ring_tip, _PRESHAPE["ring_tip"])
        angles[cfg.pinky_base] = _PRESHAPE["pinky_base"]
        set_joint(angles, cfg.pinky_tip, _PRESHAPE["pinky_tip"])
        # L25 root2
        apply_four_finger_pip(angles, cfg, 60)
        # 侧摆归中（收拢手指使钩更紧凑）
        for idx in cfg.abd_indices:
            if idx != cfg.thumb_abd:
                angles[idx] = cfg.abd_neutral
        return angles

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        finger_step = scale_per_tick_delta(FINGER_STEP, ctx.motion_control_rate)
        if load_hook_by_vision_params(ctx.hand_type) is not None:
            engine = self._ensure_engine(ctx)
            raw = engine.compute(current_angles, elapsed, ctx)
            return self._move(raw, space=u.space)

        sfce = self._ensure_sfce_engine(ctx)
        if sfce is not None:
            raw = sfce.compute(elapsed, ctx)
            return self._move(raw, space=u.space)

        cfg = self._get_cfg(ctx)

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
            # fallback 常量为 legacy 255，须经 u.pose 转到与 start_angles 同空间再 lerp
            target = u.pose(self._build_preshape_target(cfg))
            t = elapsed / PRESHAPE_DURATION
            if t >= 1.0:
                self._phase = _Phase.CLOSE
                self._close_angles = list(target)
                self._baseline_currents = list(ctx.joint_currents)
                _logger.warning("hook: 进入钩握闭合阶段")
                return self._move(list(target), space=u.space)
            return self._move(lerp_angles(self._start_angles, target, t), space=u.space)

        # === Phase 2: 力控闭合 ===
        if self._phase == _Phase.CLOSE:
            currents = ctx.joint_currents

            if self._settle_count < ctx.contact_thresholds.current_settle_frames:
                self._settle_count += 1
                if self._settle_count == ctx.contact_thresholds.current_settle_frames:
                    self._baseline_currents = list(currents)
                return self._move(list(self._close_angles), space=u.space)

            base = self._baseline_currents

            # 四指各自独立力控
            for i, digit in enumerate(_CLOSE_DIGITS):
                if self._finger_stopped[i]:
                    continue
                status = digit_flex_close_step(
                    self._close_angles,
                    cfg,
                    digit,
                    currents,
                    base,
                    self._hw_cfg,
                    ctx.contact_thresholds.current_delta_narrow,
                    u.step(getattr(cfg, f"{digit}_base"), finger_step),
                    u.close_max(getattr(cfg, f"{digit}_base")),
                )
                if status != "continue":
                    self._finger_stopped[i] = True
                    if status == "contact":
                        _logger.warning("hook: %s接触!", _CLOSE_DIGIT_LABELS[i])

            # 所有手指都停 → HOLD
            if all(self._finger_stopped):
                _logger.warning("hook: 四指钩握完成 → HOLD")
                self._phase = _Phase.HOLD
                self._grasp_state = "contact"
            else:
                self._grasp_state = "closing"

            return self._move(list(self._close_angles), space=u.space)

        # === Phase 3: 保持 ===
        self._grasp_state = "contact"
        currents = ctx.joint_currents
        monitored: List[int] = []
        for digit in _CLOSE_DIGITS:
            monitored.extend(flex_chain_indices(cfg, digit))
        for idx in monitored:
            hi = current_index_for_s30(self._hw_cfg, idx)
            if hi is None or hi < 0 or hi >= len(currents):
                continue
            if (
                currents[hi] > ctx.contact_thresholds.hold_safe_current
                and self._close_angles[idx] > 0
            ):
                over = currents[hi] - ctx.contact_thresholds.hold_safe_current
                step = u.step(idx, scaled_hold_retreat_step(over, ctx.control_rate))
                self._close_angles[idx] = max(u.value(idx, 0), self._close_angles[idx] - step)
        return self._move(list(self._close_angles), space=u.space)

    @property
    def done(self) -> bool:
        return False
