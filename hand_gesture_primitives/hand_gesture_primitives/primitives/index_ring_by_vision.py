"""食指环形包络原语 — 拇指与食指形成环状包络，其余三指张开保持。

支持根据感知物体尺寸自适应调整捏合口径，目标可随感知更新平滑过渡。
触觉闭环：到达目标后继续缓慢闭合直到接触检测；接触后冻结防止过力。
"""

from typing import List, Optional

import numpy as np

from ..angle_bridge import AngleUnits
from ..contact_detection import FingerContactTracker
from ..gesture_engine import PhasedLerpEngine, make_index_ring_by_vision_engine
from ..gesture_params import load_index_ring_by_vision_params
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
    sync_pip_from_dip,
)
from ..semantic_30 import SEMANTIC_30_IDX, l25_compat_remap_s30, s30_pose_from_channels

_I = SEMANTIC_30_IDX

# 默认目标角度 (无感知数据时的 fallback) — S30 通道
INDEX_RING_ANGLES = s30_pose_from_channels(
    {
        "thumb_mcp_flex": 200,
        "index_mcp_flex": 150,
        "middle_mcp_flex": 0,
        "ring_mcp_flex": 0,
        "pinky_mcp_flex": 0,
        "thumb_mcp_abd": 155,
        "thumb_rot": 255,
        "thumb_dip": 190,
        "index_dip": 230,
        "middle_dip": 0,
        "ring_dip": 0,
        "pinky_dip": 0,
    },
)

INDEX_RING_ANGLES_L25 = l25_compat_remap_s30(INDEX_RING_ANGLES)

# 阶段保持 / 渐进闭合用的通道
_HOLD_P1 = (
    _I["thumb_mcp_flex"],
    _I["index_mcp_flex"],
    _I["thumb_mcp_abd"],
    _I["thumb_rot"],
    _I["thumb_dip"],
    _I["index_dip"],
)
_PROGRESSIVE = (
    _I["thumb_mcp_flex"],
    _I["index_mcp_flex"],
    _I["thumb_dip"],
    _I["index_dip"],
)

# 先张开三指，再拇食合环
PHASE1_DURATION = 0.4  # 三指先张开
PHASE2_DURATION = 0.5  # 拇指食指合环到达初始目标

# 目标切换时平滑过渡时长
RETARGET_BLEND = 0.25

# 尺寸变化阈值 (m)，低于此认为无显著变化
SIZE_CHANGE_THRESHOLD = 0.002

# 渐进闭合参数 (到达目标后无接触时继续缓慢收紧)
PROGRESSIVE_CLOSE_RATE = 30.0  # 每秒增加的闭合量 (约3/tick @10Hz)
PROGRESSIVE_CLOSE_MAX = 250.0  # 闭合安全上限


def _adaptive_index_ring_by_vision_angles(object_size, base=None):
    """根据物体截面直径调整捏合闭合量。

    object_size: [sx, sy, sz] meters
    返回适配后的 S30 目标角度列表。

    物体越大 → 拇食指需要更大弯曲才能环绕包住。
    范围: d=5mm (小螺丝，轻捏) ~ d=30mm (粗柄，深握)
    """
    angles = list(base if base is not None else INDEX_RING_ANGLES)
    s = sorted(object_size)
    d_mm = (s[0] + s[1]) / 2.0 * 1000.0  # 截面直径 mm

    i_tb, i_ib = _I["thumb_mcp_flex"], _I["index_mcp_flex"]
    i_tt, i_it = _I["thumb_dip"], _I["index_dip"]

    # d=5mm → 轻捏(小弯曲); d=30mm → 深握(大弯曲)
    # L25 base 传入时以 base 值为中心，允许 ±40 范围
    if base is not None:
        b0, b1, b15, b16 = angles[i_tb], angles[i_ib], angles[i_tt], angles[i_it]
        angles[i_tb] = int(np.clip(b0 - 20 + (d_mm - 5) * 2.4, b0 - 20, b0 + 40))
        angles[i_ib] = int(np.clip(b1 - 30 + (d_mm - 5) * 2.8, b1 - 30, b1 + 40))
        angles[i_tt] = int(np.clip(b15 - 35 + (d_mm - 5) * 2.0, b15 - 35, b15 + 25))
        angles[i_it] = int(np.clip(b16 - 30 + (d_mm - 5) * 2.0, b16 - 30, b16 + 40))
    else:
        angles[i_tb] = int(np.clip(90 + (d_mm - 5) * 2.4, 90, 150))
        angles[i_ib] = int(np.clip(100 + (d_mm - 5) * 2.8, 100, 170))
        angles[i_tt] = int(np.clip(140 + (d_mm - 5) * 2.0, 140, 200))
        angles[i_it] = int(np.clip(80 + (d_mm - 5) * 2.0, 80, 150))
    return sync_pip_from_dip(angles)


class IndexRingByVision(HandGesturePrimitive):
    """拇指与食指环形包络，其余手指张开不动。

    三阶段策略:
      P1: 三指张开
      P2: 拇食指闭合到视觉目标
      P3: 若无触觉接触，继续渐进闭合直到接触或安全上限
    触觉接触后冻结，滑移时缓慢恢复。
    """

    def __init__(self):
        self._engine: Optional[PhasedLerpEngine] = None
        self._engine_hand_type: str = ""
        self._grasp_state: str = "approaching"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._engine_hand_type = ""
        self._grasp_state = "approaching"
        self._target: List[float] = sync_pip_from_dip(INDEX_RING_ANGLES)
        self._target_ready = False
        self._last_size: Optional[np.ndarray] = None
        # 平滑过渡状态
        self._blend_from: Optional[List[float]] = None
        self._blend_start_elapsed: float = 0.0
        # 触觉闭环状态
        self._frozen_pose: Optional[List[float]] = None
        # 渐进闭合状态
        self._progressive_angles: Optional[List[float]] = None
        self._last_elapsed: float = 0.0
        self._size_locked: bool = False
        self._contact = FingerContactTracker()

    def _update_target(
        self, ctx: PrimitiveContext, elapsed: float, current_output: List[float]
    ) -> None:
        """每帧评估感知数据，必要时更新目标并启动平滑过渡。"""
        if self._size_locked:
            return
        if ctx.object_size is None:
            return
        if self._last_size is not None:
            if np.linalg.norm(ctx.object_size - self._last_size) < SIZE_CHANGE_THRESHOLD:
                return
        self._last_size = ctx.object_size.copy()
        base = getattr(self, "_l25_base", None)
        new_target = _adaptive_index_ring_by_vision_angles(ctx.object_size, base=base)
        new_target = AngleUnits(ctx).pose(new_target)
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        """按阶段逻辑计算原始输出 (不含 blend)。"""
        target = self._target

        # 阶段1目标: 三指张开，拇食指保持原位
        phase1_target = list(target)
        for i in _HOLD_P1:
            phase1_target[i] = self._start_angles[i]

        t1_end = PHASE1_DURATION
        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return lerp_angles(self._start_angles, phase1_target, t)
        t = min(1.0, (elapsed - t1_end) / PHASE2_DURATION)
        return lerp_angles(phase1_target, target, t)

    def _check_contact(self, ctx: PrimitiveContext) -> bool:
        """拇指或食指接触：触觉优先，无传感器时用电流增量。"""
        return self._contact.check(ctx, (0, 1))

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        if load_index_ring_by_vision_params(ctx.hand_type) is not None:
            engine = self._ensure_engine(ctx)
            raw = engine.compute(current_angles, elapsed, ctx)
            return self._move(raw, space=u.space)

        if not getattr(self, "_target_ready", False):
            raw = INDEX_RING_ANGLES_L25 if ctx.hand_type == "l25" else INDEX_RING_ANGLES
            self._target = u.pose(sync_pip_from_dip(raw))
            if ctx.hand_type == "l25":
                self._l25_base = list(raw)
            self._target_ready = True
        raw = self._compute_raw(elapsed)
        self._update_target(ctx, elapsed, raw)
        raw = self._compute_raw(elapsed)

        # 平滑过渡
        if self._blend_from is not None:
            dt = elapsed - self._blend_start_elapsed
            if dt < RETARGET_BLEND:
                t = dt / RETARGET_BLEND
                raw = lerp_angles(self._blend_from, raw, t)
            else:
                self._blend_from = None

        # 触觉闭环: 仅在 phase2+ (拇食指闭合阶段) 生效
        if elapsed > PHASE1_DURATION:
            if not self._size_locked and elapsed >= (PHASE1_DURATION + PHASE2_DURATION):
                self._size_locked = True
                self._contact.begin_closing(ctx)

            # 已冻结 → 永远保持，等下一个指令
            if self._frozen_pose is not None:
                self._grasp_state = "contact"
                return self._move(self._frozen_pose, space=u.space)

            has_contact = self._check_contact(ctx)
            if has_contact:
                # 接触瞬间: 冻结实际输出姿态 (渐进模式下用渐进值，否则用 raw)
                freeze = self._progressive_angles if self._progressive_angles is not None else raw
                self._frozen_pose = list(freeze)
                self._grasp_state = "contact"
                return self._move(self._frozen_pose, space=u.space)

            # 无接触 + phase2 已到达目标 → 渐进闭合
            phase2_done = elapsed >= (PHASE1_DURATION + PHASE2_DURATION)
            if phase2_done:
                self._grasp_state = "progressive"
                dt = elapsed - self._last_elapsed
                self._last_elapsed = elapsed
                if self._progressive_angles is None:
                    self._progressive_angles = list(raw)
                for i in _PROGRESSIVE:
                    inc = u.step(i, PROGRESSIVE_CLOSE_RATE) * dt
                    self._progressive_angles[i] = min(
                        u.close_max(i, PROGRESSIVE_CLOSE_MAX),
                        self._progressive_angles[i] + inc,
                    )
                # O30 独立 PIP：渐进抬 dip 后同步 pip，避免 tip/root2 分叉
                self._progressive_angles = sync_pip_from_dip(self._progressive_angles)
                return self._move(list(self._progressive_angles), space=u.space)

        self._grasp_state = "approaching"
        self._last_elapsed = elapsed
        return self._move(raw, space=u.space)

    @property
    def grasp_state(self) -> str:
        if self._engine is not None:
            return self._engine.grasp_state
        return self._grasp_state

    def _ensure_engine(self, ctx: PrimitiveContext) -> PhasedLerpEngine:
        if self._engine is None or self._engine_hand_type != ctx.hand_type:
            self._engine = make_index_ring_by_vision_engine(
                ctx.hand_type, radian_converter=ctx.radian_converter
            )
            if self._engine is None:
                raise RuntimeError(
                    f"index_ring_by_vision config engine missing for {ctx.hand_type}"
                )
            self._engine.reset(self._start_angles)
            self._engine_hand_type = ctx.hand_type
        return self._engine

    @property
    def name(self) -> str:
        return "index_ring_by_vision"

    @property
    def done(self) -> bool:
        return False
