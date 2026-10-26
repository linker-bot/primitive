"""大物体包络原语 — 五指适度弯曲包络较大直径物体。

支持根据感知物体尺寸自适应调整包络弯曲量，目标可随感知更新平滑过渡。
触觉闭环：渐进闭合直到接触检测 → 冻结姿态保持，等待下一个指令。
"""

from typing import List, Optional

import numpy as np

from ..angle_bridge import AngleUnits
from ..contact_detection import FingerContactTracker
from ..gesture_engine import PhasedLerpEngine, make_large_wrap_by_vision_engine
from ..gesture_params import load_large_wrap_by_vision_params
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
    sync_pip_from_dip,
)
from ..semantic_30 import SEMANTIC_30_IDX, l25_compat_remap_s30, s30_pose_from_channels

_I = SEMANTIC_30_IDX

# 默认目标角度 (无感知数据时的 fallback) — S30
LARGE_WRAP_ANGLES = s30_pose_from_channels(
    {
        "thumb_mcp_flex": 200,
        "index_mcp_flex": 220,
        "middle_mcp_flex": 220,
        "ring_mcp_flex": 220,
        "pinky_mcp_flex": 220,
        "thumb_mcp_abd": 175,
        "thumb_rot": 210,
        "thumb_dip": 200,
        "index_dip": 210,
        "middle_dip": 210,
        "ring_dip": 210,
        "pinky_dip": 210,
    },
)

LARGE_WRAP_ANGLES_L25 = l25_compat_remap_s30(LARGE_WRAP_ANGLES)

_THUMB_ALL = (
    _I["thumb_mcp_flex"],
    _I["thumb_mcp_abd"],
    _I["thumb_rot"],
    _I["thumb_dip"],
)
_THUMB_ABD = _I["thumb_mcp_abd"]
_THUMB_ROT = _I["thumb_rot"]

_FINGER_MCP = (
    _I["index_mcp_flex"],
    _I["middle_mcp_flex"],
    _I["ring_mcp_flex"],
    _I["pinky_mcp_flex"],
)
_FINGER_DIP = (
    _I["index_dip"],
    _I["middle_dip"],
    _I["ring_dip"],
    _I["pinky_dip"],
)

PHASE1_DURATION = 0.5  # 四指先包络
PHASE2_DURATION = 0.35  # 拇指侧摆
PHASE3_DURATION = 0.35  # 拇指旋转
PHASE4_DURATION = 0.35  # 拇指弯曲闭合

# 目标切换时平滑过渡时长
RETARGET_BLEND = 0.25

# 尺寸变化阈值 (m)
SIZE_CHANGE_THRESHOLD = 0.003

# 渐进闭合参数 (到达目标后无接触时继续缓慢收紧)
PROGRESSIVE_CLOSE_RATE = 25.0  # 每秒增加的闭合量 (约2.5/tick @10Hz)
PROGRESSIVE_CLOSE_MAX = 250.0  # 闭合安全上限
POSITION_HOLD_TIMEOUT = 0.4  # 有 vision 尺寸时 position_hold 最长等待 (s)


def _adaptive_large_wrap_by_vision_angles(object_size, base=None):
    """根据物体最大截面直径调整包络弯曲量。

    object_size: [sx, sy, sz] meters
    返回适配后的 S30 目标角度列表。
    """
    angles = list(base if base is not None else LARGE_WRAP_ANGLES)
    # 取 XY 平面最大截面直径 (排除高度轴)
    s = sorted(object_size)
    D_mm = float(max(s[0], s[1])) * 1000.0

    # The legacy L25-to-O20 mapping closes farther than the canonical O20
    # geometry for compact cylindrical parts.  Calibrate that model mapping
    # in its medium-size band so the computed aperture holds the sidewall
    # instead of slowly extruding the part into the palm.
    if base is not None and 55.0 <= D_mm <= 70.0:
        D_mm += 10.0

    # D=40mm → base≈230(紧握); D=100mm → base≈150(半开)
    base_val = int(np.clip(250 - (D_mm - 30) * 1.4, 130, 240))
    tip_val = int(np.clip(240 - (D_mm - 30) * 1.2, 120, 230))
    thumb_base = int(np.clip(220 - (D_mm - 30) * 0.8, 100, 220))
    thumb_tip = int(np.clip(tip_val - 10, 100, 220))

    for i in _FINGER_MCP:
        angles[i] = base_val
    for i in _FINGER_DIP:
        angles[i] = tip_val
    angles[_I["thumb_mcp_flex"]] = thumb_base
    angles[_I["thumb_dip"]] = thumb_tip
    return sync_pip_from_dip(angles)


class LargeWrapByVision(HandGesturePrimitive):
    """五指抓握包络大直径物体（分阶段避免拇指碰撞）。

    根据 ctx.object_size 自适应调整包络弯曲量；无感知数据时使用默认值。
    感知数据变化时平滑过渡到新目标，避免关节跳变。
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
        self._target: List[float] = sync_pip_from_dip(LARGE_WRAP_ANGLES)
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
        self._position_hold_started_at: Optional[float] = None
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
        new_target = _adaptive_large_wrap_by_vision_angles(ctx.object_size, base=base)
        new_target = AngleUnits(ctx).pose(new_target)
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        """按阶段逻辑计算原始输出 (不含 blend)。"""
        target = self._target

        # 阶段1: 四指到位, 拇指保持起始
        phase1_target = list(target)
        for i in _THUMB_ALL:
            phase1_target[i] = self._start_angles[i]
        # 阶段2: 拇指侧摆到位
        phase2_target = list(phase1_target)
        phase2_target[_THUMB_ABD] = target[_THUMB_ABD]
        # 阶段3: 拇指旋转到位
        phase3_target = list(phase2_target)
        phase3_target[_THUMB_ROT] = target[_THUMB_ROT]

        t1_end = PHASE1_DURATION
        t2_end = t1_end + PHASE2_DURATION
        t3_end = t2_end + PHASE3_DURATION
        t4_end = t3_end + PHASE4_DURATION

        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return lerp_angles(self._start_angles, phase1_target, t)
        if elapsed < t2_end:
            t = (elapsed - t1_end) / PHASE2_DURATION
            return lerp_angles(phase1_target, phase2_target, t)
        if elapsed < t3_end:
            t = (elapsed - t2_end) / PHASE3_DURATION
            return lerp_angles(phase2_target, phase3_target, t)
        if elapsed < t4_end:
            t = (elapsed - t3_end) / PHASE4_DURATION
            return lerp_angles(phase3_target, target, t)
        return list(target)

    def _check_contact(self, ctx: PrimitiveContext, closing_fingers: List[int]) -> bool:
        """指定手指接触：触觉优先，无传感器时用电流增量。"""
        return self._contact.check(ctx, closing_fingers)

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        if load_large_wrap_by_vision_params(ctx.hand_type) is not None:
            engine = self._ensure_engine(ctx)
            raw = engine.compute(current_angles, elapsed, ctx)
            return self._move(raw, space=u.space)

        if not getattr(self, "_target_ready", False):
            raw = LARGE_WRAP_ANGLES_L25 if ctx.hand_type == "l25" else LARGE_WRAP_ANGLES
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

        # 触觉闭环: 仅在闭合阶段 (过了 phase1 开始后) 生效
        if elapsed > PHASE1_DURATION * 0.5:
            if not self._size_locked and elapsed > PHASE1_DURATION:
                self._size_locked = True
                self._contact.begin_closing(ctx)

            # 已冻结 → 永远保持，等下一个指令
            if self._frozen_pose is not None:
                self._grasp_state = "contact"
                return self._move(self._frozen_pose, space=u.space)

            # phase1: 四指闭合 → 检测 index/middle/ring/pinky
            # phase2+: 拇指加入 → 检测全部
            if elapsed < PHASE1_DURATION + PHASE2_DURATION:
                closing = [1, 2, 3, 4]
            else:
                closing = [0, 1, 2, 3, 4]

            has_contact = self._check_contact(ctx, closing)
            if has_contact:
                # Freeze the measured pose at first contact.  ``raw`` is the
                # still-in-flight target; holding it lets the fingers keep
                # closing after contact and can extrude the part from the hand.
                self._frozen_pose = list(current_angles)
                self._grasp_state = "contact"
                return self._move(self._frozen_pose, space=u.space)

            # 无接触 + 全部阶段完成 → 渐进闭合
            all_phases_done = elapsed >= (
                PHASE1_DURATION + PHASE2_DURATION + PHASE3_DURATION + PHASE4_DURATION
            )
            if all_phases_done:
                if self._last_size is not None:
                    if self._position_hold_started_at is None:
                        self._position_hold_started_at = elapsed
                    hold_elapsed = elapsed - self._position_hold_started_at
                    if hold_elapsed < POSITION_HOLD_TIMEOUT:
                        self._grasp_state = "position_hold"
                        self._last_elapsed = elapsed
                        return self._move(raw, space=u.space)
                self._grasp_state = "progressive"
                dt = min(elapsed - self._last_elapsed, 0.05)
                self._last_elapsed = elapsed
                if self._progressive_angles is None:
                    self._progressive_angles = list(raw)
                mcp = (
                    _I["thumb_mcp_flex"],
                    _I["index_mcp_flex"],
                    _I["middle_mcp_flex"],
                    _I["ring_mcp_flex"],
                    _I["pinky_mcp_flex"],
                )
                dip = (
                    _I["thumb_dip"],
                    _I["index_dip"],
                    _I["middle_dip"],
                    _I["ring_dip"],
                    _I["pinky_dip"],
                )
                for fi in closing:
                    for idx in (mcp[fi], dip[fi]):
                        inc = u.step(idx, PROGRESSIVE_CLOSE_RATE) * dt
                        self._progressive_angles[idx] = min(
                            u.close_max(idx, PROGRESSIVE_CLOSE_MAX),
                            self._progressive_angles[idx] + inc,
                        )
                # O30 独立 PIP：渐进抬 dip 后同步四指 pip
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
            self._engine = make_large_wrap_by_vision_engine(
                ctx.hand_type, radian_converter=ctx.radian_converter
            )
            if self._engine is None:
                raise RuntimeError(
                    f"large_wrap_by_vision config engine missing for {ctx.hand_type}"
                )
            self._engine.reset(self._start_angles)
            self._engine_hand_type = ctx.hand_type
        return self._engine

    @property
    def name(self) -> str:
        return "large_wrap_by_vision"

    @property
    def done(self) -> bool:
        return False
