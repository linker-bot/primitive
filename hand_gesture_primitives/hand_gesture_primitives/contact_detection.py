"""Contact detection for adaptive grasp primitives.

Priority (per hand type):
  O20/L25: tactile → joint current Δ (mA)
  O6:      tactile → joint torque Δ (0~100%, hand_info.torque / hand_motor_torque)
"""

from typing import List, Optional, Sequence

import numpy as np

from .contact_config import ContactThresholds
from .primitive_base import HAND_CONFIGS, HandConfig, PrimitiveContext

_FINGER_JOINT_KEYS = {
    0: ("thumb_base", "thumb_tip"),
    1: ("index_base", "index_tip"),
    2: ("middle_base", "middle_tip"),
    3: ("ring_base", "ring_tip"),
    4: ("pinky_base", "pinky_tip"),
}


def uses_torque_feedback(hand_type: str) -> bool:
    """O6 无 0x36 电流，力控走 motor_torque sense。"""
    return str(hand_type).lower() == "o6"


def hand_config(ctx: PrimitiveContext) -> HandConfig:
    return HAND_CONFIGS.get(ctx.hand_type, HAND_CONFIGS["o20"])


def finger_joint_indices(cfg: HandConfig, finger_idx: int) -> List[int]:
    base_key, tip_key = _FINGER_JOINT_KEYS[finger_idx]
    return [getattr(cfg, base_key), getattr(cfg, tip_key)]


def tactile_contact(ctx: PrimitiveContext, finger_indices: Sequence[int]) -> bool:
    """True if any listed finger exceeds tactile threshold."""
    if ctx.tactile_mode == "none" or ctx.tactile_pressure is None:
        return False
    t: ContactThresholds = ctx.contact_thresholds
    threshold = (
        t.mass_threshold if ctx.tactile_mode == "mass"
        else t.pressure_threshold
    )
    p = ctx.tactile_pressure
    return any(float(p[i]) > threshold for i in finger_indices if i < len(p))


def tactile_finger_for_group(group_id: str, joint_names: Sequence[str]) -> Optional[int]:
    """力控组 id / 关节名 → 触觉数组下标 (0=拇 … 4=小)。"""
    keys = (("thumb", 0), ("index", 1), ("middle", 2), ("ring", 3), ("pinky", 4))
    text = group_id.lower()
    for key, idx in keys:
        if key in text:
            return idx
    for jn in joint_names:
        jnl = str(jn).lower()
        for key, idx in keys:
            if key in jnl:
                return idx
    return None


def feedback_at_hw_indices(
    ctx: PrimitiveContext, hw_indices: Sequence[int],
) -> List[float]:
    """O6: 力矩 0~100；O20/L25: 电流 mA。无效采样为 -1。"""
    out: List[float] = []
    if uses_torque_feedback(ctx.hand_type):
        t = ctx.joint_torque
        for i in hw_indices:
            if t is not None and 0 <= i < len(t) and float(t[i]) >= 0:
                out.append(float(t[i]))
            else:
                out.append(-1.0)
        return out
    currents = ctx.joint_currents
    for i in hw_indices:
        if 0 <= i < len(currents) and float(currents[i]) >= 0:
            out.append(float(currents[i]))
        else:
            out.append(-1.0)
    return out


def feedback_delta_exceeded(
    ctx: PrimitiveContext,
    baseline: Sequence[float],
    hw_indices: Sequence[int],
    delta: float,
) -> bool:
    """任一监测关节相对基线超过 delta（O6=% , O20=mA）。"""
    vals = feedback_at_hw_indices(ctx, hw_indices)
    for j, hi in enumerate(hw_indices):
        if j >= len(vals):
            continue
        v = vals[j]
        if v < 0:
            continue
        if hi < len(baseline):
            b = float(baseline[hi])
        elif j < len(baseline):
            b = float(baseline[j])
        else:
            continue
        if b < 0:
            continue
        if v - b > delta:
            return True
    return False


def capture_feedback_baseline(ctx: PrimitiveContext) -> List[float]:
    """整段 joint 空间基线（按 hand_config.num_joints 长度）。"""
    cfg = hand_config(ctx)
    n = cfg.num_joints
    if uses_torque_feedback(ctx.hand_type):
        t = ctx.joint_torque
        if t is None:
            return [-1.0] * n
        return [float(t[i]) if i < len(t) else -1.0 for i in range(n)]
    return [float(c) for c in ctx.joint_currents[:n]]


def force_close_group_contact(
    ctx: PrimitiveContext,
    *,
    tactile_finger: Optional[int],
    monitor_hw: Sequence[int],
    baseline: Sequence[float],
    contact_delta: float,
) -> bool:
    """顺序力控单组接触：触觉优先，否则力矩/电流增量。"""
    if ctx.tactile_mode != "none" and tactile_finger is not None:
        return tactile_contact(ctx, [tactile_finger])
    return feedback_delta_exceeded(
        ctx, baseline, monitor_hw, contact_delta)


def hold_feedback_exceeded(
    ctx: PrimitiveContext,
    hw_idx: int,
    hold_limit: float,
) -> bool:
    """持握阶段单关节是否超过安全力矩/电流上限。"""
    vals = feedback_at_hw_indices(ctx, [hw_idx])
    if not vals or vals[0] < 0:
        return False
    return vals[0] > hold_limit


def feedback_contact(
    ctx: PrimitiveContext,
    finger_indices: Sequence[int],
    baseline: Optional[List[float]],
    settle_count: int,
) -> bool:
    """多指闭合：触觉优先，否则力矩/电流增量。"""
    t = ctx.contact_thresholds
    if settle_count < t.current_settle_frames:
        return False
    if baseline is None:
        return False
    if ctx.tactile_mode != "none":
        return tactile_contact(ctx, finger_indices)

    cfg = hand_config(ctx)
    delta = (
        t.torque_delta_pct if uses_torque_feedback(ctx.hand_type)
        else t.current_delta
    )
    hw: List[int] = []
    for finger_idx in finger_indices:
        for joint_idx in finger_joint_indices(cfg, finger_idx):
            if joint_idx >= 0:
                hw.append(joint_idx)
    if not hw:
        return False
    # baseline 与 hw 对齐：取 baseline[joint_idx]
    base_hw = [float(baseline[i]) if i < len(baseline) else -1.0 for i in hw]
    return feedback_delta_exceeded(ctx, base_hw, hw, delta)


def pinch_motion_should_stop(
    ctx: PrimitiveContext,
    monitor_hw: Sequence[int],
    tactile_fingers: Sequence[int],
    *,
    lerp_progress: float = 1.0,
    baseline: Optional[Sequence[float]] = None,
) -> tuple[bool, str]:
    """静态 timed pinch 停指。

    O6：仅压感。/hand_motor_torque 空载收指常态 60~80%，绝对值/Δ 均会误触。
    O20/L25：压感 + 关节电流绝对阈值 (mA)。
    """
    if ctx.tactile_mode != "none" and tactile_fingers:
        if tactile_contact(ctx, tactile_fingers):
            return True, "tactile"

    if uses_torque_feedback(ctx.hand_type):
        return False, ""

    if not monitor_hw:
        return False, ""

    threshold = 200.0 if ctx.hand_type == "o20" else 400.0
    currents = ctx.joint_currents
    for hi in monitor_hw:
        if hi < len(currents) and float(currents[hi]) > threshold:
            return True, f"stall_current hw{hi}={currents[hi]:.0f}mA>{threshold:.0f}"
    return False, ""


def pinch_motion_stop_detail(
    ctx: PrimitiveContext,
    monitor_hw: Sequence[int],
    tactile_fingers: Sequence[int],
    baseline: Optional[Sequence[float]],
) -> str:
    """诊断用：当前监测值与阈值。"""
    t = ctx.contact_thresholds
    parts: List[str] = []
    if ctx.tactile_pressure is not None and tactile_fingers:
        p = ctx.tactile_pressure
        for fi in tactile_fingers:
            if fi < len(p):
                parts.append(f"tactile[{fi}]={float(p[fi]):.1f}")
    vals = feedback_at_hw_indices(ctx, monitor_hw)
    if uses_torque_feedback(ctx.hand_type):
        for j, hi in enumerate(monitor_hw):
            v = vals[j] if j < len(vals) else -1.0
            parts.append(f"torque_hw{hi}={v:.0f}%")
        parts.append(f"mass>={t.mass_threshold} (O6 pinch: tactile-only)")
    else:
        for j, hi in enumerate(monitor_hw):
            if hi < len(ctx.joint_currents):
                parts.append(f"current_hw{hi}={ctx.joint_currents[hi]:.0f}mA")
        parts.append(f"current_stop>200mA")
    return " ".join(parts)


def feedback_blocks_motion(
    ctx: PrimitiveContext,
    monitor_hw: Sequence[int],
    tactile_fingers: Sequence[int],
    baseline: Optional[Sequence[float]],
    contact_delta: Optional[float] = None,
) -> bool:
    """全程监测：压感或力矩/电流超阈则不应继续加力闭合。"""
    if ctx.tactile_mode != "none" and tactile_fingers:
        if tactile_contact(ctx, tactile_fingers):
            return True
    if not monitor_hw:
        return False
    t = ctx.contact_thresholds
    delta = contact_delta
    if delta is None:
        delta = (
            t.torque_delta_pct if uses_torque_feedback(ctx.hand_type)
            else t.current_delta
        )
    if baseline is not None and feedback_delta_exceeded(
        ctx, baseline, monitor_hw, delta,
    ):
        return True
    if not uses_torque_feedback(ctx.hand_type):
        threshold = 200.0 if ctx.hand_type == "o20" else 400.0
        currents = ctx.joint_currents
        if any(
            i < len(currents) and float(currents[i]) > threshold
            for i in monitor_hw
        ):
            return True
    return False


def monitors_for_fingers(
    ctx: PrimitiveContext, finger_indices: Sequence[int],
) -> tuple[List[int], List[int]]:
    """contact_fingers 下标 → (hardware 监测关节, 触觉指下标)。"""
    cfg = hand_config(ctx)
    hw: List[int] = []
    for fi in finger_indices:
        for ji in finger_joint_indices(cfg, int(fi)):
            if ji >= 0:
                hw.append(ji)
    return sorted(set(hw)), [int(f) for f in finger_indices]


class MotionFeedbackGuard:
    """原语全程：遇触觉/力矩/电流异常则冻结在上帧指令，避免堵转强抓。"""

    def __init__(self) -> None:
        self._baseline: Optional[List[float]] = None
        self._frozen: Optional[List[float]] = None
        self._monitor_hw: List[int] = []
        self._tactile_fingers: List[int] = []
        self._contact_delta: Optional[float] = None
        self._enabled: bool = True

    def reset(
        self,
        monitor_hw: Optional[Sequence[int]] = None,
        tactile_fingers: Optional[Sequence[int]] = None,
        contact_delta: Optional[float] = None,
    ) -> None:
        self._baseline = None
        self._frozen = None
        self._monitor_hw = list(monitor_hw or [])
        self._tactile_fingers = list(tactile_fingers or [])
        self._contact_delta = contact_delta
        self._enabled = bool(self._monitor_hw or self._tactile_fingers)

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    def configured(self) -> bool:
        return bool(self._monitor_hw or self._tactile_fingers)

    @property
    def frozen(self) -> bool:
        return self._frozen is not None

    def guard(
        self,
        ctx: PrimitiveContext,
        proposed: Sequence[float],
        previous: Optional[Sequence[float]] = None,
    ) -> List[float]:
        if not self._enabled:
            return list(proposed)
        if self._frozen is not None:
            return list(self._frozen)
        prev = list(previous if previous is not None else proposed)
        if self._baseline is None:
            self._baseline = capture_feedback_baseline(ctx)
        if feedback_blocks_motion(
            ctx,
            self._monitor_hw,
            self._tactile_fingers,
            self._baseline,
            self._contact_delta,
        ):
            self._frozen = prev
            return list(self._frozen)
        return list(proposed)


def collect_force_close_monitors(
    steps: Sequence,
) -> tuple[List[int], List[int], float]:
    """从 SequentialForceCloseParams.steps 汇总监测关节与默认 Δ。"""
    hw: set = set()
    tactile: set = set()
    delta = 15.0
    for step in steps:
        for g in step.groups:
            hw.update(g.monitor_hw)
            if g.tactile_finger is not None:
                tactile.add(int(g.tactile_finger))
            delta = float(g.contact_delta)
    return sorted(hw), sorted(tactile), delta


class FingerContactTracker:
    """Tracks baseline during closing; tactile or torque/current fallback."""

    def __init__(self) -> None:
        self._baseline: Optional[List[float]] = None
        self._settle_count: int = 0
        self._active: bool = False

    def reset(self) -> None:
        self._baseline = None
        self._settle_count = 0
        self._active = False

    def begin_closing(self, ctx: PrimitiveContext) -> None:
        if self._active:
            return
        self._active = True
        self._baseline = capture_feedback_baseline(ctx)
        self._settle_count = 0

    def check(self, ctx: PrimitiveContext, finger_indices: Sequence[int]) -> bool:
        if ctx.tactile_mode != "none":
            return tactile_contact(ctx, finger_indices)
        if not self._active:
            self.begin_closing(ctx)
        settle_frames = ctx.contact_thresholds.current_settle_frames
        if self._settle_count < settle_frames:
            self._settle_count += 1
            if self._settle_count == settle_frames:
                self._baseline = capture_feedback_baseline(ctx)
        return feedback_contact(
            ctx, finger_indices, self._baseline, self._settle_count,
        )


# 兼容旧名
def current_contact(
    ctx: PrimitiveContext,
    finger_indices: Sequence[int],
    baseline: Optional[List[float]],
    settle_count: int,
) -> bool:
    return feedback_contact(ctx, finger_indices, baseline, settle_count)
