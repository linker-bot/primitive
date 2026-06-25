"""Contact detection for adaptive grasp primitives.

Priority:
1. Tactile (pressure 0-255 or matrix mass in grams) when hardware publishes data
2. Joint current rise from baseline (O20 / no tactile sensor)
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


def current_contact(
    ctx: PrimitiveContext,
    finger_indices: Sequence[int],
    baseline: Optional[List[float]],
    settle_count: int,
) -> bool:
    """True if joint current rose above baseline on any closing finger joint."""
    t = ctx.contact_thresholds
    if settle_count < t.current_settle_frames:
        return False
    if baseline is None:
        return False

    cfg = hand_config(ctx)
    currents = ctx.joint_currents
    n = min(len(currents), len(baseline))
    if n == 0:
        return False

    delta = t.current_delta
    for finger_idx in finger_indices:
        for joint_idx in finger_joint_indices(cfg, finger_idx):
            if joint_idx < 0 or joint_idx >= n:
                continue
            if currents[joint_idx] - baseline[joint_idx] > delta:
                return True
    return False


class FingerContactTracker:
    """Tracks baseline current during closing; supports tactile or current fallback."""

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
        self._baseline = list(ctx.joint_currents)
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
                self._baseline = list(ctx.joint_currents)
        return current_contact(
            ctx, finger_indices, self._baseline, self._settle_count,
        )
