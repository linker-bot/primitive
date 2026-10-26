"""原语角度空间：执行核 semantic_rad；0～255 仅边界（hardware / Pose / STREAM）。"""

from __future__ import annotations

from typing import List, Sequence, Tuple

from . import units as U

# PrimitiveResult.space / PrimitiveContext.angle_unit
SPACE_LEGACY_SEMANTIC255 = "semantic"  # 无 URDF 时的退化路径
SPACE_SEMANTIC_RAD = "semantic_rad"
SPACE_HARDWARE = "hardware"

LEGACY_FULL_SCALE = U.LEGACY_FULL_SCALE


def legacy255_to_semantic_rad(converter, semantic_255: Sequence[float]) -> List[float]:
    return converter.legacy255_to_semantic_rad(semantic_255)


def semantic_rad_to_legacy255(converter, semantic_rad: Sequence[float]) -> List[float]:
    return converter.semantic_rad_to_legacy255(semantic_rad)


def start_angles_for_rad_engine(
    start_angles: Sequence[float],
    *,
    radian_converter,
    start_unit: str = SPACE_LEGACY_SEMANTIC255,
) -> List[float]:
    """把 on_enter 缓存的起点转到 semantic_rad，供静态引擎 reset。"""
    if radian_converter is None:
        return list(start_angles)
    unit = (start_unit or SPACE_LEGACY_SEMANTIC255).lower()
    if unit in (SPACE_SEMANTIC_RAD, "radians", "rad"):
        return list(start_angles)
    return legacy255_to_semantic_rad(radian_converter, start_angles)


def result_space_for_converter(radian_converter) -> str:
    return SPACE_SEMANTIC_RAD if radian_converter is not None else SPACE_LEGACY_SEMANTIC255


def _slot_limit(converter, semantic_idx: int) -> Tuple[float, float, bool]:
    """返回 (lower, upper, inverted)；未知槽用 (0, π/2) 退化。"""
    inverted = int(semantic_idx) in set(converter.inverted_indices)
    for sidx, lim in zip(converter.semantic_indices, converter.limits):
        if int(sidx) == int(semantic_idx):
            return float(lim[0]), float(lim[1]), inverted
    return 0.0, 1.5707963267948966, inverted


def value255_to_rad_at_slot(converter, semantic_idx: int, value_255: float) -> float:
    lo, hi, inv = _slot_limit(converter, semantic_idx)
    return U.legacy255_to_rad_at_slot(value_255, lo, hi, inverted=inv)


def step255_to_rad_at_slot(converter, semantic_idx: int, step_255: float) -> float:
    lo, hi, inv = _slot_limit(converter, semantic_idx)
    return U.legacy_step255_to_rad_step(step_255, lo, hi, inverted=inv)


def full_close_rad_at_slot(converter, semantic_idx: int) -> float:
    """legacy 255（满行程）→ 该槽 semantic_rad。"""
    return value255_to_rad_at_slot(converter, semantic_idx, LEGACY_FULL_SCALE)


def abd_neutral_rad_at_slot(converter, semantic_idx: int) -> float:
    """legacy ABD_NEUTRAL(128) → rad。"""
    from .primitive_base import ABD_NEUTRAL

    return value255_to_rad_at_slot(converter, semantic_idx, float(ABD_NEUTRAL))


class AngleUnits:
    """原语侧量纲助手：有 Converter 则全部按 rad；否则退化 legacy255。"""

    def __init__(self, ctx) -> None:
        self._converter = getattr(ctx, "radian_converter", None)
        unit = str(getattr(ctx, "angle_unit", "") or "").lower()
        # 显式 legacy 语义保留 255；空 unit + 有 converter 视为 rad（避免预载等路径漏设）
        if self._converter is None:
            self.rad = False
        elif unit in (SPACE_LEGACY_SEMANTIC255, "legacy255", "255"):
            self.rad = False
        elif unit in (SPACE_SEMANTIC_RAD, "radians", "rad") or unit == "":
            self.rad = True
        else:
            self.rad = False

    @property
    def space(self) -> str:
        return SPACE_SEMANTIC_RAD if self.rad else SPACE_LEGACY_SEMANTIC255

    def pose(self, pose_255: Sequence[float]) -> List[float]:
        from .semantic_30 import as_semantic_30

        vec = as_semantic_30(pose_255)
        if not self.rad:
            return vec
        return legacy255_to_semantic_rad(self._converter, vec)

    def value(self, semantic_idx: int, value_255: float) -> float:
        if not self.rad:
            return float(value_255)
        return value255_to_rad_at_slot(self._converter, semantic_idx, value_255)

    def step(self, semantic_idx: int, step_255: float) -> float:
        if not self.rad:
            return float(step_255)
        return step255_to_rad_at_slot(self._converter, semantic_idx, step_255)

    def close_max(self, semantic_idx: int, max_255: float = LEGACY_FULL_SCALE) -> float:
        if not self.rad:
            return float(max_255)
        return value255_to_rad_at_slot(self._converter, semantic_idx, max_255)

    def abd_neutral(self, semantic_idx: int) -> float:
        if not self.rad:
            from .primitive_base import ABD_NEUTRAL

            return float(ABD_NEUTRAL)
        return abd_neutral_rad_at_slot(self._converter, semantic_idx)

    def scale_step(self, step_255: float, control_rate_hz: float, semantic_idx: int) -> float:
        """255 标定步进 → 当前频率下的 rad（或 legacy）每 tick 增量。"""
        from .contact_config import scale_per_tick_delta

        scaled = scale_per_tick_delta(step_255, control_rate_hz)
        return self.step(semantic_idx, scaled)
