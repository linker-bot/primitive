"""量纲转换：legacy semantic 0～255 / hardware 0～255 / deg ↔ semantic_rad。

原语执行核目标为内部弧度（semantic_rad，S30 槽位布局）。
0～255 为过渡态；degrees 仅作输入/显示糖衣。

本模块不依赖 ROS；需 URDF 限位时传入 ``Ros2ControlRadianConverter``
或显式 ``limits_by_semantic``。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# 纯数值（无 URDF）
# ---------------------------------------------------------------------------

LEGACY_FULL_SCALE = 255.0


def deg_to_rad(degrees: float) -> float:
    return float(degrees) * math.pi / 180.0


def rad_to_deg(radians: float) -> float:
    return float(radians) * 180.0 / math.pi


def deg_vec_to_rad(values: Sequence[float]) -> List[float]:
    return [deg_to_rad(v) for v in values]


def rad_vec_to_deg(values: Sequence[float]) -> List[float]:
    return [rad_to_deg(v) for v in values]


def clamp01(ratio: float) -> float:
    return max(0.0, min(1.0, float(ratio)))


def legacy255_ratio(value: float) -> float:
    return clamp01(float(value) / LEGACY_FULL_SCALE)


def ratio_to_legacy255(ratio: float) -> float:
    return clamp01(ratio) * LEGACY_FULL_SCALE


# ---------------------------------------------------------------------------
# 基于语义槽限位 (lower, upper) + 可选 invert
# ---------------------------------------------------------------------------

LimitsBySemantic = Dict[int, Tuple[float, float]]
InvertSet = set


def legacy255_to_rad_at_slot(
    value_255: float,
    lower: float,
    upper: float,
    *,
    inverted: bool = False,
) -> float:
    """单槽：legacy semantic 255 → URDF 弧度。"""
    if upper <= lower:
        return float(lower)
    ratio = legacy255_ratio(value_255)
    if inverted:
        ratio = 1.0 - ratio
    return float(lower) + ratio * (float(upper) - float(lower))


def rad_to_legacy255_at_slot(
    value_rad: float,
    lower: float,
    upper: float,
    *,
    inverted: bool = False,
) -> float:
    """单槽：URDF 弧度 → legacy semantic 255。"""
    if upper <= lower:
        return 0.0
    clamped = max(float(lower), min(float(upper), float(value_rad)))
    ratio = (clamped - float(lower)) / (float(upper) - float(lower))
    if inverted:
        ratio = 1.0 - ratio
    return ratio_to_legacy255(ratio)


def legacy_semantic255_to_rad(
    semantic_255: Sequence[float],
    *,
    limits_by_semantic: LimitsBySemantic,
    inverted_indices: Optional[InvertSet] = None,
    width: int = 30,
) -> List[float]:
    """S30 legacy semantic → semantic_rad（无对应限位的槽保持 0）。"""
    inverted = inverted_indices or set()
    out = [0.0] * width
    src = list(semantic_255)
    while len(src) < width:
        src.append(0.0)
    for idx, (lower, upper) in limits_by_semantic.items():
        if idx < 0 or idx >= width:
            continue
        out[idx] = legacy255_to_rad_at_slot(src[idx], lower, upper, inverted=idx in inverted)
    return out


def semantic_rad_to_legacy255(
    semantic_rad: Sequence[float],
    *,
    limits_by_semantic: LimitsBySemantic,
    inverted_indices: Optional[InvertSet] = None,
    width: int = 30,
) -> List[float]:
    """semantic_rad → S30 legacy semantic 255。"""
    inverted = inverted_indices or set()
    out = [0.0] * width
    src = list(semantic_rad)
    while len(src) < width:
        src.append(0.0)
    for idx, (lower, upper) in limits_by_semantic.items():
        if idx < 0 or idx >= width:
            continue
        out[idx] = rad_to_legacy255_at_slot(src[idx], lower, upper, inverted=idx in inverted)
    return out


def semantic_rad_to_controller(
    semantic_rad: Sequence[float],
    *,
    semantic_indices: Sequence[int],
    limits: Sequence[Tuple[float, float]],
) -> List[float]:
    """semantic_rad（S30 槽）→ controller 关节序弧度（重排 + clamp）。"""
    out: List[float] = []
    for semantic_idx, (lower, upper) in zip(semantic_indices, limits):
        if upper <= lower:
            out.append(float(lower))
            continue
        raw = float(semantic_rad[semantic_idx]) if semantic_idx < len(semantic_rad) else 0.0
        out.append(max(float(lower), min(float(upper), raw)))
    return out


def controller_to_semantic_rad(
    controller_radians: Sequence[float],
    *,
    semantic_indices: Sequence[int],
    limits: Sequence[Tuple[float, float]],
    width: int = 30,
) -> List[float]:
    """controller 序弧度 → semantic_rad。"""
    out = [0.0] * width
    for semantic_idx, value, (lower, upper) in zip(semantic_indices, controller_radians, limits):
        if semantic_idx < 0 or semantic_idx >= width:
            continue
        if upper <= lower:
            out[semantic_idx] = float(lower)
            continue
        out[semantic_idx] = max(float(lower), min(float(upper), float(value)))
    return out


def legacy_step255_to_rad_step(
    step_255: float,
    lower: float,
    upper: float,
    *,
    inverted: bool = False,
) -> float:
    """位置步进：255 刻度/tick → rad/tick（按关节全行程比例，保留符号）。

    ``inverted`` 关节：legacy255 增大对应 rad 减小，步进取反。
    """
    span = float(upper) - float(lower)
    if span <= 0.0:
        return 0.0
    signed = float(step_255) / LEGACY_FULL_SCALE * span
    return -signed if inverted else signed


# ---------------------------------------------------------------------------
# 与 Ros2ControlRadianConverter 对接的便捷封装
# ---------------------------------------------------------------------------


def limits_map_from_converter(converter) -> LimitsBySemantic:
    """从 Converter 提取 semantic_idx → (lower, upper)。"""
    out: LimitsBySemantic = {}
    for semantic_idx, lim in zip(converter.semantic_indices, converter.limits):
        out[int(semantic_idx)] = (float(lim[0]), float(lim[1]))
    return out


def converter_legacy255_to_rad(converter, semantic_255: Sequence[float]) -> List[float]:
    from .semantic_30 import SEMANTIC_DIM

    return legacy_semantic255_to_rad(
        semantic_255,
        limits_by_semantic=limits_map_from_converter(converter),
        inverted_indices=set(converter.inverted_indices),
        width=SEMANTIC_DIM,
    )


def converter_rad_to_legacy255(converter, semantic_rad: Sequence[float]) -> List[float]:
    from .semantic_30 import SEMANTIC_DIM

    return semantic_rad_to_legacy255(
        semantic_rad,
        limits_by_semantic=limits_map_from_converter(converter),
        inverted_indices=set(converter.inverted_indices),
        width=SEMANTIC_DIM,
    )


def converter_rad_to_controller(converter, semantic_rad: Sequence[float]) -> List[float]:
    return semantic_rad_to_controller(
        semantic_rad,
        semantic_indices=converter.semantic_indices,
        limits=converter.limits,
    )


def converter_controller_to_rad(converter, controller_radians: Sequence[float]) -> List[float]:
    from .semantic_30 import SEMANTIC_DIM

    return controller_to_semantic_rad(
        controller_radians,
        semantic_indices=converter.semantic_indices,
        limits=converter.limits,
        width=SEMANTIC_DIM,
    )
