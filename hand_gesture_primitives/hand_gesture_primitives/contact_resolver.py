"""按 hardware / semantic 关节名解析电流反馈索引（供力控原语使用）。"""

from __future__ import annotations

from typing import List, Optional, Sequence

from .hand_config import HandConfig
from .primitive_base import JOINT_NAMES
from .semantic_30 import (
    SEMANTIC_30_IDX,
    SEMANTIC_30_NAMES,
    o20_joint_index_to_s30,
    s30_joint_index_to_o20,
)

_SEMANTIC_JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


TORQUE_RAW_MAX = 255.0
# O6 hand.torque 官方量纲：0~100 无量纲百分比（100 对应最大电流 1657.5 mA）


def raw_torque_to_pct(value: float) -> float:
    """Legacy 驱动 raw 0~255 → 0~100%（仅当采样值 >100 时启用）。"""
    if value < 0:
        return value
    return float(value) * 100.0 / TORQUE_RAW_MAX


def normalize_motor_torque_values(
    values: Sequence[float],
    num_joints: int,
) -> Optional[List[float]]:
    """统一为 O6 官方 0~100% 量纲；若存在 >100 的采样则按 legacy raw 0~255 换算。"""
    if not values or len(values) < num_joints:
        return None
    try:
        sliced = [float(v) for v in values[:num_joints]]
    except (TypeError, ValueError):
        return None
    if any(v > 100.0 for v in sliced if v >= 0):
        return [raw_torque_to_pct(v) if v >= 0 else v for v in sliced]
    return sliced


def parse_hand_info_torque(data: dict, num_joints: int) -> Optional[List[float]]:
    """从 hand_info JSON 解析关节力矩 (0~100%)。

    O6 官方：hand.torque 读写均为 0~100 无量纲百分比（100 ≈ 1657.5 mA）。
    字段优先级：motor_torque_pct → motor_torque_joints → torque
    """
    for key in ("motor_torque_pct", "motor_torque_joints", "torque"):
        raw = data.get(key)
        if raw is None:
            continue
        normalized = normalize_motor_torque_values(raw, num_joints)
        if normalized is not None:
            return normalized
    return None


def parse_hand_info_currents(data: dict, num_joints: int) -> Optional[List[float]]:
    """从 hand_info JSON 解析关节电流 (mA)。

    LinkerHand SDK 发布字段为 ``current``；历史文档曾用 ``current_current``。
    返回长度须等于 ``num_joints``，否则返回 None。
    """
    raw = data.get("current_current")
    if raw is None:
        raw = data.get("current")
    if raw is None:
        return None
    try:
        currents = [float(c) for c in raw]
    except (TypeError, ValueError):
        return None
    if len(currents) != num_joints:
        return None
    return currents


def semantic_index(joint: str) -> Optional[int]:
    """关节名 / 数字 → **S30 通道下标**（写入姿态向量）。

    解析顺序：
    1. S30 通道名（``thumb_mcp_flex``）
    2. O20 名（``thumb_base``）→ 投影到 S30
    3. 纯数字：按历史 YAML 约定视为 **O20 下标** 再投影到 S30
    """
    s = str(joint)
    if s in SEMANTIC_30_IDX:
        return int(SEMANTIC_30_IDX[s])
    if s.isdigit():
        s30 = o20_joint_index_to_s30(int(s))
        return s30 if s30 >= 0 else None
    o20 = _SEMANTIC_JOINT_NAME_TO_IDX.get(s)
    if o20 is not None:
        return o20_joint_index_to_s30(o20)
    return None


def semantic_indices(joints: Sequence[str]) -> List[int]:
    out: List[int] = []
    for j in joints:
        idx = semantic_index(str(j))
        if idx is not None:
            out.append(idx)
    return out


def hw_indices(hand: HandConfig, joint_names: Sequence[str]) -> List[int]:
    """hardware 关节名 → 驱动电流数组下标。"""
    out: List[int] = []
    for name in joint_names:
        if name in hand.joint_names:
            out.append(hand.joint_names.index(name))
    return out


def current_delta_exceeded(
    currents: Sequence[float],
    baseline: Sequence[float],
    hw_idx: Sequence[int],
    delta: float,
) -> bool:
    """任一监测关节电流相对 baseline 超过 delta 则视为接触。"""
    for i in hw_idx:
        if i < 0 or i >= len(currents) or i >= len(baseline):
            continue
        if float(currents[i]) - float(baseline[i]) > delta:
            return True
    return False


def monitor_indices_for_semantic(
    hand: HandConfig,
    semantic_idx: int,
    explicit_hw_names: Optional[Sequence[str]] = None,
) -> List[int]:
    """O20 semantic 下标 → hardware 电流下标（可显式指定 monitor 名）。"""
    if explicit_hw_names:
        return hw_indices(hand, explicit_hw_names)
    for hi, spec in hand._reverse_map.items():
        if spec["from"] == semantic_idx:
            return [hi]
    return []


def monitor_indices_for_s30(
    hand: HandConfig,
    s30_idx: int,
    explicit_hw_names: Optional[Sequence[str]] = None,
) -> List[int]:
    """semantic_30 通道下标 → hardware 电流下标。"""
    if explicit_hw_names:
        return hw_indices(hand, explicit_hw_names)
    idx = int(s30_idx)
    if 0 <= idx < len(SEMANTIC_30_NAMES):
        spec = hand._semantic_map.get(SEMANTIC_30_NAMES[idx])
        if spec is not None:
            return [int(spec["to"])]
    o20 = s30_joint_index_to_o20(idx)
    if o20 < 0:
        return []
    return monitor_indices_for_semantic(hand, o20)


def current_index_for_s30(hand: HandConfig, s30_idx: int) -> Optional[int]:
    """单个 S30 通道对应的 hardware 电流下标；无映射返回 None。"""
    idxs = monitor_indices_for_s30(hand, s30_idx)
    return idxs[0] if idxs else None


def hw_current_delta(
    currents: Sequence[float],
    baseline: Sequence[float],
    hand: HandConfig,
    s30_idx: int,
) -> float:
    """S30 通道上的电流相对基线增量（无映射/越界 → 0）。"""
    hi = current_index_for_s30(hand, s30_idx)
    if hi is None or hi < 0 or hi >= len(currents) or hi >= len(baseline):
        return 0.0
    return float(currents[hi]) - float(baseline[hi])


# 静态 pinch 原语力反馈监测关节（O6 用 hardware 名；其它用 S30 通道下标）
_PINCH_MONITOR_HW: dict[str, tuple[str, ...]] = {
    "pinch": ("thumb_cmc_pitch", "index_mcp_pitch"),
    "index_pinch": ("thumb_cmc_pitch", "index_mcp_pitch"),
    "middle_pinch": ("thumb_cmc_pitch", "middle_mcp_pitch"),
}

# S30：thumb_mcp_flex/dip + 对指 mcp_flex/dip
_PINCH_MONITOR_S30: dict[str, tuple[int, ...]] = {
    "pinch": (0, 3, 5, 8),
    "index_pinch": (0, 3, 5, 8),
    "middle_pinch": (0, 3, 10, 13),
}


def current_monitor_indices(hand: HandConfig, primitive: str) -> List[int]:
    """返回 pinch 类原语电流监测用的 joint_currents 下标。"""
    primitive = primitive.lower()
    if hand.num_joints <= 6:
        names = _PINCH_MONITOR_HW.get(primitive)
        return hw_indices(hand, names) if names else []
    out: List[int] = []
    for s30_idx in _PINCH_MONITOR_S30.get(primitive, ()):
        out.extend(monitor_indices_for_s30(hand, s30_idx))
    return list(dict.fromkeys(out))
