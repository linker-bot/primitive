"""可名义规划 / 可终态导出的静态原语集合（单一真相源）。

PlanGesture、Framework ``plan-hand-command``、``export_primitive_poses`` 共用本集合，
避免白名单漂移。力控 / vision 闭环原语不在此列（终态依赖接触或感知）。

PlanGesture 额外要求本手型 YAML ``gestures:`` 中确有该原语配置，避免回退到
open 默认终态造成名义错误规划。
"""

from __future__ import annotations

from typing import FrozenSet, Optional

# 可离线求终态、且 PlanGesture 可复用 StaticPoseEngine 的原语。
PLANABLE_STATIC_PRIMITIVES: FrozenSet[str] = frozenset(
    {
        "open",
        "init",
        "fist",
        "release",
        "relax_grip",
        "pinch",
        "index_pinch",
        "middle_pinch",
        "grasp_ready",
        "parallel_extension",
        "point",
        "ok_sign",
        "v_sign",
    }
)

# 预定义终态导出默认名前缀（避免与内置原语名冲突）。
DEFAULT_PRESET_NAME_PREFIX = "preset_"


def is_planable_static(primitive: str) -> bool:
    return str(primitive or "").strip().lower() in PLANABLE_STATIC_PRIMITIVES


def configured_static_gestures(hand_type: str) -> FrozenSet[str]:
    """本手型 YAML gestures 段中已配置的原语名。"""
    from .gesture_params import _load_raw_gestures
    from .hand_config import resolve_hand_config_key

    canonical = resolve_hand_config_key(hand_type)
    raw = _load_raw_gestures(canonical) or {}
    return frozenset(str(k).lower() for k in raw.keys())


def is_planable_for_hand(hand_type: str, primitive: str) -> bool:
    """全局可规划且本手型 YAML 已配置终态。"""
    name = str(primitive or "").strip().lower()
    if not is_planable_static(name):
        return False
    return name in configured_static_gestures(hand_type)


def planable_reject_reason(hand_type: str, primitive: str) -> Optional[str]:
    """不可规划时返回英文原因；可规划返回 None。"""
    name = str(primitive or "").strip().lower()
    if not name:
        return "empty primitive"
    if not is_planable_static(name):
        return (
            f"unsupported nominal preview primitive: {name}; "
            f"planable={sorted(PLANABLE_STATIC_PRIMITIVES)}"
        )
    if name not in configured_static_gestures(hand_type):
        return (
            f"primitive '{name}' has no gesture config on hand '{hand_type}' "
            "(refusing default/open fallback for nominal plan)"
        )
    return None
