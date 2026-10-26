"""tripod_by_force 的手型无关接触停指策略。

运动学仍复用同手型 ``tripod``（L6 两阶段走父类）。本模块只负责：
从 YAML 解析参与手指、把 semantic 屈曲关节映射到硬件电流/力矩下标、
按手型选择反馈通道。

扩展新手型：
1. ``config/{hand}.yaml`` 的 ``primitives.supported`` 加入 ``tripod_by_force``
2. 可选 ``gestures.tripod_by_force.contact`` 覆盖通道/阈值；省略则用下方默认
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import yaml

from .contact_detection import (
    capture_feedback_baseline,
    feedback_delta_exceeded,
    force_close_group_contact,
    tactile_contact,
    uses_torque_feedback,
)
from .contact_resolver import monitor_indices_for_s30
from .hand_config import HandConfig, _find_config_path
from .primitive_base import PrimitiveContext, present_joints
from .semantic_30 import SEMANTIC_30_IDX

# 拇指 / 食指 / 中指 … 的 S30 屈曲通道（mcp_flex + dip）
_FINGER_SEMANTIC_FLEX: Dict[int, Tuple[int, int]] = {
    0: (SEMANTIC_30_IDX["thumb_mcp_flex"], SEMANTIC_30_IDX["thumb_dip"]),
    1: (SEMANTIC_30_IDX["index_mcp_flex"], SEMANTIC_30_IDX["index_dip"]),
    2: (SEMANTIC_30_IDX["middle_mcp_flex"], SEMANTIC_30_IDX["middle_dip"]),
    3: (SEMANTIC_30_IDX["ring_mcp_flex"], SEMANTIC_30_IDX["ring_dip"]),
    4: (SEMANTIC_30_IDX["pinky_mcp_flex"], SEMANTIC_30_IDX["pinky_dip"]),
}

_FINGER_NAME_TO_IDX: Dict[str, int] = {
    "thumb": 0,
    "index": 1,
    "middle": 2,
    "ring": 3,
    "pinky": 4,
}

_DEFAULT_FINGERS: Tuple[int, ...] = (0, 1, 2)
_DEFAULT_PREP_DURATION = 0.5
_DEFAULT_CLOSE_DURATION = 0.5

# 未在 YAML 声明时的反馈通道。L6 必须等电流；O6 力矩空载易误触，只走触觉。
_DEFAULT_FEEDBACK_SOURCE: Dict[str, str] = {
    "l6": "motor_current",
    "o6": "tactile",
}

VALID_FEEDBACK_SOURCES = frozenset({"auto", "motor_current", "current", "torque", "tactile"})


@dataclass(frozen=True)
class TripodForceGroup:
    """单指：冻结的 semantic 关节 + 监测用硬件下标。"""

    finger: int
    freeze_semantic: Tuple[int, ...]
    monitor_hw: Tuple[int, ...]


@dataclass(frozen=True)
class TripodForceParams:
    """按手型解析后的接触策略。运动时序仍由 tripod 原语负责。"""

    hand_type: str
    groups: Tuple[TripodForceGroup, ...]
    feedback_source: str
    feedback_delta: Optional[float]
    require_feedback: bool
    prep_duration: float = _DEFAULT_PREP_DURATION
    close_duration: float = _DEFAULT_CLOSE_DURATION

    def all_freeze_semantic(self) -> Tuple[int, ...]:
        out: List[int] = []
        for group in self.groups:
            out.extend(group.freeze_semantic)
        return tuple(dict.fromkeys(out))


def _parse_finger(token: Any) -> Optional[int]:
    if isinstance(token, bool):
        return None
    if isinstance(token, int):
        return token if token in _FINGER_SEMANTIC_FLEX else None
    key = str(token).strip().lower()
    if key.isdigit():
        idx = int(key)
        return idx if idx in _FINGER_SEMANTIC_FLEX else None
    return _FINGER_NAME_TO_IDX.get(key)


def _parse_fingers(raw: Any) -> Tuple[int, ...]:
    if not raw:
        return _DEFAULT_FINGERS
    if not isinstance(raw, (list, tuple)):
        return _DEFAULT_FINGERS
    fingers: List[int] = []
    for item in raw:
        idx = _parse_finger(item)
        if idx is not None and idx not in fingers:
            fingers.append(idx)
    return tuple(fingers) if fingers else _DEFAULT_FINGERS


def _load_raw_block(hand_type: str) -> dict:
    path = _find_config_path(hand_type)
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    gestures = data.get("gestures") or {}
    block = gestures.get("tripod_by_force") or {}
    return block if isinstance(block, dict) else {}


def _default_source(hand_type: str) -> str:
    return _DEFAULT_FEEDBACK_SOURCE.get(hand_type, "auto")


def _build_group(
    hand: HandConfig,
    finger: int,
    *,
    freeze_tips: bool,
) -> TripodForceGroup:
    base, tip = _FINGER_SEMANTIC_FLEX[finger]
    freeze = present_joints(base, tip) if freeze_tips else present_joints(base)
    monitor: List[int] = []
    for semantic in freeze:
        monitor.extend(monitor_indices_for_s30(hand, semantic))
    return TripodForceGroup(
        finger=finger,
        freeze_semantic=tuple(freeze),
        monitor_hw=tuple(dict.fromkeys(monitor)),
    )


@lru_cache(maxsize=16)
def load_tripod_force_params(hand_type: str) -> TripodForceParams:
    """加载 ``gestures.tripod_by_force``；缺省字段用可扩展默认值。"""
    key = hand_type.lower()
    hand = HandConfig(key)
    raw = _load_raw_block(key)
    contact = raw.get("contact") or {}
    motion = raw.get("motion") or {}
    if not isinstance(contact, dict):
        contact = {}
    if not isinstance(motion, dict):
        motion = {}

    source = str(contact.get("feedback_source") or _default_source(key)).lower()
    if source not in VALID_FEEDBACK_SOURCES:
        raise ValueError(
            f"{key} tripod_by_force.contact.feedback_source 无效: {source!r}，"
            f"可选: {sorted(VALID_FEEDBACK_SOURCES)}"
        )

    raw_delta = contact.get("feedback_delta")
    feedback_delta = float(raw_delta) if raw_delta is not None else None
    if feedback_delta is not None and feedback_delta < 0.0:
        raise ValueError(f"{key} tripod_by_force.contact.feedback_delta 不能为负")

    require = contact.get("require_feedback")
    if require is None:
        require_feedback = source == "motor_current"
    else:
        require_feedback = bool(require)

    freeze_tips = bool(contact.get("freeze_tips", True))
    groups = tuple(
        _build_group(hand, finger, freeze_tips=freeze_tips)
        for finger in _parse_fingers(contact.get("fingers"))
    )
    if not groups:
        raise ValueError(f"{key} tripod_by_force 未解析到任何手指组")

    return TripodForceParams(
        hand_type=key,
        groups=groups,
        feedback_source=source,
        feedback_delta=feedback_delta,
        require_feedback=require_feedback,
        prep_duration=float(motion.get("phase1", _DEFAULT_PREP_DURATION)),
        close_duration=float(motion.get("phase2", _DEFAULT_CLOSE_DURATION)),
    )


def clear_tripod_force_params_cache() -> None:
    load_tripod_force_params.cache_clear()


def resolve_contact_delta(params: TripodForceParams, ctx: PrimitiveContext) -> float:
    if params.feedback_delta is not None:
        return max(0.0, params.feedback_delta)
    thresholds = ctx.contact_thresholds
    if uses_torque_feedback(ctx.hand_type) or params.feedback_source == "torque":
        return float(thresholds.torque_delta_pct)
    return float(thresholds.current_delta)


def feedback_sources(source: str) -> tuple[str, ...]:
    """Canonical alternatives shared by execution and workbench readiness."""
    if source == "motor_current":
        return ("current",)
    if source == "auto":
        return ("tactile", "current", "torque")
    return (source,)


def feedback_ready(params: TripodForceParams, ctx: PrimitiveContext) -> bool:
    """闭合前是否已有可用反馈。require_feedback=False 时调用方可跳过等待。"""
    checks = {
        "current": lambda: _current_monitors_ready(params, ctx),
        "torque": lambda: _torque_monitors_ready(params, ctx),
        "tactile": lambda: ctx.tactile_mode != "none" and ctx.tactile_pressure is not None,
    }
    return any(checks[source]() for source in feedback_sources(params.feedback_source))


def _current_monitors_ready(params: TripodForceParams, ctx: PrimitiveContext) -> bool:
    if not ctx.joint_currents_valid:
        return False
    currents = ctx.joint_currents
    for group in params.groups:
        for hw_index in group.monitor_hw:
            if hw_index >= len(currents):
                return False
            value = float(currents[hw_index])
            if value < 0.0:
                return False
    return True


def _torque_monitors_ready(params: TripodForceParams, ctx: PrimitiveContext) -> bool:
    torque = ctx.joint_torque
    if torque is None:
        return False
    for group in params.groups:
        for hw_index in group.monitor_hw:
            if hw_index >= len(torque) or float(torque[hw_index]) < 0.0:
                return False
    return True


def group_in_contact(
    params: TripodForceParams,
    ctx: PrimitiveContext,
    group: TripodForceGroup,
    baseline: Sequence[float],
) -> bool:
    """单指是否接触。motor_current/current 不走触觉，避免 L6 占位电流被触觉抢先。"""
    delta = resolve_contact_delta(params, ctx)
    source = params.feedback_source
    if source in ("motor_current", "current"):
        return feedback_delta_exceeded(ctx, baseline, group.monitor_hw, delta)
    if source == "torque":
        return feedback_delta_exceeded(ctx, baseline, group.monitor_hw, delta)
    if source == "tactile":
        return tactile_contact(ctx, [group.finger])
    return force_close_group_contact(
        ctx,
        tactile_finger=group.finger,
        monitor_hw=group.monitor_hw,
        baseline=baseline,
        contact_delta=delta,
    )


def apply_finger_freeze(
    params: TripodForceParams,
    ctx: PrimitiveContext,
    proposed: Sequence[float],
    last_output: Sequence[float],
    baseline: Sequence[float],
    frozen: Dict[int, float],
) -> List[float]:
    """逐指冻结：接触后保持上一帧目标，避免回退卸力。"""
    for group in params.groups:
        if any(idx in frozen for idx in group.freeze_semantic):
            continue
        if not group_in_contact(params, ctx, group, baseline):
            continue
        for idx in group.freeze_semantic:
            if idx < len(last_output):
                frozen[idx] = float(last_output[idx])
    out = list(proposed)
    for idx, value in frozen.items():
        if idx < len(out):
            out[idx] = value
    return out


def capture_baseline(ctx: PrimitiveContext) -> List[float]:
    return capture_feedback_baseline(ctx)


def groups_all_frozen(params: TripodForceParams, frozen: Mapping[int, float]) -> bool:
    if not params.groups:
        return False
    return all(any(idx in frozen for idx in group.freeze_semantic) for group in params.groups)
