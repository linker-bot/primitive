"""从手型 YAML gestures 段加载原语参数，供各原语复用同一套运动逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Mapping, Optional

import yaml

from .hand_config import HandConfig, _find_config_path
from .primitive_base import ABD_NEUTRAL, JOINT_NAMES, RESERVED_INDICES

# semantic_o20 参考配置手型（原语内部 20-DOF 语义空间；非运行时 hand_type）
CANONICAL_SEMANTIC_HAND = "o20"

# O20 语义下 thumb_adduction_grip 默认（YAML 缺省时的 fallback）
_DEFAULT_THUMB_ADDUCTION: Dict[str, Any] = {
    "angles": {
        "space": "semantic_o20",
        "target": {
            "thumb_base": 255,
            "index_base": 160,
            "middle_base": 160,
            "ring_base": 160,
            "pinky_base": 160,
            "thumb_abd": 130,
            "thumb_rot": 170,
            "thumb_tip": 135,
            "index_tip": 140,
            "middle_tip": 140,
            "ring_tip": 140,
            "pinky_tip": 140,
        },
    },
    "motion": {"phase1": 0.35, "phase2": 0.40, "phase3": 0.35, "phase4": 0.65},
    "joints": {"hold": [0, 5, 10, 15], "flex": [0, 15], "rot": 10},
    "progressive": {
        "abd": 5, "flex": 15, "abd_max": 185, "flex_max": 150, "rate": 20.0,
    },
    "contact": {"fingers": [0, 1]},
}

_SEMANTIC_JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_raw_gestures(hand_type: str) -> dict:
    path = _find_config_path(hand_type)
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("gestures") or {}


def _semantic_from_sparse(sparse: Mapping[str, Any]) -> List[float]:
    """稀疏 joint 名 / O20 索引 → 20 维 semantic 向量。"""
    out = [0.0] * 20
    for key, val in sparse.items():
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            idx = int(key)
        else:
            idx = _SEMANTIC_JOINT_NAME_TO_IDX.get(str(key))
        if idx is None or idx < 0 or idx >= 20:
            continue
        out[idx] = float(val)
    for i in (6, 7, 8, 9):
        if out[i] == 0.0:
            out[i] = float(ABD_NEUTRAL)
    for i in RESERVED_INDICES:
        out[i] = 0.0
    return out


def hw_pose_to_semantic(
    hw_by_name: Mapping[str, float], hand: HandConfig,
) -> List[float]:
    """硬件关节名 → O20 semantic（利用 from_o20 invert 映射）。"""
    hw = [hand._reserved_default] * hand.num_joints
    for i, name in enumerate(hand.joint_names):
        if name in hw_by_name:
            hw[i] = float(hw_by_name[name])
    return hand.from_hardware(hw)


@dataclass(frozen=True)
class ThumbAdductionParams:
    """thumb_adduction_grip 手型相关参数（已规范为 O20 semantic 20 维）。"""

    hand_type: str
    prep_angles: List[float]
    close_angles: List[float]
    phase1: float = 0.35
    phase2: float = 0.40
    phase3: float = 0.35
    phase4: float = 0.65
    thumb_hold_joints: List[int] = field(default_factory=lambda: [0, 5, 10, 15])
    thumb_flex_joints: List[int] = field(default_factory=lambda: [0, 15])
    thumb_rot_joint: int = 10
    close_move_joints: List[int] = field(default_factory=list)
    progressive_abd_joint: int = 5
    progressive_flex_joint: int = 15
    progressive_abd_max: float = 185.0
    progressive_flex_max: float = 150.0
    progressive_rate: float = 20.0
    contact_fingers: List[int] = field(default_factory=lambda: [0, 1])
    position_stop_joint: Optional[int] = None
    position_stop_target_semantic: Optional[float] = None
    position_stop_tolerance: float = 5.0

    @property
    def prep_end(self) -> float:
        return self.phase1 + self.phase2 + self.phase3

    def close_flex_indices(self) -> List[int]:
        if self.close_move_joints:
            return list(self.close_move_joints)
        return list(self.thumb_flex_joints)


def _parse_angles(section: dict, hand: HandConfig) -> tuple:
    """返回 (prep_angles, close_angles) semantic 20 维。"""
    space = str(section.get("space", "semantic_o20")).lower()
    if space == "hardware":
        prep_hw = section.get("prep") or section.get("target") or {}
        close_hw = section.get("close") or {}
        prep = hw_pose_to_semantic(prep_hw, hand)
        close_merged = dict(prep_hw)
        close_merged.update(close_hw)
        close = hw_pose_to_semantic(close_merged, hand)
        return prep, close
    target = section.get("target") or section.get("prep") or {}
    prep = _semantic_from_sparse(target)
    close_section = section.get("close") or {}
    if close_section:
        merged = {**{JOINT_NAMES[i]: prep[i] for i in range(20)}, **close_section}
        close = _semantic_from_sparse(merged)
    else:
        close = list(prep)
    return prep, close


def _parse_thumb_adduction(raw: dict, hand_type: str, hand: HandConfig) -> ThumbAdductionParams:
    cfg = _deep_merge(_DEFAULT_THUMB_ADDUCTION, raw)
    angles_sec = cfg.get("angles") or {}
    prep, close = _parse_angles(angles_sec, hand)
    motion = cfg.get("motion") or {}
    joints = cfg.get("joints") or {}
    prog = cfg.get("progressive") or {}
    contact = cfg.get("contact") or {}
    close_sec = cfg.get("close") or {}

    rot = joints.get("rot", 10)
    pos_joint = contact.get("position_joint")
    pos_idx = None
    pos_target = None
    if pos_joint is not None:
        if isinstance(pos_joint, int):
            pos_idx = int(pos_joint)
        elif isinstance(pos_joint, str):
            if pos_joint.isdigit():
                pos_idx = int(pos_joint)
            elif pos_joint in hand.joint_names:
                pos_idx = hand.joint_names.index(pos_joint)
                rev = hand._reverse_map.get(pos_idx)
                if rev:
                    pos_idx = rev["from"]
            elif pos_joint in _SEMANTIC_JOINT_NAME_TO_IDX:
                pos_idx = _SEMANTIC_JOINT_NAME_TO_IDX[pos_joint]
        if contact.get("position_target_hw") is not None and pos_idx is not None:
            hw_name = None
            for hi, spec in hand._reverse_map.items():
                if spec["from"] == pos_idx:
                    hw_name = hand.joint_names[hi]
                    break
            if hw_name:
                pos_target = hw_pose_to_semantic(
                    {hw_name: float(contact["position_target_hw"])}, hand,
                )[pos_idx]
        elif contact.get("position_target") is not None and pos_idx is not None:
            pos_target = float(contact["position_target"])

    return ThumbAdductionParams(
        hand_type=hand_type,
        prep_angles=prep,
        close_angles=close,
        phase1=float(motion.get("phase1", 0.35)),
        phase2=float(motion.get("phase2", 0.40)),
        phase3=float(motion.get("phase3", 0.35)),
        phase4=float(motion.get("phase4", 0.65)),
        thumb_hold_joints=[int(x) for x in joints.get("hold", [0, 5, 10, 15])],
        thumb_flex_joints=[int(x) for x in joints.get("flex", [0, 15])],
        thumb_rot_joint=int(rot) if rot is not None and int(rot) >= 0 else -1,
        close_move_joints=[int(x) for x in close_sec.get("move_only", [])],
        progressive_abd_joint=int(prog.get("abd", 5)),
        progressive_flex_joint=int(prog.get("flex", 15)),
        progressive_abd_max=float(prog.get("abd_max", 185)),
        progressive_flex_max=float(prog.get("flex_max", 150)),
        progressive_rate=float(prog.get("rate", 20.0)),
        contact_fingers=[int(x) for x in contact.get("fingers", [0, 1])],
        position_stop_joint=pos_idx,
        position_stop_target_semantic=pos_target,
        position_stop_tolerance=float(contact.get("position_tolerance", 5.0)),
    )


@lru_cache(maxsize=16)
def load_thumb_adduction_params(hand_type: str) -> ThumbAdductionParams:
    """加载指定手型的 thumb_adduction_grip 参数（带缓存）。"""
    hand_type = hand_type.lower()
    hand = HandConfig(hand_type)
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get("thumb_adduction_grip") or {}
    return _parse_thumb_adduction(raw, hand_type, hand)


def clear_gesture_params_cache() -> None:
    load_thumb_adduction_params.cache_clear()
    load_static_gesture_params.cache_clear()
    load_sequential_force_close_params.cache_clear()
    load_ring_params.cache_clear()
    load_middle_ring_params.cache_clear()


# ---------------------------------------------------------------------------
# 静态 / 分阶段姿态原语 (open, init, fist, release, relax_grip)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaticGestureParams:
    """静态或分阶段姿态原语配置。"""

    hand_type: str
    primitive: str
    motion_type: str
    target_angles: List[float]
    duration: float = 0.6
    phase1: float = 0.5
    phase2: float = 0.4
    phase3: float = 0.4
    phase4: float = 0.4
    phase1_hold: List[int] = field(default_factory=lambda: [0, 5, 10, 15])
    relax_angles: List[float] = field(default_factory=list)
    grip_angles: List[float] = field(default_factory=list)
    hold_duration: float = 0.3
    grip_duration: float = 0.4
    relax_phased: Optional["StaticGestureParams"] = None
    grip_phased: Optional["StaticGestureParams"] = None


def _phased_motion_duration(params: StaticGestureParams) -> float:
    p3 = params.phase3 if params.phase3 > 0 else 0.0
    return params.phase1 + params.phase2 + p3 + params.phase4


def _resolve_relax_grip_sub(
    section: Any,
    hand_type: str,
    hand: HandConfig,
    default_ref: str,
) -> tuple[List[float], Optional[StaticGestureParams], float]:
    """解析 relax_grip 的 relax/grip 子段；支持 ref 字符串或 phased 块。

    返回 (target_angles, phased_params_or_none, phased_duration)。
    """
    if section is None or section == {}:
        ref = default_ref
        section = {}
    elif isinstance(section, str):
        ref = section
        section = {}
    else:
        ref = str(section.get("ref", default_ref))

    ref_params = load_static_gesture_params(hand_type, ref)
    if section.get("angles"):
        angles = _resolve_static_target(
            "relax_grip", hand_type, hand, section["angles"])
    else:
        angles = list(ref_params.target_angles)

    phased = _parse_phased_sub_params(
        section, hand_type, hand, angles, f"relax_grip_{ref}")
    if phased is None and ref_params.motion_type == "phased_pose":
        phased = StaticGestureParams(
            hand_type=ref_params.hand_type,
            primitive=ref_params.primitive,
            motion_type="phased_pose",
            target_angles=list(angles),
            phase1=ref_params.phase1,
            phase2=ref_params.phase2,
            phase3=ref_params.phase3,
            phase4=ref_params.phase4,
            phase1_hold=list(ref_params.phase1_hold),
        )
    dur = _phased_motion_duration(phased) if phased else 0.0
    return angles, phased, dur


def _parse_phased_sub_params(
    section: dict,
    hand_type: str,
    hand: HandConfig,
    target_angles: List[float],
    primitive: str,
) -> Optional[StaticGestureParams]:
    """relax_grip 子段 phased_pose 配置（如 O6 拇延后）。"""
    sub_motion = section.get("motion") or {}
    if str(sub_motion.get("type", "")).lower() != "phased_pose":
        return None
    joints = section.get("joints") or {}
    rot = joints.get("rot", 10)
    hold = joints.get("hold", [0, 5, 10, 15])
    if rot is not None and int(rot) < 0:
        hold = [int(x) for x in hold if int(x) != 10]
    return StaticGestureParams(
        hand_type=hand_type,
        primitive=primitive,
        motion_type="phased_pose",
        target_angles=list(target_angles),
        phase1=float(sub_motion.get("phase1", 0.5)),
        phase2=float(sub_motion.get("phase2", 0.4)),
        phase3=float(sub_motion.get("phase3", 0.4)),
        phase4=float(sub_motion.get("phase4", 0.4)),
        phase1_hold=[int(x) for x in hold],
    )


def _open_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 0, "index_base": 0, "middle_base": 0,
        "ring_base": 0, "pinky_base": 0, "thumb_abd": 0,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 0,
        "thumb_tip": 0, "index_tip": 0, "middle_tip": 0,
        "ring_tip": 0, "pinky_tip": 0,
    })


def _init_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 80, "index_base": 80, "middle_base": 80,
        "ring_base": 80, "pinky_base": 80, "thumb_abd": 60,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 50,
        "thumb_tip": 60, "index_tip": 60, "middle_tip": 60,
        "ring_tip": 60, "pinky_tip": 60,
    })


def _fist_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 60, "index_base": 255, "middle_base": 255,
        "ring_base": 255, "pinky_base": 255, "thumb_abd": 200,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 230,
        "thumb_tip": 255, "index_tip": 255, "middle_tip": 255,
        "ring_tip": 255, "pinky_tip": 255,
    })


def _v_sign_o20() -> List[float]:
    out = _semantic_from_sparse({
        "thumb_base": 200, "index_base": 0, "middle_base": 0,
        "ring_base": 255, "pinky_base": 255,
        "thumb_abd": 140,
        "index_abd": 255,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 150,
        "thumb_tip": 255, "index_tip": 0, "middle_tip": 0,
        "ring_tip": 255, "pinky_tip": 255,
    })
    out[7] = 0.0  # middle_abd: V 形张开（0 为有效值，非 abd 默认）
    return out


def _ok_sign_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 0, "index_base": 150,
        "middle_base": 100, "ring_base": 70, "pinky_base": 40,
        "thumb_abd": 170,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 220,
        "thumb_tip": 200, "index_tip": 100,
        "middle_tip": 100, "ring_tip": 70, "pinky_tip": 40,
    })


def _point_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 230, "index_base": 0,
        "middle_base": 255, "ring_base": 255, "pinky_base": 255,
        "thumb_abd": 140,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 150,
        "thumb_tip": 200, "index_tip": 0,
        "middle_tip": 255, "ring_tip": 255, "pinky_tip": 255,
    })


def _pinch_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 180, "index_base": 180,
        "middle_base": 0, "ring_base": 0, "pinky_base": 0,
        "thumb_abd": 130,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 200,
        "thumb_tip": 180, "index_tip": 180,
        "middle_tip": 0, "ring_tip": 0, "pinky_tip": 0,
    })


def _index_pinch_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 110, "index_base": 140,
        "middle_base": 0, "ring_base": 0, "pinky_base": 0,
        "thumb_abd": 140,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 200,
        "thumb_tip": 160, "index_tip": 130,
        "middle_tip": 0, "ring_tip": 0, "pinky_tip": 0,
    })


def _middle_pinch_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 120, "index_base": 0,
        "middle_base": 140, "ring_base": 0, "pinky_base": 0,
        "thumb_abd": 190,
        "index_abd": 200, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 185,
        "thumb_tip": 160, "index_tip": 0,
        "middle_tip": 130, "ring_tip": 0, "pinky_tip": 0,
    })


def _tripod_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 139, "index_base": 184, "middle_base": 163,
        "ring_base": 0, "pinky_base": 0,
        "thumb_abd": 227,
        "index_abd": 193, "middle_abd": 255,
        "ring_abd": 105, "pinky_abd": 52,
        "thumb_rot": 110,
        "thumb_tip": 143, "index_tip": 0,
        "middle_tip": 80, "ring_tip": 0, "pinky_tip": 0,
    })


def _relax_o20() -> List[float]:
    return list(_open_o20())


def _grip_o20() -> List[float]:
    return _semantic_from_sparse({
        "thumb_base": 255, "index_base": 255, "middle_base": 255,
        "ring_base": 255, "pinky_base": 255, "thumb_abd": 140,
        "index_abd": ABD_NEUTRAL, "middle_abd": ABD_NEUTRAL,
        "ring_abd": ABD_NEUTRAL, "pinky_abd": ABD_NEUTRAL,
        "thumb_rot": 160,
        "thumb_tip": 255, "index_tip": 255, "middle_tip": 255,
        "ring_tip": 255, "pinky_tip": 255,
    })


# L25 专用 25-DOF（与 open_hand.py / init_hand.py 一致）
_L25_OPEN = [
    159, 0, 0, 0, 0, 105, 141, 104, 66, 0, 75, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
]
_L25_INIT = [
    85, 55, 55, 55, 55, 135, 127, 127, 127, 127, 115, 0, 0, 0, 0,
    55, 55, 55, 55, 55, 75, 75, 75, 75, 75,
]

_DEFAULT_STATIC: Dict[str, Dict[str, Any]] = {
    "open": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "init": {
        "motion": {"type": "static", "duration": 0.8},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "fist": {
        "motion": {
            "type": "phased_pose", "phase1": 0.5, "phase2": 0.4,
            "phase3": 0.4, "phase4": 0.4,
        },
        "joints": {"hold": [0, 5, 10, 15]},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "release": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "relax_grip": {
        "motion": {
            "type": "relax_grip", "duration": 0.6,
            "hold_duration": 0.3, "grip_duration": 0.4,
        },
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "pinch": {
        "motion": {"type": "static", "duration": 0.5},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "index_pinch": {
        "motion": {"type": "static", "duration": 0.5},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "middle_pinch": {
        "motion": {"type": "static", "duration": 0.5},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "tripod": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "point": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "ok_sign": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_o20", "target": {}},
    },
    "v_sign": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_o20", "target": {}},
    },
}

_STATIC_TARGET_BUILDERS = {
    ("o20", "open"): _open_o20,
    ("o20", "init"): _init_o20,
    ("o20", "fist"): _fist_o20,
    ("o20", "release"): _open_o20,
    ("o20", "relax_grip"): _fist_o20,
    ("o6", "open"): _open_o20,
    ("o6", "init"): _init_o20,
    ("o6", "fist"): _fist_o20,
    ("o6", "release"): _open_o20,
    ("o6", "relax_grip"): _fist_o20,
    ("o20", "pinch"): _pinch_o20,
    ("o20", "index_pinch"): _index_pinch_o20,
    ("o20", "middle_pinch"): _middle_pinch_o20,
    ("o20", "tripod"): _tripod_o20,
    ("o20", "point"): _point_o20,
    ("o20", "ok_sign"): _ok_sign_o20,
    ("o20", "v_sign"): _v_sign_o20,
}

_L25_TARGET_BUILDERS = {
    "open": lambda: list(_L25_OPEN),
    "init": lambda: list(_L25_INIT),
    "release": lambda: list(_L25_OPEN),
}


def _resolve_static_target(
    primitive: str, hand_type: str, hand: HandConfig, angles_sec: dict,
) -> List[float]:
    if hand_type == "l25" and primitive in _L25_TARGET_BUILDERS:
        if angles_sec.get("target") or angles_sec.get("prep"):
            pass
        else:
            return _L25_TARGET_BUILDERS[primitive]()
    space = str(angles_sec.get("space", "semantic_o20")).lower()
    target_map = angles_sec.get("target") or angles_sec.get("prep") or {}
    if target_map:
        if space == "hardware":
            return hw_pose_to_semantic(target_map, hand)
        return _semantic_from_sparse(target_map)
    builder = _STATIC_TARGET_BUILDERS.get((hand_type, primitive))
    if builder:
        return builder()
    if hand_type == "l25" and primitive in _L25_TARGET_BUILDERS:
        return _L25_TARGET_BUILDERS[primitive]()
    return _open_o20()


def _parse_static_gesture(
    primitive: str, raw: dict, hand_type: str, hand: HandConfig,
) -> StaticGestureParams:
    base = _DEFAULT_STATIC.get(primitive, _DEFAULT_STATIC["open"])
    cfg = _deep_merge(base, raw)
    motion = cfg.get("motion") or {}
    joints = cfg.get("joints") or {}
    angles_sec = cfg.get("angles") or {}
    motion_type = str(motion.get("type", "static")).lower()

    target = _resolve_static_target(primitive, hand_type, hand, angles_sec)
    target_map = angles_sec.get("target") or angles_sec.get("prep") or {}
    if primitive == "release" and not target_map:
        target = list(load_static_gesture_params(hand_type, "open").target_angles)

    relax = list(target)
    grip = list(target)
    relax_phased = None
    grip_phased = None
    duration = float(motion.get("duration", 0.6))
    grip_duration = float(motion.get("grip_duration", 0.4))

    if motion_type == "relax_grip":
        relax_sec = cfg.get("relax")
        grip_sec = cfg.get("grip")
        relax, relax_phased, relax_dur = _resolve_relax_grip_sub(
            relax_sec, hand_type, hand, "open")
        grip, grip_phased, grip_dur = _resolve_relax_grip_sub(
            grip_sec, hand_type, hand, "fist")
        duration = max(duration, relax_dur)
        grip_duration = max(grip_duration, grip_dur)

    rot = joints.get("rot", 10)
    hold = joints.get("hold", [0, 5, 10, 15])
    if rot is not None and int(rot) < 0:
        hold = [int(x) for x in hold if int(x) != 10]

    return StaticGestureParams(
        hand_type=hand_type,
        primitive=primitive,
        motion_type=motion_type,
        target_angles=target,
        duration=duration,
        phase1=float(motion.get("phase1", 0.5)),
        phase2=float(motion.get("phase2", 0.4)),
        phase3=float(motion.get("phase3", 0.4)),
        phase4=float(motion.get("phase4", 0.4)),
        phase1_hold=[int(x) for x in hold],
        relax_angles=relax,
        grip_angles=grip,
        hold_duration=float(motion.get("hold_duration", 0.3)),
        grip_duration=grip_duration,
        relax_phased=relax_phased,
        grip_phased=grip_phased,
    )


@lru_cache(maxsize=64)
def load_static_gesture_params(hand_type: str, primitive: str) -> StaticGestureParams:
    """加载 open/init/fist/release/relax_grip 静态姿态配置。"""
    hand_type = hand_type.lower()
    primitive = primitive.lower()
    hand = HandConfig(hand_type)
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get(primitive) or {}
    return _parse_static_gesture(primitive, raw, hand_type, hand)


def resolve_static_target_angles(
    hand_type: str, primitive: str, default: List[float],
) -> List[float]:
    """若手型 YAML gestures 中有该原语则返回语义角，否则 default。"""
    ht = hand_type.lower()
    primitive = primitive.lower()
    if primitive not in (_load_raw_gestures(ht) or {}):
        return list(default)
    return list(load_static_gesture_params(ht, primitive).target_angles)


# ---------------------------------------------------------------------------
# ring — sequential_force_close (O6 MCP 力控包络)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ForceCloseGroup:
    """单组顺序力控：若干 semantic 关节 + 电流监测 hardware 关节。"""

    group_id: str
    joint_indices: List[int]
    monitor_hw: List[int]
    step: float = 8.0
    contact_delta: float = 270.0
    close_max: float = 255.0
    tactile_finger: Optional[int] = None
    # O6 力矩 Δ% 时使用；未设则与 contact_delta 相同
    contact_delta_torque_pct: Optional[float] = None


@dataclass(frozen=True)
class ForceCloseStep:
    """顺序力控中的一步（可含多组并行闭合，如食+中）。"""

    name: str
    groups: List[ForceCloseGroup]


@dataclass(frozen=True)
class SequentialForceCloseParams:
    hand_type: str
    prep_angles: List[float]
    preshape_duration: float = 0.5
    settle_frames: int = 10
    hold_safe_current: float = 800.0
    hold_safe_torque_pct: float = 55.0
    steps: List[ForceCloseStep] = field(default_factory=list)


def _parse_force_close_groups(
    step_cfg: dict, hand: HandConfig, default_delta: float, default_step: float,
) -> List[ForceCloseGroup]:
    from .contact_detection import tactile_finger_for_group
    from .contact_resolver import hw_indices, semantic_indices

    groups_raw = step_cfg.get("groups") or []
    if not groups_raw:
        joints = step_cfg.get("joints") or []
        monitor = step_cfg.get("monitor") or step_cfg.get("monitor_joints") or []
        groups_raw = [{
            "id": step_cfg.get("name", "group"),
            "joints": joints,
            "monitor": monitor,
        }]
    delta = float(step_cfg.get("contact_delta", default_delta))
    if hand.num_joints <= 6:
        delta = float(step_cfg.get(
            "contact_delta_torque_pct",
            step_cfg.get("contact_delta", default_delta),
        ))
    step = float(step_cfg.get("step", default_step))
    out: List[ForceCloseGroup] = []
    for g in groups_raw:
        joint_names = [str(x) for x in g.get("joints", [])]
        monitor_names = [str(x) for x in g.get("monitor", g.get("monitor_joints", []))]
        sem = semantic_indices(joint_names)
        mon = hw_indices(hand, monitor_names) if monitor_names else []
        if not mon and sem:
            from .contact_resolver import monitor_indices_for_semantic
            for si in sem:
                mon.extend(monitor_indices_for_semantic(hand, si))
        out.append(ForceCloseGroup(
            group_id=str(g.get("id", "group")),
            joint_indices=sem,
            monitor_hw=sorted(set(mon)),
            step=step,
            contact_delta=delta,
            close_max=float(g.get("close_max", 255)),
            tactile_finger=(
                int(g["tactile_finger"])
                if g.get("tactile_finger") is not None
                else tactile_finger_for_group(str(g.get("id", "group")), joint_names)
            ),
        ))
    return out


def _parse_sequential_force_close(
    raw: dict, hand_type: str, hand: HandConfig, primitive: str,
) -> SequentialForceCloseParams:
    angles_sec = raw.get("angles") or {}
    prep, _ = _parse_angles(angles_sec, hand)
    motion = raw.get("motion") or {}
    contact = raw.get("contact") or {}
    if hand.num_joints <= 6:
        default_delta = float(
            contact.get("contact_delta_torque_pct",
                        contact.get("contact_delta", 15.0)))
    else:
        default_delta = float(contact.get("contact_delta", 270))
    default_step = float(contact.get("finger_step", 8))
    steps_cfg = raw.get("steps") or []
    if not steps_cfg:
        if primitive == "middle_ring":
            steps_cfg = [
                {
                    "name": "close_middle",
                    "step": default_step,
                    "contact_delta": default_delta,
                    "groups": [
                        {"id": "middle", "joints": ["middle_base"],
                         "monitor": ["middle_mcp_pitch"]},
                    ],
                },
                {
                    "name": "close_thumb",
                    "step": float(contact.get("thumb_step", default_step)),
                    "contact_delta": default_delta,
                    "groups": [
                        {"id": "thumb", "joints": ["thumb_base"],
                         "monitor": ["thumb_cmc_pitch"]},
                    ],
                },
            ]
        else:
            steps_cfg = [
                {
                    "name": "close_fingers",
                    "step": default_step,
                    "contact_delta": default_delta,
                    "groups": [
                        {"id": "index", "joints": ["index_base"],
                         "monitor": ["index_mcp_pitch"]},
                        {"id": "middle", "joints": ["middle_base"],
                         "monitor": ["middle_mcp_pitch"]},
                    ],
                },
                {
                    "name": "close_thumb",
                    "step": float(contact.get("thumb_step", default_step)),
                    "contact_delta": default_delta,
                    "groups": [
                        {"id": "thumb", "joints": ["thumb_base"],
                         "monitor": ["thumb_cmc_pitch"]},
                    ],
                },
            ]
    steps = [
        ForceCloseStep(
            name=str(s.get("name", f"step{i}")),
            groups=_parse_force_close_groups(s, hand, default_delta, default_step),
        )
        for i, s in enumerate(steps_cfg)
    ]
    return SequentialForceCloseParams(
        hand_type=hand_type,
        prep_angles=prep,
        preshape_duration=float(motion.get("preshape_duration", 0.5)),
        settle_frames=int(motion.get("settle_frames", 10)),
        hold_safe_current=float(contact.get("hold_safe_current", 800)),
        hold_safe_torque_pct=float(contact.get("hold_safe_torque_pct", 55.0)),
        steps=steps,
    )


@lru_cache(maxsize=32)
def load_sequential_force_close_params(
    hand_type: str, primitive: str,
) -> Optional[SequentialForceCloseParams]:
    """加载 sequential_force_close 手势配置；无配置或非该类型返回 None。"""
    hand_type = hand_type.lower()
    primitive = primitive.lower()
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get(primitive) or {}
    motion = raw.get("motion") or {}
    if str(motion.get("type", "")).lower() != "sequential_force_close":
        return None
    hand = HandConfig(hand_type)
    return _parse_sequential_force_close(raw, hand_type, hand, primitive)


@lru_cache(maxsize=16)
def load_ring_params(hand_type: str) -> Optional[SequentialForceCloseParams]:
    """加载 ring 的 sequential_force_close 配置；无配置或非该类型返回 None。"""
    return load_sequential_force_close_params(hand_type, "ring")


@lru_cache(maxsize=16)
def load_middle_ring_params(hand_type: str) -> Optional[SequentialForceCloseParams]:
    """加载 middle_ring 的 sequential_force_close 配置；无配置或非该类型返回 None。"""
    return load_sequential_force_close_params(hand_type, "middle_ring")
