"""从手型 YAML gestures 段加载原语参数，供各原语复用同一套运动逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Mapping, Optional

import yaml

from .hand_config import HandConfig, _find_config_path
from .primitive_base import ABD_NEUTRAL, JOINT_NAMES, RESERVED_INDICES
from .semantic_30 import (
    ABD_INDICES_S30,
    SEMANTIC_30_IDX,
    SEMANTIC_30_NAMES,
    SEMANTIC_DIM,
    as_semantic_30,
    o20_joint_index_to_s30,
    project_o20_to_s30,
)

# 缺省手势参数参考手型（非运行时 hand_type）
CANONICAL_SEMANTIC_HAND = "o20"

# thumb_adduction_grip 默认（YAML 缺省时的 fallback；semantic_v3 / S30）
_DEFAULT_THUMB_ADDUCTION: Dict[str, Any] = {
    "angles": {
        "space": "semantic_v3",
        "target": {
            "thumb_mcp_flex": 255,
            "index_mcp_flex": 160,
            "middle_mcp_flex": 160,
            "ring_mcp_flex": 160,
            "pinky_mcp_flex": 160,
            "thumb_mcp_abd": 130,
            "thumb_rot": 170,
            "thumb_dip": 135,
            "index_pip": 140,
            "index_dip": 140,
            "middle_pip": 140,
            "middle_dip": 140,
            "ring_pip": 140,
            "ring_dip": 140,
            "pinky_pip": 140,
            "pinky_dip": 140,
        },
    },
    "motion": {"phase1": 0.35, "phase2": 0.40, "phase3": 0.35, "phase4": 0.65},
    # joints/progressive 缺省由解析器按 space 选 O20 或 S30 下标，避免 deep_merge 错译
    "progressive": {
        "abd_max": 185,
        "flex_max": 150,
        "rate": 20.0,
    },
    "contact": {"fingers": [0, 1]},
}

_SEMANTIC_JOINT_NAME_TO_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}


def _s30_index_for_hw_joint(hand: HandConfig, hw_idx: int) -> Optional[int]:
    """hardware 关节下标 → S30 通道下标（from_semantic 映射）。"""
    for ch, spec in hand._semantic_map.items():
        if spec["to"] == hw_idx:
            return SEMANTIC_30_IDX[ch]
    return None


def _hw_name_for_pose_idx(hand: HandConfig, pos_idx: int, *, space: str) -> Optional[str]:
    """position_stop 通道下标 → hardware 关节名（供 position_target_hw 换算）。"""
    if space == "semantic_v3" and hand.has_from_semantic:
        if 0 <= pos_idx < len(SEMANTIC_30_NAMES):
            ch = SEMANTIC_30_NAMES[pos_idx]
            spec = hand._semantic_map.get(ch)
            if spec is not None:
                return hand.joint_names[spec["to"]]
        return None
    for hi, spec in hand._reverse_map.items():
        if spec["from"] == pos_idx:
            return hand.joint_names[hi]
    return None


def _o20_style_joint_space(space: str) -> bool:
    """semantic_o20 / hardware / 物理空间的 joints 数字按历史 O20 下标理解。"""
    return str(space).lower() not in ("semantic_v3",)


def _default_thumb_hold(space: str) -> list:
    return [0, 5, 10, 15] if _o20_style_joint_space(space) else [0, 1, 4, 3]


def _default_thumb_flex(space: str) -> list:
    return [0, 15] if _o20_style_joint_space(space) else [0, 3]


def _default_thumb_rot(space: str) -> int:
    return 10 if _o20_style_joint_space(space) else 4


def _default_prog_abd(space: str) -> int:
    return 5 if _o20_style_joint_space(space) else 1


def _default_prog_flex(space: str) -> int:
    return 15 if _o20_style_joint_space(space) else 3


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
    if not isinstance(data, dict):
        raise ValueError(f"手势配置根节点须为 mapping: {path} (got {type(data).__name__})")
    gestures = data.get("gestures") or {}
    if not isinstance(gestures, dict):
        raise ValueError(f"gestures 段须为 mapping: {path}")
    return gestures


def _semantic_from_sparse_o20(sparse: Mapping[str, Any]) -> List[float]:
    """稀疏 joint 名 / O20 索引 → O20 20 维（内部辅助；不做 S30 投影）。"""
    out = [0.0] * 20
    explicitly_set: set[int] = set()
    for key, val in sparse.items():
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            idx = int(key)
        else:
            idx = _SEMANTIC_JOINT_NAME_TO_IDX.get(str(key))
        if idx is None or idx < 0 or idx >= 20:
            continue
        out[idx] = float(val)
        explicitly_set.add(idx)
    for i in (6, 7, 8, 9):
        if i not in explicitly_set and out[i] == 0.0:
            out[i] = float(ABD_NEUTRAL)
    for i in RESERVED_INDICES:
        out[i] = 0.0
    return out


def _semantic_from_sparse(sparse: Mapping[str, Any]) -> List[float]:
    """稀疏 joint 名 / O20 索引 / S30 通道名 → 30 维 semantic 向量。

    兼容 O20 名（如 thumb_base）和 S30 名（如 thumb_mcp_flex）。
    保留 CURRENT 的 explicitly_set abd：YAML 显式写 0 的 abd 槽不填中性。
    """
    out = [0.0] * SEMANTIC_DIM
    explicitly_set: set[int] = set()
    has_s30_key = False
    o20_entries: dict[int, float] = {}
    for key, val in sparse.items():
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            idx = int(key)
            if 0 <= idx < 20:
                o20_entries[idx] = float(val)
            elif 0 <= idx < SEMANTIC_DIM:
                out[idx] = float(val)
                explicitly_set.add(idx)
                has_s30_key = True
            continue
        skey = str(key)
        s30_idx = SEMANTIC_30_IDX.get(skey)
        if s30_idx is not None:
            out[s30_idx] = float(val)
            explicitly_set.add(s30_idx)
            has_s30_key = True
            continue
        o20_idx = _SEMANTIC_JOINT_NAME_TO_IDX.get(skey)
        if o20_idx is not None and 0 <= o20_idx < 20:
            o20_entries[o20_idx] = float(val)
    if o20_entries:
        vec20 = [0.0] * 20
        for idx, val in o20_entries.items():
            vec20[idx] = val
        projected = project_o20_to_s30(vec20)
        for o20_idx in o20_entries:
            s30 = o20_joint_index_to_s30(o20_idx)
            if s30 >= 0:
                explicitly_set.add(s30)
        for i in range(SEMANTIC_DIM):
            if projected[i] != 0.0 and (not has_s30_key or out[i] == 0.0):
                out[i] = projected[i]
    # 仅对未显式给出的 abd 槽填中性；显式 0 须保留（侧捏张开位）
    for i in ABD_INDICES_S30:
        if i not in explicitly_set and out[i] == 0.0:
            out[i] = float(ABD_NEUTRAL)
    return out


def hw_pose_to_semantic(
    hw_by_name: Mapping[str, float],
    hand: HandConfig,
) -> List[float]:
    """硬件关节名 → S30 semantic 30 维。"""
    hw = [hand._reserved_default] * hand.num_joints
    for i, name in enumerate(hand.joint_names):
        if name in hw_by_name:
            hw[i] = float(hw_by_name[name])
    if hand.has_from_semantic:
        return hand.to_semantic(hw)
    o20 = hand.from_hardware(hw)
    return project_o20_to_s30(o20)


@dataclass(frozen=True)
class ThumbAdductionParams:
    """PhasedLerpEngine 手型相关参数（已规范为 semantic_30 30 维）。"""

    hand_type: str
    prep_angles: List[float]
    close_angles: List[float]
    phase1: float = 0.35
    phase2: float = 0.40
    phase3: float = 0.35
    phase4: float = 0.65
    thumb_hold_joints: List[int] = field(default_factory=lambda: [0, 1, 4, 3])
    thumb_flex_joints: List[int] = field(default_factory=lambda: [0, 3])
    thumb_rot_joint: int = 4
    close_move_joints: List[int] = field(default_factory=list)
    progressive_abd_joint: int = 1
    progressive_flex_joint: int = 3
    progressive_abd_max: float = 185.0
    progressive_flex_max: float = 150.0
    progressive_rate: float = 20.0
    contact_fingers: List[int] = field(default_factory=lambda: [0, 1])
    position_stop_joint: Optional[int] = None
    position_stop_target_semantic: Optional[float] = None
    position_stop_tolerance: float = 5.0
    # by_vision 扩展
    adaptive_fn: Optional[Callable] = field(default=None, compare=False, hash=False)
    close_from_adaptive: bool = False
    progressive_all_joints: Optional[List[int]] = None
    progressive_all_maxs: Optional[List[float]] = None
    # P4a/P4b 拆分：拇指关节延后到 P4b 闭合
    close_thumb_joints: List[int] = field(default_factory=list)
    phase4_split: float = 0.5
    engine: str = ""  # "" / "phased_lerp" = 旧引擎, "rot_progressive" = 新引擎

    @property
    def prep_end(self) -> float:
        return self.phase1 + self.phase2 + self.phase3

    def close_flex_indices(self) -> List[int]:
        if self.close_move_joints:
            return list(self.close_move_joints)
        return list(self.thumb_flex_joints)


_PHYSICAL_ANGLE_SPACES = frozenset({"radians", "rad", "degrees", "deg"})


def _ensure_import_loader_space(angles_sec: dict, loader_name: str) -> None:
    """import 期公开 loader 支持 semantic_v3 / semantic_o20 / hardware。

    ``degrees``/``radians`` 需 URDF Converter，请用 ``load_*_params_for_ctx`` 在运行时加载。
    """
    space = str((angles_sec or {}).get("space", "semantic_v3")).lower()
    if space in _PHYSICAL_ANGLE_SPACES:
        raise ValueError(
            f"{loader_name}: angles.space={space!r} requires Ros2ControlRadianConverter "
            f"at load time. Use load_static_params_for_ctx / load_thumb_adduction_params_for_ctx "
            f"with PrimitiveContext at runtime, or keep space: semantic_v3 / semantic_o20 in YAML."
        )


def _parse_angles(section: dict, hand: HandConfig, radian_converter=None) -> tuple:
    """返回 (prep_angles, close_angles) legacy semantic 30 维 (0～255)。"""
    space = str(section.get("space", "semantic_v3")).lower()
    if space in _PHYSICAL_ANGLE_SPACES:
        if radian_converter is None:
            raise ValueError(
                f"angles.space={space!r} requires a Ros2ControlRadianConverter "
                "(pass radian_converter= when loading params, or use load_*_params_for_ctx)"
            )
        prep_map = section.get("prep") or section.get("target") or {}
        close_map = section.get("close") or {}
        prep = _sparse_physical_to_legacy255(prep_map, space, radian_converter)
        if close_map:
            merged = dict(prep_map)
            merged.update(close_map)
            close = _sparse_physical_to_legacy255(merged, space, radian_converter)
        else:
            close = list(prep)
        return prep, close
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
        if space == "semantic_o20":
            prep_o20 = _semantic_from_sparse_o20(target)
            merged = {**{JOINT_NAMES[i]: prep_o20[i] for i in range(20)}, **close_section}
        else:
            merged = {SEMANTIC_30_NAMES[i]: prep[i] for i in range(SEMANTIC_DIM)}
            merged.update(close_section)
        close = _semantic_from_sparse(merged)
    else:
        close = list(prep)
    return prep, close


def _sparse_physical_to_legacy255(
    sparse: Mapping[str, Any],
    space: str,
    radian_converter,
) -> List[float]:
    """radians/degrees 稀疏图 → legacy semantic 255（需 Ros2ControlRadianConverter）。

    T1 过渡：原语/缓存仍消费 legacy 255；YAML 写 rad/deg 时在加载边界换算。
    """
    if radian_converter is None:
        raise ValueError(
            f"angles.space={space!r} requires a Ros2ControlRadianConverter "
            "(pass radian_converter= when loading params)"
        )
    space_l = str(space).lower()
    raw20 = _semantic_from_sparse_physical(sparse, space_l)
    return radian_converter.semantic_rad_to_legacy255(raw20)


def _semantic_from_sparse_physical(sparse: Mapping[str, Any], space: str) -> List[float]:
    """radians/degrees 稀疏 → 30 维 semantic_rad（未指定槽为 0；不做 255 中性 abd）。"""
    from . import units as U

    # 先按名/索引写入，再经 as_semantic_30 统一到 30 维
    raw_sparse: dict = {}
    for key, val in sparse.items():
        v = float(val)
        if space in ("degrees", "deg"):
            v = U.deg_to_rad(v)
        raw_sparse[key] = v
    # 物理空间不做 abd 中性填充：用临时解析后清零未显式 abd
    out = [0.0] * SEMANTIC_DIM
    explicitly_set: set[int] = set()
    o20_entries: dict[int, float] = {}
    for key, val in raw_sparse.items():
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            idx = int(key)
            if 0 <= idx < 20:
                o20_entries[idx] = float(val)
            elif 0 <= idx < SEMANTIC_DIM:
                out[idx] = float(val)
                explicitly_set.add(idx)
            continue
        skey = str(key)
        s30_idx = SEMANTIC_30_IDX.get(skey)
        if s30_idx is not None:
            out[s30_idx] = float(val)
            explicitly_set.add(s30_idx)
            continue
        o20_idx = _SEMANTIC_JOINT_NAME_TO_IDX.get(skey)
        if o20_idx is not None and 0 <= o20_idx < 20:
            o20_entries[o20_idx] = float(val)
    if o20_entries:
        vec20 = [0.0] * 20
        for idx, val in o20_entries.items():
            vec20[idx] = val
        projected = project_o20_to_s30(vec20)
        for i in range(SEMANTIC_DIM):
            if projected[i] != 0.0 and out[i] == 0.0:
                out[i] = projected[i]
    return out


def _parse_thumb_adduction(
    raw: dict,
    hand_type: str,
    hand: HandConfig,
    radian_converter=None,
) -> ThumbAdductionParams:
    cfg = _deep_merge(_DEFAULT_THUMB_ADDUCTION, raw)
    angles_sec = cfg.get("angles") or {}
    prep, close = _parse_angles(angles_sec, hand, radian_converter=radian_converter)
    motion = cfg.get("motion") or {}
    joints = cfg.get("joints") or {}
    prog = cfg.get("progressive") or {}
    contact = cfg.get("contact") or {}
    close_sec = cfg.get("close") or {}

    # YAML joints/progressive：semantic_o20 / hardware 为 O20 索引需转 S30；semantic_v3 直用
    space = str(angles_sec.get("space", "semantic_v3")).lower()
    need_idx_translate = _o20_style_joint_space(space)

    def _idx(v) -> int:
        i = int(v)
        if need_idx_translate and i >= 0:
            return o20_joint_index_to_s30(i)
        return i

    rot = joints.get("rot", _default_thumb_rot(space))
    rot_s30 = _idx(rot) if rot is not None and int(rot) >= 0 else -1
    pos_joint = contact.get("position_joint")
    pos_idx = None
    pos_target = None
    if pos_joint is not None:
        if isinstance(pos_joint, int):
            pos_idx = _idx(pos_joint)
        elif isinstance(pos_joint, str):
            if pos_joint.isdigit():
                pos_idx = _idx(int(pos_joint))
            elif pos_joint in SEMANTIC_30_IDX:
                pos_idx = SEMANTIC_30_IDX[pos_joint]
            elif pos_joint in hand.joint_names:
                hw_idx = hand.joint_names.index(pos_joint)
                if space == "semantic_v3" and hand.has_from_semantic:
                    pos_idx = _s30_index_for_hw_joint(hand, hw_idx)
                else:
                    rev = hand._reverse_map.get(hw_idx)
                    if rev:
                        pos_idx = _idx(rev["from"])
            elif pos_joint in _SEMANTIC_JOINT_NAME_TO_IDX:
                o20_i = _SEMANTIC_JOINT_NAME_TO_IDX[pos_joint]
                pos_idx = o20_joint_index_to_s30(o20_i)
        if contact.get("position_target_hw") is not None and pos_idx is not None:
            hw_name = _hw_name_for_pose_idx(hand, pos_idx, space=space)
            if hw_name:
                pos_target = hw_pose_to_semantic(
                    {hw_name: float(contact["position_target_hw"])},
                    hand,
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
        thumb_hold_joints=[_idx(x) for x in joints.get("hold", _default_thumb_hold(space))],
        thumb_flex_joints=[_idx(x) for x in joints.get("flex", _default_thumb_flex(space))],
        thumb_rot_joint=rot_s30,
        close_move_joints=[_idx(x) for x in close_sec.get("move_only", [])],
        progressive_abd_joint=_idx(prog.get("abd", _default_prog_abd(space))),
        progressive_flex_joint=_idx(prog.get("flex", _default_prog_flex(space))),
        progressive_abd_max=float(prog.get("abd_max", 185)),
        progressive_flex_max=float(prog.get("flex_max", 150)),
        progressive_rate=float(prog.get("rate", 20.0)),
        contact_fingers=[int(x) for x in contact.get("fingers", [0, 1])],
        position_stop_joint=pos_idx,
        position_stop_target_semantic=pos_target,
        position_stop_tolerance=float(contact.get("position_tolerance", 5.0)),
        close_from_adaptive=bool(close_sec.get("from_adaptive", False)),
        progressive_all_joints=(
            [_idx(x) for x in prog["all_joints"]] if "all_joints" in prog else None
        ),
        progressive_all_maxs=(
            [float(x) for x in prog["all_maxs"]] if "all_maxs" in prog else None
        ),
        close_thumb_joints=[_idx(x) for x in joints.get("close_thumb", [])],
        phase4_split=float(motion.get("phase4_split", 0.5)),
        engine=str(raw.get("engine", "")),
    )


@lru_cache(maxsize=16)
def _cached_load_thumb_adduction_params(hand_type: str) -> ThumbAdductionParams:
    hand_type = hand_type.lower()
    hand = HandConfig(hand_type)
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get("thumb_adduction_grip") or {}
    _ensure_import_loader_space(raw.get("angles") or {}, "load_thumb_adduction_params")
    return _parse_thumb_adduction(raw, hand_type, hand)


def _copy_thumb_adduction_params(params: ThumbAdductionParams) -> ThumbAdductionParams:
    """返回深拷贝列表字段，避免污染 lru_cache。"""
    from dataclasses import replace

    return replace(
        params,
        prep_angles=list(params.prep_angles),
        close_angles=list(params.close_angles),
        thumb_hold_joints=list(params.thumb_hold_joints),
        thumb_flex_joints=list(params.thumb_flex_joints),
        close_move_joints=list(params.close_move_joints),
        contact_fingers=list(params.contact_fingers),
        progressive_all_joints=(
            list(params.progressive_all_joints)
            if params.progressive_all_joints is not None
            else None
        ),
        progressive_all_maxs=(
            list(params.progressive_all_maxs) if params.progressive_all_maxs is not None else None
        ),
        close_thumb_joints=list(params.close_thumb_joints),
    )


def load_thumb_adduction_params(hand_type: str) -> ThumbAdductionParams:
    """加载指定手型的 thumb_adduction_grip 参数（带缓存，legacy semantic 0～255）。"""
    return _copy_thumb_adduction_params(_cached_load_thumb_adduction_params(hand_type.lower()))


load_thumb_adduction_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_thumb_adduction_params.cache_clear
)


def load_thumb_adduction_params_for_ctx(ctx) -> ThumbAdductionParams:
    """运行时加载；有 Converter 时姿态/渐进上限为 semantic_rad。

    YAML ``angles.space: degrees|radians`` 仅在本路径（运行时 + Converter）支持。
    """
    converter = getattr(ctx, "radian_converter", None)
    hand_type = ctx.hand_type.lower()
    if converter is not None:
        hand = HandConfig(hand_type)
        raw = _load_raw_gestures(hand_type).get("thumb_adduction_grip") or {}
        angles_sec = raw.get("angles") or {}
        space = str(angles_sec.get("space", "semantic_v3")).lower()
        if space in _PHYSICAL_ANGLE_SPACES:
            params = _parse_thumb_adduction(raw, hand_type, hand, radian_converter=converter)
            return thumb_adduction_params_as_semantic_rad(params, converter)
    params = load_thumb_adduction_params(hand_type)
    if converter is not None:
        return thumb_adduction_params_as_semantic_rad(params, converter)
    return params


# ---------------------------------------------------------------------------
# index_middle_adduction_grip — 双关节渐进并拢参数
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FingerAdductionParams:
    """index_middle_adduction_grip 参数（已规范为 semantic_30 30 维）。"""

    hand_type: str
    prep_angles: List[float]  # P1 到位目标 (abd 在张开位)
    joint_a: int = 6  # 渐进关节A (S30 index_mcp_abd)
    joint_b: int = 11  # 渐进关节B (S30 middle_mcp_abd)
    close_a: float = 96.0  # A 并拢目标
    close_b: float = 160.0  # B 并拢目标
    phase1: float = 0.35  # P1 到位时长 (秒)
    close_duration: float = 1.0  # 并拢时长 (秒)
    thumb_joints: List[int] = field(default_factory=list)  # 拇指先行关节
    thumb_duration: float = 0.0  # 拇指先行时长 (秒)
    contact_fingers: List[int] = field(default_factory=lambda: [1, 2])
    engine: str = ""  # "" = 旧硬编码逻辑, "finger_adduction" = 新引擎


@lru_cache(maxsize=16)
def _cached_load_finger_adduction_params(hand_type: str) -> FingerAdductionParams:
    hand_type = hand_type.lower()
    hand = HandConfig(hand_type)
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get("index_middle_adduction_grip") or {}
    return _parse_finger_adduction(raw, hand_type, hand)


def _copy_finger_adduction_params(params: FingerAdductionParams) -> FingerAdductionParams:
    """返回深拷贝列表字段，避免污染 lru_cache。"""
    from dataclasses import replace

    return replace(
        params,
        prep_angles=list(params.prep_angles),
        thumb_joints=list(params.thumb_joints),
        contact_fingers=list(params.contact_fingers),
    )


def load_finger_adduction_params(hand_type: str) -> FingerAdductionParams:
    """加载指定手型的 index_middle_adduction_grip 参数（带缓存）。"""
    return _copy_finger_adduction_params(_cached_load_finger_adduction_params(hand_type.lower()))


load_finger_adduction_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_finger_adduction_params.cache_clear
)


def _parse_finger_adduction(raw: dict, hand_type: str, hand: HandConfig) -> FingerAdductionParams:
    angles_sec = raw.get("angles") or {}
    prep, _close = _parse_angles(angles_sec, hand)
    motion = raw.get("motion") or {}
    adduction = raw.get("adduction") or {}
    contact = raw.get("contact") or {}

    space = str(angles_sec.get("space", "semantic_v3")).lower()
    need_tr = _o20_style_joint_space(space)

    def _jidx(v) -> int:
        i = int(v)
        if need_tr and i >= 0:
            return o20_joint_index_to_s30(i)
        return i

    thumb = raw.get("thumb") or {}

    return FingerAdductionParams(
        hand_type=hand_type,
        prep_angles=prep,
        joint_a=_jidx(adduction.get("joint_a", 6)),
        joint_b=_jidx(adduction.get("joint_b", 7 if need_tr else 11)),
        close_a=float(adduction.get("close_a", 96)),
        close_b=float(adduction.get("close_b", 160)),
        phase1=float(motion.get("phase1", 0.35)),
        close_duration=float(adduction.get("duration", 1.0)),
        thumb_joints=[_jidx(x) for x in thumb.get("joints", [])],
        thumb_duration=float(thumb.get("duration", 0.0)),
        contact_fingers=[int(x) for x in contact.get("fingers", [1, 2])],
        engine=str(raw.get("engine", "")),
    )


def clear_gesture_params_cache() -> None:
    from .tripod_force import clear_tripod_force_params_cache

    load_thumb_adduction_params.cache_clear()
    load_finger_adduction_params.cache_clear()
    load_two_stage_speed_params.cache_clear()
    load_static_gesture_params.cache_clear()
    load_tripod_stage_durations.cache_clear()
    load_large_wrap_params.cache_clear()
    clear_tripod_force_params_cache()
    load_sequential_force_close_params.cache_clear()
    load_by_vision_phased_params.cache_clear()


# ---------------------------------------------------------------------------
# 通用两阶段限速姿态原语
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TwoStageSpeedParams:
    """两阶段姿态参数（角度为 semantic_30 30 维，速度为位置单位/秒）。"""

    hand_type: str
    primitive: str
    initial_angles: List[float]
    target_angles: List[float]
    initial_speeds: List[float]
    speeds: List[float]
    ready_tolerance: float = 5.0
    ready_frames: int = 3
    ready_timeout: float = 2.0


def _hardware_vector(raw: Any, hand: HandConfig, field_name: str) -> List[float]:
    """读取按硬件关节顺序给出的数组或 joint-name 映射。"""
    if isinstance(raw, Mapping):
        missing = [name for name in hand.joint_names if name not in raw]
        if missing:
            raise ValueError(f"{field_name} 缺少硬件关节: {missing}")
        return [float(raw[name]) for name in hand.joint_names]

    if not isinstance(raw, (list, tuple)) or len(raw) != hand.num_joints:
        raise ValueError(f"{field_name} 必须包含 {hand.num_joints} 个硬件关节值，收到: {raw!r}")
    return [float(value) for value in raw]


def _hardware_speeds_to_semantic(
    hw_speeds: List[float],
    hand: HandConfig,
) -> List[float]:
    """把硬件关节速度映射到原语使用的 semantic S30 索引。"""
    if hand.has_from_semantic:
        s30 = [0.0] * SEMANTIC_DIM
        for hw_idx, speed in enumerate(hw_speeds):
            for ch, _invert in hand._semantic_reverse.get(hw_idx, []):
                s30[SEMANTIC_30_IDX[ch]] = float(speed)
        return s30
    o20_speeds = [0.0] * 20
    for hw_idx, speed in enumerate(hw_speeds):
        reverse = hand._reverse_map.get(hw_idx)
        if reverse is not None:
            o20_speeds[reverse["from"]] = float(speed)
    return project_o20_to_s30(o20_speeds)


@lru_cache(maxsize=32)
def _cached_load_two_stage_speed_params(
    hand_type: str,
    primitive: str,
) -> TwoStageSpeedParams:
    """加载 ``angles.initial → angles.target`` 两阶段限速姿态参数。"""
    hand_type = hand_type.lower()
    hand = HandConfig(hand_type)
    raw = _load_raw_gestures(hand_type).get(primitive) or {}
    if not raw:
        raise ValueError(f"{hand_type} 未配置两阶段原语 {primitive!r}")

    angles = raw.get("angles") or {}
    space = str(angles.get("space", "hardware")).lower()
    if space != "hardware":
        raise ValueError(f"{primitive}.angles.space 仅支持 hardware，收到: {space!r}")
    motion = raw.get("motion") or {}
    motion_type = str(motion.get("type", "")).lower()
    if motion_type != "two_stage_speed":
        raise ValueError(f"{primitive}.motion.type 必须为 two_stage_speed，收到: {motion_type!r}")

    initial_hw = _hardware_vector(angles.get("initial"), hand, f"{primitive}.angles.initial")
    target_hw = _hardware_vector(angles.get("target"), hand, f"{primitive}.angles.target")
    speeds_hw = _hardware_vector(
        motion.get("speeds"),
        hand,
        f"{primitive}.motion.speeds",
    )
    initial_speed = float(motion.get("initial_speed", 200.0))
    ready_tolerance = float(motion.get("ready_tolerance", 5.0))
    ready_frames = int(motion.get("ready_frames", 3))
    ready_timeout = float(motion.get("ready_timeout", 2.0))
    if initial_speed <= 0.0:
        raise ValueError(f"{primitive}.motion.initial_speed 必须大于 0")
    if any(speed <= 0.0 for speed in speeds_hw):
        raise ValueError(f"{primitive}.motion.speeds 必须全部大于 0")
    if ready_tolerance <= 0.0:
        raise ValueError(f"{primitive}.motion.ready_tolerance 必须大于 0")
    if ready_frames < 1:
        raise ValueError(f"{primitive}.motion.ready_frames 必须至少为 1")
    if ready_timeout <= 0.0:
        raise ValueError(f"{primitive}.motion.ready_timeout 必须大于 0")

    if hand.has_from_semantic:
        initial_angles = hand.to_semantic(initial_hw)
        target_angles = hand.to_semantic(target_hw)
    else:
        initial_angles = project_o20_to_s30(hand.from_hardware(initial_hw))
        target_angles = project_o20_to_s30(hand.from_hardware(target_hw))

    return TwoStageSpeedParams(
        hand_type=hand_type,
        primitive=primitive,
        initial_angles=initial_angles,
        target_angles=target_angles,
        initial_speeds=_hardware_speeds_to_semantic([initial_speed] * hand.num_joints, hand),
        speeds=_hardware_speeds_to_semantic(speeds_hw, hand),
        ready_tolerance=ready_tolerance,
        ready_frames=ready_frames,
        ready_timeout=ready_timeout,
    )


def _copy_two_stage_speed_params(params: TwoStageSpeedParams) -> TwoStageSpeedParams:
    """返回深拷贝列表字段，避免污染 lru_cache。"""
    from dataclasses import replace

    return replace(
        params,
        initial_angles=list(params.initial_angles),
        target_angles=list(params.target_angles),
        initial_speeds=list(params.initial_speeds),
        speeds=list(params.speeds),
    )


def load_two_stage_speed_params(
    hand_type: str,
    primitive: str,
) -> TwoStageSpeedParams:
    return _copy_two_stage_speed_params(
        _cached_load_two_stage_speed_params(hand_type.lower(), primitive.lower())
    )


load_two_stage_speed_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_two_stage_speed_params.cache_clear
)


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
    pause_duration: float = 0.0
    feedback_source: str = ""
    feedback_delta: float = 30.0
    phase1_hold: List[int] = field(default_factory=lambda: [0, 1, 4, 3])
    relax_angles: List[float] = field(default_factory=list)
    grip_angles: List[float] = field(default_factory=list)
    hold_duration: float = 0.3
    grip_duration: float = 0.4
    relax_phased: Optional["StaticGestureParams"] = None
    grip_phased: Optional["StaticGestureParams"] = None


@dataclass(frozen=True)
class LargeWrapParams:
    """O6/L6 large_wrap 两阶段姿态配置。"""

    hand_type: str
    prep_angles: List[float]
    close_angles: List[float]
    prep_duration: float
    pause_duration: float
    close_duration: float
    thumb_delay: float = 0.2


def _phased_motion_duration(params: StaticGestureParams) -> float:
    p3 = params.phase3 if params.phase3 > 0 else 0.0
    return params.phase1 + params.phase2 + p3 + params.phase4


def _resolve_relax_grip_sub(
    section: Any,
    hand_type: str,
    hand: HandConfig,
    default_ref: str,
    radian_converter=None,
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
            "relax_grip",
            hand_type,
            hand,
            section["angles"],
            radian_converter=radian_converter,
        )
    else:
        angles = list(ref_params.target_angles)

    sub_space = str((section.get("angles") or {}).get("space", "")).lower()
    # semantic_v3 子段 joints 已是 S30；其余（含 hardware / 空）按历史 O20 下标翻译
    phased = _parse_phased_sub_params(
        section,
        hand_type,
        hand,
        angles,
        f"relax_grip_{ref}",
        idx_translate=_o20_style_joint_space(sub_space) if sub_space else True,
    )
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
    idx_translate: bool = True,
) -> Optional[StaticGestureParams]:
    """relax_grip 子段 phased_pose 配置（如 O6 拇延后）。"""
    sub_motion = section.get("motion") or {}
    if str(sub_motion.get("type", "")).lower() != "phased_pose":
        return None
    joints = section.get("joints") or {}
    if idx_translate:
        rot = joints.get("rot", 10)
        hold = joints.get("hold", [0, 5, 10, 15])
    else:
        rot = joints.get("rot", 4)
        hold = joints.get("hold", [0, 1, 4, 3])

    def _jidx(v) -> int:
        i = int(v)
        if idx_translate and i >= 0:
            return o20_joint_index_to_s30(i)
        return i

    rot_s30 = _jidx(rot) if rot is not None and int(rot) >= 0 else -1
    if rot_s30 < 0:
        hold = [int(x) for x in hold if _jidx(int(x)) != 4]
    return StaticGestureParams(
        hand_type=hand_type,
        primitive=primitive,
        motion_type="phased_pose",
        target_angles=list(target_angles),
        phase1=float(sub_motion.get("phase1", 0.5)),
        phase2=float(sub_motion.get("phase2", 0.4)),
        phase3=float(sub_motion.get("phase3", 0.4)),
        phase4=float(sub_motion.get("phase4", 0.4)),
        phase1_hold=[_jidx(x) for x in hold],
    )


def _open_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 0,
            "index_base": 0,
            "middle_base": 0,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 0,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 0,
            "thumb_tip": 0,
            "index_tip": 0,
            "middle_tip": 0,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _init_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 80,
            "index_base": 80,
            "middle_base": 80,
            "ring_base": 80,
            "pinky_base": 80,
            "thumb_abd": 60,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 50,
            "thumb_tip": 60,
            "index_tip": 60,
            "middle_tip": 60,
            "ring_tip": 60,
            "pinky_tip": 60,
        }
    )


def _fist_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 60,
            "index_base": 255,
            "middle_base": 255,
            "ring_base": 255,
            "pinky_base": 255,
            "thumb_abd": 200,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 230,
            "thumb_tip": 255,
            "index_tip": 255,
            "middle_tip": 255,
            "ring_tip": 255,
            "pinky_tip": 255,
        }
    )


def _v_sign_o20() -> List[float]:
    out = _semantic_from_sparse(
        {
            "thumb_base": 200,
            "index_base": 0,
            "middle_base": 0,
            "ring_base": 255,
            "pinky_base": 255,
            "thumb_abd": 140,
            "index_abd": 255,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 150,
            "thumb_tip": 255,
            "index_tip": 0,
            "middle_tip": 0,
            "ring_tip": 255,
            "pinky_tip": 255,
        }
    )
    out[11] = 0.0  # middle_mcp_abd (S30 idx 11): V 形张开（0 为有效值，非 abd 默认）
    return out


def _ok_sign_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 0,
            "index_base": 150,
            "middle_base": 100,
            "ring_base": 70,
            "pinky_base": 40,
            "thumb_abd": 170,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 220,
            "thumb_tip": 200,
            "index_tip": 100,
            "middle_tip": 100,
            "ring_tip": 70,
            "pinky_tip": 40,
        }
    )


def _ok_sign_l25() -> List[float]:
    """L25 OK 环：拇/食深弯 + 适度拇指旋转（勿用 O20 220 语义，亦勿 thumb_base 过开）。"""
    return _semantic_from_sparse(
        {
            "thumb_base": 150,
            "index_base": 255,
            "middle_base": 100,
            "ring_base": 70,
            "pinky_base": 40,
            "thumb_abd": 140,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": _L25_OK_SIGN_THUMB_ROT,
            "thumb_tip": 210,
            "index_tip": 200,
            "middle_tip": 100,
            "ring_tip": 70,
            "pinky_tip": 40,
        }
    )


def _point_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 230,
            "index_base": 0,
            "middle_base": 255,
            "ring_base": 255,
            "pinky_base": 255,
            "thumb_abd": 140,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 150,
            "thumb_tip": 200,
            "index_tip": 0,
            "middle_tip": 255,
            "ring_tip": 255,
            "pinky_tip": 255,
        }
    )


def _pinch_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 180,
            "index_base": 180,
            "middle_base": 0,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 130,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 200,
            "thumb_tip": 180,
            "index_tip": 180,
            "middle_tip": 0,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


# L25 ``thumb_rot`` (O20 slot 10) 经 invert → hw[10]：255=自然位，0=最大旋转。
# 勿照搬 O20 对指姿常用的 185~200 语义值（会落到 hw≈55~70，几乎拧到底）。
# 对指需比 init(hw≈205) 更旋转，但应留在 hw≈100~130；95 过小无法对指，192 过大。
_L25_PINCH_THUMB_ROT = 140.0
_L25_OK_SIGN_THUMB_ROT = 150.0


def _pinch_l25() -> List[float]:
    """Legacy BotClaw L25 three-finger pinch in O20 semantic space.

    BotClaw's calibrated left-hand pose is expressed in the L25 SDK's active
    16-joint, 0..100 percent space.  These values reproduce that pose after
    scaling to controller-native 0..255 and applying the L25 25-joint layout.
    """
    return _semantic_from_sparse(
        {
            "thumb_base": 128,
            "index_base": 255,
            "middle_base": 255,
            "ring_base": 252,
            "pinky_base": 223,
            "thumb_abd": 115,
            "index_abd": 125,
            "middle_abd": 123,
            "ring_abd": 114,
            "pinky_abd": 77,
            "thumb_rot": _L25_PINCH_THUMB_ROT,
            "thumb_tip": 100,
            "index_tip": 96,
            "middle_tip": 96,
            "ring_tip": 154,
            "pinky_tip": 178,
        }
    )


def _grasp_ready_l25() -> List[float]:
    """Legacy BotClaw L25 half-wrap pose in O20 semantic space.

    Unlike the generic ``init`` pose, this keeps the thumb, index, and middle
    finger aperture open while curling the ring and pinky clear of a support
    surface. Values come from the calibrated ``hand_poses_l25.json`` posture.
    """
    return _semantic_from_sparse(
        {
            "thumb_base": 193.035,
            "index_base": 129.03,
            "middle_base": 192.0,
            "ring_base": 157.08,
            "pinky_base": 155.04,
            "thumb_abd": 116.025,
            "index_abd": 126.99,
            "middle_abd": 128.01,
            "ring_abd": 119.085,
            "pinky_abd": 80.07,
            "thumb_rot": 77.01,
            "thumb_tip": 25.5,
            "index_tip": 84.915,
            "middle_tip": 129.0,
            "ring_tip": 150.96,
            "pinky_tip": 175.95,
        }
    )


def _index_pinch_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 110,
            "index_base": 140,
            "middle_base": 0,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 140,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 200,
            "thumb_tip": 160,
            "index_tip": 130,
            "middle_tip": 0,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _index_pinch_l25() -> List[float]:
    """L25 拇食对捏：食指须深弯(255)；非接触指伸开；拇指旋转适度。"""
    return _semantic_from_sparse(
        {
            "thumb_base": 128,
            "index_base": 255,
            "middle_base": 0,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 115,
            "index_abd": 125,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": _L25_PINCH_THUMB_ROT,
            "thumb_tip": 100,
            "index_tip": 96,
            "middle_tip": 0,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _middle_pinch_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 120,
            "index_base": 0,
            "middle_base": 140,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 190,
            "index_abd": 200,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 185,
            "thumb_tip": 160,
            "index_tip": 0,
            "middle_tip": 130,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _middle_pinch_l25() -> List[float]:
    """L25 拇中对捏：中指深弯(255)；食指伸开避让；与 ``_index_pinch_l25`` 对称。"""
    return _semantic_from_sparse(
        {
            "thumb_base": 128,
            "index_base": 0,
            "middle_base": 255,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 140,
            "index_abd": 200,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": _L25_PINCH_THUMB_ROT,
            "thumb_tip": 100,
            "index_tip": 0,
            "middle_tip": 96,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _tripod_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 139,
            "index_base": 184,
            "middle_base": 163,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 227,
            "index_abd": 193,
            "middle_abd": 255,
            "ring_abd": 105,
            "pinky_abd": 52,
            "thumb_rot": 110,
            "thumb_tip": 143,
            "index_tip": 180,
            "middle_tip": 80,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _tripod_l25() -> List[float]:
    """L25 三指捏：食/中根+尖对称深弯；勿复用 O20（index_tip 180 vs middle 80 在 invert 下严重偏指）。"""
    return _semantic_from_sparse(
        {
            "thumb_base": 128,
            "index_base": 230,
            "middle_base": 230,
            "ring_base": 0,
            "pinky_base": 0,
            "thumb_abd": 115,
            "index_abd": 125,
            "middle_abd": 125,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": _L25_PINCH_THUMB_ROT,
            "thumb_tip": 100,
            "index_tip": 120,
            "middle_tip": 120,
            "ring_tip": 0,
            "pinky_tip": 0,
        }
    )


def _relax_o20() -> List[float]:
    return list(_open_o20())


def _grip_o20() -> List[float]:
    return _semantic_from_sparse(
        {
            "thumb_base": 255,
            "index_base": 255,
            "middle_base": 255,
            "ring_base": 255,
            "pinky_base": 255,
            "thumb_abd": 140,
            "index_abd": ABD_NEUTRAL,
            "middle_abd": ABD_NEUTRAL,
            "ring_abd": ABD_NEUTRAL,
            "pinky_abd": ABD_NEUTRAL,
            "thumb_rot": 160,
            "thumb_tip": 255,
            "index_tip": 255,
            "middle_tip": 255,
            "ring_tip": 255,
            "pinky_tip": 255,
        }
    )


# L25 专用 25-DOF（与 open_hand.py / init_hand.py 一致）
_L25_OPEN = [
    159,
    0,
    0,
    0,
    0,
    105,
    141,
    104,
    66,
    0,
    75,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
]


_DEFAULT_STATIC: Dict[str, Dict[str, Any]] = {
    "open": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "init": {
        "motion": {"type": "static", "duration": 0.8},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "fist": {
        "motion": {
            "type": "phased_pose",
            "phase1": 0.5,
            "phase2": 0.4,
            "phase3": 0.4,
            "phase4": 0.4,
        },
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "release": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "relax_grip": {
        "motion": {
            "type": "relax_grip",
            "duration": 0.6,
            "hold_duration": 0.3,
            "grip_duration": 0.4,
        },
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "pinch": {
        "motion": {"type": "static", "duration": 0.5},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "grasp_ready": {
        "motion": {"type": "static", "duration": 0.8},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "index_pinch": {
        "motion": {"type": "static", "duration": 0.5},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "middle_pinch": {
        "motion": {"type": "static", "duration": 0.5},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "tripod": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "point": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "ok_sign": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_v3", "target": {}},
    },
    "v_sign": {
        "motion": {"type": "static", "duration": 0.6},
        "angles": {"space": "semantic_v3", "target": {}},
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
    ("l6", "open"): _open_o20,
    ("l6", "init"): _init_o20,
    ("l6", "fist"): _fist_o20,
    ("l6", "release"): _open_o20,
    ("l6", "relax_grip"): _fist_o20,
    ("l25", "open"): _open_o20,
    ("l25", "init"): _init_o20,
    ("l25", "fist"): _fist_o20,
    ("l25", "release"): _open_o20,
    ("l25", "relax_grip"): _fist_o20,
    ("l25", "pinch"): _pinch_l25,
    ("l25", "grasp_ready"): _grasp_ready_l25,
    ("l25", "index_pinch"): _index_pinch_l25,
    ("l25", "index_pinch_by_vision"): _index_pinch_l25,
    ("l25", "middle_pinch"): _middle_pinch_l25,
    ("l25", "middle_pinch_by_vision"): _middle_pinch_l25,
    ("l25", "tripod"): _tripod_l25,
    ("l25", "tripod_by_vision"): _tripod_l25,
    ("l25", "point"): _point_o20,
    ("l25", "ok_sign"): _ok_sign_l25,
    ("l25", "v_sign"): _v_sign_o20,
    ("o20", "pinch"): _pinch_o20,
    ("o20", "index_pinch"): _index_pinch_o20,
    ("o20", "middle_pinch"): _middle_pinch_o20,
    ("o20", "tripod"): _tripod_o20,
    ("o20", "point"): _point_o20,
    ("o20", "ok_sign"): _ok_sign_o20,
    ("o20", "v_sign"): _v_sign_o20,
    ("g20", "open"): _open_o20,
    ("g20", "init"): _init_o20,
    ("g20", "fist"): _fist_o20,
    ("g20", "release"): _open_o20,
    ("g20", "relax_grip"): _fist_o20,
    ("g20", "pinch"): _pinch_o20,
    ("g20", "index_pinch"): _index_pinch_o20,
    ("g20", "middle_pinch"): _middle_pinch_o20,
    ("g20", "tripod"): _tripod_o20,
    ("g20", "point"): _point_o20,
    ("g20", "ok_sign"): _ok_sign_o20,
    ("g20", "v_sign"): _v_sign_o20,
}


def _resolve_static_target(
    primitive: str,
    hand_type: str,
    hand: HandConfig,
    angles_sec: dict,
    radian_converter=None,
) -> List[float]:
    space = str(angles_sec.get("space", "semantic_v3")).lower()
    target_map = angles_sec.get("target") or angles_sec.get("prep") or {}
    if target_map:
        if space in ("radians", "rad", "degrees", "deg"):
            return _sparse_physical_to_legacy255(target_map, space, radian_converter)
        if space == "hardware":
            return hw_pose_to_semantic(target_map, hand)
        return _semantic_from_sparse(target_map)
    builder = _STATIC_TARGET_BUILDERS.get((hand_type, primitive))
    if builder:
        return builder()
    # O30 等独立 PIP 手型：禁止静默回退 O20/open（O20 builder 不含 pip 通道）
    if hand_type != "o20" and hand.channel_enabled("index_pip"):
        raise ValueError(
            f"Gesture '{primitive}' for hand '{hand_type}' has no YAML target; "
            f"add gestures.{primitive} to config/{hand_type}.yaml"
        )
    # fallback: 尝试 O20 builder（L25/O6 等价映射到 O20 语义空间）
    if hand_type != "o20":
        builder = _STATIC_TARGET_BUILDERS.get(("o20", primitive))
        if builder:
            return builder()
    return _open_o20()


def _parse_static_gesture(
    primitive: str,
    raw: dict,
    hand_type: str,
    hand: HandConfig,
    radian_converter=None,
) -> StaticGestureParams:
    base = _DEFAULT_STATIC.get(primitive, _DEFAULT_STATIC["open"])
    cfg = _deep_merge(base, raw)
    motion = cfg.get("motion") or {}
    contact = cfg.get("contact") or {}
    joints = cfg.get("joints") or {}
    angles_sec = cfg.get("angles") or {}
    motion_type = str(motion.get("type", "static")).lower()
    target_map = angles_sec.get("target") or angles_sec.get("prep") or {}

    # release / relax_grip 常只有 motion、无 angles.target。
    # O30 等独立 PIP 手禁止空 target 静默回退 O20 builder，须先落到本手 open/fist。
    if primitive == "release" and not target_map:
        target = list(load_static_gesture_params(hand_type, "open").target_angles)
    elif motion_type == "relax_grip" and not target_map:
        # 占位；真正姿态由下方 relax→open / grip→fist 解析
        target = list(load_static_gesture_params(hand_type, "open").target_angles)
    else:
        target = _resolve_static_target(
            primitive, hand_type, hand, angles_sec, radian_converter=radian_converter
        )

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
            relax_sec, hand_type, hand, "open", radian_converter=radian_converter
        )
        grip, grip_phased, grip_dur = _resolve_relax_grip_sub(
            grip_sec, hand_type, hand, "fist", radian_converter=radian_converter
        )
        duration = max(duration, relax_dur)
        grip_duration = max(grip_duration, grip_dur)

    space = str(angles_sec.get("space", "semantic_v3")).lower()
    need_idx_tr = _o20_style_joint_space(space)
    rot = joints.get("rot", _default_thumb_rot(space))
    hold = joints.get("hold", _default_thumb_hold(space))

    def _jidx(v) -> int:
        i = int(v)
        if need_idx_tr and i >= 0:
            return o20_joint_index_to_s30(i)
        return i

    rot_s30 = _jidx(rot) if rot is not None and int(rot) >= 0 else -1
    if rot_s30 < 0:
        hold = [int(x) for x in hold if _jidx(int(x)) != 4]

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
        pause_duration=float(motion.get("pause_duration", 0.0)),
        feedback_source=str(contact.get("feedback_source", "")).lower(),
        feedback_delta=float(contact.get("feedback_delta", 30.0)),
        phase1_hold=[_jidx(x) for x in hold],
        relax_angles=relax,
        grip_angles=grip,
        hold_duration=float(motion.get("hold_duration", 0.3)),
        grip_duration=grip_duration,
        relax_phased=relax_phased,
        grip_phased=grip_phased,
    )


@lru_cache(maxsize=64)
def _cached_load_static_gesture_params(hand_type: str, primitive: str) -> StaticGestureParams:
    hand = HandConfig(hand_type)
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get(primitive) or {}
    _ensure_import_loader_space((raw.get("angles") or {}), "load_static_gesture_params")
    return _parse_static_gesture(primitive, raw, hand_type, hand)


def _copy_static_gesture_params(params: StaticGestureParams) -> StaticGestureParams:
    """返回深拷贝列表字段，避免污染 lru_cache。"""
    from dataclasses import replace

    relax_phased = (
        _copy_static_gesture_params(params.relax_phased)
        if params.relax_phased is not None
        else None
    )
    grip_phased = (
        _copy_static_gesture_params(params.grip_phased) if params.grip_phased is not None else None
    )
    return replace(
        params,
        target_angles=list(params.target_angles),
        phase1_hold=list(params.phase1_hold),
        relax_angles=list(params.relax_angles),
        grip_angles=list(params.grip_angles),
        relax_phased=relax_phased,
        grip_phased=grip_phased,
    )


def load_static_gesture_params(hand_type: str, primitive: str) -> StaticGestureParams:
    """加载 open/init/fist/release/relax_grip 静态姿态配置。

    返回值中的角度向量仍为 legacy semantic 0～255。若需 semantic_rad，
    见 ``static_params_as_semantic_rad`` 或 ``load_static_params_for_ctx``。

    import 期不支持 YAML ``angles.space: degrees|radians``（无 URDF Converter）。
    """
    hand_type = hand_type.lower()
    primitive = primitive.lower()
    return _copy_static_gesture_params(_cached_load_static_gesture_params(hand_type, primitive))


load_static_gesture_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_static_gesture_params.cache_clear
)


def static_params_as_semantic_rad(
    params: StaticGestureParams,
    radian_converter,
) -> StaticGestureParams:
    """把 StaticGestureParams 内姿态向量从 legacy 255 转为 semantic_rad。

    ``feedback_delta`` 等电流/力阈值保持原单位不换算。
    """
    from dataclasses import replace

    to_rad = radian_converter.legacy255_to_semantic_rad
    relax_phased = (
        static_params_as_semantic_rad(params.relax_phased, radian_converter)
        if params.relax_phased is not None
        else None
    )
    grip_phased = (
        static_params_as_semantic_rad(params.grip_phased, radian_converter)
        if params.grip_phased is not None
        else None
    )
    return replace(
        params,
        target_angles=to_rad(as_semantic_30(params.target_angles)),
        relax_angles=to_rad(as_semantic_30(params.relax_angles)) if params.relax_angles else [],
        grip_angles=to_rad(as_semantic_30(params.grip_angles)) if params.grip_angles else [],
        relax_phased=relax_phased,
        grip_phased=grip_phased,
    )


def load_static_params_for_ctx(hand_type: str, primitive: str, ctx) -> StaticGestureParams:
    """加载静态 YAML；有 Converter 时目标已是 semantic_rad，否则 legacy255。

    YAML ``angles.space: degrees|radians`` 仅在本路径（运行时 + Converter）支持。
    """
    converter = getattr(ctx, "radian_converter", None)
    hand_type_l = hand_type.lower()
    primitive_l = primitive.lower()
    if converter is not None:
        hand = HandConfig(hand_type_l)
        gestures = _load_raw_gestures(hand_type_l)
        raw = gestures.get(primitive_l) or {}
        angles_sec = raw.get("angles") or {}
        space = str(angles_sec.get("space", "semantic_v3")).lower()
        if space in _PHYSICAL_ANGLE_SPACES:
            params = _parse_static_gesture(
                primitive_l,
                raw,
                hand_type_l,
                hand,
                radian_converter=converter,
            )
            return static_params_as_semantic_rad(params, converter)
    params = load_static_gesture_params(hand_type, primitive)
    if converter is not None:
        return static_params_as_semantic_rad(params, converter)
    return params


def static_yaml_legacy255_or_none(hand_type: str, primitive: str) -> Optional[List[float]]:
    """L25 static YAML 目标（legacy255）；全零/缺失时返回 None，供硬编码回退。"""
    params = load_static_gesture_params(hand_type, primitive)
    if any(abs(float(v)) > 1e-9 for v in params.target_angles):
        return list(params.target_angles)
    return None


def large_wrap_params_as_semantic_rad(
    params: LargeWrapParams,
    radian_converter,
) -> LargeWrapParams:
    """large_wrap prep/close 姿态 legacy255 → semantic_rad。"""
    from dataclasses import replace

    to_rad = radian_converter.legacy255_to_semantic_rad
    return replace(
        params,
        prep_angles=to_rad(as_semantic_30(params.prep_angles)),
        close_angles=to_rad(as_semantic_30(params.close_angles)),
    )


def two_stage_speed_params_as_semantic_rad(
    params: TwoStageSpeedParams,
    radian_converter,
) -> TwoStageSpeedParams:
    """两阶段限速姿态/速度/到位容差 legacy255 → semantic_rad。"""
    from dataclasses import replace

    from .angle_bridge import step255_to_rad_at_slot

    to_rad = radian_converter.legacy255_to_semantic_rad
    initial_speeds = [
        step255_to_rad_at_slot(radian_converter, i, s) for i, s in enumerate(params.initial_speeds)
    ]
    speeds = [step255_to_rad_at_slot(radian_converter, i, s) for i, s in enumerate(params.speeds)]
    tol_idx = next((i for i, s in enumerate(params.speeds) if s > 0.0), 0)
    ready_tol = step255_to_rad_at_slot(radian_converter, tol_idx, params.ready_tolerance)
    return replace(
        params,
        initial_angles=to_rad(as_semantic_30(params.initial_angles)),
        target_angles=to_rad(as_semantic_30(params.target_angles)),
        initial_speeds=as_semantic_30(initial_speeds),
        speeds=as_semantic_30(speeds),
        ready_tolerance=ready_tol,
    )


def finger_adduction_params_as_semantic_rad(
    params: FingerAdductionParams,
    radian_converter,
) -> FingerAdductionParams:
    """食中指侧向夹持 YAML 姿态/并拢目标 legacy255 → semantic_rad。"""
    from dataclasses import replace

    from .angle_bridge import value255_to_rad_at_slot

    to_rad = radian_converter.legacy255_to_semantic_rad
    return replace(
        params,
        prep_angles=to_rad(as_semantic_30(params.prep_angles)),
        close_a=value255_to_rad_at_slot(radian_converter, int(params.joint_a), params.close_a),
        close_b=value255_to_rad_at_slot(radian_converter, int(params.joint_b), params.close_b),
    )


def sequential_force_close_params_as_semantic_rad(
    params: "SequentialForceCloseParams",
    radian_converter,
) -> "SequentialForceCloseParams":
    """prep 姿态与 step/close_max 从 legacy255 转到 semantic_rad。"""
    from dataclasses import replace

    from .angle_bridge import step255_to_rad_at_slot, value255_to_rad_at_slot

    prep = radian_converter.legacy255_to_semantic_rad(as_semantic_30(params.prep_angles))
    new_steps: List[ForceCloseStep] = []
    for step in params.steps:
        new_groups: List[ForceCloseGroup] = []
        for g in step.groups:
            idx = int(g.joint_indices[0]) if g.joint_indices else 0
            new_groups.append(
                replace(
                    g,
                    step=step255_to_rad_at_slot(radian_converter, idx, g.step),
                    close_max=value255_to_rad_at_slot(radian_converter, idx, g.close_max),
                )
            )
        new_steps.append(replace(step, groups=new_groups))
    return replace(params, prep_angles=prep, steps=new_steps)


def thumb_adduction_params_as_semantic_rad(
    params: ThumbAdductionParams,
    radian_converter,
) -> ThumbAdductionParams:
    """PhasedLerp 姿态/渐进上限/速率 → semantic_rad。

    ``rot_progressive`` 下 YAML 常写 ``abd/flex: -1``，此时 ``abd_max``/``rate``
    实际作用在 ``thumb_rot``，须用 rot 槽限位换算，不能退化到 (0, π/2)。
    """
    from dataclasses import replace

    from .angle_bridge import step255_to_rad_at_slot, value255_to_rad_at_slot

    def to_pose(v):
        return radian_converter.legacy255_to_semantic_rad(as_semantic_30(v))

    abd_j = int(params.progressive_abd_joint)
    flex_j = int(params.progressive_flex_joint)
    rot_j = (
        int(params.thumb_rot_joint)
        if int(params.thumb_rot_joint) >= 0
        else SEMANTIC_30_IDX["thumb_rot"]
    )

    def _slot(joint: int, fallback: int) -> int:
        return int(joint) if int(joint) >= 0 else int(fallback)

    abd_slot = _slot(abd_j, rot_j)
    flex_slot = _slot(flex_j, abd_slot)
    # 速率：双渐进关闭时跟 rot；否则保持历史「跟 flex（或 abd）」行为
    if abd_j < 0 and flex_j < 0:
        rate_slot = rot_j
    elif flex_j >= 0:
        rate_slot = flex_slot
    else:
        rate_slot = abd_slot

    stop_tol = params.position_stop_tolerance
    stop_target = params.position_stop_target_semantic
    stop_j = params.position_stop_joint
    if stop_j is not None and stop_target is not None:
        stop_target = value255_to_rad_at_slot(radian_converter, int(stop_j), float(stop_target))
        stop_tol = step255_to_rad_at_slot(radian_converter, int(stop_j), float(stop_tol))

    all_maxs = None
    if params.progressive_all_maxs and params.progressive_all_joints:
        all_maxs = [
            value255_to_rad_at_slot(radian_converter, int(j), float(m))
            for j, m in zip(params.progressive_all_joints, params.progressive_all_maxs)
        ]

    return replace(
        params,
        prep_angles=to_pose(params.prep_angles),
        close_angles=to_pose(params.close_angles),
        progressive_abd_max=value255_to_rad_at_slot(
            radian_converter, abd_slot, params.progressive_abd_max
        ),
        progressive_flex_max=value255_to_rad_at_slot(
            radian_converter, flex_slot, params.progressive_flex_max
        ),
        progressive_rate=step255_to_rad_at_slot(
            radian_converter, rate_slot, params.progressive_rate
        ),
        position_stop_target_semantic=stop_target,
        position_stop_tolerance=float(stop_tol),
        progressive_all_maxs=all_maxs,
    )


@lru_cache(maxsize=32)
def load_tripod_stage_durations(
    hand_type: str, primitive: str = "tripod"
) -> tuple[float, float, float]:
    """返回 (prep, close, pause) 秒。

    L6 ``tripod_two_stage`` 用 motion.phase1/phase2/pause_duration。
    其它手型读 ``stages.prep/close/pause``，缺省 0.5/0.5/0。
    ``tripod_by_force`` 未写 stages 时回退到同手型 ``tripod``。
    """
    hand_type = hand_type.lower()
    primitive = primitive.lower()
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get(primitive) or {}
    motion = raw.get("motion") if isinstance(raw.get("motion"), dict) else {}
    stages = raw.get("stages") if isinstance(raw.get("stages"), dict) else {}
    if (
        primitive != "tripod"
        and str(motion.get("type", "")).lower() != "tripod_two_stage"
        and "prep" not in stages
    ):
        raw = gestures.get("tripod") or raw
        motion = raw.get("motion") if isinstance(raw.get("motion"), dict) else {}
        stages = raw.get("stages") if isinstance(raw.get("stages"), dict) else {}
    if str(motion.get("type", "")).lower() == "tripod_two_stage":
        return (
            float(motion.get("phase1", 0.5)),
            float(motion.get("phase2", 0.5)),
            float(motion.get("pause_duration", 0.0)),
        )
    return (
        float(stages.get("prep", motion.get("prep_duration", 0.5))),
        float(stages.get("close", motion.get("close_duration", 0.5))),
        float(stages.get("pause", motion.get("pause_duration", 0.0))),
    )


@lru_cache(maxsize=8)
def _cached_load_large_wrap_params(hand_type: str) -> LargeWrapParams:
    """加载 O6/L6 large_wrap 的 prep/close 姿态与 auto 时序。"""
    hand_type = hand_type.lower()
    hand = HandConfig(hand_type)
    raw = _load_raw_gestures(hand_type).get("large_wrap") or {}
    if not raw:
        raise ValueError(f"{hand_type} 未配置原语 'large_wrap'")

    motion = raw.get("motion") or {}
    motion_type = str(motion.get("type", "")).lower()
    if motion_type != "large_wrap_two_stage":
        raise ValueError(
            f"large_wrap.motion.type 必须为 large_wrap_two_stage，收到: {motion_type!r}"
        )

    prep, close = _parse_angles(raw.get("angles") or {}, hand)
    prep_duration = float(motion.get("phase1", 0.5))
    pause_duration = float(motion.get("pause_duration", 2.5))
    close_duration = float(motion.get("phase2", 0.5))
    thumb_delay = float(motion.get("thumb_delay", 0.2))
    if prep_duration < 0.0 or pause_duration < 0.0 or close_duration < 0.0:
        raise ValueError("large_wrap 的阶段时长不能为负数")
    if thumb_delay < 0.0 or thumb_delay >= close_duration:
        raise ValueError(f"thumb_delay ({thumb_delay}) 必须在 [0, close_duration) 范围内")

    return LargeWrapParams(
        hand_type=hand_type,
        prep_angles=prep,
        close_angles=close,
        prep_duration=prep_duration,
        pause_duration=pause_duration,
        close_duration=close_duration,
        thumb_delay=thumb_delay,
    )


def _copy_large_wrap_params(params: LargeWrapParams) -> LargeWrapParams:
    """返回深拷贝列表字段，避免污染 lru_cache。"""
    from dataclasses import replace

    return replace(
        params,
        prep_angles=list(params.prep_angles),
        close_angles=list(params.close_angles),
    )


def load_large_wrap_params(hand_type: str) -> LargeWrapParams:
    return _copy_large_wrap_params(_cached_load_large_wrap_params(hand_type.lower()))


load_large_wrap_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_large_wrap_params.cache_clear
)


def resolve_static_target_angles(
    hand_type: str,
    primitive: str,
    default: List[float],
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
    step_cfg: dict,
    hand: HandConfig,
    default_delta: float,
    default_step: float,
) -> List[ForceCloseGroup]:
    from .contact_detection import tactile_finger_for_group
    from .contact_resolver import hw_indices, monitor_indices_for_s30, semantic_indices

    groups_raw = step_cfg.get("groups") or []
    if not groups_raw:
        joints = step_cfg.get("joints") or []
        monitor = step_cfg.get("monitor") or step_cfg.get("monitor_joints") or []
        groups_raw = [
            {
                "id": step_cfg.get("name", "group"),
                "joints": joints,
                "monitor": monitor,
            }
        ]
    delta = float(step_cfg.get("contact_delta", default_delta))
    # O6 力矩反馈：步骤级可覆盖 contact_delta_torque_pct。
    # L6 关节数同为 6，但 uses_torque_feedback 仅为 O6；勿把 torque% 当成 mA。
    # L6 力控运行时以触觉为主（capabilities.current=false）。
    from .contact_detection import uses_torque_feedback

    if uses_torque_feedback(hand.hand_joint):
        delta = float(
            step_cfg.get(
                "contact_delta_torque_pct",
                step_cfg.get("contact_delta", default_delta),
            )
        )
    step = float(step_cfg.get("step", default_step))
    out: List[ForceCloseGroup] = []
    for g in groups_raw:
        joint_names = [str(x) for x in g.get("joints", [])]
        monitor_names = [str(x) for x in g.get("monitor", g.get("monitor_joints", []))]
        sem = semantic_indices(joint_names)  # S30 通道下标
        mon = hw_indices(hand, monitor_names) if monitor_names else []
        if not mon and sem:
            for si in sem:
                mon.extend(monitor_indices_for_s30(hand, si))
        out.append(
            ForceCloseGroup(
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
            )
        )
    return out


def _parse_sequential_force_close(
    raw: dict,
    hand_type: str,
    hand: HandConfig,
    primitive: str,
) -> SequentialForceCloseParams:
    angles_sec = raw.get("angles") or {}
    prep, _ = _parse_angles(angles_sec, hand)
    motion = raw.get("motion") or {}
    contact = raw.get("contact") or {}
    from .contact_detection import uses_torque_feedback

    if uses_torque_feedback(hand_type):
        default_delta = float(
            contact.get("contact_delta_torque_pct", contact.get("contact_delta", 15.0))
        )
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
                        {
                            "id": "middle",
                            "joints": ["middle_base"],
                            "monitor": ["middle_mcp_pitch"],
                        },
                    ],
                },
                {
                    "name": "close_thumb",
                    "step": float(contact.get("thumb_step", default_step)),
                    "contact_delta": default_delta,
                    "groups": [
                        {"id": "thumb", "joints": ["thumb_base"], "monitor": ["thumb_cmc_pitch"]},
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
                        {"id": "index", "joints": ["index_base"], "monitor": ["index_mcp_pitch"]},
                        {
                            "id": "middle",
                            "joints": ["middle_base"],
                            "monitor": ["middle_mcp_pitch"],
                        },
                    ],
                },
                {
                    "name": "close_thumb",
                    "step": float(contact.get("thumb_step", default_step)),
                    "contact_delta": default_delta,
                    "groups": [
                        {"id": "thumb", "joints": ["thumb_base"], "monitor": ["thumb_cmc_pitch"]},
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
def _cached_load_sequential_force_close_params(
    hand_type: str,
    primitive: str,
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


def _copy_force_close_group(group: ForceCloseGroup) -> ForceCloseGroup:
    from dataclasses import replace

    return replace(
        group,
        joint_indices=list(group.joint_indices),
        monitor_hw=list(group.monitor_hw),
    )


def _copy_sequential_force_close_params(
    params: SequentialForceCloseParams,
) -> SequentialForceCloseParams:
    """返回深拷贝列表字段，避免污染 lru_cache。"""
    from dataclasses import replace

    return replace(
        params,
        prep_angles=list(params.prep_angles),
        steps=[
            ForceCloseStep(
                name=step.name,
                groups=[_copy_force_close_group(g) for g in step.groups],
            )
            for step in params.steps
        ],
    )


def load_sequential_force_close_params(
    hand_type: str,
    primitive: str,
) -> Optional[SequentialForceCloseParams]:
    params = _cached_load_sequential_force_close_params(hand_type.lower(), primitive.lower())
    return _copy_sequential_force_close_params(params) if params is not None else None


load_sequential_force_close_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_sequential_force_close_params.cache_clear
)


def load_ring_params(hand_type: str) -> Optional[SequentialForceCloseParams]:
    """加载 ring 的 sequential_force_close 配置；无配置或非该类型返回 None。"""
    return load_sequential_force_close_params(hand_type, "ring")


def load_middle_ring_params(hand_type: str) -> Optional[SequentialForceCloseParams]:
    """加载 middle_ring 的 sequential_force_close 配置；无配置或非该类型返回 None。"""
    return load_sequential_force_close_params(hand_type, "middle_ring")


def _load_by_vision_phased_params(
    hand_type: str, primitive: str
) -> Optional[ThumbAdductionParams]:
    """通用: 加载 by_vision 原语的 PhasedLerp 参数；SFCE/static/无配置时返回 None。"""
    hand_type = hand_type.lower()
    gestures = _load_raw_gestures(hand_type)
    raw = gestures.get(primitive)
    if not raw or not raw.get("motion"):
        return None
    motion = raw.get("motion") or {}
    if motion.get("type") in ("sequential_force_close", "static"):
        return None
    hand = HandConfig(hand_type)
    return _parse_thumb_adduction(raw, hand_type, hand)


@lru_cache(maxsize=64)
def _cached_load_by_vision_phased_params(
    hand_type: str,
    primitive: str,
) -> Optional[ThumbAdductionParams]:
    return _load_by_vision_phased_params(hand_type, primitive)


def load_by_vision_phased_params(
    hand_type: str,
    primitive: str,
) -> Optional[ThumbAdductionParams]:
    params = _cached_load_by_vision_phased_params(hand_type.lower(), primitive.lower())
    return _copy_thumb_adduction_params(params) if params is not None else None


load_by_vision_phased_params.cache_clear = (  # type: ignore[attr-defined]
    _cached_load_by_vision_phased_params.cache_clear
)


def load_tripod_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 tripod_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "tripod_by_vision")


def load_hook_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 hook_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "hook_by_vision")


def load_small_warp_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 small_warp_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "small_warp_by_vision")


def load_no_index_warp_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 no_index_warp_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "no_index_warp_by_vision")


def load_palmar_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 palmar_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "palmar_by_vision")


def load_index_ring_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 index_ring_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "index_ring_by_vision")


def load_large_wrap_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 large_wrap_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "large_wrap_by_vision")


def load_middle_ring_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 middle_ring_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "middle_ring_by_vision")


def load_ring_by_vision_params(hand_type: str) -> Optional[ThumbAdductionParams]:
    """加载 ring_by_vision 的 PhasedLerp 参数；无配置返回 None。"""
    return load_by_vision_phased_params(hand_type, "ring_by_vision")
