"""全局原语目录 — 能力需求与元数据（与具体手型无关）。

新手型只需在 config/{hand}.yaml 中声明 capabilities + primitives.supported；
原语能力需求在此集中维护，便于校验与文档生成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import FrozenSet, Optional

# ---------------------------------------------------------------------------
# 能力 token（手型 config capabilities 段使用相同字符串）
# ---------------------------------------------------------------------------
CAP_FINGER_ABD = "finger_abd"       # 食/中/无/小侧摆
CAP_THUMB_ABD = "thumb_abd"
CAP_THUMB_ROT = "thumb_rot"
CAP_FINGER_TIPS = "finger_tips"     # 四指独立指尖
CAP_THUMB_TIP = "thumb_tip"
CAP_FINGER_ROOT2 = "finger_root2"   # L25 指中/拇指 root2
CAP_VISION_FK = "vision_fk"         # GraspGate + FK 指尖
CAP_TOUCH = "touch"                 # 压感反馈
CAP_CURRENT = "current"             # 电流接触检测 (O20/L25)
CAP_MOTOR_TORQUE = "motor_torque"   # 关节力矩 sense (O6, 0~100%)

ALL_CAPABILITIES = frozenset({
    CAP_FINGER_ABD,
    CAP_THUMB_ABD,
    CAP_THUMB_ROT,
    CAP_FINGER_TIPS,
    CAP_THUMB_TIP,
    CAP_FINGER_ROOT2,
    CAP_VISION_FK,
    CAP_TOUCH,
    CAP_CURRENT,
    CAP_MOTOR_TORQUE,
})

# 力控原语可二选一：O20 电流 / O6 力矩
_FORCE_FEEDBACK_ALTERNATIVES = {
    CAP_CURRENT: frozenset({CAP_MOTOR_TORQUE}),
    CAP_MOTOR_TORQUE: frozenset({CAP_CURRENT}),
}


@dataclass(frozen=True)
class PrimitiveCatalogEntry:
    """单条原语的全局元数据。"""
    requires: FrozenSet[str] = frozenset()
    gated: bool = False
    phased: bool = False
    category: str = "gesture"  # safe | gesture | vision | force


def _req(*caps: str) -> FrozenSet[str]:
    return frozenset(caps)


# 与 primitives/__init__.py PRIMITIVE_REGISTRY 保持同步
PRIMITIVE_CATALOG: dict[str, PrimitiveCatalogEntry] = {
    # --- 安全 / 基础 ---
    "init": PrimitiveCatalogEntry(category="safe"),
    "open": PrimitiveCatalogEntry(category="safe"),
    "release": PrimitiveCatalogEntry(category="safe"),
    "relax_grip": PrimitiveCatalogEntry(category="safe"),
    # --- 静态手势 ---
    "fist": PrimitiveCatalogEntry(),
    "pinch": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ABD)),
    "point": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ABD)),
    "ok_sign": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ABD)),
    "v_sign": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ABD)),
    # --- 力控 / 侧向夹持 ---
    "thumb_adduction_grip": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        requires=_req(CAP_THUMB_ABD, CAP_CURRENT)),
    "index_middle_adduction_grip": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        requires=_req(CAP_FINGER_ABD, CAP_CURRENT)),
    # --- 非 vision 抓取 ---
    "tripod": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_FINGER_ABD)),
    "ring": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        requires=_req(CAP_CURRENT)),
    "middle_ring": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        requires=_req(CAP_CURRENT)),
    "index_pinch": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_THUMB_ABD)),
    "middle_pinch": PrimitiveCatalogEntry(
        requires=_req(CAP_FINGER_TIPS, CAP_FINGER_ABD)),
    "parallel_extension": PrimitiveCatalogEntry(
        requires=_req(CAP_FINGER_ABD)),
    # --- vision / 门控 ---
    "index_ring_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT)),
    "large_wrap_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_CURRENT)),
    "middle_ring_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT)),
    "ring_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT)),
    "small_warp_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_CURRENT)),
    "no_index_warp_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_CURRENT)),
    "hook_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_CURRENT)),
    "index_pinch_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_CURRENT)),
    "middle_pinch_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_CURRENT)),
    "tripod_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT)),
    "palmar_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_CURRENT)),
    "parallel_extension_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_ABD, CAP_CURRENT)),
    "disk_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_CURRENT)),
}


def catalog_entry(name: str) -> Optional[PrimitiveCatalogEntry]:
    return PRIMITIVE_CATALOG.get(name)


def missing_capabilities(
    name: str,
    hand_caps: FrozenSet[str],
) -> FrozenSet[str]:
    """返回手型缺失、导致原语不可运行的能力集合。"""
    entry = catalog_entry(name)
    if entry is None:
        return frozenset()
    missing = entry.requires - hand_caps
    resolved: set = set()
    for cap in missing:
        alts = _FORCE_FEEDBACK_ALTERNATIVES.get(cap)
        if alts and alts & hand_caps:
            continue
        resolved.add(cap)
    return frozenset(resolved)
