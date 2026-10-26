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
CAP_FINGER_ABD = "finger_abd"  # 食/中/无/小侧摆
CAP_THUMB_ABD = "thumb_abd"
CAP_THUMB_ROT = "thumb_rot"
CAP_FINGER_TIPS = "finger_tips"  # 四指独立指尖
CAP_THUMB_TIP = "thumb_tip"
CAP_FINGER_ROOT2 = "finger_root2"  # L25 指中/拇指 root2
CAP_VISION_FK = "vision_fk"  # GraspGate + FK 指尖
CAP_TOUCH = "touch"  # 压感反馈
CAP_CURRENT = "current"  # 电流接触检测 (O20/L25)
CAP_MOTOR_TORQUE = "motor_torque"  # 关节力矩 sense (O6, 0~100%)

ALL_CAPABILITIES = frozenset(
    {
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
    }
)

# 力控反馈可替代：电流 / 力矩 / 触觉（L6 当前无真实电流时走触觉）
_FORCE_FEEDBACK_ALTERNATIVES = {
    CAP_CURRENT: frozenset({CAP_MOTOR_TORQUE, CAP_TOUCH}),
    CAP_MOTOR_TORQUE: frozenset({CAP_CURRENT, CAP_TOUCH}),
}


@dataclass(frozen=True)
class PrimitiveCatalogEntry:
    """单条原语的全局元数据。

    - ``gated``：须经 GraspGate 三条件（H/XY/Fwd）；与运行时门控集合同一真相源。
    - ``phased``：存在多阶段运动/力控行为（文档与能力分类）。
    - ``command_phased``：``gesture_cmd`` 支持 ``prep|close|auto|full`` 解析
      （即历史 ``PHASED_GRASP_PRIMITIVES``）；为 ``phased`` 的子集。
    """

    requires: FrozenSet[str] = frozenset()
    gated: bool = False
    phased: bool = False
    command_phased: bool = False
    category: str = "gesture"  # safe | gesture | vision | force


def _req(*caps: str) -> FrozenSet[str]:
    return frozenset(caps)


# Registry classes are discovered automatically. This catalog remains the
# explicit capability-policy layer and is validated against the registry when
# a hand profile starts.
PRIMITIVE_CATALOG: dict[str, PrimitiveCatalogEntry] = {
    # --- 安全 / 基础 ---
    "init": PrimitiveCatalogEntry(category="safe"),
    "open": PrimitiveCatalogEntry(category="safe"),
    "release": PrimitiveCatalogEntry(category="safe"),
    "relax_grip": PrimitiveCatalogEntry(category="safe"),
    "named_pose": PrimitiveCatalogEntry(category="safe"),
    # --- 静态手势 ---
    "fist": PrimitiveCatalogEntry(),
    "pinch": PrimitiveCatalogEntry(requires=_req(CAP_THUMB_ABD)),
    "point": PrimitiveCatalogEntry(requires=_req(CAP_THUMB_ABD)),
    "ok_sign": PrimitiveCatalogEntry(requires=_req(CAP_THUMB_ABD)),
    "v_sign": PrimitiveCatalogEntry(requires=_req(CAP_THUMB_ABD)),
    "grasp_ready": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_FINGER_ABD)
    ),
    "hammer_grip": PrimitiveCatalogEntry(phased=True, command_phased=True),
    "large_wrap": PrimitiveCatalogEntry(phased=True, command_phased=True),
    # --- 力控 / 侧向夹持 ---
    "thumb_adduction_grip": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        command_phased=True,
        requires=_req(CAP_THUMB_ABD, CAP_CURRENT),
    ),
    "index_middle_adduction_grip": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        command_phased=True,
        requires=_req(CAP_FINGER_ABD, CAP_CURRENT),
    ),
    # --- 非 vision 抓取 ---
    "tripod": PrimitiveCatalogEntry(
        phased=True,
        command_phased=True,
        requires=_req(CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_FINGER_ABD),
    ),
    "tripod_by_force": PrimitiveCatalogEntry(
        category="force",
        phased=True,
        command_phased=True,
        requires=_req(CAP_THUMB_ABD, CAP_CURRENT),
    ),
    # ring / middle_ring：内部多段力控，但不走 gesture_cmd prep/close 解析
    "ring": PrimitiveCatalogEntry(category="force", phased=True, requires=_req(CAP_CURRENT)),
    "middle_ring": PrimitiveCatalogEntry(
        category="force", phased=True, requires=_req(CAP_CURRENT)
    ),
    "index_pinch": PrimitiveCatalogEntry(
        requires=_req(CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_THUMB_ABD)
    ),
    "middle_pinch": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_TIPS, CAP_FINGER_ABD)),
    "parallel_extension": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_ABD)),
    "index_flick": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_TIPS)),
    "index_middle_flick": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_TIPS)),
    "twist2digit": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_TIPS, CAP_FINGER_ABD)),
    "twist3digit": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_TIPS, CAP_FINGER_ABD)),
    "twist5digit": PrimitiveCatalogEntry(requires=_req(CAP_FINGER_TIPS, CAP_FINGER_ABD)),
    # --- vision：gated=True 须过 GraspGate；下列二者按尺寸自适应，故意直通 ---
    "index_ring_by_vision": PrimitiveCatalogEntry(
        category="vision",
        gated=False,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT),
    ),
    "large_wrap_by_vision": PrimitiveCatalogEntry(
        category="vision",
        gated=False,
        requires=_req(CAP_VISION_FK, CAP_CURRENT),
    ),
    "middle_ring_by_vision": PrimitiveCatalogEntry(
        category="vision",
        gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT),
    ),
    "ring_by_vision": PrimitiveCatalogEntry(
        category="vision",
        gated=True,
        requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT),
    ),
    "small_warp_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_CURRENT)
    ),
    "no_index_warp_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_CURRENT)
    ),
    "hook_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_CURRENT)
    ),
    "index_pinch_by_vision": PrimitiveCatalogEntry(
        category="vision",
        gated=True,
        requires=_req(CAP_VISION_FK, CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_CURRENT),
    ),
    "middle_pinch_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_CURRENT)
    ),
    "tripod_by_vision": PrimitiveCatalogEntry(
        category="vision",
        gated=True,
        requires=_req(CAP_VISION_FK, CAP_THUMB_ROT, CAP_FINGER_TIPS, CAP_FINGER_ABD, CAP_CURRENT),
    ),
    "palmar_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_CURRENT)
    ),
    "parallel_extension_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_FINGER_ABD, CAP_CURRENT)
    ),
    "disk_by_vision": PrimitiveCatalogEntry(
        category="vision", gated=True, requires=_req(CAP_VISION_FK, CAP_FINGER_TIPS, CAP_CURRENT)
    ),
}


def catalog_entry(name: str) -> Optional[PrimitiveCatalogEntry]:
    return PRIMITIVE_CATALOG.get(name)


def gated_primitive_names() -> FrozenSet[str]:
    """须经 GraspGate 的原语名（单一真相源）。"""
    return frozenset(name for name, entry in PRIMITIVE_CATALOG.items() if entry.gated)


def command_phased_primitive_names() -> FrozenSet[str]:
    """gesture_cmd 支持 prep/close/auto/full 的原语名（单一真相源）。"""
    return frozenset(name for name, entry in PRIMITIVE_CATALOG.items() if entry.command_phased)


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
