"""能力矩阵单一真相源：catalog ↔ registry ↔ gate ↔ command_phased ↔ planable。"""

from __future__ import annotations

from hand_gesture_primitives.planable_primitives import PLANABLE_STATIC_PRIMITIVES
from hand_gesture_primitives.primitive_base import (
    PHASED_GRASP_PRIMITIVES,
    TRIPOD_PHASED_PRIMITIVES,
)
from hand_gesture_primitives.primitive_catalog import (
    PRIMITIVE_CATALOG,
    command_phased_primitive_names,
    gated_primitive_names,
)
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY

# 文档约定：按尺寸自适应，不过 GraspGate
_VISION_PASSTHROUGH = frozenset({"index_ring_by_vision", "large_wrap_by_vision"})


def test_catalog_covers_all_registered_primitives() -> None:
    missing = set(PRIMITIVE_REGISTRY) - set(PRIMITIVE_CATALOG)
    assert not missing, f"PRIMITIVE_CATALOG 缺少注册原语: {sorted(missing)}"


def test_catalog_has_no_orphan_entries() -> None:
    """catalog 允许 named_pose（动态）等无独立类文件的键；其余须在 registry。"""
    allowed_without_class = {"named_pose"}
    orphans = set(PRIMITIVE_CATALOG) - set(PRIMITIVE_REGISTRY) - allowed_without_class
    assert not orphans, f"catalog 有未注册条目: {sorted(orphans)}"


def test_gated_matches_catalog_flag() -> None:
    assert gated_primitive_names() == frozenset(
        name for name, entry in PRIMITIVE_CATALOG.items() if entry.gated
    )


def test_vision_passthrough_not_gated() -> None:
    for name in _VISION_PASSTHROUGH:
        assert name in PRIMITIVE_CATALOG, name
        assert PRIMITIVE_CATALOG[name].gated is False, f"{name} 应直通门控"
        assert name not in gated_primitive_names()


def test_gated_count_is_eleven() -> None:
    """与 README / interfaces 文档口径一致：11 个包络门控原语。"""
    gated = gated_primitive_names()
    assert len(gated) == 11, sorted(gated)
    assert gated.isdisjoint(_VISION_PASSTHROUGH)


def test_command_phased_matches_phased_grasp_export() -> None:
    assert command_phased_primitive_names() == PHASED_GRASP_PRIMITIVES
    assert PHASED_GRASP_PRIMITIVES == frozenset(
        {
            "thumb_adduction_grip",
            "index_middle_adduction_grip",
            "hammer_grip",
            "tripod",
            "tripod_by_force",
            "large_wrap",
        }
    )


def test_command_phased_implies_phased_flag() -> None:
    for name in command_phased_primitive_names():
        assert PRIMITIVE_CATALOG[name].phased, f"{name} command_phased 须 phased=True"


def test_tripod_phased_subset_of_command_phased() -> None:
    assert TRIPOD_PHASED_PRIMITIVES <= PHASED_GRASP_PRIMITIVES


def test_planable_subset_of_catalog() -> None:
    missing = PLANABLE_STATIC_PRIMITIVES - set(PRIMITIVE_CATALOG)
    assert not missing, f"planable 不在 catalog: {sorted(missing)}"


def test_planable_not_vision_or_force_category() -> None:
    for name in PLANABLE_STATIC_PRIMITIVES:
        cat = PRIMITIVE_CATALOG[name].category
        assert cat not in ("vision", "force"), f"{name} category={cat}"


def test_grasp_gate_exports_same_gated_set() -> None:
    """无 ROS 也可 import grasp_gate，且集合与 catalog 一致。"""
    from hand_gesture_primitives.grasp_gate import GATED_PRIMITIVES

    assert set(GATED_PRIMITIVES) == set(gated_primitive_names())
