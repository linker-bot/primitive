"""Unit tests for hand-only side resolution helpers."""

from __future__ import annotations

from hand_gesture_primitives.launch_utils import (
    parse_gesture_sides,
    parse_hand_sides,
    resolve_hand_only_sides,
)


def test_parse_hand_sides_basic() -> None:
    assert parse_hand_sides("left") == ["left"]
    assert parse_hand_sides("right") == ["right"]
    assert parse_hand_sides("both") == ["left", "right"]
    assert parse_hand_sides("") == ["left"]
    assert parse_hand_sides("", default="right") == ["right"]
    assert parse_hand_sides("nope", default="right") == ["right"]


def test_resolve_ignores_gesture_for_hw() -> None:
    hw, gesture = resolve_hand_only_sides(
        "",
        gesture_side="right",
        default_side="left",
    )
    assert hw == ["right"]
    assert gesture == []


def test_resolve_hand_side_overrides_gesture_side() -> None:
    hw, gesture = resolve_hand_only_sides(
        "1",
        gesture_side="right",
        hand_side="left",
        default_side="right",
    )
    assert hw == ["left"]
    assert gesture == ["left"]


def test_resolve_l25_default_right() -> None:
    hw, gesture = resolve_hand_only_sides("", default_side="right")
    assert hw == ["right"]
    assert gesture == []
    hw2, gesture2 = resolve_hand_only_sides("1", default_side="right")
    assert hw2 == ["right"]
    assert gesture2 == ["right"]


def test_resolve_gesture_both_enables_both_hw() -> None:
    hw, gesture = resolve_hand_only_sides("both", default_side="left")
    assert hw == ["left", "right"]
    assert gesture == ["left", "right"]
    hw2, gesture2 = resolve_hand_only_sides(
        "both",
        gesture_side="",
        hand_side="right",
        default_side="left",
    )
    # 显式 HAND_SIDE 仍可单侧（gesture=both 时右节点可能缺硬件，属调用方约束）
    assert hw2 == ["right"]
    assert gesture2 == ["left", "right"]


def test_parse_gesture_sides_still_requires_enable() -> None:
    assert parse_gesture_sides("", "right") == []
    assert parse_gesture_sides("1", "right") == ["right"]
