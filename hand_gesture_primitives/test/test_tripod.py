from __future__ import annotations

from hand_gesture_primitives.gesture_params import load_tripod_stage_durations
from hand_gesture_primitives.primitive_base import PrimitiveContext, parse_tripod_phase
from hand_gesture_primitives.primitive_catalog import PRIMITIVE_CATALOG
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY


def _tripod(phase: str = "auto"):
    return PRIMITIVE_REGISTRY["tripod"](phase=phase)


def test_tripod_catalog_is_phased() -> None:
    assert PRIMITIVE_CATALOG["tripod"].phased


def test_tripod_default_is_auto() -> None:
    assert parse_tripod_phase("") == "auto"
    assert parse_tripod_phase("full") == "auto"
    assert parse_tripod_phase("auto") == "auto"
    assert parse_tripod_phase("prep") == "prep"
    primitive = _tripod()
    assert primitive.phase == "auto"


def test_tripod_prep_waits_for_close() -> None:
    primitive = _tripod(phase="prep")
    start = [40.0] * 20
    primitive.on_enter(start)
    result = primitive.compute(start, elapsed=2.0, ctx=PrimitiveContext(hand_type="l6"))
    assert result.feasible
    assert primitive.grasp_state == "ready"


def test_tripod_auto_chains_past_pause_on_l6() -> None:
    primitive = _tripod(phase="auto")
    assert primitive.phase == "auto"
    start = [40.0] * 20
    primitive.on_enter(start)
    # phase1 0.5 + pause 1.0 → close starts after 1.5s
    result = primitive.compute(start, elapsed=1.6, ctx=PrimitiveContext(hand_type="l6"))
    assert result.feasible
    assert primitive.grasp_state == "closing"
    result = primitive.compute(start, elapsed=2.5, ctx=PrimitiveContext(hand_type="l6"))
    assert result.feasible
    assert primitive.grasp_state == "holding"


def test_tripod_bare_equals_auto_on_l6() -> None:
    bare = _tripod()
    auto = _tripod(phase="auto")
    start = [40.0] * 20
    bare.on_enter(start)
    auto.on_enter(start)
    ctx = PrimitiveContext(hand_type="l6")
    assert (
        bare.compute(start, 1.6, ctx).target_angles == auto.compute(start, 1.6, ctx).target_angles
    )
    assert bare.grasp_state == auto.grasp_state == "closing"


def test_tripod_full_aliases_auto() -> None:
    assert _tripod(phase="full").phase == "auto"


def test_tripod_prep_close_on_dexterous_and_o6() -> None:
    start = [40.0] * 20
    for hand in ("o20", "l25", "g20", "o6"):
        prep = _tripod(phase="prep")
        prep.on_enter(start)
        result = prep.compute(start, elapsed=0.6, ctx=PrimitiveContext(hand_type=hand))
        assert result.feasible, hand
        assert result.hold_reason == "", hand
        assert prep.grasp_state == "ready", hand

        close = _tripod(phase="close")
        close.on_enter(start)
        result = close.compute(start, elapsed=0.6, ctx=PrimitiveContext(hand_type=hand))
        assert result.feasible, hand
        assert close.grasp_state == "holding", hand
        # 屈曲通道应离开起始值，朝 tripod 目标收
        assert result.target_angles[1] != start[1] or result.target_angles[2] != start[2], hand


def test_tripod_auto_on_l25_still_uses_engine() -> None:
    primitive = _tripod(phase="auto")
    primitive.on_enter([0.0] * 20)
    result = primitive.compute([0.0] * 20, elapsed=0.2, ctx=PrimitiveContext(hand_type="l25"))
    assert result.feasible
    assert "仅支持" not in result.hold_reason


def test_stage_durations_from_yaml() -> None:
    prep, close, pause = load_tripod_stage_durations("l25", "tripod")
    assert prep == 0.5
    assert close == 0.5
    assert pause == 0.0
    l6_prep, l6_close, l6_pause = load_tripod_stage_durations("l6", "tripod")
    assert l6_prep == 0.5
    assert l6_close == 0.5
    assert l6_pause == 1.0
