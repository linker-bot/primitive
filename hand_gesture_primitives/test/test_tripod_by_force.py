from __future__ import annotations

import numpy as np
from hand_gesture_primitives.hand_profile import load_hand_profile
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitive_catalog import PRIMITIVE_CATALOG
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX
from hand_gesture_primitives.tripod_force import load_tripod_force_params


def _primitive(phase: str = "auto"):
    return PRIMITIVE_REGISTRY["tripod_by_force"](phase=phase)


def _currents(n: int, **spikes: float) -> list[float]:
    values = [10.0] * n
    for index, value in spikes.items():
        values[int(index)] = float(value)
    return values


def test_tripod_by_force_registered_and_phased() -> None:
    assert "tripod_by_force" in PRIMITIVE_REGISTRY
    assert PRIMITIVE_CATALOG["tripod_by_force"].phased


def test_tripod_by_force_allowlisted_on_all_configured_hands() -> None:
    for hand in ("l6", "o6", "o20", "l25", "g20"):
        profile = load_hand_profile(hand)
        assert profile.is_primitive_supported("tripod_by_force"), hand
        assert not profile.reject_reason("tripod_by_force").startswith(
            ("not_in_allowlist", "capability_missing", "unsupported_on_")
        ), (hand, profile.reject_reason("tripod_by_force"))


def test_force_params_map_semantic_flex_to_hardware() -> None:
    l6 = load_tripod_force_params("l6")
    assert [g.finger for g in l6.groups] == [0, 1, 2]
    assert l6.feedback_source == "motor_current"
    assert l6.require_feedback is True
    assert l6.feedback_delta == 30.0
    monitors = {g.finger: g.monitor_hw for g in l6.groups}
    assert monitors[0] == (0,)
    assert monitors[1] == (2,)
    assert monitors[2] == (3,)

    l25 = load_tripod_force_params("l25")
    assert l25.feedback_source == "auto"
    assert l25.require_feedback is False
    l25_monitors = {g.finger: g.monitor_hw for g in l25.groups}
    assert 1 in l25_monitors[1]

    g20 = load_tripod_force_params("g20")
    g20_monitors = {g.finger: g.monitor_hw for g in g20.groups}
    assert 12 in g20_monitors[1]  # index_tip → G20 hw[12]

    idx_flex = SEMANTIC_30_IDX["index_mcp_flex"]
    assert idx_flex in l6.groups[1].freeze_semantic


def test_l6_waits_for_current_before_close() -> None:
    primitive = _primitive()
    assert primitive.phase == "auto"
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=2.0,
        ctx=PrimitiveContext(hand_type="l6", joint_currents_valid=False),
    )
    assert not result.feasible
    assert "等待" in result.hold_reason
    assert primitive.grasp_state == "waiting_feedback"


def test_l6_prep_does_not_enter_force_close() -> None:
    primitive = _primitive(phase="prep")
    assert primitive.phase == "prep"
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=2.0,
        ctx=PrimitiveContext(hand_type="l6", joint_currents_valid=False),
    )
    assert result.feasible
    assert primitive.grasp_state == "ready"


def test_l6_freezes_index_on_current_delta() -> None:
    primitive = _primitive(phase="auto")
    start = [0.0] * 20
    primitive.on_enter(start)
    idle = PrimitiveContext(
        hand_type="l6",
        joint_currents=_currents(6),
        joint_currents_valid=True,
    )
    first = primitive.compute(start, elapsed=1.6, ctx=idle)
    assert first.feasible
    spiked = PrimitiveContext(
        hand_type="l6",
        joint_currents=_currents(6, **{"2": 50.0}),
        joint_currents_valid=True,
    )
    second = primitive.compute(start, elapsed=1.8, ctx=spiked)
    assert second.feasible
    idx_flex = SEMANTIC_30_IDX["index_mcp_flex"]
    assert idx_flex in primitive._frozen_joints
    assert second.target_angles[idx_flex] == first.target_angles[idx_flex]


def test_l25_runs_without_l6_only_hold() -> None:
    primitive = _primitive(phase="auto")
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=0.2,
        ctx=PrimitiveContext(
            hand_type="l25",
            joint_currents=_currents(25),
            joint_currents_valid=True,
        ),
    )
    assert result.feasible
    assert result.hold_reason == ""


def test_l25_prep_does_not_require_l6_two_stage() -> None:
    primitive = _primitive(phase="prep")
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=0.6,
        ctx=PrimitiveContext(hand_type="l25"),
    )
    assert result.feasible
    assert primitive.grasp_state == "ready"


def test_l25_freezes_index_on_current_delta() -> None:
    primitive = _primitive(phase="close")
    start = [0.0] * 20
    primitive.on_enter(start)
    idle = PrimitiveContext(
        hand_type="l25",
        joint_currents=_currents(25),
        joint_currents_valid=True,
    )
    first = primitive.compute(start, elapsed=0.05, ctx=idle)
    assert first.feasible
    spiked = PrimitiveContext(
        hand_type="l25",
        joint_currents=_currents(25, **{"1": 300.0}),
        joint_currents_valid=True,
    )
    second = primitive.compute(start, elapsed=0.4, ctx=spiked)
    assert second.feasible
    idx_flex = SEMANTIC_30_IDX["index_mcp_flex"]
    assert idx_flex in primitive._frozen_joints
    assert second.target_angles[idx_flex] == first.target_angles[idx_flex]


def test_g20_auto_is_feasible() -> None:
    primitive = _primitive(phase="auto")
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=0.3,
        ctx=PrimitiveContext(
            hand_type="g20",
            joint_currents=_currents(16),
            joint_currents_valid=True,
        ),
    )
    assert result.feasible


def test_o6_freezes_on_tactile_not_torque() -> None:
    primitive = _primitive(phase="close")
    start = [0.0] * 20
    primitive.on_enter(start)
    idle = PrimitiveContext(
        hand_type="o6",
        tactile_mode="pressure",
        tactile_pressure=np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        joint_torque=np.array([80.0] * 6),
    )
    first = primitive.compute(start, elapsed=0.05, ctx=idle)
    assert first.feasible
    assert not primitive._frozen_joints
    touched = PrimitiveContext(
        hand_type="o6",
        tactile_mode="pressure",
        tactile_pressure=np.array([0.0, 50.0, 0.0, 0.0, 0.0]),
        joint_torque=np.array([80.0] * 6),
    )
    second = primitive.compute(start, elapsed=0.3, ctx=touched)
    assert second.feasible
    idx_flex = SEMANTIC_30_IDX["index_mcp_flex"]
    assert idx_flex in primitive._frozen_joints
