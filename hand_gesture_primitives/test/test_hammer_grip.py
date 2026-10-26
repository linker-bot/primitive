from hand_gesture_primitives.gesture_params import load_two_stage_speed_params
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitive_catalog import PRIMITIVE_CATALOG
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY


def test_o6_hammer_grip_is_registered_and_reaches_configured_target() -> None:
    assert "hammer_grip" in PRIMITIVE_REGISTRY
    assert PRIMITIVE_CATALOG["hammer_grip"].phased

    params = load_two_stage_speed_params("o6", "hammer_grip")
    primitive = PRIMITIVE_REGISTRY["hammer_grip"]()
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=60.0,
        ctx=PrimitiveContext(hand_type="o6"),
    )

    assert result.feasible
    assert result.target_angles == params.target_angles
    assert primitive.grasp_state == "holding"


def test_hammer_grip_fails_closed_for_unsupported_hand() -> None:
    primitive = PRIMITIVE_REGISTRY["hammer_grip"]()
    primitive.on_enter([0.0] * 20)
    result = primitive.compute(
        [0.0] * 20,
        elapsed=1.0,
        ctx=PrimitiveContext(hand_type="l25"),
    )

    assert not result.feasible
    assert "仅支持 O6" in result.hold_reason
