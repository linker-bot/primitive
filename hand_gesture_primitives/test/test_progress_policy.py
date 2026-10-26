"""progress_policy 拒绝矩阵单测。"""

from hand_gesture_primitives.progress_policy import validate_primitive_progress


def test_tripod_prep_progress_ok() -> None:
    assert validate_primitive_progress("tripod", "prep", 0.3) is None
    assert validate_primitive_progress("tripod", "close", 0.5) is None
    assert validate_primitive_progress("large_wrap", "prep", 0.2) is None


def test_reject_auto_and_force() -> None:
    assert validate_primitive_progress("tripod", "auto", 0.3) is not None
    assert validate_primitive_progress("tripod_by_force", "prep", 0.3) is not None
    assert validate_primitive_progress("thumb_adduction_grip", "close", 0.3) is not None
    assert validate_primitive_progress("hammer_grip", "close", 0.1) is not None
    assert validate_primitive_progress("open", "prep", 0.1) is not None


def test_tripod_by_force_rejects_set_progress_api() -> None:
    from hand_gesture_primitives.primitives.tripod_by_force import TripodByForce

    prim = TripodByForce(phase="prep")
    assert prim.supports_progress() is False
    prim.set_progress(0.5)  # no-op, no raise


def test_none_progress_always_ok() -> None:
    assert validate_primitive_progress("tripod_by_force", "close", None) is None
