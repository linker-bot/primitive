"""引擎 reset / loader space 契约单测。"""

import pytest
from hand_gesture_primitives.gesture_engine import FingerAdductionEngine, RotProgressiveEngine
from hand_gesture_primitives.gesture_params import (
    load_static_gesture_params,
    load_static_params_for_ctx,
)
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM


def test_import_loader_rejects_degrees_space() -> None:
    from hand_gesture_primitives.gesture_params import _ensure_import_loader_space

    with pytest.raises(ValueError, match="load_static_params_for_ctx"):
        _ensure_import_loader_space(
            {"space": "degrees", "target": {}},
            "load_static_gesture_params",
        )


def test_rot_progressive_reset_clears_p2_entered() -> None:
    engine = RotProgressiveEngine(
        pose_target=[0.5] * SEMANTIC_DIM,
        rot_joint=SEMANTIC_30_IDX["thumb_rot"],
        rot_target=0.4,
        rot_rate=0.02,
        pose_duration=0.1,
        phase="full",
        contact_fingers=[],
    )
    engine.reset([0.0] * SEMANTIC_DIM)
    engine._p2_entered = True
    engine.reset([0.0] * SEMANTIC_DIM)
    assert engine._p2_entered is False


def test_finger_adduction_reset_clears_p2_entered() -> None:
    target = [0.5] * SEMANTIC_DIM
    engine = FingerAdductionEngine(
        pose_target=target,
        joint_a=SEMANTIC_30_IDX["index_mcp_abd"],
        joint_b=SEMANTIC_30_IDX["middle_mcp_abd"],
        close_a=0.3,
        close_b=0.4,
        pose_duration=0.1,
        close_duration=0.2,
        phase="full",
        contact_fingers=[],
    )
    engine.reset([0.0] * SEMANTIC_DIM)
    engine._p2_entered = True
    engine.reset([0.0] * SEMANTIC_DIM)
    assert engine._p2_entered is False


def test_rot_progressive_auto_eps_rad_not_one() -> None:
    """semantic_rad 姿态下默认阈值不应是 legacy 1.0。"""
    engine = RotProgressiveEngine(
        pose_target=[0.5] * SEMANTIC_DIM,
        rot_joint=SEMANTIC_30_IDX["thumb_rot"],
        rot_target=0.4,
        rot_rate=0.02,
        pose_duration=0.1,
    )
    assert engine._rot_arrive_threshold() < 0.5


def test_load_static_gesture_params_still_works() -> None:
    params = load_static_gesture_params("l25", "open")
    assert len(params.target_angles) == SEMANTIC_DIM


def test_load_static_params_for_ctx_without_converter() -> None:
    params = load_static_params_for_ctx("l25", "open", PrimitiveContext(hand_type="l25"))
    assert len(params.target_angles) == SEMANTIC_DIM


def test_static_pose_engine_zero_duration_no_divzero() -> None:
    from hand_gesture_primitives.gesture_engine import StaticPoseEngine
    from hand_gesture_primitives.gesture_params import StaticGestureParams

    params = StaticGestureParams(
        hand_type="o20",
        primitive="open",
        motion_type="static",
        target_angles=[1.0] * SEMANTIC_DIM,
        duration=0.0,
    )
    engine = StaticPoseEngine(params)
    engine.reset([0.0] * SEMANTIC_DIM)
    assert engine.compute(0.1) == [1.0] * SEMANTIC_DIM


def test_cached_static_params_mutation_isolated() -> None:
    load_static_gesture_params.cache_clear()
    a = load_static_gesture_params("l25", "open")
    a.target_angles[0] = 999.0
    b = load_static_gesture_params("l25", "open")
    assert b.target_angles[0] != 999.0


class _FakeRadianConverter:
    """仅暴露 angle_bridge / as_semantic_rad 所需属性。"""

    def __init__(self) -> None:
        # S30 thumb_rot=4 用更大行程，便于与默认 (0, π/2) 区分
        self.semantic_indices = list(range(SEMANTIC_DIM))
        self.limits = [(0.0, 1.5708)] * SEMANTIC_DIM
        self.limits[SEMANTIC_30_IDX["thumb_rot"]] = (0.0, 2.0)
        self.inverted_indices = frozenset()

    def legacy255_to_semantic_rad(self, values):
        return [float(v) * 0.01 for v in values]


def test_thumb_adduction_rad_uses_rot_slot_when_abd_flex_disabled() -> None:
    """L25 rot_progressive：abd/flex=-1 时 abd_max/rate 须按 thumb_rot 限位换算。"""
    from hand_gesture_primitives.angle_bridge import (
        step255_to_rad_at_slot,
        value255_to_rad_at_slot,
    )
    from hand_gesture_primitives.gesture_params import (
        load_thumb_adduction_params,
        thumb_adduction_params_as_semantic_rad,
    )

    rot_idx = SEMANTIC_30_IDX["thumb_rot"]
    params = load_thumb_adduction_params("l25")
    assert params.engine == "rot_progressive"
    assert params.progressive_abd_joint < 0
    assert params.progressive_flex_joint < 0
    assert params.thumb_rot_joint == rot_idx

    conv = _FakeRadianConverter()
    rad = thumb_adduction_params_as_semantic_rad(params, conv)
    expect_max = value255_to_rad_at_slot(conv, rot_idx, params.progressive_abd_max)
    expect_rate = step255_to_rad_at_slot(conv, rot_idx, params.progressive_rate)
    assert rad.progressive_abd_max == pytest.approx(expect_max)
    assert rad.progressive_rate == pytest.approx(expect_rate)
    # 不得退化到未知槽 (0, π/2) 上的换算
    wrong_max = value255_to_rad_at_slot(conv, -1, params.progressive_abd_max)
    assert rad.progressive_abd_max != pytest.approx(wrong_max)


def test_o20_planable_factory_covers_builders() -> None:
    from hand_gesture_primitives.export_primitive_poses import resolve_primitive_list
    from hand_gesture_primitives.gesture_params import _pinch_o20, load_static_gesture_params
    from hand_gesture_primitives.planable_primitives import PLANABLE_STATIC_PRIMITIVES

    load_static_gesture_params.cache_clear()
    prims = set(resolve_primitive_list(hand_type="o20"))
    assert prims == PLANABLE_STATIC_PRIMITIVES
    assert load_static_gesture_params("o20", "pinch").target_angles == _pinch_o20()


def test_g20_planable_factory_covers_builders() -> None:
    from hand_gesture_primitives.export_primitive_poses import resolve_primitive_list
    from hand_gesture_primitives.gesture_params import _pinch_o20, load_static_gesture_params
    from hand_gesture_primitives.planable_primitives import PLANABLE_STATIC_PRIMITIVES

    load_static_gesture_params.cache_clear()
    prims = set(resolve_primitive_list(hand_type="g20"))
    assert prims == PLANABLE_STATIC_PRIMITIVES
    assert load_static_gesture_params("g20", "pinch").target_angles == _pinch_o20()


def test_cached_finger_adduction_mutation_isolated() -> None:
    from hand_gesture_primitives.gesture_params import load_finger_adduction_params

    load_finger_adduction_params.cache_clear()
    a = load_finger_adduction_params("l25")
    a.prep_angles[0] = 999.0
    b = load_finger_adduction_params("l25")
    assert b.prep_angles[0] != 999.0


def test_resolve_progressive_joint_negative_uses_fallback() -> None:
    from hand_gesture_primitives.gesture_engine import _resolve_progressive_joint

    rot = SEMANTIC_30_IDX["thumb_rot"]
    abd = SEMANTIC_30_IDX["index_mcp_abd"]
    assert _resolve_progressive_joint(-1, rot) == rot
    assert _resolve_progressive_joint(abd, rot) == abd
    assert _resolve_progressive_joint(-1, abd) == abd
