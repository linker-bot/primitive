"""units 与 Converter：legacy255 ↔ semantic_rad ↔ controller 兼容性。"""

from __future__ import annotations

import math

import pytest
from hand_gesture_primitives.joint_state_mapping import (
    Ros2ControlRadianConverter,
    ros2_control_joint_names,
    ros2_control_semantic_indices,
)
from hand_gesture_primitives.semantic_30 import as_semantic_30

from hand_gesture_primitives import units as U


def _l25_converter() -> Ros2ControlRadianConverter:
    names = ros2_control_joint_names("l25", "left")
    joints = "".join(
        f'<joint name="{name}" type="revolute">'
        '<limit lower="-0.2" upper="1.4" effort="1" velocity="1"/>'
        "</joint>"
        for name in names
    )
    return Ros2ControlRadianConverter(f'<robot name="test">{joints}</robot>', "l25", "left")


def test_deg_rad_roundtrip() -> None:
    assert U.deg_to_rad(180.0) == pytest.approx(math.pi)
    assert U.rad_to_deg(math.pi / 2) == pytest.approx(90.0)


def test_legacy255_controller_path_matches_direct_rad_path() -> None:
    converter = _l25_converter()
    semantic_255 = as_semantic_30([float(i * 13 % 256) for i in range(20)])

    via_legacy = converter.semantic_to_radians(semantic_255)
    semantic_rad = converter.legacy255_to_semantic_rad(semantic_255)
    via_rad = converter.semantic_rad_to_controller(semantic_rad)

    assert len(via_legacy) == 16
    for a, b in zip(via_legacy, via_rad):
        assert a == pytest.approx(b, abs=1e-9)


def test_semantic_rad_controller_roundtrip_preserves_slots() -> None:
    converter = _l25_converter()
    semantic_255 = as_semantic_30([float(i * 17 % 256) for i in range(20)])
    semantic_rad = converter.legacy255_to_semantic_rad(semantic_255)
    controller = converter.semantic_rad_to_controller(semantic_rad)
    restored_rad = converter.controller_to_semantic_rad(controller)

    for index in ros2_control_semantic_indices("l25"):
        assert restored_rad[index] == pytest.approx(semantic_rad[index], abs=1e-9)


def test_legacy255_rad_legacy_roundtrip() -> None:
    converter = _l25_converter()
    semantic_255 = as_semantic_30([float(i * 19 % 256) for i in range(20)])
    rad = converter.legacy255_to_semantic_rad(semantic_255)
    back = converter.semantic_rad_to_legacy255(rad)
    for index in ros2_control_semantic_indices("l25"):
        assert back[index] == pytest.approx(semantic_255[index], abs=1e-6)


def test_step255_to_rad_step() -> None:
    # full scale 255 over 1.0 rad span → step 25.5 → 0.1 rad
    assert U.legacy_step255_to_rad_step(25.5, 0.0, 1.0) == pytest.approx(0.1)
    # 有符号步进：负向不能被 abs 抹掉
    assert U.legacy_step255_to_rad_step(-25.5, 0.0, 1.0) == pytest.approx(-0.1)
    # inverted：legacy 增大 → rad 减小
    assert U.legacy_step255_to_rad_step(25.5, 0.0, 1.0, inverted=True) == pytest.approx(-0.1)


def test_normalize_angle_space_helpers_exist_for_yaml() -> None:
    # deg vector
    assert U.deg_vec_to_rad([180.0, 0.0])[0] == pytest.approx(math.pi)


def test_gesture_params_degrees_space_matches_legacy255() -> None:
    from hand_gesture_primitives.gesture_params import _resolve_static_target
    from hand_gesture_primitives.hand_config import HandConfig

    converter = _l25_converter()
    hand = HandConfig("l25")
    semantic_255 = [0.0] * 20
    semantic_255[0] = 128.0
    expected = converter.legacy255_to_semantic_rad(semantic_255)
    deg0 = U.rad_to_deg(expected[0])

    got = _resolve_static_target(
        "open",
        "l25",
        hand,
        {"space": "degrees", "target": {"thumb_base": deg0}},
        radian_converter=converter,
    )
    assert got[0] == pytest.approx(128.0, abs=1e-4)


def test_fk_rad_path_matches_legacy255_path() -> None:
    from hand_gesture_primitives.fk_solver import HandFKSolver

    solver = object.__new__(HandFKSolver)
    solver._active_mapping = {
        0: "j0",
        1: "j1",
    }
    solver._joint_limits = {"j0": (0.0, 1.0), "j1": (-0.5, 0.5)}
    solver._invert_joints = set()

    semantic_255 = [0.0] * 20
    semantic_255[0] = 127.5
    semantic_255[1] = 255.0
    via_255 = solver.o20_to_indexed_radians(semantic_255)

    # 无 invert：ratio * span + lo
    semantic_rad = [0.0] * 20
    semantic_rad[0] = 0.5
    semantic_rad[1] = 0.5
    via_rad = solver.o20_rad_to_indexed_radians(semantic_rad)

    assert via_rad[0] == pytest.approx(via_255[0], abs=1e-9)
    assert via_rad[1] == pytest.approx(via_255[1], abs=1e-9)


def test_static_engine_rad_matches_legacy_controller_path() -> None:
    from hand_gesture_primitives.gesture_engine import make_static_engine
    from hand_gesture_primitives.gesture_params import (
        load_static_gesture_params,
        static_params_as_semantic_rad,
    )

    converter = _l25_converter()
    start_255 = as_semantic_30([float(i * 11 % 256) for i in range(20)])
    start_rad = converter.legacy255_to_semantic_rad(start_255)

    legacy = make_static_engine("l25", "open")
    legacy.reset(start_255)
    rad_engine = make_static_engine("l25", "open", radian_converter=converter)
    rad_engine.reset(start_rad)

    for t in (0.0, 0.3, 1.0):
        out_255 = legacy.compute(t)
        out_rad = rad_engine.compute(t)
        via_legacy = converter.semantic_to_radians(out_255)
        via_rad = converter.semantic_rad_to_controller(out_rad)
        for a, b in zip(via_legacy, via_rad):
            assert a == pytest.approx(b, abs=1e-6)

    params = load_static_gesture_params("l25", "open")
    as_rad = static_params_as_semantic_rad(params, converter)
    assert as_rad.target_angles[0] == pytest.approx(
        converter.legacy255_to_semantic_rad(params.target_angles)[0], abs=1e-9
    )


def test_plan_static_gesture_rad_path_matches_legacy() -> None:
    from hand_gesture_primitives.trajectory_planner import plan_static_gesture

    converter = _l25_converter()
    current = [float(i * 7 % 256) for i in range(20)]
    legacy = plan_static_gesture(
        hand_type="l25",
        primitive="open",
        current_semantic=current,
        semantic_to_positions=converter.semantic_to_radians,
        sample_period=0.1,
    )
    rad = plan_static_gesture(
        hand_type="l25",
        primitive="open",
        current_semantic=current,
        semantic_to_positions=converter.semantic_to_radians,
        sample_period=0.1,
        radian_converter=converter,
    )
    assert legacy["duration_s"] == rad["duration_s"]
    assert len(legacy["positions"]) == len(rad["positions"])
    for p_legacy, p_rad in zip(legacy["positions"], rad["positions"]):
        for a, b in zip(p_legacy, p_rad):
            assert a == pytest.approx(b, abs=1e-5)
