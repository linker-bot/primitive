from hand_gesture_primitives.keyframe_runner import KeyframeSegment, MotionProgram
from hand_gesture_primitives.trajectory_planner import (
    plan_motion_program,
    plan_static_gesture,
)


def test_fist_plan_uses_configured_phases_without_side_effects() -> None:
    from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM, as_semantic_30

    current = [0.0] * 20
    planned = plan_static_gesture(
        hand_type="o20",
        primitive="fist",
        current_semantic=current,
        semantic_to_positions=lambda values: list(values),
        sample_period=0.1,
    )

    assert planned["nominal_only"] is True
    assert planned["duration_s"] == 1.7
    assert planned["time_s"][0] == 0.0
    assert planned["time_s"][-1] == 1.7
    assert planned["positions"][0] == as_semantic_30(current)
    assert len(planned["positions"][0]) == SEMANTIC_DIM
    assert planned["positions"][-1] != planned["positions"][0]


def test_o6_fist_plan_has_controller_width_after_conversion() -> None:
    planned = plan_static_gesture(
        hand_type="o6",
        primitive="fist",
        current_semantic=[0.0] * 20,
        semantic_to_positions=lambda values: [values[index] for index in (0, 5, 1, 2, 3, 4)],
    )

    assert all(len(point) == 6 for point in planned["positions"])


def test_parse_pose_plan_ref() -> None:
    from hand_gesture_primitives.trajectory_planner import parse_pose_plan_ref

    assert parse_pose_plan_ref("pose my_grip") == "my_grip"
    assert parse_pose_plan_ref("pose:my_grip") == "my_grip"
    assert parse_pose_plan_ref("pose/my_grip") == "my_grip"
    assert parse_pose_plan_ref("named_pose:my_grip") == "my_grip"
    assert parse_pose_plan_ref("fist") is None
    assert parse_pose_plan_ref("pose ") is None


def test_plan_motion_program_multi_segment() -> None:
    program = MotionProgram(
        ref="pose/t",
        space="semantic_v3",
        unit="rad",
        segments=[
            KeyframeSegment("pre", (1.0, 1.0, 1.0, 1.0), 1.0),
            KeyframeSegment("grasp", (2.0, 2.0, 2.0, 2.0), 1.0),
        ],
    )
    planned = plan_motion_program(
        program=program,
        start_q=[0.0, 0.0, 0.0, 0.0],
        to_positions=lambda q: list(q),
        sample_period=0.5,
    )
    assert planned["duration_s"] == 2.0
    assert planned["time_s"][0] == 0.0
    assert planned["time_s"][-1] == 2.0
    assert planned["positions"][-1] == [2.0, 2.0, 2.0, 2.0]
    assert len(planned["segments"]) >= 2
    assert planned["segments"][0]["name"] == "pre"
