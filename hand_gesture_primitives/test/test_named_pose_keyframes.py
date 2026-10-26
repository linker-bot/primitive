"""NamedPose + KeyframeRunner 集成（无 ROS）。"""

from __future__ import annotations

from unittest import mock

from hand_gesture_primitives.pose_library import PoseEntry, PoseKeyframe
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitives.named_pose import NamedPose
from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM


def _ctx() -> PrimitiveContext:
    return PrimitiveContext(
        hand_type="o20",
        hand_side="left",
        angle_unit="legacy255",
    )


def test_named_pose_progress_scrub(monkeypatch=None) -> None:
    entry = PoseEntry(
        name="grip",
        positions=[10.0] * SEMANTIC_DIM,
        space="semantic_v3",
        duration=1.0,
        num_joints=SEMANTIC_DIM,
        unit="legacy255",
        keyframes=(
            PoseKeyframe("pre", [0.0] * SEMANTIC_DIM, 1.0),
            PoseKeyframe("grasp", [10.0] * SEMANTIC_DIM, 1.0),
        ),
    )
    with mock.patch("hand_gesture_primitives.primitives.named_pose.load_pose", return_value=entry):
        prim = NamedPose("grip", keyframe="grasp", progress=0.5)
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        ctx = _ctx()
        result = prim.compute(start, 0.0, ctx)
        assert result.feasible
        assert abs(result.target_angles[0] - 5.0) < 1e-6
        prim.set_progress(0.0)
        result2 = prim.compute(start, 0.0, ctx)
        assert abs(result2.target_angles[0] - 0.0) < 1e-6
        assert "drive=progress" in prim.motion_status
        assert "keyframe=grasp" in prim.motion_status


def test_named_pose_auto_multi_keyframe() -> None:
    entry = PoseEntry(
        name="grip",
        positions=[2.0] * SEMANTIC_DIM,
        space="semantic_v3",
        duration=1.0,
        num_joints=SEMANTIC_DIM,
        unit="legacy255",
        keyframes=(
            PoseKeyframe("pre", [1.0] * SEMANTIC_DIM, 1.0),
            PoseKeyframe("grasp", [2.0] * SEMANTIC_DIM, 1.0),
        ),
    )
    with mock.patch("hand_gesture_primitives.primitives.named_pose.load_pose", return_value=entry):
        prim = NamedPose("grip")
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        ctx = _ctx()
        r0 = prim.compute(start, 0.5, ctx)
        assert abs(r0.target_angles[0] - 0.5) < 1e-6
        r1 = prim.compute(start, 1.5, ctx)
        # 第二段：从 1 → 2，local 0.5 → 1.5
        assert abs(r1.target_angles[0] - 1.5) < 1e-6
        r2 = prim.compute(start, 2.0, ctx)
        assert abs(r2.target_angles[0] - 2.0) < 1e-6
        assert prim.done


def test_named_pose_auto_segment_stops_at_keyframe() -> None:
    entry = PoseEntry(
        name="grip",
        positions=[2.0] * SEMANTIC_DIM,
        space="semantic_v3",
        duration=1.0,
        num_joints=SEMANTIC_DIM,
        unit="legacy255",
        keyframes=(
            PoseKeyframe("pre", [1.0] * SEMANTIC_DIM, 1.0),
            PoseKeyframe("grasp", [2.0] * SEMANTIC_DIM, 1.0),
        ),
    )
    with mock.patch("hand_gesture_primitives.primitives.named_pose.load_pose", return_value=entry):
        prim = NamedPose("grip", keyframe="pre", auto_segment=True)
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        ctx = _ctx()
        r0 = prim.compute(start, 0.5, ctx)
        assert abs(r0.target_angles[0] - 0.5) < 1e-6
        assert not prim.progress_driven
        assert "drive=auto" in prim.motion_status
        r1 = prim.compute(start, 1.0, ctx)
        assert abs(r1.target_angles[0] - 1.0) < 1e-6
        assert prim.done
        assert prim.active_keyframe == "pre"


def test_named_pose_switch_keyframe_reanchors() -> None:
    entry = PoseEntry(
        name="grip",
        positions=[20.0] * SEMANTIC_DIM,
        space="semantic_v3",
        duration=1.0,
        num_joints=SEMANTIC_DIM,
        unit="legacy255",
        keyframes=(
            PoseKeyframe("pre", [0.0] * SEMANTIC_DIM, 1.0),
            PoseKeyframe("grasp", [10.0] * SEMANTIC_DIM, 1.0),
        ),
    )
    with mock.patch("hand_gesture_primitives.primitives.named_pose.load_pose", return_value=entry):
        prim = NamedPose("grip", keyframe="pre", progress=0.0)
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        ctx = _ctx()
        prim.compute(start, 0.0, ctx)
        # 换段：从当前 5 重锚，progress=0.5 → 5 + 0.5*(10-5)=7.5
        prim.switch_keyframe("grasp", [5.0] * SEMANTIC_DIM, progress=0.5, elapsed=0.0)
        r = prim.compute(start, 0.0, ctx)
        assert abs(r.target_angles[0] - 7.5) < 1e-6
        assert prim.active_keyframe == "grasp"


def test_executor_switch_keyframe_prefers_commanded_rad() -> None:
    """换段重锚优先 _last_target_semantic_rad，而非滞后反馈。"""
    import pytest

    pytest.importorskip("geometry_msgs")
    pytest.importorskip("rclpy")
    from unittest.mock import MagicMock

    from hand_gesture_primitives.executor import ControlMode, GestureExecutor

    entry = PoseEntry(
        name="grip",
        positions=[0.0] * SEMANTIC_DIM,
        space="semantic_v3",
        duration=1.0,
        num_joints=SEMANTIC_DIM,
        unit="legacy255",
        keyframes=(
            PoseKeyframe("pre", [0.0] * SEMANTIC_DIM, 1.0),
            PoseKeyframe("grasp", [10.0] * SEMANTIC_DIM, 1.0),
        ),
    )
    with mock.patch("hand_gesture_primitives.primitives.named_pose.load_pose", return_value=entry):
        prim = NamedPose("grip", keyframe="pre", progress=0.0)
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        prim.compute(start, 0.0, _ctx())

        commanded = [7.0] * SEMANTIC_DIM
        lagged_feedback = [1.0] * SEMANTIC_DIM

        ex = GestureExecutor.__new__(GestureExecutor)
        ex._active_primitive = prim
        ex._control_mode = ControlMode.PRIMITIVE
        ex._primitive_start_time = 0.0
        ex._motion_logged = True
        ex._last_result_msg = "x"
        ex._current_semantic_rad = list(lagged_feedback)
        ex._current_angles = list(lagged_feedback)
        ex._last_target_semantic_rad = list(commanded)
        ex._last_commanded_hw = []
        ex._last_target = []
        ex._radian_converter = object()  # truthy → use semantic_rad path
        ex._node = MagicMock()

        err = ex.set_named_pose("grip", keyframe="grasp", progress=0.0)
        assert err == ""
        # p=0 on grasp segment: target should equal commanded start (7), not feedback (1)
        r = prim.compute(lagged_feedback, 0.0, _ctx())
        assert abs(r.target_angles[0] - 7.0) < 1e-6


def test_executor_set_named_pose_hot_progress_no_reanchor() -> None:
    """同名同段重复 pose…p → set_progress，不重建原语。"""
    import pytest

    pytest.importorskip("geometry_msgs")
    pytest.importorskip("rclpy")
    from unittest.mock import MagicMock

    from hand_gesture_primitives.executor import ControlMode, GestureExecutor

    entry = PoseEntry(
        name="grip",
        positions=[10.0] * SEMANTIC_DIM,
        space="semantic_v3",
        duration=1.0,
        num_joints=SEMANTIC_DIM,
        unit="legacy255",
        keyframes=(PoseKeyframe("grasp", [10.0] * SEMANTIC_DIM, 1.0),),
    )
    with mock.patch("hand_gesture_primitives.primitives.named_pose.load_pose", return_value=entry):
        prim = NamedPose("grip", keyframe="grasp", progress=0.0)
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        prim.compute(start, 0.0, _ctx())

        ex = GestureExecutor.__new__(GestureExecutor)
        ex._active_primitive = prim
        ex._control_mode = ControlMode.PRIMITIVE
        ex._primitive_start_time = 0.0
        ex._motion_logged = True
        ex._last_result_msg = "x"
        ex._current_semantic_rad = list(start)
        ex._current_angles = list(start)
        ex._radian_converter = None
        ex._node = MagicMock()

        err = ex.set_named_pose("grip", keyframe="grasp", progress=0.5)
        assert err == ""
        assert ex._active_primitive is prim  # 同一实例，未 set_primitive
        r = prim.compute(start, 0.0, _ctx())
        assert abs(r.target_angles[0] - 5.0) < 1e-6
        assert "热更新" in str(ex._node.get_logger().info.call_args)
