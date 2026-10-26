"""motion_progress topic 名与 NamedPose.set_progress（无 ROS）。"""

from __future__ import annotations

from unittest import mock

from hand_gesture_primitives.pose_library import PoseEntry, PoseKeyframe
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitives.named_pose import NamedPose
from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM
from hand_gesture_primitives.topic_defaults import side_hand_topics


def test_named_pose_set_progress_before_runner() -> None:
    """pose 指令后、首帧 compute 前发 progress 不得丢弃。"""
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
        prim = NamedPose("grip", keyframe="grasp")
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        prim.set_progress(0.5)  # runner 尚未创建
        ctx = PrimitiveContext(hand_type="o20", hand_side="left", angle_unit="legacy255")
        r0 = prim.compute(start, 0.0, ctx)
        assert abs(r0.target_angles[0] - 5.0) < 1e-6
        assert "drive=progress" in prim.motion_status


def test_named_pose_progress_one_not_done() -> None:
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
        prim = NamedPose("grip", keyframe="grasp", progress=1.0)
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        ctx = PrimitiveContext(hand_type="o20", hand_side="left", angle_unit="legacy255")
        r0 = prim.compute(start, 0.0, ctx)
        assert abs(r0.target_angles[0] - 10.0) < 1e-6
        assert not prim.done
        assert prim.progress_driven
        prim.set_progress(0.2)
        r1 = prim.compute(start, 0.0, ctx)
        assert abs(r1.target_angles[0] - 2.0) < 1e-6


def test_side_topics_include_motion_progress() -> None:
    topics = side_hand_topics("left")
    assert topics["motion_progress"] == "/left_hand/motion_progress"
    assert side_hand_topics("right")["motion_progress"] == "/right_hand/motion_progress"


def test_named_pose_set_progress_hot_update() -> None:
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
        ctx = PrimitiveContext(hand_type="o20", hand_side="left", angle_unit="legacy255")
        r0 = prim.compute(start, 0.0, ctx)
        assert abs(r0.target_angles[0] - 0.0) < 1e-6
        prim.set_progress(0.5)
        r1 = prim.compute(start, 0.0, ctx)
        assert abs(r1.target_angles[0] - 5.0) < 1e-6
