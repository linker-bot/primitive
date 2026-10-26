"""Runtime control transitions use committed actions, not incoming request attempts."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("rclpy")
from hand_gesture_primitives.executor import ControlMode, GestureExecutor
from hand_gesture_primitives.gesture_node import GestureNode
from hand_gesture_primitives.hand_profile import load_hand_profile
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY


def executor():
    ex = GestureExecutor.__new__(GestureExecutor)
    ex._node = Mock()
    ex._hand_type, ex._side = "o6", "right"
    ex._profile = load_hand_profile("o6")
    ex._hand_config = ex._profile.hardware
    ex._radian_converter = None
    ex._current_angles = [0.] * 30
    ex._current_hw_angles = [0.] * 6
    ex._last_target = [0.] * 6
    ex._active_primitive = None
    ex._control_mode = ControlMode.PRIMITIVE
    ex._target_label = "old"
    ex._target_instance_id = None
    ex._object_size = np.ones(3)
    ex._object_orientation = None
    ex._object_center = np.ones(3)
    ex._object_center_frame = "world"
    ex._grasp_type = "old-grasp"
    ex._is_safe_primitive = lambda name: True
    ex._log_primitive_context = lambda name: None
    ex._hold_hw_for_resume = lambda: [0.] * 6
    ex._stream_target_topic, ex._stream_watchdog_sec = "unused", .5
    return ex


def test_executor_commits_epoch_only_after_success_and_stop_always_invalidates():
    ex = executor()
    assert ex.set_primitive(PRIMITIVE_REGISTRY["open"]()) == ""
    previous, epoch = ex._active_primitive, ex.control_epoch
    bad = Mock(name="broken")
    bad.name = "broken"
    bad.on_enter.side_effect = ValueError("invalid candidate")
    target = ex.target_selection_snapshot()
    assert ex.set_primitive(bad, target_label="new").startswith("rejected:")
    assert ex._active_primitive is previous and ex.control_epoch == epoch
    for name, value in target.items():
        assert getattr(ex, name) is value
    ex.stop()
    assert ex.control_epoch > epoch and ex._active_primitive is None
    epoch = ex.control_epoch
    ex.stop()
    assert ex.control_epoch > epoch


def test_actual_mode_changes_invalidate_and_repeated_mode_does_not():
    ex = executor()
    epoch = ex.control_epoch
    ex.set_control_mode(ControlMode.PRIMITIVE)
    assert ex.control_epoch == epoch
    for mode in (ControlMode.TEACH, ControlMode.STREAM, ControlMode.PRIMITIVE):
        ex.set_control_mode(mode)
        assert ex.control_epoch > epoch
        epoch = ex.control_epoch
        ex.set_control_mode(mode)
        assert ex.control_epoch == epoch


@pytest.mark.parametrize("result", ["unchanged", "disabled", "deferred", "reloaded", "failure"])
def test_only_successful_reload_invalidates(result):
    ex = executor()
    node = SimpleNamespace(
        _executor=ex, _reload_was_deferred=False,
        _primitive_reloader=Mock(), get_logger=lambda: Mock(), _set_status_override=Mock(),
    )
    ex.publish_user_result = Mock()
    if result == "failure":
        node._primitive_reloader.reload_if_changed.side_effect = ValueError("bad source")
    else:
        node._primitive_reloader.reload_if_changed.return_value = result
    GestureNode._reload_primitives_if_changed(node, force=True)
    assert (ex.control_epoch > 0) == (result == "reloaded")


def test_progress_failure_keeps_epoch_success_resets_completion():
    ex = executor()
    prim = PRIMITIVE_REGISTRY["tripod"](phase="prep")
    ex.set_primitive(prim)
    epoch = ex.control_epoch
    ex._settle_status, ex._settle_frames = "settled", 3
    assert ex.apply_motion_progress(float("nan")) is not None
    assert ex.control_epoch == epoch and ex._settle_frames == 3
    assert ex.apply_motion_progress(.2) is None
    assert ex.control_epoch > epoch and ex._settle_frames == 0
    assert ex._settle_status == "running"


@pytest.mark.parametrize("model,name", [
    ("o6", "large_wrap"), ("l6", "large_wrap"), ("o6", "hammer_grip"),
    ("l6", "tripod"), ("l25", "index_flick"), ("l25", "index_middle_flick"),
    ("o20", "parallel_extension"),
])
def test_finite_primitives_declare_actual_terminal_output(model, name):
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    p = PRIMITIVE_REGISTRY[name]()
    q = [0.] * 30
    p.on_enter(q)
    ctx = PrimitiveContext(hand_type=model, angle_unit="semantic")
    result = p.compute(q, 100., ctx)
    assert result.feasible
    assert p.trajectory_complete(result, 100.)


def test_held_clock_is_not_truncated_or_completed_by_nominal_duration():
    from hand_gesture_primitives.workbench_motion import PlaybackClock
    clock = PlaybackClock(0., duration=.1, held=True)
    assert clock.advance(2.) == 2.
    assert not clock.complete
    assert clock.snapshot()["playback_state"] == "running"


def torque_executor():
    from hand_gesture_primitives.contact_config import ContactThresholds
    ex = executor()
    ex._joint_torque = None
    ex._joint_currents = [-1.] * 6
    ex._joint_currents_valid = False
    ex._joint_overload_start = [None] * 6
    ex._contact_thresholds = ContactThresholds()
    ex._tactile_pressure = None
    ex._compute_fingertip_positions = lambda: None
    return ex


def test_torque_topic_only_keeps_updating_and_expires(monkeypatch):
    from hand_gesture_primitives.workbench_support import executor_signals
    now = [10.]
    monkeypatch.setattr("hand_gesture_primitives.executor.time.monotonic", lambda: now[0])
    ex = torque_executor()
    for stamp, value in [(10., 10.), (10.5, 20.), (11.1, 30.)]:
        now[0] = stamp
        ex._torque_callback(SimpleNamespace(data=[value] * 6))
        assert executor_signals(ex)["torque"]
        assert ex._joint_torque[0] == value
        assert ex._workbench_torque_at == stamp
    now[0] = 12.2
    assert not executor_signals(ex)["torque"]
    assert ex._joint_torque is None


def test_torque_primary_priority_and_cached_fallback_keep_original_timestamps(monkeypatch):
    import json

    from hand_gesture_primitives.workbench_support import executor_signals
    now = [10.]
    monkeypatch.setattr("hand_gesture_primitives.executor.time.monotonic", lambda: now[0])
    ex = torque_executor()
    ex._info_callback(SimpleNamespace(data=json.dumps({"motor_torque_pct": [40.] * 6})))
    now[0] = 10.6
    ex._torque_callback(SimpleNamespace(data=[20.] * 6))
    assert ex._joint_torque[0] == 40. and ex._workbench_torque_at == 10.
    now[0] = 11.1
    assert executor_signals(ex)["torque"]
    assert ex._joint_torque[0] == 20. and ex._workbench_torque_at == 10.6
    ex._info_callback(SimpleNamespace(data=json.dumps({"motor_torque_pct": [50.] * 6})))
    assert ex._joint_torque[0] == 50. and ex._workbench_torque_at == 11.1
    now[0] = 12.2
    assert not executor_signals(ex)["torque"]


@pytest.mark.parametrize("model", ["o6", "l6"])
@pytest.mark.parametrize("phase", ["prep", "close", "auto", "full"])
def test_large_wrap_completion_covers_delayed_thumb_and_stage_boundaries(model, phase):
    from hand_gesture_primitives.gesture_params import load_large_wrap_params
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    params = load_large_wrap_params(model)
    p = PRIMITIVE_REGISTRY["large_wrap"](phase=phase)
    q, ctx = [0.] * 30, PrimitiveContext(hand_type=model, angle_unit="semantic")
    p.on_enter(q)
    offset = 0 if phase in {"prep", "close"} else params.prep_duration + params.pause_duration
    end = params.prep_duration if phase == "prep" else offset + params.close_duration + params.thumb_delay
    result = p.compute(q, end - 1e-5, ctx)
    assert not p.trajectory_complete(result, end - 1e-5)
    result = p.compute(q, end, ctx)
    assert p.trajectory_complete(result, end)
    if phase in {"auto", "full"}:
        p.on_enter(q)
        result = p.compute(q, params.prep_duration, ctx)
        assert not p.trajectory_complete(result, params.prep_duration)


@pytest.mark.parametrize("model", ["o6", "l6"])
@pytest.mark.parametrize("phase", ["prep", "close", "auto", "full"])
def test_tripod_stage_terminal_and_progress_do_not_conflict(model, phase):
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    q, ctx = [0.] * 30, PrimitiveContext(hand_type=model, angle_unit="semantic")
    p = PRIMITIVE_REGISTRY["tripod"](phase=phase)
    p.on_enter(q)
    result = p.compute(q, 100., ctx)
    assert p.trajectory_complete(result, 100.)
    if phase in {"prep", "close"}:
        p.set_progress(1.)
        result = p.compute(q, 100., ctx)
        assert not p.trajectory_complete(result, 100.)
        p.set_progress(.2)
        result = p.compute(q, 100., ctx)
        assert not p.trajectory_complete(result, 100.)


def test_hammer_close_wait_and_timeout_never_finish_while_prep_can_finish():
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    q, ctx = [0.] * 30, PrimitiveContext(hand_type="o6", angle_unit="semantic")
    p = PRIMITIVE_REGISTRY["hammer_grip"](phase="close")
    p.on_enter(q)
    result = p.compute(q, 100., ctx)
    assert p.grasp_state == "prep_timeout"
    assert not p.trajectory_complete(result, 100.)
    p = PRIMITIVE_REGISTRY["hammer_grip"](phase="prep")
    p.on_enter(q)
    result = p.compute(q, 100., ctx)
    assert p.grasp_state == "ready" and p.trajectory_complete(result, 100.)
    p = PRIMITIVE_REGISTRY["hammer_grip"](phase="close")
    p.on_enter(q)
    params = p._params(ctx)
    for index in range(100):
        elapsed = index * .01
        result = p.compute(list(params.initial_angles), elapsed, ctx)
        if p._close_started_at is not None:
            break
    assert not p.trajectory_complete(result, elapsed)
    result = p.compute(list(params.initial_angles), elapsed + 100., ctx)
    assert p.trajectory_complete(result, elapsed + 100.)


@pytest.mark.parametrize("phase", ["auto", "full", "close"])
def test_force_tripod_cannot_inherit_position_completion(phase):
    from hand_gesture_primitives.primitive_base import PrimitiveResult
    p = PRIMITIVE_REGISTRY["tripod_by_force"](phase=phase)
    p.on_enter([0.] * 30)
    p._grasp_state = "contact"
    assert not p.trajectory_complete(PrimitiveResult(feasible=True, trajectory_finished=True), 100.)


def test_named_pose_only_finishes_requested_auto_program():
    from hand_gesture_primitives.pose_library import PoseEntry, PoseKeyframe
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.primitives.named_pose import NamedPose
    entry = PoseEntry(name="sequence", space="semantic_v3", unit="legacy255", positions=[0.] * 30,
                      keyframes=[PoseKeyframe(name="a", positions=[10.] * 30, duration=1.),
                                 PoseKeyframe(name="b", positions=[20.] * 30, duration=2.)])
    ctx = PrimitiveContext(hand_type="o6", angle_unit="semantic")
    for kwargs, end in [({}, 3.), ({"keyframe": "b", "auto_segment": True}, 2.)]:
        p = NamedPose(entry.name, entry=entry, **kwargs)
        p.on_enter([0.] * 30)
        r = p.compute([0.] * 30, end - .01, ctx)
        assert not p.trajectory_complete(r, end - .01)
        r = p.compute([0.] * 30, end, ctx)
        assert p.trajectory_complete(r, end)
    p = NamedPose(entry.name, entry=entry, progress=1.)
    p.on_enter([0.] * 30)
    r = p.compute([0.] * 30, 100., ctx)
    assert not p.trajectory_complete(r, 100.)


@pytest.mark.parametrize("invalid", [[float("nan")] * 6, [float("inf")] * 6, [-1.] * 6, [10.] * 5])
def test_invalid_torque_primary_does_not_hide_valid_topic(monkeypatch, invalid):
    import json

    from hand_gesture_primitives.workbench_support import executor_signals
    now = [10.]
    monkeypatch.setattr("hand_gesture_primitives.executor.time.monotonic", lambda: now[0])
    ex = torque_executor()
    ex._torque_callback(SimpleNamespace(data=[0.] * 6))
    now[0] = 10.5
    ex._info_callback(SimpleNamespace(data=json.dumps({"motor_torque_pct": invalid})))
    assert executor_signals(ex)["torque"]
    assert ex._workbench_torque_at == 10. and np.all(ex._joint_torque == 0.)
    now[0] = 11.1
    ex._torque_callback(SimpleNamespace(data=invalid))
    assert not executor_signals(ex)["torque"]


def test_non_o6_torque_topic_still_normalizes_legacy_samples(monkeypatch):
    monkeypatch.setattr("hand_gesture_primitives.executor.time.monotonic", lambda: 10.)
    ex = torque_executor()
    ex._hand_type = "l6"
    ex._torque_callback(SimpleNamespace(data=[255.] * 6))
    np.testing.assert_allclose(ex._joint_torque, [100.] * 6)
    assert ex._torque_source == "motor_torque"


def test_held_runner_ignores_short_nominal_duration_and_waits_for_feedback(monkeypatch):
    from hand_gesture_primitives.workbench_runner import HandPrimitiveRunner
    from test_workbench_core import adapter
    now = [100.]
    monkeypatch.setattr("hand_gesture_primitives.workbench_runner.time.monotonic", lambda: now[0])
    runner = HandPrimitiveRunner("o6", "right", 30, adapter=adapter())
    runner.select("large_wrap", np.zeros(6), held=True)
    runner.clock.duration = .01  # An estimate must never terminate or truncate motion.
    now[0] += .02
    target = runner.tick(np.zeros(6), feedback_id=1)
    assert runner.clock.elapsed > runner.clock.duration
    assert runner.status.overall_progress == 1.
    assert runner.status.playback_state == "running" and not runner.status.done
    now[0] += 3.
    target = runner.tick(target, feedback_id=2)
    assert runner.status.playback_state == "completed" and not runner.status.done
    for sample in range(3, 6):
        runner.tick(target, feedback_id=sample)
    assert runner.status.done and runner.status.settle_status == "settled"


def test_auto_label_can_reselect_cached_other_target_without_committing_failed_requests(monkeypatch):
    now = [10.]
    monkeypatch.setattr("hand_gesture_primitives.executor.time.monotonic", lambda: now[0])
    ex = executor()
    ex._bbox_miss_count = 0
    ex._object_pose = None
    ex._object_label = "old"
    boxes = [SimpleNamespace(label=label, score=score, instance_id=index,
                             size=[.03] * 3, center=[0., 0., .1],
                             orientation=[0., 0., 0., 1.], frame_id="world", grasp_type="power")
             for index, (label, score) in enumerate([("old", .9), ("cup", .8)])]
    msg = SimpleNamespace(boxes=boxes, header=SimpleNamespace(frame_id="world"))
    ex._bboxes_3d_callback(msg)
    node = SimpleNamespace(_executor=ex, _profile=ex._profile, get_logger=lambda: Mock(),
                           _set_status_override=Mock(), _report_primitive_reject=Mock())
    GestureNode._handle_auto_cmd(node, ["cup"])
    assert ex._active_primitive is not None
    assert ex._target_label == "cup"
    assert ex._workbench_bbox_at == 10.
    previous, epoch = ex._active_primitive, ex.control_epoch
    GestureNode._handle_auto_cmd(node, ["missing"])
    assert ex._active_primitive is previous and ex.control_epoch == epoch
    assert ex._target_label == "cup" and ex._workbench_bbox_at == 10.
    now[0] = 12.
    GestureNode._handle_auto_cmd(node, ["old"])
    assert ex._active_primitive is previous and ex.control_epoch == epoch
    assert ex._target_label == "cup" and ex._workbench_bbox_at == 10.


def _finite_cases():
    from hand_gesture_primitives.workbench_motion import FINITE_PRIMITIVES
    return [(model, name) for model in ("o6", "l6", "o20", "l25", "o30", "g20")
            for name in sorted(FINITE_PRIMITIVES)
            if load_hand_profile(model).is_primitive_supported(name)]


@pytest.mark.parametrize("model,name", _finite_cases())
def test_every_supported_finite_primitive_has_a_completion_contract(model, name):
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    p = PRIMITIVE_REGISTRY[name]()
    q = [0.] * 30
    p.on_enter(q)
    result = p.compute(q, 100., PrimitiveContext(hand_type=model, angle_unit="semantic"))
    assert result.feasible
    assert p.trajectory_complete(result, 100.)
