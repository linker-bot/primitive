# SPDX-License-Identifier: MIT
# LinkerHand workbench protocol tests; see LICENSE and NOTICE.
"""Exercise both ends of the confirmed protocol with controllable message delivery."""

import json
import time
from types import SimpleNamespace

import numpy as np
import pytest
from hand_gesture_primitives.workbench_cache import LatestCache
from hand_gesture_primitives.workbench_client import GestureNodeClient
from hand_gesture_primitives.workbench_core import (
    PoseDraft,
    PrimitiveRejectedError,
    WorkbenchService,
)
from test_workbench_core import adapter

pytest.importorskip("std_msgs.msg")
pytest.importorskip("rclpy")


@pytest.fixture
def protocol(monkeypatch):
    from hand_gesture_primitives.hand_profile import load_hand_profile
    from hand_gesture_primitives.workbench_support import WorkbenchBridge
    from std_msgs.msg import String

    published = {}

    class Executor:
        active_primitive_name = None
        control_mode = SimpleNamespace(value="primitive")
        _radian_converter = object()
        _hand_config = SimpleNamespace(num_joints=6)
        _active_primitive = None
        installed = 0
        control_epoch = 0

        def advance_control_epoch(self):
            self.control_epoch += 1

        def apply_motion_progress(self, progress):
            if not 0 <= progress <= 1:
                return "rejected:bad_progress"
            self._active_primitive.set_progress(progress)
            self.advance_control_epoch()
            return None

        def playback_status(self):
            clock = getattr(self, "_playback_clock", None)
            return (
                clock.snapshot() if clock else {"playback_state": "idle", "overall_progress": -1.0}
            )

        def set_primitive(self, primitive):
            from hand_gesture_primitives.workbench_motion import PlaybackClock

            self._playback_clock = PlaybackClock(time.monotonic(), duration=1.0)
            self._active_primitive = primitive
            self.active_primitive_name = primitive.name
            self.installed += 1
            self.advance_control_epoch()
            return ""

        def stop(self):
            self.advance_control_epoch()
            self._playback_clock = None
            self._live_active = False
            self._active_primitive = None
            self.active_primitive_name = None

        def hold_playback(self, running):
            self._playback_clock.set_running(running, time.monotonic())

        def set_motion_speed(self, speed):
            from hand_gesture_primitives.workbench_motion import validate_speed

            validate_speed(speed)
            self.speed = speed

        def live_target(self, positions):
            self.stop()
            self.positions = positions
            self._live_active = True

        def publish_user_result(self, text):
            pass

        def _object_pose_callback(self, value):
            self.selected_position = value.pose.position

        def _selected_label_callback(self, value):
            self.selected_label = value.data

    executor = Executor()
    node = SimpleNamespace(
        _executor=executor,
        _side="right",
        _hand_type="o6",
        _profile=load_hand_profile("o6"),
        create_publisher=lambda kind, name, qos: SimpleNamespace(
            publish=lambda msg: published.setdefault(name, []).append(msg)
        ),
        create_subscription=lambda *args: None,
        create_timer=lambda *args: None,
        get_logger=lambda: SimpleNamespace(warning=lambda *args: None),
        set_parameters_atomically=lambda parameters: (
            executor.set_motion_speed(parameters[0].value)
            or SimpleNamespace(successful=True, reason="")
        ),
    )
    monkeypatch.setattr(
        "hand_gesture_primitives.workbench_support.executor_signals",
        lambda e: dict.fromkeys(["position", "tactile", "current", "torque", "vision"], True),
    )
    bridge = WorkbenchBridge(node)
    bridge.library_error = ""
    client = GestureNodeClient.__new__(GestureNodeClient)
    client._init_protocol()
    client.catalog = LatestCache()
    client._String = String
    client._workbench_cmd = SimpleNamespace(
        publish=lambda msg: published.setdefault("commands", []).append(msg)
    )
    client._cmd = SimpleNamespace(
        publish=lambda msg: published.setdefault("legacy", []).append(msg)
    )
    client.side = "right"
    entry = WorkbenchService(adapter()).draft_entry(
        PoseDraft(
            name="custom", keyframes=[{"name": "a", "positions": [0.2] * 6, "duration": 0.4}]
        ),
    )
    bridge.entries[entry.name] = entry

    def sync():
        client.catalog.put(json.dumps(bridge.report()))
        client.synchronize()

    def deliver():
        bridge.command(published["commands"][-1])
        sync()

    sync()
    yield SimpleNamespace(
        client=client,
        bridge=bridge,
        executor=executor,
        entry=entry,
        published=published,
        sync=sync,
        deliver=deliver,
    )
    bridge.close()


def test_rejected_replacement_preserves_preview_and_blocks_recording(protocol):
    p = protocol
    p.client.preview(p.entry)
    p.deliver()
    assert p.client.preview_active
    p.client.send_primitive("named_pose:custom", p.entry)
    p.bridge.entries.clear()  # Concurrent edit/removal at the receiving node.
    p.deliver()
    assert p.client.preview_active and "试播" in p.client.recording_reason()
    assert not p.client.take_accepted_poses()
    p.client.stop()
    assert p.client.recording_reason()  # Sending is not confirmation.
    p.deliver()
    assert not p.client.recording_reason()


def test_expired_duplicate_reordered_commands_and_restart(protocol):
    p = protocol
    p.client.send_primitive("named_pose:custom", p.entry)
    first = p.published["commands"][-1]
    p.deliver()
    assert p.executor.installed == 1 and len(p.client.take_accepted_poses()) == 1
    p.bridge.command(first)
    p.sync()
    assert p.executor.installed == 1 and not p.client.take_accepted_poses()
    p.client.preview(p.entry)
    delayed = p.published["commands"][-1]
    p.client.stop()
    p.deliver()
    p.bridge.command(delayed)
    p.sync()
    assert p.executor.active_primitive_name is None
    p.client.preview(p.entry)
    payload = json.loads(p.published["commands"][-1].data)
    payload["expires_at_ns"] = time.time_ns() - 1
    p.bridge.command(SimpleNamespace(data=json.dumps(payload)))
    p.sync()
    assert p.executor.active_primitive_name is None
    p.bridge.node_id = "restarted"
    p.sync()
    assert p.client.recording_reason()
    p.client.stop()
    p.deliver()
    assert not p.client.recording_reason()


def test_confirmation_timeout_and_late_stop_receipt(protocol):
    p = protocol
    p.client.preview(p.entry)
    p.client._pending["deadline"] = time.monotonic() - 1
    assert p.client.recording_reason()
    with pytest.raises(PrimitiveRejectedError, match="尚未确认"):
        p.client.send_primitive("open")
    p.deliver()
    assert p.client.preview_active and p.client.recording_reason()
    p.client.stop()
    p.deliver()
    assert not p.client.recording_reason()


def test_saved_pose_requires_converter_at_receiver(protocol):
    p = protocol
    p.bridge.executor._radian_converter = None
    p.sync()
    assert "URDF" in p.bridge.report()["pose_reasons"]["custom"]
    p.client.send_primitive("named_pose:custom", p.entry)
    p.deliver()
    assert p.executor.installed == 0 and "URDF" in p.client._command_notice


def test_gated_command_waits_for_correlated_approval_and_stop_cancels_it(protocol):
    p = protocol
    p.client.send_primitive("small_warp_by_vision")
    p.deliver()
    assert p.executor.installed == 0
    p.bridge.gate_status(
        SimpleNamespace(
            data=json.dumps({"version": 2, "side": "right", "gate_id": "gate", "ready": True})
        )
    )
    p.sync()
    p.client.send_primitive("small_warp_by_vision")
    p.deliver()
    request = p.bridge.pending_gate
    assert request and p.executor.installed == 0
    reply = {
        "request_id": request["request_id"],
        "node_id": p.bridge.node_id,
        "gate_id": "gate",
        "accepted": True,
        "expires_at_ns": time.time_ns() + 100_000_000,
        "target": {"label": "cup", "position": [0, 0, 0.1], "size": [0.03] * 3},
    }
    p.client.stop()
    p.deliver()
    p.bridge.gate_reply(SimpleNamespace(data=json.dumps(reply)))
    assert p.executor.installed == 0
    p.client.send_primitive("small_warp_by_vision")
    p.deliver()
    reply["request_id"] = p.bridge.pending_gate["request_id"]
    p.bridge.gate_reply(SimpleNamespace(data=json.dumps(reply)))
    p.sync()
    assert p.executor.installed == 1 and p.executor.selected_label == "cup"


def test_gate_real_geometry_and_distinct_sample_debounce():
    from hand_gesture_primitives.grasp_gate import GraspGate
    from hand_gesture_primitives.workbench_gate import WorkbenchGate

    gate = GraspGate.__new__(GraspGate)
    now_ns = 1_000_000_000_000
    now = now_ns / 1e9
    gate.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=now_ns))
    gate._tcp = {"pos": np.zeros(3), "quat": np.array([0, 0, 0, 1]), "stamp": now}
    gate._data_timeout, gate._palm_z, gate._tip_z, gate._height_tol = 2, 0.06, 0.18, 0.05
    gate._half_w, gate._half_h, gate._de_bounce = 0.2, 0.15, 3
    gate._poses = {}
    gate._bboxes = {
        "cup": {
            "center": np.ones(3),
            "min": np.ones(3) - 0.01,
            "max": np.ones(3) + 0.01,
            "stamp": now,
        }
    }
    work = WorkbenchGate.__new__(WorkbenchGate)
    work.gate, work.last_sample, work.last_label, work.count = gate, None, None, 0
    work.update()
    assert work.current is None
    obj = gate._bboxes["cup"]
    obj.update(
        center=np.array([0, 0, 0.1]),
        min=np.array([-0.01, -0.01, 0.09]),
        max=np.array([0.01, 0.01, 0.11]),
    )
    for _ in range(5):
        work.update()
    assert work.current is None and work.count == 1
    # Three distinct synchronized samples are required by the real debounce.
    for _ in range(3):
        now_ns += 20_000_000
        obj["stamp"] = now_ns / 1e9
        work.update()
    assert work.current is not None
    now_ns += 3_000_000_000
    work.update()
    assert work.current is None


def test_held_playback_cannot_affect_a_replacement_action(protocol):
    p = protocol
    p.client.send_primitive("fist", held=True)
    p.deliver()
    assert p.executor._playback_clock.held and not p.executor._playback_clock.running
    p.client.playback(True)
    p.deliver()
    assert p.executor._playback_clock.running
    old = p.published["commands"][-1]
    p.client.send_primitive("open")
    p.deliver()
    p.bridge.command(old)
    p.sync()
    assert p.executor.active_primitive_name == "open"
    assert not p.executor._playback_clock.held


def test_speed_confirmation_and_capability_negotiation(protocol):
    p = protocol
    request = p.client.set_speed(1.5)
    assert p.client.motion_result(request) is None
    p.deliver()
    assert p.client.motion_result(request)["accepted"]
    assert p.client.motion_result(request)[
        "accepted"
    ]  # Other hands may still be awaiting acknowledgement.
    report = p.bridge.report()
    report.pop("motion_controls")
    p.client.catalog.put(json.dumps(report))
    for operation in [
        lambda: p.client.set_speed(2),
        lambda: p.client.send_primitive("fist", held=True),
        lambda: p.client.live([0.1] * 6),
    ]:
        with pytest.raises(PrimitiveRejectedError, match="升级"):
            operation()


def test_live_stop_rejects_late_updates(protocol):
    p = protocol
    p.client.live([0.1] * 6)
    p.deliver()
    assert p.executor.positions == [0.1] * 6

    p.client.live([0.2] * 6)
    late = p.published["commands"][-1]
    p.client.stop()
    p.deliver()
    p.bridge.command(late)
    p.sync()
    assert p.executor.active_primitive_name is None
    assert p.executor.positions == [0.1] * 6


def test_preview_can_replace_pending_live_stop_without_waiting_on_control_thread(protocol):
    p = protocol
    p.client.live([0.1] * 6)
    p.deliver()
    p.client.stop()
    stop = p.published["commands"][-1]
    p.client.preview(p.entry)
    preview = p.published["commands"][-1]
    p.bridge.command(stop)
    p.bridge.command(preview)
    p.sync()
    assert p.client.preview_active
    assert p.executor.active_primitive_name.startswith("named_pose:")


def test_standalone_ros_context_readonly_configuration_and_topic_overrides(monkeypatch):
    """The standalone client uses its own ROS context without publishing commands."""
    import uuid

    from hand_gesture_primitives.workbench_ros import RosContext
    from std_msgs.msg import String

    monkeypatch.setenv("ROS_DOMAIN_ID", "187")
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    fixture = RosContext.acquire()
    client = None
    received = []
    prefix = "/workbench_fixture_" + uuid.uuid4().hex[:8]
    topics = {
        "gesture_status": prefix + "/status",
        "gesture_cmd": prefix + "/command",
        "gesture_result": prefix + "/result",
        "motion_progress": prefix + "/progress",
        "workbench_command": prefix + "/workbench",
        "gesture_catalog": prefix + "/catalog",
        "workbench_result": prefix + "/confirmation",
    }
    try:
        node = fixture.node
        node.declare_parameter("primitive_speed", 1.75)
        node.declare_parameter("control_rate", 30.0)
        publisher = node.create_publisher(String, topics["gesture_status"], 10)
        node.create_timer(0.02, lambda: publisher.publish(String(data="hold|mode=primitive")))
        node.create_subscription(
            String, topics["gesture_cmd"], lambda msg: received.append(msg.data), 10
        )
        node.create_subscription(
            String, topics["workbench_command"], lambda msg: received.append(msg.data), 10
        )
        client = GestureNodeClient("right", speed_scale=0.5, topics=topics, configure=False)
        assert client.speed_scale == 1.75
        assert client.effective_control_rate == 30.0
        assert node.get_parameter("primitive_speed").value == 1.75
        assert client.topics["workbench_command"] == topics["workbench_command"]
        assert not received
    finally:
        if client is not None:
            client.close()
        fixture.release()


def test_stop_acknowledgement_survives_control_thread_poll_and_is_invalidated(protocol):
    p = protocol
    with pytest.raises(RuntimeError, match="stop unconfirmed"):
        p.client.wait_stopped(0)  # Initial idle is not evidence of a stop request.
    p.client.send_primitive("open")
    p.deliver()
    p.client.stop()
    p.deliver()  # The control thread consumes the acknowledgement before close waits.
    assert p.client._pending is None
    p.client.wait_stopped(0)
    p.client.wait_stopped(0)
    p.client.send_primitive("fist")
    p.deliver()
    with pytest.raises(RuntimeError, match="stop unconfirmed"):
        p.client.wait_stopped(0)
    p.client.stop()
    p.deliver()
    p.bridge.node_id = "replacement"
    p.sync()
    with pytest.raises(RuntimeError, match="stop unconfirmed"):
        p.client.wait_stopped(0)


def _pending_gate(protocol):
    p = protocol
    p.bridge.gate_status(SimpleNamespace(data=json.dumps({
        "version": 2, "side": "right", "gate_id": "gate", "ready": True,
    })))
    p.sync()
    p.client.send_primitive("small_warp_by_vision")
    p.deliver()
    request = p.bridge.pending_gate
    assert request is not None
    reply = {
        "request_id": request["request_id"], "node_id": p.bridge.node_id,
        "gate_id": "gate", "accepted": True,
        "expires_at_ns": time.time_ns() + 100_000_000,
        "target": {"label": "cup", "position": [0, 0, .1], "size": [.03] * 3},
    }
    return request, SimpleNamespace(data=json.dumps(reply))


def test_legacy_stop_invalidates_gate_before_any_target_side_effect(protocol):
    from unittest.mock import Mock

    from hand_gesture_primitives.gesture_node import GestureNode
    p = protocol
    _, reply = _pending_gate(p)
    p.bridge.node._cmd_topic = "/right_hand/gesture_cmd"
    p.bridge.node.get_logger = lambda: Mock()
    GestureNode._cmd_callback(p.bridge.node, SimpleNamespace(data="stop"))
    p.bridge.gate_reply(reply)
    assert p.executor.installed == 0
    assert not hasattr(p.executor, "selected_position")
    assert p.bridge.pending_gate is None


@pytest.mark.parametrize("change", ["replace", "mode", "reload", "progress"])
def test_successful_control_change_invalidates_gate(protocol, change):
    from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY
    p = protocol
    _, reply = _pending_gate(p)
    if change == "replace":
        p.executor.set_primitive(PRIMITIVE_REGISTRY["open"]())
    else:
        p.executor.advance_control_epoch()
    installed = p.executor.installed
    p.bridge.gate_reply(reply)
    assert p.executor.installed == installed
    assert not hasattr(p.executor, "selected_position")
    assert p.bridge.pending_gate is None


@pytest.mark.parametrize("invalid", ["expired", "unknown", "bad_progress", "bad_control"])
def test_invalid_command_preserves_pending_gate(protocol, invalid):
    p = protocol
    request, reply = _pending_gate(p)
    command = {**request, "request_id": "invalid", "sequence": request["sequence"] + 1}
    if invalid == "expired":
        command["expires_at_ns"] = time.time_ns() - 1
    elif invalid == "unknown":
        command.update(kind="primitive", name="missing")
    elif invalid == "bad_progress":
        command.update(kind="progress", progress=float("nan"), control_id="bad")
    else:
        command.update(kind="playback", control_id="bad", running=True)
    p.bridge.command(SimpleNamespace(data=json.dumps(command)))
    assert p.bridge.pending_gate is request
    p.bridge.gate_reply(reply)
    assert p.executor.installed == 1


def test_preview_progress_is_correlated_and_rejects_legacy_replacement(protocol):
    from hand_gesture_primitives.primitives.named_pose import NamedPose
    p = protocol
    identity = p.client.preview(p.entry, initial_progress=.2)
    assert identity
    p.deliver()
    primitive = p.executor._active_primitive
    assert primitive.progress_driven and primitive._request_progress == .2
    p.client.progress(.7, control_id=identity)
    p.deliver()
    assert p.executor._active_primitive is primitive
    assert primitive._request_progress == .7
    assert p.bridge.runtime()["control_id"] == identity
    p.executor.set_primitive(NamedPose(p.entry.name, entry=p.entry))
    p.client.progress(.9, control_id=identity)
    p.deliver()
    assert p.executor._active_primitive._request_progress is None
    assert p.bridge.runtime()["control_id"] is None
    assert p.client._command_notice


def test_invalid_initial_progress_preserves_previous_preview(protocol):
    p = protocol
    p.client.preview(p.entry)
    p.deliver()
    previous = p.executor._active_primitive
    with pytest.raises((ValueError, PrimitiveRejectedError)):
        p.client.preview(p.entry, initial_progress=float("nan"))
    assert p.executor._active_primitive is previous


def test_real_executor_natural_completion_keeps_proven_preview_identity(protocol):
    from dataclasses import replace
    from unittest.mock import Mock

    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from test_runtime_control import executor

    p = protocol
    ex = executor()
    p.bridge.executor = p.bridge.node._executor = ex
    entry = replace(p.entry, unit="legacy255")
    p.bridge._install({"kind": "preview", "control_id": "owned", "request_id": "preview"}, entry)
    ex._active_primitive.compute([0.] * 30, 100., PrimitiveContext(hand_type="o6", angle_unit="semantic"))
    ex._trajectory_complete = True
    ex._settle_status = "settled"
    ex._settle_since = time.monotonic()
    ex._playback_clock.finished = True
    ex._compute_fingertip_positions = lambda: None
    ex._publish_fingertips = False
    ex._publish_cmd = Mock()
    ex.publish_user_result = Mock()
    ex._publish_result = Mock()
    ex._tick()  # Real natural done branch exits and unloads the primitive.
    assert ex._active_primitive is None
    runtime = p.bridge.runtime()
    assert runtime["active"] is None and runtime["execution_kind"] == "idle"
    assert runtime["control_id"] == runtime["completed_control_id"] == "owned"
    assert runtime["completed_kind"] == "preview" and runtime["done"]
    assert runtime["settle_status"] == "settled"
    p.client.status, p.client.result = LatestCache(), LatestCache()
    p.client.speed_scale, p.client.effective_control_rate, p.client._last_send_ns = 1., 30., 0
    p.client.status.put("hold|settle=running")
    p.sync()
    assert p.client.snapshot()["done"]
    ex.stop()
    p.client.result.put("DONE: unrelated")
    p.sync()
    runtime = p.client.snapshot()
    assert runtime["control_id"] is None and runtime["completed_control_id"] is None
    assert not runtime["done"]


def _real_named_pose(protocol, *, kind="preview", held=False):
    from dataclasses import replace

    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from test_runtime_control import executor

    p = protocol
    ex = executor()
    ex._last_commanded_hw = []
    ex._result_pub = SimpleNamespace(publish=lambda message: None)
    p.bridge.executor = p.bridge.node._executor = ex
    first = p.entry.effective_keyframes()[0]
    entry = replace(p.entry, unit="legacy255", keyframes=[
        replace(first, name="a"), replace(first, name="b", positions=[40.] * 30),
    ])
    p.bridge._install({"kind": kind, "held": held, "control_id": "owned", "request_id": "install"}, entry)
    primitive = ex._active_primitive
    primitive.compute([0.] * 30, 0., PrimitiveContext(hand_type="o6", angle_unit="semantic"))
    p.sync()
    return ex, primitive


def test_real_hot_keyframe_change_revokes_old_preview_authorization(protocol):
    p = protocol
    ex, primitive = _real_named_pose(p)
    # The bridge's own successful updates keep one authorization across epochs.
    for progress in (.3, .6):
        p.client.progress(progress, control_id="owned")
        p.deliver()
        assert not p.client._command_notice
        assert primitive._runner.progress == progress
        assert p.bridge.runtime()["control_id"] == "owned"
    epoch = ex.control_epoch
    p.client._workbench_send({"kind": "progress", "control_id": "owned", "progress": -.1})
    p.deliver()
    assert p.client._command_notice
    assert ex.control_epoch == epoch and p.bridge.runtime()["control_id"] == "owned"
    p.client.progress(.65, control_id="owned")
    p.deliver()
    assert not p.client._command_notice
    epoch = ex.control_epoch
    assert ex.set_named_pose(primitive.pose_name, keyframe="missing", progress=.2)
    assert ex.control_epoch == epoch and p.bridge.runtime()["control_id"] == "owned"
    assert ex.set_named_pose(primitive.pose_name, keyframe="b", progress=.2) == ""
    assert ex._active_primitive is primitive and primitive.active_keyframe == "b"
    p.client.progress(.9, control_id="owned")
    p.deliver()
    assert primitive._runner.progress == .2
    assert p.client._command_notice
    assert p.bridge.runtime()["control_id"] is None
    assert p.bridge.runtime()["execution_kind"] == "preview"


def test_legacy_float_progress_keeps_saved_pose_recording_but_revokes_held_id(protocol):
    from hand_gesture_primitives.gesture_node import GestureNode

    p = protocol
    ex, primitive = _real_named_pose(p, kind="pose", held=True)
    # Own playback changes must update authorization for subsequent heartbeats.
    p.client._control_id = "owned"
    for running in (True, False, True):
        p.client.playback(running)
        p.deliver()
        assert not p.client._command_notice
        assert ex._playback_clock.running == running
    GestureNode._motion_progress_callback(p.bridge.node, SimpleNamespace(data=.4))
    assert primitive._runner.progress == .4
    p.sync()
    assert p.bridge.runtime()["execution_kind"] == "pose"
    assert p.bridge.runtime()["revision"]
    assert not p.client.recording_reason()
    assert p.bridge.runtime()["control_id"] is None
    p.client.playback(False)
    p.deliver()
    assert p.client._command_notice
    assert ex._playback_clock.running
    assert primitive._runner.progress == .4


def test_real_live_continuation_checks_authorization_epoch(protocol):
    from test_runtime_control import executor
    from test_workbench_core import adapter

    p = protocol
    ex = executor()
    ex._radian_converter = adapter().converter
    ex._result_pub = SimpleNamespace(publish=lambda message: None)
    p.bridge.executor = p.bridge.node._executor = ex
    p.sync()
    for value in (.1, .2):
        p.client.live([value] * 6)
        p.deliver()
        assert not p.client._command_notice
    target = list(ex._last_target)
    ex.set_target_label("external")  # A committed legacy control change, same live object.
    epoch = ex.control_epoch
    p.client.live([.3] * 6)
    p.deliver()
    assert p.client._command_notice
    assert ex._last_target == target and ex.control_epoch == epoch
    assert p.bridge.runtime()["control_id"] is None
