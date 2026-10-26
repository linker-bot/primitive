"""Independent core contracts: no LeRobot, robot connection or ROS context required."""

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from hand_gesture_primitives.joint_state_mapping import ros2_control_joint_names
from hand_gesture_primitives.workbench_catalog import PrimitiveMenu
from hand_gesture_primitives.workbench_core import (
    JointAdapter,
    PoseDraft,
    PrimitiveRejectedError,
    WorkbenchService,
)
from hand_gesture_primitives.workbench_edit import EditLeaseManager, LiveController
from hand_gesture_primitives.workbench_runner import HandPrimitiveRunner


def adapter(side="right"):
    names = [name.split("_hand_", 1)[-1] for name in ros2_control_joint_names("o6", side)]
    return JointAdapter.from_limits("o6", side, names, [0.0] * 6, [0.58, 1.3, 1.6, 1.6, 1.6, 1.6])


def draft(name="custom", q=0.2, revision=""):
    return PoseDraft(
        name=name,
        revision=revision,
        keyframes=[{"name": "close", "positions": [q] * 6, "duration": 0.4}],
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_controller_semantic_hardware_roundtrip(side):
    joints = adapter(side)
    q = np.array([0.1, 0.5, 0.7, 0.2, 1.0, 0.1])
    sem = joints.converter.controller_to_semantic_rad(q)
    np.testing.assert_allclose(joints.converter.semantic_rad_to_controller(sem), q)
    hardware = joints.converter.radians_to_hardware_uint8(q)
    np.testing.assert_allclose(joints.from_hardware(hardware), q)
    assert joints.joints()[0] == {"name": "thumb_cmc_pitch", "lower": 0.0, "upper": 0.58}


def test_pose_service_cas_side_isolation_and_immutable_execution():
    right = WorkbenchService(adapter())
    left = WorkbenchService(adapter("left"))
    saved = right.save(draft())
    with pytest.raises(PrimitiveRejectedError) as conflict:
        right.save(draft())
    assert conflict.value.code == "revision_conflict"
    assert not left.load_library()["public"]["poses"]
    entry = right.entries()["custom"]
    runner = HandPrimitiveRunner("o6", "right", 30, adapter=right.adapter)
    runner.select("named_pose:custom", np.zeros(6), entry=entry)
    previous = runner._active._entry_snapshot
    right.save(draft(q=0.3, revision=saved["revision"]))
    assert runner._active._entry_snapshot == previous
    assert runner._active._entry_snapshot != right.entries()["custom"]
    with pytest.raises(PrimitiveRejectedError, match="revision_conflict"):
        right.delete("custom", saved["revision"])
    newest = right.load_library()["public"]["poses"][0]
    right.delete("custom", newest["revision"])
    assert not right.entries()


@pytest.mark.parametrize("q", [[float("nan")] * 6, [float("inf")] * 6, [0.0] * 5, [2.0] * 6])
def test_invalid_targets_do_not_replace_action(q):
    joints = adapter()
    runner = HandPrimitiveRunner("o6", "right", 30, adapter=joints)
    runner.select("open", np.zeros(6))
    active = runner._active
    with pytest.raises((ValueError, PrimitiveRejectedError)):
        runner.select("fist", np.array(q))
    assert runner._active is active
    with pytest.raises(ValueError):
        runner.live_target(q)
    assert runner._active is active


def test_speed_pause_and_settle_use_distinct_feedback(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("hand_gesture_primitives.workbench_runner.time.monotonic", lambda: now[0])
    runner = HandPrimitiveRunner("o6", "right", 30, adapter=adapter())
    runner.select("pinch", np.zeros(6), held=True)
    now[0] += 0.1
    target = runner.tick(np.zeros(6), feedback_id=1)
    runner.set_running(False)
    elapsed = runner.clock.elapsed
    now[0] += 100
    np.testing.assert_allclose(runner.tick(target, feedback_id=2), target)
    runner.set_speed(2.0)
    assert runner.clock.elapsed == elapsed
    runner.set_running(True)
    now[0] += 1
    target = runner.tick(target, feedback_id=3)
    assert runner.status.playback_state == "completed"
    for _ in range(5):
        runner.tick(target, feedback_id=4)
    assert not runner.status.done
    runner.tick(target, feedback_id=5)
    runner.tick(target, feedback_id=6)
    assert runner.status.done and runner.status.settle_status == "settled"


def test_editor_lease_and_live_share_expiry_and_ownership(monkeypatch):
    now = [100.0]
    monkeypatch.setattr("hand_gesture_primitives.workbench_edit.time.monotonic", lambda: now[0])
    events = []
    edit = EditLeaseManager(
        on_pause=lambda: events.append("pause"), on_resume=lambda: events.append("resume")
    )
    joints = adapter()
    live = LiveController(
        edit,
        adapter=lambda side: joints,
        feedback=lambda side: np.zeros(6),
        send=lambda side, q: events.append(list(q)),
        stop=lambda side: events.append("stop"),
    )
    token = edit.begin()["token"]
    other = edit.begin()["token"]
    first = live.request(token, "right", 1, {joints.joint_names[0]: 0.1})
    assert first["positions"] == [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
    live.request(token, "right", 2, {joints.joint_names[1]: 0.2})
    assert events[-1] == [0.1, 0.2, 0.0, 0.0, 0.0, 0.0]
    with pytest.raises(PrimitiveRejectedError) as busy:
        live.request(other, "right", 1, {joints.joint_names[0]: 0.2})
    assert busy.value.code == "live_busy"
    now[0] += 0.6
    live.poll()
    assert events[-1] == "stop"
    with pytest.raises(PrimitiveRejectedError) as expired:
        live.request(token, "right", 3, {joints.joint_names[0]: 0.2})
    assert expired.value.code == "live_expired"
    live.request(token, "right", 4, stop=True)
    live.request(token, "right", 5, {joints.joint_names[0]: 0.2})
    now[0] += 11
    live.poll()
    assert edit.paused and not edit.leases and events[-1] == "stop"
    edit.resume()
    assert events[-1] == "resume"
    assert not edit.paused


def test_stop_barrier_and_non_neutral_input_block_resume():
    state = SimpleNamespace(ready=False, neutral=False)
    edit = EditLeaseManager(ready=lambda: state.ready, neutral=lambda: state.neutral)
    token = edit.begin()["token"]
    edit.release(token)
    with pytest.raises(PrimitiveRejectedError, match="停止"):
        edit.resume()
    state.ready = True
    with pytest.raises(PrimitiveRejectedError, match="释放"):
        edit.resume()
    state.neutral = True
    assert not edit.resume()["paused"]


def test_menu_ids_never_reused_and_side_preferences_independent():
    menu = PrimitiveMenu("o6")
    menu.configure(["open", "named_pose:alpha"], ["named_pose:alpha"])
    old_id = menu.id_of("named_pose:alpha")
    menu.configure(["open", "named_pose:beta"], ["named_pose:beta"])
    assert menu.id_of("named_pose:beta") > old_id
    assert menu.name_of(old_id) == "named_pose:alpha"
    right, left = WorkbenchService(adapter()), WorkbenchService(adapter("left"))
    right.menu_preferences(["open", "open", "pinch"])
    left.menu_preferences(["fist"])
    assert right.menu_preferences() == ["open", "pinch"]
    assert left.menu_preferences() == ["fist"]


def test_core_executes_with_host_and_ros_imports_forbidden(tmp_path):
    script = r"""
import importlib.abc, sys
class NoHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'linker_lerobot', 'lerobot', 'rclpy', 'ament_index_python', 'std_msgs', 'geometry_msgs'}:
            raise AssertionError('forbidden host dependency: ' + fullname)
sys.meta_path.insert(0, NoHost())
import numpy as np
from hand_gesture_primitives.joint_state_mapping import ros2_control_joint_names
from hand_gesture_primitives.workbench_core import JointAdapter, WorkbenchService
from hand_gesture_primitives.workbench_runner import HandPrimitiveRunner
from hand_gesture_primitives.workbench_client import GestureNodeClient
from hand_gesture_primitives.workbench_edit import EditLeaseManager
names = [n.split('_hand_', 1)[-1] for n in ros2_control_joint_names('o6', 'right')]
a = JointAdapter.from_limits('o6', 'right', names, [0.]*6, [.58,1.3,1.6,1.6,1.6,1.6])
s = WorkbenchService(a)
s.save({'name':'standalone', 'keyframes':[{'name':'frame', 'positions':[.1]*6}]})
r = HandPrimitiveRunner('o6','right',30,adapter=a)
r.select('pinch', np.zeros(6))
assert r.tick(np.zeros(6)).shape == (6,)
assert s.load_library()['public']['poses'][0]['name'] == 'standalone'
assert EditLeaseManager().begin()['paused']
"""
    env = {**os.environ, "HAND_GESTURE_POSE_DIR": str(tmp_path / "isolated")}
    result = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
