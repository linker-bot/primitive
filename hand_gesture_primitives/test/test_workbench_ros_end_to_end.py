"""Real ROS executors + real HTTP workbench with software JointState feedback only.

No hardware driver, CAN interface, or SDK is opened. Domain 187 is reserved for
this test; every child is started in its own process group and only those groups
are signalled during teardown. Child logs remain under pytest's tmp_path.
"""

from __future__ import annotations

import copy
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml
from hand_gesture_primitives.joint_state_mapping import ros2_control_joint_names
from hand_gesture_primitives.workbench_core import JointAdapter

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("hand_gesture_primitives_msgs.srv")
pytest.importorskip("hand_force_msg.msg")
pytest.importorskip("robot_perception_msgs.msg")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
Context = pytest.importorskip("rclpy.context").Context
SingleThreadedExecutor = pytest.importorskip("rclpy.executors").SingleThreadedExecutor
JointState = pytest.importorskip("sensor_msgs.msg").JointState
Float64MultiArray = pytest.importorskip("std_msgs.msg").Float64MultiArray
String = pytest.importorskip("std_msgs.msg").String

DOMAIN = 187
STATE_TOPIC = "/workbench_e2e/joint_states"


def _robot_description():
    root = ElementTree.Element("robot", name="workbench_e2e_dual_o6")
    ElementTree.SubElement(root, "link", name="base")
    for side in ("left", "right"):
        names = ros2_control_joint_names("o6", side)
        adapter = JointAdapter.from_limits(
            "o6", side, names, [0.0] * 6, [0.58, 1.3, 1.6, 1.6, 1.6, 1.6]
        )
        for spec in adapter.joints():
            name = spec["name"]
            ElementTree.SubElement(root, "link", name=name + "_link")
            joint = ElementTree.SubElement(root, "joint", name=name, type="revolute")
            ElementTree.SubElement(joint, "parent", link="base")
            ElementTree.SubElement(joint, "child", link=name + "_link")
            ElementTree.SubElement(joint, "axis", xyz="0 0 1")
            ElementTree.SubElement(
                joint,
                "limit",
                lower=str(spec["lower"]),
                upper=str(spec["upper"]),
                effort="1",
                velocity="1",
            )
    return ElementTree.tostring(root, encoding="unicode")


def _child_code(module):
    # A command can reach its executor before DDS discovers this test's observer.
    # Wait at publisher creation so one-shot requests are captured deterministically;
    # this changes only the test process, not production QoS or motion behavior.
    barrier = ""
    if module == "hand_gesture_primitives.teach_panel.main":
        barrier = """import time
from hand_gesture_primitives.workbench_ros import RosContext
_original_publisher = RosContext.publisher
def _observed_publisher(self, kind, topic):
    publisher = _original_publisher(self, kind, topic)
    if topic.endswith('/workbench_command'):
        deadline = time.monotonic() + 8
        while publisher.get_subscription_count() < 2:
            if time.monotonic() >= deadline:
                raise RuntimeError('DDS executor and test observer did not match: ' + topic)
            time.sleep(.02)
    return publisher
RosContext.publisher = _observed_publisher
"""
    return barrier + f"from {module} import main; main()"


class RosWorkbenchHarness:
    def __init__(self, directory):
        self.directory = directory
        self.logs = {}
        self.children = []
        self.lock = threading.Lock()
        self.plants = {
            "left": [0.11, 0.23, 0.34, 0.45, 0.56, 0.67],
            "right": [0.17, 0.29, 0.41, 0.53, 0.65, 0.77],
        }
        self.follow = False
        self.commands = {side: [] for side in self.plants}
        self.requests = {side: [] for side in self.plants}
        self.replies = {}
        self.catalogs = {}
        self.context = Context()
        rclpy.init(args=[], context=self.context, domain_id=DOMAIN)
        self.node = rclpy.create_node("workbench_e2e_feedback", context=self.context)
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.publisher = self.node.create_publisher(JointState, STATE_TOPIC, 10)
        self.replay = {}
        for side in self.plants:
            self.node.create_subscription(
                Float64MultiArray,
                f"/{side}_hand_controller/commands",
                lambda msg, selected=side: self.command(selected, msg),
                10,
            )
            self.node.create_subscription(
                String,
                f"/{side}_hand/gesture_catalog",
                lambda msg, selected=side: self.catalog(selected, msg),
                10,
            )
            self.node.create_subscription(
                String,
                f"/{side}_hand/workbench_command",
                lambda msg, selected=side: self.request_message(selected, msg),
                10,
            )
            self.node.create_subscription(String, f"/{side}_hand/workbench_result", self.reply, 10)
            self.replay[side] = self.node.create_publisher(
                String, f"/{side}_hand/workbench_command", 10
            )
        self.node.create_timer(0.02, self.publish_feedback)
        self.halt = threading.Event()
        self.thread = threading.Thread(target=self.spin, daemon=True)
        self.thread.start()
        with socket.socket() as server:
            server.bind(("127.0.0.1", 0))
            self.port = server.getsockname()[1]
        self.base = f"http://127.0.0.1:{self.port}/api/workbench"

    def spin(self):
        while not self.halt.is_set():
            self.executor.spin_once(timeout_sec=0.05)

    def command(self, side, msg):
        values = list(msg.data)
        with self.lock:
            self.commands[side].append(values)
            if self.follow and len(values) == 6 and np.isfinite(values).all():
                self.plants[side] = values

    def catalog(self, side, msg):
        with self.lock:
            self.catalogs[side] = json.loads(msg.data)

    def request_message(self, side, msg):
        with self.lock:
            self.requests[side].append(json.loads(msg.data))

    def reply(self, msg):
        value = json.loads(msg.data)
        with self.lock:
            self.replies[value.get("request_id")] = value

    def publish_feedback(self):
        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        with self.lock:
            for side, values in self.plants.items():
                message.name.extend(ros2_control_joint_names("o6", side))
                message.position.extend(values)
        self.publisher.publish(message)

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(
                {
                    "plants": self.plants,
                    "catalogs": self.catalogs,
                    "requests": self.requests,
                    "replies": self.replies,
                    "command_counts": {s: len(v) for s, v in self.commands.items()},
                }
            )

    def start_child(self, name, module, params):
        params_file = self.directory / f"{name}.yaml"
        params_file.write_text(yaml.safe_dump({"/**": {"ros__parameters": params}}))
        logfile = self.directory / f"{name}.log"
        self.logs[name] = logfile
        environment = {
            **os.environ,
            "ROS_DOMAIN_ID": str(DOMAIN),
            "ROS_LOCALHOST_ONLY": "1",
            "HAND_GESTURE_POSE_DIR": str(self.directory / "poses"),
            "PYTHONUNBUFFERED": "1",
        }
        source_root = str(Path(__file__).resolve().parents[1])
        environment["PYTHONPATH"] = source_root + os.pathsep + environment.get("PYTHONPATH", "")
        with logfile.open("wb") as output:
            child = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    _child_code(module),
                    "--ros-args",
                    "--params-file",
                    str(params_file),
                    "-r",
                    f"__node:={name}",
                ],
                env=environment,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.children.append((name, child))
        return child

    def log_text(self):
        return "\n".join(
            f"--- {name} ---\n{path.read_text(errors='replace')[-10000:]}"
            for name, path in self.logs.items()
        )

    def wait(self, predicate, timeout=12.0, description="condition"):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            for name, child in self.children:
                if child.poll() is not None:
                    raise AssertionError(
                        f"{name} exited with {child.returncode}\n{self.log_text()}"
                    )
            try:
                last = predicate()
                if last:
                    return last
            except (OSError, urllib.error.URLError, KeyError, ValueError):
                pass
            time.sleep(0.03)
        raise AssertionError(
            f"Timed out waiting for {description}: {last}\n{self.snapshot()}\n{self.log_text()}"
        )

    def http(self, method, path, payload=None, expected=200):
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=4.0) as response:
                status, body = response.status, json.load(response)
        except urllib.error.HTTPError as error:
            status, body = error.code, json.load(error)
        assert status == expected, (method, path, status, body)
        return body

    def idle(self):
        state = self.http("GET", "/state")
        return all(
            not h.get("command_pending")
            and h.get("state_confirmed")
            and h.get("active") in {None, "hold", "idle"}
            for h in state["hands"].values()
        )

    def stop_child(self, child):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGINT)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait(timeout=3)
        self.children = [(name, item) for name, item in self.children if item is not child]
        return child.returncode

    def close(self):
        for _, child in list(reversed(self.children)):
            self.stop_child(child)
        self.halt.set()
        self.thread.join(timeout=2)
        self.executor.shutdown(timeout_sec=1)
        self.node.destroy_node()
        self.context.try_shutdown()


@pytest.mark.timeout(90)
def test_real_dual_o6_nodes_and_ros_http_workbench(tmp_path, monkeypatch):
    monkeypatch.setenv("ROS_DOMAIN_ID", str(DOMAIN))
    monkeypatch.setenv("ROS_LOCALHOST_ONLY", "1")
    harness = RosWorkbenchHarness(tmp_path)
    try:
        description = _robot_description()
        common = {
            "hand_type": "o6",
            "state_topic": STATE_TOPIC,
            "cmd_backend": "ros2_control",
            "robot_description": description,
            "ros2_control_command_units": "radians",
            "control_rate": 30.0,
            "primitive_speed": 1.0,
        }
        for side in ("left", "right"):
            harness.start_child(
                f"gesture_e2e_{side}",
                "hand_gesture_primitives.gesture_node",
                {
                    **common,
                    "hand_side": side,
                    "publish_fingertips": False,
                    "runtime_status_path": str(tmp_path / f"runtime-{side}.json"),
                },
            )
        harness.wait(
            lambda: len(harness.snapshot()["catalogs"]) == 2, description="both gesture nodes"
        )
        panel = harness.start_child(
            "teach_e2e",
            "hand_gesture_primitives.teach_panel.main",
            {
                **common,
                "hand_side": "right",
                "http_port": harness.port,
                "workbench_hands": ["left:o6", "right:o6"],
            },
        )
        harness.wait(
            lambda: all(
                h["feedback_valid"] for h in harness.http("GET", "/state")["hands"].values()
            ),
            description="HTTP workbench and both feedback streams",
        )
        # Distinct non-quantized values expose wrong-side routing and UI-255 conversion.
        initial = harness.snapshot()["plants"]
        for side in ("left", "right"):
            captured = harness.http("GET", f"/capture?side={side}")["positions"]
            np.testing.assert_allclose(captured, initial[side], rtol=0, atol=1e-9)
            catalog = harness.http("GET", f"/catalog?side={side}")
            assert catalog["side"] == side and catalog["model"] == "o6"
            assert any(
                item["name"] == "open" and item["available"] for item in catalog["primitives"]
            )
            harness.wait(
                lambda s=side: (
                    len(
                        harness.node.get_publishers_info_by_topic(f"/{s}_hand_controller/commands")
                    )
                    == 1
                ),
                description=f"one {side} joint writer",
            )
        assert "side=right" in harness.logs["teach_e2e"].read_text()
        assert (
            "cmd=/right_hand/gesture_cmd status=/right_hand/gesture_status"
            in harness.logs["teach_e2e"].read_text()
        )

        speed = harness.http("PUT", "/speed", {"speed": 1.5})
        assert speed == {"speed": 1.5}
        harness.wait(
            lambda: all(
                c["runtime"]["primitive_speed"] == 1.5
                for c in harness.snapshot()["catalogs"].values()
            ),
            description="both speed acknowledgements",
        )
        with harness.lock:
            harness.follow = True
        token = harness.http("POST", "/edit/begin")["token"]
        harness.wait(
            lambda: harness.http("GET", "/state")["input_pause"]["ready"],
            description="edit stop barrier",
        )
        before = harness.snapshot()["plants"]
        harness.http(
            "POST",
            "/live",
            {"token": token, "side": "right", "sequence": 1, "updates": {"index_mcp_pitch": 0.31}},
        )
        harness.wait(
            lambda: abs(harness.snapshot()["plants"]["right"][2] - 0.31) < 1e-9,
            description="right live target",
        )
        after = harness.snapshot()["plants"]
        np.testing.assert_allclose(after["left"], before["left"], atol=1e-9)
        np.testing.assert_allclose(
            np.delete(after["right"], 2), np.delete(before["right"], 2), atol=1e-9
        )
        late = harness.wait(
            lambda: next(
                (
                    m
                    for m in reversed(harness.snapshot()["requests"]["right"])
                    if m["kind"] == "live"
                ),
                None,
            ),
            description="live protocol request",
        )
        harness.http("POST", "/stop", {"side": "right"})
        harness.wait(harness.idle, description="confirmed live stop")
        # Stop holds the executor's last measured feedback, which may precede
        # the loopback plant's most recent command. Observe its actual hold
        # output after the correlated ACK instead of treating a sent target as
        # an already acknowledged measurement.
        stopped_count = harness.snapshot()["command_counts"]["right"]
        harness.wait(
            lambda: harness.snapshot()["command_counts"]["right"] >= stopped_count + 2,
            description="post-stop hold output",
        )
        with harness.lock:
            stopped_target = list(harness.commands["right"][-1])
            replay_start = len(harness.commands["right"])
        rejected = harness.http(
            "POST",
            "/live",
            {"token": token, "side": "right", "sequence": 1, "updates": {"index_mcp_pitch": 0.55}},
            expected=400,
        )
        assert rejected["detail"]["code"] == "live_stale"
        late.update(
            request_id=uuid.uuid4().hex,
            positions=[0.45] * 6,
            expires_at_ns=time.time_ns() + 2_000_000_000,
        )
        harness.replay["right"].publish(String(data=json.dumps(late)))
        reply = harness.wait(
            lambda: harness.snapshot()["replies"].get(late["request_id"]),
            description="reordered request rejection",
        )
        assert not reply["accepted"] and "旧命令" in reply["reason"]
        harness.wait(
            lambda: harness.snapshot()["command_counts"]["right"] >= replay_start + 3,
            description="hold remains unchanged after rejected replay",
        )
        with harness.lock:
            replay_outputs = copy.deepcopy(harness.commands["right"][replay_start:])
        for target in replay_outputs:
            np.testing.assert_allclose(target, stopped_target, rtol=0, atol=1e-9)
        np.testing.assert_allclose(
            harness.snapshot()["plants"]["right"], stopped_target, rtol=0, atol=1e-9
        )
        harness.http("POST", "/edit/release", {"token": token})
        harness.http("POST", "/resume")

        saved = {}
        targets = {
            "left": [0.15, 0.25, 0.35, 0.45, 0.55, 0.65],
            "right": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        }
        for side in ("left", "right"):
            saved[side] = harness.http(
                "PUT",
                "/poses",
                {
                    "side": side,
                    "draft": {
                        "name": "dual_e2e",
                        "keyframes": [
                            {"name": "middle", "positions": [0.12] * 6, "duration": 0.2},
                            {"name": "final", "positions": targets[side], "duration": 0.3},
                        ],
                    },
                },
            )
        for side in ("left", "right"):
            harness.wait(
                lambda s=side: (
                    harness.snapshot()["catalogs"][s]["poses"].get("dual_e2e")
                    == saved[s]["revision"]
                ),
                description=f"{side} shared pose revision",
            )
            harness.wait(
                lambda s=side: any(
                    item["name"] == "named_pose:dual_e2e" and item["available"]
                    for item in harness.http("GET", f"/catalog?side={s}")["primitives"]
                ),
                description=f"{side} pose available",
            )
            harness.http("POST", "/select", {"side": side, "name": "named_pose:dual_e2e"})
        harness.wait(
            lambda: all(
                h["settle_status"] == "settled"
                for h in harness.http("GET", "/state")["hands"].values()
            ),
            description="both physical-feedback settle confirmations",
        )
        for side in ("left", "right"):
            np.testing.assert_allclose(
                harness.http("GET", f"/capture?side={side}")["positions"], targets[side], atol=1e-8
            )
            harness.http("POST", "/stop", {"side": side})
        harness.wait(harness.idle, description="final stopped idle")
        # Use the real legacy websocket and actual Executor completion. AUTO
        # unloads its primitive on ROS; a Mock runner retaining _active would
        # hide loss of the terminal execution identity here.
        from websockets.sync.client import connect

        with connect(f"ws://127.0.0.1:{harness.port}/ws?side=right", open_timeout=4) as ws:
            def receive_legacy(predicate):
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    message = json.loads(ws.recv(timeout=4))
                    assert message.get("type") != "error", message
                    if predicate(message):
                        return message
                raise AssertionError("legacy websocket state did not arrive")

            def legacy_command(kind, **values):
                ws.send(json.dumps({"type": kind, **values}))
                return receive_legacy(lambda message: message.get("action") == kind)

            def legacy_state(state):
                return receive_legacy(lambda message: message.get("type") == "snapshot"
                                      and message.get("panel_state") == state)

            legacy_command("enter_teach")
            receive_legacy(lambda message: message.get("type") == "snapshot" and not message["muted"])
            legacy_command("play_pose", name="dual_e2e")
            returned = legacy_state("TEACH_EDIT")
            assert not returned["muted"] and returned["playback_name"] == ""
            terminal = harness.http("GET", "/state")["hands"]["right"]
            assert terminal["active"] is None and terminal["execution_kind"] == "idle"
            assert terminal["completed_control_id"] and terminal["completed_kind"] == "preview"
            assert terminal["done"] and terminal["settle_status"] == "settled"

            legacy_command("set_prefs", return_to_teach_after_playback=False)
            legacy_command("play_pose", name="dual_e2e")
            finished = legacy_state("PRIMITIVE_IDLE")
            assert finished["muted"] and not finished["return_to_teach_after_playback"]
            assert finished["playback_name"] == ""
            assert harness.http("GET", "/state")["input_pause"]["editors"] == 0

            legacy_command("enter_teach")
            legacy_command("play_pose", name="dual_e2e")
            legacy_command("stop_playback", resume_teach=True)
            assert legacy_state("PRIMITIVE_IDLE")["muted"], "false preference overrides explicit resume"

            legacy_command("set_prefs", return_to_teach_after_playback=True)
            legacy_command("scrub_pose", name="dual_e2e", keyframe="final", progress=1.)
            time.sleep(.35)
            progress_state = legacy_state("PLAYBACK")
            assert progress_state["playback_progress_mode"] and progress_state["muted"]
            legacy_command("stop_playback", resume_teach=False)
            assert legacy_state("PRIMITIVE_IDLE")["muted"]
        harness.wait(harness.idle, description="legacy stop confirmed")
        (tmp_path / "evidence.json").write_text(
            json.dumps(
                {
                    "domain": DOMAIN,
                    "live_target": after["right"],
                    "stopped_hold": stopped_target,
                    "rejected_replay": reply,
                    "replay_output_count": len(replay_outputs),
                    "saved": saved,
                    "state": harness.http("GET", "/state"),
                    "transport": harness.snapshot(),
                },
                indent=2,
            )
        )
        assert harness.stop_child(panel) == 0, harness.log_text()
        for side in ("left", "right"):
            harness.wait(
                lambda s=side: (
                    len(
                        harness.node.get_publishers_info_by_topic(f"/{s}_hand_controller/commands")
                    )
                    == 1
                ),
                description="gesture node remains sole writer after panel shutdown",
            )
        exitcodes = {"teach_e2e": panel.returncode}
        for name, child in list(harness.children):
            exitcodes[name] = harness.stop_child(child)
            assert exitcodes[name] == 0, harness.log_text()
        (tmp_path / "shutdown.json").write_text(json.dumps(exitcodes, indent=2))
    except Exception as error:
        raise AssertionError(
            f"{error}\nROS E2E logs retained at {tmp_path}\n{harness.log_text()}"
        ) from error
    finally:
        harness.close()
