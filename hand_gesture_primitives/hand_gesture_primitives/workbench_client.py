# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Thin client for an externally running hand_gesture_primitives ``gesture_node``.

Only command/progress topics are published; gesture_node owns the hand writer. Topics follow ``hand_gesture_primitives.topic_defaults``.
"""

from __future__ import annotations

import json
import logging
import math
import time
import uuid
from collections import deque
from contextlib import suppress

from .workbench_cache import LatestCache

logger = logging.getLogger(__name__)


class GestureNodeClient:
    def __init__(
        self,
        side: str,
        speed_scale: float = 1.0,
        *,
        context_factory=None,
        topics=None,
        configure=True,
    ) -> None:
        from std_msgs.msg import Float32, String

        topics_override = topics

        try:
            from hand_gesture_primitives.topic_defaults import side_hand_topics

            topics = side_hand_topics(side)
        except ImportError:
            p = f"/{side}_hand"
            topics = {
                "gesture_cmd": f"{p}/gesture_cmd",
                "motion_progress": f"{p}/motion_progress",
                "gesture_status": f"{p}/gesture_status",
                "gesture_result": f"{p}/gesture_result",
            }
        if topics_override:
            topics.update(topics_override)
        self.side = side
        self.topics = topics
        if context_factory is None:
            from .workbench_ros import RosContext

            context_factory = RosContext.acquire
        self._ctx = context_factory()
        self.speed_scale = speed_scale
        self._last_send_ns = 0
        self._closed = False
        self._init_protocol()
        self.catalog = LatestCache()
        self._subscriptions = []
        try:
            self.effective_control_rate = self._configure_speed(speed_scale, configure=configure)
        except Exception:
            self._ctx.release()
            raise
        self._cmd = self._ctx.publisher(String, topics["gesture_cmd"])
        self._prog = self._ctx.publisher(Float32, topics["motion_progress"])
        self.status, self.result = LatestCache(), LatestCache()
        for topic, cache in [
            (topics["gesture_status"], self.status),
            (topics["gesture_result"], self.result),
        ]:
            self._subscriptions.append(
                self._ctx.node.create_subscription(
                    String, topic, lambda msg, c=cache: c.put(msg.data), self._ctx.sensor_qos()
                )
            )
        self._workbench_cmd = self._ctx.publisher(
            String, topics.get("workbench_command", f"/{side}_hand/workbench_command")
        )
        self._subscriptions.append(
            self._ctx.node.create_subscription(
                String,
                topics.get("gesture_catalog", f"/{side}_hand/gesture_catalog"),
                lambda msg: self.catalog.put(msg.data),
                10,
            )
        )
        self._subscriptions.append(
            self._ctx.node.create_subscription(
                String,
                topics.get("workbench_result", f"/{side}_hand/workbench_result"),
                lambda msg: self.confirmation.put(msg.data),
                10,
            )
        )
        self._String, self._Float32 = String, Float32

    def _init_protocol(self):
        self._client_id = uuid.uuid4().hex
        self._sequence = 0
        self._node_id = None
        self._pending = None
        self._last_stop = None
        self._confirmed_stop = None
        self._uncertain = False
        self._runtime = {"active": None, "execution_kind": "idle", "sequence": -1}
        self._outstanding = {}
        self._accepted_poses = deque()
        self._command_notice = ""
        self.confirmation = LatestCache()
        self._motion_replies = {}
        self._motion_requests = {}
        self._control_id = None
        self._live_id = None
        self._playback_sent = None
        self._playback_at = 0.0

    def _discover_status_node(self, timeout_s: float = 5.0) -> str:
        """Wait for one fully identified publisher before accessing parameters.

        DDS endpoint discovery can precede ROS node-name discovery, especially
        over host/container UDP. Unknown names are not parameter-service names.
        Multiple endpoints remain an error, even if they share a ROS node name.
        """
        deadline = time.monotonic() + timeout_s
        publishers = []
        while time.monotonic() < deadline:
            publishers = self._ctx.node.get_publishers_info_by_topic(self.topics["gesture_status"])
            if len(publishers) > 1:
                names = [f"{p.node_namespace.rstrip('/')}/{p.node_name}" for p in publishers]
                raise ValueError(
                    f"{self.side}: expected one gesture status publisher, found {names}"
                )
            if publishers:
                publisher = publishers[0]
                name, namespace = publisher.node_name, publisher.node_namespace
                if (
                    name
                    and name != "_NODE_NAME_UNKNOWN_"
                    and namespace.startswith("/")
                    and namespace != "/_NODE_NAMESPACE_UNKNOWN_"
                ):
                    return f"{namespace.rstrip('/')}/{name}"
            time.sleep(0.02)
        names = [f"{p.node_namespace.rstrip('/')}/{p.node_name}" for p in publishers]
        raise ValueError(
            f"{self.side}: expected one gesture status publisher on {self.topics['gesture_status']}, "
            f"found {names} after {timeout_s:g}s; check Framework gesture_node and ROS transport"
        )

    def _configure_speed(self, speed, *, configure=True):
        from rcl_interfaces.srv import GetParameters, SetParametersAtomically
        from rclpy.parameter import Parameter, parameter_value_to_python

        node = self._discover_status_node()

        def call(kind, service, request):
            client = self._ctx.node.create_client(kind, node + service)
            future = None
            try:
                if not client.wait_for_service(timeout_sec=1.0):
                    raise ValueError(
                        f"{node}: parameter service unavailable; upgrade gesture_node"
                    )
                future = client.call_async(request)
                limit = time.monotonic() + 2.0
                while not future.done() and time.monotonic() < limit:
                    time.sleep(0.01)
                if not future.done():
                    raise ValueError(f"{node}: parameter request timed out")
                return future.result()
            finally:
                if future is not None and not future.done():
                    future.cancel()
                self._ctx.node.destroy_client(client)

        response = call(
            GetParameters,
            "/get_parameters",
            GetParameters.Request(names=["primitive_speed", "control_rate"]),
        )
        values = [parameter_value_to_python(p) for p in response.values]
        if len(values) != 2 or any(v is None for v in values):
            raise ValueError(
                f"{node}: primitive_speed unsupported; upgrade hand_gesture_primitives and restart the node"
            )
        if not configure:
            self.speed_scale = float(values[0])
            return float(values[1])
        response = call(
            SetParametersAtomically,
            "/set_parameters_atomically",
            SetParametersAtomically.Request(
                parameters=[Parameter("primitive_speed", value=float(speed)).to_parameter_msg()]
            ),
        )
        if not response.result.successful:
            raise ValueError(f"{node}: {response.result.reason}")
        verified = call(
            GetParameters, "/get_parameters", GetParameters.Request(names=["primitive_speed"])
        )
        if parameter_value_to_python(verified.values[0]) != speed:
            raise ValueError(f"{node}: primitive_speed readback mismatch")
        return float(values[1])

    def snapshot(self):
        report = self.synchronize()
        status = (self.latest_status() or "") if self.status.age_s() <= 2.0 else ""
        parts = dict(part.split("=", 1) for part in status.split("|") if "=" in part)
        result = self.result.get()
        fresh = (
            result is not None
            and self.result.age_s() <= 2.0
            and result.receive_time_ns > self._last_send_ns
        )
        text = result.value if fresh else ""
        try:
            speed = float(parts.get("speed", self.speed_scale))
        except ValueError:
            speed = self.speed_scale
        try:
            progress = float(parts.get("progress", -1.0))
            if not math.isfinite(progress):
                progress = -1.0
        except (TypeError, ValueError):
            progress = -1.0
        return {
            "progress": progress,
            **{
                key: self._runtime.get(key)
                for key in ("playback_state", "overall_progress", "keyframe")
            },
            "control_id": self._runtime.get("control_id"),
            "completed_control_id": self._runtime.get("completed_control_id"),
            "completed_kind": self._runtime.get("completed_kind"),
            "node_id": self._node_id,
            "active": self._runtime["active"] if report else status.split("|", 1)[0] or None,
            "execution_kind": self._runtime["execution_kind"] if report else "legacy",
            "command_pending": self._pending is not None,
            "state_confirmed": not self._uncertain,
            "done": self._runtime.get("done", False) if report else text.startswith("DONE:"),
            "feasible": not text.startswith(("SETTLE_TIMEOUT", "rejected")),
            "hold_reason": self._command_notice or text,
            "settle_status": self._runtime.get("settle_status", parts.get("settle", "unknown")) if report else parts.get("settle", "unknown"),
            "primitive_backend": "gesture_node",
            "preview_active": self.preview_active,
            "primitive_speed": self._runtime.get("primitive_speed", speed),
            "effective_control_rate": self.effective_control_rate,
        }

    def capabilities(self):
        sample = self.catalog.get()
        if sample is None or self.catalog.age_s() > 2.0:
            return None
        try:
            report = json.loads(sample.value)
            return report if isinstance(report, dict) and report.get("version") in {1, 2} else None
        except (ValueError, TypeError):
            return None

    def synchronize(self):
        report = self.capabilities()
        if not report or report.get("version") != 2:
            if self._node_id is not None:
                self._uncertain = True
            return None
        node_id = report.get("node_id")
        if not node_id or not isinstance(report.get("runtime"), dict):
            self._uncertain = True
            return None
        if self._node_id != node_id:
            self._confirmed_stop = None
            self._last_stop = None
            self._uncertain = self._node_id is not None
            self._node_id = node_id
            self._runtime = {"active": None, "execution_kind": "unknown", "sequence": -1}
            if self._pending:
                self._command_notice = "外部节点已重启；请停止手势并确认状态"
                self._pending = None
            self._outstanding.clear()
        runtime = report["runtime"]
        if runtime.get("sequence", -1) >= self._runtime["sequence"]:
            self._runtime = runtime
        results = list(report.get("results", []))
        sample = self.confirmation.get()
        if sample is not None:
            with suppress(ValueError, TypeError):
                results.append(json.loads(sample.value))
        for result in results:
            if result.get("node_id") != self._node_id:
                continue
            request = self._outstanding.pop(result.get("request_id"), None)
            if request is None:
                continue
            if result.get("request_id") in self._motion_requests:
                self._motion_replies[result["request_id"]] = result
            if (
                result.get("accepted")
                and request.get("entry") is not None
                and request["kind"] == "pose"
            ):
                self._accepted_poses.append(request["entry"])
            runtime = result.get("runtime", {})
            if runtime.get("sequence", -1) >= self._runtime["sequence"]:
                self._runtime = runtime
            if (
                self._last_stop
                and result.get("request_id") == self._last_stop["request_id"]
                and result.get("accepted")
                and runtime.get("active") is None
                and runtime.get("execution_kind") == "idle"
            ):
                self._confirmed_stop = {
                    **self._last_stop,
                    "runtime_sequence": runtime.get("sequence", -1),
                }
            if self._pending and result.get("request_id") == self._pending["request_id"]:
                self._pending = None
                self._uncertain = False
                self._command_notice = (
                    "" if result.get("accepted") else result.get("reason", "命令被拒绝")
                )
        if self._confirmed_stop and (
            self._runtime.get("active") is not None
            or self._runtime.get("execution_kind") != "idle"
            or self._runtime.get("sequence") != self._confirmed_stop["runtime_sequence"]
        ):
            self._confirmed_stop = None
        if self._pending and time.monotonic() >= self._pending["deadline"]:
            self._uncertain = True
            self._command_notice = "命令确认超时，执行状态未确认；请停止手势并等待确认"
        return report

    @property
    def preview_active(self):
        return self._runtime.get("execution_kind") == "preview" or bool(
            self._pending and self._pending["kind"] == "preview"
        )

    def recording_reason(self):
        self.synchronize()
        if self._pending or self._uncertain or self._runtime.get("execution_kind") == "unknown":
            return "外部手势执行状态未确认，请停止手势并等待确认"
        if self.preview_active:
            return "请先停止自定义草稿试播，再开始录制"
        return ""

    def take_accepted_poses(self):
        self.synchronize()
        result = list(self._accepted_poses)
        self._accepted_poses.clear()
        return result

    def _workbench_send(self, data, entry=None):
        from copy import deepcopy

        from .workbench_core import PrimitiveRejectedError

        report = self.synchronize()
        if not report and (data["kind"] != "stop" or not self._node_id):
            raise PrimitiveRejectedError("请升级外部节点以支持 v2 执行确认")
        continuous = {"playback", "live", "speed", "progress"}
        supersede = data["kind"] in continuous or (
            self._pending and self._pending["kind"] in continuous
        )
        # The editor stops its live stream before previewing. Both commands use
        # the same sequenced publisher; preview may replace that pending stop.
        supersede = supersede or (
            data["kind"] == "preview" and self._pending and self._pending["kind"] == "stop"
        )
        if data["kind"] != "stop" and (self._uncertain or (self._pending and not supersede)):
            raise PrimitiveRejectedError("上一条命令尚未确认，请等待或停止手势")
        self._sequence += 1
        request_id = uuid.uuid4().hex
        payload = {
            **data,
            "version": 2,
            "node_id": self._node_id,
            "client_id": self._client_id,
            "request_id": request_id,
            "sequence": self._sequence,
            "expires_at_ns": time.time_ns()
            + (300_000_000 if data["kind"] in {"playback", "live"} else 2_000_000_000),
        }
        self._confirmed_stop = None
        self._last_stop = (
            {"request_id": request_id, "node_id": self._node_id}
            if data["kind"] == "stop"
            else None
        )
        self._pending = {
            "request_id": request_id,
            "kind": data["kind"],
            "deadline": time.monotonic() + 2.0,
        }
        self._outstanding[request_id] = {"kind": data["kind"], "entry": deepcopy(entry)}
        self._command_notice = ""
        msg = self._String()
        msg.data = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        self._last_send_ns = time.time_ns()
        self._workbench_cmd.publish(msg)
        # Continuous commands supersede older samples; keep correlation bounded.
        while len(self._outstanding) > 128:
            self._outstanding.pop(next(iter(self._outstanding)))
        return request_id

    def send_primitive(self, name, entry=None, *, held=False):
        from hand_gesture_primitives.primitive_catalog import PRIMITIVE_CATALOG
        from hand_gesture_primitives.workbench_support import pose_revision

        from .workbench_core import PrimitiveRejectedError

        if held:
            self.require_motion()
        control_id = uuid.uuid4().hex if held else None
        extra = {"held": True, "control_id": control_id} if held else {}
        if entry is not None:
            self._workbench_send(
                {"kind": "pose", "name": entry.name, "revision": pose_revision(entry), **extra},
                entry,
            )
        elif self.synchronize():
            self._workbench_send({"kind": "primitive", "name": name, **extra})
        else:
            item = PRIMITIVE_CATALOG.get(name)
            if (
                self._node_id
                or not item
                or item.category in {"force", "vision"}
                or name == "named_pose"
            ):
                raise PrimitiveRejectedError("此原语需要升级后的外部节点及 v2 执行确认")
            self.send(name)
        self._control_id, self._live_id = control_id, None
        self._playback_sent = None

    def supports_motion(self):
        return (self.capabilities() or {}).get("motion_controls") == 1

    def require_motion(self):
        from .workbench_core import PrimitiveRejectedError

        if not self.supports_motion():
            raise PrimitiveRejectedError("外部节点需升级并重启以支持暂停、实时调速和编辑跟随")

    def playback(self, running):
        if not self._control_id:
            return
        if running == self._playback_sent and time.monotonic() - self._playback_at < 0.1:
            return
        self._workbench_send(
            {"kind": "playback", "control_id": self._control_id, "running": running}
        )
        self._playback_sent, self._playback_at = running, time.monotonic()

    def live(self, positions):
        from .workbench_core import PrimitiveRejectedError

        self.require_motion()
        self.synchronize()
        if self._command_notice:
            raise PrimitiveRejectedError(self._command_notice)
        begin = self._live_id is None
        live_id = self._live_id or uuid.uuid4().hex
        self._workbench_send(
            {"kind": "live", "control_id": live_id, "begin": begin, "positions": positions}
        )
        self._live_id, self._control_id = live_id, None

    def set_speed(self, speed):
        self.require_motion()
        request = self._workbench_send({"kind": "speed", "speed": speed})
        self._motion_requests[request] = time.monotonic() + 2.0
        while len(self._motion_requests) > 128:
            old = next(iter(self._motion_requests))
            self._motion_requests.pop(old, None)
            self._motion_replies.pop(old, None)
        return request

    def motion_result(self, request):
        self.synchronize()
        reply = self._motion_replies.get(request)
        if reply is None and time.monotonic() >= self._motion_requests[request]:
            reply = {"accepted": False, "reason": "外部节点调速确认超时，请检查实际倍率"}
        if reply is not None:
            self._motion_replies[request] = reply
        return reply

    def require_progress(self):
        from .workbench_core import PrimitiveRejectedError
        if (self.capabilities() or {}).get("progress_controls") != 1:
            raise PrimitiveRejectedError("外部节点需升级并重启以支持确认的预览进度控制")

    def preview(self, entry, *, initial_progress=None):
        from dataclasses import asdict

        from .workbench_core import PrimitiveRejectedError

        report = self.synchronize()
        if not report or not report.get("preview") or report.get("mode") != "primitive":
            raise PrimitiveRejectedError("外部节点尚不支持草稿试播，请升级并配置 URDF 换算器")
        extra = {}
        if initial_progress is not None:
            self.require_progress()
            from .workbench_support import WorkbenchBridge
            extra["initial_progress"] = WorkbenchBridge._progress(initial_progress)
        control_id = uuid.uuid4().hex
        self._workbench_send({"kind": "preview", "entry": asdict(entry), "control_id": control_id, **extra})
        self._control_id, self._live_id = control_id, None
        return control_id

    def send(self, command: str) -> None:
        self._last_stop = self._confirmed_stop = None
        msg = self._String()
        msg.data = command
        self._last_send_ns = time.time_ns()
        self._cmd.publish(msg)
        logger.info("[%s hand] gesture_cmd -> %s", self.side, command)

    def stop(self) -> None:
        self._control_id, self._live_id = None, None
        if self.synchronize() or self._node_id:
            self._workbench_send({"kind": "stop"})
        else:
            self.send("stop")

    def progress(self, p: float, *, control_id=None) -> None:
        if control_id is not None:
            self.require_progress()
            from .workbench_support import WorkbenchBridge
            value = WorkbenchBridge._progress(p)
            self._workbench_send({"kind": "progress", "control_id": control_id, "progress": value})
            return
        msg = self._Float32()
        msg.data = float(p)
        self._prog.publish(msg)

    def wait_stopped(self, timeout_s: float = 2.0) -> None:
        """Require the external executor's stop acknowledgement before releasing ROS.

        This confirms that the gesture executor is idle, not physical motor standstill.
        Legacy nodes cannot confirm a correlated stop and must report that limitation.
        """
        request = self._last_stop
        if request is None:
            raise RuntimeError(f"{self.side}: gesture stop unconfirmed (no pending stop request)")
        node_id = request["node_id"]
        deadline = time.monotonic() + max(0.0, timeout_s)
        while True:
            report = self.synchronize()
            if self._node_id != node_id:
                raise RuntimeError(f"{self.side}: gesture stop unconfirmed (node restarted)")
            confirmed = self._confirmed_stop
            if (
                report
                and confirmed
                and confirmed["request_id"] == request["request_id"]
                and not self._uncertain
                and not self._command_notice
                and self._runtime.get("active") is None
                and self._runtime.get("execution_kind") == "idle"
                and self._runtime.get("sequence") == confirmed["runtime_sequence"]
            ):
                return
            if self._pending is None:
                raise RuntimeError(
                    f"{self.side}: gesture stop unconfirmed: {self._command_notice}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._uncertain = True
                raise TimeoutError(f"{self.side}: gesture stop acknowledgement timed out")
            time.sleep(min(0.01, remaining))

    def latest_status(self) -> str | None:
        s = self.status.get()
        return s.value if s else None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for subscription in self._subscriptions:
            self._ctx.node.destroy_subscription(subscription)
        for publisher in [self._cmd, self._prog, self._workbench_cmd]:
            self._ctx.node.destroy_publisher(publisher)
        self._ctx.release()
