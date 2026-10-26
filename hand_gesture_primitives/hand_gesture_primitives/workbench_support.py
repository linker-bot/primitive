"""Read-only primitive readiness and pose fingerprints shared with the workbench."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

FEEDBACK_MAX_AGE_SEC = 1.0


def pose_revision(entry) -> str:
    if entry is None:
        return ""
    return hashlib.sha256(
        json.dumps(asdict(entry), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def primitive_reason(profile, name: str, signals: dict) -> str:
    """Model overrides are authoritative; runtime sensing must actually be present."""
    from .primitive_catalog import PRIMITIVE_CATALOG

    if name == "named_pose":
        return "请在自定义原语中创建或选择已保存的姿态"
    if not profile.is_primitive_supported(name):
        return profile.reject_user_message(name) or f"{profile.hand_joint} 不支持此原语"
    entry = PRIMITIVE_CATALOG.get(name)
    if entry is None:
        return "原语未声明运行条件"
    if not signals.get("position"):
        return "等待有效、未过期的关节位置反馈"
    if entry.category == "vision":
        if not signals.get("vision"):
            return "当前后端未接入有效视觉目标／FK，或感知数据已过期"
        if entry.gated and not signals.get("gate"):
            return "视觉抓取门控条件尚未满足"
    if entry.category in {"force", "vision"}:
        if name == "tripod_by_force":
            from .tripod_force import load_tripod_force_params

            # Resolved model parameters include the actual feedback source.
            source = load_tripod_force_params(profile.hand_type).feedback_source
        else:
            from .contact_detection import uses_torque_feedback

            source = "torque" if uses_torque_feedback(profile.hand_type) else "current"
        from .tripod_force import feedback_sources

        if not any(signals.get(key) for key in feedback_sources(source)):
            label = {
                "tactile": "触觉",
                "torque": "力矩",
                "current": "电流",
                "motor_current": "电流",
                "auto": "触觉／电流／力矩",
            }.get(source, source)
            return f"当前后端缺少有效{label}反馈，或反馈已过期"
    return ""


def executor_signals(executor):
    import time

    import numpy as np

    selector = getattr(executor, "_select_torque_feedback", None)
    if selector is not None:
        selector()
    now = time.monotonic()

    def fresh(key):
        stamp = getattr(executor, key, 0.0)
        return stamp > 0 and 0 <= now - stamp <= FEEDBACK_MAX_AGE_SEC

    def valid(value, joint_feedback=False):
        if value is not None and joint_feedback:
            reserved = getattr(getattr(executor, "_hand_config", None), "reserved_indices", [])
            value = [v for i, v in enumerate(value) if i not in reserved]
        return (
            value is not None
            and np.size(value) > 0
            and bool(np.isfinite(value).all())
            and bool((np.asarray(value) >= 0).all())
        )

    return {
        "position": fresh("_feedback_received_at") and valid(executor._current_hw_angles),
        "tactile": fresh("_workbench_touch_at") and valid(executor._tactile_pressure),
        "current": fresh("_workbench_current_at")
        and executor._joint_currents_valid
        and valid(executor._joint_currents, True),
        "torque": fresh("_workbench_torque_at") and valid(executor._joint_torque, True),
        "vision": fresh("_workbench_bbox_at")
        and valid(executor._object_size)
        and executor._compute_fingertip_positions() is not None,
        "gate": False,  # Only the authoritative GraspGate may approve a target.
    }


def load_pose_snapshot(model, side):
    from .pose_library import list_poses, load_pose

    entries = {
        name: load_pose(model, name, hand_side=side) for name in list_poses(model, hand_side=side)
    }
    return {name: entry for name, entry in entries.items() if entry is not None}


def pose_reason(entry, has_converter):
    if entry.unit == "rad" and not has_converter:
        return "此弧度姿态需要 URDF radian_converter，外部节点尚未配置"
    return ""


class WorkbenchBridge:
    """Confirmed, expiring commands. ROS callbacks never wait for disk or locks."""

    def __init__(self, node):
        from std_msgs.msg import String

        self.node, self.executor, self.String = node, node._executor, String
        self.node_id = uuid.uuid4().hex
        self.managed = None
        self.managed_primitive = None
        self.managed_epoch = None
        self.execution_kind = "idle"
        self.active_revision = ""
        self.sequence = 0
        self.control_id = None
        self.results = OrderedDict()
        self.client_sequences = {}
        self.pending_gate = None
        self.pending_gate_epoch = None
        self.gate_state = None
        self.gate_received_at = 0.0
        self.entries = {}
        self.library_error = "姿态库正在加载"
        self.pool = ThreadPoolExecutor(1, thread_name_prefix="gesture-library")
        self.library_future = None
        self.next_scan = 0.0
        prefix = f"/{node._side}_hand"
        self.publisher = node.create_publisher(String, prefix + "/gesture_catalog", 10)
        self.result_publisher = node.create_publisher(String, prefix + "/workbench_result", 10)
        self.gate_publisher = node.create_publisher(String, prefix + "/workbench_gate_request", 10)
        self.subscription = node.create_subscription(
            String, prefix + "/workbench_command", self.command, 10
        )
        self.gate_subscription = node.create_subscription(
            String, prefix + "/workbench_gate_result", self.gate_reply, 10
        )
        self.gate_status_subscription = node.create_subscription(
            String, prefix + "/workbench_gate_state", self.gate_status, 10
        )
        self.timer = node.create_timer(0.1, self.publish)

    def close(self):
        self.pool.shutdown(wait=False, cancel_futures=True)

    def _refresh_library(self):
        if self.library_future is not None and self.library_future.done():
            try:
                self.entries = self.library_future.result()
                self.library_error = ""
            except Exception as exc:
                self.library_error = f"姿态库暂不可用：{exc}"
            self.library_future = None
        if self.library_future is None and time.monotonic() >= self.next_scan:
            self.next_scan = time.monotonic() + 0.5
            self.library_future = self.pool.submit(
                load_pose_snapshot, self.node._hand_type, self.node._side
            )

    def runtime(self):
        active = self.executor.active_primitive_name or None
        if active == "hold":
            active = None
        tracked = self._tracks_primitive()
        live = getattr(self.executor, "_live_active", False)
        settled = getattr(self.executor, "_settle_status", "idle") == "settled"
        completed = (active is None and self.managed_primitive is not None
                     and self.managed_primitive.done and settled
                     and self.managed_epoch == self.executor.control_epoch)
        done = completed or (tracked and settled and getattr(self.executor, "_trajectory_complete", False))
        return {
            **self.executor.playback_status(),
            "control_id": self.control_id if self._owns_primitive() or completed or (live and self.managed_epoch == self.executor.control_epoch) else None,
            "completed_control_id": self.control_id if completed else None,
            "completed_kind": self.execution_kind if completed else None,
            "done": bool(done),
            "settle_status": getattr(self.executor, "_settle_status", "idle") if tracked or completed else "idle",
            "active": active,
            "execution_kind": "live" if live else (self.execution_kind if tracked and active else "unknown" if active else "idle"),
            "revision": self.active_revision if tracked else "",
            "sequence": self.sequence,
        }

    def _tracks_primitive(self):
        """Keep provenance after compatible legacy progress updates to the same pose."""
        return (self.managed_primitive is not None
                and self.executor._active_primitive is self.managed_primitive)

    def _owns_primitive(self):
        """Object identity alone cannot authorize edits after a legacy hot change."""
        return self._tracks_primitive() and self.managed_epoch == self.executor.control_epoch

    def _cancel_changed_gate(self):
        if self.pending_gate is not None and self.pending_gate_epoch != self.executor.control_epoch:
            self._cancel_gate("门控请求已被后续控制操作取消")

    @staticmethod
    def _progress(value):
        import math
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("progress must be finite and within 0..1")
        return float(value)

    def gate_status(self, message):
        try:
            state = json.loads(message.data)
            if state.get("version") != 2 or state.get("side") != self.node._side:
                return
            self.gate_state = state
            self.gate_received_at = time.monotonic()
        except (ValueError, TypeError, AttributeError):
            return

    def report(self):
        from .primitive_catalog import PRIMITIVE_CATALOG

        signals = executor_signals(self.executor)
        gate = self.gate_state if time.monotonic() - self.gate_received_at <= 1.0 else None
        signals["gate"] = bool(gate and gate.get("ready"))
        profile = self.node._profile
        reasons = {name: primitive_reason(profile, name, signals) for name in PRIMITIVE_CATALOG}
        for name, entry in PRIMITIVE_CATALOG.items():
            if entry.gated and not signals["gate"]:
                reasons[name] = (gate or {}).get(
                    "reason"
                ) or "等待升级后的 GraspGate 和有效门控状态"
        mode = self.executor.control_mode.value
        if mode != "primitive":
            reasons = {name: f"外部节点处于 {mode} 模式" for name in reasons}
        converter = self.executor._radian_converter is not None
        return {
            "version": 2,
            "motion_controls": 1,
            "progress_controls": 1,
            "node_id": self.node_id,
            "model": profile.hand_type,
            "side": self.node._side,
            "signals": signals,
            "reasons": reasons,
            "mode": mode,
            "poses": {name: pose_revision(entry) for name, entry in self.entries.items()},
            "pose_reasons": {
                name: pose_reason(entry, converter) for name, entry in self.entries.items()
            },
            "library_error": self.library_error,
            "preview": converter,
            "pose_capabilities": {
                "radian_converter": converter,
                "units": ["legacy255"] + (["rad"] if converter else []),
            },
            "runtime": self.runtime(),
            "results": list(self.results.values())[-32:],
        }

    def _send(self, publisher, data):
        publisher.publish(self.String(data=json.dumps(data, ensure_ascii=False, allow_nan=False)))

    def _finish(self, data, accepted, reason=""):
        self.sequence += 1
        result = {
            "version": 2,
            "node_id": self.node_id,
            "request_id": data.get("request_id", ""),
            "accepted": accepted,
            "reason": reason,
            "runtime": self.runtime(),
        }
        self.results[result["request_id"]] = result
        while len(self.results) > 128:
            self.results.popitem(last=False)
        self._send(self.result_publisher, result)
        if not accepted:
            self.executor.publish_user_result("rejected:" + reason)
        return result

    def _cancel_gate(self, reason):
        if self.pending_gate is not None:
            old, self.pending_gate = self.pending_gate, None
            self.pending_gate_epoch = None
            self._finish(old, False, reason)

    def publish(self):
        try:
            self._refresh_library()
            self._cancel_changed_gate()
            if self.pending_gate and time.time_ns() >= self.pending_gate["expires_at_ns"]:
                self._cancel_gate("门控请求已超时")
            report = self.report()
            if self._tracks_primitive():
                reason = (
                    ("等待有效关节反馈" if not report["signals"]["position"] else "")
                    if self.execution_kind in {"pose", "preview"}
                    else report["reasons"].get(self.managed, "原语已失效")
                )
                if reason:
                    self.executor.stop()
                    self.managed, self.execution_kind, self.active_revision = None, "idle", ""
                    self.sequence += 1
                    self.executor.publish_user_result("rejected:" + reason)
            self._send(self.publisher, self.report())
        except Exception as exc:
            self.node.get_logger().warning(f"原语目录暂不可用: {exc}")

    def _install(self, data, entry=None):
        from .primitives import PRIMITIVE_REGISTRY
        from .primitives.named_pose import NamedPose

        kind = data["kind"]
        primitive = (
            NamedPose(entry.name, entry=entry, progress=(self._progress(data["initial_progress"]) if "initial_progress" in data else None))
            if entry is not None
            else PRIMITIVE_REGISTRY[data["name"]]()
        )
        reason = self.executor.set_primitive(primitive)
        if reason:
            raise ValueError(reason)
        self._cancel_changed_gate()
        self.control_id = data.get("control_id")
        self.managed_primitive = primitive
        self.managed_epoch = self.executor.control_epoch
        if data.get("held"):
            clock = self.executor._playback_clock
            clock.held = True
            clock.set_running(False, time.monotonic())
        self.managed = primitive.name
        self.execution_kind = kind
        self.active_revision = pose_revision(entry) if kind == "pose" else ""
        self._finish(data, True)

    def command(self, message):
        from .pose_library import _parse_entry
        from .primitive_catalog import PRIMITIVE_CATALOG

        data = {}
        try:
            data = json.loads(message.data)
            if not isinstance(data, dict):
                raise ValueError("无效命令")
            if data.get("version") != 2 or data.get("node_id") != self.node_id:
                raise ValueError("节点版本或实例已变化，请刷新")
            request_id, client_id, sequence = (
                data["request_id"],
                data["client_id"],
                int(data["sequence"]),
            )
            if not request_id or not client_id:
                raise ValueError("缺少命令标识")
            if request_id in self.results:
                self._send(self.result_publisher, self.results[request_id])
                return
            if self.pending_gate and request_id == self.pending_gate["request_id"]:
                return
            if sequence <= self.client_sequences.get(client_id, -1):
                raise ValueError("旧命令已失效")
            self.client_sequences[client_id] = sequence
            if not time.time_ns() < data["expires_at_ns"] <= time.time_ns() + 3_000_000_000:
                raise ValueError("命令已过期")
            report = self.report()
            kind = data["kind"]
            if kind == "stop":
                self.executor.stop()
                self._cancel_changed_gate()
                self.managed_primitive = None
                self.control_id = None
                self.managed, self.execution_kind, self.active_revision = None, "idle", ""
                self._finish(data, True)
                return
            if kind == "speed":
                from rclpy.parameter import Parameter
                result = self.node.set_parameters_atomically([
                    Parameter("primitive_speed", value=float(data["speed"]))
                ])
                if not result.successful:
                    raise ValueError(result.reason)
                self._finish(data, True)
                return
            if report["mode"] != "primitive" or not report["signals"]["position"]:
                raise ValueError("节点模式或关节反馈尚未就绪")
            if kind == "progress":
                progress = self._progress(data["progress"])
                if (not data.get("control_id") or data["control_id"] != self.control_id
                        or not self._owns_primitive() or self.execution_kind != "preview"):
                    raise ValueError("预览控制权已变化")
                reason = self.executor.apply_motion_progress(progress)
                if reason:
                    raise ValueError(reason)
                self.managed_epoch = self.executor.control_epoch
                self._cancel_changed_gate()
                self._finish(data, True)
                return
            if kind == "playback":
                if not data.get("control_id") or data["control_id"] != self.control_id or not self._owns_primitive():
                    raise ValueError("动作控制权已变化")
                self.executor.hold_playback(bool(data["running"]))
                self.managed_epoch = self.executor.control_epoch
                self._cancel_changed_gate()
                self._finish(data, True)
                return
            if kind == "live":
                if not data.get("control_id"):
                    raise ValueError("缺少实时编辑控制标识")
                if not data.get("begin") and (
                    data["control_id"] != self.control_id
                    or self.managed_epoch != self.executor.control_epoch
                ):
                    raise ValueError("实时编辑控制权已变化")
                if not data.get("begin") and not getattr(self.executor, "_live_active", False):
                    raise ValueError("实时跟随已超时，请重新开启跟随")
                self.executor.live_target(data["positions"])
                self._cancel_changed_gate()
                self.managed_primitive = None
                self.managed_epoch = self.executor.control_epoch
                self.control_id = data["control_id"]
                self.managed, self.execution_kind, self.active_revision = None, "live", ""
                self._finish(data, True)
                return
            if data.get("held"):
                from .workbench_motion import hold_reason
                reason = hold_reason(self.node._hand_type, "named_pose:" + data["name"] if kind == "pose" else data["name"])
                if reason:
                    raise ValueError(reason)
            if kind == "primitive":
                name = data["name"]
                reason = report["reasons"].get(name, "未知原语")
                if reason:
                    raise ValueError(reason)
                if PRIMITIVE_CATALOG[name].gated:
                    self._cancel_gate("门控请求已被新命令取消")
                    self.pending_gate = data
                    self.pending_gate_epoch = self.executor.control_epoch
                    self._send(
                        self.gate_publisher, {**data, "gate_id": self.gate_state["gate_id"]}
                    )
                else:
                    self._install(data)
                return
            if not self.node._profile.is_primitive_supported("named_pose"):
                raise ValueError("此手型不支持自定义原语")
            if kind == "pose":
                if self.library_error:
                    raise ValueError(self.library_error)
                entry = self.entries.get(data["name"])
                if entry is None or pose_revision(entry) != data["revision"]:
                    raise ValueError("自定义姿态版本不一致，请共享姿态库后刷新")
                reason = pose_reason(entry, report["preview"])
                if reason:
                    raise ValueError(reason)
            elif kind == "preview":
                if not report["preview"]:
                    raise ValueError("外部节点缺少 URDF 换算器，无法试播")
                raw = data["entry"]
                if raw.get("space") != "semantic_v3" or raw.get("unit") != "rad":
                    raise ValueError("试播必须使用 semantic_v3 + rad")
                entry = _parse_entry(
                    "draft_" + request_id.replace("-", ""),
                    raw,
                    self.executor._hand_config.num_joints,
                )
                if entry is None:
                    raise ValueError("试播姿态格式无效")
            else:
                raise ValueError("未知工作台原语命令")
            self._install(data, entry)
        except Exception as exc:
            self._finish(data if isinstance(data, dict) else {}, False, str(exc))

    def gate_reply(self, message):
        data = None
        try:
            reply = json.loads(message.data)
            data = self.pending_gate
            if (
                not data
                or reply.get("request_id") != data["request_id"]
                or reply.get("node_id") != self.node_id
            ):
                return
            if not self.gate_state or reply.get("gate_id") != self.gate_state["gate_id"]:
                return
            if self.pending_gate_epoch != self.executor.control_epoch:
                self._cancel_changed_gate()
                return
            self.pending_gate = None
            self.pending_gate_epoch = None
            if time.time_ns() >= data["expires_at_ns"] or time.time_ns() >= reply.get(
                "expires_at_ns", 0
            ):
                raise ValueError("门控批准已过期")
            if not reply.get("accepted"):
                raise ValueError(reply.get("reason") or "门控未通过")
            report = self.report()
            reason = report["reasons"].get(data["name"], "未知原语")
            if reason:
                raise ValueError(reason)
            # Apply the approved target atomically, before selecting the primitive.
            import numpy as np
            from geometry_msgs.msg import PoseStamped

            target = reply["target"]
            position = np.asarray(target["position"], dtype=float)
            size = None if target.get("size") is None else np.asarray(target["size"], dtype=float)
            if position.shape != (3,) or not np.isfinite(position).all():
                raise ValueError("门控目标位置无效")
            if size is not None and (size.shape != (3,) or not np.isfinite(size).all() or np.any(size < 0)):
                raise ValueError("门控目标尺寸无效")
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = map(float, position)
            pose.pose.orientation.w = 1.0
            missing = object()
            previous = {key: getattr(self.executor, key, missing) for key in (
                "_object_pose", "_object_label", "_object_size",
                "_workbench_object_at", "_workbench_bbox_at",
            )}
            try:
                self.executor._object_pose_callback(pose)
                self.executor._selected_label_callback(self.String(data=target["label"]))
                if size is not None:
                    self.executor._object_size = size
                    self.executor._workbench_bbox_at = time.monotonic()
                self._install(data)
            except Exception:
                for key, value in previous.items():
                    if value is missing:
                        if hasattr(self.executor, key):
                            delattr(self.executor, key)
                    else:
                        setattr(self.executor, key, value)
                raise

        except Exception as exc:
            if data:
                self._finish(data, False, str(exc))
