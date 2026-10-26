"""Read-only readiness and correlated approvals using GraspGate's real geometry."""

import json
import time
import uuid

import numpy as np
from scipy.spatial.transform import Rotation


def evaluate_gate(gate):
    now = gate.get_clock().now().nanoseconds / 1e9
    if gate._tcp is None or not 0 <= now - gate._tcp["stamp"] <= gate._data_timeout:
        return None, "无有效 TCP 数据"
    gate._cleanup(now)
    tcp = gate._tcp
    try:
        if not np.isfinite(tcp["pos"]).all() or not np.isfinite(tcp["quat"]).all():
            raise ValueError("非有限 TCP")
        rotation = Rotation.from_quat(tcp["quat"]).as_matrix()
    except ValueError:
        return None, "TCP 位姿无效"
    palm = tcp["pos"] + rotation @ np.array([0, 0, gate._palm_z])
    tip = tcp["pos"] + rotation @ np.array([0, 0, gate._tip_z])
    low, high = min(palm[2], tip[2]) - gate._height_tol, max(palm[2], tip[2]) + gate._height_tol
    candidates = []
    for objects, evaluator in [(gate._bboxes, gate._eval_obj), (gate._poses, gate._eval_pose)]:
        for label, obj in objects.items():
            if not 0 <= now - obj["stamp"] <= gate._data_timeout:
                continue
            result = evaluator(obj, rotation, tcp["pos"], low, high)
            if result["ok"]:
                candidates.append((label, obj, result))
        if candidates:
            break
    if not candidates:
        return None, gate._reject_reason(rotation, tcp["pos"], low, high)
    label, obj, result = min(candidates, key=lambda item: item[2]["cz"])
    return {"label": label, "object": obj, "result": result, "tcp_stamp": tcp["stamp"]}, ""


class WorkbenchGate:
    def __init__(self, gate):
        from std_msgs.msg import String

        self.gate, self.String = gate, String
        self.gate_id = uuid.uuid4().hex
        self.last_sample = None
        self.last_label = None
        self.count = 0
        self.current = None
        self.reason = "等待感知数据"
        prefix = f"/{gate._side}_hand/workbench_gate_"
        self.publisher = gate.create_publisher(String, prefix + "state", 10)
        self.results = gate.create_publisher(String, prefix + "result", 10)
        self.subscription = gate.create_subscription(String, prefix + "request", self.command, 10)
        self.timer = gate.create_timer(0.05, self.publish)

    def update(self):
        current, reason = evaluate_gate(self.gate)
        if current is None:
            self.last_label, self.last_sample, self.count, self.current = None, None, 0, None
            self.reason = reason
            return
        sample = (current["label"], current["object"]["stamp"], current["tcp_stamp"])
        if current["label"] != self.last_label:
            self.count = 0
        if sample != self.last_sample:
            self.count += 1
        self.last_sample, self.last_label = sample, current["label"]
        self.current = current if self.count >= self.gate._de_bounce else None
        self.reason = (
            "" if self.current else f"门控目标稳定确认中（{self.count}/{self.gate._de_bounce}）"
        )

    def publish(self):
        try:
            self.update()
        except Exception as exc:
            self.current, self.reason = None, f"门控数据无效：{exc}"
            self.count = 0
        self.publisher.publish(
            self.String(
                data=json.dumps(
                    {
                        "version": 2,
                        "side": self.gate._side,
                        "gate_id": self.gate_id,
                        "ready": self.current is not None,
                        "reason": self.reason,
                    },
                    ensure_ascii=False,
                )
            )
        )

    def command(self, message):
        request = {}
        try:
            request = json.loads(message.data)
            if request.get("version") != 2 or request.get("gate_id") != self.gate_id:
                raise ValueError("门控节点已变化")
            from .primitive_catalog import gated_primitive_names

            if request.get("name") not in gated_primitive_names():
                raise ValueError("未知门控原语")
            if time.time_ns() >= request["expires_at_ns"]:
                raise ValueError("门控请求已过期")
            self.update()
            if self.current is None:
                raise ValueError(self.reason)
            item = self.current
            obj = item["object"]
            target = {
                "label": item["label"],
                "position": list(map(float, obj.get("center", obj.get("pos")))),
                "size": list(map(float, np.asarray(obj["max"]) - obj["min"]))
                if "max" in obj
                else None,
            }
            reply = {"accepted": True, "target": target, "reason": ""}
        except Exception as exc:
            reply = {"accepted": False, "reason": str(exc)}
        if not isinstance(request, dict):
            return
        self.results.publish(
            self.String(
                data=json.dumps(
                    {
                        **reply,
                        "version": 2,
                        "gate_id": self.gate_id,
                        "node_id": request.get("node_id"),
                        "request_id": request.get("request_id"),
                        "expires_at_ns": min(
                            request.get("expires_at_ns", 0), time.time_ns() + 200_000_000
                        ),
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                )
            )
        )
