# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Host-independent editor leases and incremental live-control arbitration.

The host serializes methods on its control thread and supplies permission, input
pause and runtime callbacks. Expiring a browser lease never resumes host input.
"""

import time
import uuid

from .workbench_core import PrimitiveRejectedError


def _noop():
    return None


class EditLeaseManager:
    def __init__(
        self,
        *,
        editable=_noop,
        on_pause=_noop,
        on_resume=_noop,
        ready=lambda: True,
        neutral=lambda: True,
    ):
        self.editable = editable
        self.on_pause, self.on_resume = on_pause, on_resume
        self.ready, self.neutral = ready, neutral
        self.paused = False
        self.leases = {}
        self.stopping = False
        self.state = {"paused": False, "editors": 0, "ready": True, "neutral": True}

    def poll(self):
        now = time.monotonic()
        expired = [token for token, deadline in self.leases.items() if deadline <= now]
        for token in expired:
            self.leases.pop(token, None)
        neutral = True
        if self.paused:
            try:
                neutral = bool(self.neutral())
            except Exception:
                neutral = False
        ready = True
        if self.stopping:
            ready = bool(self.ready())
            if ready:
                self.stopping = False
        self.state = {
            "paused": self.paused,
            "editors": len(self.leases),
            "ready": ready,
            "neutral": neutral,
        }
        return self.state

    def begin(self):
        self.editable()
        if not self.paused:
            # Pause is sticky even when a stop acknowledgement fails.
            self.paused = True
            self.stopping = True
            self.on_pause()
        token = uuid.uuid4().hex
        self.leases[token] = time.monotonic() + 10.0
        self.poll()
        return {"token": token, **self.state}

    def heartbeat(self, token):
        self.poll()
        if token not in self.leases:
            raise PrimitiveRejectedError("编辑占用已过期，请关闭后重新打开编辑器", "edit_expired")
        self.leases[token] = time.monotonic() + 10.0
        return {"token": token, **self.state}

    def release(self, token):
        self.leases.pop(token, None)
        return self.poll()

    def resume(self):
        self.poll()
        if self.leases:
            raise PrimitiveRejectedError("仍有页面正在编辑原语，请先关闭编辑器")
        if not self.state["ready"]:
            raise PrimitiveRejectedError("停止手势尚未确认，请等待或重试停止")
        if not self.state["neutral"]:
            raise PrimitiveRejectedError("请释放按键、按钮和 clutch，并将摇杆与扳机复位")
        self.on_resume()
        self.paused = False
        return self.poll()


class LiveController:
    def __init__(self, edit, *, adapter, feedback, send, stop, editable=_noop, controllable=_noop):
        self.edit = edit
        self.adapter, self.feedback, self.send = adapter, feedback, send
        self._stop = stop
        self.editable, self.controllable = editable, controllable
        self.sessions = {}
        self.sequences = {}

    def poll(self):
        self.edit.poll()
        for key in list(self.sequences):
            if key[0] not in self.edit.leases:
                self.sequences.pop(key, None)
        for side, live in list(self.sessions.items()):
            if live["active"] and (
                time.monotonic() > live["deadline"] or live["token"] not in self.edit.leases
            ):
                self.stop(side)
                live["expired"] = True

    def stop(self, side):
        live = self.sessions.get(side)
        if live and live["active"]:
            live["active"] = False
            self._stop(side)

    def release(self, token):
        for side, live in self.sessions.items():
            if live["token"] == token:
                self.stop(side)
        return self.edit.release(token)

    def request(self, token, side, sequence, updates=None, stop=False):
        self.editable()
        self.controllable()
        self.edit.poll()
        if token not in self.edit.leases:
            raise PrimitiveRejectedError("编辑占用已过期，请重新打开编辑器", "edit_expired")
        if not self.edit.state["ready"]:
            raise PrimitiveRejectedError("停止手势尚未确认")
        key = (token, side)
        if sequence <= self.sequences.get(key, -1):
            raise PrimitiveRejectedError("实时跟随请求已过期", "live_stale")
        self.sequences[key] = sequence
        live = self.sessions.get(side)
        if live and live["active"] and live["token"] != token:
            raise PrimitiveRejectedError("另一页面正在实时控制此手", "live_busy")
        if stop:
            if live and live["token"] == token:
                self.stop(side)
                live["expired"] = False
            return {"ok": True}
        q = self.feedback(side)
        if q is None:
            self.stop(side)
            raise PrimitiveRejectedError("当前关节反馈缺失或过期")
        if live and live["token"] == token and live.get("expired"):
            raise PrimitiveRejectedError("实时跟随连接已超时，请重新开启跟随", "live_expired")
        adapter = self.adapter(side)
        updates = updates or {}
        if set(updates) - set(adapter.joint_names):
            raise PrimitiveRejectedError("未知关节")
        if not live or not live["active"]:
            if not updates:
                raise PrimitiveRejectedError("实时跟随尚未启动")
            live = {"token": token, "positions": list(q), "active": False}
        positions = list(live["positions"])
        for joint, value in updates.items():
            positions[adapter.joint_names.index(joint)] = value
        try:
            adapter.validate_q(positions)
        except ValueError as exc:
            self.stop(side)
            raise PrimitiveRejectedError(str(exc)) from exc
        self.send(side, positions)
        live.update(positions=positions, deadline=time.monotonic() + 0.5, active=True)
        self.sessions[side] = live
        return {"positions": positions, "sequence": sequence}
