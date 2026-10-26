"""Teach 面板状态机 — 无 Web/ROS 依赖，可单测。"""

from __future__ import annotations

import time
from enum import Enum
from typing import Callable, Dict, List, Optional, Sequence


class PanelState(str, Enum):
    DISCONNECTED = "DISCONNECTED"
    PRIMITIVE_IDLE = "PRIMITIVE_IDLE"
    TEACH_EDIT = "TEACH_EDIT"
    PLAYBACK = "PLAYBACK"
    ERROR = "ERROR"


class PanelController:
    """Authority 切换 + Mute 门控。发布关节仅在 TEACH_EDIT。"""

    def __init__(
        self,
        num_joints: int,
        *,
        publish_rate: float = 15.0,
        status_timeout_sec: float = 3.0,
        send_cmd: Optional[Callable[[str], None]] = None,
        publish_joints: Optional[Callable[[List[float]], None]] = None,
        now: Optional[Callable[[], float]] = None,
        couple_map: Optional[Dict[int, int]] = None,
        return_to_teach_after_playback: bool = True,
    ) -> None:
        self._num_joints = int(num_joints)
        self._publish_period = 1.0 / max(1.0, float(publish_rate))
        self._status_timeout = float(status_timeout_sec)
        self._send_cmd = send_cmd or (lambda _c: None)
        self._publish_joints = publish_joints or (lambda _p: None)
        self._now = now or time.monotonic
        self._couple_map: Dict[int, int] = dict(couple_map or {})
        # 从示教发起的试播成功结束后是否自动 mode teach
        self.return_to_teach_after_playback = bool(return_to_teach_after_playback)

        self.state = PanelState.DISCONNECTED
        self.muted = True
        self.control_mode = "unknown"
        self.active_primitive = "hold"
        self.last_error = ""
        self.hint = ""
        self.targets: List[float] = [128.0] * self._num_joints
        self.feedback: Optional[List[float]] = None
        self._last_status_t = 0.0
        self._last_pub_t = 0.0
        self._pending_targets: Optional[List[float]] = None
        self._playback_name = ""
        self._playback_keyframe = ""
        self._playback_progress: Optional[float] = None
        self._playback_progress_mode = False
        self._playback_auto_segment = False
        self.editing_pose_name: str = ""
        self.editing_keyframe: str = ""
        # 用户已拖动滑条改过 targets；解除 Mute 时勿用反馈覆盖（二次编辑丢改的根因）
        self._targets_dirty: bool = False
        # 等 gesture_status mode=teach 后再 unmute / 下发载入姿态
        self._pending_load_joints: Optional[List[float]] = None
        # 等 mode=primitive 后再发 pose 参数串（name [kf] [p]），避免 TEACH 下被拒
        self._pending_pose_cmd: str = ""
        # 已发出 pose 指令；见过 named_pose 执行后再允许以 hold/DONE 结束试播
        self._playback_pose_sent: bool = False
        self._playback_seen_named: bool = False
        self._resume_teach_after_playback: bool = False
        # 从 gesture_status / result 解析的 motion 字段
        self.motion_drive: str = ""
        self.motion_keyframe: str = ""
        self.motion_progress: float = 0.0
        self._publish_progress: Optional[Callable[[float], None]] = None

    @property
    def can_publish_joints(self) -> bool:
        return (not self.muted) and self.state == PanelState.TEACH_EDIT

    def on_ws_disconnected(self) -> None:
        """浏览器断开：Mute；若仍在 TEACH 则退出，避免机载停在 teach。"""
        self.muted = True
        self._pending_targets = None
        self._pending_load_joints = None
        in_teach = self.state == PanelState.TEACH_EDIT or self.control_mode == "teach"
        if self.state == PanelState.PLAYBACK:
            self._clear_playback_flags()
            self.state = PanelState.PRIMITIVE_IDLE
            self.hint = "连接断开：试播已中止（已 Mute）"
        if in_teach:
            self._send_cmd("mode primitive")
            self.state = PanelState.PRIMITIVE_IDLE
            self.hint = "连接断开：已 Mute 并退出 TEACH"
        elif not self.hint:
            self.hint = "连接断开：已 Mute"

    def on_ws_connected(self) -> None:
        self.hint = ""
        if self.state == PanelState.TEACH_EDIT and self.control_mode == "teach":
            self.muted = False
            self._pull_feedback_to_targets()
        elif self.state == PanelState.DISCONNECTED and self._last_status_t > 0:
            self._sync_state_from_mode()

    def _clear_playback_flags(self) -> None:
        self._playback_name = ""
        self._playback_keyframe = ""
        self._playback_progress = None
        self._playback_progress_mode = False
        self._playback_auto_segment = False
        self._pending_pose_cmd = ""
        self._playback_pose_sent = False
        self._playback_seen_named = False
        self._resume_teach_after_playback = False

    @staticmethod
    def _pose_cmd_args(
        name: str,
        keyframe: Optional[str] = None,
        progress: Optional[float] = None,
        *,
        auto_segment: bool = False,
    ) -> str:
        parts = [(name or "").strip().lower()]
        kf = (keyframe or "").strip().lower()
        if kf:
            parts.append(kf)
        if auto_segment and kf and progress is None:
            parts.append("auto")
        elif progress is not None:
            parts.append(f"{float(progress):.4g}")
        return " ".join(p for p in parts if p)

    def begin_playback(
        self,
        name: str,
        *,
        keyframe: Optional[str] = None,
        progress: Optional[float] = None,
        auto_segment: bool = False,
    ) -> None:
        self.muted = True
        self._pending_targets = None
        self._pending_load_joints = None
        self.last_error = ""
        n = (name or "").strip().lower()
        if not n:
            self.last_error = "姿态名为空"
            self.hint = "姿态名为空"
            return
        kf = (keyframe or "").strip().lower()
        prog: Optional[float] = None
        if progress is not None:
            try:
                prog = max(0.0, min(1.0, float(progress)))
            except (TypeError, ValueError):
                self.last_error = "progress 无效"
                self.hint = "progress 无效"
                return
        want_auto_seg = bool(auto_segment) and bool(kf) and prog is None
        # 指定 kf（含仅 hold p=0）或显式 progress → PROGRESS，不自动结束
        # auto_segment → 单段 AUTO，可 DONE 结束
        progress_mode = (bool(kf) or prog is not None) and not want_auto_seg
        args = self._pose_cmd_args(n, kf or None, prog, auto_segment=want_auto_seg)
        self._playback_name = n
        self._playback_keyframe = kf
        self._playback_progress = prog
        self._playback_progress_mode = progress_mode
        self._playback_auto_segment = want_auto_seg
        self._playback_pose_sent = False
        self._playback_seen_named = False
        was_teach = self.control_mode == "teach" or self.state == PanelState.TEACH_EDIT
        self._resume_teach_after_playback = bool(
            self.return_to_teach_after_playback and was_teach and not progress_mode
        )
        self.state = PanelState.PLAYBACK
        if want_auto_seg:
            self.hint = f"单帧 AUTO「{n}/{kf}」…"
        elif progress_mode:
            self.hint = f"细控「{args}」（PROGRESS，点停止结束）…"
        else:
            self.hint = f"试播「{n}」…"
        if was_teach or self.control_mode != "primitive":
            self._pending_pose_cmd = ""
            self._send_cmd("mode primitive")
            self._send_cmd(f"pose {args}")
            self._playback_pose_sent = True
        else:
            self._pending_pose_cmd = ""
            self._playback_pose_sent = True
            self._send_cmd(f"pose {args}")
        if prog is not None:
            self.motion_progress = prog
        if kf:
            self.motion_keyframe = kf
        self.motion_drive = "auto" if (want_auto_seg or not progress_mode) else "progress"

    def _parse_motion_fields(self, parts: Sequence[str]) -> None:
        for p in parts:
            if p.startswith("drive="):
                self.motion_drive = p.split("=", 1)[1].strip()
            elif p.startswith("keyframe="):
                self.motion_keyframe = p.split("=", 1)[1].strip()
            elif p.startswith("progress="):
                try:
                    self.motion_progress = max(0.0, min(1.0, float(p.split("=", 1)[1])))
                except ValueError:
                    pass

    def _finish_playback(self, *, resume_teach: Optional[bool] = None) -> None:
        """结束试播。resume_teach=None 时按发起时记录的标志决定是否回示教。"""
        do_resume = (
            self._resume_teach_after_playback if resume_teach is None else bool(resume_teach)
        )
        do_resume = do_resume and self.return_to_teach_after_playback
        self._clear_playback_flags()
        self.muted = True
        # 试播结束以机上实际姿态为准，允许随后 pull 同步滑条
        self._targets_dirty = False
        if do_resume:
            self.enter_teach()
            self.hint = "试播完成：正在回到示教（等待 mode=teach）"
        else:
            self.state = PanelState.PRIMITIVE_IDLE
            self.hint = "试播完成：仍在 primitive，点「开始示教」后再调滑条"

    def _flush_pending_pose(self) -> bool:
        """若已回到 primitive 且仍有待试播姿态，发送 pose。"""
        if self.state != PanelState.PLAYBACK:
            return False
        if not self._pending_pose_cmd or self.control_mode != "primitive":
            return False
        self._send_cmd(f"pose {self._pending_pose_cmd}")
        self._pending_pose_cmd = ""
        self._playback_pose_sent = True
        return True

    def on_status(self, raw: str, *, stamp: Optional[float] = None) -> None:
        self._last_status_t = stamp if stamp is not None else self._now()
        text = (raw or "").strip()
        if text.startswith("rejected:"):
            self.last_error = text
            if self.state == PanelState.PLAYBACK:
                # 抢跑 pose 时可能 rejected:teach_mode：保留试播，等 mode ACK 再补发
                if text.endswith("_mode") and self._playback_name:
                    self._pending_pose_cmd = self._pose_cmd_args(
                        self._playback_name,
                        self._playback_keyframe or None,
                        self._playback_progress,
                        auto_segment=self._playback_auto_segment,
                    )
                    self._playback_pose_sent = False
                    self.hint = "试播被拒（模式未就绪），等待切回 primitive 后重试"
                    return
                self._finish_playback(resume_teach=False)
                self.hint = f"试播失败：{text}"
            elif self.state == PanelState.TEACH_EDIT:
                self.muted = True
                self.hint = "指令被拒：已 Mute，待 status 确认 mode=teach 后恢复"
            return
        # pose_list 会短暂覆盖 status 为 POSES:...，勿当成原语名、勿打断试播
        if text.startswith("POSES:"):
            return

        parts = [p for p in text.split("|") if p]
        if not parts:
            return
        self.active_primitive = parts[0]
        mode = "unknown"
        for p in parts[1:]:
            if p.startswith("mode="):
                mode = p.split("=", 1)[1].strip().lower()
        if mode != "unknown":
            self.control_mode = mode
        self._parse_motion_fields(parts[1:])

        if self.state == PanelState.DISCONNECTED:
            self._sync_state_from_mode()
            # 断线恢复时若仍有待试播，拉回 PLAYBACK 并补发
            if self._pending_pose_cmd and self.control_mode == "primitive":
                self.state = PanelState.PLAYBACK
                self._flush_pending_pose()
        elif self.state == PanelState.PLAYBACK:
            if self._flush_pending_pose():
                return
            if self.active_primitive.startswith("named_pose:"):
                self._playback_seen_named = True
                self._pending_pose_cmd = ""  # 已在执行，无需再补发
                return
            # PROGRESS 细控不自动结束（永不 DONE）；AUTO 见 hold/init 才结束
            if self._playback_progress_mode:
                return
            if self._playback_seen_named and self.active_primitive in ("hold", "init"):
                self._finish_playback()
            elif (
                self.control_mode == "teach"
                and self._playback_pose_sent
                and self._playback_seen_named
            ):
                # 外部已回到 teach：结束试播标志并 unmute
                self._clear_playback_flags()
                self.state = PanelState.TEACH_EDIT
                self.muted = False
                self._targets_dirty = False
                self._pull_feedback_to_targets()
                self.hint = "试播结束：已回到示教"
        elif self.state == PanelState.TEACH_EDIT and self.control_mode == "teach":
            # mode teach 已确认：允许发关节；若有待载入姿态则立刻下发
            if self.muted:
                self.muted = False
                # 有待载入姿态或用户未保存的拖动：禁止用反馈冲掉 targets
                if self._pending_load_joints is None and not self._targets_dirty:
                    self._pull_feedback_to_targets()
                if self.hint.startswith("试播完成：正在回到示教") or self.hint.startswith(
                    "指令被拒：已 Mute"
                ):
                    self.hint = "示教已就绪，可拖动滑条"
            if self._pending_load_joints is not None:
                vals = list(self._pending_load_joints)
                self._pending_load_joints = None
                self.targets = vals
                self._targets_dirty = False
                self._pending_targets = list(vals)
                self._publish_joints(vals)
                self._last_pub_t = self._now()
        elif self.state == PanelState.TEACH_EDIT and self.control_mode == "primitive":
            # 外部切回 primitive
            self.muted = True
            self._pending_load_joints = None
            self.state = PanelState.PRIMITIVE_IDLE
        elif self.state == PanelState.PRIMITIVE_IDLE and self.control_mode == "teach":
            self.state = PanelState.TEACH_EDIT
            self.muted = False
            if not self._targets_dirty:
                self._pull_feedback_to_targets()

    def on_result(self, raw: str) -> None:
        text = (raw or "").strip()
        # 始终尝试解析 motion 字段（细控 UI）
        if "|" in text:
            self._parse_motion_fields([p for p in text.split("|") if p][1:])
        if self.state != PanelState.PLAYBACK:
            return
        if text.startswith("EXECUTING:") and "named_pose:" in text:
            self._playback_seen_named = True
            self._pending_pose_cmd = ""
            self._playback_pose_sent = True
            return
        if text.startswith("DONE:"):
            if self._playback_progress_mode:
                return
            if self._playback_pose_sent or self._playback_seen_named:
                self._finish_playback()
            return
        if text == "IDLE":
            if self._playback_progress_mode:
                return
            if self._playback_seen_named:
                self._finish_playback()
            return

    def tick(self, *, stamp: Optional[float] = None) -> None:
        now = stamp if stamp is not None else self._now()
        if self._last_status_t > 0 and (now - self._last_status_t) > self._status_timeout:
            if self.state != PanelState.DISCONNECTED:
                self.state = PanelState.DISCONNECTED
                self.muted = True
                self._pending_targets = None
            return

        # status 可能丢一拍：定时补发待试播 pose
        self._flush_pending_pose()

        if not self.can_publish_joints:
            return
        if self._pending_targets is None:
            return
        if (now - self._last_pub_t) < self._publish_period:
            return
        self._last_pub_t = now
        self._publish_joints(list(self._pending_targets))

    def enter_teach(self) -> None:
        self.last_error = ""
        self.hint = "正在进入示教（等待 mode=teach）"
        self._send_cmd("mode teach")
        # 先 Mute，等 gesture_status 含 mode=teach 再 unmute，避免与 Executor 抢手
        self.state = PanelState.TEACH_EDIT
        self.muted = True
        self._pending_targets = None

    def exit_teach(self) -> None:
        self.muted = True
        self._pending_targets = None
        self._pending_load_joints = None
        self._resume_teach_after_playback = False
        self._send_cmd("mode primitive")
        self.state = PanelState.PRIMITIVE_IDLE
        self.hint = "已退出示教"

    def _apply_coupling(self, vals: List[float]) -> List[float]:
        """tip 跟随 mid（只读耦合）。"""
        if not self._couple_map:
            return vals
        out = list(vals)
        for tip, mid in self._couple_map.items():
            if 0 <= tip < len(out) and 0 <= mid < len(out):
                out[tip] = out[mid]
        return out

    def set_joints(self, positions: Sequence[float]) -> Optional[str]:
        """更新面板 targets。

        即使 Mute / 未进 TEACH，也保留本地拖动值（避免 UI 被 snapshot 打回 128）。
        仅在 can_publish_joints 时排队下发。
        """
        if len(positions) != self._num_joints:
            return f"joint_count:{len(positions)}!={self._num_joints}"
        vals = self._apply_coupling([max(0.0, min(255.0, float(v))) for v in positions])
        self.targets = vals
        self._targets_dirty = True
        if not self.can_publish_joints:
            self._pending_targets = None
            return "muted_or_not_teach"
        self._pending_targets = vals
        return None

    def flush_joints(self) -> None:
        """松手：立即发一帧终值（仍受 Mute 约束）。"""
        if not self.can_publish_joints or self._pending_targets is None:
            return
        self._last_pub_t = 0.0
        self.tick()

    def update_feedback(self, positions: Sequence[float]) -> None:
        if len(positions) != self._num_joints:
            return
        self.feedback = self._apply_coupling([float(v) for v in positions])

    def capture_positions(self) -> Optional[List[float]]:
        """保存优先用反馈；无反馈则 None。"""
        if self.feedback is None:
            return None
        return self._apply_coupling(list(self.feedback))

    def set_playback_progress(self, progress: float) -> Optional[str]:
        """同段热更新 progress（优先 motion_progress topic）。"""
        try:
            p = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            return "progress 无效"
        if self.state != PanelState.PLAYBACK or not self._playback_name:
            return "未在试播/细控中"
        self.motion_progress = p
        self.motion_drive = "progress"
        self._playback_progress_mode = True
        self._playback_progress = p
        if self._publish_progress is not None:
            self._publish_progress(p)
        else:
            # 无 topic 时退回 gesture_cmd（同段热更新由 executor.set_named_pose 处理）
            args = self._pose_cmd_args(
                self._playback_name,
                self._playback_keyframe or None,
                p,
            )
            self._send_cmd(f"pose {args}")
        self.hint = (
            f"细控 progress={p:.3f}"
            f"{(' kf=' + self._playback_keyframe) if self._playback_keyframe else ''}"
        )
        return None

    def scrub_to(
        self,
        name: str,
        *,
        keyframe: str,
        progress: float = 0.0,
    ) -> Optional[str]:
        """细控定位到某关键帧的 local progress。

        同名同段 → 热更新；换段/新启 → begin_playback（换段重锚）。
        """
        n = (name or "").strip().lower()
        kf = (keyframe or "").strip().lower()
        if not n:
            return "姿态名为空"
        if not kf:
            return "关键帧名为空"
        try:
            p = max(0.0, min(1.0, float(progress)))
        except (TypeError, ValueError):
            return "progress 无效"
        same = (
            self.state == PanelState.PLAYBACK
            and self._playback_name == n
            and self._playback_keyframe == kf
            and self._playback_seen_named
        )
        if same:
            return self.set_playback_progress(p)
        self.begin_playback(n, keyframe=kf, progress=p)
        if self.last_error:
            return self.last_error
        return None

    def stop_playback(self, *, resume_teach: Optional[bool] = None) -> None:
        """手动结束试播/细控。"""
        if self.state != PanelState.PLAYBACK and not self._playback_name:
            self.hint = "当前无试播"
            return
        self._send_cmd("stop")
        self._finish_playback(resume_teach=resume_teach)

    def reinit(self) -> None:
        """重新初始化：Mute → primitive → init，清空编辑态。"""
        self.muted = True
        self._pending_targets = None
        self._pending_load_joints = None
        self._clear_playback_flags()
        self.editing_pose_name = ""
        self.editing_keyframe = ""
        self.last_error = ""
        self.hint = "正在重新初始化…"
        if self.control_mode == "teach" or self.state == PanelState.TEACH_EDIT:
            self._send_cmd("mode primitive")
        self._send_cmd("init")
        self.state = PanelState.PRIMITIVE_IDLE

    def load_pose_for_edit(
        self,
        name: str,
        positions: Sequence[float],
        *,
        enter_teach: bool = True,
        keyframe: str = "",
    ) -> Optional[str]:
        """载入库中姿态到滑条，供二次修改。"""
        n = (name or "").strip().lower()
        if not n:
            return "姿态名为空"
        if len(positions) != self._num_joints:
            return f"joint_count:{len(positions)}!={self._num_joints}"
        vals = self._apply_coupling([max(0.0, min(255.0, float(v))) for v in positions])
        self.targets = vals
        self._targets_dirty = False
        self.editing_pose_name = n
        self.editing_keyframe = (keyframe or "").strip().lower()
        self.last_error = ""
        if enter_teach:
            self.state = PanelState.TEACH_EDIT
            if self.control_mode == "teach":
                self.muted = False
                self._pending_load_joints = None
                self._pending_targets = list(vals)
                self._publish_joints(list(vals))
                self._last_pub_t = self._now()
            else:
                # 等 mode=teach 确认后再下发，避免与 PRIMITIVE Executor 抢命令
                self.muted = True
                self._pending_targets = None
                self._pending_load_joints = list(vals)
                self._send_cmd("mode teach")
        return None

    def clear_editing_pose(self) -> None:
        self.editing_pose_name = ""
        self.editing_keyframe = ""

    def mark_targets_clean(self) -> None:
        """保存成功后清除脏标记，允许后续用反馈同步。"""
        self._targets_dirty = False

    def sync_from_feedback(self) -> Optional[str]:
        """用最新 hand_state 覆盖滑条目标。"""
        if self.feedback is None:
            return "no_hand_state"
        vals = self._apply_coupling(list(self.feedback))
        self.targets = vals
        self._targets_dirty = False
        self._pending_targets = list(vals)
        return None

    def snapshot(self) -> dict:
        return {
            "panel_state": self.state.value,
            "mode": self.control_mode,
            "muted": self.muted,
            "active_primitive": self.active_primitive,
            "targets": list(self.targets),
            "feedback": list(self.feedback) if self.feedback else None,
            "last_error": self.last_error,
            "hint": self.hint,
            "playback_name": self._playback_name,
            "playback_keyframe": self._playback_keyframe,
            "playback_progress_mode": self._playback_progress_mode,
            "editing_pose_name": self.editing_pose_name,
            "editing_keyframe": self.editing_keyframe,
            "motion_drive": self.motion_drive,
            "motion_keyframe": self.motion_keyframe,
            "motion_progress": self.motion_progress,
            "num_joints": self._num_joints,
            "return_to_teach_after_playback": self.return_to_teach_after_playback,
        }

    def _sync_state_from_mode(self) -> None:
        if self.control_mode == "teach":
            self.state = PanelState.TEACH_EDIT
            self.muted = False
            if not self._targets_dirty and self._pending_load_joints is None:
                self._pull_feedback_to_targets()
        elif self.control_mode in ("primitive", "stream"):
            self.state = PanelState.PRIMITIVE_IDLE
            self.muted = True
        else:
            self.state = PanelState.PRIMITIVE_IDLE
            self.muted = True

    def _pull_feedback_to_targets(self) -> None:
        if self.feedback is not None:
            vals = self._apply_coupling(list(self.feedback))
            self.targets = vals
            self._targets_dirty = False
            self._pending_targets = list(vals)
