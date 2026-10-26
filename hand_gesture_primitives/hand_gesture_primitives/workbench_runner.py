# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Host-independent tick-driven primitive execution with injected joint conversion.

The host owns feedback freshness, scheduling and the single hardware writer.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .workbench_core import JointAdapter, PrimitiveRejectedError

logger = logging.getLogger(__name__)


@dataclass
class RunnerStatus:
    active: str | None = None
    execution_kind: str = "idle"
    done: bool = False
    feasible: bool = True
    hold_reason: str = ""
    motion_status: str = ""
    progress: float = -1.0
    elapsed_s: float = 0.0
    supports_progress: bool = False
    settle_status: str = "idle"
    playback_state: str = "idle"
    overall_progress: float = -1.0
    keyframe: str = ""
    history: list[str] = field(default_factory=list)


class HandPrimitiveRunner:
    """Drive one hand's gesture primitives tick by tick.

    Args:
        model: hand model key (o6 / l6 / l25 / o20 ...).
        side: "left" | "right".
        control_rate: ticks per second the runner is called at (fps of the record loop).
        joint_names: controller-order names (must match the injected adapter).
    """

    def __init__(
        self,
        model: str,
        side: str,
        control_rate: float,
        joint_names: list[str] | None = None,
        speed_scale: float = 1.0,
        *,
        adapter: JointAdapter,
    ) -> None:
        from hand_gesture_primitives.contact_config import (
            clamp_control_rate_hz,
            load_contact_thresholds,
        )
        from hand_gesture_primitives.hand_profile import load_hand_profile

        self.model = model.lower()
        self.side = side
        self.speed_scale = float(speed_scale)
        if not 0.25 <= self.speed_scale <= 3.0:
            raise ValueError("speed_scale must be between 0.25 and 3.0")
        self.control_rate = clamp_control_rate_hz(float(control_rate))
        self.profile = load_hand_profile(self.model)
        if adapter.model != self.model or adapter.side != side:
            raise ValueError("Joint adapter model/side mismatch")
        self.adapter = adapter
        self.converter = adapter.converter
        self.joint_names = list(joint_names or adapter.joint_names)
        if self.joint_names != adapter.joint_names:
            raise ValueError("Joint adapter order mismatch")
        self.contact_thresholds = load_contact_thresholds().scaled_for_control_rate(
            self.control_rate
        )
        self._settle_since = None
        self._settle_frames = 0
        self._settle_feedback_id = None
        self._active = None
        self._t0 = 0.0
        self._last_target: np.ndarray | None = None
        self._last_cmd_hw: list[float] | None = None
        self.status = RunnerStatus()
        self.clock = None

    # ------------------------------------------------------------------ selection
    def live_target(self, positions):
        target = self.adapter.validate_q(positions).copy()
        self.stop()
        self._last_target = target
        self.status.execution_kind = "live"
        self.status.playback_state = "live"
        return target.copy()

    def supported(self) -> list[str]:
        return list(self.profile.supported_primitives())

    def select(
        self, name: str, current_q: np.ndarray, *, entry=None, execution_kind=None, held=False
    ) -> bool:
        """Prepare before swapping: a bad request never destroys the previous action."""
        from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY
        from hand_gesture_primitives.primitives.named_pose import NamedPose

        key = self.profile.resolve_primitive_name(name)
        if key == "named_pose":
            raise PrimitiveRejectedError("请在自定义原语中选择一个已保存的姿态")
        if key.startswith("named_pose:"):
            if entry is None or not self.profile.is_primitive_supported("named_pose"):
                raise PrimitiveRejectedError("自定义姿态不存在或此手型不支持姿态回放")

            def candidate_factory():
                return NamedPose(entry.name, entry=entry)
        else:
            if not self.profile.is_primitive_supported(key) or key not in PRIMITIVE_REGISTRY:
                return False
            candidate_factory = PRIMITIVE_REGISTRY[key]
        try:
            self.adapter.validate_q(current_q)
            candidate = candidate_factory()
            candidate.on_enter(self._to_semantic(current_q))
            supports_progress = candidate.supports_progress()
            from hand_gesture_primitives.workbench_motion import PlaybackClock, trajectory_duration

            clock = PlaybackClock(
                time.monotonic(),
                self.speed_scale,
                trajectory_duration(
                    self.model,
                    key,
                    entry=entry,
                    start=self._to_semantic(current_q),
                    converter=self.converter,
                ),
                held=held,
            )
        except Exception as exc:
            raise PrimitiveRejectedError(
                f"原语初始化失败：{type(exc).__name__}: {exc}", "primitive_init_failed"
            ) from exc
        self.stop()
        self._active = candidate
        self._t0 = time.monotonic()
        self.clock = clock
        self._settle_since = None
        self._settle_frames = 0
        self._settle_feedback_id = None
        self._last_target = None
        self._last_cmd_hw = None
        self.status = RunnerStatus(
            active=key,
            execution_kind=execution_kind or ("pose" if entry is not None else "primitive"),
            supports_progress=supports_progress,
            settle_status="running",
            playback_state="running",
            history=self.status.history[-20:] + [key],
        )
        return True

    def stop(self) -> None:
        if self._active is not None:
            try:
                self._active.on_exit()
            except Exception:
                logger.exception("primitive cleanup failed")
        self._active = None
        self.clock = None
        self._last_target = None
        self._last_cmd_hw = None
        self.status = RunnerStatus(history=self.status.history)

    def hold(self, current_q, reason=""):
        self.stop()
        self.status.feasible = not bool(reason)
        self.status.hold_reason = reason
        self.status.settle_status = "failed" if reason else "idle"
        if current_q is not None and np.isfinite(current_q).all():
            self._last_target = np.asarray(current_q, dtype=float).copy()
            return self._last_target.copy()
        return None

    def set_running(self, running):
        if self.clock and self.clock.held and self.clock.running != running:
            self.clock.set_running(running, time.monotonic())
            self._playback_status()

    def set_speed(self, speed):
        from hand_gesture_primitives.workbench_motion import validate_speed

        self.speed_scale = validate_speed(speed)
        if self.clock:
            self.clock.set_speed(speed, time.monotonic())

    def _playback_status(self):
        if self.clock:
            for key, value in self.clock.snapshot().items():
                setattr(self.status, key, value)
        self.status.keyframe = getattr(self._active, "active_keyframe", "") or ""

    def set_progress(self, p: float) -> None:
        if self._active is not None and self._active.supports_progress():
            self._active.set_progress(float(np.clip(p, 0.0, 1.0)))
            self.status.progress = float(p)
            self._settle_since = None
            self._settle_frames = 0
            self._settle_feedback_id = None
            self.status.settle_status, self.status.done = "running", False
            if self.clock:
                self.clock.finished = False
            self._playback_status()

    @property
    def active_name(self) -> str | None:
        return self._active.name if self._active is not None else None

    # ------------------------------------------------------------------ stepping
    def tick(self, current_q, force=None, currents=None, *, feedback_id=None):
        if self._active is None:
            return (
                self._last_target.copy()
                if self.status.execution_kind == "live" and self._last_target is not None
                else None
            )
        try:
            return self._tick_active(current_q, force, currents, feedback_id=feedback_id)
        except Exception as exc:
            logger.exception("primitive execution failed; holding this hand")
            return self.hold(current_q, f"原语执行失败：{type(exc).__name__}: {exc}")

    def _tick_active(
        self,
        current_q: np.ndarray,
        force: np.ndarray | None = None,
        currents: list[float] | None = None,
        *,
        feedback_id=None,
    ) -> np.ndarray | None:
        """Advance the active primitive. Returns joint targets (rad, ROS order) or ``None`` to hold."""
        if self._active is None:
            return None
        from hand_gesture_primitives.primitive_base import PrimitiveContext

        if not np.isfinite(current_q).all():
            self.status.feasible = False
            self.status.hold_reason = "invalid_feedback"
            return None
        if self.clock and not self.clock.running:
            return self._last_target.copy() if self._last_target is not None else current_q.copy()
        if self._active.done and self._last_target is not None:
            self._settle(current_q, feedback_id)
            return self._last_target.copy()
        elapsed = self.clock.advance(time.monotonic())
        sem = self._to_semantic(current_q)
        hw_now = self.converter.radians_to_hardware_uint8([float(v) for v in current_q])
        thr = float(getattr(self.contact_thresholds, "pressure_threshold", 20.0))
        contact = bool(force is not None and np.any(np.asarray(force, dtype=np.float64) > thr))
        ctx = PrimitiveContext(
            joint_currents=list(currents) if currents else [0.0] * 20,
            joint_currents_valid=bool(currents),
            hand_type=self.profile.hand_type,
            hand_side=self.side,
            hand_hw_angles=[float(v) for v in hw_now],
            hand_cmd_angles=self._last_cmd_hw,
            tactile_pressure=np.asarray(force, dtype=np.float64) if force is not None else None,
            tactile_mode="pressure" if force is not None else "none",
            contact_detected=contact,
            contact_thresholds=self.contact_thresholds,
            control_rate=self.control_rate,
            motion_speed=self.speed_scale,
            angle_unit="semantic_rad",
            radian_converter=self.converter,
        )
        result = self._active.compute(sem, elapsed, ctx)
        self.status.elapsed_s = elapsed
        self.status.feasible = bool(result.feasible)
        self.status.hold_reason = result.hold_reason
        self.status.motion_status = self._active.motion_status
        self.status.progress = float(result.progress)
        self._playback_status()
        if not result.feasible:
            return None
        target = self._from_result(result)
        self.adapter.validate_q(target)
        self._last_target = target
        self._last_cmd_hw = self.converter.radians_to_hardware_uint8([float(v) for v in target])
        # Hold-style primitives can expose trajectory completion through progress
        # without setting done. Do not infer completion from a stationary intermediate pose.
        self.clock.finished = self._active.trajectory_complete(result, elapsed)
        self._playback_status()
        if self.clock.finished:
            self._settle(current_q, feedback_id)
        return target

    def _settle(self, current_q, feedback_id):
        now = time.monotonic()
        if self._settle_since is None:
            self._settle_since = now
        if self.status.settle_status in {"settled", "settle_timeout"}:
            return
        close = (
            np.isfinite(current_q).all() and np.max(np.abs(current_q - self._last_target)) <= 0.03
        )
        if feedback_id is None or feedback_id != self._settle_feedback_id:
            self._settle_feedback_id = feedback_id
            self._settle_frames = self._settle_frames + 1 if close else 0
        self.status.settle_status = "settling"
        if self._settle_frames >= 3:
            self.status.settle_status, self.status.done = "settled", True
        elif now - self._settle_since >= 1.0:
            self.status.settle_status = "settle_timeout"
            self.status.hold_reason = "settle_timeout"
            self.status.done = False

    # ------------------------------------------------------------------ conversions
    def _to_semantic(self, q: np.ndarray) -> list[float]:
        return self.converter.controller_to_semantic_rad([float(v) for v in q])

    def _from_result(self, result) -> np.ndarray:
        space = result.space
        angles = list(result.target_angles)
        if space == "semantic_rad":
            q = self.converter.semantic_rad_to_controller(angles)
        elif space == "semantic":
            q = self.converter.semantic_to_radians(angles)
        elif space == "hardware":
            q = self.adapter.from_hardware(angles)
        else:
            raise ValueError(f"unknown PrimitiveResult.space {space!r}")
        return np.asarray(q, dtype=np.float64)[: len(self.joint_names)]

    def plan_static(self, name: str, current_q: np.ndarray, sample_period: float = 0.1):
        """Offline nominal trajectory (for UI preview). Uses hand_gesture_primitives.trajectory_planner."""
        from hand_gesture_primitives.trajectory_planner import plan_static_gesture

        return plan_static_gesture(
            hand_type=self.profile.hand_type,
            primitive=self.profile.resolve_primitive_name(name),
            current_semantic=self._to_semantic(current_q),
            semantic_to_positions=self.converter.semantic_rad_to_controller,
            sample_period=sample_period,
            radian_converter=self.converter,
        )
