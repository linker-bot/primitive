"""Shared finite-trajectory playback used by the workbench and the ROS executor.

The clock is integrated, so changing speed never rescales elapsed history. Safety
and feedback deadlines continue to use wall time, independently of this clock.
"""

import math
from functools import lru_cache

FINITE_PRIMITIVES = frozenset(
    {
        "init",
        "open",
        "release",
        "relax_grip",
        "fist",
        "pinch",
        "point",
        "ok_sign",
        "v_sign",
        "grasp_ready",
        "tripod",
        "large_wrap",
        "hammer_grip",
        "index_pinch",
        "middle_pinch",
        "parallel_extension",
        "index_flick",
        "index_middle_flick",
    }
)


@lru_cache(maxsize=32)
def _profile(model):
    from .hand_profile import load_hand_profile

    return load_hand_profile(model)


def hold_reason(model, name):
    profile = _profile(model)
    key = "named_pose" if name.startswith("named_pose:") else name
    if not profile.is_primitive_supported(key):
        return "当前手型不支持此原语"
    if key != "named_pose" and key not in FINITE_PRIMITIVES:
        return "此原语依赖感知或持续力控，请使用一键执行"
    return ""


def trajectory_duration(model, name, *, entry=None, start=None, converter=None):
    if entry is not None:
        return sum(frame.duration for frame in entry.effective_keyframes())
    if name.startswith("named_pose:"):
        return None
    if hold_reason(model, name):
        return None
    from .gesture_params import load_static_gesture_params
    from .trajectory_planner import static_gesture_duration

    if name == "large_wrap":
        from .gesture_params import load_large_wrap_params

        p = load_large_wrap_params(model)
        return p.prep_duration + p.pause_duration + p.close_duration + p.thumb_delay
    if name == "hammer_grip":
        from .gesture_params import (
            load_two_stage_speed_params,
            two_stage_speed_params_as_semantic_rad,
        )
        from .primitives.hammer_grip import _travel_duration

        if start is None:
            return None
        p = load_two_stage_speed_params(model, name)
        if converter is not None:
            p = two_stage_speed_params_as_semantic_rad(p, converter)
        return _travel_duration(start, p.initial_angles, p.initial_speeds) + _travel_duration(
            p.initial_angles, p.target_angles, p.speeds
        )
    if name in {"index_flick", "index_middle_flick"}:
        from .primitives import PRIMITIVE_REGISTRY

        return PRIMITIVE_REGISTRY[name].END_DURATION
    p = load_static_gesture_params(model, name)
    if name == "tripod" and model == "l6":
        return p.phase1 + p.pause_duration + p.phase2
    return static_gesture_duration(p)


def validate_speed(speed):
    if isinstance(speed, bool) or not math.isfinite(speed) or not 0.25 <= speed <= 3.0:
        raise ValueError("primitive_speed must be finite and within 0.25..3.0")
    return float(speed)


class PlaybackClock:
    def __init__(self, now, speed=1.0, duration=None, held=False):
        self.last = now
        self.speed = validate_speed(speed)
        self.duration = duration
        self.elapsed = 0.0
        self.held = held
        self.running = True
        self.finished = False

    def advance(self, now):
        if self.running:
            self.elapsed += max(0.0, now - self.last) * self.speed
        self.last = now
        return self.elapsed

    def set_running(self, running, now):
        # Input edges take effect at the last computed target, never synthesize
        # extra movement on release or catch up time spent paused.
        self.last = now
        self.running = running

    def set_speed(self, speed, now):
        speed = validate_speed(speed)
        self.advance(now)
        self.speed = speed

    @property
    def complete(self):
        return self.finished

    def snapshot(self):
        return {
            "playback_state": "completed"
            if self.complete
            else "running"
            if self.running
            else "paused",
            "overall_progress": (
                min(1.0, self.elapsed / self.duration) if self.duration else -1.0
            ),
        }
