"""Side-effect-free nominal trajectory planning for static hand gestures."""

from __future__ import annotations

import math
from typing import Callable, List, Sequence

from .gesture_engine import make_static_engine
from .gesture_params import StaticGestureParams, load_static_gesture_params
from .keyframe_runner import KeyframeRunner, MotionProgram
from .semantic_30 import as_semantic_30


def parse_pose_plan_ref(raw: str) -> str | None:
    """从 PlanGesture.primitive 解析自定义姿态名。

    支持 ``pose name`` / ``pose:name`` / ``pose/name`` / ``named_pose:name``。
    """
    s = (raw or "").strip().lower()
    for prefix in ("pose ", "pose:", "pose/", "named_pose:"):
        if s.startswith(prefix):
            name = s[len(prefix) :].strip()
            return name or None
    return None


def static_gesture_duration(params: StaticGestureParams) -> float:
    if params.motion_type == "phased_pose":
        return params.phase1 + params.phase2 + max(params.phase3, 0.0) + params.phase4
    if params.motion_type == "relax_grip":
        return params.duration + params.hold_duration + params.grip_duration
    return params.duration


def plan_static_gesture(
    *,
    hand_type: str,
    primitive: str,
    current_semantic: Sequence[float],
    semantic_to_positions: Callable[[Sequence[float]], Sequence[float]],
    sample_period: float = 0.04,
    radian_converter=None,
) -> dict[str, object]:
    """Run the same static gesture engine as execution without publishing commands.

    ``current_semantic`` 为 legacy255 S30（或已是 semantic_rad）。若传入
    ``radian_converter``，引擎在 semantic_rad 内插值，再经
    ``semantic_to_positions`` 输出 controller 弧度。
    """

    period = float(sample_period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be a positive finite number")
    # Keep previews bounded while retaining enough samples for smooth playback.
    period = max(0.01, min(period, 0.25))
    params = load_static_gesture_params(hand_type, primitive)
    engine = make_static_engine(hand_type, primitive, radian_converter=radian_converter)
    start = as_semantic_30(current_semantic)
    to_positions = semantic_to_positions
    if radian_converter is not None:
        start = radian_converter.legacy255_to_semantic_rad(start)
        to_positions = radian_converter.semantic_rad_to_controller
    engine.reset(start)
    duration = round(max(0.0, static_gesture_duration(params)), 9)
    point_count = max(2, int(math.ceil(duration / period)) + 1)
    times = [round(min(index * period, duration), 9) for index in range(point_count)]
    times[-1] = duration
    positions: List[List[float]] = []
    for elapsed in times:
        semantic = engine.compute(elapsed)
        positions.append([float(value) for value in to_positions(semantic)])
    return {
        "primitive": primitive,
        "duration_s": duration,
        "time_s": times,
        "positions": positions,
        "nominal_only": True,
    }


def plan_motion_program(
    *,
    program: MotionProgram,
    start_q: Sequence[float],
    to_positions: Callable[[Sequence[float]], Sequence[float]],
    sample_period: float = 0.04,
) -> dict[str, object]:
    """按 MotionProgram 多段 AUTO 模式采样轨迹。

    返回格式与 ``plan_static_gesture`` 一致，增加 ``segments`` 字段标记段边界。
    """
    period = float(sample_period)
    if not math.isfinite(period) or period <= 0.0:
        raise ValueError("sample_period must be a positive finite number")
    period = max(0.01, min(period, 0.25))

    if not program.segments:
        raise ValueError("MotionProgram.segments 为空")

    total_duration = round(sum(s.duration for s in program.segments), 9)
    if total_duration <= 0.0:
        raise ValueError("MotionProgram total duration <= 0")

    runner = KeyframeRunner()
    runner.reset(program, list(start_q))

    point_count = max(2, int(math.ceil(total_duration / period)) + 1)
    times: List[float] = [round(min(i * period, total_duration), 9) for i in range(point_count)]
    times[-1] = total_duration

    positions: List[List[float]] = []
    segments: List[dict] = []
    current_seg_name = ""

    for elapsed in times:
        q = runner.compute(elapsed)
        positions.append([float(v) for v in to_positions(q)])
        seg_name = runner.keyframe
        if seg_name != current_seg_name:
            segments.append({"name": seg_name, "start_index": len(positions) - 1})
            current_seg_name = seg_name

    return {
        "primitive": f"named_pose:{program.ref}",
        "duration_s": total_duration,
        "time_s": times,
        "positions": positions,
        "segments": segments,
        "nominal_only": True,
    }
