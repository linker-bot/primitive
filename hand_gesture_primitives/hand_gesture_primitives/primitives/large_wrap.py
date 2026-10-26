"""O6/L6 五指大包络原语。

prep 拇指完全张开（legacy 屈曲 255）、侧摆 0，四指使用 init 姿态；close 四指先闭合，
拇指延迟 thumb_delay 秒后开始闭合。auto 按配置串联 prep、等待和 close。

YAML 姿态为 legacy255；有 Converter 时经 AngleUnits 转到 semantic_rad 再插值。
"""

from typing import List, Optional

from ..angle_bridge import AngleUnits
from ..gesture_params import (
    LargeWrapParams,
    large_wrap_params_as_semantic_rad,
    load_large_wrap_params,
)
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
    parse_large_wrap_phase,
)
from ..semantic_30 import SEMANTIC_30_IDX

# S30 屈曲槽：close 交错只动这些通道（姿态向量已是 30 维）
_THUMB = SEMANTIC_30_IDX["thumb_mcp_flex"]
_FINGERS = (
    SEMANTIC_30_IDX["index_mcp_flex"],
    SEMANTIC_30_IDX["middle_mcp_flex"],
    SEMANTIC_30_IDX["ring_mcp_flex"],
    SEMANTIC_30_IDX["pinky_mcp_flex"],
)


class LargeWrap(HandGesturePrimitive):
    """O6/L6 两阶段五指包络，支持 prep / close / auto。"""

    def __init__(self, phase: str = "auto", *, progress: float | None = None) -> None:
        self._phase = parse_large_wrap_phase(phase)
        self._progress_override: float | None = (
            None if progress is None else max(0.0, min(1.0, float(progress)))
        )
        self._params: Optional[LargeWrapParams] = None
        self._params_hand_type = ""
        self._result_space = "semantic_rad"
        self._grasp_state = "approaching"

    def supports_progress(self) -> bool:
        return self._phase in ("prep", "close")

    @property
    def progress_driven(self) -> bool:
        return self._progress_override is not None

    def set_progress(self, p: float) -> None:
        import math

        if self._phase not in ("prep", "close"):
            return
        value = float(p)
        if not math.isfinite(value):
            return
        self._progress_override = max(0.0, min(1.0, value))

    @property
    def name(self) -> str:
        return "large_wrap"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def grasp_state(self) -> str:
        return self._grasp_state

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._params = None
        self._params_hand_type = ""
        self._result_space = "semantic_rad"
        self._grasp_state = "closing" if self._phase == "close" else "approaching"

    def _ensure_params(self, ctx: PrimitiveContext) -> LargeWrapParams:
        hand_type = ctx.hand_type.lower()
        if self._params is None or self._params_hand_type != hand_type:
            params = load_large_wrap_params(hand_type)
            u = AngleUnits(ctx)
            self._result_space = u.space
            if ctx.radian_converter is not None:
                params = large_wrap_params_as_semantic_rad(params, ctx.radian_converter)
            self._params = params
            self._params_hand_type = hand_type
        return self._params

    def _direct_close_target(self, params: LargeWrapParams) -> List[float]:
        """close 单独执行时只移动五个屈曲关节，拇指侧摆保持当前值。"""
        target = list(self._start_angles)
        for index in (_THUMB,) + _FINGERS:
            if index < len(target) and index < len(params.close_angles):
                target[index] = params.close_angles[index]
        return target

    def _progress(self, elapsed: float, duration: float) -> float:
        from ..keyframe_runner import resolve_t

        return resolve_t(elapsed, duration, self._progress_override)

    def _lerp_staggered_close(
        self,
        start: List[float],
        target: List[float],
        elapsed: float,
        params: LargeWrapParams,
    ) -> tuple[List[float], bool]:
        """四指按 elapsed/duration 线性插值，拇指延迟后以同样速度闭合，不追赶。"""
        result = list(start)
        finger_progress = self._progress(elapsed, params.close_duration)
        thumb_p = self._progress(max(0.0, elapsed - params.thumb_delay), params.close_duration)
        for i in _FINGERS:
            if i < len(result) and i < len(target):
                result[i] = start[i] + (target[i] - start[i]) * finger_progress
        if _THUMB < len(result) and _THUMB < len(target):
            result[_THUMB] = start[_THUMB] + (target[_THUMB] - start[_THUMB]) * thumb_p
        return result, finger_progress >= 1.0 - 1e-12 and thumb_p >= 1.0 - 1e-12

    def _compute_six_dof(
        self,
        elapsed: float,
        ctx: PrimitiveContext,
    ) -> PrimitiveResult:
        params = self._ensure_params(ctx)
        space = self._result_space

        if self._phase == "prep":
            progress = self._progress(elapsed, params.prep_duration)
            self._grasp_state = "ready" if progress >= 1.0 else "approaching"
            return self._move(
                lerp_angles(self._start_angles, params.prep_angles, progress),
                space=space, trajectory_finished=progress >= 1.0,
            )

        if self._phase == "close":
            output, finished = self._lerp_staggered_close(
                self._start_angles, self._direct_close_target(params), elapsed, params,
            )
            self._grasp_state = "holding" if finished else "closing"
            return self._move(output, space=space, trajectory_finished=finished)

        if elapsed < params.prep_duration:
            self._grasp_state = "approaching"
            return self._move(
                lerp_angles(self._start_angles, params.prep_angles,
                            self._progress(elapsed, params.prep_duration)),
                space=space, trajectory_finished=False,
            )

        close_start = params.prep_duration + params.pause_duration
        if elapsed < close_start:
            self._grasp_state = "ready"
            return self._move(list(params.prep_angles), space=space, trajectory_finished=False)

        output, finished = self._lerp_staggered_close(
            list(params.prep_angles), params.close_angles, elapsed - close_start, params,
        )
        self._grasp_state = "holding" if finished else "closing"
        return self._move(output, space=space, trajectory_finished=finished)

    def compute(
        self,
        current_angles: List[float],
        elapsed: float,
        ctx: PrimitiveContext,
    ) -> PrimitiveResult:
        hand_type = ctx.hand_type.lower()
        if hand_type not in ("o6", "l6"):
            return self._hold(f"large_wrap 仅支持 O6/L6，当前手型为 {ctx.hand_type}")
        return self._compute_six_dof(elapsed, ctx)

    @property
    def done(self) -> bool:
        return False
