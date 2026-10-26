"""O6 握锤原语：初始化位姿后再闭合到握持位姿。

YAML angles/speeds 为 legacy255 量纲；有 Converter 时转到 semantic_rad 再限速运动。
"""

from typing import List, Optional

from ..angle_bridge import AngleUnits
from ..contact_config import scale_tick_count
from ..gesture_params import (
    TwoStageSpeedParams,
    load_two_stage_speed_params,
    two_stage_speed_params_as_semantic_rad,
)
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    parse_grasp_phase,
)


def _travel_duration(
    start: List[float],
    target: List[float],
    speeds: List[float],
) -> float:
    """返回所有受控关节到达目标所需的最长时间。"""
    durations = [
        abs(goal - origin) / speed
        for origin, goal, speed in zip(start, target, speeds)
        if speed > 0.0
    ]
    return max(durations, default=0.0)


def _speed_limited_pose(
    start: List[float],
    target: List[float],
    speeds: List[float],
    elapsed: float,
) -> List[float]:
    """按各关节最大位置变化率生成同步开始、独立到达的目标。"""
    t = max(0.0, float(elapsed))
    result = list(start)
    for index, (origin, goal, speed) in enumerate(zip(start, target, speeds)):
        delta = goal - origin
        if speed <= 0.0 or delta == 0.0:
            result[index] = goal if delta == 0.0 else origin
            continue
        step = min(abs(delta), speed * t)
        result[index] = origin + step if delta > 0.0 else origin - step
    return result


def _params_in_exec_space(ctx: PrimitiveContext) -> tuple[TwoStageSpeedParams, str]:
    """加载参数并按执行核量纲换算角度/速度/到位容差。"""
    params = load_two_stage_speed_params(ctx.hand_type, "hammer_grip")
    u = AngleUnits(ctx)
    if ctx.radian_converter is not None:
        params = two_stage_speed_params_as_semantic_rad(params, ctx.radian_converter)
    return params, u.space


class HammerGrip(HandGesturePrimitive):
    """O6 两阶段握锤动作。

    ``prep`` 以配置速度运动到初始化位姿并保持；``close`` 从配置的
    初始化位姿运动到握锤目标位姿；``full`` 不暂停，依次完成两个阶段。

    ``close`` 可直接下发：若手尚未到初始化位姿，先持续下发该位姿；
    连续确认到位后才按第二阶段速度闭合。等待超时则保持初始化目标，
    不盲目闭合。
    """

    def __init__(self, phase: str = "full") -> None:
        self._phase = parse_grasp_phase(phase)
        self._grasp_state = "approaching"
        self._ready_count = 0
        self._close_started_at: Optional[float] = None
        self._close_start_angles: List[float] = []
        self._prep_timed_out = False
        self._result_space = "semantic_rad"

    @property
    def name(self) -> str:
        return "hammer_grip"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def grasp_state(self) -> str:
        return self._grasp_state

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._ready_count = 0
        self._close_started_at = None
        self._close_start_angles = []
        self._prep_timed_out = False
        self._result_space = "semantic_rad"
        self._grasp_state = "waiting_prep" if self._phase == "close" else "approaching"

    def _params(self, ctx: PrimitiveContext) -> TwoStageSpeedParams:
        if ctx.hand_type.lower() != "o6":
            raise ValueError("hammer_grip 仅支持 O6")
        params, space = _params_in_exec_space(ctx)
        self._result_space = space
        return params

    @staticmethod
    def _ready_error(
        current_angles: List[float],
        params: TwoStageSpeedParams,
    ) -> float:
        """返回 O6 受控关节相对初始化位姿的最大位置误差。"""
        errors = [
            abs(current - target)
            for current, target, speed in zip(
                current_angles,
                params.initial_angles,
                params.initial_speeds,
            )
            if speed > 0.0
        ]
        return max(errors, default=0.0)

    def _compute_close(
        self,
        current_angles: List[float],
        elapsed: float,
        params: TwoStageSpeedParams,
        control_rate: float = 10.0,
    ) -> PrimitiveResult:
        """先确保 prep 到位，再从实际位置开始第二阶段闭合。"""
        space = self._result_space
        if self._prep_timed_out:
            self._grasp_state = "prep_timeout"
            return self._move(list(params.initial_angles), space=space, trajectory_finished=False)

        if self._close_started_at is None:
            ready_error = self._ready_error(current_angles, params)
            if ready_error <= params.ready_tolerance:
                self._ready_count += 1
            else:
                self._ready_count = 0

            if self._ready_count >= scale_tick_count(params.ready_frames, control_rate):
                self._close_started_at = max(0.0, float(elapsed))
                self._close_start_angles = list(current_angles)
                self._grasp_state = "closing"
                return self._move(list(self._close_start_angles), space=space, trajectory_finished=False)

            if elapsed >= params.ready_timeout:
                self._prep_timed_out = True
                self._grasp_state = "prep_timeout"
            else:
                self._grasp_state = "waiting_prep"
            return self._move(list(params.initial_angles), space=space, trajectory_finished=False)

        close_elapsed = max(0.0, elapsed - self._close_started_at)
        duration = _travel_duration(self._close_start_angles, params.target_angles, params.speeds)
        output = _speed_limited_pose(
            self._close_start_angles,
            params.target_angles,
            params.speeds,
            close_elapsed,
        )
        self._grasp_state = "holding" if close_elapsed >= duration else "closing"
        return self._move(output, space=space, trajectory_finished=close_elapsed >= duration)

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        try:
            params = self._params(ctx)
        except ValueError as error:
            return self._hold(str(error))

        space = self._result_space
        if self._phase == "prep":
            duration = _travel_duration(
                self._start_angles,
                params.initial_angles,
                params.initial_speeds,
            )
            output = _speed_limited_pose(
                self._start_angles,
                params.initial_angles,
                params.initial_speeds,
                elapsed,
            )
            self._grasp_state = "ready" if elapsed >= duration else "approaching"
            return self._move(output, space=space, trajectory_finished=elapsed >= duration)

        if self._phase == "close":
            return self._compute_close(current_angles, elapsed, params, ctx.control_rate)

        prep_duration = _travel_duration(
            self._start_angles, params.initial_angles, params.initial_speeds
        )
        if elapsed < prep_duration:
            self._grasp_state = "approaching"
            return self._move(
                _speed_limited_pose(
                    self._start_angles,
                    params.initial_angles,
                    params.initial_speeds,
                    elapsed,
                ),
                space=space, trajectory_finished=False,
            )

        close_elapsed = elapsed - prep_duration
        close_duration = _travel_duration(
            params.initial_angles, params.target_angles, params.speeds
        )
        self._grasp_state = "holding" if close_elapsed >= close_duration else "closing"
        return self._move(
            _speed_limited_pose(
                params.initial_angles,
                params.target_angles,
                params.speeds,
                close_elapsed,
            ),
            space=space, trajectory_finished=close_elapsed >= close_duration,
        )

    @property
    def done(self) -> bool:
        return False
