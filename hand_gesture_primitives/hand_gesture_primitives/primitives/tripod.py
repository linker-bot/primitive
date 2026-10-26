"""三指捏取原语 — 拇指与食指+中指对捏，形成三角支撑。

所有手型支持 ``prep`` / ``close`` / ``auto``（``full`` 为 auto 别名）：
  auto  = prep → 停顿 → 自动 close（**默认**，裸发 ``tripod`` 同效）
  prep  = 侧摆/旋转就位，三指屈曲保持 init 预成型；到位后等待 close
  close = 仅闭合拇/食/中屈曲通道
"""

from typing import List, Optional, Tuple

from ..angle_bridge import AngleUnits
from ..gesture_engine import StaticPoseEngine, activate_static_engine
from ..gesture_params import (
    CANONICAL_SEMANTIC_HAND,
    StaticGestureParams,
    load_static_gesture_params,
    load_tripod_stage_durations,
    static_params_as_semantic_rad,
)
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
    parse_tripod_phase,
    present_joints,
)
from ..semantic_30 import SEMANTIC_30_IDX

_ref_gesture_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "tripod")
TRIPOD_ANGLES = list(_ref_gesture_params.target_angles)
TRANSITION_DURATION = _ref_gesture_params.duration

# L6 init 四指自然弯曲位置，prep 阶段复用（S30 槽位；运行时 lazy 加载）
_L6_INDEX = SEMANTIC_30_IDX["index_mcp_flex"]
_L6_MIDDLE = SEMANTIC_30_IDX["middle_mcp_flex"]
_L6_RING = SEMANTIC_30_IDX["ring_mcp_flex"]
_L6_PINKY = SEMANTIC_30_IDX["pinky_mcp_flex"]
_INIT_PARAMS: Optional[StaticGestureParams] = None


def _l6_init_params() -> StaticGestureParams:
    global _INIT_PARAMS
    if _INIT_PARAMS is None:
        _INIT_PARAMS = load_static_gesture_params("l6", "init")
    return _INIT_PARAMS


def _l6_init_fingers() -> tuple[float, float, float, float]:
    params = _l6_init_params()
    return (
        params.target_angles[_L6_INDEX],
        params.target_angles[_L6_MIDDLE],
        params.target_angles[_L6_RING],
        params.target_angles[_L6_PINKY],
    )


_L6_THUMB_FLEX_JOINT = SEMANTIC_30_IDX["thumb_mcp_flex"]
_L6_THUMB_YAW_JOINT = SEMANTIC_30_IDX["thumb_mcp_abd"]
_L6_FINGER_JOINTS = (_L6_INDEX, _L6_MIDDLE, _L6_RING, _L6_PINKY)
_L6_THUMB_FULLY_OPEN = 0.0
_L6_CLOSE_JOINTS = (_L6_THUMB_FLEX_JOINT, _L6_INDEX, _L6_MIDDLE)

# S30：拇/食/中 MCP 屈曲 + PIP + DIP（O30 独立 root2 须进 close 集）
_TRIPOD_CLOSE_SEMANTIC = (
    SEMANTIC_30_IDX["thumb_mcp_flex"],
    SEMANTIC_30_IDX["index_mcp_flex"],
    SEMANTIC_30_IDX["middle_mcp_flex"],
    SEMANTIC_30_IDX["thumb_dip"],
    SEMANTIC_30_IDX["index_pip"],
    SEMANTIC_30_IDX["index_dip"],
    SEMANTIC_30_IDX["middle_pip"],
    SEMANTIC_30_IDX["middle_dip"],
)


class Tripod(HandGesturePrimitive):
    """拇指与食指+中指三指对捏。

    所有手型：
      auto  = prep → 停顿 → 自动 close（**默认**；``full`` 同义）
      prep  = 非屈曲关节到 tripod 目标，三指屈曲到 init 预成型（等 close）
      close = 只闭合三指屈曲通道

    L6 ``auto``：两阶段（拇指先完全张开）。
    其它手型 ``auto``：默认一次到位引擎；``tripod_by_force`` 拆成 prep→close。
    """

    def __init__(self, phase: str = "auto", *, progress: float | None = None) -> None:
        self._phase = parse_tripod_phase(phase)
        self._progress_override: float | None = (
            None if progress is None else max(0.0, min(1.0, float(progress)))
        )
        self._engine: Optional[StaticPoseEngine] = None
        self._result_space = "semantic_rad"
        self._hand_type = ""
        self._l6_params: Optional[StaticGestureParams] = None
        self._grasp_state = "approaching"

    def supports_progress(self) -> bool:
        return self._phase in ("prep", "close")

    @property
    def progress_driven(self) -> bool:
        return self._progress_override is not None

    def set_progress(self, p: float) -> None:
        """段内 progress 热更新（prep/close；auto 忽略）。"""
        import math

        if self._phase not in ("prep", "close"):
            return
        value = float(p)
        if not math.isfinite(value):
            return
        self._progress_override = max(0.0, min(1.0, value))

    def _phase_t(self, elapsed: float, duration: float) -> float:
        from ..keyframe_runner import resolve_t

        return resolve_t(elapsed, duration, self._progress_override)

    @property
    def name(self) -> str:
        return "tripod"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def grasp_state(self) -> str:
        return self._grasp_state

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""
        self._l6_params = None
        self._grasp_state = "closing" if self._phase == "close" else "approaching"

    def _ensure_engine(self, ctx: PrimitiveContext) -> StaticPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            self._engine, self._result_space = activate_static_engine(
                ctx.hand_type,
                "tripod",
                self._start_angles,
                radian_converter=ctx.radian_converter,
                start_unit=ctx.angle_unit,
            )
            self._hand_type = ctx.hand_type
        return self._engine

    def _ensure_l6_params(self) -> StaticGestureParams:
        if self._l6_params is None:
            params = load_static_gesture_params("l6", self.name)
            if params.motion_type != "tripod_two_stage":
                raise ValueError(
                    f"L6 {self.name}.motion.type 必须为 tripod_two_stage，"
                    f"收到: {params.motion_type!r}"
                )
            self._l6_params = params
        return self._l6_params

    def _uses_two_stage_full(self, ctx: PrimitiveContext) -> bool:
        """``auto`` 是否拆成 prep→close。子类（tripod_by_force）可打开。"""
        del ctx
        return False

    def _target_for(self, joints: tuple, target: List[float]) -> List[float]:
        """仅把指定关节设为 tripod 目标，其余保持原语进入时姿态。"""
        out = list(self._start_angles)
        for index in joints:
            if index < len(out) and index < len(target):
                out[index] = target[index]
        return out

    def _close_semantic(self) -> Tuple[int, ...]:
        return tuple(present_joints(*_TRIPOD_CLOSE_SEMANTIC))

    def _generic_targets(
        self, ctx: PrimitiveContext
    ) -> Tuple[List[float], List[float], List[float]]:
        """非 L6：prep=瞄准+init 屈曲；close=三指收到 tripod 目标。"""
        u = AngleUnits(ctx)
        tripod = load_static_gesture_params(ctx.hand_type, "tripod")
        init_p = load_static_gesture_params(ctx.hand_type, "init")
        if ctx.radian_converter is not None:
            tripod = static_params_as_semantic_rad(tripod, ctx.radian_converter)
            init_p = static_params_as_semantic_rad(init_p, ctx.radian_converter)
        target = list(tripod.target_angles)
        init = list(init_p.target_angles)
        self._result_space = u.space
        close_set = set(self._close_semantic())
        prep_target = list(self._start_angles)
        for index, value in enumerate(target):
            if index >= len(prep_target):
                break
            if index in close_set:
                if index < len(init):
                    prep_target[index] = init[index]
            else:
                prep_target[index] = value
        close_from_prep = list(prep_target)
        for index in close_set:
            if index < len(close_from_prep) and index < len(target):
                close_from_prep[index] = target[index]
        close_direct = self._target_for(tuple(close_set), target)
        return prep_target, close_from_prep, close_direct

    def _compute_generic_two_stage(self, elapsed: float, ctx: PrimitiveContext) -> PrimitiveResult:
        prep_target, close_from_prep, close_direct = self._generic_targets(ctx)
        space = self._result_space
        prep_duration, close_duration, pause_duration = load_tripod_stage_durations(
            ctx.hand_type, "tripod"
        )
        prep_duration = max(0.0, prep_duration)
        close_duration = max(0.0, close_duration)
        pause_duration = max(0.0, pause_duration)

        if self._phase == "prep":
            progress = self._phase_t(elapsed, prep_duration)
            self._grasp_state = "ready" if progress >= 1.0 else "approaching"
            return self._move(lerp_angles(self._start_angles, prep_target, progress), space=space,
                              trajectory_finished=self._phase == "prep" and progress >= 1.0)

        if self._phase == "close":
            progress = self._phase_t(elapsed, close_duration)
            self._grasp_state = "holding" if progress >= 1.0 else "closing"
            return self._move(lerp_angles(self._start_angles, close_direct, progress), space=space,
                              trajectory_finished=progress >= 1.0)

        if elapsed < prep_duration:
            self._grasp_state = "approaching"
            progress = 1.0 if prep_duration == 0.0 else elapsed / prep_duration
            return self._move(lerp_angles(self._start_angles, prep_target, progress), space=space,
                              trajectory_finished=self._phase == "prep" and progress >= 1.0)

        close_start = prep_duration + pause_duration
        if elapsed < close_start:
            self._grasp_state = "ready"
            return self._move(prep_target, space=space, trajectory_finished=False)

        close_elapsed = elapsed - close_start
        progress = 1.0 if close_duration == 0.0 else close_elapsed / close_duration
        self._grasp_state = "holding" if progress >= 1.0 else "closing"
        return self._move(lerp_angles(prep_target, close_from_prep, progress), space=space,
                          trajectory_finished=progress >= 1.0)

    def _compute_l6(self, elapsed: float, ctx: PrimitiveContext) -> PrimitiveResult:
        u = AngleUnits(ctx)
        params = self._ensure_l6_params()
        if ctx.radian_converter is not None:
            params = static_params_as_semantic_rad(params, ctx.radian_converter)
        init_fingers = u.pose(
            [
                0.0,
                *_l6_init_fingers(),
            ]
            + [0.0] * 15
        )
        thumb_open = u.value(_L6_THUMB_FLEX_JOINT, _L6_THUMB_FULLY_OPEN)
        self._result_space = u.space
        space = u.space

        prep_target = self._target_for((_L6_THUMB_YAW_JOINT,), params.target_angles)
        if _L6_THUMB_FLEX_JOINT < len(prep_target):
            # L6 语义：legacy0=伸直（硬件侧常映射为 255）；此处已是执行核量纲
            prep_target[_L6_THUMB_FLEX_JOINT] = thumb_open
        for idx in _L6_FINGER_JOINTS:
            finger_i = _L6_FINGER_JOINTS.index(idx)
            if idx < len(prep_target) and finger_i < 4:
                prep_target[idx] = init_fingers[idx]
        close_target = list(prep_target)
        for index in _L6_CLOSE_JOINTS:
            if index < len(close_target) and index < len(params.target_angles):
                close_target[index] = params.target_angles[index]

        phase1_duration = max(0.0, params.phase1)
        phase2_duration = max(0.0, params.phase2)

        if self._phase == "prep":
            progress = self._phase_t(elapsed, phase1_duration)
            self._grasp_state = "ready" if progress >= 1.0 else "approaching"
            return self._move(lerp_angles(self._start_angles, prep_target, progress), space=space,
                              trajectory_finished=self._phase == "prep" and progress >= 1.0)

        if self._phase == "close":
            # close 可以单独下发：拇指侧摆及非参与关节保持收到指令时的位置。
            direct_close_target = self._target_for(_L6_CLOSE_JOINTS, params.target_angles)
            progress = self._phase_t(elapsed, phase2_duration)
            self._grasp_state = "holding" if progress >= 1.0 else "closing"
            return self._move(
                lerp_angles(self._start_angles, direct_close_target, progress),
                space=space, trajectory_finished=progress >= 1.0,
            )

        if elapsed < phase1_duration:
            self._grasp_state = "approaching"
            progress = 1.0 if phase1_duration == 0.0 else elapsed / phase1_duration
            return self._move(lerp_angles(self._start_angles, prep_target, progress), space=space,
                              trajectory_finished=self._phase == "prep" and progress >= 1.0)

        close_start = phase1_duration + max(0.0, params.pause_duration)
        if elapsed < close_start:
            self._grasp_state = "ready"
            return self._move(prep_target, space=space, trajectory_finished=False)

        close_elapsed = elapsed - close_start
        progress = 1.0 if phase2_duration == 0.0 else close_elapsed / phase2_duration
        self._grasp_state = "holding" if progress >= 1.0 else "closing"
        return self._move(lerp_angles(prep_target, close_target, progress), space=space,
                          trajectory_finished=progress >= 1.0)

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        if ctx.hand_type.lower() == "l6":
            return self._compute_l6(elapsed, ctx)
        if self._phase == "auto" and not self._uses_two_stage_full(ctx):
            return self._move(self._ensure_engine(ctx).compute(elapsed), space=self._result_space)
        return self._compute_generic_two_stage(elapsed, ctx)

    @property
    def done(self) -> bool:
        return False
