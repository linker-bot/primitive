"""张开手势原语 — 从当前位置相对张开。"""

from typing import List

from ..angle_bridge import (
    result_space_for_converter,
    start_angles_for_rad_engine,
)
from ..gesture_engine import StaticPoseEngine
from ..gesture_params import (
    CANONICAL_SEMANTIC_HAND,
    load_static_gesture_params,
    static_params_as_semantic_rad,
)
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_ref_open_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "open")
OPEN_ANGLES = list(_ref_open_params.target_angles)


class Release(HandGesturePrimitive):
    """从当前位置相对张开。degree=1.0 完全张开，0.5 张开剩余距离 50%。"""

    TRANSITION_DURATION = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "release").duration

    def __init__(self, degree: str = "1.0"):
        self._degree = max(0.0, min(1.0, float(degree)))
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "release"

    def _ensure_engine(self, ctx: PrimitiveContext) -> StaticPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            params = load_static_gesture_params(ctx.hand_type, "release")
            converter = ctx.radian_converter
            start = start_angles_for_rad_engine(
                self._start_angles,
                radian_converter=converter,
                start_unit=ctx.angle_unit,
            )
            if converter is not None:
                params = static_params_as_semantic_rad(params, converter)
            open_target = list(params.target_angles)
            release_target = [s + (o - s) * self._degree for s, o in zip(start, open_target)]
            engine = StaticPoseEngine(params)
            engine.reset(start, release_target)
            self._engine = engine
            self._result_space = result_space_for_converter(converter)
            self._hand_type = ctx.hand_type
        return self._engine

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        return self._move(self._ensure_engine(ctx).compute(elapsed), space=self._result_space)

    @property
    def done(self) -> bool:
        return False
