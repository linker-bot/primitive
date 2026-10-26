"""Table-safe L25 pre-grasp posture."""

from typing import List

from ..gesture_engine import activate_static_engine
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult


class GraspReady(HandGesturePrimitive):
    """Keep the three-finger aperture open while curling rear fingers."""

    def __init__(self) -> None:
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "grasp_ready"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""

    def _ensure_engine(self, ctx: PrimitiveContext):
        if self._engine is None or self._hand_type != ctx.hand_type:
            self._engine, self._result_space = activate_static_engine(
                ctx.hand_type,
                "grasp_ready",
                self._start_angles,
                radian_converter=ctx.radian_converter,
                start_unit=ctx.angle_unit,
            )
            self._hand_type = ctx.hand_type
        return self._engine

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        return self._move(self._ensure_engine(ctx).compute(elapsed), space=self._result_space)

    @property
    def done(self) -> bool:
        return False
