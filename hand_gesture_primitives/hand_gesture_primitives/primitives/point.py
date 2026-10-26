"""指向手势原语 — 食指伸出，其余握拢。

O6：config 静态 MCP 姿态 + StaticPoseEngine。
O20/L25：gestures 默认 / YAML + StaticPoseEngine（行为与历史 POINT_ANGLES 一致）。
"""

from typing import List

from ..gesture_engine import activate_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_ref_gesture_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "point")
POINT_ANGLES = list(_ref_gesture_params.target_angles)
TRANSITION_DURATION = _ref_gesture_params.duration


class Point(HandGesturePrimitive):
    """食指伸出指向 — StaticPoseEngine + gestures 配置。"""

    def __init__(self) -> None:
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "point"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._result_space = "semantic_rad"
        self._hand_type = ""

    def _ensure_engine(self, ctx: PrimitiveContext):
        if self._engine is None or self._hand_type != ctx.hand_type:
            self._engine, self._result_space = activate_static_engine(
                ctx.hand_type,
                "point",
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
