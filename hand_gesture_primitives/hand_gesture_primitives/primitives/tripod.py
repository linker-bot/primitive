"""三指捏取原语 — 拇指与食指+中指对捏，形成三角支撑。

O6：config 静态 MCP 姿态 + StaticPoseEngine。
O20/L25：gestures 默认 / YAML + StaticPoseEngine。
"""

from typing import List, Optional

from ..gesture_engine import StaticPoseEngine, make_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_ref_gesture_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "tripod")
TRIPOD_ANGLES = list(_ref_gesture_params.target_angles)
TRANSITION_DURATION = _ref_gesture_params.duration


class Tripod(HandGesturePrimitive):
    """拇指与食指+中指三指对捏 — StaticPoseEngine + gestures 配置。"""

    def __init__(self) -> None:
        self._engine: Optional[StaticPoseEngine] = None
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "tripod"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._hand_type = ""

    def _ensure_engine(self, ctx: PrimitiveContext) -> StaticPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            engine = make_static_engine(ctx.hand_type, "tripod")
            engine.reset(self._start_angles)
            self._engine = engine
            self._hand_type = ctx.hand_type
        return self._engine

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        return self._move(self._ensure_engine(ctx).compute(elapsed))

    @property
    def done(self) -> bool:
        return False
