"""张开手势原语 — 五指完全伸直。"""

from typing import List, Optional

from ..gesture_engine import StaticPoseEngine, make_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult, ABD_NEUTRAL,
)

_ref_gesture_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "open")
OPEN_ANGLES_O20 = list(_ref_gesture_params.target_angles)

OPEN_ANGLES_L25 = list(load_static_gesture_params("l25", "open").target_angles)

OPEN_ANGLES = {"o20": OPEN_ANGLES_O20, "l25": OPEN_ANGLES_L25}


class OpenHand(HandGesturePrimitive):
    """五指完全张开 — StaticPoseEngine + gestures 配置。"""

    TRANSITION_DURATION = _ref_gesture_params.duration

    def __init__(self) -> None:
        self._engine: Optional[StaticPoseEngine] = None
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "open"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._hand_type = ""

    def _ensure_engine(self, ctx: PrimitiveContext) -> StaticPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            engine = make_static_engine(ctx.hand_type, "open")
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
