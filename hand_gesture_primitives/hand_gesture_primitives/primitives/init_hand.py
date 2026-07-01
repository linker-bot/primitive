"""初始位手势原语 — 半张开自然姿态。"""

from typing import Optional

from ..gesture_engine import StaticPoseEngine, make_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_ref_gesture_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "init")
INIT_ANGLES_O20 = list(_ref_gesture_params.target_angles)
INIT_ANGLES_L25 = list(load_static_gesture_params("l25", "init").target_angles)
INIT_ANGLES = {"o20": INIT_ANGLES_O20, "l25": INIT_ANGLES_L25}


class InitHand(HandGesturePrimitive):
    """回到自然松弛初始位。"""

    TRANSITION_DURATION = _ref_gesture_params.duration

    def __init__(self) -> None:
        self._engine: Optional[StaticPoseEngine] = None
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "init"

    def on_enter(self, current_angles) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._hand_type = ""

    def _ensure_engine(self, ctx: PrimitiveContext) -> StaticPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            engine = make_static_engine(ctx.hand_type, "init")
            engine.reset(self._start_angles)
            self._engine = engine
            self._hand_type = ctx.hand_type
        return self._engine

    def compute(
        self, current_angles, elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        return self._move(self._ensure_engine(ctx).compute(elapsed))

    @property
    def done(self) -> bool:
        return False
