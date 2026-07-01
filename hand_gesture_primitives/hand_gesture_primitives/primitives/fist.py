"""握拳手势原语 — 分阶段避免拇指与食指碰撞。"""

from typing import List, Optional

from ..gesture_engine import PhasedPoseEngine, make_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_ref_gesture_params = load_static_gesture_params(CANONICAL_SEMANTIC_HAND, "fist")
FIST_ANGLES = list(_ref_gesture_params.target_angles)
THUMB_ALL_INDICES = list(_ref_gesture_params.phase1_hold)
PHASE1_DURATION = _ref_gesture_params.phase1
PHASE2_DURATION = _ref_gesture_params.phase2
PHASE3_DURATION = _ref_gesture_params.phase3
PHASE4_DURATION = _ref_gesture_params.phase4


class Fist(HandGesturePrimitive):
    """五指握拳 — PhasedPoseEngine + gestures 配置。"""

    def __init__(self) -> None:
        self._engine: Optional[PhasedPoseEngine] = None
        self._hand_type = ""

    @property
    def name(self) -> str:
        return "fist"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._hand_type = ""

    def _ensure_engine(self, ctx: PrimitiveContext) -> PhasedPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            engine = make_static_engine(ctx.hand_type, "fist")
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
