"""放松再抓紧原语 — 先张开放松，再握紧。"""

from typing import List, Optional

from ..gesture_engine import RelaxGripEngine, make_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_ref_gesture_params = load_static_gesture_params(
    CANONICAL_SEMANTIC_HAND, "relax_grip")
RELAX_ANGLES = list(_ref_gesture_params.relax_angles)
GRIP_ANGLES = list(_ref_gesture_params.grip_angles)
RELAX_DURATION = _ref_gesture_params.duration
HOLD_DURATION = _ref_gesture_params.hold_duration
GRIP_DURATION = _ref_gesture_params.grip_duration


class RelaxGrip(HandGesturePrimitive):
    """放松再抓紧 — RelaxGripEngine + gestures 配置。"""

    def __init__(self) -> None:
        self._engine: Optional[RelaxGripEngine] = None
        self._hand_type = ""
        self._done = False

    @property
    def name(self) -> str:
        return "relax_grip"

    @property
    def done(self) -> bool:
        return self._done

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._hand_type = ""
        self._done = False

    def _ensure_engine(self, ctx: PrimitiveContext) -> RelaxGripEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            engine = make_static_engine(ctx.hand_type, "relax_grip")
            engine.reset(self._start_angles)
            self._engine = engine
            self._hand_type = ctx.hand_type
        return self._engine

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        engine = self._ensure_engine(ctx)
        angles = engine.compute(elapsed)
        self._done = engine.done
        return self._move(angles)
