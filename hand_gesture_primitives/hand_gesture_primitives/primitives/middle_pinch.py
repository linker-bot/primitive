"""中指捏取原语 — 拇指与中指对捏。

O6：timed lerp + 压感/堵转力矩停指（空载不用 Δ，避免误触 done）。
O20/L25：gestures 默认 + StaticPoseEngine，电流/触觉力反馈停指。
"""

import logging
from typing import List, Optional

from ..contact_detection import (
    capture_feedback_baseline,
    pinch_motion_should_stop,
    pinch_motion_stop_detail,
)
from ..contact_resolver import current_monitor_indices
from ..gesture_engine import StaticPoseEngine, make_static_engine
from ..gesture_params import CANONICAL_SEMANTIC_HAND, load_static_gesture_params
from ..hand_config import HandConfig
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult

_logger = logging.getLogger(__name__)

_ref_gesture_params = load_static_gesture_params(
    CANONICAL_SEMANTIC_HAND, "middle_pinch")
MIDDLE_PINCH_ANGLES = list(_ref_gesture_params.target_angles)
TRANSITION_DURATION = _ref_gesture_params.duration


class MiddlePinch(HandGesturePrimitive):
    """拇指与中指对捏 — StaticPoseEngine + 压感/堵转停指。"""

    def __init__(self) -> None:
        self._engine: Optional[StaticPoseEngine] = None
        self._hand_type = ""
        self._frozen_pose: Optional[List[float]] = None
        self._baseline: Optional[List[float]] = None
        self._done = False

    @property
    def name(self) -> str:
        return "middle_pinch"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._hand_type = ""
        self._frozen_pose = None
        self._baseline = None
        self._done = False

    def _ensure_engine(self, ctx: PrimitiveContext) -> StaticPoseEngine:
        if self._engine is None or self._hand_type != ctx.hand_type:
            engine = make_static_engine(ctx.hand_type, "middle_pinch")
            engine.reset(self._start_angles)
            self._engine = engine
            self._hand_type = ctx.hand_type
        return self._engine

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        if self._frozen_pose is not None:
            return self._move(list(self._frozen_pose))

        engine = self._ensure_engine(ctx)
        duration = max(engine._params.duration, 1e-6)
        progress = min(elapsed / duration, 1.0)
        target = engine.compute(elapsed)

        hw = current_monitor_indices(HandConfig(ctx.hand_type), "middle_pinch")
        if self._baseline is None:
            self._baseline = capture_feedback_baseline(ctx)

        stop, reason = pinch_motion_should_stop(
            ctx, hw, [0, 2],
            lerp_progress=progress,
            baseline=self._baseline,
        )
        if stop:
            self._frozen_pose = list(current_angles)
            self._done = True
            detail = pinch_motion_stop_detail(ctx, hw, [0, 2], self._baseline)
            _logger.warning(
                "middle_pinch: 停指 reason=%s progress=%.0f%% | %s",
                reason, progress * 100.0, detail,
            )
            return self._move(list(self._frozen_pose))

        return self._move(target)

    @property
    def done(self) -> bool:
        return self._done
