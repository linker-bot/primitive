"""五指扭动手势 — 五指捏取后反复扭动。"""

import logging
from typing import List

import numpy as np

from ..angle_bridge import AngleUnits
from ..contact_detection import (
    monitors_for_fingers,
    pinch_motion_should_stop,
)
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
)
from ..semantic_30 import SEMANTIC_30_IDX
from ..twist_poses import TwistPoseSet, load_twist_poses

_logger = logging.getLogger(__name__)

_ABD_SCALE_IDX = (
    SEMANTIC_30_IDX["thumb_mcp_abd"],
    SEMANTIC_30_IDX["index_mcp_abd"],
    SEMANTIC_30_IDX["middle_mcp_abd"],
    SEMANTIC_30_IDX["ring_mcp_abd"],
    SEMANTIC_30_IDX["pinky_mcp_abd"],
)


def _poses(ctx: PrimitiveContext, u: AngleUnits) -> TwistPoseSet:
    raw = load_twist_poses(ctx.hand_type or "l25", "twist5digit")
    return TwistPoseSet(
        ready=u.pose(raw.ready),
        end=u.pose(raw.end),
        twist_delta=[u.step(i, d) for i, d in enumerate(raw.twist_delta)],
    )


class Twist5Digit(HandGesturePrimitive):
    TRANSITION_DURATION = 1.0

    PINCH_DURATION = 1.5
    CHECK_TOUCH = 0.8
    TOUCH_DURATION = 3.0
    PINCH_END_DURATION = PINCH_DURATION + CHECK_TOUCH

    TWIST_CIRCLE = 1.5
    TWIST_RELEASE_CIRCLE = 1.0
    TWIST_RESET_CIRCLE = 1.0
    TWIST_END_DURATION = PINCH_END_DURATION + 5 * (
        TWIST_CIRCLE + TWIST_RELEASE_CIRCLE + TWIST_RESET_CIRCLE
    )

    @property
    def name(self) -> str:
        return "twist5digit"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        poses = _poses(ctx, u)
        t = elapsed
        if t < self.TRANSITION_DURATION:
            self.CHECK_TOUCH = 0.8
            self.PINCH_END_DURATION = self.PINCH_DURATION + self.TOUCH_DURATION
            return self._move(
                lerp_angles(self._start_angles, poses.ready, t / self.TRANSITION_DURATION),
                space=u.space,
            )
        if t < self.PINCH_DURATION:
            return self._move(list(poses.ready), space=u.space)
        elif t < self.PINCH_END_DURATION:
            dt = (t - self.PINCH_DURATION) / self.TOUCH_DURATION
            monitor_hw, _ = monitors_for_fingers(ctx, [0, 1, 2, 3, 4])
            stop, reason = pinch_motion_should_stop(
                ctx,
                monitor_hw,
                [0, 1, 2, 3, 4],
                lerp_progress=0,
                baseline=None,
            )
            if stop:
                _logger.warning(
                    "twist5digit: 停指 reason=%s progress=%.0f%% check_touch=%f",
                    reason,
                    t * 100.0,
                    dt,
                )
                self.CHECK_TOUCH = dt
                self.PINCH_END_DURATION = t
            return self._move(lerp_angles(poses.ready, poses.end, dt), space=u.space)
        elif t < self.TWIST_END_DURATION:
            touch_state_angles = lerp_angles(poses.ready, poses.end, self.CHECK_TOUCH)
            twist_t = t - self.PINCH_END_DURATION

            stages = [
                ("TWIST_CIRCLE", self.TWIST_CIRCLE),
                ("TWIST_RELEASE_CIRCLE", self.TWIST_RELEASE_CIRCLE),
                ("TWIST_RESET_CIRCLE", self.TWIST_RESET_CIRCLE),
            ]
            total_duration = sum(d for _, d in stages)
            local_t = twist_t % total_duration

            stage_name = stages[-1][0]
            ratio = 1.0
            for i, (name, duration) in enumerate(stages):
                if i == len(stages) - 1 or local_t < duration:
                    ratio = max(0.0, min(1.0, local_t / duration))
                    stage_name = name
                    break
                local_t -= duration

            delta = list(poses.twist_delta)
            for idx in _ABD_SCALE_IDX:
                delta[idx] *= 0.8

            if stage_name == "TWIST_CIRCLE":
                base = np.array(
                    lerp_angles(
                        touch_state_angles,
                        lerp_angles(poses.ready, poses.end, min(1.0, self.CHECK_TOUCH + 0.2)),
                        ratio,
                    )
                )
                result = base + np.array(delta) * ratio
                return self._move(result.tolist(), space=u.space)
            elif stage_name == "TWIST_RELEASE_CIRCLE":
                base = np.array(
                    lerp_angles(
                        poses.ready,
                        poses.end,
                        max(0.0, self.CHECK_TOUCH - 0.5),
                    )
                )
                result = base + np.array(delta) * (1 - ratio)
                return self._move(result.tolist(), space=u.space)
            else:
                return self._move(list(touch_state_angles), space=u.space)

        return self._move(list(poses.ready), space=u.space)
