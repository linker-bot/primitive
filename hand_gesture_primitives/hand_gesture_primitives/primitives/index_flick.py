"""食指拨动手势 — 食指伸出做圆周摆动。"""

import math
from typing import List

from ..angle_bridge import AngleUnits
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
    sync_pip_from_dip,
)
from ..semantic_30 import SEMANTIC_30_IDX, s30_pose_from_channels

POINT_ANGLES_255 = s30_pose_from_channels(
    {
        "thumb_mcp_flex": 120,
        "index_mcp_flex": 0,
        "middle_mcp_flex": 255,
        "ring_mcp_flex": 255,
        "pinky_mcp_flex": 255,
        "thumb_mcp_abd": 140,
        "thumb_rot": 150,
        "thumb_dip": 200,
        "index_dip": 77,
        "middle_dip": 255,
        "ring_dip": 255,
        "pinky_dip": 255,
    },
)

_IDX_INDEX_MCP = SEMANTIC_30_IDX["index_mcp_flex"]
_IDX_INDEX_DIP = SEMANTIC_30_IDX["index_dip"]


class IndexFlick(HandGesturePrimitive):
    TRANSITION_DURATION = 0.5

    FLICK_DURATION = 1.5
    CIRCLE = 3.0
    ROOT_SWING_RANGE = 102
    TIP_SWING_RANGE = 77

    END_DURATION = FLICK_DURATION + 2 * CIRCLE

    @property
    def name(self) -> str:
        return "index_flick"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        point = u.pose(sync_pip_from_dip(POINT_ANGLES_255))
        root_swing = u.step(_IDX_INDEX_MCP, self.ROOT_SWING_RANGE)
        tip_swing = u.step(_IDX_INDEX_DIP, self.TIP_SWING_RANGE)
        t = elapsed
        if t >= self.TRANSITION_DURATION and t < self.FLICK_DURATION:
            return self._move(list(point), space=u.space)
        elif t >= self.FLICK_DURATION and t < self.END_DURATION:
            dt = t - self.FLICK_DURATION
            circle_root_pos = math.cos(dt / self.CIRCLE * 2 * math.pi)
            circle_tip_pos = math.sin(dt / self.CIRCLE * 2 * math.pi)

            flick_angles = list(point)
            flick_angles[_IDX_INDEX_MCP] = point[_IDX_INDEX_MCP] + root_swing * (
                1 - circle_root_pos
            )
            flick_angles[_IDX_INDEX_DIP] = point[_IDX_INDEX_DIP] - tip_swing * circle_tip_pos
            return self._move(sync_pip_from_dip(flick_angles), space=u.space)
        elif t >= self.END_DURATION:
            return self._move(list(point), space=u.space, trajectory_finished=True)
        return self._move(
            lerp_angles(self._start_angles, point, t / self.TRANSITION_DURATION),
            space=u.space,
        )
