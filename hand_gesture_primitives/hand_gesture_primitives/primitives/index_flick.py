"""食指拨动手势 — 食指伸出做圆周摆动。"""

import math
from typing import List

from ..primitive_base import (
    ABD_NEUTRAL,
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    RESERVED_INDICES,
    lerp_angles,
)

# O20 semantic 20-DOF: 食指伸出、其余握拳
POINT_ANGLES = [
    120.0,       # [0]  thumb_base: 收拢
    0.0,         # [1]  index_base: 伸直
    255.0,       # [2]  middle_base: 握拳
    255.0,       # [3]  ring_base: 握拳
    255.0,       # [4]  pinky_base: 握拳
    140.0,       # [5]  thumb_abd
    ABD_NEUTRAL, # [6]  index_abd
    ABD_NEUTRAL, # [7]  middle_abd
    ABD_NEUTRAL, # [8]  ring_abd
    ABD_NEUTRAL, # [9]  pinky_abd
    150.0,       # [10] thumb_rot
    0.0, 0.0, 0.0, 0.0,  # [11-14] reserved
    200.0,       # [15] thumb_tip: 收拢
    77.0,        # [16] index_tip: 微弯（摆动中心）
    255.0,       # [17] middle_tip: 握拳
    255.0,       # [18] ring_tip: 握拳
    255.0,       # [19] pinky_tip: 握拳
]


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
        t = elapsed
        if t >= self.TRANSITION_DURATION and t < self.FLICK_DURATION:
            return self._move(list(POINT_ANGLES))
        elif t >= self.FLICK_DURATION and t < self.END_DURATION:
            dt = t - self.FLICK_DURATION
            circle_root_pos = math.cos(dt / self.CIRCLE * 2 * math.pi)
            circle_tip_pos = math.sin(dt / self.CIRCLE * 2 * math.pi)

            flick_angles = list(POINT_ANGLES)
            # O20: 值越大越弯, (1-cos) 从0到2, 使食指做弯-伸循环
            flick_angles[1] = POINT_ANGLES[1] + self.ROOT_SWING_RANGE * (1 - circle_root_pos)
            # tip 正弦摆动 (O20 方向: 减=伸, 加=弯)
            flick_angles[16] = POINT_ANGLES[16] - self.TIP_SWING_RANGE * circle_tip_pos
            for i in RESERVED_INDICES:
                flick_angles[i] = 0.0
            return self._move(flick_angles)
        elif t >= self.END_DURATION:
            return self._move(list(POINT_ANGLES))
        return self._move(lerp_angles(self._start_angles, POINT_ANGLES, t / self.TRANSITION_DURATION))
