"""捏合手势原语 — 拇指与食指对捏。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 捏合: 拇指弯曲+旋转对向食指，食指弯曲，其余伸直
PINCH_ANGLES = [
    180,        # [0]  thumb_base: 中等弯曲 (~85°)
    180,        # [1]  index_base: 中等弯曲 (~180°)
    0,          # [2]  middle_base: 伸直
    0,          # [3]  ring_base: 伸直
    0,          # [4]  pinky_base: 伸直
    130,        # [5]  thumb_abd: 侧摆内收 (~92°)
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    200,        # [10] thumb_rot: 旋转对向食指 (~102°)
    0, 0, 0, 0,  # [11-14] rsv
    180,        # [15] thumb_tip: 弯曲 (~106°)
    180,        # [16] index_tip: 弯曲 (~127°)
    0,          # [17] middle_tip: 伸直
    0,          # [18] ring_tip: 伸直
    0,          # [19] pinky_tip: 伸直
]


class Pinch(HandGesturePrimitive):
    """拇指与食指对捏。"""

    TRANSITION_DURATION = 0.5

    @property
    def name(self) -> str:
        return "pinch"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(PINCH_ANGLES))
        return self._move(lerp_angles(self._start_angles, PINCH_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
