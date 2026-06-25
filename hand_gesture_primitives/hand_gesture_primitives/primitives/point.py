"""指向手势原语 — 食指伸出，其余握拢。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 指向: 食指完全伸直，其余弯曲
POINT_ANGLES = [
    230,        # [0]  thumb_base: 弯曲
    0,          # [1]  index_base: 伸直 (指向)
    255,        # [2]  middle_base: 弯曲
    255,        # [3]  ring_base: 弯曲
    255,        # [4]  pinky_base: 弯曲
    140,        # [5]  thumb_abd: 内收
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    150,        # [10] thumb_rot: 旋转压住中指
    0, 0, 0, 0,  # [11-14] rsv
    200,        # [15] thumb_tip: 弯曲
    0,          # [16] index_tip: 伸直 (指向)
    255,        # [17] middle_tip: 弯曲
    255,        # [18] ring_tip: 弯曲
    255,        # [19] pinky_tip: 弯曲
]


class Point(HandGesturePrimitive):
    """食指伸出指向。"""

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "point"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(POINT_ANGLES))
        return self._move(lerp_angles(self._start_angles, POINT_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
