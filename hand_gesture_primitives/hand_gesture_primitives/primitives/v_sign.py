"""剪刀手原语 — 食指与中指伸直张开，其余握拢。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 剪刀手: 食指和中指伸直并张开，拇指/无名指/小指弯曲
V_SIGN_ANGLES = [
    200,        # [0]  thumb_base: 弯曲握拢
    0,          # [1]  index_base: 伸直
    0,          # [2]  middle_base: 伸直
    255,        # [3]  ring_base: 弯曲握拢
    255,        # [4]  pinky_base: 弯曲握拢
    140,        # [5]  thumb_abd: 内收压住
    255,        # [6]  index_abd: 略微张开（V形）
    0,        # [7]  middle_abd: 略微张开（V形）
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    150,        # [10] thumb_rot: 旋转压住无名指
    0, 0, 0, 0,  # [11-14] rsv
    255,        # [15] thumb_tip: 弯曲握拢
    0,          # [16] index_tip: 伸直
    0,          # [17] middle_tip: 伸直
    255,        # [18] ring_tip: 弯曲握拢
    255,        # [19] pinky_tip: 弯曲握拢
]


class VSign(HandGesturePrimitive):
    """食指与中指伸直张开成V形。"""

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "v_sign"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(V_SIGN_ANGLES))
        return self._move(lerp_angles(self._start_angles, V_SIGN_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
