"""张开手势原语 — 从当前位置相对张开，参数为 0.0~1.0 的张开比例。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 完全张开的角度（张开方向的极限）
OPEN_ANGLES = [
    0,          # [0]  thumb_base
    0,          # [1]  index_base
    0,          # [2]  middle_base
    0,          # [3]  ring_base
    0,          # [4]  pinky_base
    0,          # [5]  thumb_abd
    ABD_NEUTRAL,  # [6]  index_abd
    ABD_NEUTRAL,  # [7]  middle_abd
    ABD_NEUTRAL,  # [8]  ring_abd
    ABD_NEUTRAL,  # [9]  pinky_abd
    0,          # [10] thumb_rot
    0, 0, 0, 0,  # [11-14] rsv
    0,          # [15] thumb_tip
    0,          # [16] index_tip
    0,          # [17] middle_tip
    0,          # [18] ring_tip
    0,          # [19] pinky_tip
]


class Release(HandGesturePrimitive):
    """从当前位置相对张开。degree=1.0 完全张开到底，degree=0.5 张开剩余距离的50%。"""

    TRANSITION_DURATION = 0.6

    def __init__(self, degree: str = "1.0"):
        self._degree = max(0.0, min(1.0, float(degree)))

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._target = [
            s + (o - s) * self._degree
            for s, o in zip(self._start_angles, OPEN_ANGLES)
        ]

    @property
    def name(self) -> str:
        return "release"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(self._target))
        return self._move(lerp_angles(self._start_angles, self._target, t))

    @property
    def done(self) -> bool:
        return False
