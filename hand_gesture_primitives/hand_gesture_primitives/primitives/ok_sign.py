"""OK手势原语 — 拇指与食指捏成圆圈，其余三指伸直张开。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# OK手势: 拇指与食指指尖对捏成圆，中指/无名指/小指伸直
OK_ANGLES = [
    0,         # [0]  thumb_base: 很轻微弯曲
    150,        # [1]  index_base: 适度弯曲
    100,        # [2]  middle_base: 弯曲
    70,         # [3]  ring_base: 中等弯曲
    40,         # [4]  pinky_base: 轻微弯曲
    170,        # [5]  thumb_abd: 往掌心偏移，微调右移
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    220,        # [10] thumb_rot: 大幅旋转对准食指
    0, 0, 0, 0,  # [11-14] rsv
    200,        # [15] thumb_tip: 指尖适度弯曲
    100,        # [16] index_tip: 指尖很轻微弯曲
    100,        # [17] middle_tip: 弯曲
    70,         # [18] ring_tip: 中等弯曲
    40,         # [19] pinky_tip: 轻微弯曲
]


class OkSign(HandGesturePrimitive):
    """拇指与食指捏圆，其余三指伸直。"""

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "ok_sign"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(OK_ANGLES))
        return self._move(lerp_angles(self._start_angles, OK_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
