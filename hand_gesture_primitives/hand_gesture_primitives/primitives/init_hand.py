"""初始位手势原语 — 半张开自然姿态。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# O20 初始位: 各关节半弯曲，自然松弛
INIT_ANGLES_O20 = [
    80,         # [0]  thumb_base: 轻微弯曲 (~38°)
    80,         # [1]  index_base: 轻微弯曲 (~56°)
    80,         # [2]  middle_base: 轻微弯曲 (~56°)
    80,         # [3]  ring_base: 轻微弯曲 (~56°)
    80,         # [4]  pinky_base: 轻微弯曲 (~56°)
    60,         # [5]  thumb_abd: 轻微侧摆 (~42°)
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    50,         # [10] thumb_rot: 轻微旋转 (~25°)
    0, 0, 0, 0,  # [11-14] rsv
    60,         # [15] thumb_tip: 轻微弯曲
    60,         # [16] index_tip: 轻微弯曲
    60,         # [17] middle_tip: 轻微弯曲
    60,         # [18] ring_tip: 轻微弯曲
    60,         # [19] pinky_tip: 轻微弯曲
]

# L25 初始位: 半张开自然松弛姿态 (O20约定: 0=张开, 255=弯曲)
INIT_ANGLES_L25 = [
    85, 55, 55, 55, 55,           # ROOT1[0-4]: 五指微弯
    135, 127, 127, 127, 127,      # YAW[5-9]: 拇指微偏 + 四指中立
    115, 0, 0, 0, 0,              # ROLL[10-14]: 拇指微旋 + 预留
    55, 55, 55, 55, 55,           # ROOT2[15-19]: 各指第二根部微弯
    75, 75, 75, 75, 75,           # TIP[20-24]: 各指尖微弯
]

INIT_ANGLES = {"o20": INIT_ANGLES_O20, "l25": INIT_ANGLES_L25}


class InitHand(HandGesturePrimitive):
    """回到自然松弛初始位。"""

    TRANSITION_DURATION = 0.8

    @property
    def name(self) -> str:
        return "init"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        target = INIT_ANGLES.get(ctx.hand_type, INIT_ANGLES_O20)
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(target))
        return self._move(lerp_angles(self._start_angles, target, t))

    @property
    def done(self) -> bool:
        return False
