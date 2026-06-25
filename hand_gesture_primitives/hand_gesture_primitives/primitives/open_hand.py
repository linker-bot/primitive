"""张开手势原语 — 五指完全伸直。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# O20 全张开: 弯曲关节=0(伸直)，侧摆=128(中立)
OPEN_ANGLES_O20 = [
    0, 0, 0, 0, 0,              # [0-4] thumb/index/middle/ring/pinky_base
    0,                           # [5]  thumb_abd
    ABD_NEUTRAL, ABD_NEUTRAL,    # [6-7] index/middle_abd
    ABD_NEUTRAL, ABD_NEUTRAL,    # [8-9] ring/pinky_abd
    0,                           # [10] thumb_rot
    0, 0, 0, 0,                 # [11-14] rsv
    0, 0, 0, 0, 0,              # [15-19] thumb/index/middle/ring/pinky_tip
]

# L25 张开: SDK L25_positions.yaml 张开位姿, 转为 O20 约定 (0=张开, 255=弯曲)
# SDK 原始 [96,255,255,255,255,150,114,151,189,255,180,
#           255,255,255,255,255,255,255,255,255,255,255,255,255,255]
# O20约定 = 255 - SDK值:
OPEN_ANGLES_L25 = [
    159, 0, 0, 0, 0,            # ROOT1[0-4]: 拇指微弯, 四指全张
    105, 141, 104, 66, 0,       # YAW[5-9]
    75, 0, 0, 0, 0,             # ROLL[10-14]: 拇指旋转 + 预留
    0, 0, 0, 0, 0,             # ROOT2[15-19]: 全张
    0, 0, 0, 0, 0,             # TIP[20-24]: 全张
]

OPEN_ANGLES = {"o20": OPEN_ANGLES_O20, "l25": OPEN_ANGLES_L25}


class OpenHand(HandGesturePrimitive):
    """五指完全张开。"""

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "open"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        target = OPEN_ANGLES.get(ctx.hand_type, OPEN_ANGLES_O20)
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(target))
        return self._move(lerp_angles(self._start_angles, target, t))

    @property
    def done(self) -> bool:
        return False
