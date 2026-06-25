"""握拳手势原语 — 五指完全弯曲握紧（分阶段避免拇指与食指碰撞）。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 握拳: 所有弯曲关节拉满，侧摆中立，拇指包住其他手指
FIST_ANGLES = [
    60,        # [0]  thumb_base: 弯曲但不接触食指
    255,        # [1]  index_base: 最大弯曲
    255,        # [2]  middle_base: 最大弯曲
    255,        # [3]  ring_base: 最大弯曲
    255,        # [4]  pinky_base: 最大弯曲
    200,        # [5]  thumb_abd: 内收包住
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    230,        # [10] thumb_rot: 旋转压住食指
    0, 0, 0, 0,  # [11-14] rsv
    255,        # [15] thumb_tip: 弯曲但不接触食指
    255,        # [16] index_tip: 最大弯曲
    255,        # [17] middle_tip: 最大弯曲
    255,        # [18] ring_tip: 最大弯曲
    255,        # [19] pinky_tip: 最大弯曲
]

# 拇指相关的索引
THUMB_ALL_INDICES = [0, 5, 10, 15]  # thumb_base, thumb_abd, thumb_rot, thumb_tip

PHASE1_DURATION = 0.5  # 四指闭合，拇指完全不动
PHASE2_DURATION = 0.4  # 拇指侧摆立起（thumb_abd）
PHASE3_DURATION = 0.4  # 拇指旋转（thumb_rot）
PHASE4_DURATION = 0.4  # 拇指弯曲压住（thumb_base + thumb_tip）


class Fist(HandGesturePrimitive):
    """五指握拳（四阶段：四指闭合 → 拇指立起 → 拇指旋转 → 拇指弯曲）。"""

    @property
    def name(self) -> str:
        return "fist"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        # 阶段1目标：四指到位，拇指完全保持起始位置
        self._phase1_target = list(FIST_ANGLES)
        for i in THUMB_ALL_INDICES:
            self._phase1_target[i] = self._start_angles[i]
        # 阶段2目标：拇指侧摆(abd)到位，其余拇指关节不动
        self._phase2_target = list(self._phase1_target)
        self._phase2_target[5] = FIST_ANGLES[5]   # thumb_abd
        # 阶段3目标：拇指旋转(rot)到位，弯曲仍不动
        self._phase3_target = list(self._phase2_target)
        self._phase3_target[10] = FIST_ANGLES[10]  # thumb_rot

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t1_end = PHASE1_DURATION
        t2_end = t1_end + PHASE2_DURATION
        t3_end = t2_end + PHASE3_DURATION
        t4_end = t3_end + PHASE4_DURATION

        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return self._move(lerp_angles(self._start_angles, self._phase1_target, t))
        elif elapsed < t2_end:
            t = (elapsed - t1_end) / PHASE2_DURATION
            return self._move(lerp_angles(self._phase1_target, self._phase2_target, t))
        elif elapsed < t3_end:
            t = (elapsed - t2_end) / PHASE3_DURATION
            return self._move(lerp_angles(self._phase2_target, self._phase3_target, t))
        elif elapsed < t4_end:
            t = (elapsed - t3_end) / PHASE4_DURATION
            return self._move(lerp_angles(self._phase3_target, FIST_ANGLES, t))
        else:
            return self._move(list(FIST_ANGLES))

    @property
    def done(self) -> bool:
        return False
