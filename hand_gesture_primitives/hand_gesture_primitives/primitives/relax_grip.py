"""放松再抓紧原语 — 先张开放松，再握紧。"""

from typing import List

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

RELAX_ANGLES = [
    0, 0, 0, 0, 0,               # base: 全部伸直
    0, ABD_NEUTRAL, ABD_NEUTRAL, ABD_NEUTRAL, ABD_NEUTRAL,  # abd
    0,                            # thumb_rot
    0, 0, 0, 0,                   # rsv
    0, 0, 0, 0, 0,               # tip: 全部伸直
]

GRIP_ANGLES = [
    255, 255, 255, 255, 255,      # base: 最大弯曲
    140, ABD_NEUTRAL, ABD_NEUTRAL, ABD_NEUTRAL, ABD_NEUTRAL,  # abd
    160,                          # thumb_rot: 压住
    0, 0, 0, 0,                   # rsv
    255, 255, 255, 255, 255,      # tip: 最大弯曲
]

# 时间轴
RELAX_DURATION = 0.6   # 放松用时
HOLD_DURATION = 0.3    # 放松后停顿
GRIP_DURATION = 0.4    # 抓紧用时


class RelaxGrip(HandGesturePrimitive):
    """放松再抓紧：张开 → 停顿 → 握紧。

    时间线:
      0.0 ~ 0.6s  从当前位置插值到全张开 (放松)
      0.6 ~ 0.9s  保持张开 (停顿)
      0.9 ~ 1.3s  从张开插值到握紧
      1.3s+       保持握紧，标记 done
    """

    @property
    def name(self) -> str:
        return "relax_grip"

    @property
    def done(self) -> bool:
        return self._done

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._done = False

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t_relax_end = RELAX_DURATION
        t_hold_end = t_relax_end + HOLD_DURATION
        t_grip_end = t_hold_end + GRIP_DURATION

        if elapsed < t_relax_end:
            t = elapsed / RELAX_DURATION
            return self._move(lerp_angles(self._start_angles, RELAX_ANGLES, t))

        elif elapsed < t_hold_end:
            return self._move(list(RELAX_ANGLES))

        elif elapsed < t_grip_end:
            t = (elapsed - t_hold_end) / GRIP_DURATION
            return self._move(lerp_angles(RELAX_ANGLES, GRIP_ANGLES, t))

        else:
            self._done = True
            return self._move(list(GRIP_ANGLES))
