"""五指扭动手势 — 五指捏取后反复扭动。"""

import math
from typing import List

import numpy as np

from ..primitive_base import (
    ABD_NEUTRAL,
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    RESERVED_INDICES,
    lerp_angles,
)
from ..contact_detection import (
    capture_feedback_baseline,
    pinch_motion_should_stop_toque,
    pinch_motion_stop_detail,
)
from ..contact_resolver import current_monitor_indices
from ..hand_config import HandConfig
import logging

_logger = logging.getLogger(__name__)
# O20 semantic 20-DOF: 五指捏取预备姿态
PINCH_ANGLES_READY = [
    0.0,         # [0]  thumb_base: 伸出准备捏
    102.0,       # [1]  index_base: 半弯
    77.0,        # [2]  middle_base: 半弯
    89.0,        # [3]  ring_base: 半弯
    128.0,       # [4]  pinky_base: 半弯
    220.0,        # [5]  thumb_abd
    ABD_NEUTRAL + 50,  # [6]  index_abd: 稍外展
    ABD_NEUTRAL, # [7]  middle_abd
    ABD_NEUTRAL - 25,  # [8]  ring_abd: 稍内收
    ABD_NEUTRAL - 50,  # [9]  pinky_abd: 稍内收
    150.0,         # [10] thumb_rot
    0.0, 0.0, 0.0, 0.0,  # [11-14] reserved
    77.0,        # [15] thumb_tip
    51.0,        # [16] index_tip
    77.0,        # [17] middle_tip
    77.0,        # [18] ring_tip
    77.0,        # [19] pinky_tip
]

# 捏合完成姿态（手指更弯）
PINCH_ANGLES_END = [
    102.0,       # [0]  thumb_base: 更弯
    191.0,       # [1]  index_base: 更弯
    179.0,       # [2]  middle_base: 更弯
    191.0,       # [3]  ring_base: 更弯
    255.0,       # [4]  pinky_base: 更弯
    220.0,        # [5]  thumb_abd
    ABD_NEUTRAL + 50,  # [6]  index_abd
    ABD_NEUTRAL, # [7]  middle_abd
    ABD_NEUTRAL - 25,  # [8]  ring_abd
    ABD_NEUTRAL - 50,  # [9]  pinky_abd
    150.0,         # [10] thumb_rot
    0.0, 0.0, 0.0, 0.0,  # [11-14] reserved
    77.0,        # [15] thumb_tip
    80.0,       # [16] index_tip: 更弯
    80.0,       # [17] middle_tip: 更弯
    90.0,       # [18] ring_tip: 更弯
    100.0,       # [19] pinky_tip: 更弯
]

# 扭动增量（侧摆方向 + 微调）
TWIST_DELTA_ANGLE = [
    -16.0,       # [0]  thumb_base
    -35.0,       # [1]  index_base
    26.0,        # [2]  middle_base
    35.0,        # [3]  ring_base
    0.0,         # [4]  pinky_base
    -150.0,        # [5]  thumb_abd: 拇指侧摆
    -200.0,      # [6]  index_abd: 食指侧摆
    -200.0,      # [7]  middle_abd: 中指侧摆
    -100.0,      # [8]  ring_abd: 无名指侧摆
    -50.0,       # [9]  pinky_abd: 小指侧摆
    0.0,         # [10] thumb_rot
    0.0, 0.0, 0.0, 0.0,  # [11-14] reserved
    13.0,        # [15] thumb_tip
    0.0,         # [16] index_tip
    0.0,         # [17] middle_tip
    0.0,         # [18] ring_tip
    0.0,         # [19] pinky_tip
]


class Twist5Digit(HandGesturePrimitive):

    TRANSITION_DURATION = 1.0

    PINCH_DURATION = 1.5
    CHECK_TOUCH = 0.8
    TOUCH_DURATION = 3.0
    PINCH_END_DURATION = PINCH_DURATION + CHECK_TOUCH

    TWIST_CIRCLE = 1.5
    TWIST_RELEASE_CIRCLE = 1.0
    TWIST_RESET_CIRCLE = 1.0
    TWIST_END_DURATION = PINCH_END_DURATION + 5 * (TWIST_CIRCLE + TWIST_RELEASE_CIRCLE + TWIST_RESET_CIRCLE)

    @property
    def name(self) -> str:
        return "twist5digit"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed
        if t < self.TRANSITION_DURATION:
            self.CHECK_TOUCH = 0.8
            self.PINCH_END_DURATION = self.PINCH_DURATION + self.TOUCH_DURATION

            return self._move(lerp_angles(self._start_angles, PINCH_ANGLES_READY, t / self.TRANSITION_DURATION))
        if t < self.PINCH_DURATION:
            return self._move(list(PINCH_ANGLES_READY))
        elif t < self.PINCH_END_DURATION:
            dt = (t - self.PINCH_DURATION)/self.TOUCH_DURATION
            stop, reason = pinch_motion_should_stop_toque(
                ctx, [6,21,22,23,24], [0, 1],
                lerp_progress=0,
                baseline=None,
            )
            # _logger.warning(reason)
            if stop:
                _logger.warning(
                    "index_pinch: 停指 reason=%s progress=%.0f%% check_touch=%f",
                    reason, t * 100.0, dt
                )
                self.CHECK_TOUCH = dt
                self.PINCH_END_DURATION = t
            return self._move(lerp_angles(PINCH_ANGLES_READY, PINCH_ANGLES_END, dt))
        elif t < self.TWIST_END_DURATION:
            touch_state_angles = lerp_angles(PINCH_ANGLES_READY, PINCH_ANGLES_END, self.CHECK_TOUCH)
            twist_t = t - self.PINCH_END_DURATION

            stages = [
                ("TWIST_CIRCLE", self.TWIST_CIRCLE),
                ("TWIST_RELEASE_CIRCLE", self.TWIST_RELEASE_CIRCLE),
                ("TWIST_RESET_CIRCLE", self.TWIST_RESET_CIRCLE),
            ]
            total_duration = sum(d for _, d in stages)
            local_t = twist_t % total_duration

            stage_name = stages[-1][0]
            ratio = 1.0
            for i, (name, duration) in enumerate(stages):
                if i == len(stages) - 1 or local_t < duration:
                    ratio = max(0.0, min(1.0, local_t / duration))
                    stage_name = name
                    break
                local_t -= duration

            delta = list(TWIST_DELTA_ANGLE)
            delta[5] *= 1 - 0.2#self.CHECK_TOUCH
            delta[6] *= 1 - 0.2#self.CHECK_TOUCH
            delta[7] *= 1 - 0.2#self.CHECK_TOUCH
            delta[8] *= 1 - 0.2#self.CHECK_TOUCH
            delta[9] *= 1 - 0.2#self.CHECK_TOUCH

            if stage_name == "TWIST_CIRCLE":
                base = np.array(lerp_angles(
                    touch_state_angles,
                    lerp_angles(PINCH_ANGLES_READY, PINCH_ANGLES_END, min(1.0, self.CHECK_TOUCH + 0.2)),
                    ratio,
                ))
                result = base + np.array(delta) * ratio
                for i in RESERVED_INDICES:
                    result[i] = 0.0
                return self._move(result.tolist())
            elif stage_name == "TWIST_RELEASE_CIRCLE":
                base = np.array(lerp_angles(
                    PINCH_ANGLES_READY, PINCH_ANGLES_END,
                    max(0.0, self.CHECK_TOUCH - 0.5),
                ))
                result = base + np.array(delta) * (1 - ratio)
                for i in RESERVED_INDICES:
                    result[i] = 0.0
                return self._move(result.tolist())
            else:
                return self._move(list(touch_state_angles))

        return self._move(list(PINCH_ANGLES_READY))
