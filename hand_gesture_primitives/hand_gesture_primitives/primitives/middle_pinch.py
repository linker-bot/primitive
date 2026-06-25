"""中指捏取原语 — 拇指与中指指尖对捏。

抓取类型: 1 vs 3 (拇指 vs 中指)

适用场景: 需要食指保持自由（如指向、按压其他按钮）的同时捏取物体。
拇指与中指指尖对捏，食指伸直外展远离捏合区域，无名指和小指自然弯曲避让。
"""

import logging
from typing import List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

_logger = logging.getLogger(__name__)

# 力反馈: 活动关节电流(绝对值, mA)超过此阈值 → 停止闭合并完成
CURRENT_STOP_THRESHOLD = 400

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# 中指捏合: 拇指旋转对向中指，中指弯曲对捏，食指外展避让
MIDDLE_PINCH_ANGLES = [
    120,        # [0]  thumb_base: 适度弯曲 (~56°)
    0,          # [1]  index_base: 伸直避让
    140,        # [2]  middle_base: 适度弯曲 (~99°)
    0,          # [3]  ring_base: 轻微弯曲避让
    0,          # [4]  pinky_base: 轻微弯曲避让
    190,        # [5]  thumb_abd: 内收 (~92°)
    200,        # [6]  index_abd: 外展远离中指 (~+28°)
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    185,        # [10] thumb_rot: 旋转对向中指 (~94°)
    0, 0, 0, 0,  # [11-14] rsv
    160,        # [15] thumb_tip: 指尖弯曲 (~94°)
    0,          # [16] index_tip: 伸直
    130,        # [17] middle_tip: 指尖弯曲对捏 (~92°)
    0,          # [18] ring_tip: 轻微弯曲
    0,          # [19] pinky_tip: 轻微弯曲
]


class MiddlePinch(HandGesturePrimitive):
    """拇指与中指指尖对捏，食指伸直外展避让。"""

    TRANSITION_DURATION = 0.5
    # 监测关节: thumb_base, thumb_tip, middle_base, middle_tip
    _MONITORED = (0, 15, 2, 17)

    @property
    def name(self) -> str:
        return "middle_pinch"

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._frozen_pose: Optional[List[float]] = None
        self._done = False

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        # 本帧闭合目标
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            target = list(MIDDLE_PINCH_ANGLES)
        else:
            target = lerp_angles(self._start_angles, MIDDLE_PINCH_ANGLES, t)

        # 力反馈: 拇指+中指任一关节电流(绝对值)超阈值 → 冻结姿态并完成
        currents = ctx.joint_currents
        if any(i < len(currents) and currents[i] > CURRENT_STOP_THRESHOLD
               for i in self._MONITORED):
            if self._frozen_pose is None:
                self._frozen_pose = list(target)
                self._done = True
                _logger.warning(
                    "middle_pinch: 电流>%dma，停止闭合并保持(done)", CURRENT_STOP_THRESHOLD)
            return self._move(list(self._frozen_pose))

        return self._move(target)

    @property
    def done(self) -> bool:
        return self._done
