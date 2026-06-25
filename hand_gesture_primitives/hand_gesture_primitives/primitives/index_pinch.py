"""食指捏取原语 — 拇指与食指指尖对捏。

抓取类型: 1 vs 2 (拇指 vs 食指)

适用场景: 精细捏取小物体，如螺丝、针、薄片等。
拇指指尖与食指指尖精确对捏，其余三指伸开避让。
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
CURRENT_STOP_THRESHOLD = 200

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# 食指捏合: 拇指向食指侧摆+旋转对向食指，食指弯曲，其余伸直
INDEX_PINCH_ANGLES = [
    110,        # [0]  thumb_base: 适度弯曲 (~52°)
    140,        # [1]  index_base: 适度弯曲 (~99°)
    0,          # [2]  middle_base: 伸直避让
    0,          # [3]  ring_base: 伸直避让
    0,          # [4]  pinky_base: 伸直避让
    140,        # [5]  thumb_abd: 内收对准食指 (~99°)
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    200,        # [10] thumb_rot: 旋转对向食指 (~102°)
    0, 0, 0, 0,  # [11-14] rsv
    160,        # [15] thumb_tip: 指尖弯曲对捏 (~94°)
    130,        # [16] index_tip: 指尖弯曲对捏 (~92°)
    0,          # [17] middle_tip: 伸直
    0,          # [18] ring_tip: 伸直
    0,          # [19] pinky_tip: 伸直
]


class IndexPinch(HandGesturePrimitive):
    """拇指与食指指尖对捏，其余手指伸直避让。"""

    TRANSITION_DURATION = 0.5
    # 监测关节: thumb_base, thumb_tip, index_base, index_tip
    _MONITORED = (0, 15, 1, 16)

    @property
    def name(self) -> str:
        return "index_pinch"

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
            target = list(INDEX_PINCH_ANGLES)
        else:
            target = lerp_angles(self._start_angles, INDEX_PINCH_ANGLES, t)

        # 力反馈: 拇指+食指任一关节电流(绝对值)超阈值 → 冻结姿态并完成
        currents = ctx.joint_currents
        if any(i < len(currents) and currents[i] > CURRENT_STOP_THRESHOLD
               for i in self._MONITORED):
            if self._frozen_pose is None:
                self._frozen_pose = list(target)
                self._done = True
                _logger.warning(
                    "index_pinch: 电流>%dma，停止闭合并保持(done)", CURRENT_STOP_THRESHOLD)
            return self._move(list(self._frozen_pose))

        return self._move(target)

    @property
    def done(self) -> bool:
        return self._done
