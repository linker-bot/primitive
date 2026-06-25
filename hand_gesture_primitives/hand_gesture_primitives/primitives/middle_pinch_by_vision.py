"""中指捏取原语 — 拇指与中指指尖对捏。

抓取类型: 1 vs 3 (拇指 vs 中指)

适用场景: 需要食指保持自由（如指向、按压其他按钮）的同时捏取物体。
拇指与中指指尖对捏，食指伸直外展远离捏合区域，无名指和小指自然弯曲避让。
"""

from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

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


class MiddlePinchByVision(HandGesturePrimitive):
    """拇指与中指指尖对捏，食指伸直外展避让。"""

    TRANSITION_DURATION = 0.5

    @property
    def name(self) -> str:
        return "middle_pinch_by_vision"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        if ctx.tcp_pose is None:
            return self._hold("缺少 tcp_pose")

        # 无 object_pose 时跳过可达性检查
        # (GraspGate 已完成三判定)
        if ctx.object_pose is not None:
            tcp_to_obj = ctx.object_pose.position - ctx.tcp_pose.position
            tcp_dist = float(np.linalg.norm(tcp_to_obj))

            tcp_quat = ctx.tcp_pose.orientation
            if not np.allclose(tcp_quat, [0, 0, 0, 1]):
                R_tcp = Rotation.from_quat(tcp_quat).as_matrix()
                palm_forward = R_tcp[:, 2]
                forward_dist = float(np.dot(tcp_to_obj, palm_forward))
                if forward_dist < PALM_FORWARD_MIN or forward_dist > PALM_FORWARD_MAX:
                    return self._hold("物体不在掌心前方 2–15cm")
            else:
                if tcp_dist > REACH_THRESHOLD:
                    return self._hold("TCP到物体距离过远")

        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(MIDDLE_PINCH_ANGLES))
        return self._move(lerp_angles(self._start_angles, MIDDLE_PINCH_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
