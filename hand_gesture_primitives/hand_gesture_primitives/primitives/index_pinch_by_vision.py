"""食指捏取原语 — 拇指与食指指尖对捏。

抓取类型: 1 vs 2 (拇指 vs 食指)

适用场景: 精细捏取小物体，如螺丝、针、薄片等。
拇指指尖与食指指尖精确对捏，其余三指伸开避让。
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


class IndexPinchByVision(HandGesturePrimitive):
    """拇指与食指指尖对捏，其余手指伸直避让。"""

    TRANSITION_DURATION = 0.5

    @property
    def name(self) -> str:
        return "index_pinch_by_vision"

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
            return self._move(list(INDEX_PINCH_ANGLES))
        return self._move(lerp_angles(self._start_angles, INDEX_PINCH_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
