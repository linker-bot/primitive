"""平行伸展捏取原语 — 拇指与四指平行伸展对捏，五指均保持伸展，如平行夹爪。

抓取类型: 1 vs 2-5 (拇指 vs 四指，全部平行伸展)

适用场景: 抓取扁平物体如书本、卡片、平板、薄板等。
五指全部保持伸展姿态，拇指在一侧、四指并拢在另一侧，
形成"平行夹爪"——两面平行贴合物体。

与 pinch 的区别: pinch 是指尖捏合(bend to pinch)，
parallel_extension 是整指伸展贴合(extend to clamp)。
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


PARALLEL_EXT_ANGLES = [
    120,          # [0]  thumb_base: 伸直
    235,          # [1]  index_base: 伸直
    235,          # [2]  middle_base: 伸直
    235,          # [3]  ring_base: 伸直
    235,          # [4]  pinky_base: 伸直
    235,        # [5]  thumb_abd: 外展张开对向四指
    193,         # [6]  index_abd: 内收并拢
    150,         # [7]  middle_abd: 内收并拢
    105,         # [8]  ring_abd: 内收并拢
    45,         # [9]  pinky_abd: 内收并拢
    245,        # [10] thumb_rot: 旋转使指腹平行于四指面
    0, 0, 0, 0,  # [11-14] rsv
    0,          # [15] thumb_tip: 伸直
    0,          # [16] index_tip: 伸直
    0,          # [17] middle_tip: 伸直
    0,          # [18] ring_tip: 伸直
    0,          # [19] pinky_tip: 伸直
]


class ParallelExtensionByVision(HandGesturePrimitive):
    """拇指与四指平行伸展对捏 (1 vs 2-5)。

    五指全部保持伸展姿态，拇指在一侧、四指并拢在另一侧，
    形成平行夹爪结构，适合抓取书本、卡片等扁平物体。
    """

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "parallel_extension_by_vision"

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
            return self._move(list(PARALLEL_EXT_ANGLES))
        return self._move(lerp_angles(self._start_angles, PARALLEL_EXT_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
