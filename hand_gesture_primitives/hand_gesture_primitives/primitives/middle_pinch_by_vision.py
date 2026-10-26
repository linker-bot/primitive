"""中指捏取原语 — 拇指与中指指尖对捏。

抓取类型: 1 vs 3 (拇指 vs 中指)

适用场景: 需要食指保持自由（如指向、按压其他按钮）的同时捏取物体。
拇指与中指指尖对捏，食指伸直外展远离捏合区域，无名指和小指自然弯曲避让。
"""

from typing import List

import numpy as np
from scipy.spatial.transform import Rotation

from ..angle_bridge import AngleUnits
from ..gesture_params import load_static_params_for_ctx
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    lerp_angles,
)
from ..semantic_30 import l25_compat_remap_s30, s30_pose_from_channels

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

# 中指捏合: 拇指旋转对向中指，中指弯曲对捏，食指外展避让
MIDDLE_PINCH_ANGLES = s30_pose_from_channels(
    {
        "thumb_mcp_flex": 120,
        "index_mcp_flex": 0,
        "middle_mcp_flex": 140,
        "ring_mcp_flex": 0,
        "pinky_mcp_flex": 0,
        "thumb_mcp_abd": 190,
        "index_mcp_abd": 200,
        "thumb_rot": 185,
        "thumb_dip": 160,
        "index_dip": 0,
        "middle_dip": 130,
        "ring_dip": 0,
        "pinky_dip": 0,
    },
)

MIDDLE_PINCH_ANGLES_L25 = l25_compat_remap_s30(MIDDLE_PINCH_ANGLES)


class MiddlePinchByVision(HandGesturePrimitive):
    """拇指与中指指尖对捏，食指伸直外展避让。"""

    TRANSITION_DURATION = 0.5

    @property
    def name(self) -> str:
        return "middle_pinch_by_vision"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        if ctx.hand_type in ("o6", "l6", "l25", "o30"):
            params = load_static_params_for_ctx(ctx.hand_type, self.name, ctx)
            duration = params.duration if params.duration > 0 else self.TRANSITION_DURATION
            t = elapsed / duration
            if t >= 1.0:
                return self._move(list(params.target_angles), space=u.space)
            return self._move(
                lerp_angles(self._start_angles, params.target_angles, t), space=u.space
            )

        raw_target = MIDDLE_PINCH_ANGLES_L25 if ctx.hand_type == "l25" else MIDDLE_PINCH_ANGLES
        target = u.pose(raw_target)

        if ctx.hand_type != "l25":
            if ctx.tcp_pose is None:
                return self._hold("缺少 tcp_pose")

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
            return self._move(list(target), space=u.space)
        return self._move(lerp_angles(self._start_angles, target, t), space=u.space)

    @property
    def done(self) -> bool:
        return False
