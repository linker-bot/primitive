"""抓盘子原语 — 五指C形夹持，适合抓取盘/碟/扁平圆形物体。

适用场景: 抓取盘子、碟子、圆形扁平物体。
三阶段: 拇指向中指侧摆定位 → 四指弯成C弧 → 拇指合拢握住

抓取类型: 1 vs 2-5 (拇指 vs 四指C形)
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
    sync_pip_from_dip,
)
from ..semantic_30 import SEMANTIC_30_IDX, l25_compat_remap_s30, s30_pose_from_channels

REACH_THRESHOLD = 0.15
PALM_FORWARD_MIN = 0.02
PALM_FORWARD_MAX = 0.15

_I = SEMANTIC_30_IDX

# 最终目标: 四指C弧 + 拇指最大限度侧摆对齐中指握住（S30）
# pip 默认 0（与历史 O20 rsv 一致）；O30 root2 由 from_semantic mask 消化
DISK_ANGLES = s30_pose_from_channels(
    {
        "thumb_mcp_flex": 255,
        "index_mcp_flex": 140,
        "middle_mcp_flex": 140,
        "ring_mcp_flex": 140,
        "pinky_mcp_flex": 140,
        "thumb_mcp_abd": 255,
        "thumb_rot": 255,
        "thumb_dip": 255,
        "index_dip": 130,
        "middle_dip": 130,
        "ring_dip": 130,
        "pinky_dip": 130,
    },
)

DISK_ANGLES_L25 = l25_compat_remap_s30(DISK_ANGLES)

# 拇指通道
_THUMB_ABD = _I["thumb_mcp_abd"]
_THUMB_FLEX = (_I["thumb_mcp_flex"], _I["thumb_rot"], _I["thumb_dip"])
_FINGER_MCP = (
    _I["index_mcp_flex"],
    _I["middle_mcp_flex"],
    _I["ring_mcp_flex"],
    _I["pinky_mcp_flex"],
)
_FINGER_PIP = (
    _I["index_pip"],
    _I["middle_pip"],
    _I["ring_pip"],
    _I["pinky_pip"],
)
_FINGER_DIP = (
    _I["index_dip"],
    _I["middle_dip"],
    _I["ring_dip"],
    _I["pinky_dip"],
)

PHASE1_DURATION = 0.4  # 拇指侧摆先过去定位
PHASE2_DURATION = 0.5  # 四指弯成C弧
PHASE3_DURATION = 0.5  # 拇指弯曲合拢握住


class DiskByVision(HandGesturePrimitive):
    """五指C形夹持，适合抓取盘/碟/扁平圆形物体。

    三阶段:
      P1(0-0.4s): 拇指abd大幅侧摆到中指位置，其余不动
      P2(0.4-0.9s): 四指弯成C弧，拇指保持侧摆位但弯曲关节不动
      P3(0.9-1.4s): 拇指base/rot/tip弯曲合拢握住
    """

    @property
    def name(self) -> str:
        return "disk_by_vision"

    def on_enter(self, current_angles: List[float]) -> None:
        # 阶段目标在 compute 内经 AngleUnits.pose 现算，避免 on_enter 无 ctx 时写错量纲
        super().on_enter(current_angles)

    def _phase_targets(
        self, u: AngleUnits, start: List[float], raw_final: List[float]
    ) -> tuple[List[float], List[float], List[float]]:
        final = u.pose(sync_pip_from_dip(raw_final))
        p1 = list(start)
        p1[_THUMB_ABD] = final[_THUMB_ABD]
        p2 = list(final)
        for i in _THUMB_FLEX:
            p2[i] = start[i]
        return p1, p2, final

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        if ctx.hand_type in ("o6", "l6", "l25", "o30"):
            from ..semantic_30 import SEMANTIC_30_IDX, as_semantic_30

            params = load_static_params_for_ctx(ctx.hand_type, self.name, ctx)
            target = as_semantic_30(params.target_angles)
            start = as_semantic_30(self._start_angles)
            p1_target = list(start)
            p1_target[SEMANTIC_30_IDX["thumb_mcp_abd"]] = target[SEMANTIC_30_IDX["thumb_mcp_abd"]]
            p2_target = list(p1_target)
            for ch in (
                "index_mcp_flex",
                "middle_mcp_flex",
                "ring_mcp_flex",
                "pinky_mcp_flex",
                "index_pip",
                "middle_pip",
                "ring_pip",
                "pinky_pip",
                "index_dip",
                "middle_dip",
                "ring_dip",
                "pinky_dip",
            ):
                i = SEMANTIC_30_IDX[ch]
                p2_target[i] = target[i]
            t1 = PHASE1_DURATION
            t2 = t1 + PHASE2_DURATION
            t3 = t2 + PHASE3_DURATION
            if elapsed < t1:
                return self._move(lerp_angles(start, p1_target, elapsed / t1), space=u.space)
            elif elapsed < t2:
                return self._move(
                    lerp_angles(p1_target, p2_target, (elapsed - t1) / PHASE2_DURATION),
                    space=u.space,
                )
            elif elapsed < t3:
                return self._move(
                    lerp_angles(p2_target, target, (elapsed - t2) / PHASE3_DURATION),
                    space=u.space,
                )
            return self._move(list(target), space=u.space)

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

        p1_target, p2_target, final_target = self._phase_targets(
            u, self._start_angles, DISK_ANGLES
        )
        if elapsed < PHASE1_DURATION:
            t = elapsed / PHASE1_DURATION
            return self._move(lerp_angles(self._start_angles, p1_target, t), space=u.space)
        elif elapsed < PHASE1_DURATION + PHASE2_DURATION:
            t = (elapsed - PHASE1_DURATION) / PHASE2_DURATION
            return self._move(lerp_angles(p1_target, p2_target, t), space=u.space)
        elif elapsed < PHASE1_DURATION + PHASE2_DURATION + PHASE3_DURATION:
            t = (elapsed - PHASE1_DURATION - PHASE2_DURATION) / PHASE3_DURATION
            return self._move(lerp_angles(p2_target, final_target, t), space=u.space)
        return self._move(list(final_target), space=u.space)

    @property
    def done(self) -> bool:
        return False
