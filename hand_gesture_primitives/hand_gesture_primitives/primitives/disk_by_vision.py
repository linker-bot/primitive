"""抓盘子原语 — 五指C形夹持，适合抓取盘/碟/扁平圆形物体。

适用场景: 抓取盘子、碟子、圆形扁平物体。
三阶段: 拇指向中指侧摆定位 → 四指弯成C弧 → 拇指合拢握住

抓取类型: 1 vs 2-5 (拇指 vs 四指C形)
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

# 最终目标: 四指C弧 + 拇指最大限度侧摆对齐中指握住
DISK_ANGLES = [
    255,        # [0]  thumb_base: 最大限度弯曲 (255=120°)
    140,        # [1]  index_base: C弧 (~99°)
    140,        # [2]  middle_base: C弧
    140,        # [3]  ring_base: C弧 (~99°)
    140,        # [4]  pinky_base: C弧 (~99°)
    255,        # [5]  thumb_abd: 最大限度侧摆 (255=180°) — 对齐中指位置
    ABD_NEUTRAL,
    ABD_NEUTRAL,
    ABD_NEUTRAL,
    ABD_NEUTRAL,
    255,        # [10] thumb_rot: 最大限度旋转 (255=130°)
    0, 0, 0, 0,
    255,        # [15] thumb_tip: 最大限度弯曲 (255=150°)
    130,        # [16] index_tip: C弧 (~92°)
    130,        # [17] middle_tip: C弧 (~92°)
    130,        # [18] ring_tip: C弧 (~92°)
    130,        # [19] pinky_tip: C弧 (~92°)
]

# 拇指所有关节
THUMB_INDICES = [0, 5, 10, 15]          # base, abd, rot, tip
# 拇指弯曲关节 (除去abd), 最后阶段才合拢
THUMB_FLEX_INDICES = [0, 10, 15]         # base, rot, tip

PHASE1_DURATION = 0.4   # 拇指侧摆先过去定位
PHASE2_DURATION = 0.5   # 四指弯成C弧
PHASE3_DURATION = 0.5   # 拇指弯曲合拢握住


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
        super().on_enter(current_angles)

        # P1目标: 只有thumb_abd侧摆过去，其余全保持起始
        self._p1_target = list(self._start_angles)
        self._p1_target[5] = DISK_ANGLES[5]  # thumb_abd = 255

        # P2目标: 四指C弧到位 + thumb_abd保持255, thumb_flex仍起始
        self._p2_target = list(DISK_ANGLES)
        for i in THUMB_FLEX_INDICES:
            self._p2_target[i] = self._start_angles[i]

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

        if elapsed < PHASE1_DURATION:
            # P1: 拇指侧摆先定位
            t = elapsed / PHASE1_DURATION
            return self._move(lerp_angles(self._start_angles, self._p1_target, t))
        elif elapsed < PHASE1_DURATION + PHASE2_DURATION:
            # P2: 四指弯C弧
            t = (elapsed - PHASE1_DURATION) / PHASE2_DURATION
            return self._move(lerp_angles(self._p1_target, self._p2_target, t))
        elif elapsed < PHASE1_DURATION + PHASE2_DURATION + PHASE3_DURATION:
            # P3: 拇指合拢握住
            t = (elapsed - PHASE1_DURATION - PHASE2_DURATION) / PHASE3_DURATION
            return self._move(lerp_angles(self._p2_target, DISK_ANGLES, t))
        else:
            return self._move(list(DISK_ANGLES))

    @property
    def done(self) -> bool:
        return False
