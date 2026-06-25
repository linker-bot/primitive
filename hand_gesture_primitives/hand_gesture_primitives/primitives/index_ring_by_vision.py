"""食指环形包络原语 — 拇指与食指形成环状包络，其余三指张开保持。

支持根据感知物体尺寸自适应调整捏合口径，目标可随感知更新平滑过渡。
触觉闭环：到达目标后继续缓慢闭合直到接触检测；接触后冻结防止过力。
"""

from typing import List, Optional

import numpy as np

from ..contact_detection import FingerContactTracker
from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 默认目标角度 (无感知数据时的 fallback)
INDEX_RING_ANGLES = [
    110,        # [0]  thumb_base: 适度弯曲参与环
    130,        # [1]  index_base: 适度弯曲形成环弧
    0,          # [2]  middle_base: 伸直张开
    0,          # [3]  ring_base: 伸直张开
    0,          # [4]  pinky_base: 伸直张开
    165,        # [5]  thumb_abd: 内收对准食指
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    210,        # [10] thumb_rot: 旋转对准食指
    0, 0, 0, 0,  # [11-14] rsv
    175,        # [15] thumb_tip: 指尖弯曲闭合环
    110,        # [16] index_tip: 指尖轻微弯曲闭合环
    0,          # [17] middle_tip: 伸直
    0,          # [18] ring_tip: 伸直
    0,          # [19] pinky_tip: 伸直
]

# 先张开三指，再拇食合环
PHASE1_DURATION = 0.4   # 三指先张开
PHASE2_DURATION = 0.5   # 拇指食指合环到达初始目标

# 目标切换时平滑过渡时长
RETARGET_BLEND = 0.25

# 尺寸变化阈值 (m)，低于此认为无显著变化
SIZE_CHANGE_THRESHOLD = 0.002

# 触觉闭环参数 (无触觉时 fallback 到 joint current)

# 渐进闭合参数 (到达目标后无接触时继续缓慢收紧)
PROGRESSIVE_CLOSE_RATE = 30.0       # 每秒增加的闭合量 (约3/tick @10Hz)
PROGRESSIVE_CLOSE_MAX = 250.0       # 闭合安全上限


def _adaptive_index_ring_by_vision_angles(object_size):
    """根据物体截面直径调整捏合闭合量。

    object_size: [sx, sy, sz] meters
    返回适配后的 20-DOF 目标角度列表。

    物体越大 → 拇食指需要更大弯曲才能环绕包住。
    范围: d=5mm (小螺丝，轻捏) ~ d=30mm (粗柄，深握)
    """
    angles = list(INDEX_RING_ANGLES)
    s = sorted(object_size)
    d_mm = (s[0] + s[1]) / 2.0 * 1000.0  # 截面直径 mm

    # d=5mm → 轻捏(小弯曲); d=30mm → 深握(大弯曲)
    angles[0] = int(np.clip(90 + (d_mm - 5) * 2.4, 90, 150))     # thumb_base
    angles[1] = int(np.clip(100 + (d_mm - 5) * 2.8, 100, 170))   # index_base
    angles[15] = int(np.clip(140 + (d_mm - 5) * 2.0, 140, 200))  # thumb_tip
    angles[16] = int(np.clip(80 + (d_mm - 5) * 2.0, 80, 150))    # index_tip
    return angles


class IndexRingByVision(HandGesturePrimitive):
    """拇指与食指环形包络，其余手指张开不动。

    三阶段策略:
      P1: 三指张开
      P2: 拇食指闭合到视觉目标
      P3: 若无触觉接触，继续渐进闭合直到接触或安全上限
    触觉接触后冻结，滑移时缓慢恢复。
    """

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._target: List[float] = list(INDEX_RING_ANGLES)
        self._last_size: Optional[np.ndarray] = None
        # 平滑过渡状态
        self._blend_from: Optional[List[float]] = None
        self._blend_start_elapsed: float = 0.0
        # 触觉闭环状态
        self._frozen_pose: Optional[List[float]] = None
        # 渐进闭合状态
        self._progressive_angles: Optional[List[float]] = None
        self._last_elapsed: float = 0.0
        self._size_locked: bool = False
        self._contact = FingerContactTracker()
        # 状态追踪 (供 executor 日志使用)
        self.grasp_state: str = "approaching"

    def _update_target(self, ctx: PrimitiveContext, elapsed: float,
                       current_output: List[float]) -> None:
        """每帧评估感知数据，必要时更新目标并启动平滑过渡。"""
        if self._size_locked:
            return
        if ctx.object_size is None:
            return
        if self._last_size is not None:
            if np.linalg.norm(ctx.object_size - self._last_size) < SIZE_CHANGE_THRESHOLD:
                return
        self._last_size = ctx.object_size.copy()
        new_target = _adaptive_index_ring_by_vision_angles(ctx.object_size)
        # 启动 blend: 从当前输出位置平滑过渡到新目标
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        """按阶段逻辑计算原始输出 (不含 blend)。"""
        target = self._target

        # 阶段1目标: 三指张开，拇食指保持原位
        phase1_target = list(target)
        for i in [0, 1, 5, 10, 15, 16]:
            phase1_target[i] = self._start_angles[i]

        t1_end = PHASE1_DURATION
        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return lerp_angles(self._start_angles, phase1_target, t)
        t = min(1.0, (elapsed - t1_end) / PHASE2_DURATION)
        return lerp_angles(phase1_target, target, t)

    def _check_contact(self, ctx: PrimitiveContext) -> bool:
        """拇指或食指接触：触觉优先，无传感器时用电流增量。"""
        return self._contact.check(ctx, (0, 1))

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        raw = self._compute_raw(elapsed)
        self._update_target(ctx, elapsed, raw)
        raw = self._compute_raw(elapsed)

        # 平滑过渡
        if self._blend_from is not None:
            dt = elapsed - self._blend_start_elapsed
            if dt < RETARGET_BLEND:
                t = dt / RETARGET_BLEND
                raw = lerp_angles(self._blend_from, raw, t)
            else:
                self._blend_from = None

        # 触觉闭环: 仅在 phase2+ (拇食指闭合阶段) 生效
        if elapsed > PHASE1_DURATION:
            if not self._size_locked and elapsed >= (PHASE1_DURATION + PHASE2_DURATION):
                self._size_locked = True
                self._contact.begin_closing(ctx)

            # 已冻结 → 永远保持，等下一个指令
            if self._frozen_pose is not None:
                self.grasp_state = "contact"
                return self._move(self._frozen_pose)

            has_contact = self._check_contact(ctx)
            if has_contact:
                # 接触瞬间: 冻结实际输出姿态 (渐进模式下用渐进值，否则用 raw)
                freeze = self._progressive_angles if self._progressive_angles is not None else raw
                self._frozen_pose = list(freeze)
                self.grasp_state = "contact"
                return self._move(self._frozen_pose)

            # 无接触 + phase2 已到达目标 → 渐进闭合
            phase2_done = elapsed >= (PHASE1_DURATION + PHASE2_DURATION)
            if phase2_done:
                self.grasp_state = "progressive"
                dt = elapsed - self._last_elapsed
                self._last_elapsed = elapsed
                if self._progressive_angles is None:
                    self._progressive_angles = list(raw)
                increment = PROGRESSIVE_CLOSE_RATE * dt
                # 拇食指各关节缓慢递增
                for i in [0, 1, 15, 16]:
                    self._progressive_angles[i] = min(
                        PROGRESSIVE_CLOSE_MAX,
                        self._progressive_angles[i] + increment
                    )
                return self._move(list(self._progressive_angles))

        self.grasp_state = "approaching"
        self._last_elapsed = elapsed
        return self._move(raw)

    @property
    def name(self) -> str:
        return "index_ring_by_vision"

    @property
    def done(self) -> bool:
        return False
