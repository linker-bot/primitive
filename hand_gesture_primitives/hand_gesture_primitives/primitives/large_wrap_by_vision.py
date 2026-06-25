"""大物体包络原语 — 五指适度弯曲包络较大直径物体。

支持根据感知物体尺寸自适应调整包络弯曲量，目标可随感知更新平滑过渡。
触觉闭环：渐进闭合直到接触检测 → 冻结姿态保持，等待下一个指令。
"""

from typing import List, Optional

import numpy as np

from ..contact_detection import FingerContactTracker
from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL,
)

# 默认目标角度 (无感知数据时的 fallback)
LARGE_WRAP_ANGLES = [
    200,        # [0]  thumb_base: 弯曲对掌
    220,        # [1]  index_base: 深度弯曲
    220,        # [2]  middle_base: 深度弯曲
    220,        # [3]  ring_base: 深度弯曲
    220,        # [4]  pinky_base: 深度弯曲
    175,        # [5]  thumb_abd: 外展对掌包络
    ABD_NEUTRAL,  # [6]  index_abd: 中立
    ABD_NEUTRAL,  # [7]  middle_abd: 中立
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    210,        # [10] thumb_rot: 旋转对向四指
    0, 0, 0, 0,  # [11-14] rsv
    200,        # [15] thumb_tip: 弯曲包络
    210,        # [16] index_tip: 弯曲包络
    210,        # [17] middle_tip: 弯曲包络
    210,        # [18] ring_tip: 弯曲包络
    210,        # [19] pinky_tip: 弯曲包络
]

THUMB_ALL_INDICES = [0, 5, 10, 15]

PHASE1_DURATION = 0.5   # 四指先包络
PHASE2_DURATION = 0.35  # 拇指侧摆
PHASE3_DURATION = 0.35  # 拇指旋转
PHASE4_DURATION = 0.35  # 拇指弯曲闭合

# 目标切换时平滑过渡时长
RETARGET_BLEND = 0.25

# 尺寸变化阈值 (m)
SIZE_CHANGE_THRESHOLD = 0.003

# 触觉闭环参数
# 渐进闭合参数 (到达目标后无接触时继续缓慢收紧)
PROGRESSIVE_CLOSE_RATE = 25.0       # 每秒增加的闭合量 (约2.5/tick @10Hz)
PROGRESSIVE_CLOSE_MAX = 250.0       # 闭合安全上限


def _adaptive_large_wrap_by_vision_angles(object_size):
    """根据物体最大截面直径调整包络弯曲量。

    object_size: [sx, sy, sz] meters
    返回适配后的 20-DOF 目标角度列表。
    """
    angles = list(LARGE_WRAP_ANGLES)
    # 取 XY 平面最大截面直径 (排除高度轴)
    s = sorted(object_size)
    D_mm = float(max(s[0], s[1])) * 1000.0

    # D=40mm → base≈230(紧握); D=100mm → base≈150(半开)
    base_val = int(np.clip(250 - (D_mm - 30) * 1.4, 130, 240))
    tip_val = int(np.clip(240 - (D_mm - 30) * 1.2, 120, 230))
    thumb_base = int(np.clip(220 - (D_mm - 30) * 0.8, 100, 220))
    thumb_tip = int(np.clip(tip_val - 10, 100, 220))

    angles[1] = angles[2] = angles[3] = angles[4] = base_val   # 四指 base
    angles[16] = angles[17] = angles[18] = angles[19] = tip_val  # 四指 tip
    angles[0] = thumb_base
    angles[15] = thumb_tip
    return angles


class LargeWrapByVision(HandGesturePrimitive):
    """五指抓握包络大直径物体（分阶段避免拇指碰撞）。

    根据 ctx.object_size 自适应调整包络弯曲量；无感知数据时使用默认值。
    感知数据变化时平滑过渡到新目标，避免关节跳变。
    """

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._target: List[float] = list(LARGE_WRAP_ANGLES)
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
        new_target = _adaptive_large_wrap_by_vision_angles(ctx.object_size)
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        """按阶段逻辑计算原始输出 (不含 blend)。"""
        target = self._target

        # 阶段1: 四指到位, 拇指保持起始
        phase1_target = list(target)
        for i in THUMB_ALL_INDICES:
            phase1_target[i] = self._start_angles[i]
        # 阶段2: 拇指侧摆到位
        phase2_target = list(phase1_target)
        phase2_target[5] = target[5]
        # 阶段3: 拇指旋转到位
        phase3_target = list(phase2_target)
        phase3_target[10] = target[10]

        t1_end = PHASE1_DURATION
        t2_end = t1_end + PHASE2_DURATION
        t3_end = t2_end + PHASE3_DURATION
        t4_end = t3_end + PHASE4_DURATION

        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return lerp_angles(self._start_angles, phase1_target, t)
        if elapsed < t2_end:
            t = (elapsed - t1_end) / PHASE2_DURATION
            return lerp_angles(phase1_target, phase2_target, t)
        if elapsed < t3_end:
            t = (elapsed - t2_end) / PHASE3_DURATION
            return lerp_angles(phase2_target, phase3_target, t)
        if elapsed < t4_end:
            t = (elapsed - t3_end) / PHASE4_DURATION
            return lerp_angles(phase3_target, target, t)
        return list(target)

    def _check_contact(self, ctx: PrimitiveContext, closing_fingers: List[int]) -> bool:
        """指定手指接触：触觉优先，无传感器时用电流增量。"""
        return self._contact.check(ctx, closing_fingers)

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

        # 触觉闭环: 仅在闭合阶段 (过了 phase1 开始后) 生效
        if elapsed > PHASE1_DURATION * 0.5:
            if not self._size_locked and elapsed > PHASE1_DURATION:
                self._size_locked = True
                self._contact.begin_closing(ctx)

            # 已冻结 → 永远保持，等下一个指令
            if self._frozen_pose is not None:
                self.grasp_state = "contact"
                return self._move(self._frozen_pose)

            # phase1: 四指闭合 → 检测 index/middle/ring/pinky
            # phase2+: 拇指加入 → 检测全部
            if elapsed < PHASE1_DURATION + PHASE2_DURATION:
                closing = [1, 2, 3, 4]
            else:
                closing = [0, 1, 2, 3, 4]

            has_contact = self._check_contact(ctx, closing)
            if has_contact:
                # 接触瞬间: 冻结实际输出姿态 (渐进模式下用渐进值，否则用 raw)
                freeze = self._progressive_angles if self._progressive_angles is not None else raw
                self._frozen_pose = list(freeze)
                self.grasp_state = "contact"
                return self._move(self._frozen_pose)

            # 无接触 + 全部阶段完成 → 渐进闭合
            all_phases_done = elapsed >= (PHASE1_DURATION + PHASE2_DURATION
                                          + PHASE3_DURATION + PHASE4_DURATION)
            if all_phases_done:
                self.grasp_state = "progressive"
                dt = elapsed - self._last_elapsed
                self._last_elapsed = elapsed
                if self._progressive_angles is None:
                    self._progressive_angles = list(raw)
                increment = PROGRESSIVE_CLOSE_RATE * dt
                for i in closing:
                    base_idx = i
                    tip_idx = i + 15
                    self._progressive_angles[base_idx] = min(
                        PROGRESSIVE_CLOSE_MAX,
                        self._progressive_angles[base_idx] + increment
                    )
                    self._progressive_angles[tip_idx] = min(
                        PROGRESSIVE_CLOSE_MAX,
                        self._progressive_angles[tip_idx] + increment
                    )
                return self._move(list(self._progressive_angles))

        self.grasp_state = "approaching"
        self._last_elapsed = elapsed
        return self._move(raw)

    @property
    def name(self) -> str:
        return "large_wrap_by_vision"

    @property
    def done(self) -> bool:
        return False
