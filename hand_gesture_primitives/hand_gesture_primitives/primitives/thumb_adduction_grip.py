"""大拇指侧向夹持原语 — 拇指侧面下压贴紧食指侧面形成侧捏。

典型场景：夹持薄片(卡片/纸张)、钥匙等扁平物体。
控制顺序: abd 外摆(130) → rot(170) → base(255)+tip(135) 下压闭合。
触觉闭环：渐进夹紧直到接触检测 → 冻结防过力。
"""

from typing import List, Optional

import numpy as np

from ..contact_detection import FingerContactTracker
from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL, parse_grasp_phase,
)

# 拇指目标控制量 (0~255，实机标定值)
THUMB_ABD_PREP = 130   # P2 外摆先行
THUMB_ROT = 170        # P3 旋转
THUMB_BASE = 255       # P3 指根弯曲 (满量程)
THUMB_TIP = 135        # P3 指尖弯曲

THUMB_TIP_PROGRESSIVE_MAX = 150
THUMB_ABD_PROGRESSIVE_MAX = 185

# 默认目标角度 (无感知数据时的 fallback)
THUMB_ADDUCTION_ANGLES = [
    THUMB_BASE,   # [0]  thumb_base
    160,        # [1]  index_base: 中度弯曲形成夹持承接面
    160,        # [2]  middle_base: 与食指同曲
    160,        # [3]  ring_base: 与食指同曲
    160,        # [4]  pinky_base: 与食指同曲
    THUMB_ABD_PREP,  # [5]  thumb_abd: 外摆 (P2)
    ABD_NEUTRAL,  # [6]  index_abd
    ABD_NEUTRAL,  # [7]  middle_abd
    ABD_NEUTRAL,  # [8]  ring_abd
    ABD_NEUTRAL,  # [9]  pinky_abd
    THUMB_ROT,    # [10] thumb_rot
    0, 0, 0, 0,  # [11-14] rsv
    THUMB_TIP,    # [15] thumb_tip
    140,        # [16] index_tip: 弯曲形成承接面
    140,        # [17] middle_tip
    140,        # [18] ring_tip
    140,        # [19] pinky_tip
]

THUMB_ALL_INDICES = [0, 5, 10, 15]
THUMB_ROT_INDEX = 10
THUMB_FLEX_INDICES = [0, 15]  # P4: base + tip 下压

PHASE1_DURATION = 0.35  # 四指弯曲收拢
PHASE2_DURATION = 0.40  # thumb_abd 外摆
PHASE3_DURATION = 0.35  # thumb_rot 旋转就位
PHASE4_DURATION = 0.65  # thumb_base + tip 下压闭合 (较慢，避免猛合)

PREP_END = PHASE1_DURATION + PHASE2_DURATION + PHASE3_DURATION

RETARGET_BLEND = 0.25
SIZE_CHANGE_THRESHOLD = 0.002

PROGRESSIVE_CLOSE_RATE = 20.0


def _adaptive_thumb_adduction_angles(object_size):
    """根据物体厚度调整食指弯曲；拇指 rot/base/abd/tip 固定。"""
    angles = list(THUMB_ADDUCTION_ANGLES)
    s = sorted(object_size)
    thickness_mm = s[0] * 1000.0

    angles[1] = int(np.clip(170 - (thickness_mm - 1) * 1.5, 120, 170))   # index_base
    angles[16] = int(np.clip(150 - (thickness_mm - 1) * 1.0, 100, 150))  # index_tip
    angles[2] = angles[3] = angles[4] = angles[1]
    angles[17] = angles[18] = angles[19] = angles[16]
    angles[0] = THUMB_BASE
    angles[5] = THUMB_ABD_PREP
    angles[10] = THUMB_ROT
    angles[15] = THUMB_TIP
    return angles


class ThumbAdductionGrip(HandGesturePrimitive):
    """拇指侧向夹持: 四指弯曲收拢，拇指先侧摆再下压闭合。

    四阶段 (拇指):
      P1: 四指弯曲 (拇指保持起始位)
      P2: thumb_abd 外摆 (130)
      P3: thumb_rot 旋转 (170)
      P4: thumb_base(255) + tip(135) 下压闭合
    触觉闭环: P4 起检测，完成后 progressive 继续 abd/tip → 接触冻结。

    phase 模式:
      prep  — 执行 P1~P3 后 hold，grasp_state=ready
      close — 从当前姿态执行 P4 + 触觉 progressive
      full  — 一次跑完全流程 (默认，向后兼容)
    """

    def __init__(self, phase: str = "full") -> None:
        self._phase = parse_grasp_phase(phase)

    @property
    def phase(self) -> str:
        return self._phase

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._target: List[float] = list(THUMB_ADDUCTION_ANGLES)
        self._last_size: Optional[np.ndarray] = None
        self._blend_from: Optional[List[float]] = None
        self._blend_start_elapsed: float = 0.0
        self._frozen_pose: Optional[List[float]] = None
        self._progressive_angles: Optional[List[float]] = None
        self._last_elapsed: float = 0.0
        self._size_locked: bool = False
        self._contact = FingerContactTracker()
        self.grasp_state: str = "approaching"

    def _update_target(self, ctx: PrimitiveContext, elapsed: float,
                       current_output: List[float]) -> None:
        if self._size_locked:
            return
        if ctx.object_size is None:
            return
        if self._last_size is not None:
            if np.linalg.norm(ctx.object_size - self._last_size) < SIZE_CHANGE_THRESHOLD:
                return
        self._last_size = ctx.object_size.copy()
        new_target = _adaptive_thumb_adduction_angles(ctx.object_size)
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        target = self._target
        t1_end = PHASE1_DURATION
        t2_end = t1_end + PHASE2_DURATION
        t3_end = t2_end + PHASE3_DURATION
        t4_end = t3_end + PHASE4_DURATION

        # P1: 四指弯曲，拇指完全保持起始
        phase1_target = list(target)
        for i in THUMB_ALL_INDICES:
            phase1_target[i] = self._start_angles[i]

        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return lerp_angles(self._start_angles, phase1_target, t)

        # P2: thumb_abd 外摆先行
        phase2_target = list(phase1_target)
        phase2_target[5] = target[5]

        if elapsed < t2_end:
            t = (elapsed - t1_end) / PHASE2_DURATION
            return lerp_angles(phase1_target, phase2_target, t)

        # P3: thumb_rot 旋转就位
        phase3_target = list(phase2_target)
        phase3_target[THUMB_ROT_INDEX] = target[THUMB_ROT_INDEX]

        if elapsed < t3_end:
            t = (elapsed - t2_end) / PHASE3_DURATION
            return lerp_angles(phase2_target, phase3_target, t)

        # P4: base + tip 下压闭合
        phase4_target = list(phase3_target)
        for i in THUMB_FLEX_INDICES:
            phase4_target[i] = target[i]

        if elapsed < t4_end:
            t = (elapsed - t3_end) / PHASE4_DURATION
            return lerp_angles(phase3_target, phase4_target, t)

        return list(target)

    def _compute_close(self, elapsed: float) -> List[float]:
        """close 阶段: 仅 P4 下压 (从 on_enter 时的关节态插值)。"""
        close_to = list(self._start_angles)
        for i in THUMB_FLEX_INDICES:
            close_to[i] = self._target[i]
        if elapsed < PHASE4_DURATION:
            t = elapsed / PHASE4_DURATION
            return lerp_angles(self._start_angles, close_to, t)
        return close_to

    def _check_contact(self, ctx: PrimitiveContext) -> bool:
        return self._contact.check(ctx, (0, 1))

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        if self._phase == "close":
            compute_fn = self._compute_close
            motion_elapsed = elapsed
            close_phase_start = 0.0
            phase4_done_at = PHASE4_DURATION
        else:
            compute_fn = self._compute_raw
            motion_elapsed = min(elapsed, PREP_END) if self._phase == "prep" else elapsed
            close_phase_start = PREP_END
            phase4_done_at = PREP_END + PHASE4_DURATION

        raw = compute_fn(motion_elapsed)
        self._update_target(ctx, elapsed, raw)
        raw = compute_fn(motion_elapsed)

        if self._blend_from is not None:
            dt = elapsed - self._blend_start_elapsed
            if dt < RETARGET_BLEND:
                t = dt / RETARGET_BLEND
                raw = lerp_angles(self._blend_from, raw, t)
            else:
                self._blend_from = None

        if self._phase == "prep":
            self.grasp_state = "ready" if elapsed >= PREP_END else "approaching"
            self._last_elapsed = elapsed
            return self._move(raw)

        if elapsed > close_phase_start:
            if not self._size_locked:
                self._size_locked = True
                self._contact.begin_closing(ctx)

            if self._frozen_pose is not None:
                self.grasp_state = "contact"
                return self._move(self._frozen_pose)

            has_contact = self._check_contact(ctx)
            if has_contact:
                freeze = self._progressive_angles if self._progressive_angles is not None else raw
                self._frozen_pose = list(freeze)
                self.grasp_state = "contact"
                return self._move(self._frozen_pose)

            if elapsed >= phase4_done_at:
                self.grasp_state = "progressive"
                dt = elapsed - self._last_elapsed
                self._last_elapsed = elapsed
                if self._progressive_angles is None:
                    self._progressive_angles = list(raw)
                increment = PROGRESSIVE_CLOSE_RATE * dt
                self._progressive_angles[5] = min(
                    THUMB_ABD_PROGRESSIVE_MAX, self._progressive_angles[5] + increment)
                self._progressive_angles[15] = min(
                    THUMB_TIP_PROGRESSIVE_MAX, self._progressive_angles[15] + increment * 0.5)
                return self._move(list(self._progressive_angles))

        self.grasp_state = "approaching"
        self._last_elapsed = elapsed
        return self._move(raw)

    @property
    def name(self) -> str:
        return "thumb_adduction_grip"

    @property
    def done(self) -> bool:
        return False
