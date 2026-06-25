"""食中指侧向夹持原语 — 食指与中指并拢侧面夹持物体。

典型场景：夹持香烟、笔杆、细棒等细长物体 (食中指像筷子一样夹住)。
无名/小指握拢 → P2 拇指 abd→rot→base/tip → 食中指侧摆张开 → 并拢夹取。
触觉闭环：渐进夹紧直到接触检测 → 冻结防过力。
"""

from typing import List, Optional, Tuple

import numpy as np

from ..contact_detection import FingerContactTracker
from ..primitive_base import (
    HandGesturePrimitive, PrimitiveContext, PrimitiveResult,
    lerp_angles, ABD_NEUTRAL, parse_grasp_phase,
)

# 准备姿态: 无名/小指握拢，拇指让位到尺侧
RING_PINKY_CLOSE = 255
THUMB_BASE_PREP = 178
THUMB_ABD_PREP = 255
THUMB_ROT_PREP = 255
THUMB_TIP_PREP = 255

THUMB_ALL_INDICES = [0, 5, 10, 15]
THUMB_ABD_INDEX = 5
THUMB_ROT_INDEX = 10
THUMB_FLEX_INDICES = [0, 15]  # P2 末步: base + tip 闭合

# 食中指侧摆：v_sign 方向为张开 (先 prep)，并拢为闭合目标
INDEX_ABD_SPREAD = 255
MIDDLE_ABD_SPREAD = 0

# 默认目标角度 (无感知数据时的 fallback) — 最终并拢夹持姿态
INDEX_MIDDLE_ADDUCTION_ANGLES = [
    THUMB_BASE_PREP,  # [0]  thumb_base: P2-3 闭合
    120,        # [1]  index_base: 中度弯曲提供夹持角度
    120,        # [2]  middle_base: 同食指对称
    RING_PINKY_CLOSE,  # [3]  ring_base: 握拢
    RING_PINKY_CLOSE,  # [4]  pinky_base: 握拢
    THUMB_ABD_PREP,  # [5]  thumb_abd: P2-1 内扣
    96,         # [6]  index_abd: 并拢 (<128, 与 spread 255 相反)
    160,        # [7]  middle_abd: 并拢 (>128, 与 spread 0 相反)
    ABD_NEUTRAL,  # [8]  ring_abd: 中立
    ABD_NEUTRAL,  # [9]  pinky_abd: 中立
    THUMB_ROT_PREP,  # [10] thumb_rot: P2-2 旋转
    0, 0, 0, 0,  # [11-14] rsv
    THUMB_TIP_PREP,  # [15] thumb_tip: P2-3 闭合
    100,        # [16] index_tip: 中度弯曲形成夹持面
    100,        # [17] middle_tip: 同食指
    RING_PINKY_CLOSE,  # [18] ring_tip: 握拢
    RING_PINKY_CLOSE,  # [19] pinky_tip: 握拢
]

PHASE1_DURATION = 0.35  # 无名/小指握拢 + 食中指弯曲
PHASE2_ABD_DURATION = 0.20  # P2-1: abd 内扣
PHASE2_ROT_DURATION = 0.20   # P2-2: rot 旋转
PHASE2_FLEX_DURATION = 0.20  # P2-3: base + tip 闭合
PHASE2_DURATION = PHASE2_ABD_DURATION + PHASE2_ROT_DURATION + PHASE2_FLEX_DURATION
PHASE3_DURATION = 0.35  # 食中指侧摆张开 (prep)
PHASE4_DURATION = 0.40  # 食中指侧摆并拢夹取

PREP_END = PHASE1_DURATION + PHASE2_DURATION + PHASE3_DURATION

RETARGET_BLEND = 0.25
SIZE_CHANGE_THRESHOLD = 0.002

PROGRESSIVE_CLOSE_RATE = 15.0
PROGRESSIVE_CLOSE_MAX = 200.0


def _close_abd_from_diameter(d_mm: float) -> Tuple[int, int]:
    """根据物体直径计算并拢时的食中指 abd 目标。"""
    abd_offset = int(np.clip(40 - (d_mm - 3) * 2.5, 10, 40))
    return ABD_NEUTRAL - abd_offset, ABD_NEUTRAL + abd_offset


def _spread_abd_from_diameter(d_mm: float) -> Tuple[int, int]:
    """根据物体直径计算 prep 张开幅度 (粗物体需更大开口)。"""
    spread = int(np.clip(90 + (d_mm - 3) * 4.0, 90, 127))
    return min(255, ABD_NEUTRAL + spread), max(0, ABD_NEUTRAL - spread)


def _adaptive_index_middle_adduction_angles(object_size):
    """根据物体直径调整食中指侧向夹持开合量。

    侧捏主要关注物体最薄维度 (直径)。
    直径越大 → 并拢 abd 偏移越小、prep 张开越大。
    """
    angles = list(INDEX_MIDDLE_ADDUCTION_ANGLES)
    s = sorted(object_size)
    d_mm = s[0] * 1000.0  # 最薄维度 mm

    angles[6], angles[7] = _close_abd_from_diameter(d_mm)

    # 弯曲量: 细物体需要更多弯曲来包住
    base_val = int(np.clip(140 - (d_mm - 3) * 3.0, 80, 140))
    tip_val = int(np.clip(120 - (d_mm - 3) * 2.5, 60, 120))
    angles[1] = angles[2] = base_val
    angles[16] = angles[17] = tip_val
    angles[0] = THUMB_BASE_PREP
    angles[3] = angles[4] = RING_PINKY_CLOSE
    angles[5] = THUMB_ABD_PREP
    angles[10] = THUMB_ROT_PREP
    angles[15] = THUMB_TIP_PREP
    angles[18] = angles[19] = RING_PINKY_CLOSE
    return angles


class IndexMiddleAdductionGrip(HandGesturePrimitive):
    """食指与中指侧向并拢夹持。

    四阶段:
      P1: 无名/小指握拢 + 食中指弯曲 (拇指保持起始位)
      P2: 拇指准备 — abd 内扣 → rot 旋转 → base/tip 闭合
      P3: 食中指侧摆张开 (prep)
      P4: 食中指侧摆并拢夹取
    触觉闭环: P4 完成后 progressive 继续并拢 → 接触冻结。

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
        self._target: List[float] = list(INDEX_MIDDLE_ADDUCTION_ANGLES)
        self._spread_abd: Tuple[int, int] = (INDEX_ABD_SPREAD, MIDDLE_ABD_SPREAD)
        self._last_size: Optional[np.ndarray] = None
        self._blend_from: Optional[List[float]] = None
        self._blend_start_elapsed: float = 0.0
        self._frozen_pose: Optional[List[float]] = None
        self._progressive_angles: Optional[List[float]] = None
        self._last_elapsed: float = 0.0
        self._size_locked: bool = False
        self._contact = FingerContactTracker()
        self.grasp_state: str = "approaching"

    def _update_spread_abd(self, object_size: Optional[np.ndarray]) -> None:
        if object_size is None:
            self._spread_abd = (INDEX_ABD_SPREAD, MIDDLE_ABD_SPREAD)
            return
        d_mm = sorted(object_size)[0] * 1000.0
        self._spread_abd = _spread_abd_from_diameter(d_mm)

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
        new_target = _adaptive_index_middle_adduction_angles(ctx.object_size)
        self._update_spread_abd(ctx.object_size)
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        target = self._target
        t1_end = PHASE1_DURATION
        t2_abd_end = t1_end + PHASE2_ABD_DURATION
        t2_rot_end = t2_abd_end + PHASE2_ROT_DURATION
        t2_end = t2_rot_end + PHASE2_FLEX_DURATION
        t3_end = t2_end + PHASE3_DURATION
        t4_end = t3_end + PHASE4_DURATION

        # P1: 无名/小指 + 食中指弯曲；拇指与食中指 abd 保持起始
        phase1_target = list(target)
        for i in THUMB_ALL_INDICES:
            phase1_target[i] = self._start_angles[i]
        phase1_target[6] = self._start_angles[6]
        phase1_target[7] = self._start_angles[7]

        if elapsed < t1_end:
            t = elapsed / PHASE1_DURATION
            return lerp_angles(self._start_angles, phase1_target, t)

        # P2-1: thumb_abd 内扣
        phase2_abd_target = list(phase1_target)
        phase2_abd_target[THUMB_ABD_INDEX] = target[THUMB_ABD_INDEX]

        if elapsed < t2_abd_end:
            t = (elapsed - t1_end) / PHASE2_ABD_DURATION
            return lerp_angles(phase1_target, phase2_abd_target, t)

        # P2-2: thumb_rot 旋转
        phase2_rot_target = list(phase2_abd_target)
        phase2_rot_target[THUMB_ROT_INDEX] = target[THUMB_ROT_INDEX]

        if elapsed < t2_rot_end:
            t = (elapsed - t2_abd_end) / PHASE2_ROT_DURATION
            return lerp_angles(phase2_abd_target, phase2_rot_target, t)

        # P2-3: thumb_base + tip 闭合
        phase2_flex_target = list(phase2_rot_target)
        for i in THUMB_FLEX_INDICES:
            phase2_flex_target[i] = target[i]

        if elapsed < t2_end:
            t = (elapsed - t2_rot_end) / PHASE2_FLEX_DURATION
            return lerp_angles(phase2_rot_target, phase2_flex_target, t)

        # P3: 食中指 abd 侧摆张开 (prep)
        phase3_target = list(phase2_flex_target)
        phase3_target[6] = self._spread_abd[0]
        phase3_target[7] = self._spread_abd[1]

        if elapsed < t3_end:
            t = (elapsed - t2_end) / PHASE3_DURATION
            return lerp_angles(phase2_flex_target, phase3_target, t)

        # P4: 食中指 abd 并拢到夹持目标
        if elapsed < t4_end:
            t = (elapsed - t3_end) / PHASE4_DURATION
            return lerp_angles(phase3_target, target, t)

        return list(target)

    def _compute_close(self, elapsed: float) -> List[float]:
        """close 阶段: 仅 P4 食中指 abd 并拢 (从 on_enter 时的关节态插值)。"""
        close_to = list(self._start_angles)
        close_to[6] = self._target[6]
        close_to[7] = self._target[7]
        if elapsed < PHASE4_DURATION:
            t = elapsed / PHASE4_DURATION
            return lerp_angles(self._start_angles, close_to, t)
        return close_to

    def _check_contact(self, ctx: PrimitiveContext) -> bool:
        return self._contact.check(ctx, (1, 2))

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        self._update_spread_abd(ctx.object_size)

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
                self._progressive_angles[6] = max(
                    0, self._progressive_angles[6] - increment)
                self._progressive_angles[7] = min(
                    PROGRESSIVE_CLOSE_MAX, self._progressive_angles[7] + increment)
                return self._move(list(self._progressive_angles))

        self.grasp_state = "approaching"
        self._last_elapsed = elapsed
        return self._move(raw)

    @property
    def name(self) -> str:
        return "index_middle_adduction_grip"

    @property
    def done(self) -> bool:
        return False
