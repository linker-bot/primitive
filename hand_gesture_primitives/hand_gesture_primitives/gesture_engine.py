"""分阶段 lerp 运动引擎 — 供 thumb_adduction_grip 等 phased 原语复用。"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .contact_detection import (
    FingerContactTracker,
    MotionFeedbackGuard,
    capture_feedback_baseline,
    force_close_group_contact,
    hold_feedback_exceeded,
    monitors_for_fingers,
    tactile_contact,
    uses_torque_feedback,
)
from .gesture_params import (
    SequentialForceCloseParams,
    StaticGestureParams,
    ThumbAdductionParams,
    load_ring_params,
    load_middle_ring_params,
    load_sequential_force_close_params,
    load_static_gesture_params,
    load_thumb_adduction_params,
)
from .primitive_base import lerp_angles, parse_grasp_phase

RETARGET_BLEND = 0.25
SIZE_CHANGE_THRESHOLD = 0.002


def adaptive_thumb_adduction_angles(
    object_size, params: ThumbAdductionParams,
) -> List[float]:
    """根据物体厚度调整食指弯曲；其余关节取自配置。"""
    angles = list(params.prep_angles)
    s = sorted(object_size)
    thickness_mm = s[0] * 1000.0

    angles[1] = int(np.clip(170 - (thickness_mm - 1) * 1.5, 120, 170))
    if params.thumb_rot_joint >= 0:
        angles[params.thumb_rot_joint] = params.prep_angles[params.thumb_rot_joint]
    for i in params.thumb_flex_joints:
        if i >= 0:
            angles[i] = params.close_angles[i]
    angles[5] = params.prep_angles[5]
    if not params.close_move_joints:
        angles[2] = angles[3] = angles[4] = angles[1]
        for tip_i in (16, 17, 18, 19):
            angles[tip_i] = int(np.clip(150 - (thickness_mm - 1) * 1.0, 100, 150))
        angles[17] = angles[18] = angles[19] = angles[16]
    return angles


class PhasedLerpEngine:
    """P1~P4 分阶段插值 + 接触/位置停指 + 渐进加力。"""

    def __init__(self, params: ThumbAdductionParams, phase: str = "full") -> None:
        self._params = params
        self._phase = parse_grasp_phase(phase)
        self._start_angles: List[float] = [0.0] * 20
        self._target: List[float] = list(params.prep_angles)
        self._last_size: Optional[np.ndarray] = None
        self._blend_from: Optional[List[float]] = None
        self._blend_start_elapsed: float = 0.0
        self._frozen_pose: Optional[List[float]] = None
        self._progressive_angles: Optional[List[float]] = None
        self._last_elapsed: float = 0.0
        self._size_locked: bool = False
        self._contact = FingerContactTracker()
        self._guard = MotionFeedbackGuard()
        self._last_output: Optional[List[float]] = None
        self.grasp_state: str = "approaching"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def params(self) -> ThumbAdductionParams:
        return self._params

    def reset(self, start_angles: List[float]) -> None:
        self._start_angles = list(start_angles)
        self._target = list(self._params.prep_angles)
        self._last_size = None
        self._blend_from = None
        self._blend_start_elapsed = 0.0
        self._frozen_pose = None
        self._progressive_angles = None
        self._last_elapsed = 0.0
        self._size_locked = False
        self._contact.reset()
        self._guard.reset()
        self._last_output = None
        self.grasp_state = "approaching"

    def set_params(self, params: ThumbAdductionParams) -> None:
        """手型切换时更新配置并重置目标。"""
        self._params = params
        self._target = list(params.prep_angles)

    def _phase3_duration(self) -> float:
        p = self._params
        if p.thumb_rot_joint < 0 or p.phase3 <= 0:
            return 0.0
        return p.phase3

    def _update_target(self, ctx, elapsed: float, current_output: List[float]) -> None:
        if self._size_locked:
            return
        if ctx.object_size is None:
            return
        if self._last_size is not None:
            if np.linalg.norm(ctx.object_size - self._last_size) < SIZE_CHANGE_THRESHOLD:
                return
        self._last_size = ctx.object_size.copy()
        fn = self._params.adaptive_fn
        if fn is None:
            return
        new_target = fn(ctx.object_size, self._params)
        self._blend_from = list(current_output)
        self._blend_start_elapsed = elapsed
        self._target = new_target

    def _compute_raw(self, elapsed: float) -> List[float]:
        p = self._params
        target = self._target
        close = self._target if p.close_from_adaptive else p.close_angles
        thumb_hold = p.thumb_hold_joints
        flex_indices = p.close_flex_indices()
        rot_idx = p.thumb_rot_joint
        p3_dur = self._phase3_duration()

        t1_end = p.phase1
        t2_end = t1_end + p.phase2
        t3_end = t2_end + p3_dur
        t4_end = t3_end + p.phase4

        phase1_target = list(target)
        for i in thumb_hold:
            phase1_target[i] = self._start_angles[i]

        if elapsed < t1_end:
            return lerp_angles(self._start_angles, phase1_target, elapsed / p.phase1)

        phase2_target = list(phase1_target)
        phase2_target[5] = target[5]

        if elapsed < t2_end:
            t = (elapsed - t1_end) / p.phase2
            return lerp_angles(phase1_target, phase2_target, t)

        phase3_target = list(phase2_target)
        if p3_dur > 0 and rot_idx >= 0:
            phase3_target[rot_idx] = target[rot_idx]

        if elapsed < t3_end:
            if p3_dur > 0 and rot_idx >= 0:
                t = (elapsed - t2_end) / p3_dur
                return lerp_angles(phase2_target, phase3_target, t)
            phase3_target = list(phase2_target)

        close_thumb = p.close_thumb_joints
        if close_thumb:
            finger_flex = [i for i in flex_indices if i not in close_thumb]
            thumb_flex = [i for i in flex_indices if i in close_thumb]
            p4a_dur = p.phase4 * p.phase4_split
            p4b_dur = p.phase4 - p4a_dur
            t4a_end = t3_end + p4a_dur
            t4b_end = t4a_end + p4b_dur

            # P4a: 非拇指 flex 闭合
            p4a_target = list(phase3_target)
            for i in finger_flex:
                p4a_target[i] = close[i]
            if elapsed < t4a_end:
                return lerp_angles(phase3_target, p4a_target,
                                   (elapsed - t3_end) / p4a_dur if p4a_dur > 0 else 1.0)

            # P4b: 拇指 flex 闭合
            p4b_target = list(p4a_target)
            for i in thumb_flex:
                p4b_target[i] = close[i]
            if elapsed < t4b_end:
                return lerp_angles(p4a_target, p4b_target,
                                   (elapsed - t4a_end) / p4b_dur if p4b_dur > 0 else 1.0)

            out = list(target)
            for i in flex_indices:
                out[i] = close[i]
            return out
        else:
            phase4_target = list(phase3_target)
            for i in flex_indices:
                phase4_target[i] = close[i]

            if elapsed < t4_end:
                t = (elapsed - t3_end) / p.phase4
                return lerp_angles(phase3_target, phase4_target, t)

            out = list(target)
            for i in flex_indices:
                out[i] = close[i]
            return out

    def _compute_close(self, elapsed: float) -> List[float]:
        p = self._params
        flex_indices = p.close_flex_indices()
        close_thumb = p.close_thumb_joints
        if close_thumb:
            finger_flex = [i for i in flex_indices if i not in close_thumb]
            thumb_flex = [i for i in flex_indices if i in close_thumb]
            p4a_dur = p.phase4 * p.phase4_split
            p4b_dur = p.phase4 - p4a_dur
            t4a_end = p4a_dur
            t4b_end = t4a_end + p4b_dur

            # P4a: 非拇指 flex 闭合
            p4a_target = list(self._start_angles)
            for i in finger_flex:
                p4a_target[i] = p.close_angles[i]
            if elapsed < t4a_end:
                return lerp_angles(self._start_angles, p4a_target,
                                   elapsed / p4a_dur if p4a_dur > 0 else 1.0)

            # P4b: 拇指 flex 闭合
            p4b_target = list(p4a_target)
            for i in thumb_flex:
                p4b_target[i] = p.close_angles[i]
            if elapsed < t4b_end:
                return lerp_angles(p4a_target, p4b_target,
                                   (elapsed - t4a_end) / p4b_dur if p4b_dur > 0 else 1.0)
            return p4b_target
        else:
            close_to = list(self._start_angles)
            for i in flex_indices:
                close_to[i] = p.close_angles[i]
            if elapsed < p.phase4:
                return lerp_angles(self._start_angles, close_to, elapsed / p.phase4)
            return close_to

    def _position_reached(self, feedback_angles: List[float]) -> bool:
        p = self._params
        if p.position_stop_joint is None or p.position_stop_target_semantic is None:
            return False
        idx = p.position_stop_joint
        if idx < 0 or idx >= len(feedback_angles):
            return False
        return abs(
            feedback_angles[idx] - p.position_stop_target_semantic
        ) <= p.position_stop_tolerance

    def _check_contact(self, ctx, feedback_angles: List[float]) -> bool:
        """压感 → 力矩/电流增量 → 位置到达。O6/O20 统一走 FingerContactTracker。"""
        if ctx.tactile_mode != "none":
            if tactile_contact(ctx, tuple(self._params.contact_fingers)):
                return True
        if self._contact.check(ctx, tuple(self._params.contact_fingers)):
            return True
        return self._position_reached(feedback_angles)

    def _init_motion_guard(self, ctx) -> None:
        if uses_torque_feedback(ctx.hand_type):
            # O6 motor_torque 空载 60~80%，guard Δ 会在 prep 误冻
            self._guard.reset()
            return
        hw, tactile = monitors_for_fingers(ctx, self._params.contact_fingers)
        self._guard.reset(monitor_hw=hw, tactile_fingers=tactile)

    def _apply_guard(
        self, ctx, proposed: List[float], *, active: bool = True,
    ) -> List[float]:
        if uses_torque_feedback(ctx.hand_type):
            self._last_output = list(proposed)
            return list(proposed)
        self._guard.set_enabled(active)
        prev = self._last_output if self._last_output is not None else proposed
        out = self._guard.guard(ctx, proposed, prev)
        self._last_output = list(out)
        if self._guard.frozen and self.grasp_state not in ("contact", "progressive"):
            self.grasp_state = "contact"
        return out

    def compute(
        self,
        feedback_angles: List[float],
        elapsed: float,
        ctx,
    ) -> List[float]:
        """返回 semantic 20-DOF 目标角；更新 self.grasp_state。"""
        if self._last_output is None:
            self._init_motion_guard(ctx)

        p = self._params
        prep_end = p.prep_end

        if self._phase == "close":
            compute_fn = self._compute_close
            motion_elapsed = elapsed
            close_phase_start = 0.0
            phase4_done_at = p.phase4
        else:
            compute_fn = self._compute_raw
            motion_elapsed = min(elapsed, prep_end) if self._phase == "prep" else elapsed
            close_phase_start = prep_end
            phase4_done_at = prep_end + p.phase4

        raw = compute_fn(motion_elapsed)
        self._update_target(ctx, elapsed, raw)
        raw = compute_fn(motion_elapsed)

        if self._blend_from is not None:
            dt = elapsed - self._blend_start_elapsed
            if dt < RETARGET_BLEND:
                raw = lerp_angles(self._blend_from, raw, dt / RETARGET_BLEND)
            else:
                self._blend_from = None

        if self._phase == "prep":
            self.grasp_state = "ready" if elapsed >= prep_end else "approaching"
            self._last_elapsed = elapsed
            return self._apply_guard(ctx, raw)

        if elapsed > close_phase_start:
            if not self._size_locked:
                self._size_locked = True
                self._contact.begin_closing(ctx)

            if self._frozen_pose is not None:
                self.grasp_state = "contact"
                return self._apply_guard(ctx, list(self._frozen_pose))

            if self._check_contact(ctx, feedback_angles):
                freeze = (
                    self._progressive_angles
                    if self._progressive_angles is not None else raw
                )
                self._frozen_pose = list(freeze)
                self.grasp_state = "contact"
                return self._apply_guard(ctx, list(self._frozen_pose))

            if elapsed >= phase4_done_at:
                self.grasp_state = "progressive"
                dt = elapsed - self._last_elapsed
                self._last_elapsed = elapsed
                if self._progressive_angles is None:
                    self._progressive_angles = list(raw)
                if self._guard.frozen:
                    return self._apply_guard(ctx, list(self._progressive_angles))
                increment = p.progressive_rate * dt
                if p.progressive_all_joints:
                    for idx, max_val in zip(
                        p.progressive_all_joints, p.progressive_all_maxs
                    ):
                        self._progressive_angles[idx] = min(
                            max_val, self._progressive_angles[idx] + increment
                        )
                else:
                    abd_idx = p.progressive_abd_joint
                    flex_idx = p.progressive_flex_joint
                    self._progressive_angles[abd_idx] = min(
                        p.progressive_abd_max,
                        self._progressive_angles[abd_idx] + increment,
                    )
                    self._progressive_angles[flex_idx] = min(
                        p.progressive_flex_max,
                        self._progressive_angles[flex_idx] + increment * 0.5,
                    )
                return self._apply_guard(ctx, list(self._progressive_angles))

        self.grasp_state = "approaching"
        self._last_elapsed = elapsed
        return self._apply_guard(ctx, raw)


def make_phased_lerp_engine(hand_type: str, phase: str = "full") -> PhasedLerpEngine:
    return PhasedLerpEngine(load_thumb_adduction_params(hand_type), phase=phase)


class RotProgressiveEngine:
    """rot 先到位再渐进引擎。

    P1: 所有关节（含 rot）lerp 到 YAML angles.target（含 thumb_rot=255）。
    P2: 非 rot 关节保持 target 不变，rot 从 255 渐进到 abd_max。
        期间持续检测接触，触碰到即冻结。

    prep 模式: P1 执行 rot 到位 → grasp_state="ready"
    close/full 模式: P1+P2 全部执行 → grasp_state="contact"
    """

    def __init__(self, pose_target: List[float], rot_joint: int = 10,
                 rot_target: float = 200, rot_rate: float = 10.0,
                 pose_duration: float = 0.35, phase: str = "full",
                 contact_fingers: Optional[List[int]] = None) -> None:
        self._pose_target = list(pose_target)
        self._rot_joint = rot_joint
        self._rot_target = float(rot_target)
        self._rot_rate = float(rot_rate)
        self._pose_duration = float(pose_duration)
        self._phase = phase
        self._contact_fingers = list(contact_fingers or [])
        self._start: List[float] = []
        self._contact = FingerContactTracker()
        self._frozen_rot: Optional[float] = None
        self.grasp_state: str = "approaching"

    def reset(self, start_angles: List[float]) -> None:
        self._start = list(start_angles[:20])
        self._contact.reset()
        self._frozen_rot = None
        self.grasp_state = "approaching"

    def compute(
        self,
        feedback_angles: List[float],
        elapsed: float,
        ctx,
    ) -> List[float]:
        if not self._start:
            self.reset(feedback_angles)

        # prep 模式: P1 rot 奔向 255，到位即停
        if self._phase == "prep":
            if elapsed >= self._pose_duration:
                self.grasp_state = "ready"
                return list(self._pose_target)
            t = max(0.0, min(1.0, elapsed / self._pose_duration))
            return [s + (g - s) * t for s, g in zip(self._start, self._pose_target)]

        # full / close 模式
        #
        # P1: 所有关节（含 rot）lerp 到 target（rot=255）
        if elapsed < self._pose_duration:
            t = max(0.0, min(1.0, elapsed / self._pose_duration))
            self.grasp_state = "approaching"
            return [s + (g - s) * t for s, g in zip(self._start, self._pose_target)]

        # P2 开始：捕获基线
        if not hasattr(self, '_p2_entered'):
            self._p2_entered = True
            if self._contact_fingers:
                self._contact.begin_closing(ctx)

        # 接触冻结后保持当前姿态
        if self._frozen_rot is not None:
            self.grasp_state = "contact"
            out = list(self._pose_target)
            out[self._rot_joint] = self._frozen_rot
            return out

        # P2 每 tick 检测接触
        if self._contact_fingers and self._contact.check(ctx, tuple(self._contact_fingers)):
            self._frozen_rot = None  # 下面会取当前计算值
            # 用当前 rot 值冻结
            rot_elapsed = elapsed - self._pose_duration
            rot_p1_end = self._pose_target[self._rot_joint]
            if rot_p1_end > self._rot_target:
                current_rot = max(rot_p1_end - self._rot_rate * rot_elapsed, self._rot_target)
            else:
                current_rot = min(rot_p1_end + self._rot_rate * rot_elapsed, self._rot_target)
            self._frozen_rot = current_rot
            self.grasp_state = "contact"
            out = list(self._pose_target)
            out[self._rot_joint] = self._frozen_rot
            return out

        # P2: rot 渐进
        rot_p1_end = self._pose_target[self._rot_joint]
        rot_elapsed = elapsed - self._pose_duration

        if rot_p1_end > self._rot_target:
            new_rot = max(rot_p1_end - self._rot_rate * rot_elapsed, self._rot_target)
        else:
            new_rot = min(rot_p1_end + self._rot_rate * rot_elapsed, self._rot_target)

        out = list(self._pose_target)
        out[self._rot_joint] = new_rot

        if abs(new_rot - self._rot_target) < 1.0:
            self.grasp_state = "contact"
        else:
            self.grasp_state = "progressive"

        return out


class FingerAdductionEngine:
    """双关节渐进并拢引擎 — 专为 index_middle_adduction_grip 设计。

    P0 (可选): 拇指关节 lerp 到 target，其余保持起始值不动。
    P1: 其余关节 lerp 到 target (abd 在张开位，拇指已是 target)。
    P2: 两个 abd 从张开位 lerp 到并拢目标，期间接触检测冻结。

    prep 模式: P0+P1 执行到位 → grasp_state="ready"
    close/full 模式: P0+P1+P2 全部执行 → grasp_state="contact"
    """

    def __init__(self, pose_target: List[float], joint_a: int = 6, joint_b: int = 7,
                 close_a: float = 96, close_b: float = 160,
                 pose_duration: float = 0.35, close_duration: float = 1.0,
                 phase: str = "full",
                 contact_fingers: Optional[List[int]] = None,
                 thumb_joints: Optional[List[int]] = None,
                 thumb_duration: float = 0.0) -> None:
        self._pose_target = list(pose_target)
        self._joint_a = joint_a
        self._joint_b = joint_b
        self._close_a = float(close_a)
        self._close_b = float(close_b)
        self._pose_duration = float(pose_duration)
        self._close_duration = float(close_duration)
        self._phase = phase
        self._contact_fingers = list(contact_fingers or [])
        self._thumb_joints = list(thumb_joints or [])
        self._thumb_duration = float(thumb_duration) if thumb_joints else 0.0
        self._start: List[float] = []
        self._contact = FingerContactTracker()
        self._frozen_pose: Optional[List[float]] = None
        self.grasp_state: str = "approaching"

    @property
    def _total_prep(self) -> float:
        return self._thumb_duration + self._pose_duration

    def reset(self, start_angles: List[float]) -> None:
        self._start = list(start_angles[:20])
        self._contact.reset()
        self._frozen_pose = None
        self.grasp_state = "approaching"

    def _thumb_target(self) -> List[float]:
        """P0 终点: 拇指到 target，其余保持初始。"""
        out = list(self._start)
        for i in self._thumb_joints:
            if 0 <= i < 20:
                out[i] = self._pose_target[i]
        return out

    def compute(
        self,
        feedback_angles: List[float],
        elapsed: float,
        ctx,
    ) -> List[float]:
        if not self._start:
            self.reset(feedback_angles)

        total_prep = self._total_prep

        # prep 模式: 到位即停
        if self._phase == "prep":
            if elapsed >= total_prep:
                self.grasp_state = "ready"
                return list(self._pose_target)
            # P0: 拇指先行
            if self._thumb_duration > 0 and elapsed < self._thumb_duration:
                t = max(0.0, min(1.0, elapsed / self._thumb_duration))
                thumb_tgt = self._thumb_target()
                return lerp_angles(self._start, thumb_tgt, t)
            # P1: 其余到位
            t0 = self._thumb_duration
            from_pose = self._thumb_target() if self._thumb_duration > 0 else self._start
            t = max(0.0, min(1.0, (elapsed - t0) / self._pose_duration)) if self._pose_duration > 0 else 1.0
            return lerp_angles(from_pose, self._pose_target, t)

        # full / close 模式
        #
        # P0: 拇指先行
        if self._thumb_duration > 0 and elapsed < self._thumb_duration:
            t = max(0.0, min(1.0, elapsed / self._thumb_duration))
            self.grasp_state = "approaching"
            return lerp_angles(self._start, self._thumb_target(), t)

        # P1: 其余关节 lerp 到 target
        t0 = self._thumb_duration
        from_pose = self._thumb_target() if self._thumb_duration > 0 else self._start
        if elapsed < total_prep:
            t = max(0.0, min(1.0, (elapsed - t0) / self._pose_duration)) if self._pose_duration > 0 else 1.0
            self.grasp_state = "approaching"
            return lerp_angles(from_pose, self._pose_target, t)

        # P2 开始：基线捕获
        if not hasattr(self, '_p2_entered'):
            self._p2_entered = True
            if self._contact_fingers:
                self._contact.begin_closing(ctx)

        # 接触冻结
        if self._frozen_pose is not None:
            self.grasp_state = "contact"
            return list(self._frozen_pose)

        # 每 tick 检测接触
        if self._contact_fingers and self._contact.check(ctx, tuple(self._contact_fingers)):
            close_elapsed = elapsed - total_prep
            progress = min(close_elapsed / self._close_duration, 1.0)
            freeze = list(self._pose_target)
            freeze[self._joint_a] = self._pose_target[self._joint_a] + \
                (self._close_a - self._pose_target[self._joint_a]) * progress
            freeze[self._joint_b] = self._pose_target[self._joint_b] + \
                (self._close_b - self._pose_target[self._joint_b]) * progress
            self._frozen_pose = freeze
            self.grasp_state = "contact"
            return list(self._frozen_pose)

        # P2: abd 从张开位 lerp 到并拢目标
        close_elapsed = elapsed - total_prep
        progress = min(close_elapsed / self._close_duration, 1.0)

        out = list(self._pose_target)
        out[self._joint_a] = self._pose_target[self._joint_a] + \
            (self._close_a - self._pose_target[self._joint_a]) * progress
        out[self._joint_b] = self._pose_target[self._joint_b] + \
            (self._close_b - self._pose_target[self._joint_b]) * progress

        if progress >= 1.0:
            self.grasp_state = "contact"
        else:
            self.grasp_state = "progressive"

        return out





def adaptive_index_ring_angles(
    object_size, params: "ThumbAdductionParams",
) -> List[float]:
    """根据物体截面直径调整拇食指闭合量。"""
    angles = list(params.prep_angles)
    s = sorted(object_size)
    d_mm = (s[0] + s[1]) / 2.0 * 1000.0
    angles[0] = int(np.clip(150 - (d_mm - 5) * 2.4, 90, 150))    # thumb_base
    angles[1] = int(np.clip(170 - (d_mm - 5) * 2.8, 100, 170))   # index_base
    if params.thumb_rot_joint >= 0:
        angles[15] = int(np.clip(200 - (d_mm - 5) * 2.0, 140, 200))
        angles[16] = int(np.clip(150 - (d_mm - 5) * 2.0, 80, 150))
    return angles


def make_index_ring_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from dataclasses import replace
    from .gesture_params import load_index_ring_by_vision_params
    params = load_index_ring_by_vision_params(hand_type)
    if params is None:
        return None
    params = replace(params, adaptive_fn=adaptive_index_ring_angles)
    return PhasedLerpEngine(params, phase=phase)


def adaptive_large_wrap_angles(
    object_size, params: "ThumbAdductionParams",
) -> List[float]:
    """根据物体最大截面直径调整五指包络弯曲量。

    D=40mm → base≈230(紧握); D=100mm → base≈150(半开)
    """
    angles = list(params.prep_angles)
    s = sorted(object_size)
    D_mm = float(max(s[0], s[1])) * 1000.0
    base_val = int(np.clip(250 - (D_mm - 30) * 1.4, 130, 240))
    for i in params.close_flex_indices():
        angles[i] = base_val
    return angles


def make_large_wrap_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from dataclasses import replace
    from .gesture_params import load_large_wrap_by_vision_params
    params = load_large_wrap_by_vision_params(hand_type)
    if params is None:
        return None
    params = replace(params, adaptive_fn=adaptive_large_wrap_angles)
    return PhasedLerpEngine(params, phase=phase)


def make_tripod_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_tripod_by_vision_params
    params = load_tripod_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


def make_hook_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_hook_by_vision_params
    params = load_hook_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


def make_small_warp_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_small_warp_by_vision_params
    params = load_small_warp_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


def make_no_index_warp_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_no_index_warp_by_vision_params
    params = load_no_index_warp_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


def make_palmar_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_palmar_by_vision_params
    params = load_palmar_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


def make_middle_ring_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_middle_ring_by_vision_params
    params = load_middle_ring_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


def make_ring_by_vision_engine(
    hand_type: str, phase: str = "full",
) -> Optional[PhasedLerpEngine]:
    from .gesture_params import load_ring_by_vision_params
    params = load_ring_by_vision_params(hand_type)
    if params is None:
        return None
    return PhasedLerpEngine(params, phase=phase)


class StaticPoseEngine:
    """单目标姿态线性插值。"""

    def __init__(self, params: StaticGestureParams) -> None:
        self._params = params
        self._start_angles: List[float] = [0.0] * len(params.target_angles)
        self._target: List[float] = list(params.target_angles)

    def reset(self, start_angles: List[float], target: Optional[List[float]] = None) -> None:
        self._start_angles = list(start_angles)
        self._target = list(target if target is not None else self._params.target_angles)

    def compute(self, elapsed: float) -> List[float]:
        t = elapsed / self._params.duration
        if t >= 1.0:
            return list(self._target)
        return lerp_angles(self._start_angles, self._target, t)


class PhasedPoseEngine:
    """分阶段姿态（fist）：P1 四指/非 hold 关节 → P2 abd → P3 rot → P4 全到位。"""

    def __init__(self, params: StaticGestureParams) -> None:
        self._params = params
        self._start_angles: List[float] = [0.0] * len(params.target_angles)
        self._phase1_target: List[float] = []
        self._phase2_target: List[float] = []
        self._phase3_target: List[float] = []

    def reset(self, start_angles: List[float]) -> None:
        self._start_angles = list(start_angles)
        target = self._params.target_angles
        p = self._params

        self._phase1_target = list(target)
        for i in p.phase1_hold:
            if i < len(self._phase1_target):
                self._phase1_target[i] = self._start_angles[i]

        self._phase2_target = list(self._phase1_target)
        if 5 < len(target):
            self._phase2_target[5] = target[5]

        self._phase3_target = list(self._phase2_target)
        if p.phase3 > 0 and 10 < len(target):
            self._phase3_target[10] = target[10]

    def compute(self, elapsed: float) -> List[float]:
        p = self._params
        target = p.target_angles
        t1 = p.phase1
        t2 = t1 + p.phase2
        t3 = t2 + (p.phase3 if p.phase3 > 0 else 0.0)
        t4 = t3 + p.phase4

        if elapsed < t1:
            return lerp_angles(self._start_angles, self._phase1_target, elapsed / t1)
        if elapsed < t2:
            return lerp_angles(
                self._phase1_target, self._phase2_target, (elapsed - t1) / p.phase2)
        if p.phase3 > 0 and elapsed < t3:
            return lerp_angles(
                self._phase2_target, self._phase3_target, (elapsed - t2) / p.phase3)
        if elapsed < t4:
            from_target = self._phase3_target if p.phase3 > 0 else self._phase2_target
            t_start = t3 if p.phase3 > 0 else t2
            dur = p.phase4
            return lerp_angles(from_target, target, (elapsed - t_start) / dur)
        return list(target)


class RelaxGripEngine:
    """放松 → 停顿 → 握紧；可选子段 phased_pose（O6 拇延后等）。"""

    def __init__(self, params: StaticGestureParams) -> None:
        self._params = params
        self._start_angles: List[float] = [0.0] * len(params.relax_angles)
        self._done = False
        self._grip_armed = False
        self._relax_phased: Optional[PhasedPoseEngine] = None
        self._grip_phased: Optional[PhasedPoseEngine] = None
        if params.relax_phased is not None:
            self._relax_phased = PhasedPoseEngine(params.relax_phased)
        if params.grip_phased is not None:
            self._grip_phased = PhasedPoseEngine(params.grip_phased)

    def reset(self, start_angles: List[float]) -> None:
        self._start_angles = list(start_angles)
        self._done = False
        self._grip_armed = False
        if self._relax_phased is not None:
            self._relax_phased.reset(start_angles)

    @property
    def done(self) -> bool:
        return self._done

    def compute(self, elapsed: float) -> List[float]:
        p = self._params
        t_relax = p.duration
        t_hold = t_relax + p.hold_duration
        t_grip = t_hold + p.grip_duration

        if elapsed < t_relax:
            if self._relax_phased is not None:
                return self._relax_phased.compute(elapsed)
            return lerp_angles(
                self._start_angles, p.relax_angles, elapsed / t_relax)
        if elapsed < t_hold:
            return list(p.relax_angles)
        if elapsed < t_grip:
            t_grip_elapsed = elapsed - t_hold
            if self._grip_phased is not None:
                if not self._grip_armed:
                    self._grip_phased.reset(p.relax_angles)
                    self._grip_armed = True
                return self._grip_phased.compute(t_grip_elapsed)
            return lerp_angles(
                p.relax_angles, p.grip_angles,
                t_grip_elapsed / p.grip_duration)
        self._done = True
        return list(p.grip_angles)


def make_static_engine(hand_type: str, primitive: str):
    """按 motion_type 构造静态/分阶段引擎。"""
    params = load_static_gesture_params(hand_type, primitive)
    if params.motion_type == "phased_pose":
        return PhasedPoseEngine(params)
    if params.motion_type == "relax_grip":
        return RelaxGripEngine(params)
    return StaticPoseEngine(params)


class SequentialForceCloseEngine:
    """预成型 lerp → 分步顺序力控闭合（MCP 版 ring 等）。

    力/触觉反馈按 **组** 隔离：食指接触只停食指组，中指/拇指可继续。
    """

    def __init__(self, params: SequentialForceCloseParams) -> None:
        self._params = params
        self._start_angles: List[float] = [0.0] * 20
        self._close_angles: List[float] = [0.0] * 20
        self._baseline: List[float] = []
        self._motion_baseline: List[float] = []
        self._settle_count = 0
        self._step_index = 0
        self._stopped: dict = {}
        self._phase = "preshape"
        self.grasp_state = "approaching"
        self._last_output: Optional[List[float]] = None

    def reset(self, start_angles: List[float]) -> None:
        self._start_angles = list(start_angles[:20])
        while len(self._start_angles) < 20:
            self._start_angles.append(0.0)
        self._close_angles = list(self._start_angles)
        self._baseline = []
        self._motion_baseline = []
        self._settle_count = 0
        self._step_index = 0
        self._stopped = {}
        self._phase = "preshape"
        self.grasp_state = "approaching"
        self._last_output = None

    def _joints_for_group(self, group) -> List[int]:
        return list(group.joint_indices)

    def _apply_per_group_hold(self, proposed: List[float]) -> List[float]:
        """已接触组锁定上帧角度，其余关节继续运动。"""
        if not self._last_output:
            return list(proposed)
        out = list(proposed)
        for step in self._params.steps:
            for g in step.groups:
                if not self._stopped.get(g.group_id):
                    continue
                for ji in self._joints_for_group(g):
                    if 0 <= ji < len(out) and ji < len(self._last_output):
                        out[ji] = self._last_output[ji]
        return out

    def _finalize_output(self, proposed: List[float]) -> List[float]:
        out = self._apply_per_group_hold(proposed)
        self._last_output = list(out)
        self._update_grasp_state()
        return out

    def _update_grasp_state(self) -> None:
        if self._phase == "hold":
            self.grasp_state = "contact"
        elif self._phase == "close":
            self.grasp_state = "closing"
        elif self._phase == "preshape":
            self.grasp_state = (
                "closing" if any(self._stopped.values()) else "approaching"
            )

    def _mark_blocked_groups(self, ctx) -> None:
        """全程扫描：已接触的组单独标记，不影响其他组。"""
        import sys as _sys
        for step in self._params.steps:
            for g in step.groups:
                if self._stopped.get(g.group_id):
                    continue
                if self._group_contact(ctx, g):
                    print(f"[SFCE] CONTACT step={step.name}[{self._step_index}] "
                          f"group={g.group_id} mon={g.monitor_hw} "
                          f"delta={g.contact_delta}", file=_sys.stderr)
                    self._stopped[g.group_id] = True

    def _init_stopped_flags(self) -> None:
        if self._step_index < len(self._params.steps):
            for g in self._params.steps[self._step_index].groups:
                self._stopped[g.group_id] = False  # force reset (setdefault could leave stale True from preshape)

    def _feedback_baseline(self, ctx) -> List[float]:
        """全程力反馈基线：close 阶段 settle 后刷新 _baseline，此前用 motion 起始基线。"""
        if self._baseline:
            return self._baseline
        if not self._motion_baseline:
            self._motion_baseline = capture_feedback_baseline(ctx)
        return self._motion_baseline

    def _group_contact(self, ctx, group) -> bool:
        return force_close_group_contact(
            ctx,
            tactile_finger=group.tactile_finger,
            monitor_hw=group.monitor_hw,
            baseline=self._feedback_baseline(ctx),
            contact_delta=group.contact_delta,
        )

    def _tick_close_groups(self, ctx) -> None:
        step_cfg = self._params.steps[self._step_index]
        all_done = True
        import sys as _sys
        for g in step_cfg.groups:
            if self._stopped.get(g.group_id):
                print(f"[SFCE] group {g.group_id} already stopped", file=_sys.stderr)
                continue
            all_done = False
            if self._group_contact(ctx, g):
                print(f"[SFCE] CONTACT on {g.group_id}!", file=_sys.stderr)
                self._stopped[g.group_id] = True
                continue
            moved = False
            for ji in g.joint_indices:
                if ji < 0 or ji >= len(self._close_angles):
                    continue
                if self._close_angles[ji] < g.close_max:
                    old = self._close_angles[ji]
                    self._close_angles[ji] = min(
                        g.close_max, self._close_angles[ji] + g.step)
                    moved = True
                    print(f"[SFCE] inc ji={ji} {old:.0f}->{self._close_angles[ji]:.0f} (step={g.step})", file=_sys.stderr)
                else:
                    self._stopped[g.group_id] = True
                    print(f"[SFCE] ji={ji} at max, stopping group {g.group_id}", file=_sys.stderr)
            if not moved and not self._stopped.get(g.group_id):
                self._stopped[g.group_id] = True
                print(f"[SFCE] no move, stopping {g.group_id}", file=_sys.stderr)
        if all_done or all(self._stopped.get(g.group_id) for g in step_cfg.groups):
            print(f"[SFCE] step {self._step_index} complete, advancing", file=_sys.stderr)
            self._step_index += 1
            self._settle_count = 0
            if self._step_index < len(self._params.steps):
                self._init_stopped_flags()
            else:
                self._phase = "hold"
                self.grasp_state = "contact"

    def _tick_hold(self, ctx) -> None:
        p = self._params
        from .hand_config import HandConfig as YamlHandConfig

        hand = YamlHandConfig(p.hand_type)
        monitor_hw: set = set()
        for step in p.steps:
            for g in step.groups:
                monitor_hw.update(g.monitor_hw)
        hold_limit = (
            p.hold_safe_torque_pct if uses_torque_feedback(p.hand_type)
            else p.hold_safe_current
        )
        for hi, spec in hand._reverse_map.items():
            if hi not in monitor_hw:
                continue
            sem = spec["from"]
            if hi >= len(self._close_angles):
                continue
            if not hold_feedback_exceeded(ctx, hi, hold_limit):
                continue
            vals = capture_feedback_baseline(ctx)
            cur = float(vals[hi]) if hi < len(vals) and vals[hi] >= 0 else hold_limit
            over = cur - hold_limit
            step = max(1, int(over / (30.0 if not uses_torque_feedback(p.hand_type) else 3.0)))
            if sem < len(self._close_angles) and self._close_angles[sem] > 0:
                self._close_angles[sem] = max(0.0, self._close_angles[sem] - step)

    def compute(self, elapsed: float, ctx) -> List[float]:
        p = self._params
        if self._phase == "preshape":
            target = list(p.prep_angles)
            t = min(elapsed / p.preshape_duration, 1.0)
            proposed = (
                list(target) if t >= 1.0
                else lerp_angles(self._start_angles, target, t)
            )
            self._mark_blocked_groups(ctx)
            out = self._finalize_output(proposed)
            if t >= 1.0:
                self._phase = "close"
                self._close_angles = list(out)
                self._baseline = capture_feedback_baseline(ctx)
                self._settle_count = 0
                self._step_index = 0
                self._init_stopped_flags()
                self._mark_blocked_groups(ctx)
                self.grasp_state = "closing"
                return self._finalize_output(list(self._close_angles))
            return out

        if self._phase == "close":
            if self._settle_count < p.settle_frames:
                self._settle_count += 1
                if self._settle_count == p.settle_frames:
                    self._baseline = capture_feedback_baseline(ctx)
                    import sys as _sys
                    print(f"[SFCE] settle done, baseline={self._baseline}", file=_sys.stderr)
                # 不在此处 _mark_blocked_groups：马达尚未稳定，运动力矩 ＞ 静息基线 → 误触
                self._close_angles = self._finalize_output(list(self._close_angles))
                return list(self._close_angles)
            self._mark_blocked_groups(ctx)
            self._tick_close_groups(ctx)
            self._close_angles = self._finalize_output(list(self._close_angles))
            import sys as _sys
            print(f"[SFCE] close phase mid[2]={self._close_angles[2]:.1f} step={self._step_index} stopped={self._stopped}", file=_sys.stderr)
            return list(self._close_angles)

        self._tick_hold(ctx)
        out = list(self._close_angles)
        self._last_output = out
        self.grasp_state = "contact"
        return out


def make_sequential_force_close_engine(
    hand_type: str, primitive: str,
) -> Optional[SequentialForceCloseEngine]:
    params = load_sequential_force_close_params(hand_type, primitive)
    if params is None:
        return None
    return SequentialForceCloseEngine(params)


def make_ring_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "ring")


def make_middle_ring_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "middle_ring")


def make_middle_ring_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    """middle_ring_by_vision SFCE 引擎（O6 顺序力控）。"""
    return make_sequential_force_close_engine(hand_type, "middle_ring_by_vision")


def make_tripod_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "tripod_by_vision")


def make_hook_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "hook_by_vision")


def make_small_warp_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "small_warp_by_vision")


def make_no_index_warp_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "no_index_warp_by_vision")


def make_ring_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "ring_by_vision")


def make_palmar_by_vision_sfce_engine(hand_type: str) -> Optional[SequentialForceCloseEngine]:
    return make_sequential_force_close_engine(hand_type, "palmar_by_vision")
