"""三指接触停指捏取 — 运动复用 tripod，闭合阶段逐指冻结。

时序与 ``tripod`` 相同：默认 ``auto``（裸发同效）；``prep`` 等待 close。
L6：电流增量作为力矩代理（SDK torque 是上限回读）。
其它手型：``auto`` 亦走 prep→close，力控只在 close 生效。
接触策略见 ``tripod_force.py``，按 YAML ``gestures.tripod_by_force.contact`` 扩展。
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..gesture_params import load_tripod_stage_durations
from ..primitive_base import PrimitiveContext, PrimitiveResult
from ..tripod_force import (
    TripodForceParams,
    apply_finger_freeze,
    capture_baseline,
    feedback_ready,
    groups_all_frozen,
    load_tripod_force_params,
)
from .tripod import Tripod


class TripodByForce(Tripod):
    """tripod 运动 + 逐指接触停指。"""

    def __init__(self, phase: str = "auto") -> None:
        super().__init__(phase=phase)
        self._force_params: Optional[TripodForceParams] = None
        self._feedback_baseline: Optional[List[float]] = None
        self._force_close_started_at: Optional[float] = None
        self._frozen_joints: Dict[int, float] = {}
        self._last_output: List[float] = []

    @property
    def name(self) -> str:
        return "tripod_by_force"

    def trajectory_complete(self, result: PrimitiveResult, elapsed: float) -> bool:
        # Only the explicit positioning prep is finite; closing remains force controlled.
        return self.phase == "prep" and super().trajectory_complete(result, elapsed)

    def supports_progress(self) -> bool:
        # 力控 close；与 progress_policy 一致，topic/cmd 均拒绝
        return False

    def set_progress(self, p: float) -> None:
        del p
        return

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._force_params = None
        self._feedback_baseline = None
        self._force_close_started_at = None
        self._frozen_joints = {}
        self._last_output = list(current_angles)

    def _uses_two_stage_full(self, ctx: PrimitiveContext) -> bool:
        return ctx.hand_type.lower() != "l6"

    def _params_for(self, ctx: PrimitiveContext) -> TripodForceParams:
        key = ctx.hand_type.lower()
        if self._force_params is None or self._force_params.hand_type != key:
            self._force_params = load_tripod_force_params(key)
        return self._force_params

    def _scheduled_close_start(self, ctx: PrimitiveContext) -> Optional[float]:
        if self.phase == "prep":
            return None
        if self.phase == "close":
            return 0.0
        # auto（含 full）：与 tripod 同节奏进入 close
        if ctx.hand_type.lower() == "l6":
            l6 = self._ensure_l6_params()
            return max(0.0, l6.phase1) + max(0.0, l6.pause_duration)
        prep, _close, pause = load_tripod_stage_durations(ctx.hand_type, "tripod")
        return max(0.0, prep) + max(0.0, pause)

    def _adjusted_elapsed(self, elapsed: float, close_start: float) -> float:
        """反馈等待不占用第二阶段动作时间。"""
        assert self._force_close_started_at is not None
        close_elapsed = max(0.0, elapsed - self._force_close_started_at)
        return close_elapsed if self.phase == "close" else close_start + close_elapsed

    def compute(
        self,
        current_angles: List[float],
        elapsed: float,
        ctx: PrimitiveContext,
    ) -> PrimitiveResult:
        try:
            params = self._params_for(ctx)
        except (ValueError, FileNotFoundError) as exc:
            return self._hold(str(exc))

        close_start = self._scheduled_close_start(ctx)
        if close_start is None or elapsed < close_start:
            result = super().compute(current_angles, elapsed, ctx)
            if result.feasible:
                self._last_output = list(result.target_angles)
            return result

        if params.require_feedback and not feedback_ready(params, ctx):
            self._grasp_state = "waiting_feedback"
            return self._hold(f"tripod_by_force 等待 {params.feedback_source} 反馈后再闭合")

        if self._force_close_started_at is None:
            self._force_close_started_at = elapsed
            self._feedback_baseline = capture_baseline(ctx)

        if self._feedback_baseline is None:
            self._feedback_baseline = capture_baseline(ctx)

        adjusted_elapsed = self._adjusted_elapsed(elapsed, close_start)
        result = super().compute(current_angles, adjusted_elapsed, ctx)
        if not result.feasible:
            return result

        out = apply_finger_freeze(
            params,
            ctx,
            result.target_angles,
            self._last_output,
            self._feedback_baseline,
            self._frozen_joints,
        )
        if groups_all_frozen(params, self._frozen_joints):
            self._grasp_state = "contact"
        self._last_output = list(out)
        return self._move(out, space=result.space, trajectory_finished=False)
