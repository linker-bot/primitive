"""大拇指侧向夹持原语 — 姿态到位 + rot 关节渐进。

engine 字段控制行为:
  - 不填 / "phased_lerp": PhasedLerpEngine (O20 旧行为, 4 阶段)
  - "rot_progressive": RotProgressiveEngine (L25 新行为, P1 到位 + P2 rot 渐进)
"""

from typing import List, Optional

from ..angle_bridge import AngleUnits
from ..gesture_engine import (
    PhasedLerpEngine,
    RotProgressiveEngine,
    adaptive_thumb_adduction_angles,
    make_phased_lerp_engine,
)
from ..gesture_params import (
    CANONICAL_SEMANTIC_HAND,
    ThumbAdductionParams,
    load_thumb_adduction_params,
    load_thumb_adduction_params_for_ctx,
)
from ..primitive_base import (
    HandGesturePrimitive,
    PrimitiveContext,
    PrimitiveResult,
    parse_grasp_phase,
)
from ..semantic_30 import SEMANTIC_30_IDX

# 向后兼容：canonical semantic_o20 默认常量（与 gestures 缺省 / o20.yaml 一致）
_ref_gesture_params = load_thumb_adduction_params(CANONICAL_SEMANTIC_HAND)
THUMB_ABD_PREP = float(_ref_gesture_params.prep_angles[SEMANTIC_30_IDX["thumb_mcp_abd"]])
THUMB_ROT = float(_ref_gesture_params.prep_angles[_ref_gesture_params.thumb_rot_joint])
THUMB_BASE = float(_ref_gesture_params.close_angles[SEMANTIC_30_IDX["thumb_mcp_flex"]])
THUMB_TIP = float(_ref_gesture_params.close_angles[SEMANTIC_30_IDX["thumb_dip"]])
THUMB_ADDUCTION_ANGLES = list(_ref_gesture_params.prep_angles)
THUMB_ALL_INDICES = list(_ref_gesture_params.thumb_hold_joints)
THUMB_ROT_INDEX = _ref_gesture_params.thumb_rot_joint
THUMB_FLEX_INDICES = list(_ref_gesture_params.thumb_flex_joints)
THUMB_TIP_PROGRESSIVE_MAX = _ref_gesture_params.progressive_flex_max
THUMB_ABD_PROGRESSIVE_MAX = _ref_gesture_params.progressive_abd_max
THUMB_BASE_PROGRESSIVE_MAX = 255

PHASE1_DURATION = _ref_gesture_params.phase1
PHASE2_DURATION = _ref_gesture_params.phase2
PHASE3_DURATION = _ref_gesture_params.phase3
PHASE4_DURATION = _ref_gesture_params.phase4
PREP_END_REF = _ref_gesture_params.prep_end
PROGRESSIVE_CLOSE_RATE = _ref_gesture_params.progressive_rate


def _params(ctx: PrimitiveContext) -> ThumbAdductionParams:
    return load_thumb_adduction_params_for_ctx(ctx)


def _default_angles(ctx: PrimitiveContext) -> List[float]:
    return list(_params(ctx).prep_angles)


def _thumb_hold_indices(ctx: PrimitiveContext) -> List[int]:
    return list(_params(ctx).thumb_hold_joints)


def _thumb_flex_indices(ctx: PrimitiveContext) -> List[int]:
    return _params(ctx).close_flex_indices()


def _phase3_duration(ctx: PrimitiveContext) -> float:
    p = _params(ctx)
    if p.thumb_rot_joint < 0 or p.phase3 <= 0:
        return 0.0
    return p.phase3


def _prep_end(ctx: PrimitiveContext) -> float:
    return _params(ctx).prep_end


def _progressive_indices(ctx: PrimitiveContext) -> tuple:
    p = _params(ctx)
    return p.progressive_abd_joint, p.progressive_flex_joint


def _adaptive_thumb_adduction_angles(object_size, ctx: PrimitiveContext):
    return adaptive_thumb_adduction_angles(object_size, _params(ctx))


class ThumbAdductionGrip(HandGesturePrimitive):
    """拇指侧向夹持。

    engine="rot_progressive" (L25 新): RotProgressiveEngine
        P1: 所有关节 lerp → YAML angles.target
        P2: rot 渐进 → YAML progressive.abd_max

    engine 未填 / "phased_lerp" (O20 旧): PhasedLerpEngine
        4 阶段运动，接触检测 + progressive abd/flex。
    """

    def __init__(self, phase: str = "full") -> None:
        self._phase = parse_grasp_phase(phase)
        self._engine: Optional[PhasedLerpEngine] = None
        self._engine_new: Optional[RotProgressiveEngine] = None
        self._engine_hand_type: str = ""
        self._use_new_engine: bool = False
        self._result_space = "semantic_rad"

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def grasp_state(self) -> str:
        if self._use_new_engine and self._engine_new is not None:
            return self._engine_new.grasp_state
        if self._engine is not None:
            return self._engine.grasp_state
        return "approaching"

    @grasp_state.setter
    def grasp_state(self, value: str) -> None:
        if self._use_new_engine and self._engine_new is not None:
            self._engine_new.grasp_state = value
        elif self._engine is not None:
            self._engine.grasp_state = value

    def _ensure_engine(self, ctx: PrimitiveContext):
        if self._engine_hand_type != ctx.hand_type:
            self._engine = None
            self._engine_new = None
            self._engine_hand_type = ""

        p = _params(ctx)
        self._use_new_engine = p.engine == "rot_progressive"

        if self._use_new_engine:
            if self._engine_new is None:
                p_eng = p
                u = AngleUnits(ctx)
                rot_joint = (
                    p_eng.thumb_rot_joint
                    if p_eng.thumb_rot_joint >= 0
                    else SEMANTIC_30_IDX["thumb_rot"]
                )
                # progressive_* 已是 rad（有 Converter）或 legacy255
                rot_target = (
                    p_eng.progressive_abd_max
                    if p_eng.progressive_abd_max > 0
                    else u.value(rot_joint, 200.0)
                )
                rot_rate = (
                    p_eng.progressive_rate
                    if p_eng.progressive_rate > 0
                    else u.step(rot_joint, 10.0)
                )
                contact_fingers = list(p_eng.contact_fingers) if p_eng.contact_fingers else None
                self._engine_new = RotProgressiveEngine(
                    pose_target=list(p_eng.prep_angles),
                    rot_joint=rot_joint,
                    rot_target=rot_target,
                    rot_rate=rot_rate,
                    pose_duration=p_eng.phase1,
                    phase=self._phase,
                    contact_fingers=contact_fingers,
                    rot_arrive_eps=u.step(rot_joint, 1.0),
                )
                self._engine_new.reset(self._start_angles)
                self._engine_hand_type = ctx.hand_type
                self._result_space = u.space
            return self._engine_new
        else:
            if self._engine is None:
                self._engine = make_phased_lerp_engine(
                    ctx.hand_type, self._phase, radian_converter=ctx.radian_converter
                )
                self._engine.reset(self._start_angles)
                self._engine_hand_type = ctx.hand_type
                self._result_space = AngleUnits(ctx).space
            return self._engine

    def on_enter(self, current_angles: List[float]) -> None:
        super().on_enter(current_angles)
        self._engine = None
        self._engine_new = None
        self._engine_hand_type = ""
        self._result_space = "semantic_rad"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        engine = self._ensure_engine(ctx)
        raw = engine.compute(current_angles, elapsed, ctx)
        return self._move(raw, space=getattr(self, "_result_space", AngleUnits(ctx).space))

    @property
    def name(self) -> str:
        return "thumb_adduction_grip"

    @property
    def done(self) -> bool:
        return False
