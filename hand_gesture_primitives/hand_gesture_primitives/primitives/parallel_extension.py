"""平行伸展捏取原语（无 vision 别名）— 固定姿态 lerp，不依赖 tcp/object_pose。"""

from typing import List

from ..angle_bridge import AngleUnits
from ..gesture_params import load_static_params_for_ctx, static_yaml_legacy255_or_none
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult, lerp_angles
from .parallel_extension_by_vision import PARALLEL_EXT_ANGLES, PARALLEL_EXT_ANGLES_L25


class ParallelExtension(HandGesturePrimitive):
    """拇指与四指平行伸展对捏 (1 vs 2-5)，固定 0.6s lerp。"""

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "parallel_extension"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        u = AngleUnits(ctx)
        yaml255 = static_yaml_legacy255_or_none(ctx.hand_type, self.name)
        if yaml255 is not None:
            params = load_static_params_for_ctx(ctx.hand_type, self.name, ctx)
            target = list(params.target_angles)
            duration = params.duration if params.duration > 0 else self.TRANSITION_DURATION
        else:
            fallback = PARALLEL_EXT_ANGLES_L25 if ctx.hand_type == "l25" else PARALLEL_EXT_ANGLES
            target = u.pose(fallback)
            duration = self.TRANSITION_DURATION
        t = elapsed / duration
        if t >= 1.0:
            return self._move(list(target), space=u.space, trajectory_finished=True)
        return self._move(lerp_angles(self._start_angles, target, t), space=u.space)

    @property
    def done(self) -> bool:
        return False
