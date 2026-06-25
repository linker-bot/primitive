"""平行伸展捏取原语（无 vision 别名）— 固定姿态 lerp，不依赖 tcp/object_pose。

与 parallel_extension_by_vision 的区别:
  - 不经 GraspGate 三判定（短名不在 GATED_PRIMITIVES）
  - 无 tcp_pose / object_pose 前置检查，适合 bench 快速测手
  - 产线经 Gate + 感知请用 parallel_extension_by_vision
"""

from typing import List

from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult, lerp_angles
from .parallel_extension_by_vision import PARALLEL_EXT_ANGLES


class ParallelExtension(HandGesturePrimitive):
    """拇指与四指平行伸展对捏 (1 vs 2-5)，固定 0.6s lerp。"""

    TRANSITION_DURATION = 0.6

    @property
    def name(self) -> str:
        return "parallel_extension"

    def compute(
        self, current_angles: List[float], elapsed: float, ctx: PrimitiveContext
    ) -> PrimitiveResult:
        t = elapsed / self.TRANSITION_DURATION
        if t >= 1.0:
            return self._move(list(PARALLEL_EXT_ANGLES))
        return self._move(lerp_angles(self._start_angles, PARALLEL_EXT_ANGLES, t))

    @property
    def done(self) -> bool:
        return False
