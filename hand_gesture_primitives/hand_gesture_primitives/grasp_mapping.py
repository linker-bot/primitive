"""感知 grasp_type → 自适应原语映射。"""

from typing import Optional

# 与 perception classify_grasp_type 输出一致
GRASP_TYPE_TO_PRIMITIVE = {
    'precision': 'index_ring_by_vision',
    'lateral': 'index_middle_adduction_grip',
    'power': 'large_wrap_by_vision',
}


def primitive_for_grasp_type(grasp_type: str) -> Optional[str]:
    if not grasp_type:
        return None
    return GRASP_TYPE_TO_PRIMITIVE.get(grasp_type.lower().strip())
