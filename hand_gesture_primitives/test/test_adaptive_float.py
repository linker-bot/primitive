"""自适应公式输出 float，避免 int 量化在 rad 往返中产生台阶。"""

import numpy as np
from hand_gesture_primitives.gesture_engine import (
    adaptive_index_ring_angles,
    adaptive_large_wrap_angles,
    adaptive_thumb_adduction_angles,
)
from hand_gesture_primitives.gesture_params import ThumbAdductionParams
from hand_gesture_primitives.primitives.index_ring_by_vision import (
    _adaptive_index_ring_by_vision_angles,
)
from hand_gesture_primitives.primitives.large_wrap_by_vision import (
    _adaptive_large_wrap_by_vision_angles,
)
from hand_gesture_primitives.primitives.middle_ring import _width_to_thumb_rot
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM


def _params() -> ThumbAdductionParams:
    return ThumbAdductionParams(
        hand_type="o20",
        prep_angles=[128.0] * SEMANTIC_DIM,
        close_angles=[200.0] * SEMANTIC_DIM,
        thumb_rot_joint=SEMANTIC_30_IDX["thumb_rot"],
        thumb_flex_joints=[
            SEMANTIC_30_IDX["thumb_mcp_flex"],
            SEMANTIC_30_IDX["thumb_dip"],
        ],
        close_move_joints=[
            SEMANTIC_30_IDX["index_mcp_flex"],
            SEMANTIC_30_IDX["middle_mcp_flex"],
            SEMANTIC_30_IDX["ring_mcp_flex"],
            SEMANTIC_30_IDX["pinky_mcp_flex"],
            SEMANTIC_30_IDX["thumb_mcp_flex"],
            SEMANTIC_30_IDX["thumb_dip"],
        ],
    )


def test_adaptive_thumb_adduction_is_float_and_continuous() -> None:
    p = _params()
    idx = SEMANTIC_30_IDX["index_mcp_flex"]
    a = adaptive_thumb_adduction_angles(np.array([0.005, 0.02, 0.03]), p)
    b = adaptive_thumb_adduction_angles(np.array([0.0051, 0.02, 0.03]), p)
    assert isinstance(a[idx], float)
    assert abs(a[idx] - b[idx]) > 0.0


def test_adaptive_index_ring_is_float_and_continuous() -> None:
    p = _params()
    a = adaptive_index_ring_angles(np.array([0.01, 0.012, 0.02]), p)
    b = adaptive_index_ring_angles(np.array([0.01005, 0.01205, 0.02]), p)
    for i in (
        SEMANTIC_30_IDX["thumb_mcp_flex"],
        SEMANTIC_30_IDX["index_mcp_flex"],
        SEMANTIC_30_IDX["thumb_dip"],
        SEMANTIC_30_IDX["index_dip"],
    ):
        assert isinstance(a[i], float)
    assert abs(a[SEMANTIC_30_IDX["index_mcp_flex"]] - b[SEMANTIC_30_IDX["index_mcp_flex"]]) > 0.0


def test_adaptive_large_wrap_is_float_and_continuous() -> None:
    p = _params()
    idx = SEMANTIC_30_IDX["index_mcp_flex"]
    a = adaptive_large_wrap_angles(np.array([0.05, 0.05, 0.08]), p)
    b = adaptive_large_wrap_angles(np.array([0.0501, 0.05, 0.08]), p)
    assert isinstance(a[idx], float)
    assert abs(a[idx] - b[idx]) > 0.0


def test_legacy_index_ring_by_vision_adaptive_float() -> None:
    a = _adaptive_index_ring_by_vision_angles(np.array([0.01, 0.012, 0.02]))
    b = _adaptive_index_ring_by_vision_angles(np.array([0.0105, 0.0125, 0.02]))
    assert abs(a[SEMANTIC_30_IDX["thumb_mcp_flex"]] - b[SEMANTIC_30_IDX["thumb_mcp_flex"]]) > 0.0


def test_legacy_large_wrap_by_vision_adaptive_float() -> None:
    a = _adaptive_large_wrap_by_vision_angles(np.array([0.05, 0.05, 0.08]))
    b = _adaptive_large_wrap_by_vision_angles(np.array([0.0505, 0.05, 0.08]))
    assert abs(a[SEMANTIC_30_IDX["index_mcp_flex"]] - b[SEMANTIC_30_IDX["index_mcp_flex"]]) > 0.0


def test_width_to_thumb_rot_is_float_and_continuous() -> None:
    a = _width_to_thumb_rot(20.0)
    b = _width_to_thumb_rot(20.5)
    assert isinstance(a, float)
    assert abs(a - b) > 0.0
