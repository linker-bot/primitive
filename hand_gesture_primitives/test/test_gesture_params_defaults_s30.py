"""gesture_params 内置默认表已迁 semantic_v3 / S30。"""

from hand_gesture_primitives.gesture_params import (
    _DEFAULT_STATIC,
    _DEFAULT_THUMB_ADDUCTION,
    _fist_o20,
    _open_o20,
    _parse_thumb_adduction,
    clear_gesture_params_cache,
    load_static_gesture_params,
    load_thumb_adduction_params,
)
from hand_gesture_primitives.hand_config import HandConfig
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM


def test_default_thumb_adduction_is_semantic_v3() -> None:
    assert _DEFAULT_THUMB_ADDUCTION["angles"]["space"] == "semantic_v3"
    target = _DEFAULT_THUMB_ADDUCTION["angles"]["target"]
    assert "thumb_mcp_flex" in target
    assert "thumb_base" not in target
    p = _parse_thumb_adduction({}, "o20", HandConfig("o20"))
    assert p.thumb_hold_joints == [0, 1, 4, 3]
    assert p.thumb_rot_joint == SEMANTIC_30_IDX["thumb_rot"]
    assert p.progressive_abd_joint == 1


def test_default_static_is_semantic_v3() -> None:
    for name, block in _DEFAULT_STATIC.items():
        assert block["angles"]["space"] == "semantic_v3", name
    load_static_gesture_params.cache_clear()
    fist = load_static_gesture_params("o20", "fist")
    assert fist.phase1_hold == [0, 1, 4, 3]


def test_builtin_open_fist_are_s30() -> None:
    open_pose = _open_o20()
    fist_pose = _fist_o20()
    assert len(open_pose) == SEMANTIC_DIM
    assert len(fist_pose) == SEMANTIC_DIM
    assert open_pose[SEMANTIC_30_IDX["index_mcp_abd"]] == 128.0
    assert fist_pose[SEMANTIC_30_IDX["index_mcp_flex"]] == 255.0


def test_empty_raw_thumb_adduction_uses_s30_indices() -> None:
    p = _parse_thumb_adduction({}, "o20", HandConfig("o20"))
    assert p.thumb_rot_joint == SEMANTIC_30_IDX["thumb_rot"]
    assert p.thumb_hold_joints == [0, 1, 4, 3]
    assert p.progressive_abd_joint == 1
    assert p.progressive_flex_joint == 3
    assert len(p.prep_angles) == SEMANTIC_DIM


def test_fallback_static_without_yaml_target() -> None:
    """angles.target 为空时走内置 S30 builder。"""
    clear_gesture_params_cache()
    try:
        p = load_static_gesture_params("o20", "open")
        assert len(p.target_angles) == SEMANTIC_DIM
        ta = load_thumb_adduction_params("o20")
        assert ta.thumb_rot_joint == SEMANTIC_30_IDX["thumb_rot"]
    finally:
        clear_gesture_params_cache()
