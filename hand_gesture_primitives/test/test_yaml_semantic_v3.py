"""YAML gestures 已迁 semantic_v3；加载结果仍为 S30。"""

import yaml
from hand_gesture_primitives.gesture_params import (
    load_static_gesture_params,
    load_thumb_adduction_params,
)
from hand_gesture_primitives.hand_config import _find_config_path
from hand_gesture_primitives.primitive_base import SemanticPoseConfig, semantic_pose_config
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM


def test_o20_l25_gestures_are_semantic_v3() -> None:
    for hand in ("o20", "l25"):
        path = _find_config_path(hand)
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        for name, block in (raw.get("gestures") or {}).items():
            angles = (block or {}).get("angles") or {}
            if not angles:
                continue
            space = str(angles.get("space", "")).lower()
            assert space in ("", "semantic_v3", "hardware", "semantic_o20"), (hand, name, space)
            target = angles.get("target") or {}
            if space == "semantic_v3":
                for key in target:
                    assert "thumb_base" not in str(key), (hand, name, key)
                    assert key not in (
                        "thumb_base",
                        "index_base",
                        "thumb_tip",
                        "index_tip",
                    ), (hand, name, key)


def test_o20_thumb_adduction_indices_are_s30() -> None:
    load_thumb_adduction_params.cache_clear()
    try:
        p = load_thumb_adduction_params("o20")
        assert p.thumb_rot_joint == SEMANTIC_30_IDX["thumb_rot"]
        assert SEMANTIC_30_IDX["thumb_mcp_flex"] in p.thumb_hold_joints
        assert SEMANTIC_30_IDX["thumb_mcp_abd"] in p.thumb_hold_joints
        assert len(p.prep_angles) == SEMANTIC_DIM
    finally:
        load_thumb_adduction_params.cache_clear()


def test_l25_init_loads_s30() -> None:
    load_static_gesture_params.cache_clear()
    try:
        p = load_static_gesture_params("l25", "init")
        assert len(p.target_angles) == SEMANTIC_DIM
        assert p.target_angles[SEMANTIC_30_IDX["index_mcp_abd"]] == 128.0
    finally:
        load_static_gesture_params.cache_clear()


def test_all_hands_have_yaml_from_semantic() -> None:
    from hand_gesture_primitives.hand_config import HandConfig, _load_config_data

    _load_config_data.cache_clear()
    for ht in ("o20", "l25", "g20", "o6", "l6", "o30"):
        h = HandConfig(ht)
        assert h.has_from_semantic, ht
        assert h._semantic_map_source == "yaml", (ht, h._semantic_map_source)
        assert len(h._semantic_map) >= 6, ht


def test_semantic_pose_config_alias() -> None:
    cfg = semantic_pose_config("o30")
    assert isinstance(cfg, SemanticPoseConfig)
    assert cfg.index_pip == 7
