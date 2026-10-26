"""L25 static by_vision 须读取 l25.yaml，而非旁路硬编码。"""

from hand_gesture_primitives.gesture_params import (
    _fist_o20,
    _grasp_ready_l25,
    _index_pinch_l25,
    _middle_pinch_l25,
    _ok_sign_l25,
    _pinch_l25,
    _semantic_from_sparse,
    _tripod_l25,
    load_finger_adduction_params,
    load_static_gesture_params,
    static_yaml_legacy255_or_none,
)
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitives import PRIMITIVE_REGISTRY
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM

_I = SEMANTIC_30_IDX
_ZERO = [0.0] * SEMANTIC_DIM


def test_l25_static_yaml_targets_nonzero() -> None:
    for name in (
        "disk_by_vision",
        "index_pinch_by_vision",
        "middle_pinch_by_vision",
        "parallel_extension_by_vision",
        "parallel_extension",
        "index_ring_by_vision",
        "large_wrap_by_vision",
    ):
        yaml255 = static_yaml_legacy255_or_none("l25", name)
        assert yaml255 is not None, name
        assert any(v > 0 for v in yaml255), name


def test_l25_disk_by_vision_uses_yaml_target() -> None:
    params = load_static_gesture_params("l25", "disk_by_vision")
    prim = PRIMITIVE_REGISTRY["disk_by_vision"]()
    prim.on_enter(_ZERO)
    result = prim.compute(
        _ZERO,
        elapsed=10.0,
        ctx=PrimitiveContext(hand_type="l25"),
    )
    assert result.feasible
    assert result.space == "semantic"  # 无 Converter → legacy255
    assert result.target_angles == list(params.target_angles)


def test_l25_index_pinch_by_vision_uses_calibrated_builder() -> None:
    load_static_gesture_params.cache_clear()
    params = load_static_gesture_params("l25", "index_pinch_by_vision")
    assert params.target_angles == _index_pinch_l25()
    prim = PRIMITIVE_REGISTRY["index_pinch_by_vision"]()
    prim.on_enter(_ZERO)
    result = prim.compute(
        _ZERO,
        elapsed=10.0,
        ctx=PrimitiveContext(hand_type="l25"),
    )
    assert result.feasible
    assert result.target_angles == list(params.target_angles)


def test_l25_index_ring_by_vision_yaml_base() -> None:
    params = load_static_gesture_params("l25", "index_ring_by_vision")
    assert params.target_angles[_I["thumb_mcp_flex"]] == 200.0
    assert params.target_angles[_I["index_mcp_flex"]] == 130.0
    assert params.target_angles[_I["thumb_rot"]] == 120.0


def test_open_hand_initial_result_space_is_semantic_rad() -> None:
    prim = PRIMITIVE_REGISTRY["open"]()
    assert getattr(prim, "_result_space") == "semantic_rad"


def test_l25_init_and_open_targets_differ() -> None:
    init = load_static_gesture_params("l25", "init").target_angles
    open_ = load_static_gesture_params("l25", "open").target_angles
    assert init != open_
    assert init[_I["thumb_mcp_flex"]] == 80.0
    assert open_[_I["thumb_mcp_flex"]] == 0.0


def test_l25_contact_gestures_use_calibrated_builders() -> None:
    """对指类空 target 须回退 builder，避免 O20 弱半弯被 YAML 盖掉。"""
    load_static_gesture_params.cache_clear()
    cases = (
        ("pinch", _pinch_l25),
        ("index_pinch", _index_pinch_l25),
        ("index_pinch_by_vision", _index_pinch_l25),
        ("middle_pinch", _middle_pinch_l25),
        ("middle_pinch_by_vision", _middle_pinch_l25),
        ("ok_sign", _ok_sign_l25),
        ("tripod", _tripod_l25),
        ("tripod_by_vision", _tripod_l25),
    )
    for name, builder in cases:
        got = load_static_gesture_params("l25", name).target_angles
        assert got == builder(), name

    # 关键关节：语义深弯，L25 invert 后才是对指位
    assert load_static_gesture_params("l25", "pinch").target_angles[_I["index_mcp_flex"]] == 255.0
    assert (
        load_static_gesture_params("l25", "index_pinch").target_angles[_I["index_mcp_flex"]]
        == 255.0
    )
    assert (
        load_static_gesture_params("l25", "middle_pinch").target_angles[_I["middle_mcp_flex"]]
        == 255.0
    )
    assert (
        load_static_gesture_params("l25", "ok_sign").target_angles[_I["index_mcp_flex"]] == 255.0
    )
    assert load_static_gesture_params("l25", "tripod").target_angles[_I["thumb_mcp_flex"]] == 128.0
    assert load_static_gesture_params("l25", "tripod").target_angles[_I["index_mcp_flex"]] == 230.0
    assert (
        load_static_gesture_params("l25", "tripod").target_angles[_I["middle_mcp_flex"]] == 230.0
    )
    # ok_sign 须深于 O20 通用姿（L25 invert 下 150 过浅）
    assert load_static_gesture_params("l25", "ok_sign").target_angles[_I["index_mcp_flex"]] > 150.0


def test_l25_pinch_family_thumb_rot_within_natural_range() -> None:
    """L25 对指：thumb_rot 须在 init 与极限旋转之间（hw 约 100~165）。"""
    from hand_gesture_primitives.hand_config import HandConfig

    load_static_gesture_params.cache_clear()
    hand = HandConfig("l25")
    init_rot_hw = hand.from_semantic(load_static_gesture_params("l25", "init").target_angles)[10]
    for name in ("pinch", "index_pinch", "middle_pinch"):
        rot_hw = hand.from_semantic(load_static_gesture_params("l25", name).target_angles)[10]
        assert 100.0 <= rot_hw <= init_rot_hw - 40.0, f"{name} hw_rot={rot_hw}"
    ok_rot_hw = hand.from_semantic(load_static_gesture_params("l25", "ok_sign").target_angles)[10]
    assert 95.0 <= ok_rot_hw <= init_rot_hw - 35.0


def test_l25_index_pinch_opposition_joints_closed() -> None:
    """index_pinch 参与对指的 flex 须在 L25 invert 下深弯 (hw 低)。"""
    from hand_gesture_primitives.hand_config import HandConfig

    load_static_gesture_params.cache_clear()
    hand = HandConfig("l25")
    hw = hand.from_semantic(load_static_gesture_params("l25", "index_pinch").target_angles)
    assert hw[1] <= 20.0  # index_base
    assert hw[0] <= 140.0  # thumb_base
    assert hw[2] >= 200.0  # middle_base open


def test_semantic_from_sparse_preserves_explicit_abd_zero() -> None:
    """侧捏 prep 的 middle_abd: 0 不能被 ABD_NEUTRAL 覆盖。"""
    prep = _semantic_from_sparse({"index_abd": 255, "middle_abd": 0})
    assert prep[_I["index_mcp_abd"]] == 255.0
    assert prep[_I["middle_mcp_abd"]] == 0.0


def test_l25_index_middle_adduction_symmetric_abd_travel() -> None:
    """P2 并拢时食/中 abd 行程应对称（修复前 middle 被误填 128，仅动 32）。"""
    load_finger_adduction_params.cache_clear()
    p = load_finger_adduction_params("l25")
    assert p.prep_angles[_I["middle_mcp_abd"]] == 0.0
    assert (
        abs((p.prep_angles[p.joint_a] - p.close_a) - (p.close_b - p.prep_angles[p.joint_b])) <= 2.0
    )


def test_l25_tripod_index_middle_symmetric_hw() -> None:
    from hand_gesture_primitives.hand_config import HandConfig

    load_static_gesture_params.cache_clear()
    hand = HandConfig("l25")
    hw = hand.from_semantic(load_static_gesture_params("l25", "tripod").target_angles)
    assert abs(hw[1] - hw[2]) <= 5.0
    assert abs(hw[16] - hw[17]) <= 5.0


def test_l25_planable_factory_covers_builders() -> None:
    """L25 YAML 桩齐全后，出厂导出应覆盖全部 planable（含 fist/point 等）。"""
    from hand_gesture_primitives.export_primitive_poses import (
        export_primitive_poses,
        resolve_primitive_list,
    )
    from hand_gesture_primitives.hand_config import HandConfig
    from hand_gesture_primitives.planable_primitives import PLANABLE_STATIC_PRIMITIVES
    from hand_gesture_primitives.pose_library import pose_entry_to_hardware

    load_static_gesture_params.cache_clear()
    prims = resolve_primitive_list(hand_type="l25")
    assert set(prims) == PLANABLE_STATIC_PRIMITIVES

    hand = HandConfig("l25")
    pinch_sem = load_static_gesture_params("l25", "pinch").target_angles
    assert hand.from_semantic(pinch_sem)[1] == 0.0  # semantic 255 → L25 invert

    expect_fist = hand.from_semantic(_fist_o20())
    hw_fist_entry = load_static_gesture_params("l25", "fist")
    assert hand.from_semantic(hw_fist_entry.target_angles) == expect_fist

    expect_gr = hand.from_semantic(_grasp_ready_l25())
    hw_gr_entry = load_static_gesture_params("l25", "grasp_ready")
    assert hand.from_semantic(hw_gr_entry.target_angles) == expect_gr

    import os
    import tempfile
    from unittest import mock

    from hand_gesture_primitives import pose_library as pl

    with tempfile.TemporaryDirectory() as td:
        with mock.patch.dict(os.environ, {"HAND_GESTURE_POSE_DIR": td}):
            count, _, names = export_primitive_poses(
                "l25",
                "left",
                set_name="planable",
                write_library=True,
                overwrite=True,
            )
            assert count == len(PLANABLE_STATIC_PRIMITIVES)
            assert "preset_fist" in names
            assert "preset_pinch" in names
            entry = pl.load_pose("l25", "preset_pinch", hand_side="left")
            assert entry is not None
            hw = pose_entry_to_hardware(entry, "l25")
            assert hw is not None
            assert hw[1] == 0.0
