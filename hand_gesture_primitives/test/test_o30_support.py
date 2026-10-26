"""O30 HOP 手型档案与 ros2_control 关节序。"""

import os

import pytest
from hand_gesture_primitives.hand_config import HandConfig, resolve_hand_config_key
from hand_gesture_primitives.hand_profile import load_hand_profile
from hand_gesture_primitives.joint_state_mapping import (
    O30_ACTIVE_SUFFIXES,
    extract_o30_from_joint_state,
    ros2_control_joint_names,
    ros2_control_semantic_indices,
)
from hand_gesture_primitives.primitive_base import HAND_CONFIGS, semantic_pose_config


def test_resolve_o30_config_key():
    assert resolve_hand_config_key("O30") == "o30"


def test_o30_yaml_loads():
    cfg = HandConfig("o30")
    assert cfg.num_joints == 20
    assert cfg.joint_names[0] == "thumb_roll"
    assert cfg.joint_names[6] == "thumb_root1"
    hw = cfg.to_hardware([0.0] * 20)
    assert len(hw) == 20
    semantic = [0.0] * 20
    semantic[0] = 200.0  # thumb_base
    hw2 = cfg.to_hardware(semantic)
    assert hw2[6] == 200.0


def test_o30_hand_profile():
    profile = load_hand_profile("o30")
    assert profile.hand_type == "o30"
    assert profile.hardware.num_joints == 20
    assert profile.is_primitive_supported("init")
    assert profile.is_primitive_supported("open")


def test_o30_primitive_hand_config():
    cfg = HAND_CONFIGS["o30"]
    assert cfg.num_joints == 20
    assert cfg.thumb_base == 0
    assert cfg.thumb_tip == 3
    assert cfg.thumb_rot == 4
    assert cfg.middle_root2 == 12
    pose = semantic_pose_config("o30")
    assert pose.num_joints == 30
    assert pose.middle_root2 == 12


def test_o30_joint_state_extract():
    pos = [float(i) for i in range(20)]
    names = [f"l_hand_{s}" for s in O30_ACTIVE_SUFFIXES]
    hw = extract_o30_from_joint_state(names, pos, "left")
    assert hw == pos
    assert ros2_control_joint_names("o30", "left") == names
    assert len(ros2_control_semantic_indices("o30")) == 20


def test_o30_pinch_and_parallel_extension_set_pip():
    """O30 独立 PIP：pinch/parallel_extension/index_pinch 须在 YAML 显式写 *_pip。"""
    from hand_gesture_primitives.gesture_params import load_static_gesture_params
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    load_static_gesture_params.cache_clear()
    try:
        idx = SEMANTIC_30_IDX
        pinch = load_static_gesture_params("o30", "pinch").target_angles
        assert pinch[idx["index_pip"]] == 210.0
        assert pinch[idx["thumb_rot"]] == 255.0
        assert pinch[idx["thumb_mcp_flex"]] == 200.0
        assert pinch[idx["thumb_dip"]] == 200.0

        pe = load_static_gesture_params("o30", "parallel_extension").target_angles
        for ch in (
            "index_pip",
            "middle_pip",
            "ring_pip",
            "pinky_pip",
            "index_dip",
            "middle_dip",
            "ring_dip",
            "pinky_dip",
        ):
            assert pe[idx[ch]] == 235.0, ch

        pev = load_static_gesture_params("o30", "parallel_extension_by_vision").target_angles
        for ch in ("index_pip", "index_dip", "pinky_pip", "pinky_dip"):
            assert pev[idx[ch]] == 235.0, ch

        disk = load_static_gesture_params("o30", "disk_by_vision").target_angles
        assert disk[idx["index_pip"]] == 130.0
        assert disk[idx["index_dip"]] == 130.0
        assert disk[idx["thumb_mcp_abd"]] == 255.0

        ir = load_static_gesture_params("o30", "index_ring_by_vision").target_angles
        assert ir[idx["index_pip"]] == 240.0
        assert ir[idx["index_dip"]] == 230.0
        assert ir[idx["thumb_rot"]] == 255.0

        lw = load_static_gesture_params("o30", "large_wrap_by_vision").target_angles
        assert lw[idx["middle_pip"]] == 210.0
        assert lw[idx["middle_dip"]] == 210.0

        ip = load_static_gesture_params("o30", "index_pinch").target_angles
        assert ip[idx["index_pip"]] == 240.0
        assert ip[idx["index_mcp_flex"]] == 150.0
        assert ip[idx["thumb_mcp_flex"]] == 200.0
        assert ip[idx["thumb_rot"]] == 255.0

        mp = load_static_gesture_params("o30", "middle_pinch").target_angles
        assert mp[idx["middle_pip"]] == 180.0
        assert mp[idx["middle_mcp_flex"]] == 200.0

        open_ = load_static_gesture_params("o30", "open").target_angles
        assert open_[idx["index_pip"]] == 0.0
        assert open_[idx["thumb_mcp_flex"]] == 0.0
    finally:
        load_static_gesture_params.cache_clear()


def test_o30_relax_grip_and_release_load():
    """O30 无 empty-target builder：relax_grip/release 须能解析到 open/fist。"""
    from hand_gesture_primitives.gesture_params import load_static_gesture_params
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    load_static_gesture_params.cache_clear()
    try:
        idx = SEMANTIC_30_IDX
        rg = load_static_gesture_params("o30", "relax_grip")
        assert rg.motion_type == "relax_grip"
        assert rg.relax_angles[idx["thumb_mcp_flex"]] == 0.0  # open
        assert rg.grip_angles[idx["index_mcp_flex"]] == 255.0  # fist
        assert rg.grip_phased is not None  # fist 为 phased_pose

        rel = load_static_gesture_params("o30", "release")
        op = load_static_gesture_params("o30", "open")
        assert rel.target_angles == op.target_angles
    finally:
        load_static_gesture_params.cache_clear()


def test_o30_ring_sfce_includes_index_pip():
    """长期修法：O30 ring 走 SFCE，食指闭合链含 mcp/pip/dip。"""
    from hand_gesture_primitives.gesture_params import (
        load_middle_ring_params,
        load_ring_params,
        load_sequential_force_close_params,
    )
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    load_sequential_force_close_params.cache_clear()

    ring = load_ring_params("o30")
    assert ring is not None
    assert ring.prep_angles[SEMANTIC_30_IDX["index_pip"]] == 30.0
    assert ring.prep_angles[SEMANTIC_30_IDX["thumb_rot"]] == 255.0
    assert len(ring.steps) == 2
    index_group = ring.steps[0].groups[0]
    assert index_group.group_id == "index"
    assert SEMANTIC_30_IDX["index_mcp_flex"] in index_group.joint_indices
    assert SEMANTIC_30_IDX["index_pip"] in index_group.joint_indices
    assert SEMANTIC_30_IDX["index_dip"] in index_group.joint_indices
    # monitor 为 HOP 硬件下标：index_root1=7, root2=11, tip=16
    assert set(index_group.monitor_hw) == {7, 11, 16}

    mid = load_middle_ring_params("o30")
    assert mid is not None
    mid_group = mid.steps[0].groups[0]
    assert SEMANTIC_30_IDX["middle_pip"] in mid_group.joint_indices

    rbv = load_sequential_force_close_params("o30", "ring_by_vision")
    assert rbv is not None
    assert SEMANTIC_30_IDX["index_pip"] in rbv.steps[0].groups[0].joint_indices

    for name in (
        "hook_by_vision",
        "small_warp_by_vision",
        "no_index_warp_by_vision",
        "palmar_by_vision",
        "tripod_by_vision",
    ):
        params = load_sequential_force_close_params("o30", name)
        assert params is not None, name
        # 第一个闭合组应含 pip（no_index 首组为 middle）
        first = params.steps[0].groups[0]
        assert any(
            SEMANTIC_30_IDX[ch] in first.joint_indices
            for ch in ("index_pip", "middle_pip", "ring_pip", "pinky_pip")
        ), name


def test_o30_twist_and_flick_sync_pip():
    """twist/flick：独立 PIP 须跟随 dip，避免 tip 弯而 root2 仍 0。"""
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.primitives.index_flick import IndexFlick
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX
    from hand_gesture_primitives.twist_poses import clear_twist_poses_cache, load_twist_poses

    clear_twist_poses_cache()
    idx = SEMANTIC_30_IDX
    for prim in ("twist2digit", "twist3digit", "twist5digit"):
        poses = load_twist_poses("o30", prim)
        assert poses.ready[idx["index_pip"]] == poses.ready[idx["index_dip"]], prim
        assert poses.end[idx["index_pip"]] == poses.end[idx["index_dip"]], prim

    ctx = PrimitiveContext(hand_type="o30")
    start = [0.0] * 30
    flick = IndexFlick()
    flick.on_enter(start)
    out = flick.compute(start, 0.6, ctx).target_angles
    assert out[idx["index_pip"]] == out[idx["index_dip"]]
    assert out[idx["middle_pip"]] == out[idx["middle_dip"]]


def test_o30_index_ring_progressive_keeps_pip():
    """precision 路径：渐进闭合后 index_pip 跟随 index_dip。"""
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.primitives.index_ring_by_vision import IndexRingByVision
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    idx = SEMANTIC_30_IDX
    ctx = PrimitiveContext(hand_type="o30", joint_currents=[0.0] * 20)
    prim = IndexRingByVision()
    start = [0.0] * 30
    prim.on_enter(start)
    # 越过 phase2 进入 progressive
    out = prim.compute(start, 1.2, ctx).target_angles
    assert prim.grasp_state == "progressive"
    assert out[idx["index_pip"]] == out[idx["index_dip"]]


def test_o30_grasp_ready_not_open():
    from hand_gesture_primitives.gesture_params import load_static_gesture_params
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    load_static_gesture_params.cache_clear()
    try:
        gr = load_static_gesture_params("o30", "grasp_ready").target_angles
        op = load_static_gesture_params("o30", "open").target_angles
        idx = SEMANTIC_30_IDX
        assert gr[idx["ring_mcp_flex"]] > 50.0
        assert gr != op
        tripod = load_static_gesture_params("o30", "tripod").target_angles
        assert tripod[idx["index_pip"]] == 184.0
        assert tripod[idx["middle_pip"]] == 220.0
        assert tripod[idx["thumb_mcp_abd"]] == 255.0
        assert tripod[idx["middle_mcp_flex"]] == 170.0
    finally:
        load_static_gesture_params.cache_clear()


def test_o30_missing_yaml_raises_not_open_fallback():
    from hand_gesture_primitives.gesture_params import load_static_gesture_params

    load_static_gesture_params.cache_clear()
    try:
        # allowlist 有、但无 angles.target 的静态原语仍禁止静默 O20 fallback
        with pytest.raises(ValueError, match="no YAML target"):
            load_static_gesture_params("o30", "index_flick")
    finally:
        load_static_gesture_params.cache_clear()


def test_o30_fist_drives_root2_pip():
    from hand_gesture_primitives.gesture_params import load_static_gesture_params
    from hand_gesture_primitives.semantic_30 import project_s30_to_o20

    params = load_static_gesture_params("o30", "fist")
    o20 = project_s30_to_o20(params.target_angles)
    assert o20[11] == 255.0
    assert o20[12] == 255.0
    hw = HandConfig("o30").to_hardware(o20)
    assert hw[11] == 255.0
    assert hw[12] == 255.0


def test_o6_rejects_vision_without_fk():
    profile = load_hand_profile("o6")
    reason = profile.reject_reason("tripod_by_vision")
    assert reason.startswith(("capability_missing:", "not_in_allowlist:", "capability_warning:"))


def test_o30_fist_hold_is_thumb_chain_s30():
    from hand_gesture_primitives.gesture_params import load_static_gesture_params

    load_static_gesture_params.cache_clear()
    try:
        for hand in ("o30", "l25", "o20"):
            params = load_static_gesture_params(hand, "fist")
            assert params.phase1_hold == [0, 1, 4, 3], (hand, params.phase1_hold)
    finally:
        load_static_gesture_params.cache_clear()


def test_o30_from_semantic_matches_from_o20_path():
    """P5：from_semantic 与 project+from_o20 对映射通道应一致。"""
    from hand_gesture_primitives.hand_config import _load_config_data
    from hand_gesture_primitives.semantic_30 import (
        SEMANTIC_30_IDX,
        SEMANTIC_DIM,
        project_o20_to_s30,
        project_s30_to_o20,
    )

    _load_config_data.cache_clear()
    cfg = HandConfig("o30")
    assert cfg.has_from_semantic

    o20 = [float(i * 10 % 256) for i in range(20)]
    s30 = project_o20_to_s30(o20)
    # thumb_pip 无电机：coupling ignore，不应影响硬件
    s30[SEMANTIC_30_IDX["thumb_pip"]] = 200.0

    hw_direct = cfg.from_semantic(s30)
    hw_compat = cfg.to_hardware(project_s30_to_o20(s30))
    assert hw_direct == hw_compat
    assert len(hw_direct) == 20

    back = cfg.to_semantic(hw_direct)
    assert len(back) == SEMANTIC_DIM
    for ch in cfg.enabled_channel_names():
        i = SEMANTIC_30_IDX[ch]
        assert abs(back[i] - s30[i]) < 1e-6, (ch, back[i], s30[i])
    assert back[SEMANTIC_30_IDX["thumb_pip"]] == 0.0


def test_o30_finger_abd_invert_for_v_sign_polarity():
    """四指 yaw 与 S30 侧摆约定相反：v_sign middle 低语义 → 高 HW。"""
    from hand_gesture_primitives.gesture_params import load_static_gesture_params
    from hand_gesture_primitives.hand_config import _load_config_data
    from hand_gesture_primitives.joint_state_mapping import (
        Ros2ControlRadianConverter,
        o30_single_hand_stub_urdf,
    )
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    _load_config_data.cache_clear()
    load_static_gesture_params.cache_clear()
    cfg = HandConfig("o30")
    names = cfg.joint_names

    vs = load_static_gesture_params("o30", "v_sign").target_angles
    hw = cfg.from_semantic(vs)
    # 语义 index=200 / middle=60 → invert 后 HW 55 / 195
    assert abs(hw[names.index("index_yaw")] - (255.0 - 200.0)) < 1e-6
    assert abs(hw[names.index("middle_yaw")] - (255.0 - 60.0)) < 1e-6
    # 拇指 yaw 不 invert
    assert abs(hw[names.index("thumb_yaw")] - vs[SEMANTIC_30_IDX["thumb_mcp_abd"]]) < 1e-6

    conv = Ros2ControlRadianConverter(o30_single_hand_stub_urdf("right"), "o30", "right")
    assert SEMANTIC_30_IDX["index_mcp_abd"] in conv.inverted_indices
    assert SEMANTIC_30_IDX["middle_mcp_abd"] in conv.inverted_indices
    # radians 路径：语义变小 → URDF 靠近 upper（与 HW invert 同向）
    mid = SEMANTIC_30_IDX["middle_mcp_abd"]
    rad60 = conv.legacy255_to_semantic_rad([60.0 if i == mid else 128.0 for i in range(30)])[mid]
    rad200 = conv.legacy255_to_semantic_rad([200.0 if i == mid else 128.0 for i in range(30)])[mid]
    assert rad60 > rad200  # invert：legacy 低 → rad 高


def test_o30_channels_and_allowlist():
    from hand_gesture_primitives.hand_config import _load_config_data

    _load_config_data.cache_clear()
    cfg = HandConfig("o30")
    assert not cfg.channel_enabled("thumb_pip")
    assert cfg.channel_enabled("index_pip")
    assert cfg.channel_enabled("thumb_mcp_flex")

    profile = load_hand_profile("o30")
    for name in (
        "ok_sign",
        "v_sign",
        "grasp_ready",
        "index_pinch",
        "middle_pinch",
        "parallel_extension",
        "thumb_adduction_grip",
        "index_middle_adduction_grip",
        "tripod",
        "tripod_by_force",
        "ring",
        "middle_ring",
        "index_flick",
        "index_middle_flick",
        "twist2digit",
        "twist3digit",
        "twist5digit",
    ):
        assert profile.is_primitive_supported(name), name
        assert profile.reject_reason(name) == "", name

    assert profile.primitive_for_grasp_type("lateral") == "index_middle_adduction_grip"
    assert profile.primitive_for_grasp_type("precision") == "index_ring_by_vision"
    assert profile.primitive_for_grasp_type("power") == "large_wrap_by_vision"

    for vision in (
        "tripod_by_vision",
        "index_ring_by_vision",
        "large_wrap_by_vision",
        "disk_by_vision",
    ):
        assert profile.is_primitive_supported(vision), vision
        assert profile.reject_reason(vision) == "", vision


def test_o30_fk_solver_uses_o30_mesh():
    """O30 FK 使用 evo_work/o30 mesh；pinocchio / URDF 不可用时跳过。"""
    from hand_gesture_primitives.fk_solver import (
        _PINOCCHIO_AVAILABLE,
        _load_fk_config,
        create_fk_solver,
    )
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM
    from hand_gesture_primitives.urdf_utils import resolve_urdf_path

    fk_cfg = _load_fk_config("o30")
    assert fk_cfg is not None
    assert fk_cfg.get("urdf_model") == "o30"
    assert fk_cfg.get("fold_pip_into_dip") is False
    assert "semantic_to_urdf_joint" in fk_cfg

    try:
        left_urdf = resolve_urdf_path("O30", "left")
    except FileNotFoundError:
        pytest.skip("O30 URDF 未部署（evo_work/o30 缺失）")
    assert "o30" in left_urdf.replace("\\", "/").lower()
    assert os.path.isfile(left_urdf)

    if not _PINOCCHIO_AVAILABLE:
        return

    solver = create_fk_solver("O30", "left")
    assert solver is not None
    assert "o30" in solver.urdf_path.replace("\\", "/").lower()
    tips0 = solver.compute_fingertips([0.0] * 20)
    assert tips0.shape == (5, 3)
    # S30：仅动 index_pip 应改变食指尖（O30 URDF 有独立 PIP）
    s30 = [0.0] * SEMANTIC_DIM
    tips_pip = solver.compute_fingertips(s30)
    s30[SEMANTIC_30_IDX["index_pip"]] = 200.0
    tips_pip2 = solver.compute_fingertips(s30)
    assert (
        abs(float(tips_pip2[1, 2] - tips_pip[1, 2])) > 1e-4
        or abs(float(tips_pip2[1, 0] - tips_pip[1, 0])) > 1e-4
    )
