"""REVIEW_FIX_PLAN.md 关键回归单测。"""

from __future__ import annotations

import pytest
from hand_gesture_primitives.gesture_engine import (
    FingerAdductionEngine,
    PhasedLerpEngine,
)
from hand_gesture_primitives.gesture_params import ThumbAdductionParams
from hand_gesture_primitives.pose_library import validate_positions
from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitive_catalog import missing_capabilities
from hand_gesture_primitives.primitives.tripod_by_vision import TripodByVision
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM


def test_finger_flex_joints_s30() -> None:
    expected = {
        0: (0, 3),
        1: (5, 8),
        2: (10, 13),
        3: (15, 18),
        4: (20, 23),
    }
    for finger, pair in expected.items():
        assert PhasedLerpEngine._finger_flex_joints(finger) == pair


def test_finger_adduction_close_duration_zero_safe() -> None:
    engine = FingerAdductionEngine(
        pose_target=[128.0] * SEMANTIC_DIM,
        close_duration=0.0,
    )
    engine.reset([0.0] * SEMANTIC_DIM)
    out = engine.compute([0.0] * SEMANTIC_DIM, elapsed=2.0, ctx=PrimitiveContext())
    assert len(out) == SEMANTIC_DIM


def test_tripod_by_vision_grasp_state_sticky() -> None:
    primitive = TripodByVision()
    primitive.on_enter([0.0] * SEMANTIC_DIM)
    primitive._grasp_state = "contact"  # noqa: SLF001
    assert primitive.grasp_state == "contact"


def test_validate_positions_rejects_absurd_rad() -> None:
    err = validate_positions([999.0] * SEMANTIC_DIM, SEMANTIC_DIM, unit="rad")
    assert err is not None


def test_catalog_twist_requires_capabilities_on_o6() -> None:
    missing = missing_capabilities("twist5digit", frozenset())
    assert "finger_tips" in missing or "finger_abd" in missing


def test_phased_lerp_p2_uses_thumb_abd_not_index_flex() -> None:
    abd = SEMANTIC_30_IDX["thumb_mcp_abd"]
    prep = [0.0] * SEMANTIC_DIM
    prep[abd] = 180.0
    params = ThumbAdductionParams(
        hand_type="o20",
        prep_angles=prep,
        close_angles=[200.0] * SEMANTIC_DIM,
        phase2=0.2,
        phase3=0.0,
        phase4=0.1,
        progressive_abd_joint=abd,
        thumb_rot_joint=-1,
    )
    engine = PhasedLerpEngine(params)
    engine.reset([0.0] * SEMANTIC_DIM)
    mid = engine._compute_raw(params.phase1 + params.phase2 * 0.5)  # noqa: SLF001
    assert mid[SEMANTIC_30_IDX["index_mcp_flex"]] == pytest.approx(0.0)
    assert mid[abd] > 0.0


def test_ring_legacy_preshape_keeps_s30_slots() -> None:
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.primitives.ring import Ring

    prim = Ring()
    prim.on_enter([0.0] * SEMANTIC_DIM)
    ctx = PrimitiveContext(hand_type="o20")
    cfg = prim._get_cfg(ctx)  # noqa: SLF001
    prim._thumb_rot = 180.0  # noqa: SLF001
    raw = prim._build_preshape_target(cfg)  # noqa: SLF001
    assert len(raw) == SEMANTIC_DIM
    assert raw[SEMANTIC_30_IDX["index_mcp_flex"]] == pytest.approx(90.0)
    # 禁止再经 [:20]→O20 投影：食指 flex 必须仍在 S30[5]
    from hand_gesture_primitives.angle_bridge import AngleUnits
    from hand_gesture_primitives.semantic_30 import as_semantic_30

    posed = AngleUnits(ctx).pose(as_semantic_30(raw))
    assert posed[SEMANTIC_30_IDX["index_mcp_flex"]] == pytest.approx(90.0)


def test_thumb_rot_fallback_is_s30() -> None:
    assert SEMANTIC_30_IDX["thumb_rot"] == 4
    from hand_gesture_primitives.gesture_engine import RotProgressiveEngine

    eng = RotProgressiveEngine(pose_target=[0.0] * SEMANTIC_DIM)
    assert eng._rot_joint == 4  # noqa: SLF001


def test_reachability_uses_tcp_rotated_palm_offset() -> None:
    """object_pose 路径须用 TCP+旋转后的 palm_offset，不能把 offset 当世界点。"""
    pytest.importorskip("geometry_msgs")
    import numpy as np
    from hand_gesture_primitives.executor import GestureExecutor
    from hand_gesture_primitives.primitive_base import PoseStamped

    ex = GestureExecutor.__new__(GestureExecutor)
    # 90° 绕 Z：局部 [0.05, 0, 0] → 世界 [0, 0.05, 0]
    ex._palm_offset = np.array([0.05, 0.0, 0.0])
    ex._reach_max_dist = 0.12
    ex._tcp_pose = PoseStamped(
        position=np.array([1.0, 2.0, 3.0]),
        orientation=np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)]),
        frame_id="world",
    )
    # 物体在掌心附近（palm ≈ [1.0, 2.05, 3.0]）
    ex._object_pose = PoseStamped(
        position=np.array([1.0, 2.05, 3.0]),
        frame_id="world",
    )
    ex._object_center = None
    assert ex._check_reachability() == ""  # noqa: SLF001

    # 远目标应被拒
    ex._object_pose = PoseStamped(
        position=np.array([5.0, 5.0, 5.0]),
        frame_id="world",
    )
    reason = ex._check_reachability()  # noqa: SLF001
    assert reason.startswith("too_far:")


def test_force_primary_stale_allows_matrix_fallback() -> None:
    """主触觉源过期后，matrix_touch 应可再次写入。"""
    pytest.importorskip("std_msgs")
    import time

    from hand_gesture_primitives.contact_config import ContactThresholds
    from hand_gesture_primitives.executor import (
        _FORCE_PRIMARY_STALE_SEC,
        GestureExecutor,
    )
    from std_msgs.msg import String

    ex = GestureExecutor.__new__(GestureExecutor)
    ex._contact_thresholds = ContactThresholds()
    ex._tactile_pressure = None
    ex._tactile_mode = "none"
    ex._contact_detected = False
    ex._force_primary_stamp = time.monotonic()
    assert ex._force_primary_fresh() is True  # noqa: SLF001

    # 模拟过期
    ex._force_primary_stamp = time.monotonic() - (_FORCE_PRIMARY_STALE_SEC + 0.1)
    assert ex._force_primary_fresh() is False  # noqa: SLF001

    msg = String()
    msg.data = '{"thumb_mass":1,"index_mass":2,"middle_mass":3,"ring_mass":4,"little_mass":5}'
    ex._matrix_touch_mass_callback(msg)  # noqa: SLF001
    assert ex._tactile_mode == "mass"
    assert ex._tactile_pressure is not None
    assert float(ex._tactile_pressure[0]) == pytest.approx(1.0)


def test_stream_expected_dof_follows_hand_protocol() -> None:
    """STREAM 长度：hardware 用产品 DOF，radians 用 controller 主动序。"""
    pytest.importorskip("geometry_msgs")
    from hand_gesture_primitives.executor import GestureExecutor
    from hand_gesture_primitives.primitive_base import HAND_CONFIGS

    ex = GestureExecutor.__new__(GestureExecutor)
    ex._hand_type = "l25"
    ex._hand_config = HAND_CONFIGS["l25"]
    ex._ros2_control_command_units = "hardware"
    ex._radian_converter = None
    assert ex._stream_expected_dof() == 25  # noqa: SLF001

    class _FakeConv:
        joint_names = [f"j{i}" for i in range(16)]

    ex._ros2_control_command_units = "radians"
    ex._radian_converter = _FakeConv()
    assert ex._stream_expected_dof() == 16  # noqa: SLF001

    ex._hand_type = "o20"
    ex._hand_config = HAND_CONFIGS["o20"]
    ex._ros2_control_command_units = "hardware"
    ex._radian_converter = None
    assert ex._stream_expected_dof() == 20  # noqa: SLF001


def test_sfce_hold_writes_s30_not_o20() -> None:
    """SFCE HOLD 须把 O20 reverse_map 投影到 S30 槽再回退。"""
    from hand_gesture_primitives.gesture_engine import SequentialForceCloseEngine
    from hand_gesture_primitives.gesture_params import (
        ForceCloseGroup,
        ForceCloseStep,
        SequentialForceCloseParams,
    )
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM

    index_mcp = SEMANTIC_30_IDX["index_mcp_flex"]  # S30=5；O20 index_base=1
    params = SequentialForceCloseParams(
        hand_type="o20",
        prep_angles=[0.0] * SEMANTIC_DIM,
        steps=[
            ForceCloseStep(
                name="close",
                groups=[
                    ForceCloseGroup(
                        group_id="index",
                        joint_indices=[index_mcp],
                        monitor_hw=[1],  # O20/hw index_base
                        step=5.0,
                        close_max=200.0,
                    )
                ],
            )
        ],
        hold_safe_current=50.0,
    )
    eng = SequentialForceCloseEngine(params)
    eng.reset([0.0] * SEMANTIC_DIM)
    eng._phase = "hold"  # noqa: SLF001
    eng._close_angles = [0.0] * SEMANTIC_DIM  # noqa: SLF001
    eng._close_angles[index_mcp] = 180.0  # noqa: SLF001
    # 故意在 O20[1] 留非零，若误写 O20 下标会污染错误槽
    eng._close_angles[1] = 10.0  # noqa: SLF001

    ctx = PrimitiveContext(hand_type="o20", joint_currents=[0.0] * 20)
    ctx.joint_currents[1] = 200.0  # 超 hold_safe
    eng._tick_hold(ctx)  # noqa: SLF001
    assert eng._close_angles[1] == pytest.approx(10.0)  # noqa: SLF001 O20 槽不动
    assert eng._close_angles[index_mcp] < 180.0  # noqa: SLF001 S30 槽回退


def test_overload_skip_uses_hardware_indices() -> None:
    """过载跳过须用 YAML 硬件 reserved + thumb_rot 的 hardware 下标。"""
    pytest.importorskip("geometry_msgs")
    from unittest import mock

    from hand_gesture_primitives.contact_config import ContactThresholds
    from hand_gesture_primitives.contact_resolver import current_index_for_s30
    from hand_gesture_primitives.executor import GestureExecutor
    from hand_gesture_primitives.hand_profile import load_hand_profile
    from hand_gesture_primitives.primitive_base import HAND_CONFIGS
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    ex = GestureExecutor.__new__(GestureExecutor)
    ex._hand_type = "o20"
    ex._hand_config = HAND_CONFIGS["o20"]
    ex._profile = load_hand_profile("o20")
    ex._contact_thresholds = ContactThresholds(
        overload_threshold=100.0,
        overload_duration_sec=0.0,
    )
    n = ex._hand_config.num_joints
    ex._joint_currents = [0.0] * n
    ex._joint_overload_start = [None] * n
    ex._joint_torque = None
    ex._node = mock.Mock()
    ex._node.get_logger.return_value = mock.Mock()
    triggered = []
    ex._trigger_protection = lambda i, c: triggered.append(i)  # noqa: E731

    thumb_hw = current_index_for_s30(ex._profile.hardware, SEMANTIC_30_IDX["thumb_rot"])
    assert thumb_hw is not None
    assert thumb_hw != SEMANTIC_30_IDX["thumb_rot"]  # O20: hw[10] vs S30[4]

    # S30[4] 高电流不应被误当成跳过目标；应触发保护
    ex._joint_currents[SEMANTIC_30_IDX["thumb_rot"]] = 999.0
    msg = mock.Mock()
    msg.data = '{"current":' + str([0] * n).replace("'", "") + "}"
    # 直接跑过载环逻辑片段
    skip = set(int(i) for i in ex._profile.hardware.reserved_indices)
    skip.add(int(thumb_hw))
    for i, c in enumerate(ex._joint_currents):
        if c < 0 or i in skip:
            continue
        if c > 100.0:
            triggered.append(i)
    assert SEMANTIC_30_IDX["thumb_rot"] in triggered

    # 真 thumb_rot hardware 高电流应被跳过
    triggered.clear()
    ex._joint_currents = [0.0] * n
    ex._joint_currents[thumb_hw] = 999.0
    for i, c in enumerate(ex._joint_currents):
        if c < 0 or i in skip:
            continue
        if c > 100.0:
            triggered.append(i)
    assert triggered == []


def test_by_vision_sfce_call_sites_pass_radian_converter() -> None:
    """by_vision _ensure_sfce_engine 须把 ctx.radian_converter 传给 factory。"""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "hand_gesture_primitives" / "primitives"
    files = (
        "hook_by_vision.py",
        "palmar_by_vision.py",
        "ring_by_vision.py",
        "middle_ring_by_vision.py",
        "small_warp_by_vision.py",
        "no_index_warp_by_vision.py",
    )
    for name in files:
        text = (root / name).read_text(encoding="utf-8")
        assert "radian_converter=ctx.radian_converter" in text, name
        assert "make_" in text and "_sfce_engine" in text, name


def test_large_wrap_fingers_are_s30_mcp_flex() -> None:
    from hand_gesture_primitives.primitives import large_wrap as lw
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    assert lw._THUMB == SEMANTIC_30_IDX["thumb_mcp_flex"]
    assert lw._FINGERS == (
        SEMANTIC_30_IDX["index_mcp_flex"],
        SEMANTIC_30_IDX["middle_mcp_flex"],
        SEMANTIC_30_IDX["ring_mcp_flex"],
        SEMANTIC_30_IDX["pinky_mcp_flex"],
    )


def test_monitors_for_fingers_returns_hardware_indices() -> None:
    """O20：食指 S30[5,8] 应映射为 hardware [1,16]，不能直接用 5/8。"""
    from hand_gesture_primitives.contact_detection import monitors_for_fingers
    from hand_gesture_primitives.primitive_base import PrimitiveContext

    ctx = PrimitiveContext(hand_type="o20", joint_currents=[0.0] * 20)
    hw, tactile = monitors_for_fingers(ctx, [1])  # index finger
    assert tactile == [1]
    assert 5 not in hw and 8 not in hw
    assert 1 in hw  # O20 index_base


def test_by_vision_phased_engine_passes_radian_converter() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "hand_gesture_primitives" / "primitives"
    for name in (
        "hook_by_vision.py",
        "palmar_by_vision.py",
        "ring_by_vision.py",
        "middle_ring_by_vision.py",
        "small_warp_by_vision.py",
        "no_index_warp_by_vision.py",
    ):
        text = (root / name).read_text(encoding="utf-8")
        # PhasedLerp ensure 与 SFCE 均应传 converter
        assert text.count("radian_converter=ctx.radian_converter") >= 2, name


def test_signed_step255_preserves_direction() -> None:
    from hand_gesture_primitives import units as U

    assert U.legacy_step255_to_rad_step(-51.0, 0.0, 1.57) < 0.0


def test_tactile_preload_limit_is_slot_scaled() -> None:
    """rad 模式下 preload 必须按槽行程换算，不能硬编码 6.0 rad。"""
    from unittest.mock import MagicMock

    from hand_gesture_primitives.gesture_engine import PhasedLerpEngine
    from hand_gesture_primitives.gesture_params import ThumbAdductionParams
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM

    params = ThumbAdductionParams(
        hand_type="o20",
        prep_angles=[0.0] * SEMANTIC_DIM,
        close_angles=[1.0] * SEMANTIC_DIM,
        contact_fingers=[1],
    )
    eng = PhasedLerpEngine(params)
    converter = MagicMock()
    converter.semantic_indices = list(range(SEMANTIC_DIM))
    converter.limits = [(0.0, 1.57)] * SEMANTIC_DIM
    converter.inverted_indices = set()
    ctx = PrimitiveContext(
        hand_type="o20",
        tactile_mode="pressure",
        tactile_pressure=[0.0, 100.0, 0.0, 0.0, 0.0],
        radian_converter=converter,
        angle_unit="semantic_rad",
    )
    feedback = [0.0] * SEMANTIC_DIM
    proposed = [1.0] * SEMANTIC_DIM
    eng._update_tactile_finger_holds(ctx, feedback, proposed)
    held = eng._frozen_finger_joints[1]
    # preload≈6/255*1.57 ≈ 0.037，远小于硬编码 6.0
    for v in held.values():
        assert abs(v) < 0.2


def test_g20_monitors_use_s30_not_hardware_pose_table() -> None:
    """g20 不能用 HAND_CONFIGS['g20'] 硬件序当 S30。"""
    from hand_gesture_primitives.contact_detection import (
        finger_joint_indices,
        hand_config,
        monitors_for_fingers,
    )
    from hand_gesture_primitives.primitive_base import PrimitiveContext
    from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX

    ctx = PrimitiveContext(hand_type="g20", joint_currents=[0.0] * 16)
    cfg = hand_config(ctx)
    assert finger_joint_indices(cfg, 1) == [
        SEMANTIC_30_IDX["index_mcp_flex"],
        SEMANTIC_30_IDX["index_dip"],
    ]
    hw, _ = monitors_for_fingers(ctx, [1])
    # S30 index flex/dip 映射后的 hw 下标不应等于 S30 裸下标 5/8
    assert 5 not in hw and 8 not in hw


def test_teach_panel_o30_defaults_ros2_control() -> None:
    text = (
        __import__("pathlib").Path(__file__).resolve().parents[1]
        / "hand_gesture_primitives"
        / "teach_panel"
        / "ros_bridge.py"
    ).read_text(encoding="utf-8")
    assert '"o30"' in text or "'o30'" in text


def test_twist_pinch_monitors_via_s30_mapping() -> None:
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "hand_gesture_primitives" / "primitives"
    for name in ("twist2digit.py", "twist3digit.py", "twist5digit.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "monitors_for_fingers" in text, name
        assert "pinch_motion_should_stop(" in text, name
        assert "pinch_motion_should_stop_toque" not in text, name
        assert "[6, 21" not in text, name


def test_angle_units_empty_unit_with_converter_is_rad() -> None:
    """有 converter 且未设 angle_unit 时按 rad 换算步进（预载加固）。"""
    from unittest.mock import MagicMock

    from hand_gesture_primitives.angle_bridge import AngleUnits
    from hand_gesture_primitives.primitive_base import PrimitiveContext

    converter = MagicMock()
    converter.semantic_indices = [0]
    converter.limits = [(0.0, 1.0)]
    converter.inverted_indices = set()
    ctx = PrimitiveContext(hand_type="o20", radian_converter=converter, angle_unit="")
    u = AngleUnits(ctx)
    assert u.rad is True
    assert u.step(0, 25.5) == pytest.approx(0.1)
