"""G20 ↔ L25 URDF 序映射（与 linkerhand_hardware.cpp 一致）。"""

from hand_gesture_primitives.hand_config import HandConfig, resolve_hand_config_key
from hand_gesture_primitives.hand_profile import load_hand_profile
from hand_gesture_primitives.joint_state_mapping import (
    g20_native_to_l25_urdf16,
    joint_state_to_g20_hardware,
    l25_urdf16_to_g20_native,
)
from hand_gesture_primitives.primitive_base import HAND_CONFIGS


def test_resolve_g20_config_key():
    assert resolve_hand_config_key("G20") == "g20"
    assert resolve_hand_config_key("g20") == "g20"


def test_g20_yaml_loads():
    cfg = HandConfig("g20")
    assert cfg.num_joints == 16
    assert cfg.joint_names[11] == "thumb_tip"
    hw = cfg.to_hardware([255.0] * 20)
    assert len(hw) == 16


def test_g20_hand_profile():
    profile = load_hand_profile("g20")
    assert profile.hand_type == "g20"
    assert profile.hardware.num_joints == 16


def test_g20_primitive_hand_config():
    cfg = HAND_CONFIGS["g20"]
    assert cfg.num_joints == 16
    assert cfg.thumb_tip == 11
    assert cfg.pinky_tip == 15
    assert cfg.invert_angles is True


def test_l25_urdf_g20_roundtrip():
    urdf16 = [float(i + 1) for i in range(16)]
    g20 = l25_urdf16_to_g20_native(urdf16)
    back = g20_native_to_l25_urdf16(g20)
    assert back == urdf16


def test_joint_state_g20_sdk_and_l25_urdf():
    from hand_gesture_primitives.joint_state_mapping import (
        ACTIVE16_SUFFIXES,
        G20_ACTIVE_SUFFIXES,
    )

    pos = [float(i) for i in range(16)]
    names = [f"l_hand_{s}" for s in G20_ACTIVE_SUFFIXES]
    hw, uses_l25 = joint_state_to_g20_hardware(names, pos, "left")
    assert uses_l25 is False
    assert hw == pos

    l25_names = [f"l_hand_{s}" for s in ACTIVE16_SUFFIXES]
    hw2, uses_l25_2 = joint_state_to_g20_hardware(l25_names, pos, "left")
    assert uses_l25_2 is True
    assert hw2 == l25_urdf16_to_g20_native(pos)


def test_resolve_g20_l25_urdf_priority():
    from hand_gesture_primitives.joint_state_mapping import resolve_g20_l25_urdf

    assert resolve_g20_l25_urdf("l25_urdf", "", "left") is True
    assert resolve_g20_l25_urdf("native", "l_hand_thumb_cmc_roll", "left") is False
    assert resolve_g20_l25_urdf("", "l_hand_thumb_cmc_roll", "left") is True
    assert resolve_g20_l25_urdf("", "l_hand_thumb_base", "left") is False
    assert resolve_g20_l25_urdf("", "", "left") is False


def test_resolve_radian_hand_type_g20_l25_urdf():
    from hand_gesture_primitives.joint_state_mapping import (
        Ros2ControlRadianConverter,
        resolve_radian_hand_type,
        ros2_control_joint_names,
    )

    assert (
        resolve_radian_hand_type(
            "g20", g20_joint_order="l25_urdf", robot_description="", hand_side="left"
        )
        == "g20_l25_urdf"
    )
    assert (
        resolve_radian_hand_type(
            "g20", g20_joint_order="native", robot_description="", hand_side="left"
        )
        == "g20"
    )
    assert resolve_radian_hand_type("l25") == "l25"

    # 整机 URDF 只有 L25 名：用 g20 会缺限位；用 g20_l25_urdf 则成功
    names = ros2_control_joint_names("g20_l25_urdf", "left")
    joints = "".join(
        f'<joint name="{name}" type="revolute">'
        '<limit lower="0" upper="1.0" effort="1" velocity="1"/>'
        "</joint>"
        for name in names
    )
    urdf = f'<robot name="test">{joints}</robot>'
    try:
        Ros2ControlRadianConverter(urdf, "g20", "left")
        raised = False
    except ValueError as exc:
        raised = True
        assert "缺少手部关节限位" in str(exc)
    assert raised
    converter = Ros2ControlRadianConverter(urdf, "g20_l25_urdf", "left")
    assert len(converter.semantic_to_radians([0.0] * 20)) == 16


def test_g20_fk_config_matches_l25_fingertip_names():
    """整机 G20 复用 L25 mesh/官方 URDF，指尖 link 须与 l25.yaml 一致。"""
    from hand_gesture_primitives.fk_solver import _load_fk_config

    g20_fk = _load_fk_config("G20")
    l25_fk = _load_fk_config("L25")
    assert g20_fk is not None and l25_fk is not None
    assert g20_fk["fingertip_links"] == l25_fk["fingertip_links"]
    assert "thumb_distal_link" not in g20_fk["fingertip_links"]
