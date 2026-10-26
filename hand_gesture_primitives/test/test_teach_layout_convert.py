"""Teach panel product/layout conversion (G20 UI uses L25 native25)."""

from hand_gesture_primitives.hand_config import HandConfig
from hand_gesture_primitives.joint_state_mapping import (
    active16_to_native25,
    native25_to_active16,
    ros2_control_semantic_indices,
    semantic_to_controller_hardware,
)
from hand_gesture_primitives.teach_panel.layout_convert import (
    panel_layout_hand_key,
    product_hardware_to_ui,
    ui_to_product_hardware,
)


def test_panel_layout_g20_l25_urdf():
    assert panel_layout_hand_key("g20", g20_joint_order="l25_urdf") == "l25"
    assert panel_layout_hand_key("g20", g20_joint_order="native") == "g20"
    assert panel_layout_hand_key("l25") == "l25"


def test_l25_abd_uses_yaml_index_not_rsv_slot():
    l25 = HandConfig("l25")
    ui = [128.0] * 25
    ui[6] = 200.0  # index_abd
    active = native25_to_active16(ui)
    sem = l25.from_hardware(ui)
    via_sem = semantic_to_controller_hardware("l25", sem, l25)
    idx = ros2_control_semantic_indices("l25")
    # 侧摆在 active[4]（index_mcp_roll），来自 native[6]，不是 reserved[11]
    assert active[4] == 200.0
    assert via_sem[idx.index(6)] == 200.0
    # 往返保留 abd / mid
    back = active16_to_native25(active)
    assert back[6] == 200.0
    assert back[15] == ui[15]


def test_thumb_mid_slot_same_for_left_and_right():
    """拇指指中：左右仅前缀不同（l_hand_/r_hand_），控制器槽位同为 slot3=thumb_mcp。"""
    from hand_gesture_primitives.joint_state_mapping import (
        ACTIVE16_SUFFIXES,
        Ros2ControlRadianConverter,
        active_joint_names,
        l25_urdf16_to_g20_native,
    )

    assert ACTIVE16_SUFFIXES[3] == "thumb_mcp"
    assert active_joint_names("left")[3] == "l_hand_thumb_mcp"
    assert active_joint_names("right")[3] == "r_hand_thumb_mcp"

    l25 = HandConfig("l25")
    ui_open = [255.0] * 25
    ui_flex = list(ui_open)
    ui_flex[15] = 0.0  # thumb_mid
    ui_flex[20] = 0.0

    def rad_for(side: str, ui):
        parts = ['<robot name="x">']
        for s in ACTIVE16_SUFFIXES:
            p = "l_hand_" if side == "left" else "r_hand_"
            parts.append(
                f'<joint name="{p}{s}" type="revolute"><limit lower="0" upper="1.25"/></joint>'
            )
        parts.append("</robot>")
        conv = Ros2ControlRadianConverter("".join(parts), "l25", side)
        return conv.semantic_to_radians(l25.from_hardware(ui))

    for side in ("left", "right"):
        r0 = rad_for(side, ui_open)
        r1 = rad_for(side, ui_flex)
        assert abs(r1[3] - r0[3]) > 0.5  # thumb_mcp 有明显变化

    # G20 l25_urdf：指中槽驱动 tip 电机
    ctrl = native25_to_active16(ui_flex)
    g20 = l25_urdf16_to_g20_native(ctrl)
    assert g20[11] == ctrl[3]  # tip motor ← thumb_mcp slot


def test_g20_ui25_pose_roundtrip():
    ui = [float((i * 17) % 256) for i in range(25)]
    # tip follows mid after product roundtrip via L25 also=
    g20_hw = ui_to_product_hardware(ui, product_key="g20", layout_key="l25")
    assert len(g20_hw) == 16
    back = product_hardware_to_ui(g20_hw, product_key="g20", layout_key="l25")
    assert len(back) == 25
    # base/abd/rot channels that exist on both should match after semantic roundtrip
    sem_l = HandConfig("l25").from_hardware(ui)
    sem_g = HandConfig("g20").from_hardware(g20_hw)
    for i in (0, 1, 5, 6, 10, 15, 16):
        assert abs(sem_l[i] - sem_g[i]) < 1e-6
