"""joint_state_bridge O30 / ros2_control 路径单测。"""

from hand_gesture_primitives.joint_state_bridge import (
    hardware_vector_from_joint_state,
    semantic_from_hardware,
)
from hand_gesture_primitives.joint_state_mapping import (
    O30_ACTIVE_SUFFIXES,
    Ros2ControlRadianConverter,
    o30_joint_names,
    o30_single_hand_stub_urdf,
)
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX, SEMANTIC_DIM


class _Msg:
    def __init__(self, names, position):
        self.name = names
        self.position = position


def test_o30_stub_urdf_loads_converter() -> None:
    stub = o30_single_hand_stub_urdf("left")
    conv = Ros2ControlRadianConverter(stub, "o30", "left")
    assert len(conv.limits) == 20
    assert SEMANTIC_30_IDX["index_mcp_abd"] in conv.inverted_indices
    assert SEMANTIC_30_IDX["pinky_mcp_abd"] in conv.inverted_indices
    assert SEMANTIC_30_IDX["thumb_mcp_abd"] not in conv.inverted_indices


def test_hardware_vector_from_joint_states_radians() -> None:
    stub = o30_single_hand_stub_urdf("left")
    conv = Ros2ControlRadianConverter(stub, "o30", "left")
    names = o30_joint_names("left")
    hw_u8 = [float(i * 12) for i in range(20)]
    rad = conv.hardware_uint8_to_radians(hw_u8)
    msg = _Msg(names, rad)
    out = hardware_vector_from_joint_state(
        msg, hand_joint="O30", hand_side="left", radian_converter=conv
    )
    assert out is not None
    assert len(out) == 20
    for a, b in zip(out, hw_u8):
        assert abs(a - b) < 2.0


def test_semantic_from_o30_hardware_preserves_root2() -> None:
    hw = [0.0] * 20
    idx_root2 = O30_ACTIVE_SUFFIXES.index("index_root2")
    hw[idx_root2] = 180.0
    s30 = semantic_from_hardware("O30", hw)
    assert len(s30) == SEMANTIC_DIM
    assert s30[SEMANTIC_30_IDX["index_pip"]] == 180.0


def test_o30_active_suffixes_count() -> None:
    assert len(O30_ACTIVE_SUFFIXES) == 20
