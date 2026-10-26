import pytest
from hand_gesture_primitives.fk_solver import HandFKSolver
from hand_gesture_primitives.hand_config import HandConfig
from hand_gesture_primitives.joint_state_mapping import (
    Ros2ControlRadianConverter,
    ros2_control_joint_names,
    ros2_control_semantic_indices,
)


def _fake_l25_solver() -> HandFKSolver:
    solver = object.__new__(HandFKSolver)
    solver._active_mapping = {
        0: "thumb_cmc_pitch",
        1: "index_mcp_pitch",
        5: "thumb_cmc_yaw",
        6: "index_mcp_roll",
        10: "thumb_cmc_roll",
        15: "thumb_mcp",
        16: "index_pip",
    }
    solver._joint_limits = {
        "thumb_cmc_pitch": (0.0, 0.84),
        "index_mcp_pitch": (0.0, 1.22),
        "thumb_cmc_yaw": (0.0, 1.57),
        "index_mcp_roll": (-0.23, 0.23),
        "thumb_cmc_roll": (0.0, 1.40),
        "thumb_mcp": (0.0, 1.26),
        "index_pip": (0.0, 1.74),
    }
    solver._invert_joints = set()
    return solver


def test_indexed_radian_conversion_round_trips() -> None:
    solver = _fake_l25_solver()
    semantic = [0.0] * 20
    semantic[0] = 64.0
    semantic[1] = 128.0
    semantic[5] = 192.0
    semantic[6] = 255.0
    semantic[10] = 32.0
    semantic[15] = 200.0
    semantic[16] = 96.0

    radians = solver.o20_to_indexed_radians(semantic)

    assert 0.0 <= radians[0] <= 0.84
    assert -0.23 <= radians[6] <= 0.23
    assert 0.0 <= radians[10] <= 1.40
    restored = solver.indexed_radians_to_o20(radians)
    for index in radians:
        assert restored[index] == pytest.approx(semantic[index])


def test_l25_controller_order_uses_semantic_joint_mapping() -> None:
    # O20 表投影为 S30 下标
    assert ros2_control_semantic_indices("l25") == [
        4,
        1,
        0,
        3,
        6,
        5,
        8,
        11,
        10,
        13,
        16,
        15,
        18,
        21,
        20,
        23,
    ]


def test_l25_hardware_round_trip_keeps_semantic_channels() -> None:
    config = HandConfig("L25")
    semantic = [float(i * 11 % 256) for i in range(20)]
    hardware = config.to_hardware(semantic)
    restored = config.from_hardware(hardware)

    for index in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19):
        assert restored[index] == pytest.approx(semantic[index])


def test_robot_description_converter_respects_controller_order_and_limits() -> None:
    names = ros2_control_joint_names("l25", "left")
    joints = "".join(
        f'<joint name="{name}" type="revolute">'
        '<limit lower="-0.2" upper="1.4" effort="1" velocity="1"/>'
        "</joint>"
        for name in names
    )
    converter = Ros2ControlRadianConverter(f'<robot name="test">{joints}</robot>', "l25", "left")
    from hand_gesture_primitives.semantic_30 import as_semantic_30

    semantic = as_semantic_30([float(i * 13 % 256) for i in range(20)])

    radians = converter.semantic_to_radians(semantic)

    assert len(radians) == 16
    assert all(-0.2 <= value <= 1.4 for value in radians)
    restored = converter.radians_to_semantic(radians)
    for index in ros2_control_semantic_indices("l25"):
        assert restored[index] == pytest.approx(semantic[index])


@pytest.mark.parametrize(("hand_type", "thumb_pitch_limit"), [("o6", 0.58), ("l6", 0.837758)])
@pytest.mark.parametrize("side", ["left", "right"])
def test_six_dof_thumb_pitch_uses_positive_urdf_flexion(
    hand_type: str, thumb_pitch_limit: float, side: str
) -> None:
    names = ros2_control_joint_names(hand_type, side)
    limits = [thumb_pitch_limit, 1.36, 1.60, 1.60, 1.60, 1.60]
    joints = "".join(
        f'<joint name="{name}" type="revolute">'
        f'<limit lower="0" upper="{upper}" effort="1" velocity="1"/>'
        "</joint>"
        for name, upper in zip(names, limits)
    )
    converter = Ros2ControlRadianConverter(f'<robot name="test">{joints}</robot>', hand_type, side)

    open_semantic = [0.0] * 20
    flexed_semantic = [0.0] * 20
    flexed_semantic[0] = 255.0

    assert converter.semantic_to_radians(open_semantic)[0] == pytest.approx(0.0)
    assert converter.semantic_to_radians(flexed_semantic)[0] == pytest.approx(thumb_pitch_limit)
    assert converter.radians_to_semantic([thumb_pitch_limit, 0.0, 0.0, 0.0, 0.0, 0.0])[
        0
    ] == pytest.approx(255.0)
