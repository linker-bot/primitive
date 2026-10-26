"""从 JointState 提取 Pose Library 用的 hardware 向量（0~255）。

供 save_pose / pose_library_server 共用，避免 radians 栈误把弧度当 0~255 落盘。
"""

from __future__ import annotations

from typing import List, Optional

from sensor_msgs.msg import JointState

from .hand_config import HandConfig, semantic_legacy255_to_hardware
from .joint_state_mapping import (
    Ros2ControlRadianConverter,
    extract_o6_from_joint_state,
    extract_o20_from_joint_state,
    extract_o30_from_joint_state,
    g20_native_to_l25_urdf16,
    joint_state_to_g20_hardware,
    joint_state_to_l25_native,
    native25_to_active16,
)


def extract_hardware_from_joint_state(
    msg: JointState,
    *,
    hand_key: str,
    hand_side: str,
    hand: HandConfig,
    units: str,
    radian_converter: Optional[Ros2ControlRadianConverter],
) -> Optional[List[float]]:
    """返回长度=hand.num_joints 的 hardware 向量；无法解析则 None。"""
    n = hand.num_joints
    units_l = (units or "hardware").strip().lower()

    if units_l == "radians" and radian_converter is not None and msg.name:
        controller_state = None
        if hand_key == "l25":
            native = joint_state_to_l25_native(msg.name, msg.position, hand_side)
            if native is not None:
                controller_state = native25_to_active16(native)
        elif hand_key in ("o6", "l6"):
            controller_state = extract_o6_from_joint_state(msg.name, msg.position, hand_side)
        elif hand_key == "o20":
            controller_state = extract_o20_from_joint_state(msg.name, msg.position, hand_side)
        elif hand_key == "o30":
            controller_state = extract_o30_from_joint_state(msg.name, msg.position, hand_side)
        elif hand_key == "g20":
            hw, uses_l25 = joint_state_to_g20_hardware(msg.name, msg.position, hand_side)
            if hw is not None:
                controller_state = g20_native_to_l25_urdf16(hw) if uses_l25 else hw
        if controller_state is not None:
            semantic = radian_converter.radians_to_semantic(controller_state)
            return semantic_legacy255_to_hardware(hand, semantic)

    if msg.name:
        if hand_key == "l25":
            native = joint_state_to_l25_native(msg.name, msg.position, hand_side)
            if native is not None:
                return native
        elif hand_key in ("o6", "l6"):
            hw = extract_o6_from_joint_state(msg.name, msg.position, hand_side)
            if hw is not None:
                return hw
        elif hand_key == "o20":
            hw = extract_o20_from_joint_state(msg.name, msg.position, hand_side)
            if hw is not None:
                return list(hw)
        elif hand_key == "o30":
            hw = extract_o30_from_joint_state(msg.name, msg.position, hand_side)
            if hw is not None:
                return list(hw)
        elif hand_key == "g20":
            hw, _uses_l25 = joint_state_to_g20_hardware(msg.name, msg.position, hand_side)
            if hw is not None:
                return list(hw)
    if len(msg.position) == n:
        return [float(v) for v in msg.position]
    return None
