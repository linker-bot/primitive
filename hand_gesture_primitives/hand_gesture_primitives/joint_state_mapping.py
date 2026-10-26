"""机载 /joint_states 与各型号硬件向量映射。

- L25: 16 主动 ↔ native 25（与 linkerhand_hardware.cpp 一致）
- O6/L6: 6 主动直通（关节名与 o6/l6.yaml joint_names 一致；L6 复用 O6 抽取）
- O20: 20 主动直通（与 o20_hand.ros2_control.xacro / o20.yaml joint_names 一致）
- O30: 20 主动直通（HOP NAMES_EN / o30.yaml；语义经 from_semantic 映射）
- G20: 16 主动；手-only 为 SDK 序；整机可走 L25 URDF 序再映射（与 HW 一致）
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple
from xml.etree import ElementTree

# ForwardCommandController / joint_state_broadcaster 关节序（不含左右前缀）
ACTIVE16_SUFFIXES: List[str] = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "thumb_mcp",
    "index_mcp_roll",
    "index_mcp_pitch",
    "index_pip",
    "middle_mcp_roll",
    "middle_mcp_pitch",
    "middle_pip",
    "ring_mcp_roll",
    "ring_mcp_pitch",
    "ring_pip",
    "pinky_mcp_roll",
    "pinky_mcp_pitch",
    "pinky_pip",
]

# O6 硬件序 = o6.yaml joint_names / ros2_control 关节序
O6_ACTIVE_SUFFIXES: List[str] = [
    "thumb_cmc_pitch",
    "thumb_cmc_yaw",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

# O30 硬件序 = o30.yaml joint_names / HOP NAMES_EN
O30_ACTIVE_SUFFIXES: List[str] = [
    "thumb_roll",
    "thumb_yaw",
    "index_yaw",
    "middle_yaw",
    "ring_yaw",
    "little_yaw",
    "thumb_root1",
    "index_root1",
    "middle_root1",
    "ring_root1",
    "little_root1",
    "index_root2",
    "middle_root2",
    "ring_root2",
    "little_root2",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "little_tip",
]

# O20 硬件序 = o20.yaml joint_names / ros2_control 关节序
O20_ACTIVE_SUFFIXES: List[str] = [
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "pinky_base",
    "thumb_abd",
    "index_abd",
    "middle_abd",
    "ring_abd",
    "pinky_abd",
    "thumb_rot",
    "rsv_11",
    "rsv_12",
    "rsv_13",
    "rsv_14",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]

# G20 SDK 序 = g20.yaml joint_names（L20 去 reserved）
G20_ACTIVE_SUFFIXES: List[str] = [
    "thumb_base",
    "index_base",
    "middle_base",
    "ring_base",
    "pinky_base",
    "thumb_abd",
    "index_abd",
    "middle_abd",
    "ring_abd",
    "pinky_abd",
    "thumb_rot",
    "thumb_tip",
    "index_tip",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
]

# ForwardCommandController 每个位置对应的 **O20 槽**（表内仍写 O20；
# ``ros2_control_semantic_indices`` 返回时投影为 S30 下标）。
#
# ros2_control 对外契约为 URDF 弧度；gesture 侧经 Ros2ControlRadianConverter
# 在 semantic_rad@S30 与 controller 弧度之间换算。
ROS2_CONTROL_SEMANTIC_INDICES = {
    "l25": [10, 5, 0, 15, 6, 1, 16, 7, 2, 17, 8, 3, 18, 9, 4, 19],
    "o6": [0, 5, 1, 2, 3, 4],
    "l6": [0, 5, 1, 2, 3, 4],
    "o20": list(range(20)),
    "o30": [10, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    # 手-only G20 SDK 序；整机 l25_urdf 见 g20_l25_urdf 键
    "g20": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19],
    "g20_l25_urdf": [10, 5, 0, 15, 6, 1, 16, 7, 2, 17, 8, 3, 18, 9, 4, 19],
}

NATIVE_JOINTS = 25
ACTIVE_JOINTS = 16
O6_JOINTS = 6
O20_JOINTS = 20
G20_JOINTS = 16


def ros2_control_semantic_indices(hand_type: str) -> List[int]:
    """Return **S30** semantic indices in this hand model's controller order.

    表内仍按 O20 槽维护；此处经 ``o20_joint_index_to_s30`` 投影。
    O30 表项本身已是相对 from_o20 兼容槽，同样投影到 S30。
    """
    from .semantic_30 import o20_joint_index_to_s30

    key = (hand_type or "").strip().lower()
    try:
        o20_idxs = list(ROS2_CONTROL_SEMANTIC_INDICES[key])
    except KeyError as exc:
        raise ValueError(f"未知 hand_type: {hand_type!r}") from exc
    out: List[int] = []
    for o20_idx in o20_idxs:
        s30 = o20_joint_index_to_s30(int(o20_idx))
        out.append(s30 if s30 >= 0 else int(o20_idx))
    return out


def semantic_to_controller_hardware(
    hand_type: str,
    semantic: Sequence[float],
    hand_config: "object",
) -> List[float]:
    """语义向量 → ForwardCommandController 硬件向量（按 URDF/控制器主动序）。

    优先 ``from_semantic``（S30）；否则 O20 ``to_hardware`` 兼容路径。
    """
    from .semantic_30 import SEMANTIC_30_NAMES, as_semantic_30

    key = (hand_type or "").strip().lower()
    if key == "g20_l25_urdf":
        key = "l25"
    indices = ros2_control_semantic_indices(key)
    out: List[float] = []
    if getattr(hand_config, "has_from_semantic", False):
        hw_full = hand_config.from_semantic(as_semantic_30(semantic))
        smap = getattr(hand_config, "_semantic_map", {}) or {}
        for s30_idx in indices:
            ch = (
                SEMANTIC_30_NAMES[int(s30_idx)]
                if 0 <= int(s30_idx) < len(SEMANTIC_30_NAMES)
                else ""
            )
            spec = smap.get(ch)
            if not spec:
                default = float(getattr(hand_config, "_reserved_default", 255.0))
                out.append(default)
                continue
            hw_idx = int(spec["to"])
            out.append(float(hw_full[hw_idx]) if 0 <= hw_idx < len(hw_full) else 0.0)
        return out

    from .semantic_30 import project_s30_to_o20

    vec = list(semantic)
    if len(vec) >= 30:
        vec = project_s30_to_o20(vec)
    hw_full = hand_config.to_hardware(vec)
    mapping = getattr(hand_config, "_mapping", {})
    from .semantic_30 import s30_joint_index_to_o20

    for s30_idx in indices:
        o20_idx = s30_joint_index_to_o20(int(s30_idx))
        spec = mapping.get(int(o20_idx)) if o20_idx >= 0 else None
        if not spec:
            out.append(0.0)
            continue
        hw_idx = int(spec["to"])
        out.append(float(hw_full[hw_idx]) if 0 <= hw_idx < len(hw_full) else 0.0)
    return out


def g20_uses_l25_urdf_layout(robot_description: str, hand_side: str) -> bool:
    """整机 G20 复用 L25 关节名时为 True；手-only SDK 名为 False。

    robot_description 为空时返回 False（由显式 g20_joint_order 或 /joint_states 推断）。
    """
    desc = robot_description or ""
    if not desc.strip():
        return False
    prefix = _prefix(hand_side)
    if f"{prefix}thumb_cmc_roll" in desc:
        return True
    if f"{prefix}thumb_base" in desc:
        return False
    return False


def resolve_g20_l25_urdf(
    g20_joint_order: str = "",
    robot_description: str = "",
    hand_side: str = "left",
) -> bool:
    """决定 G20 命令是否按整机 L25 URDF 16 序发布。

    优先级：显式 g20_joint_order（l25_urdf|native）> robot_description 探测。
    """
    order = (g20_joint_order or "").strip().lower()
    if order in ("l25_urdf", "l25"):
        return True
    if order in ("native", "sdk", "g20"):
        return False
    if order:
        raise ValueError(f"未知 g20_joint_order: {g20_joint_order!r}（期望 l25_urdf|native|空）")
    return g20_uses_l25_urdf_layout(robot_description, hand_side)


def resolve_radian_hand_type(
    hand_type: str,
    *,
    g20_joint_order: str = "",
    robot_description: str = "",
    hand_side: str = "left",
) -> str:
    """radians 换算用的 hand_type：整机 G20 返回 ``g20_l25_urdf``，其余原样。"""
    key = (hand_type or "").strip().lower()
    if key != "g20":
        return key
    if resolve_g20_l25_urdf(g20_joint_order, robot_description, hand_side):
        return "g20_l25_urdf"
    return "g20"


def ros2_control_joint_names(hand_type: str, hand_side: str) -> List[str]:
    """Return controller joint names in ForwardCommandController order."""
    key = (hand_type or "").strip().lower()
    if key == "l25":
        return active_joint_names(hand_side)
    if key in ("o6", "l6"):
        return o6_joint_names(hand_side)
    if key == "o20":
        return o20_joint_names(hand_side)
    if key == "o30":
        return o30_joint_names(hand_side)
    if key == "g20":
        return g20_joint_names(hand_side)
    if key == "g20_l25_urdf":
        return active_joint_names(hand_side)
    raise ValueError(f"未知 hand_type: {hand_type!r}")


def _inverted_semantic_indices(hand_type: str, hand_side: str) -> Set[int]:
    from .semantic_30 import SEMANTIC_30_IDX, o20_joint_index_to_s30

    key = (hand_type or "").strip().lower()
    side = (hand_side or "").strip().lower()
    # 来自各 hand config 的 fk.invert_joints / from_semantic invert。
    # L25/O6/L6/G20 的语义正方向与整机 URDF 一致；
    # O20 仅右手 thumb yaw 相反（O20 idx 5 → S30）。
    # O30 四指 yaw：HOP 物理方向与 S30 侧摆约定相反（真机 v_sign 验证），
    # 须在 255↔rad 层 invert，否则 radians 发布路径不会走 from_semantic。
    if key == "o20" and side == "right":
        return {o20_joint_index_to_s30(5)}
    if key == "o30":
        return {
            SEMANTIC_30_IDX["index_mcp_abd"],
            SEMANTIC_30_IDX["middle_mcp_abd"],
            SEMANTIC_30_IDX["ring_mcp_abd"],
            SEMANTIC_30_IDX["pinky_mcp_abd"],
        }
    return set()


class Ros2ControlRadianConverter:
    """Convert S30 semantic values to/from exact joint limits in robot_description.

    内部同时支持：
    - legacy semantic 0～255（``semantic_to_radians`` / ``radians_to_semantic``）
    - semantic_rad（``semantic_rad_to_controller`` / ``controller_to_semantic_rad``）
    - 控制器序 uint8（``hardware_uint8_to_radians``）

    ``semantic_indices`` 为 **S30** 下标。
    """

    def __init__(self, robot_description: str, hand_type: str, hand_side: str):
        if not robot_description.strip():
            raise ValueError("radians 模式需要非空 robot_description")
        self.hand_type = (hand_type or "").strip().lower()
        self.semantic_indices = ros2_control_semantic_indices(hand_type)
        self.joint_names = ros2_control_joint_names(hand_type, hand_side)
        self.inverted_indices = _inverted_semantic_indices(hand_type, hand_side)

        root = ElementTree.fromstring(robot_description)
        limits_by_name: Dict[str, Tuple[float, float]] = {}
        for joint in root.findall("joint"):
            name = joint.get("name", "")
            limit = joint.find("limit")
            if not name or limit is None:
                continue
            lower = float(limit.get("lower", "0"))
            upper = float(limit.get("upper", "0"))
            limits_by_name[name] = (lower, upper)

        self.limits: List[Tuple[float, float]] = []
        missing: List[str] = []
        for ctrl_i, (semantic_idx, joint_name) in enumerate(
            zip(self.semantic_indices, self.joint_names)
        ):
            limit = limits_by_name.get(joint_name)
            # O20 的 rsv_* 是保留 command interface，不对应可视化关节。
            if (
                limit is None
                and self.hand_type == "o20"
                and ("rsv_" in joint_name or ctrl_i in range(11, 15))
            ):
                limit = (0.0, 0.0)
            if limit is None:
                missing.append(joint_name)
                limit = (0.0, 0.0)
            self.limits.append(limit)
        if missing:
            raise ValueError("robot_description 缺少手部关节限位: " + ", ".join(missing))

    def hardware_uint8_to_radians(self, hw: Sequence[float]) -> List[float]:
        """控制器序 0~255 → 弧度。"""
        out: List[float] = []
        for i, ((lower, upper), semantic_idx) in enumerate(
            zip(self.limits, self.semantic_indices)
        ):
            if upper <= lower:
                out.append(lower)
                continue
            raw = float(hw[i]) if i < len(hw) else 0.0
            ratio = max(0.0, min(255.0, raw)) / 255.0
            if semantic_idx in self.inverted_indices:
                ratio = 1.0 - ratio
            out.append(lower + ratio * (upper - lower))
        return out

    def radians_to_hardware_uint8(self, radians: Sequence[float]) -> List[float]:
        """控制器序弧度 → 0~255。"""
        out: List[float] = []
        for i, ((lower, upper), semantic_idx) in enumerate(
            zip(self.limits, self.semantic_indices)
        ):
            if upper <= lower:
                out.append(0.0)
                continue
            value = float(radians[i]) if i < len(radians) else lower
            clamped = max(lower, min(upper, value))
            ratio = (clamped - lower) / (upper - lower)
            if semantic_idx in self.inverted_indices:
                ratio = 1.0 - ratio
            out.append(max(0.0, min(255.0, ratio * 255.0)))
        return out

    def semantic_to_radians(self, semantic: Sequence[float]) -> List[float]:
        """Legacy S30/O20 semantic 0～255 → controller 序 URDF 弧度。"""
        from . import units as U
        from .semantic_30 import as_semantic_30

        semantic_rad = U.converter_legacy255_to_rad(self, as_semantic_30(semantic))
        return U.converter_rad_to_controller(self, semantic_rad)

    def radians_to_semantic(self, radians: Sequence[float]) -> List[float]:
        """controller 序 URDF 弧度 → legacy S30 semantic 0～255。"""
        from . import units as U

        semantic_rad = U.converter_controller_to_rad(self, radians)
        return U.converter_rad_to_legacy255(self, semantic_rad)

    def legacy255_to_semantic_rad(self, semantic_255: Sequence[float]) -> List[float]:
        from . import units as U
        from .semantic_30 import as_semantic_30

        return U.converter_legacy255_to_rad(self, as_semantic_30(semantic_255))

    def semantic_rad_to_legacy255(self, semantic_rad: Sequence[float]) -> List[float]:
        from . import units as U

        return U.converter_rad_to_legacy255(self, semantic_rad)

    def semantic_rad_to_controller(self, semantic_rad: Sequence[float]) -> List[float]:
        """semantic_rad（S30 槽，单位 rad）→ controller 序弧度。"""
        from . import units as U

        return U.converter_rad_to_controller(self, semantic_rad)

    def controller_to_semantic_rad(self, radians: Sequence[float]) -> List[float]:
        from . import units as U

        return U.converter_controller_to_rad(self, radians)


def _prefix(hand_side: str) -> str:
    return "l_hand_" if hand_side == "left" else "r_hand_"


def active_joint_names(hand_side: str) -> List[str]:
    p = _prefix(hand_side)
    return [p + s for s in ACTIVE16_SUFFIXES]


def o6_joint_names(hand_side: str) -> List[str]:
    p = _prefix(hand_side)
    return [p + s for s in O6_ACTIVE_SUFFIXES]


def o20_joint_names(hand_side: str) -> List[str]:
    p = _prefix(hand_side)
    return [p + s for s in O20_ACTIVE_SUFFIXES]


def o30_joint_names(hand_side: str) -> List[str]:
    p = _prefix(hand_side)
    return [p + s for s in O30_ACTIVE_SUFFIXES]


def o30_single_hand_stub_urdf(hand_side: str) -> str:
    """单 side O30 stub URDF（HOP 关节限位），供 RadianConverter / joint_state_bridge。"""
    p = _prefix(hand_side)
    limits = [
        ("thumb_roll", "0.000", "0.611"),
        ("thumb_yaw", "0.000", "2.094"),
        ("index_yaw", "-0.400", "0.037"),
        ("middle_yaw", "-0.380", "0.054"),
        ("ring_yaw", "-0.280", "0.188"),
        ("little_yaw", "-0.280", "0.281"),
        ("thumb_root1", "0.000", "1.713"),
        ("index_root1", "0.000", "1.729"),
        ("middle_root1", "0.000", "1.884"),
        ("ring_root1", "0.000", "1.963"),
        ("little_root1", "0.000", "1.850"),
        ("index_root2", "0.000", "1.661"),
        ("middle_root2", "0.000", "1.707"),
        ("ring_root2", "0.000", "1.662"),
        ("little_root2", "0.000", "1.703"),
        ("thumb_tip", "0.000", "1.733"),
        ("index_tip", "0.000", "1.763"),
        ("middle_tip", "0.000", "1.586"),
        ("ring_tip", "0.000", "1.675"),
        ("little_tip", "0.000", "1.733"),
    ]
    parts = ['<?xml version="1.0"?><robot name="o30_stub">', f'<link name="{p}base_link"/>']
    parent = f"{p}base_link"
    for suffix, lo, hi in limits:
        link = f"{p}{suffix}_link"
        parts.append(f'<link name="{link}"/>')
        parts.append(
            f'<joint name="{p}{suffix}" type="revolute">'
            f'<parent link="{parent}"/><child link="{link}"/>'
            f'<origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/>'
            f'<limit lower="{lo}" upper="{hi}" effort="1" velocity="10"/>'
            f"</joint>"
        )
        parent = link
    parts.append("</robot>")
    return "".join(parts)


def g20_joint_names(hand_side: str) -> List[str]:
    p = _prefix(hand_side)
    return [p + s for s in G20_ACTIVE_SUFFIXES]


def _extract_named(
    names: Sequence[str],
    positions: Sequence[float],
    wanted: Sequence[str],
) -> Optional[List[float]]:
    if not names or len(names) != len(positions):
        return None
    index: Dict[str, int] = {n: i for i, n in enumerate(names)}
    out: List[float] = []
    for jn in wanted:
        if jn not in index:
            return None
        out.append(float(positions[index[jn]]))
    return out


def active16_to_native25(active: Sequence[float]) -> List[float]:
    """ros2_control 16 主动序 → L25 native25（与 linkerhand_hardware.cpp 一致）。

    侧摆走 native[6-9]（YAML ``*_abd``），不是 reserved[11-14]。
    """
    native = [0.0] * NATIVE_JOINTS
    if len(active) < ACTIVE_JOINTS:
        return native
    native[10] = float(active[0])  # thumb rot
    native[5] = float(active[1])  # thumb abd
    native[0] = float(active[2])  # thumb base
    native[15] = float(active[3])  # thumb mid（URDF thumb_mcp）
    native[6] = float(active[4])  # index abd
    native[1] = float(active[5])  # index base
    native[16] = float(active[6])  # index mid
    native[7] = float(active[7])  # middle abd
    native[2] = float(active[8])  # middle base
    native[17] = float(active[9])  # middle mid
    native[8] = float(active[10])  # ring abd
    native[3] = float(active[11])  # ring base
    native[18] = float(active[12])  # ring mid
    native[9] = float(active[13])  # pinky abd
    native[4] = float(active[14])  # pinky base
    native[19] = float(active[15])  # pinky mid

    native[11] = native[12] = native[13] = native[14] = 255.0  # reserved

    native[20] = native[15]
    native[21] = native[16]
    native[22] = native[17]
    native[23] = native[18]
    native[24] = native[19]
    return native


def native25_to_active16(native: Sequence[float]) -> List[float]:
    """L25 native 25 → ros2_control 16 主动关节（与 HW native25_to_active16 一致）。"""
    active = [0.0] * ACTIVE_JOINTS
    if len(native) < NATIVE_JOINTS:
        return active
    active[0] = float(native[10])
    active[1] = float(native[5])
    active[2] = float(native[0])
    active[3] = float(native[15])  # thumb mid → thumb_mcp
    active[4] = float(native[6])  # index abd
    active[5] = float(native[1])
    active[6] = float(native[16])
    active[7] = float(native[7])  # middle abd
    active[8] = float(native[2])
    active[9] = float(native[17])
    active[10] = float(native[8])  # ring abd
    active[11] = float(native[3])
    active[12] = float(native[18])
    active[13] = float(native[9])  # pinky abd
    active[14] = float(native[4])
    active[15] = float(native[19])
    return active


def extract_active16_from_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Optional[List[float]]:
    """从 JointState 按名抽取 16 主动关节；缺任一关节则返回 None。"""
    return _extract_named(names, positions, active_joint_names(hand_side))


def joint_state_to_l25_native(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Optional[List[float]]:
    active = extract_active16_from_joint_state(names, positions, hand_side)
    if active is None:
        return None
    return active16_to_native25(active)


def extract_o6_from_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Optional[List[float]]:
    """从 JointState 按名抽取 O6 6 维硬件向量；缺任一关节则返回 None。"""
    return _extract_named(names, positions, o6_joint_names(hand_side))


def extract_o20_from_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Optional[List[float]]:
    """从 JointState 按名抽取 O20 20 维硬件向量；缺任一关节则返回 None。"""
    return _extract_named(names, positions, o20_joint_names(hand_side))


def extract_o30_from_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Optional[List[float]]:
    """从 JointState 按名抽取 O30 20 维硬件向量；缺任一关节则返回 None。"""
    return _extract_named(names, positions, o30_joint_names(hand_side))


def extract_g20_from_joint_state(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Optional[List[float]]:
    """从 JointState 按名抽取 G20 SDK 16 维；缺任一关节则返回 None。"""
    return _extract_named(names, positions, g20_joint_names(hand_side))


def l25_urdf16_to_g20_native(urdf16: Sequence[float]) -> List[float]:
    """L25 整机 URDF 16 主动序 → G20 SDK（与 linkerhand_hardware.cpp 一致）。"""
    g20 = [0.0] * G20_JOINTS
    if len(urdf16) < G20_JOINTS:
        return g20
    g20[10] = float(urdf16[0])
    g20[5] = float(urdf16[1])
    g20[0] = float(urdf16[2])
    g20[11] = float(urdf16[3])
    g20[6] = float(urdf16[4])
    g20[1] = float(urdf16[5])
    g20[12] = float(urdf16[6])
    g20[7] = float(urdf16[7])
    g20[2] = float(urdf16[8])
    g20[13] = float(urdf16[9])
    g20[8] = float(urdf16[10])
    g20[3] = float(urdf16[11])
    g20[14] = float(urdf16[12])
    g20[9] = float(urdf16[13])
    g20[4] = float(urdf16[14])
    g20[15] = float(urdf16[15])
    return g20


def g20_native_to_l25_urdf16(g20: Sequence[float]) -> List[float]:
    """G20 SDK → L25 整机 URDF 16 主动序。"""
    urdf16 = [0.0] * G20_JOINTS
    if len(g20) < G20_JOINTS:
        return urdf16
    urdf16[0] = float(g20[10])
    urdf16[1] = float(g20[5])
    urdf16[2] = float(g20[0])
    urdf16[3] = float(g20[11])
    urdf16[4] = float(g20[6])
    urdf16[5] = float(g20[1])
    urdf16[6] = float(g20[12])
    urdf16[7] = float(g20[7])
    urdf16[8] = float(g20[2])
    urdf16[9] = float(g20[13])
    urdf16[10] = float(g20[8])
    urdf16[11] = float(g20[3])
    urdf16[12] = float(g20[14])
    urdf16[13] = float(g20[9])
    urdf16[14] = float(g20[4])
    urdf16[15] = float(g20[15])
    return urdf16


def joint_state_to_g20_hardware(
    names: Sequence[str],
    positions: Sequence[float],
    hand_side: str,
) -> Tuple[Optional[List[float]], bool]:
    """抽取 G20 硬件向量。

    Returns:
        (hw16, uses_l25_urdf_order)
    """
    hw = extract_g20_from_joint_state(names, positions, hand_side)
    if hw is not None:
        return hw, False
    active = extract_active16_from_joint_state(names, positions, hand_side)
    if active is not None:
        return l25_urdf16_to_g20_native(active), True
    return None, False
