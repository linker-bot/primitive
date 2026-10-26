"""Helpers for embedding hand_gesture_primitives in robot launch files.

Dual-hand public topics (side = left|right):
  /{side}_hand/gesture_cmd[_exec]
  /{side}_hand/gesture_status|gesture_result
  /{side}_hand/tcp_pose|object_pose
  /{side}_hand/grasp_gate/status|selected_label
  /{side}_hand/stream_target
  /{side}_hand/save_pose|list_poses|...

hand_type / hand_joint and the matching robot_description are supplied by the
caller. Optional gate, pose server and teach panel nodes use the same configured
hand type and side. An explicit urdf_path overrides FK model discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from launch_ros.actions import Node

_BOTH = frozenset({"both", "bimanual", "lr", "dual", "all"})


def parse_hand_sides(side_env: str = "", *, default: str = "left") -> List[str]:
    """硬件侧 left|right|both；与 GESTURE 无关。空串回落 default。"""
    raw = (side_env or "").strip().lower()
    s = raw if raw else (default or "left").strip().lower()
    if s in _BOTH:
        return ["left", "right"]
    if s == "right":
        return ["right"]
    if s == "left":
        return ["left"]
    # 未知值回落 default
    d = (default or "left").strip().lower()
    if d in _BOTH:
        return ["left", "right"]
    return ["right"] if d == "right" else ["left"]


def parse_gesture_sides(gesture_env: str, side_env: str = "left") -> List[str]:
    """Return [] / ['left'] / ['right'] / ['left','right'].

    Enable with GESTURE=1|true|on, or GESTURE=both.
    Side via GESTURE_SIDE=left|right|both (default **left** for launch).

    Callers sharing command topics should pass the same explicit side.
    """
    g = (gesture_env or "").strip().lower()
    s = (side_env or "left").strip().lower()
    if not g or g in ("0", "false", "no", "off"):
        return []
    if g in _BOTH or s in _BOTH:
        return ["left", "right"]
    if s == "right":
        return ["right"]
    return ["left"]


def resolve_hand_only_sides(
    gesture_env: str = "",
    *,
    gesture_side: str = "",
    hand_side: str = "",
    default_side: str = "left",
) -> Tuple[List[str], List[str]]:
    """手-only launch 的 (hw_sides, gesture_sides)。

    - 硬件侧：HAND_SIDE → GESTURE_SIDE →（GESTURE=both 时 both）→ default_side。
    - gesture 节点仅在 GESTURE=1|both 时启动；GESTURE=both 时左右都起。
    - L25 等可用 default_side='right' 保持旧右手基线。
    """
    g = (gesture_env or "").strip().lower()
    explicit = (hand_side or "").strip() or (gesture_side or "").strip()
    if explicit:
        side = explicit
    elif g in _BOTH:
        # GESTURE=both 且未指定 SIDE：双手硬件与 gesture 对齐
        side = "both"
    else:
        side = default_side or "left"
    hw_sides = parse_hand_sides(side, default=default_side)
    gesture_sides = parse_gesture_sides(gesture_env, side)
    return hw_sides, gesture_sides


def gesture_nodes_for_side(
    side: str,
    *,
    hand_type: str,
    hand_joint: str,
    with_gate: bool,
    cmd_backend: str = "ros2_control",
    ros2_control_command_units: str = "radians",
    robot_description: str = "",
    g20_joint_order: str = "",
    state_topic: str = "/joint_states",
    urdf_path: str = "",
    with_pose_server: bool = False,
    with_teach_panel: bool = False,
    teach_http_host: str = "127.0.0.1",
    teach_http_port: Optional[int] = None,
    control_rate: float = 10.0,
    primitive_speed: float = 1.0,
    pose_dir: str = "",
) -> List["Node"]:
    """Build gesture_node (+ optional gate / pose_server / teach_panel) for one side.

    ros2_control_command_units 默认 radians：mock/mujoco/真机 ros2_control 对外均为 URDF 弧度。
    """
    from launch_ros.actions import Node

    if side not in ("left", "right"):
        raise ValueError(f"side must be left|right, got {side!r}")
    from .workbench_motion import validate_speed

    primitive_speed = validate_speed(primitive_speed)
    environment = {"PYTHONNOUSERSITE": "1"}
    if pose_dir:
        environment["HAND_GESTURE_POSE_DIR"] = str(pose_dir)
    prefix = f"/{side}_hand"
    params = {
        "hand_side": side,
        "hand_type": hand_type,
        "hand_joint": hand_joint,
        "state_topic": state_topic,
        "cmd_backend": cmd_backend,
        "ros2_control_command_units": ros2_control_command_units,
        "robot_description": robot_description,
        "g20_joint_order": (g20_joint_order or "").strip(),
        "control_rate": float(control_rate),
        "primitive_speed": primitive_speed,
        "cmd_topic": (f"{prefix}/gesture_cmd_exec" if with_gate else f"{prefix}/gesture_cmd"),
    }
    if urdf_path:
        params["urdf_path"] = urdf_path
    nodes: List[Node] = [
        Node(
            package="hand_gesture_primitives",
            executable="gesture_node",
            name=f"{side}_hand_gesture_node",
            output="screen",
            parameters=[params],
            # 避开用户级 NumPy 2（pyzed 需要）与 ROS pinocchio（NumPy 1.x ABI）冲突。
            # gesture_node 不依赖 pyzed；隔离后走 apt numpy 1.26 + ros-jazzy-pinocchio。
            additional_env=environment,
        ),
    ]
    if with_gate:
        nodes.append(
            Node(
                package="hand_gesture_primitives",
                executable="grasp_gate",
                name=f"{side}_grasp_gate",
                output="screen",
                parameters=[{"hand_side": side}],
                additional_env=environment,
            ),
        )
    if with_pose_server:
        nodes.append(
            Node(
                package="hand_gesture_primitives",
                executable="pose_library_server",
                name=f"{side}_pose_library_server",
                output="screen",
                additional_env=environment,
                parameters=[
                    {
                        "hand_side": side,
                        "hand_type": hand_type,
                        "hand_joint": hand_joint,
                        "state_topic": state_topic,
                        # 与 gesture/teach 同量纲，避免空 positions 缓存把弧度当 0~255 落盘
                        "ros2_control_command_units": ros2_control_command_units,
                        "robot_description": robot_description,
                        "g20_joint_order": (g20_joint_order or "").strip(),
                    }
                ],
            ),
        )
    if with_teach_panel:
        port = (
            teach_http_port if teach_http_port is not None else (8001 if side == "left" else 8002)
        )
        # 始终发公开 API /{side}_hand/gesture_cmd（与 CLI/teleop 一致）。
        # 有 gate 时由 gate 转发到 gesture_cmd_exec；mode/pose/open 等非包络原语会直通。
        # 勿绑 gesture_cmd_exec，否则与 CLI 路径分叉，排障易误判「topic 不对」。
        panel_params = {
            "hand_side": side,
            "hand_type": hand_type,
            "hand_joint": hand_joint,
            "state_topic": state_topic,
            "cmd_backend": cmd_backend,
            "ros2_control_command_units": ros2_control_command_units,
            "robot_description": robot_description,
            "g20_joint_order": (g20_joint_order or "").strip(),
            "cmd_topic": f"{prefix}/gesture_cmd",
            "http_host": teach_http_host,
            "http_port": port,
            "primitive_speed": primitive_speed,
        }
        nodes.append(
            Node(
                package="hand_gesture_primitives",
                executable="teach_panel",
                name=f"{side}_teach_panel",
                output="screen",
                additional_env=environment,
                parameters=[panel_params],
            ),
        )
    return nodes


def all_gesture_nodes(sides: Sequence[str], **kwargs) -> List["Node"]:
    out: List["Node"] = []
    for side in sides:
        out.extend(gesture_nodes_for_side(side, **kwargs))
    return out
