"""在 RViz 中可视化 LinkerHand URDF（自动修复 mesh 绝对路径）。

用法:
    # GUI 模式 (手动拖动滑条控制关节)
    ros2 launch hand_gesture_primitives view_hand_urdf.launch.py
    ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
      hand_model:=L6 hand_side:=left

    ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
      mode:=live hand_model:=O30 hand_joint:=O30 hand_side:=left

    # Live 模式 (订阅 ros2_control /joint_states 或 /cb_* — 验证 FK)
    # 须同时设 hand_model + hand_joint（或只设 hand_joint：live 会自动对齐 URDF）
    ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
      mode:=live hand_model:=L6 hand_joint:=L6
    ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
      mode:=live hand_joint:=O20

    # 无显示器 / SSH 无 X11: 不启 RViz，只跑 state publisher + bridge
    ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
      mode:=live hand_joint:=L6 launch_rviz:=false
"""

import os
import tempfile

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def _make_rviz_config(fixed_frame: str) -> str:
    """按根 link 生成 RViz 配置 (Fixed Frame 随型号变化)。"""
    return f"""Panels:
  - Class: rviz_common/Displays
    Name: Displays
  - Class: rviz_common/Views
    Name: Views
Visualization Manager:
  Class: ""
  Displays:
    - Class: rviz_default_plugins/Grid
      Enabled: true
      Name: Grid
      Reference Frame: {fixed_frame}
    - Class: rviz_default_plugins/TF
      Enabled: true
      Frame Timeout: 15
      Marker Scale: 0.15
      Name: TF
      Show Arrows: true
      Show Axes: true
      Show Names: true
    - Class: rviz_default_plugins/RobotModel
      Description Source: topic
      Description Topic:
        Depth: 5
        Durability Policy: Volatile
        History Policy: Keep Last
        Reliability Policy: Reliable
        Value: /robot_description
      Enabled: true
      Name: RobotModel
      Visual Enabled: true
  Global Options:
    Fixed Frame: {fixed_frame}
    Frame Rate: 30
  Name: root
  Tools:
    - Class: rviz_default_plugins/Interact
    - Class: rviz_default_plugins/MoveCamera
    - Class: rviz_default_plugins/Select
  Value: true
  Views:
    Current:
      Class: rviz_default_plugins/Orbit
      Distance: 0.35
      Focal Point:
        X: 0.05
        Y: 0
        Z: 0.08
      Name: Current View
      Pitch: 0.5
      Yaw: 0.8
Window Geometry:
  Height: 800
  Width: 1200
"""


def _launch_setup(context, *args, **kwargs):
    try:
        from hand_gesture_primitives.urdf_utils import (
            get_root_link_name,
            resolve_urdf_path,
            urdf_with_absolute_meshes,
        )
    except ImportError:
        import sys

        pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sys.path.insert(0, pkg_root)
        from hand_gesture_primitives.urdf_utils import (
            get_root_link_name,
            resolve_urdf_path,
            urdf_with_absolute_meshes,
        )

    hand_model = LaunchConfiguration("hand_model").perform(context)
    hand_side = LaunchConfiguration("hand_side").perform(context)
    hand_joint = LaunchConfiguration("hand_joint").perform(context)
    urdf_path_arg = LaunchConfiguration("urdf_path").perform(context).strip()
    fixed_frame_arg = LaunchConfiguration("fixed_frame").perform(context).strip()
    mode = LaunchConfiguration("mode").perform(context).strip().lower()
    launch_rviz_raw = LaunchConfiguration("launch_rviz").perform(context).strip().lower()
    launch_rviz = launch_rviz_raw in ("1", "true", "yes", "on")
    if launch_rviz and not os.environ.get("DISPLAY"):
        print(
            "[view_hand_urdf] WARN: DISPLAY 未设置，跳过 RViz（无头环境可用 launch_rviz:=false）"
        )
        launch_rviz = False

    # live 以 hand_joint 驱动 FK；URDF 型号须与之对齐，避免只改 hand_joint 仍加载默认 L25
    urdf_model = hand_model
    if mode == "live" and hand_joint and hand_joint.upper() != hand_model.upper():
        print(
            f"[view_hand_urdf] live: hand_model={hand_model} → {hand_joint} "
            f"(与 hand_joint 对齐，避免 URDF/FK 错配)"
        )
        urdf_model = hand_joint

    urdf_path = resolve_urdf_path(urdf_model, hand_side, urdf_path_arg)
    urdf_xml = urdf_with_absolute_meshes(urdf_path)
    root_link = fixed_frame_arg or get_root_link_name(urdf_path)

    tmp_urdf = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="linkerhand_view_", delete=False, encoding="utf-8"
    )
    tmp_urdf.write(urdf_xml)
    tmp_urdf.close()

    tmp_rviz = tempfile.NamedTemporaryFile(
        mode="w", suffix=".rviz", prefix="linkerhand_view_", delete=False, encoding="utf-8"
    )
    tmp_rviz.write(_make_rviz_config(root_link))
    tmp_rviz.close()

    print(f"[view_hand_urdf] mode={mode} model={urdf_model} side={hand_side}")
    print(f"[view_hand_urdf] URDF: {urdf_path}")
    print(f"[view_hand_urdf] Fixed Frame / 根 link: {root_link}")
    print(f"[view_hand_urdf] 绝对 mesh URDF: {tmp_urdf.name}")
    if launch_rviz:
        print(f"[view_hand_urdf] RViz 配置: {tmp_rviz.name}")

    mesh_js_topic = "/mesh_joint_states"
    rsp = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "robot_description": ParameterValue(urdf_xml, value_type=str),
            }
        ],
        remappings=[("joint_states", mesh_js_topic)] if mode == "live" else [],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        output="screen",
        arguments=["-d", tmp_rviz.name],
    )

    nodes = [rsp]

    if mode == "live":
        # Live：控制侧 /joint_states(HOP) → bridge → /mesh_joint_states(解剖名)
        bridge = Node(
            package="hand_gesture_primitives",
            executable="joint_state_bridge",
            output="screen",
            parameters=[
                {
                    "hand_side": hand_side,
                    "hand_joint": hand_joint,
                    "urdf_path": urdf_path,
                    "state_source": "auto",
                    "output_topic": mesh_js_topic,
                }
            ],
        )
        nodes.append(bridge)
        src = (
            "/joint_states (ros2_control)"
            if hand_joint.strip().lower() in ("o30", "o6", "l6", "l25", "o20", "g20")
            else f"/cb_{hand_side}_hand_state"
        )
        print(f"[view_hand_urdf] Live 模式: {src} → {mesh_js_topic} → mesh RSP")
    else:
        # GUI 模式: 手动滑条控制（需要 DISPLAY）
        if not os.environ.get("DISPLAY"):
            raise RuntimeError(
                "mode=gui 需要图形界面 (DISPLAY)。请改用 mode:=live launch_rviz:=false，"
                "或 SSH -X / 在有显示器的终端运行。"
            )
        joint_gui = Node(
            package="joint_state_publisher_gui",
            executable="joint_state_publisher_gui",
            output="screen",
        )
        nodes.append(joint_gui)

    if launch_rviz:
        nodes.append(TimerAction(period=1.0, actions=[rviz]))
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hand_model",
                default_value="L25",
                description="手部型号: L25 / L20 / O20 / O6 / L6 / O30 / G20",
            ),
            DeclareLaunchArgument("hand_side", default_value="left", description="left 或 right"),
            DeclareLaunchArgument(
                "hand_joint",
                default_value="L25",
                description="hand_joint (L25/L20/O20/O6/L6/O30/G20), live 模式 FK 转换用手型",
            ),
            DeclareLaunchArgument(
                "urdf_path",
                default_value="",
                description="URDF 绝对路径 (空=自动搜索 evo_work / linkerhand-urdf，兼容 l6v3.1 等版本化文件名)",
            ),
            DeclareLaunchArgument(
                "fixed_frame",
                default_value="",
                description="RViz Fixed Frame (空=自动检测根 link)",
            ),
            DeclareLaunchArgument(
                "mode", default_value="gui", description="gui=滑条手动控制, live=订阅真实硬件反馈"
            ),
            DeclareLaunchArgument(
                "launch_rviz",
                default_value="true",
                description="是否启动 RViz；无 DISPLAY 时自动跳过",
            ),
            OpaqueFunction(function=_launch_setup),
        ]
    )
