"""从 /joint_states 或 /cb_{side}_hand_state 捕获硬件关节并写入 Pose Library。

用法（机载侧化 topic，真机/mock 默认 radians）::

  ros2 run hand_gesture_primitives pub_gesture_cmd -- left "mode teach"
  ros2 run hand_gesture_primitives save_pose --ros-args \\
    -p name:=my_grip -p keyframe:=pre -p hand_side:=left -p hand_type:=g20 \\
    -p ros2_control_command_units:=radians \\
    -p robot_description:="$(xacro ...)" \\
    -p g20_joint_order:=l25_urdf
  # 再调到下一姿态后：
  ros2 run hand_gesture_primitives save_pose --ros-args -p name:=my_grip -p keyframe:=grasp ...
  ros2 run hand_gesture_primitives pub_gesture_cmd -- left "mode primitive"
  ros2 run hand_gesture_primitives pub_gesture_cmd -- left "pose my_grip"
  ros2 run hand_gesture_primitives pub_gesture_cmd -- left "pose my_grip grasp 0.4"

说明:
  - 默认 ``ros2_control_command_units:=radians``（与机载 FCC/gesture 一致）
  - ``-p space:=hardware`` 可显式落盘机型序（默认 ``semantic_v3``）；**多关键帧禁止 hardware**
  - ``-p keyframe:=pre`` 追加/替换关键帧（见 ``upsert_pose_keyframe``）
  - 落盘目录见 ``pose_library.pose_dir()``（可用 HAND_GESTURE_POSE_DIR）
  - 示教前请 mode teach，避免 gesture_node 抢发命令
"""

from __future__ import annotations

import sys
from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .hand_config import HandConfig, resolve_hand_config_key
from .hand_state_extract import extract_hardware_from_joint_state
from .joint_coupling import apply_tip_mid_coupling, tip_mid_couple_map
from .joint_state_mapping import Ros2ControlRadianConverter, resolve_radian_hand_type
from .pose_library import (
    DEFAULT_SAVE_SPACE,
    commit_saved_pose,
    pose_dir,
    prepare_pose_for_save,
)


class SavePoseNode(Node):
    def __init__(self) -> None:
        super().__init__("save_pose")
        self.declare_parameter("name", "")
        self.declare_parameter("keyframe", "")
        self.declare_parameter("hand_side", "left")
        self.declare_parameter("hand_type", "l25")
        self.declare_parameter("hand_joint", "")
        self.declare_parameter("duration", 0.6)
        self.declare_parameter("description", "")
        self.declare_parameter("space", DEFAULT_SAVE_SPACE)
        self.declare_parameter("timeout_sec", 5.0)
        self.declare_parameter("state_topic", "/joint_states")
        # 与机载 gesture / FCC 默认一致；遗留 SDK 直连可显式设 hardware
        self.declare_parameter("ros2_control_command_units", "radians")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("g20_joint_order", "")

        name = self.get_parameter("name").value.strip().lower()
        if not name:
            self.get_logger().error("参数 name 必填，例如 -p name:=my_open")
            raise SystemExit(2)

        side = self.get_parameter("hand_side").value.strip().lower()
        if side not in ("left", "right"):
            self.get_logger().error(f"hand_side 无效: {side}")
            raise SystemExit(2)

        hand_type = self.get_parameter("hand_type").value.strip()
        hand_joint = self.get_parameter("hand_joint").value.strip()
        if hand_joint:
            hand_type = hand_joint
        self._hand_key = resolve_hand_config_key(hand_type)
        self._hand = HandConfig(self._hand_key)
        self._side = side
        self._name = name
        self._keyframe = str(self.get_parameter("keyframe").value or "").strip().lower()
        self._duration = float(self.get_parameter("duration").value)
        self._description = str(self.get_parameter("description").value or "")
        self._space = str(self.get_parameter("space").value or DEFAULT_SAVE_SPACE).strip().lower()
        self._done = False
        self._exit_code = 1

        units = (
            (self.get_parameter("ros2_control_command_units").value or "radians").strip().lower()
        )
        if units not in ("hardware", "radians"):
            self.get_logger().error(f"ros2_control_command_units 无效: {units}")
            raise SystemExit(2)
        self._units = units
        self._radian_converter: Optional[Ros2ControlRadianConverter] = None
        if units == "radians":
            robot_description = str(self.get_parameter("robot_description").value or "")
            g20_joint_order = str(self.get_parameter("g20_joint_order").value or "")
            try:
                radian_hand_type = resolve_radian_hand_type(
                    self._hand_key,
                    g20_joint_order=g20_joint_order,
                    robot_description=robot_description,
                    hand_side=self._side,
                )
                self._radian_converter = Ros2ControlRadianConverter(
                    robot_description, radian_hand_type, self._side
                )
            except ValueError as exc:
                self.get_logger().error(
                    f"radians 模式初始化失败: {exc}；"
                    "请传入 robot_description（及 G20 的 g20_joint_order:=l25_urdf）"
                )
                raise SystemExit(2) from exc

        topic = self.get_parameter("state_topic").value.strip() or "/joint_states"
        self._state_topic = topic
        self.create_subscription(JointState, topic, self._on_state, 10)
        if topic == "/joint_states":
            self.create_subscription(
                JointState, f"/cb_{side}_hand_state", self._on_legacy_state, 10
            )
        timeout = float(self.get_parameter("timeout_sec").value)
        self._timer = self.create_timer(timeout, self._on_timeout)
        self.get_logger().info(
            f"等待一手状态: {topic} (+ /cb_{side}_hand_state) → 保存 '{name}'"
            f"{f' keyframe={self._keyframe}' if self._keyframe else ''} "
            f"(hand={self._hand_key}, side={side}, joints={self._hand.num_joints}, "
            f"space={self._space}, units={units}, pose_dir={pose_dir()})"
        )

    def _extract_hardware(self, msg: JointState) -> Optional[List[float]]:
        return extract_hardware_from_joint_state(
            msg,
            hand_key=self._hand_key,
            hand_side=self._side,
            hand=self._hand,
            units=self._units,
            radian_converter=self._radian_converter,
        )

    def _on_timeout(self) -> None:
        if self._done:
            return
        self.get_logger().error("等待 hand_state 超时，未保存")
        self._done = True
        self._exit_code = 1
        rclpy.shutdown()

    def _on_legacy_state(self, msg: JointState) -> None:
        self._save_from_msg(msg)

    def _on_state(self, msg: JointState) -> None:
        self._save_from_msg(msg)

    def _save_from_msg(self, msg: JointState) -> None:
        if self._done:
            return
        positions = self._extract_hardware(msg)
        if positions is None:
            return
        # L25：保存前 tip=mid，与 teach 面板 / FCC 能力一致
        cmap = tip_mid_couple_map(self._hand_key, self._hand.joint_names)
        if cmap:
            positions = apply_tip_mid_coupling(positions, cmap)
        prepared = prepare_pose_for_save(
            self._hand_key,
            positions,
            space=self._space,
            radian_converter=self._radian_converter,
        )
        if isinstance(prepared, str):
            self.get_logger().error(f"保存失败: {prepared}")
            self._done = True
            self._exit_code = 1
            rclpy.shutdown()
            return
        save_positions, space, unit = prepared
        err = commit_saved_pose(
            self._hand_key,
            self._name,
            save_positions,
            duration=self._duration,
            description=self._description,
            source="teach",
            space=space,
            unit=unit,
            hand_side=self._side,
            keyframe=self._keyframe,
        )
        self._done = True
        if err:
            self.get_logger().error(f"保存失败: {err}")
            self._exit_code = 1
        else:
            kf_msg = f" keyframe={self._keyframe}" if self._keyframe else ""
            self.get_logger().info(
                f"已保存姿态 '{self._name}'{kf_msg} ({space}, unit={unit}) → "
                f"{pose_dir() / f'{self._hand_key}_{self._side}.json'}"
            )
            self._exit_code = 0
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    try:
        node = SavePoseNode()
    except SystemExit as e:
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(int(e.code) if isinstance(e.code, int) else 2)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        code = getattr(node, "_exit_code", 1)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        sys.exit(code)


if __name__ == "__main__":
    main()
