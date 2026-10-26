"""Pose Library ROS 服务节点 + 可选状态缓存供 SavePose。

默认 ``ros2_control_command_units=radians``，与机载 gesture / FCC 一致；
须传入 ``robot_description``（G20 整机另加 ``g20_joint_order:=l25_urdf``）。
"""

from __future__ import annotations

from typing import List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from .export_primitive_poses import export_primitive_poses
from .hand_config import HandConfig, resolve_hand_config_key
from .hand_state_extract import extract_hardware_from_joint_state
from .import_gui_presets import import_gui_presets
from .joint_coupling import apply_tip_mid_coupling, tip_mid_couple_map
from .joint_state_mapping import Ros2ControlRadianConverter, resolve_radian_hand_type
from .pose_bundle import export_library_to_file, import_bundle_file
from .pose_library import (
    DEFAULT_SAVE_SPACE,
    commit_saved_pose,
    delete_pose,
    list_poses,
    pose_dir,
    prepare_pose_for_save,
)
from .topic_defaults import side_hand_prefix


class PoseLibraryServer(Node):
    def __init__(self) -> None:
        super().__init__("pose_library_server")
        self.declare_parameter("hand_side", "left")
        self.declare_parameter("hand_type", "l25")
        self.declare_parameter("hand_joint", "")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("ros2_control_command_units", "radians")
        self.declare_parameter("robot_description", "")
        self.declare_parameter("g20_joint_order", "")

        side = self.get_parameter("hand_side").value.strip().lower()
        hand_type = self.get_parameter("hand_type").value.strip()
        hand_joint = self.get_parameter("hand_joint").value.strip()
        if hand_joint:
            hand_type = hand_joint
        self._hand_key = resolve_hand_config_key(hand_type)
        self._hand = HandConfig(self._hand_key)
        self._side = side if side in ("left", "right") else "left"
        self._last_state: Optional[List[float]] = None

        units = (
            (self.get_parameter("ros2_control_command_units").value or "radians").strip().lower()
        )
        if units not in ("hardware", "radians"):
            raise SystemExit(f"ros2_control_command_units 无效: {units}")
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
                raise SystemExit(
                    f"radians 模式初始化失败: {exc}；"
                    "请传 robot_description（G20 整机加 g20_joint_order:=l25_urdf）"
                ) from exc

        # 与侧化公开 API 一致：/{side}_hand/save_pose 等
        svc_prefix = side_hand_prefix(self._side)

        state_topic = self.get_parameter("state_topic").value.strip() or "/joint_states"
        self.create_subscription(JointState, state_topic, self._on_state, 10)
        if state_topic == "/joint_states":
            self.create_subscription(
                JointState,
                f"/cb_{self._side}_hand_state",
                self._on_legacy_state,
                10,
            )

        try:
            from hand_gesture_primitives_msgs.srv import (
                DeletePose,
                ExportPoseBundle,
                ExportPrimitivePoses,
                ImportGuiPresets,
                ImportPoseBundle,
                ListPoses,
                SavePose,
            )
        except ImportError as exc:
            raise SystemExit(
                f"需要先编译 hand_gesture_primitives_msgs 并 source 工作空间。\n{exc}"
            ) from exc

        self.create_service(SavePose, f"{svc_prefix}/save_pose", self._save)
        self.create_service(ListPoses, f"{svc_prefix}/list_poses", self._list)
        self.create_service(DeletePose, f"{svc_prefix}/delete_pose", self._delete)
        self.create_service(
            ImportGuiPresets,
            f"{svc_prefix}/import_gui_presets",
            self._import,
        )
        self.create_service(
            ExportPoseBundle,
            f"{svc_prefix}/export_pose_bundle",
            self._export_bundle,
        )
        self.create_service(
            ImportPoseBundle,
            f"{svc_prefix}/import_pose_bundle",
            self._import_bundle,
        )
        self.create_service(
            ExportPrimitivePoses,
            f"{svc_prefix}/export_primitive_poses",
            self._export_primitives,
        )
        self.get_logger().info(
            f"PoseLibraryServer: hand={self._hand_key} side={self._side} "
            f"units={self._units} pose_dir={pose_dir()} "
            f"services={svc_prefix}/* state={state_topic}"
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

    def _on_legacy_state(self, msg: JointState) -> None:
        hw = self._extract_hardware(msg)
        if hw is not None:
            self._last_state = hw

    def _on_state(self, msg: JointState) -> None:
        hw = self._extract_hardware(msg)
        if hw is not None:
            self._last_state = hw

    def _resolve_side(self, request_side: str) -> str:
        s = (request_side or "").strip().lower()
        if s in ("left", "right"):
            return s
        return self._side

    def _save(self, request, response):
        hand = resolve_hand_config_key(request.hand_type or self._hand_key)
        side = self._resolve_side(getattr(request, "hand_side", "") or "")
        positions = list(request.positions)
        if not positions:
            if self._last_state is None:
                response.success = False
                response.message = "无 positions 且尚无 hand_state 缓存"
                return response
            if hand != self._hand_key:
                response.success = False
                response.message = (
                    f"空 positions 时仅能用本节点缓存 hand={self._hand_key}，"
                    f"请求 hand={hand} 不一致"
                )
                return response
            positions = list(self._last_state)
        cmap = tip_mid_couple_map(hand, HandConfig(hand).joint_names)
        if cmap:
            positions = apply_tip_mid_coupling(positions, cmap)
        space = str(getattr(request, "space", "") or DEFAULT_SAVE_SPACE).strip().lower()
        converter = self._radian_converter if hand == self._hand_key else None
        prepared = prepare_pose_for_save(
            hand,
            positions,
            space=space,
            radian_converter=converter,
        )
        if isinstance(prepared, str):
            response.success = False
            response.message = prepared
            return response
        save_positions, space, unit = prepared
        keyframe = str(getattr(request, "keyframe", "") or "").strip().lower()
        err = commit_saved_pose(
            hand,
            request.name,
            save_positions,
            duration=float(request.duration or 0.6),
            description=str(request.description or ""),
            source="service",
            space=space,
            unit=unit,
            hand_side=side,
            keyframe=keyframe,
        )
        response.success = err is None
        if err:
            response.message = err
        elif keyframe:
            response.message = f"ok keyframe={keyframe}"
        else:
            response.message = "ok"
        return response

    def _list(self, request, response):
        hand = resolve_hand_config_key(request.hand_type or self._hand_key)
        side = self._resolve_side(getattr(request, "hand_side", "") or "")
        response.names = list_poses(hand, hand_side=side)
        response.success = True
        response.message = "ok"
        return response

    def _delete(self, request, response):
        hand = resolve_hand_config_key(request.hand_type or self._hand_key)
        side = self._resolve_side(getattr(request, "hand_side", "") or "")
        ok = delete_pose(hand, request.name, hand_side=side)
        response.success = ok
        response.message = "ok" if ok else f"pose_missing:{request.name}"
        return response

    def _import(self, request, response):
        hand = resolve_hand_config_key(request.hand_type or self._hand_key)
        side = self._resolve_side(getattr(request, "hand_side", "") or "")
        imported, skipped, message = import_gui_presets(
            hand,
            request.presets_path,
            overwrite=bool(request.overwrite),
            hand_side=side,
        )
        response.imported = int(imported)
        response.skipped = int(skipped)
        # 与 pose_bundle 一致：无实际写入则失败（全 skip / 读失败）
        response.success = imported > 0
        response.message = message or ("ok" if imported > 0 else "no poses imported")
        return response

    def _export_bundle(self, request, response):
        hand = resolve_hand_config_key(request.hand_type or self._hand_key)
        side = self._resolve_side(getattr(request, "hand_side", "") or "")
        output_path = str(request.output_path or "").strip()
        if not output_path:
            response.success = False
            response.message = "output_path is required"
            response.count = 0
            return response
        names = [str(n).strip() for n in (request.names or []) if str(n).strip()]
        try:
            path, count = export_library_to_file(
                output_path,
                hand_type=hand,
                hand_side=side,
                names=names or None,
                exporter="pose_library_server",
            )
        except Exception as exc:  # noqa: BLE001 — service boundary
            response.success = False
            response.message = str(exc) or type(exc).__name__
            response.count = 0
            return response
        response.success = True
        response.message = f"ok:{path}"
        response.count = int(count)
        return response

    def _import_bundle(self, request, response):
        input_path = str(request.input_path or "").strip()
        if not input_path:
            response.success = False
            response.message = "input_path is required"
            response.imported = 0
            response.skipped = 0
            response.failed = 0
            return response
        strategy = str(request.strategy or "skip").strip().lower() or "skip"
        hand_filter = str(request.hand_type or "").strip() or None
        side_filter = str(getattr(request, "hand_side", "") or "").strip() or None
        default_side = str(getattr(request, "default_side", "") or "").strip() or self._side
        try:
            result = import_bundle_file(
                input_path,
                strategy=strategy,
                hand_type=hand_filter,
                hand_side=side_filter,
                default_side=default_side,
            )
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc) or type(exc).__name__
            response.imported = 0
            response.skipped = 0
            response.failed = 0
            return response
        response.imported = int(result.imported)
        response.skipped = int(result.skipped)
        response.failed = int(result.failed)
        response.success = result.success
        response.message = result.summary
        return response

    def _export_primitives(self, request, response):
        hand = resolve_hand_config_key(request.hand_type or self._hand_key)
        side = self._resolve_side(getattr(request, "hand_side", "") or "")
        primitives = [str(p).strip() for p in (request.primitives or []) if str(p).strip()]
        name_prefix = str(request.name_prefix or "preset_").strip() or "preset_"
        write_library = bool(request.write_library)
        output_path = str(request.output_path or "").strip()
        if not write_library and not output_path:
            response.success = False
            response.message = "specify output_path and/or write_library=true"
            response.count = 0
            response.names = []
            return response
        try:
            count, path, names = export_primitive_poses(
                hand,
                side,
                primitives=primitives or None,
                name_prefix=name_prefix,
                write_library=write_library,
                output_path=output_path or None,
                overwrite=bool(request.overwrite),
            )
        except Exception as exc:  # noqa: BLE001
            response.success = False
            response.message = str(exc) or type(exc).__name__
            response.count = 0
            response.names = []
            return response
        response.success = True
        response.message = f"ok:{path}" if path else "ok"
        response.count = int(count)
        response.names = list(names)
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PoseLibraryServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
