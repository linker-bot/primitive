"""手势原语 ROS2 控制节点。

公开 API（空 topic 参数自动侧化，side=left|right）:
  /{side}_hand/gesture_cmd[_exec]
  /{side}_hand/gesture_status|gesture_result
  /{side}_hand/tcp_pose|object_pose
"""

import os
import time
from pathlib import Path
from typing import Union

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterType
from rclpy.node import Node
from std_msgs.msg import Float32, String

from .contact_config import (
    DEFAULT_CONTROL_RATE_HZ,
    ContactThresholds,
    clamp_control_rate_hz,
    load_contact_thresholds,
)
from .executor import ControlMode, GestureExecutor
from .grasp_mapping import GRASP_TYPE_TO_PRIMITIVE
from .hand_profile import (
    format_reject_message,
    list_configured_hand_joints,
    load_hand_profile,
)
from .label_utils import normalize_label
from .planable_primitives import planable_reject_reason
from .pose_library import list_poses, load_pose, pose_dir
from .primitive_base import (
    GRASP_PHASES,
    LARGE_WRAP_PHASES,
    PHASED_GRASP_PRIMITIVES,
    TRIPOD_PHASED_PRIMITIVES,
    TRIPOD_PHASES,
)
from .primitive_catalog import gated_primitive_names
from .primitive_reloader import PrimitiveHotReloader
from .primitives import PRIMITIVE_REGISTRY
from .progress_policy import (
    POSITION_PROGRESS_PRIMITIVES,
    validate_primitive_progress,
)
from .topic_defaults import (
    bboxes_3d_topic,
    labeled_poses_topic,
    resolve_side_topic,
    side_hand_topics,
)
from .trajectory_planner import parse_pose_plan_ref as _parse_plan_pose_ref

# 门控集合：派生自 catalog（勿在 grasp_gate 再维护一份）
GATED_PRIMITIVES = gated_primitive_names()


class GestureNode(Node):
    """接收手势指令并调度对应原语执行。"""

    _STATUS_OVERRIDE_SEC = 3.0

    @staticmethod
    def _coerce_numeric(value: Union[int, float, str], kind: str):
        if kind == "int":
            return int(value)
        return float(value)

    def _declare_numeric_parameter(
        self,
        name: str,
        default: Union[int, float],
        kind: str = "float",
    ) -> None:
        """Declare float/int param; coerce launch INTEGER overrides to float."""
        if self.has_parameter(name):
            existing = self.get_parameter(name)
            if existing.type_ == ParameterType.PARAMETER_NOT_SET:
                pass
            elif kind == "float" and existing.type_ == ParameterType.PARAMETER_INTEGER:
                coerced = float(existing.value)
                self.undeclare_parameter(name)
                self.declare_parameter(name, coerced)
                return
            elif existing.type_ in (
                ParameterType.PARAMETER_DOUBLE,
                ParameterType.PARAMETER_INTEGER,
            ):
                self.undeclare_parameter(name)
                self.declare_parameter(name, self._coerce_numeric(existing.value, kind))
                return
        self.declare_parameter(name, default)

    def __init__(self):
        # Resolve station storage before constructing any ROS command publishers.
        pose_dir()
        super().__init__("hand_gesture_node")

        self.declare_parameter("hand_side", "left")
        self.declare_parameter("hand_type", "o20")
        self.declare_parameter("hand_joint", "")
        # Empty => auto /{side}_hand/... (dual-hand safe). Set explicitly for legacy globals.
        self.declare_parameter("cmd_topic", "")
        self.declare_parameter("tcp_pose_topic", "")
        self.declare_parameter("object_pose_topic", "")
        self.declare_parameter("bboxes_3d_topic", bboxes_3d_topic())
        self.declare_parameter("labeled_pose_topic", labeled_poses_topic())
        self.declare_parameter("target_object_label", "")
        self.declare_parameter("result_topic", "")
        self.declare_parameter("selected_label_topic", "")
        self.declare_parameter("status_topic", "")
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("cmd_backend", "")
        self.declare_parameter("hand_cmd_topic", "")
        self.declare_parameter("ros2_control_command_units", "radians")
        self.declare_parameter("robot_description", "")
        # G20：整机 competition_device 用 l25_urdf；手-only SDK 名用 native；空则探测 URDF/状态
        self.declare_parameter("g20_joint_order", "")
        self._declare_numeric_parameter("control_rate", DEFAULT_CONTROL_RATE_HZ)
        self._declare_numeric_parameter("primitive_speed", 1.0)
        self.declare_parameter("reach_check_strict", False)
        self.declare_parameter("reach_max_distance", 0.15)
        self.declare_parameter("palm_offset_x", 0.0)
        self.declare_parameter("palm_offset_y", 0.0)
        self.declare_parameter("palm_offset_z", -0.05)
        self.declare_parameter("urdf_path", "")
        self.declare_parameter("publish_fingertips", True)
        self.declare_parameter("fingertip_frame", "")
        self.declare_parameter("contact_config_path", "")
        self.declare_parameter("stream_target_topic", "")
        self.declare_parameter(
            "runtime_status_path",
            os.environ.get(
                "HAND_GESTURE_RUNTIME_STATUS_PATH",
                str(Path.home() / ".local/state/hand_gesture_primitives/runtime-status.json"),
            ),
        )
        self.declare_parameter("motion_progress_topic", "")
        self._declare_numeric_parameter("stream_watchdog_sec", 0.5, "float")
        config_path = self.get_parameter("contact_config_path").value.strip()
        loaded_thresholds = load_contact_thresholds(config_path or None)
        self._declare_numeric_parameter(
            "contact_pressure_threshold", loaded_thresholds.pressure_threshold
        )
        self._declare_numeric_parameter("contact_mass_threshold", loaded_thresholds.mass_threshold)
        self._declare_numeric_parameter("contact_current_delta", loaded_thresholds.current_delta)
        self._declare_numeric_parameter(
            "contact_current_delta_narrow", loaded_thresholds.current_delta_narrow
        )
        self._declare_numeric_parameter(
            "contact_current_settle_frames",
            loaded_thresholds.current_settle_frames,
            kind="int",
        )
        self._declare_numeric_parameter("hold_safe_current", loaded_thresholds.hold_safe_current)
        self._declare_numeric_parameter(
            "overload_current_threshold", loaded_thresholds.overload_threshold
        )
        self._declare_numeric_parameter(
            "overload_duration_sec", loaded_thresholds.overload_duration_sec
        )

        self._side = self.get_parameter("hand_side").value
        hand_type = self.get_parameter("hand_type").value
        hand_joint = self.get_parameter("hand_joint").value.strip()
        if hand_joint:
            hand_type = hand_joint.lower()
        self._hand_type = hand_type

        side_topics = side_hand_topics(self._side)
        # Default empty -> gesture_cmd (no-gate UX). With gate, launch sets cmd_exec.
        cmd_topic = resolve_side_topic(
            self.get_parameter("cmd_topic").value, side_topics["gesture_cmd"]
        )
        tcp_topic = resolve_side_topic(
            self.get_parameter("tcp_pose_topic").value, side_topics["tcp_pose"]
        )
        obj_topic = resolve_side_topic(
            self.get_parameter("object_pose_topic").value, side_topics["object_pose"]
        )
        result_topic = resolve_side_topic(
            self.get_parameter("result_topic").value, side_topics["gesture_result"]
        )
        sel_label_topic = resolve_side_topic(
            self.get_parameter("selected_label_topic").value, side_topics["selected_label"]
        )
        status_topic = resolve_side_topic(
            self.get_parameter("status_topic").value, side_topics["gesture_status"]
        )
        bboxes_topic = self.get_parameter("bboxes_3d_topic").value
        labeled_topic = self.get_parameter("labeled_pose_topic").value
        target_label = self.get_parameter("target_object_label").value
        state_topic = self.get_parameter("state_topic").value
        cmd_backend = self.get_parameter("cmd_backend").value
        hand_cmd_topic = self.get_parameter("hand_cmd_topic").value
        ros2_control_command_units = self.get_parameter("ros2_control_command_units").value
        robot_description = self.get_parameter("robot_description").value
        g20_joint_order = self.get_parameter("g20_joint_order").value
        control_rate = clamp_control_rate_hz(self.get_parameter("control_rate").value)
        reach_strict = self.get_parameter("reach_check_strict").value
        reach_max_dist = self.get_parameter("reach_max_distance").value
        palm_offset = np.array(
            [
                self.get_parameter("palm_offset_x").value,
                self.get_parameter("palm_offset_y").value,
                self.get_parameter("palm_offset_z").value,
            ]
        )
        urdf_path = self.get_parameter("urdf_path").value
        publish_fingertips = self.get_parameter("publish_fingertips").value
        fingertip_frame = self.get_parameter("fingertip_frame").value
        stream_target_topic = self.get_parameter("stream_target_topic").value
        stream_watchdog_sec = float(self.get_parameter("stream_watchdog_sec").value)
        runtime_status_path = self.get_parameter("runtime_status_path").value
        motion_progress_topic = resolve_side_topic(
            self.get_parameter("motion_progress_topic").value,
            side_topics["motion_progress"],
        )

        contact_thresholds = ContactThresholds(
            pressure_threshold=float(self.get_parameter("contact_pressure_threshold").value),
            mass_threshold=float(self.get_parameter("contact_mass_threshold").value),
            current_delta=float(self.get_parameter("contact_current_delta").value),
            current_delta_narrow=float(self.get_parameter("contact_current_delta_narrow").value),
            current_settle_frames=int(self.get_parameter("contact_current_settle_frames").value),
            hold_safe_current=float(self.get_parameter("hold_safe_current").value),
            overload_threshold=float(self.get_parameter("overload_current_threshold").value),
            overload_duration_sec=float(self.get_parameter("overload_duration_sec").value),
        ).scaled_for_control_rate(control_rate)

        if self._side not in ("left", "right"):
            self.get_logger().error(f"hand_side 参数无效: '{self._side}'，必须为 left 或 right")
            raise ValueError(f"Invalid hand_side: {self._side}")

        try:
            self._profile = load_hand_profile(self._hand_type)
        except FileNotFoundError as exc:
            available = list_configured_hand_joints()
            self.get_logger().error(
                f"hand_type 无效: '{self._hand_type}'，"
                f"未找到 config/{{hand}}.yaml；可用: {available}"
            )
            raise ValueError(str(exc)) from exc

        self._hand_type = self._profile.hand_type

        self.get_logger().info(f"手势节点启动: side={self._side}, hand_type={self._hand_type}")
        for line in self._profile.summary_lines():
            self.get_logger().info(line)
        n_prims = len(PRIMITIVE_REGISTRY)
        self.get_logger().info(f"全局已注册原语 ({n_prims}): {sorted(PRIMITIVE_REGISTRY.keys())}")
        self.get_logger().info(
            f"包络门控原语 ({len(GATED_PRIMITIVES)}): {sorted(GATED_PRIMITIVES)}"
        )
        self.get_logger().info(
            f"grasp_type 自动映射: {GRASP_TYPE_TO_PRIMITIVE} (命令: auto [label])"
        )

        self.get_logger().info(
            f"接触/电流阈值: delta={contact_thresholds.current_delta}mA, "
            f"narrow={contact_thresholds.current_delta_narrow}mA, "
            f"hold_safe={contact_thresholds.hold_safe_current}mA, "
            f"overload={contact_thresholds.overload_threshold}mA, "
            f"settle_frames={contact_thresholds.current_settle_frames} "
            f"(@{control_rate:g}Hz)"
        )
        self.get_logger().info(f"原语控制下发频率: {control_rate:g} Hz")

        self._executor = GestureExecutor(
            self,
            self._side,
            hand_type=self._hand_type,
            hand_profile=self._profile,
            tcp_pose_topic=tcp_topic,
            object_pose_topic=obj_topic,
            bboxes_3d_topic=bboxes_topic,
            labeled_pose_topic=labeled_topic,
            target_object_label=target_label,
            result_topic=result_topic,
            selected_label_topic=sel_label_topic,
            state_topic=state_topic,
            cmd_backend=cmd_backend,
            hand_cmd_topic=hand_cmd_topic,
            ros2_control_command_units=ros2_control_command_units,
            robot_description=robot_description,
            g20_joint_order=g20_joint_order,
            reach_strict=reach_strict,
            reach_max_dist=reach_max_dist,
            palm_offset=palm_offset,
            urdf_path=urdf_path,
            publish_fingertips=publish_fingertips,
            fingertip_frame=fingertip_frame,
            contact_thresholds=contact_thresholds,
            stream_target_topic=stream_target_topic,
            stream_watchdog_sec=stream_watchdog_sec,
            control_rate=control_rate,
        )

        initial_speed = float(self.get_parameter("primitive_speed").value)
        if not 0.25 <= initial_speed <= 3.0:
            raise ValueError("primitive_speed must be finite and within 0.25..3.0")
        self._executor._configured_speed = initial_speed
        self.add_on_set_parameters_callback(self._motion_parameters)

        try:
            from hand_gesture_primitives_msgs.srv import PlanGesture
        except ImportError as exc:
            raise RuntimeError(
                "需要先编译 hand_gesture_primitives_msgs 并 source 工作空间"
            ) from exc
        self.create_service(
            PlanGesture,
            f"/{self._side}_hand/plan_gesture",
            self._plan_gesture,
        )

        self.get_logger().info(
            f"订阅指令 topic: {cmd_topic}, tcp={tcp_topic}, object={obj_topic}, "
            f"status={status_topic}, 感知 topic: {labeled_topic}"
        )
        # 公开 API 始终是 gesture_cmd；有 gate 时本节点订的是 gesture_cmd_exec。
        pub_hint_side = self.get_parameter("hand_side").value
        self.get_logger().info(
            "下发指令请用公开 topic /{0}_hand/gesture_cmd；"
            "若 `ros2 topic pub` 卡在 Waiting for subscription，改用: "
            "ros2 run hand_gesture_primitives pub_gesture_cmd -- {0} open".format(pub_hint_side)
        )

        self.create_subscription(String, cmd_topic, self._cmd_callback, 10)
        self.create_subscription(
            Float32, motion_progress_topic, self._motion_progress_callback, 10
        )
        self._motion_progress_warn_t = 0.0
        self.get_logger().info(
            f"motion_progress topic: {motion_progress_topic} "
            f"(Float32；仅热更新当前 pose 段内 progress，切换关键帧仍用 gesture_cmd)"
        )

        self._cmd_topic = cmd_topic
        self._status_pub = self.create_publisher(String, status_topic, 10)
        self._status_override = ""
        self._status_override_until = 0.0
        self._primitive_reloader = PrimitiveHotReloader(runtime_status_path)
        self._reload_was_deferred = False
        self.create_timer(0.5, self._reload_primitives_if_changed)
        self.create_timer(0.5, self._publish_status)
        from .workbench_support import WorkbenchBridge
        self._workbench = WorkbenchBridge(self)

    def destroy_node(self):
        if hasattr(self, "_workbench"):
            self._workbench.close()
        return super().destroy_node()

    def _motion_parameters(self, parameters):
        import math

        from rcl_interfaces.msg import SetParametersResult

        speed = None
        for parameter in parameters:
            if parameter.name == "primitive_speed":
                if isinstance(parameter.value, bool) or not isinstance(parameter.value, (int, float)) or not math.isfinite(parameter.value) or not 0.25 <= parameter.value <= 3.0:
                    return SetParametersResult(successful=False, reason="primitive_speed must be finite and within 0.25..3.0")
                speed = float(parameter.value)
            if parameter.name == "control_rate":
                return SetParametersResult(successful=False, reason="control_rate is fixed at startup; use primitive_speed for motion speed")
        if speed is not None:
            # Integrated clock changes only subsequent motion increments.
            self._executor.set_motion_speed(speed)
        return SetParametersResult(successful=True)

    def _plan_gesture(self, request, response):
        raw = str(request.primitive or "").strip().lower()
        pose_name = _parse_plan_pose_ref(raw)
        sample_period = float(request.sample_period or 0.04)
        if pose_name:
            try:
                planned = self._executor.plan_pose_trajectory(
                    pose_name,
                    sample_period=sample_period,
                )
            except Exception as exc:
                response.success = False
                response.message = str(exc) or type(exc).__name__
                return response
            positions = planned["positions"]
            response.success = True
            response.message = "planned"
            response.joint_names = list(planned["joint_names"])
            response.time_s = [float(value) for value in planned["time_s"]]
            response.positions = [float(value) for point in positions for value in point]
            response.point_count = len(positions)
            response.nominal_only = True
            segments = planned.get("segments")
            if segments:
                import json

                response.segments_json = json.dumps(segments, ensure_ascii=False)
            return response

        primitive = self._profile.resolve_primitive_name(raw)
        reject = planable_reject_reason(self._hand_type, primitive)
        if reject:
            response.success = False
            response.message = reject
            return response
        profile_reject = self._profile.reject_reason(primitive)
        if profile_reject.startswith(
            ("unknown_primitive", "unsupported_on_", "not_in_allowlist", "capability_missing")
        ):
            response.success = False
            response.message = f"primitive not allowed on this hand: {profile_reject}"
            return response
        try:
            planned = self._executor.plan_static_primitive(
                primitive,
                sample_period=sample_period,
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc) or type(exc).__name__
            return response
        positions = planned["positions"]
        response.success = True
        response.message = "planned"
        response.joint_names = list(planned["joint_names"])
        response.time_s = [float(value) for value in planned["time_s"]]
        response.positions = [float(value) for point in positions for value in point]
        response.point_count = len(positions)
        response.nominal_only = True
        return response

    def _set_status_override(self, message: str) -> None:
        self._status_override = message
        self._status_override_until = time.monotonic() + self._STATUS_OVERRIDE_SEC

    def _report_primitive_reject(self, prim_name: str, reason: str) -> None:
        """记录并发布原语被拒绝的用户可读说明。"""
        user_msg = self._profile.reject_user_message(prim_name)
        if not user_msg:
            user_msg = format_reject_message(
                self._profile.hand_joint,
                self._hand_type,
                reason,
                num_joints=self._profile.hardware.num_joints,
            )
        self.get_logger().warn(f"{user_msg} [{reason}]")
        self._set_status_override(user_msg)

    def _motion_progress_callback(self, msg: Float32) -> None:
        """高频段内 progress 热更新；切换 keyframe / 启停仍用 gesture_cmd。"""
        err = self._executor.apply_motion_progress(float(msg.data))
        if err is None:
            return
        # 无 active pose / 非 PRIMITIVE 时静默忽略，避免刷屏；仅偶发提示
        now = time.monotonic()
        if now - self._motion_progress_warn_t < 2.0:
            return
        self._motion_progress_warn_t = now
        self.get_logger().warn(f"motion_progress 未应用: {err}")

    def _cmd_callback(self, msg: String) -> None:
        parts = msg.data.strip().lower().split()
        if not parts:
            return
        cmd = parts[0]
        args = parts[1:]
        cmd = self._profile.resolve_primitive_name(cmd)
        self.get_logger().info(f"收到指令: '{msg.data.strip()}' (topic: {self._cmd_topic})")

        if cmd == "stop":
            self._executor.stop()
            return

        if cmd == "reload_primitives":
            self._reload_primitives_if_changed(force=True)
            return

        if cmd == "mode":
            self._handle_mode_cmd(args)
            return

        if cmd == "pose_list":
            names = list_poses(self._hand_type, hand_side=self._side)
            listing = ",".join(names) if names else ""
            # 机器可读：POSES:a,b（pub_gesture_cmd / teach_panel / 脚本订阅）
            result = String()
            result.data = f"POSES:{listing}"
            self._executor.publish_user_result(result.data)
            self._set_status_override(result.data)
            pretty = f"Pose Library ({self._hand_type}/{self._side}): " + (
                ", ".join(names) if names else "(空)"
            )
            self.get_logger().info(pretty)
            # 提示用户用 CLI 看 stdout，避免只翻 launch 日志
            self.get_logger().info(
                "查询结果已发到 gesture_result；终端可读: "
                f"ros2 run hand_gesture_primitives pub_gesture_cmd -- {self._side} pose_list"
            )
            return

        if cmd == "pose":
            self._handle_pose_cmd(args)
            return

        if cmd == "named_pose":
            self.get_logger().warn("请使用 `pose <name>` 回放示教姿态，勿直接发 named_pose")
            return

        if cmd == "target":
            label = normalize_label(args[0]) if args else ""
            self._executor.set_target_label(label)
            self.get_logger().info(f"目标物体过滤: '{label}' (空=最高score)")
            return

        if cmd == "target_id":
            if not args:
                self._executor.set_target_instance_id(0)
                self.get_logger().info("清除 instance_id 过滤")
                return
            try:
                instance_id = int(args[0])
            except ValueError:
                self.get_logger().warn(f"target_id 需要整数: '{args[0]}'")
                return
            self._executor.set_target_instance_id(instance_id)
            self.get_logger().info(f"目标 instance_id: {instance_id}")
            return

        if self._executor.control_mode in (ControlMode.TEACH, ControlMode.STREAM):
            mode = self._executor.control_mode.value
            self.get_logger().warn(
                f"当前为 {mode.upper()} 模式，拒绝执行 '{cmd}'；请先 `mode primitive`"
            )
            self._set_status_override(f"rejected:{mode}_mode")
            return

        if cmd == "auto":
            self._handle_auto_cmd(args)
            return

        primitive_cls = PRIMITIVE_REGISTRY.get(cmd)
        if primitive_cls is None:
            hint = self._unknown_primitive_hint(cmd)
            supported = self._profile.supported_primitives()
            self.get_logger().warn(
                f"未知原语指令: '{cmd}'，"
                f"本手型支持: {supported}；全局: {sorted(PRIMITIVE_REGISTRY.keys()) + ['auto', 'pose', 'mode']}"
                + (f"；{hint}" if hint else "")
            )
            return

        reject = self._profile.reject_reason(cmd)
        if reject.startswith(
            ("unknown_primitive", "unsupported_on_", "not_in_allowlist", "capability_missing")
        ):
            self._report_primitive_reject(cmd, reject)
            return
        if reject.startswith("capability_warning:"):
            self.get_logger().warn(f"原语能力警告: {reject}")

        phase, target_label, prim_args, progress = self._parse_primitive_args(cmd, args)

        try:
            if cmd in PHASED_GRASP_PRIMITIVES:
                leftover = [
                    a
                    for a in prim_args
                    if not str(a).lower().startswith("progress=")
                    and not str(a).lstrip("-").replace(".", "", 1).isdigit()
                ]
                if leftover:
                    phase_hint = (
                        "prep/close/auto/full"
                        if cmd in TRIPOD_PHASED_PRIMITIVES or cmd == "large_wrap"
                        else "prep/close/full"
                    )
                    raise ValueError(
                        f"{cmd} 仅支持 phase ({phase_hint}) 与可选 progress，多余参数: {leftover}"
                    )
                progress_err = validate_primitive_progress(cmd, phase, progress)
                if progress_err:
                    self.get_logger().warn(progress_err)
                    self._set_status_override(f"rejected:{progress_err}")
                    return
                if progress is not None and cmd in POSITION_PROGRESS_PRIMITIVES:
                    primitive = primitive_cls(phase=phase, progress=progress)
                else:
                    primitive = primitive_cls(phase=phase)
            else:
                if phase != "full":
                    self.get_logger().warn(f"原语 {cmd} 不支持 prep/close，忽略 phase={phase}")
                if progress is not None:
                    self.get_logger().warn(
                        f"原语 {cmd} 不支持 progress，请用 pose <name> <kf> <p>"
                    )
                    self._set_status_override("rejected:progress_not_supported")
                    return
                primitive = primitive_cls(*prim_args)
        except (TypeError, ValueError) as e:
            self.get_logger().warn(f"原语参数错误: '{msg.data}' → {e}")
            return

        reject_reason = self._executor.set_primitive(primitive, target_label=target_label or None)
        if reject_reason:
            self.get_logger().warn(f"原语被拒绝: {reject_reason}")
            self._set_status_override(reject_reason)

    def _handle_auto_cmd(self, args):
        phase = "full"
        label_args = []
        for a in args:
            if a.lower() in GRASP_PHASES:
                phase = a.lower()
            else:
                label_args.append(a)
        label = normalize_label(label_args[0]) if label_args else ""
        snapshot = self._executor.target_selection_snapshot()
        accepted = False
        try:
            if label:
                self._executor.set_target_label(label, invalidate=False)
                self._executor.refresh_target_from_cached_bbox()
            prim_name = self._executor.suggested_primitive_name()
            if not prim_name:
                grasp = self._executor.grasp_type or "(无)"
                self.get_logger().warn(
                    f"auto 失败: 无可用 grasp_type={grasp}，请先确保 bboxes_3d 已发布或指定 label"
                )
                self._set_status_override(f"rejected:no_grasp_type:{grasp}")
                return
            primitive_cls = PRIMITIVE_REGISTRY.get(prim_name)
            if primitive_cls is None:
                self.get_logger().error(f"映射原语未注册: {prim_name}")
                return
            reject = self._profile.reject_reason(prim_name)
            if reject.startswith(
                ("unknown_primitive", "unsupported_on_", "not_in_allowlist", "capability_missing")
            ):
                self._report_primitive_reject(prim_name, reject)
                return
            if reject.startswith("capability_warning:"):
                self.get_logger().warn(f"原语能力警告: {reject}")
            phase_info = f", phase={phase}" if prim_name in PHASED_GRASP_PRIMITIVES else ""
            self.get_logger().info(
                f"auto: grasp_type={self._executor.grasp_type} → {prim_name}{phase_info}"
                + (f", label={label}" if label else "")
            )
            if prim_name in PHASED_GRASP_PRIMITIVES:
                primitive = primitive_cls(phase=phase)
            else:
                if phase != "full":
                    self.get_logger().warn(
                        f"原语 {prim_name} 不支持 prep/close，忽略 phase={phase}"
                    )
                primitive = primitive_cls()
            reject_reason = self._executor.set_primitive(primitive)
            accepted = not reject_reason
            if reject_reason:
                self.get_logger().warn(f"原语被拒绝: {reject_reason}")
                self._set_status_override(reject_reason)
            return

        finally:
            if not accepted:
                self._executor.restore_target_selection(snapshot)

    def _reload_primitives_if_changed(self, force: bool = False) -> None:
        idle = self._executor.can_reload_primitives
        try:
            result = self._primitive_reloader.reload_if_changed(
                idle=idle,
                force=force,
            )
        except Exception as error:
            message = f"primitive_reload_failed:{type(error).__name__}:{error}"
            self.get_logger().error(message)
            self._executor.publish_user_result(message)
            self._set_status_override(message)
            return
        if result == "deferred":
            if not self._reload_was_deferred:
                self.get_logger().info(
                    "Primitive source changed; waiting for idle primitive mode before reload"
                )
                self._reload_was_deferred = True
            return
        if result != "reloaded":
            return
        self._reload_was_deferred = False
        self._executor.advance_control_epoch()
        message = f"PRIMITIVES_RELOADED:{len(PRIMITIVE_REGISTRY)}"
        self.get_logger().info(f"Debug gesture primitives reloaded: {sorted(PRIMITIVE_REGISTRY)}")
        self._executor.publish_user_result(message)
        self._set_status_override(message)

    def _parse_primitive_args(self, cmd: str, args: list) -> tuple:
        """解析原语命令参数，返回 (phase, target_label, prim_args, progress)。"""
        if cmd == "large_wrap":
            phase = "auto"
            phase_values = LARGE_WRAP_PHASES
        elif cmd in TRIPOD_PHASED_PRIMITIVES:
            phase = "auto"
            phase_values = TRIPOD_PHASES
        else:
            phase = "full"
            phase_values = GRASP_PHASES
        target_label = ""
        prim_args = []
        progress = None
        phased = cmd in PHASED_GRASP_PRIMITIVES

        for a in args:
            al = a.lower()
            if al.startswith("progress="):
                try:
                    progress = float(al.split("=", 1)[1])
                except ValueError:
                    prim_args.append(a)
                continue
            if phased and al in phase_values:
                phase = al
                continue
            # 纯数字：phased 时作 progress；否则留给 prim_args
            try:
                num = float(a)
            except ValueError:
                num = None
            if num is not None and phased:
                progress = num
                continue
            if (
                not target_label
                and a not in PRIMITIVE_REGISTRY
                and al not in GRASP_PHASES
                and al not in TRIPOD_PHASES
                and not a.lstrip("-").replace(".", "", 1).isdigit()
            ):
                target_label = normalize_label(a)
            else:
                prim_args.append(a)
        return phase, target_label, prim_args, progress

    def _unknown_primitive_hint(self, cmd: str) -> str:
        """原语名不匹配时给出可操作的提示。"""
        if load_pose(self._hand_type, cmd, hand_side=self._side) is not None:
            return f"若为自定义姿态请用 `pose {cmd}`（勿省略 pose 前缀）"
        if cmd.endswith("_by_vision"):
            short = cmd[: -len("_by_vision")]
            if short in PRIMITIVE_REGISTRY:
                return f"可尝试短名 '{short}'（不经门控/旧版别名）"
            return "当前版本应含 *_by_vision，请 colcon build 并 source install"
        vision_name = f"{cmd}_by_vision"
        if vision_name in PRIMITIVE_REGISTRY:
            return f"产线推荐 '{vision_name}'（经 GraspGate 门控）"
        return ""

    def _handle_mode_cmd(self, args: list) -> None:
        if not args:
            self.get_logger().info(
                f"当前控制模式: {self._executor.control_mode.value} "
                f"（可用: teach|primitive|stream）"
            )
            return
        mode_name = args[0].strip().lower()
        if mode_name == "stream":
            self._executor.set_control_mode(ControlMode.STREAM)
            return
        if mode_name == "teach":
            self._executor.set_control_mode(ControlMode.TEACH)
            return
        if mode_name == "primitive":
            self._executor.set_control_mode(ControlMode.PRIMITIVE)
            return
        self.get_logger().warn(f"未知模式 '{mode_name}'，可用: teach|primitive|stream")

    def _handle_pose_cmd(self, args: list) -> None:
        if not args:
            self.get_logger().warn(
                "用法: pose <name> [<keyframe>] [<progress>|auto] 或 pose <name> progress=<float>"
            )
            return
        if self._executor.control_mode in (ControlMode.TEACH, ControlMode.STREAM):
            self.get_logger().warn(
                f"当前为 {self._executor.control_mode.value.upper()} 模式，"
                "拒绝 pose；请先 `mode primitive`"
            )
            self._set_status_override(f"rejected:{self._executor.control_mode.value}_mode")
            return
        reject = self._profile.reject_reason("named_pose")
        if reject.startswith(
            ("unknown_primitive", "unsupported_on_", "not_in_allowlist", "capability_missing")
        ):
            self._report_primitive_reject("named_pose", reject)
            return
        pose_name = args[0].strip().lower()
        keyframe: str | None = None
        progress: float | None = None
        auto_segment = False
        for tok in args[1:]:
            t = str(tok).strip()
            tl = t.lower()
            if tl in ("auto", "auto_segment", "segment"):
                auto_segment = True
                continue
            if tl.startswith("progress="):
                try:
                    progress = float(tl.split("=", 1)[1])
                except ValueError:
                    self.get_logger().warn(f"progress 无效: {t}")
                    self._set_status_override("rejected:bad_progress")
                    return
                continue
            # 纯数字 → progress
            try:
                progress = float(t)
                continue
            except ValueError:
                pass
            if keyframe is None and t.replace("_", "").isalnum():
                keyframe = tl
                continue
            self.get_logger().warn(f"pose 忽略无法识别的参数: {t}")

        if load_pose(self._hand_type, pose_name, hand_side=self._side) is None:
            available = list_poses(self._hand_type, hand_side=self._side)
            self.get_logger().warn(
                f"姿态不存在: '{pose_name}'；可用: {available if available else '(空)'}"
            )
            self._set_status_override(f"rejected:pose_missing:{pose_name}")
            return
        if keyframe:
            entry = load_pose(self._hand_type, pose_name, hand_side=self._side)
            names = [kf.name for kf in entry.effective_keyframes()] if entry else []
            if keyframe not in names:
                self.get_logger().warn(
                    f"关键帧不存在: '{keyframe}'；可用: {names if names else '(空)'}"
                )
                self._set_status_override(f"rejected:keyframe_missing:{keyframe}")
                return
        if auto_segment and not keyframe:
            self.get_logger().warn("pose … auto 需要指定 keyframe")
            self._set_status_override("rejected:auto_needs_keyframe")
            return
        if auto_segment and progress is not None:
            self.get_logger().warn("pose … auto 与 progress 互斥，已忽略 auto")
            auto_segment = False
        reject_reason = self._executor.set_named_pose(
            pose_name,
            keyframe=keyframe,
            progress=progress,
            auto_segment=auto_segment,
        )
        if reject_reason:
            self.get_logger().warn(f"原语被拒绝: {reject_reason}")
            self._set_status_override(reject_reason)
            return
        motion = self._executor.active_motion_status
        if motion:
            self.get_logger().info(f"named_pose 状态: {motion}")

    def _publish_status(self) -> None:
        msg = String()
        if time.monotonic() < self._status_override_until:
            msg.data = self._status_override
        else:
            active = self._executor.active_primitive_name
            grasp_state = self._executor.grasp_state
            grasp = self._executor.grasp_type
            suggested = self._profile.primitive_for_grasp_type(grasp) if grasp else ""
            parts = [active, f"mode={self._executor.control_mode.value}",
                     f"speed={getattr(self._executor, '_primitive_speed', 1.0)}",
                     f"control_rate={self._executor._control_rate}",
                     f"settle={getattr(self._executor, '_settle_status', 'idle')}"]
            motion = self._executor.active_motion_status
            if motion:
                parts.append(motion)
            if grasp_state:
                parts.append(f"grasp={grasp_state}")
            elif suggested and active in ("hold", "init"):
                parts.append(f"type={grasp}")
                parts.append(f"suggest={suggested}")
            msg.data = "|".join(parts)
        self._status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
