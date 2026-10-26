#!/usr/bin/env python3
"""抓取可行性门控节点 — 三条件判定 (H / XY / Fwd) + 去抖。

判定逻辑 (与 auto_grasp_node 一致):
1. H (高度):  物体 3D bbox 世界 Z 范围 ∩ 手世界 Z 可达范围 ≠ ∅
2. XY (掌面): 物体中心在手坐标系掌面矩形内 (±half_w × ±half_h)
3. Fwd (前方): 物体在手坐标系中 cz > 0 (手指伸出方向)

坐标系转换: 世界 → TCP → R_z(90°) → 手坐标系
  hand +X = -TCP_Y (掌面左右),  hand +Y = +TCP_X (掌面上下),  hand +Z = +TCP_Z (手指前方)

包络原语需通过门控，其余原语直接放行。
门控集合以 ``primitive_catalog`` 的 ``gated=True`` 为单一真相源
（当前 11 个；``index_ring_by_vision`` / ``large_wrap_by_vision`` 故意直通）。
"""

import numpy as np

from .primitive_catalog import gated_primitive_names
from .topic_defaults import (
    bboxes_3d_topic,
    labeled_poses_topic,
    resolve_side_topic,
    side_hand_topics,
)

try:
    import rclpy
    from geometry_msgs.msg import PoseStamped
    from rclpy.node import Node
    from std_msgs.msg import String
except ImportError:  # pragma: no cover — 无 ROS 环境仍可 import 门控集合
    rclpy = None  # type: ignore[assignment]
    Node = object  # type: ignore[misc,assignment]
    PoseStamped = object  # type: ignore[misc,assignment]
    String = object  # type: ignore[misc,assignment]

# ── 包络原语集合 (需通过三判定: H / XY / Fwd) — 派生自 catalog ──
GATED_PRIMITIVES = set(gated_primitive_names())

# ── 忽略的物体标签 ────────────────────────────────────────────
IGNORE_LABELS = ["robotic hand", "hand", "arm", "robot", "claw"]


class GraspGate(Node):
    """三条件抓取门控: H(Z交叠) ∩ XY(掌面矩形) ∩ Fwd(手指前方)。"""

    def __init__(self):
        super().__init__("grasp_gate")

        # ── 参数 ──
        self.declare_parameter("hand_side", "left")
        # Empty => auto /{side}_hand/... (dual-hand safe).
        self.declare_parameter("tcp_pose_topic", "")
        self.declare_parameter("bbox_topic", bboxes_3d_topic())
        self.declare_parameter("labeled_pose_topic", labeled_poses_topic())
        self.declare_parameter("input_cmd_topic", "")
        self.declare_parameter("output_cmd_topic", "")
        self.declare_parameter("status_topic", "")
        self.declare_parameter("selected_pose_topic", "")
        self.declare_parameter("selected_label_topic", "")
        # 手几何
        self.declare_parameter("palm_z", 0.06)  # TCP → 掌心 Z 偏移 (m)
        self.declare_parameter("tip_z", 0.18)  # TCP → 指尖 Z 偏移 (m)
        self.declare_parameter("half_w", 0.20)  # 掌面左右半宽 (m)
        self.declare_parameter("half_h", 0.15)  # 掌面上下半高 (m)
        self.declare_parameter("height_tol", 0.05)  # 高度 Z 容差 (m)
        self.declare_parameter("data_timeout", 2.0)  # 数据过期 (s)
        self.declare_parameter("de_bounce", 3)  # 去抖帧数

        hand_side = self.get_parameter("hand_side").value
        if hand_side not in ("left", "right"):
            self.get_logger().error(f"hand_side 参数无效: '{hand_side}'，必须为 left 或 right")
            raise ValueError(f"Invalid hand_side: {hand_side}")
        side_topics = side_hand_topics(hand_side)
        self._side = hand_side
        self._tcp_topic = resolve_side_topic(
            self.get_parameter("tcp_pose_topic").value, side_topics["tcp_pose"]
        )
        self._bbox_topic = self.get_parameter("bbox_topic").value
        self._labeled_topic = self.get_parameter("labeled_pose_topic").value
        self._input_topic = resolve_side_topic(
            self.get_parameter("input_cmd_topic").value, side_topics["gesture_cmd"]
        )
        self._output_topic = resolve_side_topic(
            self.get_parameter("output_cmd_topic").value, side_topics["gesture_cmd_exec"]
        )
        self._status_topic = resolve_side_topic(
            self.get_parameter("status_topic").value, side_topics["gate_status"]
        )
        self._sel_pose_topic = resolve_side_topic(
            self.get_parameter("selected_pose_topic").value, side_topics["object_pose"]
        )
        self._sel_label_topic = resolve_side_topic(
            self.get_parameter("selected_label_topic").value, side_topics["selected_label"]
        )
        self._palm_z = self.get_parameter("palm_z").value
        self._tip_z = self.get_parameter("tip_z").value
        self._half_w = self.get_parameter("half_w").value
        self._half_h = self.get_parameter("half_h").value
        self._height_tol = self.get_parameter("height_tol").value
        self._data_timeout = self.get_parameter("data_timeout").value
        self._de_bounce = self.get_parameter("de_bounce").value

        # ── 状态 ──
        self._tcp = None  # dict: pos(3,), quat(4,), stamp
        self._bboxes = {}  # label -> {center, min, max, stamp}
        self._poses = {}  # label -> {pos, stamp} (fallback)
        self._last_label = None  # 上一帧选中的物体
        self._ok_count = 0  # 连续 OK 帧数

        # ── 订阅 ──
        self.create_subscription(PoseStamped, self._tcp_topic, self._tcp_cb, 10)

        # 首选: 3D bbox (有 min/max，H 判定精确)
        try:
            from robot_perception_msgs.msg import LabeledBBox3DArray

            self.create_subscription(LabeledBBox3DArray, self._bbox_topic, self._bbox_cb, 10)
            self.get_logger().info(f"已订阅 bbox_3d: {self._bbox_topic}")
        except ImportError:
            self.get_logger().warn("robot_perception_msgs 未安装，仅使用 LabeledPose 输入")

        # 备选: LabeledPose (无 bbox，H 判定降级为点±容差)
        try:
            from robot_perception_msgs.msg import LabeledPoseArray

            self.create_subscription(LabeledPoseArray, self._labeled_topic, self._pose_cb, 10)
            self.get_logger().info(f"已订阅 labeled_pose: {self._labeled_topic}")
        except ImportError:
            pass

        # 手势指令
        self.create_subscription(String, self._input_topic, self._cmd_cb, 10)

        # ── 发布 ──
        self._cmd_pub = self.create_publisher(String, self._output_topic, 10)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self._pose_pub = self.create_publisher(PoseStamped, self._sel_pose_topic, 10)
        self._label_pub = self.create_publisher(String, self._sel_label_topic, 10)
        from .workbench_gate import WorkbenchGate
        self._workbench = WorkbenchGate(self)

        self.get_logger().info(
            f"GraspGate side={self._side}: "
            f"H(Z交叠) XY(±{self._half_w * 100:.0f}x{self._half_h * 100:.0f}cm) "
            f"Fwd(cz>0) 去抖x{self._de_bounce}"
        )
        self.get_logger().info(
            f"包络原语={sorted(GATED_PRIMITIVES)}, "
            f"掌心Z={self._palm_z * 100:.0f}cm 指尖Z={self._tip_z * 100:.0f}cm"
        )
        self.get_logger().info(
            f"topics: in={self._input_topic} out={self._output_topic} "
            f"tcp={self._tcp_topic} object={self._sel_pose_topic} "
            f"label={self._sel_label_topic}"
        )

    # ──────────────────────────────────────────────────────────
    #  回调
    # ──────────────────────────────────────────────────────────

    def _tcp_cb(self, msg: PoseStamped) -> None:
        q = msg.pose.orientation
        self._tcp = {
            "pos": np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]),
            "quat": np.array([q.x, q.y, q.z, q.w]),
            "stamp": msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
        }

    def _bbox_cb(self, msg) -> None:
        """缓存 3D bbox (首选，有 min/max 可精确 H 判定)。"""
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for b in msg.boxes:
            if any(w in b.label.lower() for w in IGNORE_LABELS):
                continue
            self._bboxes[b.label] = {
                "center": np.array([b.center[0], b.center[1], b.center[2]]),
                "min": np.array([b.min[0], b.min[1], b.min[2]]),
                "max": np.array([b.max[0], b.max[1], b.max[2]]),
                "stamp": now,
            }

    def _pose_cb(self, msg) -> None:
        """缓存 LabeledPose (备选，用于无 bbox 时降级判定)。"""
        now = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        for lp in msg.poses:
            if any(w in lp.label.lower() for w in IGNORE_LABELS):
                continue
            self._poses[lp.label] = {
                "pos": np.array([lp.pose.position.x, lp.pose.position.y, lp.pose.position.z]),
                "stamp": now,
            }

    # ──────────────────────────────────────────────────────────
    #  指令门控
    # ──────────────────────────────────────────────────────────

    def _cmd_cb(self, msg: String) -> None:
        parts = msg.data.strip().lower().split()
        if not parts:
            return
        cmd = parts[0]
        self.get_logger().info(f"Gate 收到指令: '{msg.data.strip()}' (topic: {self._input_topic})")

        # 非包络原语直接放行（含 mode / pose / pose_list / auto / stop / 固定手势）
        if cmd not in GATED_PRIMITIVES:
            self.get_logger().info(f"Gate PASS (非门控): {cmd} → 转发")
            self._cmd_pub.publish(msg)
            return

        self.get_logger().info(f"Gate 收到门控指令: {cmd}，开始三判定...")
        from .workbench_gate import evaluate_gate
        selected, reason = evaluate_gate(self)
        if selected is None:
            self._reject(cmd, reason)
            return
        best_label, best_r = selected["label"], selected["result"]

        # 6. 去抖
        if best_label == self._last_label:
            self._ok_count += 1
        else:
            self._last_label = best_label
            self._ok_count = 1

        if self._ok_count >= self._de_bounce:
            self._accept(cmd, best_label, best_r, msg)
        else:
            self.get_logger().info(
                f"Gate pending [{cmd}]: {best_label} "
                f"H:{best_r['wz_min']:.2f}~{best_r['wz_max']:.2f} "
                f"XY:({best_r['cx'] * 100:.0f},{best_r['cy'] * 100:.0f})cm "
                f"Fwd:{best_r['cz'] * 100:.0f}cm "
                f"({self._ok_count}/{self._de_bounce})"
            )

    # ──────────────────────────────────────────────────────────
    #  三判定核心
    # ──────────────────────────────────────────────────────────

    def _to_hand(self, world_pt, R_tcp, tcp_pos):
        """世界坐标 → 手坐标系 (R_z(90°) 旋转后)。
        hand +X = 掌面左右, +Y = 掌面上下, +Z = 手指前方。
        """
        p = R_tcp.T @ (world_pt - tcp_pos)
        return np.array([-p[1], p[0], p[2]])

    def _eval_obj(self, obj, R_tcp, tcp_pos, wz_min_h, wz_max_h):
        """三判定 — bbox 模式 (有 min/max)。"""
        # H: 物体世界 Z 范围 ∩ 手世界 Z 范围
        wz_min_obj = min(obj["min"][2], obj["max"][2])
        wz_max_obj = max(obj["min"][2], obj["max"][2])
        ok_h = not (wz_min_obj > wz_max_h or wz_max_obj < wz_min_h)

        # XY + Fwd: 物体中心在手坐标系
        h = self._to_hand(obj["center"], R_tcp, tcp_pos)
        cx, cy, cz = float(h[0]), float(h[1]), float(h[2])
        ok_xy = abs(cx) < self._half_w and abs(cy) < self._half_h
        ok_fwd = cz > 0

        return {
            "ok_h": ok_h,
            "ok_xy": ok_xy,
            "ok_fwd": ok_fwd,
            "ok": ok_h and ok_xy and ok_fwd,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "wz_min": wz_min_obj,
            "wz_max": wz_max_obj,
        }

    def _eval_pose(self, obj, R_tcp, tcp_pos, wz_min_h, wz_max_h):
        """三判定 — pose 模式 (无 bbox，H 降级为中心点±容差)。"""
        center = obj["pos"]

        # H: 点 ± 容差
        wz_min_obj = center[2] - self._height_tol
        wz_max_obj = center[2] + self._height_tol
        ok_h = not (wz_min_obj > wz_max_h or wz_max_obj < wz_min_h)

        h = self._to_hand(center, R_tcp, tcp_pos)
        cx, cy, cz = float(h[0]), float(h[1]), float(h[2])
        ok_xy = abs(cx) < self._half_w and abs(cy) < self._half_h
        ok_fwd = cz > 0

        return {
            "ok_h": ok_h,
            "ok_xy": ok_xy,
            "ok_fwd": ok_fwd,
            "ok": ok_h and ok_xy and ok_fwd,
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "wz_min": wz_min_obj,
            "wz_max": wz_max_obj,
        }

    # ──────────────────────────────────────────────────────────
    #  辅助
    # ──────────────────────────────────────────────────────────

    def _cleanup(self, now):
        for d in [self._bboxes, self._poses]:
            stale = [label for label, o in d.items() if now - o["stamp"] > self._data_timeout]
            for label in stale:
                del d[label]

    def _reject(self, cmd: str, reason: str) -> None:
        self._last_label = None
        self._ok_count = 0
        self.get_logger().warn(f"Gate REJECT [{cmd}]: {reason}")
        self._status_pub.publish(String(data=f"REJECT: {cmd}, {reason}"))

    def _reject_reason(self, R_tcp, tcp_pos, wz_min_h, wz_max_h) -> str:
        """生成详细拒绝原因——列出最近物体的判定失败项。"""
        # 合并所有物体
        all_objs = []
        for label, obj in self._bboxes.items():
            r = self._eval_obj(obj, R_tcp, tcp_pos, wz_min_h, wz_max_h)
            all_objs.append((label, r["cz"], r))
        for label, obj in self._poses.items():
            r = self._eval_pose(obj, R_tcp, tcp_pos, wz_min_h, wz_max_h)
            all_objs.append((label, r["cz"], r))
        if not all_objs:
            return "无物体"

        # 按 cz 排序 (最近在前)
        all_objs.sort(key=lambda x: x[1])
        nearest = all_objs[0]
        label, cz, r = nearest
        fails = []
        if not r["ok_h"]:
            fails.append(
                f"H(wz_obj=[{r['wz_min']:.2f},{r['wz_max']:.2f}] "
                f"vs hand=[{wz_min_h:.2f},{wz_max_h:.2f}])"
            )
        if not r["ok_xy"]:
            fails.append(f"XY({r['cx'] * 100:.0f},{r['cy'] * 100:.0f}cm)")
        if not r["ok_fwd"]:
            fails.append(f"Fwd({r['cz'] * 100:.0f}cm≤0)")
        return f"'{label}' {'+'.join(fails)}"

    def _accept(self, cmd: str, label: str, r: dict, orig_msg: String) -> None:
        """通过 → 发布选中物体位姿 + 转发指令。"""
        # 发布物体位姿
        obj = self._bboxes.get(label) or self._poses.get(label)
        if obj is not None:
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "world"
            pos = obj.get("center", obj.get("pos"))
            pose_msg.pose.position.x = float(pos[0])
            pose_msg.pose.position.y = float(pos[1])
            pose_msg.pose.position.z = float(pos[2])
            pose_msg.pose.orientation.w = 1.0
            self._pose_pub.publish(pose_msg)

        # 发布选中 label（供 gesture_node / executor 过滤）
        self._label_pub.publish(String(data=label))

        # 转发指令
        self._cmd_pub.publish(orig_msg)

        self.get_logger().info(
            f"Gate OK [{cmd}]: '{label}' "
            f"H:{r['wz_min']:.2f}~{r['wz_max']:.2f} "
            f"XY:({r['cx'] * 100:.0f},{r['cy'] * 100:.0f})cm "
            f"Fwd:{r['cz'] * 100:.0f}cm ✅"
        )
        self._status_pub.publish(String(data=f"OK: {cmd}, '{label}' fwd={r['cz']:.3f}m"))


def main(args=None):
    if rclpy is None:
        raise RuntimeError("rclpy is required to run GraspGate")
    rclpy.init(args=args)
    node = GraspGate()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
