#!/usr/bin/env python3
"""可靠下发 /{side}_hand/gesture_cmd，绕过 ros2cli daemon 发现卡死。

Docker --network host 下 `ros2 topic pub --once` 常永久 Waiting for subscription，
而同进程内 rclpy 发布可正常匹配 gesture_node / grasp_gate。

用法::

  ros2 run hand_gesture_primitives pub_gesture_cmd -- right open
  ros2 run hand_gesture_primitives pub_gesture_cmd -- left "mode primitive"
  ros2 run hand_gesture_primitives pub_gesture_cmd -- right "pose my_open"
  ros2 run hand_gesture_primitives pub_gesture_cmd -- left pose_list
  ros2 run hand_gesture_primitives pub_gesture_cmd -- left pose_list --local --hand-type g20
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence

from .topic_defaults import side_hand_topics

# 查询类指令：发布后等待 gesture_result 并打印到 stdout
_QUERY_COMMANDS = frozenset({"pose_list"})


def parse_poses_result(data: str) -> Optional[List[str]]:
    """解析 ``POSES:a,b``；非该前缀返回 None；空库返回 []。"""
    text = (data or "").strip()
    if not text.startswith("POSES:"):
        return None
    body = text[len("POSES:") :].strip()
    if not body:
        return []
    return [n.strip() for n in body.split(",") if n.strip()]


def format_pose_list(
    names: Sequence[str],
    *,
    side: str,
    source: str,
    hand_type: str = "",
) -> str:
    """人类可读的姿态列表（stdout）。"""
    scope = f"{hand_type}/{side}" if hand_type else side
    header = f"Pose Library ({scope})  [{source}]"
    if not names:
        return f"{header}\n  (空)"
    lines = [header, f"  共 {len(names)} 个:"]
    lines.extend(f"  - {n}" for n in names)
    return "\n".join(lines)


def _parse_argv(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="pub_gesture_cmd",
        description=(
            "Publish std_msgs/String to /{side}_hand/gesture_cmd (no ros2cli daemon). "
            "For pose_list, also wait on gesture_result and print names to stdout."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  pub_gesture_cmd -- left open\n"
            "  pub_gesture_cmd -- left pose_list\n"
            "  pub_gesture_cmd -- left pose_list --local --hand-type g20\n"
        ),
    )
    p.add_argument(
        "side",
        choices=("left", "right"),
        help="hand side",
    )
    p.add_argument(
        "command",
        nargs="+",
        help="command tokens, e.g. open | mode teach | pose my_open | pose_list",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=3.0,
        help="seconds to wait for ≥1 matching subscription (default: 3)",
    )
    p.add_argument(
        "--result-timeout",
        type=float,
        default=2.0,
        help="for pose_list: seconds to wait for POSES: on gesture_result (default: 2)",
    )
    p.add_argument(
        "--topic",
        default="",
        help="override cmd topic (default: /{side}_hand/gesture_cmd)",
    )
    p.add_argument(
        "--result-topic",
        default="",
        help="override result topic (default: /{side}_hand/gesture_result)",
    )
    p.add_argument(
        "--local",
        action="store_true",
        help="pose_list only: read Pose Library from disk (no gesture_node needed)",
    )
    p.add_argument(
        "--hand-type",
        default="",
        help="with --local: hand type bucket (g20|l25|o20|o6|l6|…)",
    )
    return p.parse_args(argv)


def _list_poses_local(side: str, hand_type: str) -> int:
    from .pose_library import list_poses, pose_dir

    ht = (hand_type or "").strip()
    if not ht:
        print(
            "本地查询需要 --hand-type（如 g20 / l25 / o20）。\n"
            "示例: ros2 run hand_gesture_primitives pub_gesture_cmd -- "
            f"{side} pose_list --local --hand-type g20",
            file=sys.stderr,
        )
        return 2
    try:
        names = list_poses(ht, hand_side=side)
    except Exception as e:  # noqa: BLE001 — CLI 出口
        print(f"读取姿态库失败: {e}", file=sys.stderr)
        return 1
    print(format_pose_list(names, side=side, source=f"local:{pose_dir()}", hand_type=ht))
    return 0


def _is_pose_list_cmd(tokens: Sequence[str]) -> bool:
    joined = " ".join(tokens).strip().lower()
    return joined == "pose_list" or (len(tokens) == 1 and tokens[0].lower() in _QUERY_COMMANDS)


def main(argv: list[str] | None = None) -> int:
    # ros2 run ... -- <args>；若无 --，整段 argv 都可能含 ros 参数，只取非 - 开头段较脆。
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        raw = raw[raw.index("--") + 1 :]
    else:
        # 过滤 ros2 注入的 -r / --ros-args 之后内容
        if "--ros-args" in raw:
            raw = raw[: raw.index("--ros-args")]

    args = _parse_argv(raw)
    data = " ".join(args.command).strip()
    if not data:
        print("empty command", file=sys.stderr)
        return 2

    is_pose_list = _is_pose_list_cmd(args.command)

    if args.local and not is_pose_list:
        print("--local 仅对 pose_list 有效，已忽略", file=sys.stderr)

    if is_pose_list and args.local:
        return _list_poses_local(args.side, args.hand_type)

    topics = side_hand_topics(args.side)
    topic = args.topic.strip() or topics["gesture_cmd"]
    result_topic = args.result_topic.strip() or topics["gesture_result"]

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("pub_gesture_cmd")
    pub = node.create_publisher(String, topic, 10)

    received: dict[str, Optional[str]] = {"data": None}
    armed = {"ok": False}

    def _on_result(msg: String) -> None:
        if not armed["ok"]:
            return
        if parse_poses_result(msg.data) is not None:
            received["data"] = msg.data

    result_sub = None
    if is_pose_list:
        result_sub = node.create_subscription(String, result_topic, _on_result, 10)

    deadline = time.monotonic() + max(0.1, float(args.timeout))
    while pub.get_subscription_count() < 1 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)

    subs = pub.get_subscription_count()
    if subs < 1:
        print(
            f"超时未见订阅者 topic={topic}（gesture_node/gate 未起？或 DDS 发现失败）。\n"
            f"可检查: ros2 topic info {topic} -v\n"
            f"或离线查库: ros2 run hand_gesture_primitives pub_gesture_cmd -- "
            f"{args.side} pose_list --local --hand-type <手型>",
            file=sys.stderr,
        )
        if result_sub is not None:
            node.destroy_subscription(result_sub)
        node.destroy_node()
        rclpy.shutdown()
        return 1

    # 给 result 订阅一点匹配时间，再发指令，避免丢首包
    if is_pose_list:
        match_deadline = time.monotonic() + min(0.5, max(0.1, float(args.result_timeout)))
        while time.monotonic() < match_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

    armed["ok"] = True
    msg = String()
    msg.data = data
    pub.publish(msg)

    if is_pose_list:
        result_deadline = time.monotonic() + max(0.2, float(args.result_timeout))
        while received["data"] is None and time.monotonic() < result_deadline:
            rclpy.spin_once(node, timeout_sec=0.05)

        if received["data"] is None:
            print(
                f"已发布 → {topic}: '{data}'，但 {args.result_timeout:g}s 内未收到 "
                f"{result_topic} 的 POSES: 回复。\n"
                "请确认 gesture_node 在跑；或改用 --local --hand-type <手型> 直接读盘。",
                file=sys.stderr,
            )
            if result_sub is not None:
                node.destroy_subscription(result_sub)
            node.destroy_node()
            rclpy.shutdown()
            return 3

        names = parse_poses_result(received["data"]) or []
        print(
            format_pose_list(
                names,
                side=args.side,
                source=result_topic,
            )
        )
        if result_sub is not None:
            node.destroy_subscription(result_sub)
        node.destroy_node()
        rclpy.shutdown()
        return 0

    # 普通指令：给匹配端一点时间收包
    end = time.monotonic() + 0.2
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)

    print(f"已发布 → {topic}: '{data}' (subs={subs})")
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
