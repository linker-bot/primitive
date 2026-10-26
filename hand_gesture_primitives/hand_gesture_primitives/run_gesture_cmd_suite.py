#!/usr/bin/env python3
"""按序下发全套 gesture_cmd 原语，用于 MOCK/真机逐条目测回归。

绕过 ``ros2 topic pub --once`` 在 Docker host 网络下的 DDS 发现卡死，与
``pub_gesture_cmd`` 相同走进程内 rclpy 发布。

按 ``--hand-type``（或自动探测运行中的 gesture 节点）过滤本手型
allowlist 外原语：打印不支持原因并跳过，避免误发。

用法::

  ros2 run hand_gesture_primitives run_gesture_cmd_suite -- left --hand-type l25
  ros2 run hand_gesture_primitives run_gesture_cmd_suite -- left --pause
  ros2 run hand_gesture_primitives run_gesture_cmd_suite -- left --wait 3
  ros2 run hand_gesture_primitives run_gesture_cmd_suite -- left --dry-run --hand-type o6
  ros2 run hand_gesture_primitives run_gesture_cmd_suite -- left --from pinch
  ros2 run hand_gesture_primitives run_gesture_cmd_suite -- left --tags focus
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

from .topic_defaults import side_hand_topics

# 控制类指令（非 PRIMITIVE_REGISTRY 原语）：始终保留，不做手型过滤。
_CONTROL_BASES = frozenset(
    {
        "mode",
        "stop",
        "pose",
        "pose_list",
        "target",
        "target_id",
        "auto",
    }
)

# 全集顺序 = 各手型文档清单并集；具体手型由 HandProfile allowlist 裁剪。
# 支持范围见仓库 README 的手型与原语清单。
DEFAULT_SUITE: tuple[tuple[str, float, str], ...] = (
    ("mode primitive", 1.0, "setup"),
    ("init", 2.5, "safe"),
    ("open", 2.5, "safe"),
    ("release", 2.5, "safe"),
    ("relax_grip", 2.5, "safe"),
    ("fist", 3.0, "static"),
    ("pinch", 3.0, "focus"),
    ("index_pinch", 3.0, "focus"),
    ("middle_pinch", 3.0, "focus"),
    ("ok_sign", 3.0, "focus"),
    ("grasp_ready", 3.0, "static"),
    ("point", 3.0, "static"),
    ("v_sign", 3.0, "static"),
    ("parallel_extension", 3.0, "static"),
    # O6 / L6 独有分阶段
    ("large_wrap", 4.0, "phased"),
    ("large_wrap prep", 3.0, "phased"),
    ("large_wrap close", 4.0, "phased"),
    ("large_wrap auto", 5.0, "phased"),
    ("hammer_grip prep", 3.0, "phased"),
    ("hammer_grip close", 4.0, "phased"),
    ("thumb_adduction_grip", 4.0, "phased"),
    ("thumb_adduction_grip prep", 3.0, "phased"),
    ("thumb_adduction_grip close", 4.0, "phased"),
    ("index_middle_adduction_grip", 5.0, "phased"),
    ("index_middle_adduction_grip prep", 3.0, "phased"),
    ("index_middle_adduction_grip close", 4.0, "phased"),
    ("tripod", 3.5, "phased"),
    ("tripod prep", 3.0, "phased"),
    ("tripod close", 3.0, "phased"),
    ("tripod_by_force", 6.0, "phased"),
    ("tripod_by_force prep", 3.0, "phased"),
    ("tripod_by_force close", 5.0, "phased"),
    ("ring", 8.0, "force"),
    ("middle_ring", 8.0, "force"),
    ("index_ring_by_vision", 4.0, "vision"),
    ("large_wrap_by_vision", 4.0, "vision"),
    ("middle_ring_by_vision", 4.0, "vision"),
    ("ring_by_vision", 4.0, "vision"),
    ("small_warp_by_vision", 4.0, "vision"),
    ("no_index_warp_by_vision", 4.0, "vision"),
    ("hook_by_vision", 4.0, "vision"),
    ("index_pinch_by_vision", 3.5, "vision"),
    ("middle_pinch_by_vision", 3.5, "vision"),
    ("tripod_by_vision", 4.0, "vision"),
    ("palmar_by_vision", 4.0, "vision"),
    ("parallel_extension_by_vision", 3.5, "vision"),
    ("disk_by_vision", 4.0, "vision"),
    ("index_flick", 3.0, "motion"),
    ("index_middle_flick", 3.0, "motion"),
    ("twist2digit", 3.0, "motion"),
    ("twist3digit", 3.0, "motion"),
    ("twist5digit", 3.0, "motion"),
    ("open", 2.5, "safe"),
)


@dataclass(frozen=True)
class SuiteStep:
    command: str
    dwell_s: float
    tag: str


def _parse_suite(raw: Sequence[tuple[str, float, str]]) -> List[SuiteStep]:
    return [SuiteStep(c, d, t) for c, d, t in raw]


def command_base(command: str) -> str:
    """取指令首词作为原语/控制基名（``thumb_adduction_grip prep`` → ``thumb_adduction_grip``）。"""
    parts = (command or "").strip().split()
    return parts[0].lower() if parts else ""


def _filter_steps(
    steps: Sequence[SuiteStep],
    *,
    tags: Optional[set[str]],
    from_cmd: str,
    only_cmd: str,
) -> List[SuiteStep]:
    out = list(steps)
    if tags:
        out = [s for s in out if s.tag in tags]
    if from_cmd:
        needle = from_cmd.strip().lower()
        for i, s in enumerate(out):
            if s.command.lower() == needle or s.command.lower().startswith(needle):
                out = out[i:]
                break
        else:
            raise SystemExit(f"--from 未找到指令: {from_cmd!r}")
    if only_cmd:
        needle = only_cmd.strip().lower()
        matched = [s for s in out if s.command.lower() == needle]
        if not matched:
            raise SystemExit(f"--only 未找到指令: {only_cmd!r}")
        out = matched
    return out


def partition_by_hand_support(
    steps: Sequence[SuiteStep],
    hand_type: str,
    *,
    include_unsupported: bool = False,
) -> Tuple[List[SuiteStep], List[Tuple[SuiteStep, str]]]:
    """按手型 allowlist 拆成可跑步骤 / 跳过步骤（附中文原因）。

    ``mode`` 等控制指令始终保留。``include_unsupported=True`` 时全部可跑、跳过表为空。
    """
    from .hand_profile import load_hand_profile

    profile = load_hand_profile(hand_type)
    runnable: List[SuiteStep] = []
    skipped: List[Tuple[SuiteStep, str]] = []

    for step in steps:
        base = command_base(step.command)
        if not base or base in _CONTROL_BASES:
            runnable.append(step)
            continue
        if profile.is_primitive_supported(base):
            runnable.append(step)
            continue
        reason = profile.reject_user_message(base) or (
            f"{profile.hand_joint.upper()} 不支持原语 '{base}'"
        )
        if include_unsupported:
            runnable.append(step)
        else:
            skipped.append((step, reason))
    return runnable, skipped


def _try_discover_hand_type(side: str, timeout_s: float = 2.0) -> Optional[str]:
    """从运行中的 ``{side}_hand_gesture_node`` 读取 ``hand_type`` 参数。"""
    import rclpy
    from rcl_interfaces.msg import ParameterType
    from rcl_interfaces.srv import GetParameters
    from rclpy.node import Node

    node_name = f"{side}_hand_gesture_node"
    service = f"/{node_name}/get_parameters"
    owned_init = False
    if not rclpy.ok():
        rclpy.init()
        owned_init = True
    node = Node("run_gesture_cmd_suite_discover")
    try:
        client = node.create_client(GetParameters, service)
        deadline = time.monotonic() + max(0.3, timeout_s)
        while time.monotonic() < deadline:
            if not client.wait_for_service(timeout_sec=0.15):
                rclpy.spin_once(node, timeout_sec=0.05)
                continue
            req = GetParameters.Request()
            req.names = ["hand_type"]
            future = client.call_async(req)
            while not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.05)
            if not future.done() or future.result() is None:
                return None
            values = future.result().values
            if not values:
                return None
            v = values[0]
            if v.type == ParameterType.PARAMETER_STRING and v.string_value:
                return str(v.string_value).strip()
            return None
        return None
    finally:
        node.destroy_node()
        if owned_init and rclpy.ok():
            rclpy.shutdown()


def _parse_argv(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="run_gesture_cmd_suite",
        description=(
            "Sequentially publish gesture_cmd for hand-type-aware primitive regression. "
            "Unsupported primitives are skipped with a reason (unless --include-unsupported)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  run_gesture_cmd_suite -- left --hand-type l25\n"
            "  run_gesture_cmd_suite -- left --pause\n"
            "  run_gesture_cmd_suite -- left --hand-type o6 --dry-run\n"
            "  run_gesture_cmd_suite -- left --wait 3 --wait-motion\n"
            "  run_gesture_cmd_suite -- left --tags focus\n"
            "  run_gesture_cmd_suite -- left --from pinch --dry-run\n"
        ),
    )
    p.add_argument("side", choices=("left", "right"), help="hand side")
    p.add_argument(
        "--hand-type",
        "-t",
        default="",
        help="hand model: l25|o20|g20|o6|l6（省略则尝试从 gesture 节点探测）",
    )
    p.add_argument(
        "--no-auto-hand",
        action="store_true",
        help="不探测运行中节点的 hand_type；未给 --hand-type 时跑全集不裁剪",
    )
    p.add_argument(
        "--include-unsupported",
        action="store_true",
        help="仍发布本手型不支持的原语（用于验证节点拒绝路径）",
    )
    p.add_argument(
        "--wait",
        type=float,
        default=0.0,
        help="override per-step dwell (seconds); 0=use built-in defaults",
    )
    p.add_argument(
        "--wait-motion",
        action="store_true",
        help="after each step, also wait for /{side}_hand/motion_done true (timeout)",
    )
    p.add_argument(
        "--motion-timeout",
        type=float,
        default=12.0,
        help="max seconds to wait for motion_done per step (default: 12)",
    )
    p.add_argument(
        "--pub-timeout",
        type=float,
        default=5.0,
        help="seconds to wait for gesture_cmd subscriber before each publish",
    )
    p.add_argument(
        "--from",
        dest="from_cmd",
        default="",
        help="start from this command (prefix match ok)",
    )
    p.add_argument("--only", dest="only_cmd", default="", help="run a single command")
    p.add_argument(
        "--tags",
        default="",
        help="comma-separated filter: setup,safe,static,focus,phased,force,vision,motion",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print planned steps without publishing",
    )
    p.add_argument(
        "--pause",
        action="store_true",
        help="press Enter before each step (manual inspect)",
    )
    p.add_argument("--topic", default="", help="override gesture_cmd topic")
    return p.parse_args(argv)


def _wait_motion_done(
    node,
    motion_topic: str,
    timeout_s: float,
) -> bool:
    from rclpy.qos import qos_profile_sensor_data
    from std_msgs.msg import Bool

    done = {"ok": False}

    def _cb(msg: Bool) -> None:
        if msg.data:
            done["ok"] = True

    sub = node.create_subscription(Bool, motion_topic, _cb, qos_profile_sensor_data)
    deadline = time.monotonic() + max(0.2, timeout_s)
    try:
        while not done["ok"] and time.monotonic() < deadline:
            import rclpy

            rclpy.spin_once(node, timeout_sec=0.05)
    finally:
        node.destroy_subscription(sub)
    return done["ok"]


def _publish_one(
    node,
    pub,
    data: str,
    pub_timeout: float,
) -> int:
    import rclpy

    deadline = time.monotonic() + max(0.1, pub_timeout)
    while pub.get_subscription_count() < 1 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    subs = pub.get_subscription_count()
    if subs < 1:
        return 1
    from std_msgs.msg import String

    msg = String()
    msg.data = data
    pub.publish(msg)
    end = time.monotonic() + 0.15
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)
    return 0


def run_suite(
    side: str,
    steps: Iterable[SuiteStep],
    *,
    wait_override: float,
    wait_motion: bool,
    motion_timeout: float,
    pub_timeout: float,
    topic: str,
    dry_run: bool,
    pause: bool,
    hand_type: str = "",
    skipped: Optional[Sequence[Tuple[SuiteStep, str]]] = None,
) -> int:
    topics = side_hand_topics(side)
    cmd_topic = topic.strip() or topics["gesture_cmd"]
    motion_topic = f"/{side}_hand/motion_done"

    steps_list = list(steps)
    skipped_list = list(skipped or [])

    hand_label = hand_type.strip() or "unfiltered"
    print(f"=== gesture_cmd 回归 ({side}, hand={hand_label}) 共 {len(steps_list)} 步 ===")
    print(f"cmd topic: {cmd_topic}")
    if wait_motion:
        print(f"motion_done: {motion_topic} (timeout={motion_timeout:g}s)")

    if skipped_list:
        print(f"--- 跳过不支持 {len(skipped_list)} 步 ---")
        seen_bases: set[str] = set()
        for step, reason in skipped_list:
            base = command_base(step.command)
            print(f"  SKIP [{step.tag}] {step.command!r}")
            if base not in seen_bases:
                print(f"        {reason}")
                seen_bases.add(base)

    if not steps_list:
        print("无步骤可执行（全部被手型过滤跳过？）", file=sys.stderr)
        return 2

    if dry_run:
        for i, step in enumerate(steps_list, 1):
            dwell = wait_override if wait_override > 0 else step.dwell_s
            print(
                f"[{i:02d}/{len(steps_list):02d}] [{step.tag}] "
                f"wait={dwell:g}s  →  {step.command!r}"
            )
        return 0

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import String

    rclpy.init()
    node = Node("run_gesture_cmd_suite")
    pub = node.create_publisher(String, cmd_topic, 10)

    failures = 0
    try:
        for i, step in enumerate(steps_list, 1):
            dwell = wait_override if wait_override > 0 else step.dwell_s
            if pause:
                try:
                    input(f"[{i}/{len(steps_list)}] Enter 继续 → {step.command!r} ")
                except EOFError:
                    print("中断", file=sys.stderr)
                    return 130

            print(f"[{i:02d}/{len(steps_list):02d}] [{step.tag}] publish → {step.command!r}")
            rc = _publish_one(node, pub, step.command, pub_timeout)
            if rc != 0:
                print(
                    f"  WARN: 未见订阅者 ({cmd_topic})，仍等待 {dwell:g}s",
                    file=sys.stderr,
                )
                failures += 1

            if wait_motion:
                if _wait_motion_done(node, motion_topic, motion_timeout):
                    print("  motion_done=true")
                else:
                    print(
                        f"  WARN: {motion_timeout:g}s 内未见 motion_done=true，"
                        f"fallback sleep {dwell:g}s",
                        file=sys.stderr,
                    )
                    time.sleep(dwell)
            else:
                time.sleep(dwell)
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print(
        f"=== 完成 (run={len(steps_list)}, skip={len(skipped_list)}, publish_warn={failures}) ==="
    )
    return 1 if failures else 0


def resolve_hand_type_for_suite(
    side: str,
    hand_type_arg: str,
    *,
    auto_hand: bool,
    dry_run: bool,
) -> Tuple[str, str]:
    """返回 (hand_type, source_note)。hand_type 空表示不裁剪。"""
    explicit = (hand_type_arg or "").strip()
    if explicit:
        return explicit, "cli --hand-type"

    if not auto_hand:
        return "", "disabled (--no-auto-hand)"

    # dry-run 未起 ROS 时仍可探测；探测失败则不裁剪
    try:
        discovered = _try_discover_hand_type(side)
    except Exception as exc:  # noqa: BLE001 — 探测失败仅降级
        return "", f"auto-discover failed: {exc}"

    if discovered:
        return discovered, f"auto from /{side}_hand_gesture_node"

    note = "auto-discover miss"
    if dry_run:
        note += "（dry-run 且节点未就绪时可显式 --hand-type）"
    return "", note


def main(argv: List[str] | None = None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    if "--" in raw:
        raw = raw[raw.index("--") + 1 :]
    elif "--ros-args" in raw:
        raw = raw[: raw.index("--ros-args")]

    args = _parse_argv(raw)
    tags = {t.strip() for t in args.tags.split(",") if t.strip()} or None
    steps = _filter_steps(
        _parse_suite(DEFAULT_SUITE),
        tags=tags,
        from_cmd=args.from_cmd,
        only_cmd=args.only_cmd,
    )

    hand_type, hand_src = resolve_hand_type_for_suite(
        args.side,
        args.hand_type,
        auto_hand=not bool(args.no_auto_hand),
        dry_run=bool(args.dry_run),
    )

    skipped: List[Tuple[SuiteStep, str]] = []
    if hand_type:
        print(f"hand_type={hand_type} ({hand_src})")
        try:
            steps, skipped = partition_by_hand_support(
                steps,
                hand_type,
                include_unsupported=bool(args.include_unsupported),
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"手型无效: {exc}", file=sys.stderr)
            return 2
        if args.only_cmd and not steps and skipped:
            print(
                f"--only {args.only_cmd!r} 对本手型不支持，已跳过:\n  {skipped[0][1]}",
                file=sys.stderr,
            )
            return 2
    else:
        print(
            f"WARN: 未确定 hand_type（{hand_src}），将跑全集不裁剪；"
            f"建议: --hand-type l25|o20|g20|o6|l6",
            file=sys.stderr,
        )

    return run_suite(
        args.side,
        steps,
        wait_override=float(args.wait),
        wait_motion=bool(args.wait_motion),
        motion_timeout=float(args.motion_timeout),
        pub_timeout=float(args.pub_timeout),
        topic=args.topic,
        dry_run=bool(args.dry_run),
        pause=bool(args.pause),
        hand_type=hand_type,
        skipped=skipped,
    )


if __name__ == "__main__":
    sys.exit(main())
