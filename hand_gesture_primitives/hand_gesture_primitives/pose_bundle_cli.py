"""Pose Library bundle 导入/导出 CLI。

用法::

  ros2 run hand_gesture_primitives pose_bundle_export -- \\
    --hand-type g20 --hand-side left --output /tmp/left.pose_bundle.json

  ros2 run hand_gesture_primitives pose_bundle_import -- \\
    --input /tmp/left.pose_bundle.json --strategy skip

  # 旧目录迁移到当前 HAND_GESTURE_POSE_DIR
  ros2 run hand_gesture_primitives pose_bundle_import -- \\
    --from-dir ~/.config/hand_gesture_primitives/poses --strategy skip
"""

from __future__ import annotations

import argparse
import socket
import sys
from typing import Optional, Sequence


def _machine_hint() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return ""


def export_main(argv: Optional[Sequence[str]] = None) -> int:
    from .pose_bundle import export_library_to_file

    parser = argparse.ArgumentParser(description="Export Pose Library to pose_bundle JSON")
    parser.add_argument("--output", "-o", required=True)
    parser.add_argument("--hand-type", default="")
    parser.add_argument("--hand-side", default="", help="left|right；空=该手型两侧+legacy")
    parser.add_argument(
        "--names",
        default="",
        help="comma-separated pose names；空=全部",
    )
    parser.add_argument(
        "--from-dir",
        default="",
        help="optional source pose dir (default: HAND_GESTURE_POSE_DIR)",
    )
    ns, _ = parser.parse_known_args(args=list(argv) if argv is not None else None)
    names = [n.strip() for n in ns.names.split(",") if n.strip()] if ns.names else None
    hand_type = ns.hand_type.strip() or None
    hand_side = ns.hand_side.strip() or None
    root = ns.from_dir.strip() or None
    try:
        from pathlib import Path

        path, count = export_library_to_file(
            ns.output,
            hand_type=hand_type,
            hand_side=hand_side,
            names=names,
            exporter="pose_bundle_export",
            machine_hint=_machine_hint(),
            root=Path(root) if root else None,
        )
    except Exception as exc:  # noqa: BLE001 — CLI 边界
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    print(f"exported={count} bundle={path}")
    return 0


def import_main(argv: Optional[Sequence[str]] = None) -> int:
    from .pose_bundle import import_bundle_file, import_library_dir

    parser = argparse.ArgumentParser(description="Import pose_bundle JSON into Pose Library")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", "-i", default="", help="*.pose_bundle.json")
    src.add_argument("--from-dir", default="", help="legacy/other Pose Library directory")
    parser.add_argument(
        "--strategy",
        default="skip",
        choices=("skip", "overwrite", "rename"),
    )
    parser.add_argument("--hand-type", default="", help="optional filter")
    parser.add_argument("--hand-side", default="", help="optional filter")
    parser.add_argument(
        "--default-side",
        default="",
        help="legacy entries without side → this side",
    )
    parser.add_argument(
        "--no-verify-checksum",
        action="store_true",
        help="skip bundle checksum check",
    )
    ns, _ = parser.parse_known_args(args=list(argv) if argv is not None else None)
    hand_type = ns.hand_type.strip() or None
    hand_side = ns.hand_side.strip() or None
    default_side = ns.default_side.strip() or None
    try:
        if ns.from_dir:
            result = import_library_dir(
                ns.from_dir,
                strategy=ns.strategy,
                hand_type=hand_type,
                hand_side=hand_side,
                default_side=default_side,
            )
        else:
            result = import_bundle_file(
                ns.input,
                strategy=ns.strategy,
                hand_type=hand_type,
                hand_side=hand_side,
                default_side=default_side,
                verify_checksum=not ns.no_verify_checksum,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"失败: {exc}", file=sys.stderr)
        return 1
    print(result.summary)
    return 0 if result.success else 2


def main_export(args=None) -> None:
    raise SystemExit(export_main(args))


def main_import(args=None) -> None:
    raise SystemExit(import_main(args))


if __name__ == "__main__":
    # 默认走 export；import 由 console_scripts 入口区分
    raise SystemExit(export_main())
