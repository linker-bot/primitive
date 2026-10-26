"""从 gui_control presets JSON 单向导入到 PoseLibrary。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

from .hand_config import HandConfig, resolve_hand_config_key
from .pose_library import list_poses, upsert_pose, validate_pose_name


def _default_presets_path(hand_type: str) -> Path:
    # gui_control 常见布局：~/.config/gui_control/presets/{Hand}.json
    key = resolve_hand_config_key(hand_type)
    home = Path.home() / ".config" / "gui_control" / "presets"
    for name in (key.upper(), key, f"{key}.json"):
        p = home / name if not str(name).endswith(".json") else home / name
        if p.is_file():
            return p
        if (home / f"{name}.json").is_file():
            return home / f"{name}.json"
    return home / f"{key.upper()}.json"


def _extract_poses(raw: dict) -> dict:
    """兼容几种 gui_control / preset 文件形状。"""
    if not isinstance(raw, dict):
        return {}
    if isinstance(raw.get("poses"), dict):
        return raw["poses"]
    if isinstance(raw.get("presets"), dict):
        return raw["presets"]
    # 顶层即为 name -> {positions|angles|joints}
    skip = {"version", "hand_joint", "hand_type", "canonical_hand_type"}
    out = {}
    for k, v in raw.items():
        if k in skip or not isinstance(v, dict):
            continue
        if any(x in v for x in ("positions", "angles", "joints", "joint_angles")):
            out[str(k)] = v
    return out


def _positions_from_entry(entry: dict, num_joints: int) -> Optional[list]:
    for key in ("positions", "angles", "joints", "joint_angles"):
        if key in entry and isinstance(entry[key], list):
            vals = [float(x) for x in entry[key]]
            if len(vals) == num_joints:
                return vals
    return None


def import_gui_presets(
    hand_type_or_joint: str,
    presets_path: str = "",
    *,
    overwrite: bool = False,
    hand_side: Optional[str] = None,
) -> Tuple[int, int, Optional[str]]:
    """返回 (imported, skipped, error_or_None)。"""
    canonical = resolve_hand_config_key(hand_type_or_joint)
    hand = HandConfig(canonical)
    path = Path(presets_path) if presets_path else _default_presets_path(canonical)
    if not path.is_file():
        return 0, 0, f"文件不存在: {path}"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return 0, 0, f"读取失败: {exc}"

    poses = _extract_poses(raw if isinstance(raw, dict) else {})
    if not poses:
        return 0, 0, "未找到可导入的姿态条目"

    existing = set(list_poses(canonical, hand_side=hand_side))
    imported = 0
    skipped = 0
    for name, entry in poses.items():
        n = str(name).strip().lower()
        err = validate_pose_name(n)
        if err:
            skipped += 1
            continue
        if n in existing and not overwrite:
            skipped += 1
            continue
        positions = _positions_from_entry(entry, hand.num_joints)
        if positions is None:
            skipped += 1
            continue
        duration = float(entry.get("duration", 0.6))
        desc = str(entry.get("description") or entry.get("desc") or "")
        up_err = upsert_pose(
            canonical,
            n,
            positions,
            duration=duration,
            description=desc,
            source="import_gui_control",
            space="hardware",
            hand_side=hand_side,
            replace_multi_keyframe=bool(overwrite),
        )
        if up_err:
            skipped += 1
            continue
        imported += 1
        existing.add(n)
    if imported == 0:
        return (
            0,
            skipped,
            f"all {skipped} entries skipped (reserved/exists/invalid; try --overwrite)",
        )
    return imported, skipped, None


def main(args=None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="单向导入 gui_control presets → PoseLibrary")
    parser.add_argument("--hand-type", default="l25")
    parser.add_argument("--hand-side", default="left")
    parser.add_argument("--presets-path", default="")
    parser.add_argument("--overwrite", action="store_true")
    ns, _ = parser.parse_known_args(args=args)
    side = (ns.hand_side or "left").strip().lower()
    if side not in ("left", "right"):
        print(f"失败: hand_side 无效: {ns.hand_side}", file=sys.stderr)
        sys.exit(2)
    imported, skipped, err = import_gui_presets(
        ns.hand_type,
        ns.presets_path,
        overwrite=ns.overwrite,
        hand_side=side,
    )
    if imported == 0:
        print(f"失败: {err or 'no poses imported'}", file=sys.stderr)
        sys.exit(1)
    print(f"imported={imported} skipped={skipped}" + (f" note={err}" if err else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
