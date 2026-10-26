"""将可规划静态原语终态导出为 Pose Library / bundle。

离线读取 gesture YAML → semantic 终态 → hardware 向量，不依赖关节状态。
默认名 ``preset_<primitive>``，避免与内置原语保留字冲突。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from .gesture_params import load_static_gesture_params
from .hand_config import HandConfig, resolve_hand_config_key
from .planable_primitives import (
    DEFAULT_PRESET_NAME_PREFIX,
    PLANABLE_STATIC_PRIMITIVES,
    is_planable_static,
)
from .pose_bundle import BundleEntry, build_bundle, write_bundle
from .pose_library import (
    DEFAULT_SAVE_SPACE,
    list_poses,
    normalize_hand_side,
    upsert_pose,
    validate_pose_name,
)
from .semantic_30 import SEMANTIC_DIM, as_semantic_30


def _gestures_configured(hand_type: str) -> set[str]:
    from .gesture_params import _load_raw_gestures

    return {str(k).lower() for k in (_load_raw_gestures(hand_type) or {})}


def resolve_primitive_list(
    primitives: Optional[Sequence[str]] = None,
    *,
    hand_type: str,
    set_name: str = "planable",
) -> List[str]:
    """解析待导出原语列表。

    set_name:
      - planable: 与 PlanGesture 对齐的静态集 ∩ 本手型 YAML 已配置
      - static_all: 本手型 gestures 中属于 PLANABLE 的全部
    """
    canonical = resolve_hand_config_key(hand_type)
    configured = _gestures_configured(canonical)
    if primitives:
        out = []
        for p in primitives:
            name = str(p).strip().lower()
            if not name:
                continue
            if not is_planable_static(name):
                raise ValueError(f"primitive not planable/exportable: {name}")
            out.append(name)
        return out

    set_l = (set_name or "planable").strip().lower()
    if set_l not in ("planable", "static_all"):
        raise ValueError("set_name must be planable|static_all")
    base = PLANABLE_STATIC_PRIMITIVES
    # 仅导出本手型 YAML 中实际存在的项，避免空默认姿态污染库
    return sorted(p for p in base if p in configured)


def primitive_final_semantic(
    hand_type: str,
    primitive: str,
) -> Tuple[List[float], float]:
    """返回 (semantic_v3 S30 positions legacy255, duration)。"""
    canonical = resolve_hand_config_key(hand_type)
    if not is_planable_static(primitive):
        raise ValueError(f"primitive not planable/exportable: {primitive}")
    params = load_static_gesture_params(canonical, primitive)
    semantic = as_semantic_30(list(params.target_angles))
    duration = float(params.duration)
    if params.motion_type == "phased_pose":
        duration = (
            float(params.phase1)
            + float(params.phase2)
            + max(float(params.phase3), 0.0)
            + float(params.phase4)
        )
    elif params.motion_type == "relax_grip":
        duration = (
            float(params.duration) + float(params.hold_duration) + float(params.grip_duration)
        )
    return [float(v) for v in semantic], max(duration, 0.2)


def primitive_final_hardware(
    hand_type: str,
    primitive: str,
) -> Tuple[List[float], float]:
    """返回 (hardware positions, duration)。"""
    canonical = resolve_hand_config_key(hand_type)
    hand = HandConfig(canonical)
    semantic, duration = primitive_final_semantic(canonical, primitive)
    semantic30 = as_semantic_30(semantic)
    if hand.has_from_semantic:
        hw = hand.from_semantic(semantic30)
    else:
        from .semantic_30 import project_s30_to_o20

        hw = hand.to_hardware(project_s30_to_o20(semantic30))
    return [float(v) for v in hw], duration


def build_primitive_entries(
    hand_type: str,
    hand_side: str,
    primitives: Sequence[str],
    *,
    name_prefix: str = DEFAULT_PRESET_NAME_PREFIX,
) -> List[BundleEntry]:
    canonical = resolve_hand_config_key(hand_type)
    side = normalize_hand_side(hand_side)
    if side is None:
        raise ValueError("hand_side must be left|right for primitive export")
    prefix = name_prefix or DEFAULT_PRESET_NAME_PREFIX
    entries: List[BundleEntry] = []
    for prim in primitives:
        positions, duration = primitive_final_semantic(canonical, prim)
        name = f"{prefix}{prim}".lower()
        err = validate_pose_name(name)
        if err:
            raise ValueError(f"generated pose name invalid for {prim}: {err}")
        entries.append(
            BundleEntry(
                canonical_hand_type=canonical,
                hand_side=side,
                name=name,
                positions=positions,
                num_joints=SEMANTIC_DIM,
                space=DEFAULT_SAVE_SPACE,
                unit="legacy255",
                duration=duration,
                description=f"exported final pose of primitive '{prim}'",
                source="primitive_export",
                origin_primitive=prim,
            )
        )
    return entries


def export_primitive_poses(
    hand_type: str,
    hand_side: str,
    *,
    primitives: Optional[Sequence[str]] = None,
    set_name: str = "planable",
    name_prefix: str = DEFAULT_PRESET_NAME_PREFIX,
    write_library: bool = False,
    output_path: Optional[str] = None,
    overwrite: bool = False,
) -> Tuple[int, Optional[Path], List[str]]:
    """导出预定义终态。

    返回 (count, bundle_path_or_None, written_names)。
    write_library=True 时 upsert；output_path 非空时写 bundle（可同时）。
    """
    if not write_library and not output_path:
        raise ValueError("specify output_path and/or write_library=True")

    prims = resolve_primitive_list(primitives, hand_type=hand_type, set_name=set_name)
    if not prims:
        return 0, None, []

    entries = build_primitive_entries(
        hand_type,
        hand_side,
        prims,
        name_prefix=name_prefix,
    )
    written: List[str] = []
    if write_library:
        existing = set(list_poses(hand_type, hand_side=hand_side))
        for entry in entries:
            if entry.name in existing and not overwrite:
                continue
            err = upsert_pose(
                entry.canonical_hand_type,
                entry.name,
                entry.positions,
                duration=entry.duration,
                description=entry.description,
                source=entry.source,
                space=entry.space,
                unit=entry.unit,
                hand_side=entry.hand_side,
                replace_multi_keyframe=bool(overwrite),
            )
            if err:
                raise ValueError(err)
            written.append(entry.name)
    bundle_path: Optional[Path] = None
    if output_path:
        bundle = build_bundle(entries, exporter="export_primitive_poses")
        bundle_path = write_bundle(bundle, output_path)
        if not write_library:
            written = [e.name for e in entries]
    return len(written) if write_library else len(entries), bundle_path, written


def main(args=None) -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Export planable static primitive final poses → Pose Library / bundle"
    )
    parser.add_argument("--hand-type", required=True)
    parser.add_argument("--hand-side", default="left", choices=("left", "right"))
    parser.add_argument(
        "--set",
        dest="set_name",
        default="planable",
        choices=("planable", "static_all"),
        help="default primitive set when --primitive omitted",
    )
    parser.add_argument(
        "--primitive",
        action="append",
        default=[],
        help="primitive name (repeatable); overrides --set",
    )
    parser.add_argument("--name-prefix", default=DEFAULT_PRESET_NAME_PREFIX)
    parser.add_argument("--output", default="", help="write *.pose_bundle.json")
    parser.add_argument(
        "--write-library",
        action="store_true",
        help="also upsert into HAND_GESTURE_POSE_DIR",
    )
    parser.add_argument("--overwrite", action="store_true")
    ns, _ = parser.parse_known_args(args=args)
    try:
        count, path, names = export_primitive_poses(
            ns.hand_type,
            ns.hand_side,
            primitives=ns.primitive or None,
            set_name=ns.set_name,
            name_prefix=ns.name_prefix,
            write_library=bool(ns.write_library),
            output_path=ns.output or None,
            overwrite=bool(ns.overwrite),
        )
    except (ValueError, FileNotFoundError) as exc:
        print(f"失败: {exc}", file=sys.stderr)
        sys.exit(1)
    print(f"exported={count} names={','.join(names)}" + (f" bundle={path}" if path else ""))
    sys.exit(0)


if __name__ == "__main__":
    main()
