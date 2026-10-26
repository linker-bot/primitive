"""Pose Library 跨机迁移 bundle（传输层）。

落盘真相源仍是 ``HAND_GESTURE_POSE_DIR`` 下的 ``{canonical}_{side}.json``；
本模块只负责打包/解包与合并策略，最终写入仍走 ``upsert_pose``。

schema v2：semantic_v3 默认 30 维 + ``unit``；v1 仍可读（hardware 0~255）。
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .hand_config import HandConfig, resolve_hand_config_key
from .pose_library import (
    DEFAULT_SAVE_SPACE,
    DEFAULT_SAVE_UNIT,
    PoseKeyframe,
    bucket_exists,
    list_bucket_poses,
    list_poses,
    load_bucket_pose,
    normalize_hand_side,
    pose_dir,
    positions_named_dict,
    upsert_pose,
    validate_pose_name,
    validate_positions,
)
from .semantic_30 import SEMANTIC_DIM, project_o20_to_s30

BUNDLE_KIND = "hand_gesture_pose_bundle"
BUNDLE_SCHEMA_VERSION = 2
BUNDLE_SCHEMA_VERSION_LEGACY = 1
MERGE_STRATEGIES = frozenset({"skip", "overwrite", "rename"})

_SIDE_FILE_RE = re.compile(r"^([a-z0-9]+)_(left|right)\.json$", re.IGNORECASE)
_LEGACY_FILE_RE = re.compile(r"^([a-z0-9]+)\.json$", re.IGNORECASE)
_POSE_DIR_LOCK = __import__("threading").Lock()


@dataclass
class BundleEntry:
    canonical_hand_type: str
    hand_side: str
    name: str
    positions: List[float]
    num_joints: int
    space: str = DEFAULT_SAVE_SPACE
    unit: str = DEFAULT_SAVE_UNIT
    duration: float = 0.6
    description: str = ""
    source: str = "teach"
    origin_primitive: Optional[str] = None
    updated_at: str = ""
    keyframes: Tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        d = {
            "canonical_hand_type": self.canonical_hand_type,
            "hand_side": self.hand_side,
            "name": self.name,
            "space": self.space,
            "unit": self.unit,
            "positions": [float(v) for v in self.positions],
            "num_joints": int(self.num_joints),
            "duration": float(self.duration),
            "description": self.description or "",
            "source": self.source or "teach",
            "updated_at": self.updated_at or "",
        }
        if self.origin_primitive:
            d["origin_primitive"] = self.origin_primitive
        # 旁路可读字段：导入仍只认 positions
        hand = None
        try:
            hand = HandConfig(self.canonical_hand_type)
            jnames, named = positions_named_dict(list(self.positions), space=self.space, hand=hand)
            d["joint_names"] = jnames
            d["positions_named"] = named
        except (FileNotFoundError, ValueError, OSError):
            hand = None
        if self.keyframes:
            kf_out = []
            for kf in self.keyframes:
                item = {
                    "name": str(kf.get("name") or "").strip().lower(),
                    "positions": [float(v) for v in (kf.get("positions") or [])],
                    "duration": float(kf.get("duration") or self.duration),
                }
                if hand is not None:
                    _, kf_named = positions_named_dict(
                        list(item["positions"]), space=self.space, hand=hand
                    )
                    item["positions_named"] = kf_named
                kf_out.append(item)
            d["keyframes"] = kf_out
        return d


@dataclass
class Bundle:
    entries: List[BundleEntry] = field(default_factory=list)
    exported_at: str = ""
    exporter: str = "cli"
    machine_hint: str = ""
    schema_version: int = BUNDLE_SCHEMA_VERSION

    def to_dict(self, *, with_checksum: bool = True) -> dict:
        entries = [e.to_dict() for e in self.entries]
        payload = {
            "schema_version": int(self.schema_version),
            "kind": BUNDLE_KIND,
            "exported_at": self.exported_at or _now_iso(),
            "exporter": self.exporter or "cli",
            "machine_hint": self.machine_hint or "",
            "entries": entries,
        }
        if with_checksum:
            payload["checksum"] = {
                "algo": "sha256",
                "of": "entries",
                "value": checksum_entries(entries),
            }
        return payload


@dataclass
class ImportResult:
    imported: int = 0
    skipped: int = 0
    failed: int = 0
    messages: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        parts = [
            f"imported={self.imported}",
            f"skipped={self.skipped}",
            f"failed={self.failed}",
        ]
        if self.messages:
            parts.append("notes=" + "; ".join(self.messages[:8]))
        return " ".join(parts)

    @property
    def success(self) -> bool:
        """有失败，或过滤后零导入（全被 skip）视为不成功。"""
        if self.failed > 0:
            return False
        if self.imported == 0 and self.skipped > 0:
            return False
        return True


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def checksum_entries(entries: Sequence[dict]) -> str:
    """对 entries 的规范 JSON 做 sha256（稳定键序）。"""
    canonical = json.dumps(
        list(entries), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _normalize_bundle_entry(entry: BundleEntry) -> BundleEntry:
    """旧 20 维 semantic 条目 → S30；导出/导入统一维度。"""
    space = (entry.space or DEFAULT_SAVE_SPACE).strip().lower()
    unit = (entry.unit or DEFAULT_SAVE_UNIT).strip().lower()
    positions = [float(v) for v in entry.positions]
    keyframes = tuple(entry.keyframes or ())

    def _with(**kwargs) -> BundleEntry:
        return BundleEntry(
            canonical_hand_type=entry.canonical_hand_type,
            hand_side=entry.hand_side,
            name=entry.name,
            positions=kwargs.get("positions", positions),
            num_joints=kwargs.get("num_joints", entry.num_joints),
            space=kwargs.get("space", space),
            unit=kwargs.get("unit", unit if unit in ("rad", "legacy255") else "legacy255"),
            duration=entry.duration,
            description=entry.description,
            source=entry.source,
            origin_primitive=entry.origin_primitive,
            updated_at=entry.updated_at,
            keyframes=kwargs.get("keyframes", keyframes),
        )

    if space == "hardware":
        return entry
    if space == "semantic_o20" or (space == "semantic_v3" and len(positions) == 20):
        positions = project_o20_to_s30(positions[:20])
        # keyframes 若仍是 20 维也投影
        new_kfs = []
        for kf in keyframes:
            pos = [float(v) for v in (kf.get("positions") or [])]
            if len(pos) == 20:
                pos = project_o20_to_s30(pos)
            elif len(pos) >= SEMANTIC_DIM:
                pos = pos[:SEMANTIC_DIM]
            new_kfs.append(
                {
                    "name": str(kf.get("name") or "").strip().lower(),
                    "positions": pos,
                    "duration": float(kf.get("duration") or entry.duration),
                }
            )
        return _with(
            positions=positions,
            num_joints=SEMANTIC_DIM,
            space="semantic_v3",
            keyframes=tuple(new_kfs),
        )
    if space == "semantic_v3" and len(positions) >= SEMANTIC_DIM:
        new_kfs = []
        for kf in keyframes:
            pos = [float(v) for v in (kf.get("positions") or [])]
            if len(pos) >= SEMANTIC_DIM:
                pos = pos[:SEMANTIC_DIM]
            new_kfs.append(
                {
                    "name": str(kf.get("name") or "").strip().lower(),
                    "positions": pos,
                    "duration": float(kf.get("duration") or entry.duration),
                }
            )
        return _with(
            positions=positions[:SEMANTIC_DIM],
            num_joints=SEMANTIC_DIM,
            space="semantic_v3",
            keyframes=tuple(new_kfs),
        )
    return entry


def _expected_dims(space: str, hand: HandConfig) -> Tuple[int, str]:
    space_l = (space or DEFAULT_SAVE_SPACE).strip().lower()
    if space_l == "semantic_v3":
        return SEMANTIC_DIM, space_l
    if space_l == "semantic_o20":
        return 20, space_l
    return hand.num_joints, "hardware"


def _parse_bucket_filename(name: str) -> Optional[Tuple[str, Optional[str]]]:
    """返回 (canonical, side|None)；无法识别则 None。"""
    m = _SIDE_FILE_RE.match(name)
    if m:
        return resolve_hand_config_key(m.group(1)), m.group(2).lower()
    m = _LEGACY_FILE_RE.match(name)
    if m:
        key = m.group(1).lower()
        if key.endswith("corrupt"):
            return None
        return resolve_hand_config_key(key), None
    return None


def discover_library_buckets(
    root: Optional[Path] = None,
) -> List[Tuple[str, Optional[str]]]:
    """扫描姿态库目录，返回 (canonical, side) 列表（legacy side=None）。"""
    base = Path(root) if root is not None else pose_dir()
    if not base.is_dir():
        return []
    found: List[Tuple[str, Optional[str]]] = []
    for path in sorted(base.glob("*.json")):
        parsed = _parse_bucket_filename(path.name)
        if parsed is None:
            continue
        found.append(parsed)
    return found


def collect_library_entries(
    *,
    hand_type: Optional[str] = None,
    hand_side: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    root: Optional[Path] = None,
) -> List[BundleEntry]:
    """从本地 Pose Library 收集条目（导出时规范为 S30 semantic_v3）。"""
    name_filter = {n.strip().lower() for n in (names or []) if str(n).strip()}
    side_filter = normalize_hand_side(hand_side) if hand_side else None
    entries: List[BundleEntry] = []

    if hand_type:
        canonical = resolve_hand_config_key(hand_type)
        if side_filter:
            if bucket_exists(canonical, side_filter):
                buckets: List[Tuple[str, Optional[str], bool]] = [(canonical, side_filter, False)]
            elif bucket_exists(canonical, None):
                buckets = [(canonical, None, True)]
            else:
                buckets = []
        else:
            buckets = []
            for s in ("left", "right"):
                if bucket_exists(canonical, s):
                    buckets.append((canonical, s, False))
            if bucket_exists(canonical, None):
                buckets.append((canonical, None, False))
    else:
        buckets = [(c, s, False) for c, s in discover_library_buckets(root)]

    seen: set[Tuple[str, str, str]] = set()
    side_owned: Dict[str, set[str]] = {}
    for canonical, side, remap_legacy_to_filter in buckets:
        if side in ("left", "right") and not remap_legacy_to_filter:
            side_owned.setdefault(canonical, set()).update(list_bucket_poses(canonical, side))

    for canonical, side, remap_legacy_to_filter in buckets:
        try:
            hand = HandConfig(canonical)
        except FileNotFoundError:
            continue
        if remap_legacy_to_filter:
            pose_names = list_bucket_poses(canonical, None)
            out_side = side_filter or ""

            def _load(n: str, _c=canonical):
                return load_bucket_pose(_c, n, hand_side=None)

        else:
            pose_names = list_bucket_poses(canonical, side)
            out_side = side or ""

            def _load(n: str, _c=canonical, _s=side):
                return load_bucket_pose(_c, n, hand_side=_s)

        for pname in pose_names:
            if name_filter and pname not in name_filter:
                continue
            if (
                side is None
                and not remap_legacy_to_filter
                and not side_filter
                and pname in side_owned.get(canonical, set())
            ):
                continue
            entry = _load(pname)
            if entry is None:
                continue
            key = (canonical, out_side or "_", pname)
            if key in seen:
                continue
            seen.add(key)
            bundle_entry = BundleEntry(
                canonical_hand_type=canonical,
                hand_side=out_side,
                name=entry.name,
                positions=list(entry.positions),
                num_joints=int(entry.num_joints or hand.num_joints),
                space=entry.space,
                unit=getattr(entry, "unit", DEFAULT_SAVE_UNIT),
                duration=float(entry.duration),
                description=entry.description,
                source=entry.source,
                updated_at=entry.updated_at,
                keyframes=tuple(
                    {
                        "name": kf.name,
                        "positions": list(kf.positions),
                        "duration": float(kf.duration),
                    }
                    for kf in (entry.keyframes or ())
                ),
            )
            entries.append(_normalize_bundle_entry(bundle_entry))
    return entries


def build_bundle(
    entries: Iterable[BundleEntry],
    *,
    exporter: str = "cli",
    machine_hint: str = "",
) -> Bundle:
    normalized = [_normalize_bundle_entry(e) for e in entries]
    return Bundle(
        entries=normalized,
        exported_at=_now_iso(),
        exporter=exporter,
        machine_hint=machine_hint,
    )


def write_bundle(bundle: Bundle, output_path: str | Path) -> Path:
    path = Path(output_path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(bundle.to_dict(with_checksum=True), ensure_ascii=False, indent=2) + "\n"
    path.write_text(content, encoding="utf-8")
    return path


def read_bundle(input_path: str | Path, *, verify_checksum: bool = True) -> Bundle:
    path = Path(input_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"bundle not found: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("bundle root must be an object")
    if str(raw.get("kind") or "") != BUNDLE_KIND:
        raise ValueError(f"unexpected kind: {raw.get('kind')!r}")
    version = int(raw.get("schema_version") or 0)
    if version not in (BUNDLE_SCHEMA_VERSION, BUNDLE_SCHEMA_VERSION_LEGACY):
        raise ValueError(f"unsupported schema_version: {version}")
    entries_raw = raw.get("entries")
    if not isinstance(entries_raw, list):
        raise ValueError("entries must be a list")
    if verify_checksum and isinstance(raw.get("checksum"), dict):
        expected = str(raw["checksum"].get("value") or "")
        actual = checksum_entries(entries_raw)
        if expected and expected != actual:
            raise ValueError("bundle checksum mismatch")
    entries: List[BundleEntry] = []
    for item in entries_raw:
        if not isinstance(item, dict):
            continue
        space = str(
            item.get("space") or ("hardware" if version == 1 else DEFAULT_SAVE_SPACE)
        ).lower()
        unit = str(
            item.get("unit") or ("legacy255" if version == 1 else DEFAULT_SAVE_UNIT)
        ).lower()
        positions = [float(v) for v in (item.get("positions") or [])]
        raw_kfs = item.get("keyframes") or []
        keyframes: Tuple[dict, ...] = ()
        if isinstance(raw_kfs, list) and raw_kfs:
            parsed = []
            for kf in raw_kfs:
                if not isinstance(kf, dict):
                    continue
                parsed.append(
                    {
                        "name": str(kf.get("name") or "").strip().lower(),
                        "positions": [float(x) for x in (kf.get("positions") or [])],
                        "duration": float(kf.get("duration") or item.get("duration") or 0.6),
                    }
                )
            keyframes = tuple(parsed)
        entry = BundleEntry(
            canonical_hand_type=resolve_hand_config_key(
                str(item.get("canonical_hand_type") or "")
            ),
            hand_side=str(item.get("hand_side") or "").strip().lower(),
            name=str(item.get("name") or "").strip().lower(),
            positions=positions,
            num_joints=int(item.get("num_joints") or len(positions)),
            space=space,
            unit=unit,
            duration=float(item.get("duration") or 0.6),
            description=str(item.get("description") or ""),
            source=str(item.get("source") or "import"),
            origin_primitive=(
                str(item["origin_primitive"]).strip().lower()
                if item.get("origin_primitive")
                else None
            ),
            updated_at=str(item.get("updated_at") or ""),
            keyframes=keyframes,
        )
        entries.append(_normalize_bundle_entry(entry))
    return Bundle(
        entries=entries,
        exported_at=str(raw.get("exported_at") or ""),
        exporter=str(raw.get("exporter") or ""),
        machine_hint=str(raw.get("machine_hint") or ""),
        schema_version=version,
    )


def _unique_name(base: str, existing: set[str]) -> str:
    if base not in existing:
        return base
    for i in range(2, 1000):
        cand = f"{base}_{i}" if not base.endswith("_imp") else f"{base}{i}"
        if cand not in existing:
            return cand
    return f"{base}_{int(time.time())}"


def import_bundle(
    bundle: Bundle,
    *,
    strategy: str = "skip",
    hand_type: Optional[str] = None,
    hand_side: Optional[str] = None,
    default_side: Optional[str] = None,
) -> ImportResult:
    """将 bundle 写入 Pose Library。"""
    strategy_l = (strategy or "skip").strip().lower()
    if strategy_l not in MERGE_STRATEGIES:
        raise ValueError(f"strategy must be one of {sorted(MERGE_STRATEGIES)}")

    filter_hand = resolve_hand_config_key(hand_type) if hand_type else None
    filter_side = normalize_hand_side(hand_side) if hand_side else None
    fallback_side = normalize_hand_side(default_side) if default_side else None

    result = ImportResult()
    existing_cache: Dict[Tuple[str, Optional[str]], set[str]] = {}

    for raw_entry in bundle.entries:
        entry = _normalize_bundle_entry(raw_entry)
        canonical = resolve_hand_config_key(entry.canonical_hand_type)
        if filter_hand and canonical != filter_hand:
            result.skipped += 1
            continue
        side = normalize_hand_side(entry.hand_side) if entry.hand_side else fallback_side
        if filter_side is not None and side != filter_side:
            result.skipped += 1
            continue

        name_err = validate_pose_name(entry.name)
        if name_err:
            result.failed += 1
            result.messages.append(f"{canonical}/{side or '-'}/{entry.name}: {name_err}")
            continue
        try:
            hand = HandConfig(canonical)
        except FileNotFoundError as exc:
            result.failed += 1
            result.messages.append(f"{canonical}: {exc}")
            continue

        expect_n, space_l = _expected_dims(entry.space, hand)
        unit_l = (entry.unit or DEFAULT_SAVE_UNIT).strip().lower()
        if space_l == "hardware":
            unit_l = "legacy255"
        if entry.num_joints and entry.num_joints != expect_n and space_l == "hardware":
            result.failed += 1
            result.messages.append(
                f"{canonical}/{entry.name}: joint count {entry.num_joints} != {expect_n}"
            )
            continue
        pos_err = validate_positions(list(entry.positions), expect_n, unit=unit_l)
        if pos_err:
            result.failed += 1
            result.messages.append(f"{canonical}/{entry.name}: {pos_err}")
            continue

        cache_key = (canonical, side)
        if cache_key not in existing_cache:
            existing_cache[cache_key] = set(list_poses(canonical, hand_side=side))
        existing = existing_cache[cache_key]
        target_name = entry.name.strip().lower()

        if target_name in existing:
            if strategy_l == "skip":
                result.skipped += 1
                continue
            if strategy_l == "rename":
                target_name = _unique_name(f"{target_name}_imp", existing)
                rename_err = validate_pose_name(target_name)
                if rename_err:
                    result.failed += 1
                    result.messages.append(f"{entry.name}→{target_name}: {rename_err}")
                    continue

        kf_objs = None
        if entry.keyframes:
            kf_objs = [
                PoseKeyframe(
                    name=str(kf.get("name") or "").strip().lower(),
                    positions=[float(v) for v in (kf.get("positions") or [])],
                    duration=float(kf.get("duration") or entry.duration),
                )
                for kf in entry.keyframes
            ]
        up_err = upsert_pose(
            canonical,
            target_name,
            list(entry.positions),
            duration=float(entry.duration),
            description=entry.description,
            source=entry.source or "import_bundle",
            space=space_l,
            unit=unit_l,
            hand_side=side,
            keyframes=kf_objs,
            # overwrite 允许用无 keyframes 的条目整表替换已有多帧姿态
            replace_multi_keyframe=(strategy_l == "overwrite"),
        )
        if up_err:
            result.failed += 1
            result.messages.append(f"{canonical}/{target_name}: {up_err}")
            continue
        existing.add(target_name)
        result.imported += 1

    if result.imported == 0 and result.skipped > 0 and result.failed == 0:
        filt = []
        if filter_hand:
            filt.append(f"hand={filter_hand}")
        if filter_side:
            filt.append(f"side={filter_side}")
        result.messages.append(
            "no poses imported (all skipped"
            + (f" by filter {' '.join(filt)}" if filt else "")
            + "; check hand_type/hand_side match or use overwrite)"
        )
    return result


def export_library_to_file(
    output_path: str | Path,
    *,
    hand_type: Optional[str] = None,
    hand_side: Optional[str] = None,
    names: Optional[Sequence[str]] = None,
    exporter: str = "cli",
    machine_hint: str = "",
    root: Optional[Path] = None,
) -> Tuple[Path, int]:
    """导出库到 bundle 文件，返回 (path, count)。"""
    if root is not None:
        entries = _collect_from_root(
            Path(root),
            hand_type=hand_type,
            hand_side=hand_side,
            names=names,
        )
    else:
        entries = collect_library_entries(
            hand_type=hand_type,
            hand_side=hand_side,
            names=names,
        )
    bundle = build_bundle(entries, exporter=exporter, machine_hint=machine_hint)
    path = write_bundle(bundle, output_path)
    return path, len(entries)


def _collect_from_root(
    root: Path,
    *,
    hand_type: Optional[str],
    hand_side: Optional[str],
    names: Optional[Sequence[str]],
) -> List[BundleEntry]:
    """从任意目录读桶文件（加锁临时覆盖 HAND_GESTURE_POSE_DIR）。"""
    import os

    root = Path(root).expanduser()
    with _POSE_DIR_LOCK:
        prev = os.environ.get("HAND_GESTURE_POSE_DIR")
        os.environ["HAND_GESTURE_POSE_DIR"] = str(root)
        try:
            return collect_library_entries(
                hand_type=hand_type,
                hand_side=hand_side,
                names=names,
                root=root,
            )
        finally:
            if prev is None:
                os.environ.pop("HAND_GESTURE_POSE_DIR", None)
            else:
                os.environ["HAND_GESTURE_POSE_DIR"] = prev


def import_bundle_file(
    input_path: str | Path,
    *,
    strategy: str = "skip",
    hand_type: Optional[str] = None,
    hand_side: Optional[str] = None,
    default_side: Optional[str] = None,
    verify_checksum: bool = True,
) -> ImportResult:
    bundle = read_bundle(input_path, verify_checksum=verify_checksum)
    return import_bundle(
        bundle,
        strategy=strategy,
        hand_type=hand_type,
        hand_side=hand_side,
        default_side=default_side,
    )


def import_library_dir(
    source_dir: str | Path,
    *,
    strategy: str = "skip",
    hand_type: Optional[str] = None,
    hand_side: Optional[str] = None,
    default_side: Optional[str] = None,
) -> ImportResult:
    """把另一目录下的 Pose Library JSON 当作源库导入当前 HAND_GESTURE_POSE_DIR。"""
    entries = _collect_from_root(
        Path(source_dir).expanduser(),
        hand_type=hand_type,
        hand_side=hand_side,
        names=None,
    )
    for e in entries:
        if not e.hand_side and (hand_side or default_side):
            e.hand_side = (hand_side or default_side or "").strip().lower()
    return import_bundle(
        build_bundle(entries, exporter="legacy_dir"),
        strategy=strategy,
        hand_type=hand_type,
        hand_side=hand_side,
        default_side=default_side,
    )
