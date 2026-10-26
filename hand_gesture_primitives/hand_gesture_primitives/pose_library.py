"""用户示教姿态库 — 按规范手型（及可选手侧）分桶持久化。

存储目录必须由 ``HAND_GESTURE_POSE_DIR`` 显式指定。
带侧文件为 ``{canonical}_{side}.json``，旧版无侧文件为 ``{canonical}.json``。
不读取用户 HOME 下的旧姿态目录，也不自动迁移或跟随该目录的链接。例如::

  export HAND_GESTURE_POSE_DIR=/path/to/persistent/hand_gesture_poses

坐标系: 新建默认 ``semantic_v3``（30 维 S30）+ ``unit: rad``；仍可读
``hardware`` / ``semantic_o20`` / ``unit: legacy255`` 以兼容旧落盘。

可读旁路: 写入时附带 ``joint_names``（与 ``positions`` 同序）与
``positions_named``（名字→值）；**真相源仍是 ``positions`` 数组**，导入忽略旁路字段。

多中间态: 可选 ``keyframes[]``（schema version=3）；禁止与 ``space=hardware``
混用。位置与量纲约定见仓库根目录的 ``docs/workbench.md``。
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Set, Tuple, Union

from .hand_config import HandConfig, resolve_hand_config_key
from .semantic_30 import SEMANTIC_30_NAMES, SEMANTIC_DIM, project_o20_to_s30

if TYPE_CHECKING:
    from .keyframe_runner import MotionProgram

logger = logging.getLogger(__name__)

POSE_SCHEMA_VERSION = 3
# Compatibility symbol for downstream tooling/tests; never used as a directory fallback.
POSE_DIR: Optional[Path] = None

_POSE_NAME_RE = re.compile(r"^[a-z0-9_]+$")
_MIN_DURATION = 0.2
_DEFAULT_DURATION = 0.6
_DEFAULT_KEYFRAME_NAME = "default"
_ALLOWED_SPACES = frozenset({"hardware", "semantic_v3", "semantic_o20"})
_ALLOWED_UNITS = frozenset({"rad", "legacy255"})
# 示教新建默认落盘空间（跨机型；hardware 仍可读）
DEFAULT_SAVE_SPACE = "semantic_v3"
DEFAULT_SAVE_UNIT = "rad"

# 指令 / 系统保留字（注册原语名在 validate 时动态并入）
_COMMAND_RESERVED: Set[str] = {
    "stop",
    "mode",
    "pose",
    "pose_list",
    "save_pose",
    "auto",
    "target",
    "target_id",
    "teach",
    "stream",
    "primitive",
    "named_pose",
}


def joint_names_for_space(space: str, hand: HandConfig) -> List[str]:
    """与 ``positions`` 同序的关节/通道名（供落盘旁路字段，非执行契约）。"""
    space_l = (space or "hardware").strip().lower()
    if space_l == "semantic_v3":
        return list(SEMANTIC_30_NAMES)
    if space_l == "semantic_o20":
        # O20 语义序与 o20.yaml joint_names 一致
        if resolve_hand_config_key(hand.hand_joint) == "o20":
            return list(hand.joint_names)
        return list(HandConfig("o20").joint_names)
    return list(hand.joint_names)


def positions_named_dict(
    positions: List[float],
    *,
    space: str,
    hand: HandConfig,
) -> Tuple[List[str], Dict[str, float]]:
    """构造 ``joint_names`` + ``positions_named``；长度不足时用 ``j{i}`` 补齐。"""
    names = joint_names_for_space(space, hand)
    n = len(positions)
    if len(names) < n:
        names = list(names) + [f"j{i}" for i in range(len(names), n)]
    elif len(names) > n:
        names = names[:n]
    named = {names[i]: float(positions[i]) for i in range(n)}
    return names, named


def is_legal_keyframe_name(name: str) -> bool:
    """关键帧名合法且非纯数字（避免 ``pose grip 2`` 被解析成 progress）。"""
    n = (name or "").strip().lower()
    if not n or not _POSE_NAME_RE.match(n):
        return False
    try:
        float(n)
    except ValueError:
        return True
    return False


@dataclass(frozen=True)
class PoseKeyframe:
    """Pose 内单个位置关键帧（勿与 YAML gestures.stages 时长旋钮混淆）。"""

    name: str
    positions: List[float]
    duration: float = _DEFAULT_DURATION


@dataclass(frozen=True)
class PoseEntry:
    name: str
    positions: List[float]
    space: str = "hardware"
    duration: float = _DEFAULT_DURATION
    num_joints: int = 0
    description: str = ""
    source: str = "teach"
    updated_at: str = ""
    unit: str = "legacy255"
    keyframes: Tuple[PoseKeyframe, ...] = ()

    def effective_keyframes(self) -> Tuple[PoseKeyframe, ...]:
        """无 keyframes 时退化为单帧 default（顶层 positions）。"""
        if self.keyframes:
            return self.keyframes
        return (
            PoseKeyframe(
                name=_DEFAULT_KEYFRAME_NAME,
                positions=list(self.positions),
                duration=float(self.duration),
            ),
        )


def pose_dir() -> Path:
    """Explicit station/library path; never fall back to an unrelated HOME library."""
    override = os.environ.get("HAND_GESTURE_POSE_DIR", "").strip()
    if not override:
        raise ValueError(
            "HAND_GESTURE_POSE_DIR is required. Set it to an explicit pose library directory "
            "or pass --pose-dir PATH to teach_panel (pose_dir:=PATH in ROS launch)."
        )
    path = Path(override).expanduser().absolute()
    if path.exists() and not path.is_dir():
        raise ValueError(f"Pose directory is not a directory: {path}")
    if path.resolve() != path:
        raise ValueError(f"Pose directory must not traverse a symlink: {path}")
    return path


def normalize_hand_side(hand_side: Optional[str]) -> Optional[str]:
    """合法 left/right 返回小写；空/None 表示使用旧版无侧文件。"""
    if hand_side is None:
        return None
    s = str(hand_side).strip().lower()
    if not s:
        return None
    if s in ("left", "right"):
        return s
    raise ValueError(f"hand_side 无效: {hand_side!r}（应为 left/right）")


def _pose_path(canonical: str, side: Optional[str] = None) -> Path:
    root = pose_dir()
    if side:
        return root / f"{canonical}_{side}.json"
    return root / f"{canonical}.json"


def bucket_exists(hand_type_or_joint: str, hand_side: Optional[str] = None) -> bool:
    """对应桶文件是否存在（不触发 legacy 回退）。"""
    canonical = resolve_hand_config_key(hand_type_or_joint)
    side = normalize_hand_side(hand_side)
    return _pose_path(canonical, side).is_file()


def list_bucket_poses(
    hand_type_or_joint: str,
    hand_side: Optional[str] = None,
) -> List[str]:
    """仅列指定桶文件中的姿态（无 legacy 回退）。"""
    canonical = resolve_hand_config_key(hand_type_or_joint)
    side = normalize_hand_side(hand_side)
    data = _load_file(canonical, side)
    return sorted(str(k).lower() for k in (data.get("poses") or {}).keys())


def load_bucket_pose(
    hand_type_or_joint: str,
    name: str,
    hand_side: Optional[str] = None,
) -> Optional[PoseEntry]:
    """仅从指定桶文件加载（无 legacy 回退）。"""
    canonical = resolve_hand_config_key(hand_type_or_joint)
    n = (name or "").strip().lower()
    if not n:
        return None
    hand = HandConfig(canonical)
    side = normalize_hand_side(hand_side)
    raw = (_load_file(canonical, side).get("poses") or {}).get(n)
    if raw is None:
        return None
    return _parse_entry(n, raw, hand.num_joints)


def reserved_pose_names() -> Set[str]:
    from .primitives import PRIMITIVE_REGISTRY

    return set(_COMMAND_RESERVED) | set(PRIMITIVE_REGISTRY.keys())


def validate_pose_name(name: str) -> Optional[str]:
    """合法返回 None，否则返回错误说明。"""
    n = (name or "").strip().lower()
    if not n:
        return "姿态名为空"
    if not _POSE_NAME_RE.match(n):
        return f"姿态名非法 '{name}'（仅允许小写字母/数字/下划线）"
    if n in reserved_pose_names():
        return f"姿态名 '{n}' 为保留字或已注册原语，请换名"
    return None


def validate_positions(
    positions: List[float],
    num_joints: int,
    *,
    unit: str = "legacy255",
) -> Optional[str]:
    if len(positions) != num_joints:
        return f"关节数不匹配: got {len(positions)}, expect {num_joints}"
    unit_l = (unit or "legacy255").strip().lower()
    for i, v in enumerate(positions):
        try:
            fv = float(v)
        except (TypeError, ValueError):
            return f"positions[{i}] 非数值"
        if not math.isfinite(fv):
            return f"positions[{i}]={fv} 非有限数值"
        if unit_l == "legacy255" and (fv < 0.0 or fv > 255.0):
            return f"positions[{i}]={fv} 超出 [0, 255]"
        if unit_l == "rad" and (fv < -2.0 * math.pi or fv > 2.0 * math.pi):
            return f"positions[{i}]={fv} 超出弧度合理范围 [-2π, 2π]"
    return None


def _resolve_unit_for_space(space: str, unit: Optional[str]) -> str:
    space_l = (space or DEFAULT_SAVE_SPACE).strip().lower()
    if space_l == "hardware":
        return "legacy255"
    unit_l = (unit or DEFAULT_SAVE_UNIT).strip().lower()
    if unit_l not in _ALLOWED_UNITS:
        return DEFAULT_SAVE_UNIT
    return unit_l


def _empty_file(canonical: str) -> dict:
    return {
        "version": POSE_SCHEMA_VERSION,
        "hand_joint": canonical.upper(),
        "canonical_hand_type": canonical,
        "poses": {},
    }


def _quarantine_corrupt(path: Path) -> Optional[Path]:
    """将损坏文件改名为 *.corrupt.<ts>，避免后续 upsert 覆盖未备份内容。"""
    if not path.is_file():
        return None
    dest = path.with_name(f"{path.name}.corrupt.{int(time.time())}")
    try:
        path.replace(dest)
        logger.error("姿态库文件损坏，已隔离: %s → %s", path, dest)
        return dest
    except OSError as exc:
        logger.error("姿态库损坏且隔离失败 %s: %s", path, exc)
        return None


def _load_file(canonical: str, side: Optional[str] = None) -> dict:
    path = _pose_path(canonical, side)
    if not path.is_file():
        return _empty_file(canonical)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        _quarantine_corrupt(path)
        return _empty_file(canonical)
    except OSError:
        return _empty_file(canonical)
    if not isinstance(raw, dict):
        _quarantine_corrupt(path)
        return _empty_file(canonical)
    poses = raw.get("poses") or {}
    if not isinstance(poses, dict):
        poses = {}
    return {
        "version": _parse_pose_version(raw.get("version")),
        "hand_joint": str(raw.get("hand_joint") or canonical.upper()),
        "canonical_hand_type": str(raw.get("canonical_hand_type") or canonical).lower(),
        "poses": poses,
        "hand_side": str(raw.get("hand_side") or (side or "")),
    }


def _parse_pose_version(raw_version) -> int:
    try:
        return int(raw_version if raw_version is not None else POSE_SCHEMA_VERSION)
    except (TypeError, ValueError):
        logger.warning("姿态库 version 字段无效 %r，回退为 %s", raw_version, POSE_SCHEMA_VERSION)
        return POSE_SCHEMA_VERSION


@contextmanager
def _pose_file_lock(canonical: str, side: Optional[str]) -> Iterator[None]:
    """对姿态库 JSON 的读写加排他锁，避免并发 upsert/delete 静默丢写。"""
    root = pose_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = _pose_path(canonical, side)
    lock_path = path.with_name(path.name + ".lock")
    with open(lock_path, "a+", encoding="utf-8") as lock_f:
        import time
        deadline = time.monotonic() + 1.0
        while True:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("姿态库忙，请稍后重试")
                time.sleep(0.01)
        try:
            yield
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def _load_file_with_fallback(canonical: str, side: Optional[str]) -> dict:
    """优先读侧文件；侧文件不存在则回退旧版无侧文件。"""
    if side:
        side_path = _pose_path(canonical, side)
        if side_path.is_file():
            return _load_file(canonical, side)
        return _load_file(canonical, None)
    return _load_file(canonical, None)


def _save_file(canonical: str, data: dict, side: Optional[str] = None) -> None:
    root = pose_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = _pose_path(canonical, side)
    payload = {
        "version": POSE_SCHEMA_VERSION,
        "hand_joint": data.get("hand_joint") or canonical.upper(),
        "canonical_hand_type": canonical,
        "poses": data.get("poses") or {},
    }
    if side:
        payload["hand_side"] = side
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(root), suffix=".tmp")
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    except BaseException:
        os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    os.close(fd)
    os.replace(tmp, str(path))


def _parse_keyframes(
    raw_kfs: object,
    *,
    num_joints: int,
    unit: str,
) -> Optional[Tuple[PoseKeyframe, ...]]:
    if raw_kfs is None:
        return ()
    if not isinstance(raw_kfs, list) or not raw_kfs:
        return None
    seen: Set[str] = set()
    out: List[PoseKeyframe] = []
    for item in raw_kfs:
        if not isinstance(item, dict):
            return None
        kf_name = str(item.get("name") or "").strip().lower()
        if not kf_name or not is_legal_keyframe_name(kf_name):
            return None
        if kf_name in seen:
            return None
        if kf_name in _COMMAND_RESERVED:
            return None
        seen.add(kf_name)
        positions = item.get("positions")
        if not isinstance(positions, list):
            return None
        if validate_positions(positions, num_joints, unit=unit) is not None:
            return None
        duration = float(item.get("duration", _DEFAULT_DURATION))
        if duration < _MIN_DURATION:
            duration = _MIN_DURATION
        out.append(
            PoseKeyframe(
                name=kf_name,
                positions=[float(v) for v in positions],
                duration=duration,
            )
        )
    return tuple(out)


def _parse_entry(name: str, raw: dict, expect_joints: int) -> Optional[PoseEntry]:
    if not isinstance(raw, dict):
        return None
    positions = raw.get("positions")
    if not isinstance(positions, list):
        return None
    num_joints = int(raw.get("num_joints") or len(positions))
    space = str(raw.get("space", "hardware")).lower()
    if space not in _ALLOWED_SPACES:
        space = "hardware"
    if space == "hardware":
        if expect_joints and num_joints != expect_joints:
            return None
        unit = "legacy255"
    elif space == "semantic_v3":
        if num_joints != SEMANTIC_DIM:
            return None
        # 旧 semantic_v3 无 unit 字段：默认 legacy255（SRC 兼容）
        unit = str(raw.get("unit") or "legacy255").strip().lower()
        if unit not in _ALLOWED_UNITS:
            unit = "legacy255"
    elif space == "semantic_o20":
        if num_joints != 20:
            return None
        unit = str(raw.get("unit") or "legacy255").strip().lower()
        if unit not in _ALLOWED_UNITS:
            unit = "legacy255"
    else:
        unit = "legacy255"
    if validate_positions(positions, num_joints, unit=unit) is not None:
        return None
    duration = float(raw.get("duration", _DEFAULT_DURATION))
    if duration < _MIN_DURATION:
        duration = _MIN_DURATION
    keyframes = _parse_keyframes(raw.get("keyframes"), num_joints=num_joints, unit=unit)
    if keyframes is None:
        return None
    # 多关键帧禁止 hardware（避免混空间）；旧单点 hardware 仍可读
    if keyframes and space == "hardware":
        return None
    if keyframes:
        # 顶层 positions/duration 应与末帧一致；容错以末帧为准
        last = keyframes[-1]
        positions = list(last.positions)
        duration = float(last.duration)
        num_joints = len(positions)
    return PoseEntry(
        name=name,
        positions=[float(v) for v in positions],
        space=space,
        duration=duration,
        num_joints=num_joints,
        description=str(raw.get("description") or ""),
        source=str(raw.get("source") or "teach"),
        updated_at=str(raw.get("updated_at") or ""),
        unit=unit,
        keyframes=keyframes,
    )


def motion_program_from_entry(entry: PoseEntry) -> MotionProgram:
    """从 PoseEntry 构建 MotionProgram（供 NamedPose / 规划使用）。"""
    from .keyframe_runner import KeyframeSegment, MotionProgram

    segs = [
        KeyframeSegment(
            name=kf.name,
            target=tuple(float(v) for v in kf.positions),
            duration=float(kf.duration),
        )
        for kf in entry.effective_keyframes()
    ]
    return MotionProgram(
        ref=f"pose/{entry.name}",
        space=entry.space,
        unit=entry.unit,
        segments=segs,
    )


def pose_entry_to_hardware(
    entry: PoseEntry,
    hand_type_or_joint: str,
    *,
    radian_converter=None,
) -> Optional[List[float]]:
    """将库条目转为产品 hardware 0~255（Teach 滑条 / NamedPose hardware 路径）。"""
    canonical = resolve_hand_config_key(hand_type_or_joint)
    hand = HandConfig(canonical)
    if entry.space == "hardware":
        return [float(v) for v in entry.positions]

    semantic = [float(v) for v in entry.positions]
    if entry.space == "semantic_o20":
        semantic = project_o20_to_s30(semantic)
    while len(semantic) < SEMANTIC_DIM:
        semantic.append(0.0)
    semantic = semantic[:SEMANTIC_DIM]

    if entry.unit == "rad":
        if radian_converter is None:
            return None
        from .units import converter_rad_to_legacy255

        semantic = converter_rad_to_legacy255(radian_converter, semantic)

    if hand.has_from_semantic:
        return hand.from_semantic(semantic)
    from .semantic_30 import project_s30_to_o20

    return hand.to_hardware(project_s30_to_o20(semantic))


def list_poses(hand_type_or_joint: str, hand_side: Optional[str] = None) -> List[str]:
    """列出可成功 ``load_pose`` 的姿态名（跳过损坏/关节数不符条目）。"""
    return [s["name"] for s in list_pose_summaries(hand_type_or_joint, hand_side)]


def list_pose_summaries(
    hand_type_or_joint: str, hand_side: Optional[str] = None
) -> List[Dict[str, Any]]:
    """姿态摘要列表：``[{name, keyframe_count, keyframes:[{name,duration}]}]``。"""
    canonical = resolve_hand_config_key(hand_type_or_joint)
    side = normalize_hand_side(hand_side)
    data = _load_file_with_fallback(canonical, side)
    hand = HandConfig(canonical)
    expect = hand.num_joints
    out: List[Dict[str, Any]] = []
    for key, raw in (data.get("poses") or {}).items():
        name = str(key).lower()
        entry = _parse_entry(name, raw, expect)
        if entry is None:
            continue
        kfs = entry.effective_keyframes()
        # 真·多帧才标 count；单点退化 default 显示 0（UI 不强调）
        multi = bool(entry.keyframes)
        out.append(
            {
                "name": name,
                "keyframe_count": len(entry.keyframes) if multi else 0,
                "keyframes": [{"name": kf.name, "duration": float(kf.duration)} for kf in kfs]
                if multi
                else [],
                "duration": float(entry.duration),
                "description": entry.description or "",
                "space": entry.space,
                "unit": entry.unit,
            }
        )
    return sorted(out, key=lambda x: x["name"])


def prepare_pose_for_save(
    hand_type_or_joint: str,
    positions: List[float],
    *,
    space: Optional[str] = None,
    unit: Optional[str] = None,
    radian_converter=None,
) -> Union[Tuple[List[float], str, str], str]:
    """将 hardware 或已规范化的语义向量转为 ``upsert_pose`` 可用的三元组。

    成功返回 ``(positions, space, unit)``；失败返回错误字符串。
    有 ``radian_converter`` 且目标为 semantic 时默认落盘 ``unit=rad``。
    """
    canonical = resolve_hand_config_key(hand_type_or_joint)
    hand = HandConfig(canonical)
    space_l = (space or DEFAULT_SAVE_SPACE).strip().lower()
    if space_l not in _ALLOWED_SPACES:
        return f"space 无效: {space!r}（支持 hardware / semantic_v3 / semantic_o20）"

    n = len(positions)
    came_from_hardware_dof = False
    if space_l == "semantic_v3":
        if not hand.has_from_semantic:
            return f"{canonical} 无 from_semantic，无法落盘 semantic_v3"
        if n == hand.num_joints:
            save_positions = hand.to_semantic(list(positions))
            came_from_hardware_dof = True
        elif n == SEMANTIC_DIM:
            save_positions = [float(v) for v in positions]
        else:
            return (
                f"semantic_v3 需要 {hand.num_joints} 维 hardware 或 {SEMANTIC_DIM} 维 S30，"
                f"收到 {n} 维"
            )
    elif space_l == "semantic_o20":
        if n == hand.num_joints:
            save_positions = hand.from_hardware(list(positions))
            came_from_hardware_dof = True
        elif n == 20:
            save_positions = [float(v) for v in positions]
        else:
            return f"semantic_o20 需要 {hand.num_joints} 维 hardware 或 20 维 O20，收到 {n} 维"
    else:
        if n != hand.num_joints:
            return f"hardware 需要 {hand.num_joints} 维，收到 {n} 维"
        save_positions = [float(v) for v in positions]
        return save_positions, "hardware", "legacy255"

    unit_l = _resolve_unit_for_space(space_l, unit)
    if unit_l == "rad" and radian_converter is not None:
        from .semantic_30 import as_semantic_30
        from .units import converter_legacy255_to_rad

        semantic30 = as_semantic_30(save_positions)
        # 硬件维刚转出的语义恒为 legacy255，必须换算；纯 S30 输入则用 |v|>π 判定
        # （避免近开/近零 legacy255 被误当成已是 rad 而跳过转换）。
        if came_from_hardware_dof or max(abs(v) for v in semantic30) > math.pi:
            save_positions = converter_legacy255_to_rad(radian_converter, semantic30)
        else:
            save_positions = semantic30
    elif unit_l == "rad" and radian_converter is None:
        unit_l = "legacy255"
    return save_positions, space_l, unit_l


def load_pose(
    hand_type_or_joint: str,
    name: str,
    hand_side: Optional[str] = None,
) -> Optional[PoseEntry]:
    canonical = resolve_hand_config_key(hand_type_or_joint)
    n = (name or "").strip().lower()
    if not n:
        return None
    hand = HandConfig(canonical)
    side = normalize_hand_side(hand_side)
    data = _load_file_with_fallback(canonical, side)
    raw = (data.get("poses") or {}).get(n)
    if raw is None:
        return None
    return _parse_entry(n, raw, hand.num_joints)


def upsert_pose(
    hand_type_or_joint: str,
    name: str,
    positions: List[float],
    *,
    duration: float = _DEFAULT_DURATION,
    description: str = "",
    source: str = "teach",
    space: str = DEFAULT_SAVE_SPACE,
    unit: Optional[str] = None,
    hand_side: Optional[str] = None,
    keyframes: Optional[List[PoseKeyframe]] = None,
    replace_multi_keyframe: bool = False,
    expected_revision: Optional[str] = None,
) -> Optional[str]:
    """写入姿态。成功返回 None，失败返回错误字符串。

    传入 hand_side 时写入 {canonical}_{side}.json；
    若侧文件尚不存在，则以旧版无侧文件为种子（若有）。
    未传 hand_side 时写回旧版 {canonical}.json（兼容）。

    新建默认 ``space=semantic_v3``、``unit=rad``；显式传 ``hardware`` 仍可落盘机型序。
    多关键帧（``keyframes`` 非空）禁止 ``space=hardware``。

    未传 ``keyframes``（或空列表）时若库中已有多关键帧条目，默认拒绝覆盖以免误抹；
    显式 ``replace_multi_keyframe=True``（如 bundle overwrite）才允许整姿态替换为单点。
    """
    err = validate_pose_name(name)
    if err:
        return err
    space_l = (space or DEFAULT_SAVE_SPACE).strip().lower()
    if space_l not in _ALLOWED_SPACES:
        return f"space 无效: {space!r}（支持 hardware / semantic_v3 / semantic_o20）"
    kf_list = list(keyframes or [])
    if kf_list and space_l == "hardware":
        return "多关键帧姿态禁止 space=hardware，请使用 semantic_v3 + rad"
    canonical = resolve_hand_config_key(hand_type_or_joint)
    hand = HandConfig(canonical)
    if space_l == "semantic_v3":
        expect_n = SEMANTIC_DIM
    elif space_l == "semantic_o20":
        expect_n = 20
    else:
        expect_n = hand.num_joints
    unit_l = _resolve_unit_for_space(space_l, unit)
    if kf_list:
        seen: Set[str] = set()
        for kf in kf_list:
            kn = (kf.name or "").strip().lower()
            if not kn or not is_legal_keyframe_name(kn):
                return f"keyframe 名非法（禁止纯数字）: {kf.name!r}"
            if kn in seen:
                return f"keyframe 名重复: {kn}"
            if kn in _COMMAND_RESERVED:
                return f"keyframe 名保留字: {kn}"
            seen.add(kn)
            err = validate_positions(list(kf.positions), expect_n, unit=unit_l)
            if err:
                return f"keyframe '{kn}': {err}"
            if float(kf.duration) < _MIN_DURATION:
                return f"keyframe '{kn}' duration 过短（最小 {_MIN_DURATION}s）"
        positions = list(kf_list[-1].positions)
        duration = float(kf_list[-1].duration)
    err = validate_positions(positions, expect_n, unit=unit_l)
    if err:
        return err
    dur = float(duration)
    if dur < _MIN_DURATION:
        return f"duration 过短（最小 {_MIN_DURATION}s）"
    n = name.strip().lower()
    side = normalize_hand_side(hand_side)

    with _pose_file_lock(canonical, side):
        if side:
            side_path = _pose_path(canonical, side)
            if side_path.is_file():
                data = _load_file(canonical, side)
            else:
                data = _load_file(canonical, None)
        else:
            data = _load_file(canonical, None)

        poses: Dict[str, dict] = dict(data.get("poses") or {})
        existing_raw = poses.get(n) or {}
        if expected_revision is not None:
            from .workbench_support import pose_revision
            existing = _parse_entry(n, existing_raw, hand.num_joints) if existing_raw else None
            if pose_revision(existing) != expected_revision:
                return "revision_conflict: 姿态已存在或已被修改，请重新加载或另存为"
        existing_kfs = existing_raw.get("keyframes") or []
        if not kf_list and existing_kfs and not replace_multi_keyframe:
            return (
                f"姿态 '{n}' 已有 {len(existing_kfs)} 个 keyframe，"
                "整姿态保存会抹除多帧；请指定 keyframe 名"
                "（upsert_pose_keyframe / save_pose -p keyframe:=），"
                "或 replace_multi_keyframe=True"
            )
        record = {
            "space": space_l,
            "positions": [float(v) for v in positions],
            "num_joints": expect_n,
            "duration": dur,
            "description": description or "",
            "source": source or "teach",
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        jnames, named = positions_named_dict(list(positions), space=space_l, hand=hand)
        record["joint_names"] = jnames
        record["positions_named"] = named
        if space_l != "hardware":
            record["unit"] = unit_l
        if kf_list:
            record["keyframes"] = []
            for kf in kf_list:
                _, kf_named = positions_named_dict(list(kf.positions), space=space_l, hand=hand)
                record["keyframes"].append(
                    {
                        "name": kf.name.strip().lower(),
                        "positions": [float(v) for v in kf.positions],
                        "duration": float(kf.duration),
                        "positions_named": kf_named,
                    }
                )
        poses[n] = record
        data["poses"] = poses
        data["hand_joint"] = hand.hand_joint
        data["canonical_hand_type"] = canonical
        _save_file(canonical, data, side)
    return None


def upsert_pose_keyframe(
    hand_type_or_joint: str,
    name: str,
    keyframe: str,
    positions: List[float],
    *,
    duration: float = _DEFAULT_DURATION,
    description: str = "",
    source: str = "teach",
    space: str = DEFAULT_SAVE_SPACE,
    unit: Optional[str] = None,
    hand_side: Optional[str] = None,
) -> Optional[str]:
    """追加或替换某一关键帧；顶层 positions/duration 镜像末帧。

    多关键帧新建要求 ``semantic_v3``（推荐 ``unit=rad``）；禁止 hardware。
    若姿态尚不存在，则创建仅含该关键帧的条目。
    若已存在无 keyframes 的单点姿态，则先把原 positions 迁为 ``default`` 再写入新帧
    （当新帧名不是 default 时）；若新帧名就是 default，则直接替换单点。
    """
    kf_name = (keyframe or "").strip().lower()
    if not kf_name or not is_legal_keyframe_name(kf_name):
        return f"keyframe 名非法（禁止纯数字）: {keyframe!r}"
    if kf_name in _COMMAND_RESERVED:
        return f"keyframe 名保留字: {kf_name}"
    space_l = (space or DEFAULT_SAVE_SPACE).strip().lower()
    if space_l == "hardware":
        return "多关键帧姿态禁止 space=hardware，请使用 semantic_v3 + rad"

    existing = load_pose(hand_type_or_joint, name, hand_side=hand_side)
    new_kf = PoseKeyframe(
        name=kf_name,
        positions=[float(v) for v in positions],
        duration=float(duration),
    )
    if existing is None:
        return upsert_pose(
            hand_type_or_joint,
            name,
            positions,
            duration=duration,
            description=description,
            source=source,
            space=space_l,
            unit=unit,
            hand_side=hand_side,
            keyframes=[new_kf],
        )

    if existing.space == "hardware" or space_l != existing.space:
        # 已有 hardware 单点不可就地升级为多帧
        if existing.space == "hardware":
            return f"姿态 '{name}' 为 hardware 单点，无法追加 keyframe；请新建 semantic_v3 姿态"
        if space_l != existing.space:
            return f"姿态 '{name}' space={existing.space}，与请求 space={space_l} 不一致"

    unit_l = _resolve_unit_for_space(space_l, unit or existing.unit)
    if existing.unit != unit_l and existing.keyframes:
        return f"姿态 '{name}' unit={existing.unit}，与请求 unit={unit_l} 不一致"

    kfs: List[PoseKeyframe] = list(existing.keyframes)
    if not kfs:
        if kf_name == _DEFAULT_KEYFRAME_NAME:
            kfs = [new_kf]
        else:
            kfs = [
                PoseKeyframe(
                    name=_DEFAULT_KEYFRAME_NAME,
                    positions=list(existing.positions),
                    duration=float(existing.duration),
                ),
                new_kf,
            ]
    else:
        replaced = False
        for i, old in enumerate(kfs):
            if old.name == kf_name:
                kfs[i] = new_kf
                replaced = True
                break
        if not replaced:
            kfs.append(new_kf)

    return upsert_pose(
        hand_type_or_joint,
        name,
        list(kfs[-1].positions),
        duration=float(kfs[-1].duration),
        description=description or existing.description,
        source=source or existing.source,
        space=existing.space,
        unit=existing.unit if existing.keyframes else unit_l,
        hand_side=hand_side,
        keyframes=kfs,
    )


def commit_saved_pose(
    hand_type_or_joint: str,
    name: str,
    positions: List[float],
    *,
    duration: float = _DEFAULT_DURATION,
    description: str = "",
    source: str = "teach",
    space: str = DEFAULT_SAVE_SPACE,
    unit: Optional[str] = None,
    hand_side: Optional[str] = None,
    keyframe: Optional[str] = None,
    replace_multi_keyframe: bool = False,
) -> Optional[str]:
    """统一落盘入口：有 keyframe 名则 upsert_pose_keyframe，否则 upsert_pose。

    供 SavePose 服务 / CLI / Teach 共用，避免各路径分支漂移。
    """
    kf = (keyframe or "").strip().lower()
    if kf:
        return upsert_pose_keyframe(
            hand_type_or_joint,
            name,
            kf,
            positions,
            duration=duration,
            description=description,
            source=source,
            space=space,
            unit=unit,
            hand_side=hand_side,
        )
    return upsert_pose(
        hand_type_or_joint,
        name,
        positions,
        duration=duration,
        description=description,
        source=source,
        space=space,
        unit=unit,
        hand_side=hand_side,
        replace_multi_keyframe=replace_multi_keyframe,
    )


def delete_pose_keyframe(
    hand_type_or_joint: str,
    name: str,
    keyframe: str,
    *,
    hand_side: Optional[str] = None,
) -> Optional[str]:
    """删除姿态中某一关键帧。

    - 删后仍剩 ≥1 帧：重写 ``keyframes``，顶层镜像末帧。
    - 删后无帧：删除整个姿态。
    - 单点（无 keyframes 字段）：仅当 keyframe 为 ``default`` 时删除整姿态。

    成功返回 None；失败返回错误字符串。
    """
    kf_name = (keyframe or "").strip().lower()
    if not kf_name:
        return "keyframe 名为空"
    entry = load_pose(hand_type_or_joint, name, hand_side=hand_side)
    if entry is None:
        return f"姿态不存在: {name}"

    if not entry.keyframes:
        if kf_name != _DEFAULT_KEYFRAME_NAME:
            return f"单点姿态无关键帧 '{kf_name}'（仅有 default）"
        ok = delete_pose(hand_type_or_joint, name, hand_side=hand_side)
        return None if ok else f"删除失败: {name}"

    remaining = [kf for kf in entry.keyframes if kf.name != kf_name]
    if len(remaining) == len(entry.keyframes):
        return f"关键帧不存在: {kf_name}"
    if not remaining:
        ok = delete_pose(hand_type_or_joint, name, hand_side=hand_side)
        return None if ok else f"删除失败: {name}"

    last = remaining[-1]
    return upsert_pose(
        hand_type_or_joint,
        name,
        list(last.positions),
        duration=float(last.duration),
        description=entry.description,
        source=entry.source,
        space=entry.space,
        unit=entry.unit,
        hand_side=hand_side,
        keyframes=remaining,
        replace_multi_keyframe=True,
    )


def move_pose_keyframe(
    hand_type_or_joint: str,
    name: str,
    keyframe: str,
    *,
    delta: int,
    hand_side: Optional[str] = None,
) -> Optional[str]:
    """上下移动关键帧顺序。``delta=-1`` 上移，``+1`` 下移。成功返回 None。"""
    kf_name = (keyframe or "").strip().lower()
    if not kf_name:
        return "keyframe 名为空"
    try:
        step = int(delta)
    except (TypeError, ValueError):
        return "delta 无效"
    if step == 0:
        return None
    entry = load_pose(hand_type_or_joint, name, hand_side=hand_side)
    if entry is None:
        return f"姿态不存在: {name}"
    if not entry.keyframes:
        return "单点姿态无法调整关键帧顺序"
    kfs = list(entry.keyframes)
    idx = next((i for i, kf in enumerate(kfs) if kf.name == kf_name), -1)
    if idx < 0:
        return f"关键帧不存在: {kf_name}"
    new_idx = idx + step
    if new_idx < 0 or new_idx >= len(kfs):
        return "已到边界，无法继续移动"
    kfs[idx], kfs[new_idx] = kfs[new_idx], kfs[idx]
    last = kfs[-1]
    return upsert_pose(
        hand_type_or_joint,
        name,
        list(last.positions),
        duration=float(last.duration),
        description=entry.description,
        source=entry.source,
        space=entry.space,
        unit=entry.unit,
        hand_side=hand_side,
        keyframes=kfs,
        replace_multi_keyframe=True,
    )


def rename_pose_keyframe(
    hand_type_or_joint: str,
    name: str,
    keyframe: str,
    new_keyframe: str,
    *,
    hand_side: Optional[str] = None,
) -> Optional[str]:
    """重命名关键帧。成功返回 None。"""
    old = (keyframe or "").strip().lower()
    new = (new_keyframe or "").strip().lower()
    if not old:
        return "keyframe 名为空"
    if not new:
        return "新关键帧名为空"
    if old == new:
        return None
    entry = load_pose(hand_type_or_joint, name, hand_side=hand_side)
    if entry is None:
        return f"姿态不存在: {name}"
    if not entry.keyframes:
        return "单点姿态无法重命名关键帧"
    if any(kf.name == new for kf in entry.keyframes):
        return f"关键帧名已存在: {new}"
    kfs: List[PoseKeyframe] = []
    found = False
    for kf in entry.keyframes:
        if kf.name == old:
            found = True
            kfs.append(PoseKeyframe(new, list(kf.positions), float(kf.duration)))
        else:
            kfs.append(kf)
    if not found:
        return f"关键帧不存在: {old}"
    last = kfs[-1]
    return upsert_pose(
        hand_type_or_joint,
        name,
        list(last.positions),
        duration=float(last.duration),
        description=entry.description,
        source=entry.source,
        space=entry.space,
        unit=entry.unit,
        hand_side=hand_side,
        keyframes=kfs,
        replace_multi_keyframe=True,
    )


def delete_pose(
    hand_type_or_joint: str,
    name: str,
    hand_side: Optional[str] = None,
    *,
    expected_revision: Optional[str] = None,
) -> bool:
    """删除姿态。

    带 hand_side 且侧文件尚不存在时：若旧无侧文件含该名，则先种子复制到侧文件再删，
    避免 list 可见但 delete 失败；不直接改写共享的旧无侧文件。
    """
    canonical = resolve_hand_config_key(hand_type_or_joint)
    n = (name or "").strip().lower()
    side = normalize_hand_side(hand_side)

    with _pose_file_lock(canonical, side):
        if expected_revision is not None:
            from .workbench_support import pose_revision
            if pose_revision(load_pose(canonical, n, hand_side=side)) != expected_revision:
                raise ValueError("revision_conflict: 姿态已被修改，请重新加载")
        if side:
            side_path = _pose_path(canonical, side)
            if side_path.is_file():
                data = _load_file(canonical, side)
            else:
                legacy = _load_file(canonical, None)
                legacy_poses = dict(legacy.get("poses") or {})
                if n not in legacy_poses:
                    return False
                data = legacy
            poses = dict(data.get("poses") or {})
            if n not in poses:
                return False
            del poses[n]
            data["poses"] = poses
            data["hand_joint"] = HandConfig(canonical).hand_joint
            data["canonical_hand_type"] = canonical
            _save_file(canonical, data, side)
            return True

        data = _load_file(canonical, None)
        poses = dict(data.get("poses") or {})
        if n not in poses:
            return False
        del poses[n]
        data["poses"] = poses
        _save_file(canonical, data, None)
        return True
