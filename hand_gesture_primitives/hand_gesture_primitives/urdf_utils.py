"""URDF 路径与 mesh 工具（FK / RViz 共用）。

职责:
  - 搜索官方手 URDF，顺序为 `data/urdf`（本仓约定，使用方自备）→
    `LINKERHAND_URDF_DIR` → 安装 share/hand_gesture_primitives/urdf → 外仓 →
    `/opt/linkerhand/urdf`
  - 兼容版本化文件名（如 `linkerhand_l6v3.1_left.urdf`）与稳定别名
    （`linkerhand_l6_left.urdf`）
  - 将相对 mesh 路径改写为 `file://` 绝对路径，供 RViz RobotModel 加载

本模块解析的是含 mesh 的官方模型，供 `view_hand_urdf` 与 Pinocchio FK 使用。
未找到时 `create_fk_solver` 返回 None：`executor` 自动停发指尖话题，
而 `joint_state_bridge` 会直接报错退出（该节点职责即 FK 驱动 mesh 模型）。
"""

from __future__ import annotations

import glob
import os
import re
import sys
from typing import Iterable, List, Optional, Set


def _repo_root() -> str:
    """向上定位源码仓库根（兼容 .git 目录与 worktree 的 .git 文件）。

    **限制**：靠 ``.git`` 识别仓库根，因此只在源码树内有效。经 colcon 安装后
    install space 无 ``.git``，回退的 ``data/urdf`` 通常不存在；此时由
    ``_default_urdf_search_dirs`` 的 ament / sys.prefix share 路径找到随包模型。
    外部模型仍可通过 ``LINKERHAND_URDF_DIR`` 或节点 ``urdf_path`` 指定。
    """
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.exists(os.path.join(d, ".git")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _repo_robot_model_dirs() -> List[str]:
    """本仓手 URDF 根。

    优先使用 ``data/urdf/``，并保留外部集成的模型布局兼容搜索。
    随包模型和外部模型配置见仓库根目录 data/urdf/README.md。
    """
    dirs: List[str] = [os.path.join(_repo_root(), "data", "urdf")]

    # 向上查找兼容的外部模型布局。
    here = os.path.abspath(os.path.dirname(__file__))
    cur = here
    for _ in range(10):
        evo = os.path.join(cur, "src", "robot_model", "evo_work")
        if os.path.isdir(evo):
            dirs.append(evo)
            break
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent

    return dirs


def _default_urdf_search_dirs() -> List[str]:
    """本仓 data/urdf → 显式环境目录 → 安装 share → 外仓 / CAD 仓 → /opt。"""
    dirs: List[str] = []
    dirs.extend(_repo_robot_model_dirs())

    env_root = os.environ.get("LINKERHAND_URDF_DIR", "").strip()
    if env_root:
        dirs.append(os.path.expanduser(env_root))

    # Explicit external configuration still precedes the installed defaults.
    # ament supports colcon overlays; sys.prefix supports ordinary pip/venv installs.
    try:
        from ament_index_python.packages import get_package_share_directory

        dirs.append(os.path.join(get_package_share_directory("hand_gesture_primitives"), "urdf"))
    except Exception:
        pass
    dirs.append(os.path.join(sys.prefix, "share", "hand_gesture_primitives", "urdf"))

    home = os.path.expanduser("~")
    for ws in ("gcl_ws", "gitea_ws", "ros_ws", "cad_ws"):
        dirs.extend(
            [
                os.path.join(home, ws, "src", "linkerhand-urdf"),
                os.path.join(home, ws, "src", "linkerhand", "linkerhand-urdf"),
                os.path.join(home, ws, "linkerhand", "linkerhand-urdf"),
                os.path.join(home, ws, "online", "src", "linkerhand-urdf"),
            ]
        )
    dirs.append("/opt/linkerhand/urdf")

    seen = set()
    unique: List[str] = []
    for d in dirs:
        norm = os.path.normpath(d)
        if norm not in seen:
            seen.add(norm)
            unique.append(norm)
    return unique


_URDF_SEARCH_DIRS = _default_urdf_search_dirs()

# 可视化 / FK 共用的 URDF 型号映射
_MODEL_ALIASES = {
    "O20": "o20",
    "L20": "l20",
    "L25": "l25",
    "O6": "o6",
    "L6": "l6",
    "O30": "o30",
    # 整机 G20 无独立 mesh；FK / RViz 复用 L25 URDF
    "G20": "l25",
    # YAML fk.urdf_model / assets 目录名
    "LINKERHAND_L6": "l6",
    "LINKERHAND_G20": "l25",
}


def _normalize_model(hand_model: str) -> str:
    model = _MODEL_ALIASES.get(hand_model.upper(), hand_model.lower())
    if model.startswith("linkerhand_"):
        model = model[len("linkerhand_") :]
    return model


def hand_joint_to_urdf_model(hand_joint: str) -> str:
    """hand_joint 参数 (O20/L25/...) → URDF 目录名 (o20/l25/...)。"""
    return _normalize_model(hand_joint)


def _version_key(path: str) -> tuple:
    """从文件名提取版本排序键；无版本视为 (0,)。

    例: linkerhand_l6v3.1_left.urdf → (3, 1)
        linkerhand_l6_left.urdf     → (0,)
    """
    name = os.path.basename(path)
    m = re.search(r"v(\d+(?:\.\d+)*)", name, flags=re.IGNORECASE)
    if not m:
        return (0,)
    parts = []
    for p in m.group(1).split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(0)
    return tuple(parts) if parts else (0,)


def _pick_best_urdf(candidates: Iterable[str], model: str, side: str) -> Optional[str]:
    """在候选中选最优：精确名 > 更高版本号 > 字典序。"""
    files = [os.path.abspath(p) for p in candidates if os.path.isfile(p)]
    if not files:
        return None
    exact = f"linkerhand_{model}_{side}.urdf"
    for p in files:
        if os.path.basename(p) == exact:
            return p
    # 版本化优先取最新（如 l6v3.1 > l6v2）
    files.sort(key=lambda p: (_version_key(p), os.path.basename(p)))
    return files[-1]


def _find_urdf_in_dir(side_dir: str, model: str, side: str) -> Optional[str]:
    """在 {model}/{side}/ 目录下查找 URDF。

    兼容:
      linkerhand_l6_left.urdf
      linkerhand_l6v3.1_left.urdf
      linkerhand_o6_left.urdf
    """
    if not os.path.isdir(side_dir):
        return None

    exact = os.path.join(side_dir, f"linkerhand_{model}_{side}.urdf")
    if os.path.isfile(exact):
        return exact

    patterns = [
        os.path.join(side_dir, f"linkerhand_{model}v*_{side}.urdf"),
        os.path.join(side_dir, f"linkerhand_{model}*_{side}.urdf"),
        os.path.join(side_dir, f"*_{side}.urdf"),
    ]
    found: List[str] = []
    model_l = model.lower()
    for pat in patterns:
        for path in glob.glob(pat):
            base = os.path.basename(path).lower()
            compact = re.sub(r"[^a-z0-9]", "", base)
            # 要求文件名含型号 token（l6 / o6 / l25 ...）
            if model_l not in compact:
                continue
            if not (base.startswith("linkerhand_") or model_l in base):
                continue
            found.append(path)
    return _pick_best_urdf(found, model, side)


def resolve_urdf_path(hand_model: str, hand_side: str, urdf_path: str = "") -> str:
    """查找手部 URDF 文件路径。"""
    if urdf_path and os.path.isfile(urdf_path):
        return os.path.abspath(urdf_path)

    model = _normalize_model(hand_model)
    side = hand_side.lower()

    for base in _URDF_SEARCH_DIRS:
        hit = _find_urdf_in_dir(os.path.join(base, model, side), model, side)
        if hit:
            return hit

    # ament_index: o6_left_description / l6_right_description 等
    try:
        from ament_index_python.packages import get_package_share_directory

        pkg = f"{model}_{side}_description"
        pkg_dir = get_package_share_directory(pkg)
        hit = _find_urdf_in_dir(pkg_dir, model, side)
        if hit:
            return hit
        # 部分包把 urdf 放在子目录
        for sub in ("urdf", "robots", side, os.path.join(model, side)):
            hit = _find_urdf_in_dir(os.path.join(pkg_dir, sub), model, side)
            if hit:
                return hit
    except Exception:
        pass

    # ~/assets/components/hands/linkerhand_{model}/variants/{side}/hand.urdf
    assets_hands = os.path.join(os.path.expanduser("~"), "assets", "components", "hands")
    if os.path.isdir(assets_hands):
        target_names = {f"linkerhand_{model}".lower(), model.lower()}
        for hand_name in sorted(os.listdir(assets_hands)):
            hand_dir = os.path.join(assets_hands, hand_name)
            name_l = hand_name.lower()
            matched = name_l in target_names or model.lower() in re.sub(r"[^a-z0-9]", "", name_l)
            if not matched:
                meta = os.path.join(hand_dir, "meta.yaml")
                if os.path.isfile(meta):
                    try:
                        import yaml as _yaml

                        with open(meta, "r", encoding="utf-8") as f:
                            m = _yaml.safe_load(f) or {}
                        meta_name = str(m.get("name", "")).lower()
                        matched = meta_name in target_names or model.lower() in meta_name
                    except Exception:
                        matched = False
            if not matched:
                continue
            variant = os.path.join(hand_dir, "variants", side, "hand.urdf")
            if os.path.isfile(variant):
                return variant
            # 兼容 variants/{side}/*.urdf
            hit = _find_urdf_in_dir(os.path.join(hand_dir, "variants", side), model, side)
            if hit:
                return hit

    raise FileNotFoundError(
        f"找不到 URDF: model={hand_model}, side={hand_side}, "
        f"expected linkerhand_{model}_{side}.urdf "
        f"(或版本化名如 linkerhand_{model}v3.1_{side}.urdf) under "
        f"src/robot_model/evo_work/{{model}}/{{side}}/ "
        f"或 <linkerhand-urdf>/{{model}}/{{side}}/. "
        f"已搜索: {_URDF_SEARCH_DIRS}. "
        f"请设置 launch 参数 urdf_path:=/abs/path/to.urdf "
        f"或环境变量 LINKERHAND_URDF_DIR=/path/to/linkerhand-urdf"
    )


def get_root_link_name(urdf_path: str) -> str:
    """解析 URDF 根 link 名 (无 parent joint 的 link)。"""
    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()

    child_links: Set[str] = set(re.findall(r'<child\s+link="([^"]+)"', content))
    all_links: Set[str] = set(re.findall(r'<link\s+name="([^"]+)"', content))
    roots = all_links - child_links
    if len(roots) == 1:
        return roots.pop()
    for preferred in (
        "lh_hand_base_link",
        "rh_hand_base_link",
        "hand_base_link",
        "hand_link",
    ):
        if preferred in all_links:
            return preferred
    return sorted(all_links)[0] if all_links else "base_link"


def urdf_with_absolute_meshes(urdf_path: str) -> str:
    """读取 URDF，把相对 mesh 路径替换为 file:// 绝对路径。"""
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))

    with open(urdf_path, "r", encoding="utf-8") as f:
        content = f.read()

    def _replace(match: re.Match) -> str:
        rel = match.group(1)
        if rel.startswith(("file://", "package://", "http://", "https://")):
            return match.group(0)
        abs_path = os.path.normpath(os.path.join(urdf_dir, rel))
        return f'filename="file://{abs_path}"'

    return re.sub(r'filename="([^"]+)"', _replace, content)


def list_search_dirs() -> List[str]:
    return list(_URDF_SEARCH_DIRS)


def refresh_search_dirs() -> List[str]:
    """Re-read env / defaults (tests or runtime config)."""
    global _URDF_SEARCH_DIRS
    _URDF_SEARCH_DIRS = _default_urdf_search_dirs()
    return list(_URDF_SEARCH_DIRS)
