"""URDF 路径工具：将相对 mesh 路径转为 file:// 绝对路径供 RViz 加载。"""

import os
import re
from typing import List, Set


def _default_urdf_search_dirs() -> List[str]:
    """Common clone locations for linkerhand-urdf (newest / env overrides first)."""
    dirs: List[str] = []
    env_root = os.environ.get("LINKERHAND_URDF_DIR", "").strip()
    if env_root:
        dirs.append(os.path.expanduser(env_root))

    home = os.path.expanduser("~")
    for ws in ("ros_ws", "gcl_ws", "gitea_ws"):
        dirs.extend([
            os.path.join(home, ws, "src", "linkerhand-urdf"),
            os.path.join(home, ws, "online", "src", "linkerhand-urdf"),
        ])
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
}


def hand_joint_to_urdf_model(hand_joint: str) -> str:
    """hand_joint 参数 (O20/L25/...) → URDF 目录名 (o20/l25/...)。"""
    return _MODEL_ALIASES.get(hand_joint.upper(), hand_joint.lower())


def resolve_urdf_path(hand_model: str, hand_side: str, urdf_path: str = "") -> str:
    """查找手部 URDF 文件路径。"""
    if urdf_path and os.path.isfile(urdf_path):
        return os.path.abspath(urdf_path)

    model = _MODEL_ALIASES.get(hand_model.upper(), hand_model.lower())
    side = hand_side.lower()
    filename = f"linkerhand_{model}_{side}.urdf"

    for base in _URDF_SEARCH_DIRS:
        path = os.path.join(base, model, side, filename)
        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"找不到 URDF: model={hand_model}, side={hand_side}, "
        f"expected linkerhand_{model.lower()}_{side.lower()}.urdf under "
        f"<linkerhand-urdf>/{model.lower()}/{side.lower()}/. "
        f"已搜索: {_URDF_SEARCH_DIRS}. "
        f"请设置 launch 参数 urdf_path:=/abs/path/to.urdf "
        f"或环境变量 LINKERHAND_URDF_DIR=/path/to/linkerhand-urdf "
        f"（常见 clone 路径: ~/ros_ws/src/linkerhand-urdf）"
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
    if "hand_base_link" in all_links:
        return "hand_base_link"
    if "hand_link" in all_links:
        return "hand_link"
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
