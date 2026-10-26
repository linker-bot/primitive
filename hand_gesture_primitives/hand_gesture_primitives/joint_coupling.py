"""手型关节耦合：示教面板只读 / 强制同步。

L25：指尖 (tip) 与指中 (mid) 机械耦合；ros2_control 仅 16 主动关节，
不含 tip，独立拖 tip 无效。示教时 tip 只读并跟随 mid。
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

from .hand_config import HandConfig, resolve_hand_config_key


def tip_mid_couple_map(
    hand_type: str,
    joint_names: Optional[Sequence[str]] = None,
) -> Dict[int, int]:
    """返回 tip_index → mid_index。

    优先用 YAML ``from_o20`` 的 ``also``（mid→tip 复制）；
    再按关节名 ``*_tip`` / ``*_mid`` 对齐补齐。
    """
    key = resolve_hand_config_key(hand_type)
    hand = HandConfig(key)
    names = list(joint_names) if joint_names else list(hand.joint_names)
    couple: Dict[int, int] = {}

    # from_o20: also 目标视为只读，源为 to（通常是 mid）
    for _o20_idx, spec in getattr(hand, "_mapping", {}).items():
        mid = int(spec.get("to", -1))
        for tip in spec.get("also") or []:
            tip_i = int(tip)
            if 0 <= tip_i < hand.num_joints and 0 <= mid < hand.num_joints:
                couple[tip_i] = mid

    # 名面对齐：thumb_tip ← thumb_mid（仅 L25/L20 系）
    if key in ("l25",) or key.startswith("l2"):
        by_name = {n: i for i, n in enumerate(names)}
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            tip_n = f"{finger}_tip"
            mid_n = f"{finger}_mid"
            if tip_n in by_name and mid_n in by_name:
                couple[by_name[tip_n]] = by_name[mid_n]

    return couple


def couple_meta(
    hand_type: str,
    joint_names: Optional[Sequence[str]] = None,
) -> List[dict]:
    """供 snapshot / 前端：只读 tip 的元数据列表。"""
    key = resolve_hand_config_key(hand_type)
    hand = HandConfig(key)
    names = list(joint_names) if joint_names else list(hand.joint_names)
    cmap = tip_mid_couple_map(key, names)
    out: List[dict] = []
    for tip, mid in sorted(cmap.items()):
        out.append(
            {
                "index": tip,
                "source_index": mid,
                "name": names[tip] if tip < len(names) else f"j{tip}",
                "source_name": names[mid] if mid < len(names) else f"j{mid}",
                "label": "耦合/只读·随指中",
                "reason": "l25_tip_follows_mid",
            }
        )
    return out


def apply_tip_mid_coupling(
    positions: Sequence[float],
    couple_map: Mapping[int, int],
) -> List[float]:
    """把 tip 值强制写成对应 mid（下发与保存前调用）。"""
    out = [float(v) for v in positions]
    for tip, mid in couple_map.items():
        if 0 <= tip < len(out) and 0 <= mid < len(out):
            out[tip] = out[mid]
    return out


def coupling_hint(hand_type: str) -> str:
    key = resolve_hand_config_key(hand_type)
    if key == "l25":
        return "L25：指尖与指中机械耦合（ros2_control 不可独立控 tip）；指尖滑条只读，随指中变化"
    return ""
