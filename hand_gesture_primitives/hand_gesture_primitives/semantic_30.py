"""semantic_30 协议：30 通道名称表、O20↔S30 投影、SemanticHand30 视图。"""

from __future__ import annotations

from typing import List, Mapping, Sequence

SEMANTIC_30_NAMES: List[str] = [
    # Block A — 手指 25 (5 指 × 5 槽)
    # thumb (finger_index=0)
    "thumb_mcp_flex",  # 0
    "thumb_mcp_abd",  # 1
    "thumb_pip",  # 2
    "thumb_dip",  # 3
    "thumb_rot",  # 4
    # index (finger_index=1)
    "index_mcp_flex",  # 5
    "index_mcp_abd",  # 6
    "index_pip",  # 7
    "index_dip",  # 8
    "index_extra",  # 9
    # middle (finger_index=2)
    "middle_mcp_flex",  # 10
    "middle_mcp_abd",  # 11
    "middle_pip",  # 12
    "middle_dip",  # 13
    "middle_extra",  # 14
    # ring (finger_index=3)
    "ring_mcp_flex",  # 15
    "ring_mcp_abd",  # 16
    "ring_pip",  # 17
    "ring_dip",  # 18
    "ring_extra",  # 19
    # pinky (finger_index=4)
    "pinky_mcp_flex",  # 20
    "pinky_mcp_abd",  # 21
    "pinky_pip",  # 22
    "pinky_dip",  # 23
    "pinky_extra",  # 24
    # Block B — 腕与掌 5
    "wrist_flex",  # 25
    "wrist_deviation",  # 26
    "wrist_pronation",  # 27
    "palm_arch",  # 28
    "reserved",  # 29
]

SEMANTIC_30_IDX: dict[str, int] = {name: i for i, name in enumerate(SEMANTIC_30_NAMES)}

SEMANTIC_DIM = 30

# O20 → S30 投影映射: o20_idx → s30_idx
_O20_TO_S30: List[int] = [
    0,  # o20[0]  thumb_base   → s30[0]  thumb_mcp_flex
    5,  # o20[1]  index_base   → s30[5]  index_mcp_flex
    10,  # o20[2]  middle_base  → s30[10] middle_mcp_flex
    15,  # o20[3]  ring_base    → s30[15] ring_mcp_flex
    20,  # o20[4]  pinky_base   → s30[20] pinky_mcp_flex
    1,  # o20[5]  thumb_abd    → s30[1]  thumb_mcp_abd
    6,  # o20[6]  index_abd    → s30[6]  index_mcp_abd
    11,  # o20[7]  middle_abd   → s30[11] middle_mcp_abd
    16,  # o20[8]  ring_abd     → s30[16] ring_mcp_abd
    21,  # o20[9]  pinky_abd    → s30[21] pinky_mcp_abd
    4,  # o20[10] thumb_rot    → s30[4]  thumb_rot
    7,  # o20[11] rsv_11       → s30[7]  index_pip
    12,  # o20[12] rsv_12       → s30[12] middle_pip
    17,  # o20[13] rsv_13       → s30[17] ring_pip
    22,  # o20[14] rsv_14       → s30[22] pinky_pip
    3,  # o20[15] thumb_tip    → s30[3]  thumb_dip
    8,  # o20[16] index_tip    → s30[8]  index_dip
    13,  # o20[17] middle_tip   → s30[13] middle_dip
    18,  # o20[18] ring_tip     → s30[18] ring_dip
    23,  # o20[19] pinky_tip    → s30[23] pinky_dip
]

# S30 → O20 反向映射: s30_idx → o20_idx (仅有映射的通道)
_S30_TO_O20: dict[int, int] = {s30: o20 for o20, s30 in enumerate(_O20_TO_S30)}

# S30 中没有 O20 来源的通道（填 0 + mask）
_S30_UNMAPPED: frozenset[int] = frozenset(i for i in range(SEMANTIC_DIM) if i not in _S30_TO_O20)

# S30 中的 abd 通道索引（中立值 128）
ABD_INDICES_S30: List[int] = [1, 6, 11, 16, 21]


def o20_joint_index_to_s30(o20_idx: int) -> int:
    """将 O20 关节索引转为对应的 S30 索引。超界返回 -1。"""
    if 0 <= o20_idx < 20:
        return _O20_TO_S30[o20_idx]
    return -1


def s30_joint_index_to_o20(s30_idx: int) -> int:
    """将 S30 通道索引转为 O20；无映射返回 -1。"""
    return int(_S30_TO_O20.get(s30_idx, -1))


def project_o20_to_s30(vec20: Sequence[float]) -> List[float]:
    """O20 20 维 → S30 30 维投影（无损：O20 全部信息保留）。"""
    if len(vec20) < 20:
        raise ValueError(f"project_o20_to_s30: 需要 20 维输入，收到 {len(vec20)}")
    out = [0.0] * SEMANTIC_DIM
    for o20_idx, s30_idx in enumerate(_O20_TO_S30):
        out[s30_idx] = float(vec20[o20_idx])
    return out


def project_s30_to_o20(vec30: Sequence[float]) -> List[float]:
    """S30 30 维 → O20 20 维反向投影（lossy：Block B / extra / thumb_pip 丢弃）。"""
    if len(vec30) < 30:
        raise ValueError(f"project_s30_to_o20: 需要 30 维输入，收到 {len(vec30)}")
    out = [0.0] * 20
    for o20_idx, s30_idx in enumerate(_O20_TO_S30):
        out[o20_idx] = float(vec30[s30_idx])
    return out


def as_semantic_30(vec: Sequence[float]) -> List[float]:
    """将 O20（20）或已是 S30（≥30）的向量规范为 30 维。"""
    if len(vec) >= SEMANTIC_DIM:
        return [float(v) for v in vec[:SEMANTIC_DIM]]
    if len(vec) >= 20:
        return project_o20_to_s30(vec)
    raise ValueError(f"as_semantic_30: 需要 ≥20 维，收到 {len(vec)}")


class _FingerView:
    """单指 5 通道视图。"""

    __slots__ = ("_data", "_offset")

    def __init__(self, data: List[float], finger_index: int) -> None:
        self._data = data
        self._offset = finger_index * 5

    @property
    def mcp_flex(self) -> float:
        return self._data[self._offset]

    @mcp_flex.setter
    def mcp_flex(self, v: float) -> None:
        self._data[self._offset] = v

    @property
    def mcp_abd(self) -> float:
        return self._data[self._offset + 1]

    @mcp_abd.setter
    def mcp_abd(self, v: float) -> None:
        self._data[self._offset + 1] = v

    @property
    def pip(self) -> float:
        return self._data[self._offset + 2]

    @pip.setter
    def pip(self, v: float) -> None:
        self._data[self._offset + 2] = v

    @property
    def dip(self) -> float:
        return self._data[self._offset + 3]

    @dip.setter
    def dip(self, v: float) -> None:
        self._data[self._offset + 3] = v

    @property
    def extra(self) -> float:
        return self._data[self._offset + 4]

    @extra.setter
    def extra(self, v: float) -> None:
        self._data[self._offset + 4] = v


class _ThumbView(_FingerView):
    """拇指视图：offset 4 语义为 rot 而非 extra。"""

    @property
    def rot(self) -> float:
        return self._data[self._offset + 4]

    @rot.setter
    def rot(self, v: float) -> None:
        self._data[self._offset + 4] = v


class SemanticHand30:
    """30 维手势语义向量的命名视图。"""

    __slots__ = ("_data",)

    def __init__(self, data: list[float] | None = None) -> None:
        if data is None:
            self._data = [0.0] * SEMANTIC_DIM
        else:
            if len(data) != SEMANTIC_DIM:
                raise ValueError(f"SemanticHand30: 需要 {SEMANTIC_DIM} 维，收到 {len(data)}")
            self._data = list(data)

    @classmethod
    def from_vector(cls, vec: Sequence[float]) -> SemanticHand30:
        return cls(list(vec))

    def as_vector(self) -> List[float]:
        return list(self._data)

    @property
    def thumb(self) -> _ThumbView:
        return _ThumbView(self._data, 0)

    @property
    def index(self) -> _FingerView:
        return _FingerView(self._data, 1)

    @property
    def middle(self) -> _FingerView:
        return _FingerView(self._data, 2)

    @property
    def ring(self) -> _FingerView:
        return _FingerView(self._data, 3)

    @property
    def pinky(self) -> _FingerView:
        return _FingerView(self._data, 4)

    def finger(self, name: str) -> _FingerView:
        idx = {"thumb": 0, "index": 1, "middle": 2, "ring": 3, "pinky": 4}.get(name)
        if idx is None:
            raise KeyError(f"未知手指名: {name!r}")
        if idx == 0:
            return _ThumbView(self._data, 0)
        return _FingerView(self._data, idx)

    def __getitem__(self, idx: int) -> float:
        return self._data[idx]

    def __setitem__(self, idx: int, val: float) -> None:
        self._data[idx] = val

    def __len__(self) -> int:
        return SEMANTIC_DIM


def s30_pose_from_channels(
    channels: Mapping[str, float],
    *,
    abd_neutral: bool = True,
) -> List[float]:
    """稀疏通道名 → 完整 30 维向量。

    ``abd_neutral=True`` 时先把五指 abd 置 128，再用 ``channels`` 覆盖。
    """
    abd_neutral_val = 128.0
    hand = SemanticHand30()
    if abd_neutral:
        for idx in ABD_INDICES_S30:
            hand[idx] = abd_neutral_val
    for name, value in channels.items():
        if name not in SEMANTIC_30_IDX:
            raise KeyError(f"未知 semantic_30 通道: {name}")
        hand[SEMANTIC_30_IDX[name]] = float(value)
    out = hand.as_vector()
    assert len(out) == SEMANTIC_DIM
    return out


def l25_compat_remap_s30(s30: Sequence[float]) -> List[float]:
    """旧 L25 字面量表经 blanket invert 的兼容投影 → S30。

    输入为「旧常量语义」（与历史 O20 裸表数值相同），输出为 L25 硬件路径
    实际使用的 S30 目标。
    """
    from .hand_config import l25_legacy_angles_to_o20

    o20_style = project_s30_to_o20(s30)
    return project_o20_to_s30(l25_legacy_angles_to_o20(o20_style))
