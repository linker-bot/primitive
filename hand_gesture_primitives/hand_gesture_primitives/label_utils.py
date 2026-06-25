"""感知 label 归一化与 bbox 候选筛选。"""

from typing import Any, List, Optional, Sequence


def normalize_label(label: str) -> str:
    """小写、去首尾空白、去掉末尾标点。"""
    s = label.lower().strip()
    while s and s[-1] in '.,;:!?':
        s = s[:-1].strip()
    return s


def label_match_rank(target: str, candidate: str) -> int:
    """label 匹配优先级，越大越优先。target 为空时返回 0。"""
    if not target:
        return 0
    t = normalize_label(target)
    c = normalize_label(candidate)
    if not t or not c:
        return 0
    if t == c:
        return 3
    if t in c:
        return 2
    if c in t:
        return 1
    return -1


def labels_match(target: str, candidate: str) -> bool:
    return label_match_rank(target, candidate) > 0


def select_best_bbox(
    boxes: Sequence[Any],
    target_label: str = '',
    target_instance_id: Optional[int] = None,
) -> Optional[Any]:
    """按 instance_id / label 过滤后取最佳 bbox。"""
    if not boxes:
        return None

    candidates: List[Any] = list(boxes)
    if target_instance_id is not None and target_instance_id > 0:
        by_id = [b for b in candidates if b.instance_id == target_instance_id]
        if not by_id:
            return None
        candidates = by_id

    if target_label:
        matched = [b for b in candidates if labels_match(target_label, b.label)]
        if not matched:
            return None
        candidates = matched

    return max(
        candidates,
        key=lambda b: (label_match_rank(target_label, b.label), b.score),
    )
