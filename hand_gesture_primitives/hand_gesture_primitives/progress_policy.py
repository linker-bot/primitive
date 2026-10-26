"""内置原语 progress 细控策略（与 NamedPose keyframe progress 分离）。

仅纯定位 prep/close 允许 progress；力控/progressive close 与 auto/full 拒绝。
"""

from __future__ import annotations

from typing import Optional

# 纯定位、可接 progress 的 command_phased 原语
POSITION_PROGRESS_PRIMITIVES = frozenset({"tripod", "large_wrap"})

# close 相位含力控 / progressive，禁止 progress
FORCE_CLOSE_NO_PROGRESS = frozenset(
    {
        "tripod_by_force",
        "thumb_adduction_grip",
        "index_middle_adduction_grip",
        "hammer_grip",
    }
)


def validate_primitive_progress(
    cmd: str,
    phase: str,
    progress: Optional[float],
) -> Optional[str]:
    """若 progress 非法返回英文错误串；合法返回 None。"""
    if progress is None:
        return None
    name = (cmd or "").strip().lower()
    ph = (phase or "").strip().lower()
    if name == "tripod_by_force":
        return "tripod_by_force does not support progress (force close)"
    if ph in ("auto", "full"):
        return "progress requires phase prep|close (not auto/full)"
    if ph == "close" and name in FORCE_CLOSE_NO_PROGRESS:
        return f"{name} close rejects progress (force/progressive)"
    if name not in POSITION_PROGRESS_PRIMITIVES:
        return f"{name} does not support progress yet (use pose keyframes)"
    if ph not in ("prep", "close"):
        return f"invalid phase for progress: {phase!r}"
    return None
