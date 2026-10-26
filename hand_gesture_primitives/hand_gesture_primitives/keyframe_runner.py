"""MotionProgram / KeyframeRunner — 位置关键帧插值与 progress 细控。

与 YAML ``gestures.*.stages``（prep/close **时长**）无关；本模块仅服务
Pose Library ``keyframes[]`` 与 ``NamedPose`` 回放。

锚定规则：
- 进入新 segment / 新 program → ``q_enter = current``
- 同段仅更新 progress → **不**重锚
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from .primitive_base import lerp_angles


class DriveMode(str, Enum):
    AUTO = "auto"
    PROGRESS = "progress"


def resolve_t(
    elapsed: float,
    duration: float,
    progress_override: Optional[float] = None,
) -> float:
    """progress_override 优先；否则 elapsed/duration；结果 clamp 到 [0, 1]。

    非有限 progress_override（NaN/Inf）视为未提供，回退到墙钟。
    """
    if progress_override is not None:
        p = float(progress_override)
        if p == p and abs(p) != float("inf"):  # isfinite without importing math
            return max(0.0, min(1.0, p))
    if duration <= 0.0:
        return 1.0
    return max(0.0, min(1.0, float(elapsed) / float(duration)))


@dataclass(frozen=True)
class KeyframeSegment:
    name: str
    target: Tuple[float, ...]
    duration: float


@dataclass
class MotionProgram:
    ref: str
    space: str
    unit: str
    segments: List[KeyframeSegment] = field(default_factory=list)

    def segment_names(self) -> List[str]:
        return [s.name for s in self.segments]

    def find_segment_index(self, name: str) -> int:
        key = (name or "").strip().lower()
        for i, seg in enumerate(self.segments):
            if seg.name == key:
                return i
        raise KeyError(key)


class KeyframeRunner:
    """唯一位置关键帧插值实现（auto / progress）。"""

    def __init__(self) -> None:
        self._program: Optional[MotionProgram] = None
        self._index: int = 0
        self._q_enter: List[float] = []
        self._drive: DriveMode = DriveMode.AUTO
        self._progress: float = 0.0
        self._segment_t0: float = 0.0
        self._done: bool = False
        self._last_t: float = 0.0
        self._stop_after_segment: bool = False

    def reset(
        self,
        program: MotionProgram,
        start_q: Sequence[float],
        *,
        drive: DriveMode = DriveMode.AUTO,
        segment_index: int = 0,
        progress: float = 0.0,
        elapsed: float = 0.0,
        stop_after_segment: bool = False,
    ) -> None:
        if not program.segments:
            raise ValueError("MotionProgram.segments 为空")
        if segment_index < 0 or segment_index >= len(program.segments):
            raise IndexError(f"segment_index 越界: {segment_index}")
        self._program = program
        self._index = int(segment_index)
        self._q_enter = [float(v) for v in start_q]
        self._drive = DriveMode(drive)
        self._progress = max(0.0, min(1.0, float(progress)))
        self._segment_t0 = float(elapsed)
        self._done = False
        self._last_t = self._progress if self._drive == DriveMode.PROGRESS else 0.0
        self._stop_after_segment = bool(stop_after_segment)
        self._pad_enter_to_target()

    def _pad_enter_to_target(self) -> None:
        assert self._program is not None
        n = len(self._program.segments[self._index].target)
        while len(self._q_enter) < n:
            self._q_enter.append(0.0)
        self._q_enter = self._q_enter[:n]

    def select_segment(
        self,
        name: str,
        current_q: Sequence[float],
        *,
        elapsed: float = 0.0,
        drive: Optional[DriveMode] = None,
        progress: Optional[float] = None,
    ) -> None:
        if self._program is None:
            raise RuntimeError("KeyframeRunner 未 reset")
        idx = self._program.find_segment_index(name)
        self._index = idx
        self._q_enter = [float(v) for v in current_q]
        self._pad_enter_to_target()
        self._segment_t0 = float(elapsed)
        self._done = False
        if drive is not None:
            self._drive = DriveMode(drive)
        if progress is not None:
            self._progress = max(0.0, min(1.0, float(progress)))
        elif self._drive == DriveMode.PROGRESS:
            self._progress = 0.0

    def set_drive_progress(self, p: float) -> None:
        """同段热更新 progress，不重锚。非有限值忽略。"""
        value = float(p)
        if value != value or abs(value) == float("inf"):
            return
        self._drive = DriveMode.PROGRESS
        self._progress = max(0.0, min(1.0, value))
        # PROGRESS 细控永不因 p=1 标记完成，便于 scrub 回拉
        self._done = False

    def set_drive_auto(self, *, elapsed: float) -> None:
        """切回 AUTO：以当前几何进度反推段内起点，避免跳变（尽力）。"""
        t = self.progress
        seg = self._current_segment()
        # 将当前 t 视为已走过；之后用 elapsed 继续
        self._drive = DriveMode.AUTO
        self._segment_t0 = float(elapsed) - t * max(seg.duration, 1e-6)
        self._done = False

    def _current_segment(self) -> KeyframeSegment:
        assert self._program is not None
        return self._program.segments[self._index]

    def compute(self, elapsed: float) -> List[float]:
        if self._program is None:
            raise RuntimeError("KeyframeRunner 未 reset")

        while True:
            seg = self._current_segment()
            local = float(elapsed) - self._segment_t0
            override = self._progress if self._drive == DriveMode.PROGRESS else None
            t = resolve_t(local, seg.duration, override)
            self._last_t = t
            q = lerp_angles(self._q_enter, list(seg.target), t)

            if self._drive == DriveMode.PROGRESS:
                # 外部 scrub：保持 active，不置 done（避免 Executor 卸载后无法再改 p）
                self._done = False
                return q

            # AUTO：段完成则前进；末段 / stop_after_segment 则 hold+done
            if t < 1.0:
                self._done = False
                return q
            if self._stop_after_segment or self._index >= len(self._program.segments) - 1:
                self._done = True
                return list(seg.target)
            # 换段重锚：以上一目标为进入角；结转超额时间，避免大 dt 丢进度
            overshoot = max(0.0, local - float(seg.duration))
            self._index += 1
            self._q_enter = list(seg.target)
            self._pad_enter_to_target()
            self._segment_t0 = float(elapsed) - overshoot

    @property
    def drive_mode(self) -> str:
        return self._drive.value

    @property
    def keyframe(self) -> str:
        if self._program is None:
            return ""
        return self._current_segment().name

    @property
    def progress(self) -> float:
        if self._drive == DriveMode.PROGRESS:
            return self._progress
        return float(self._last_t)

    def progress_at(self, elapsed: float) -> float:
        if self._program is None:
            return 0.0
        seg = self._current_segment()
        local = float(elapsed) - self._segment_t0
        override = self._progress if self._drive == DriveMode.PROGRESS else None
        return resolve_t(local, seg.duration, override)

    @property
    def done(self) -> bool:
        return self._done

    @property
    def program(self) -> Optional[MotionProgram]:
        return self._program
