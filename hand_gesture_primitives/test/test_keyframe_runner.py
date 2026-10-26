"""KeyframeRunner / resolve_t 单测。"""

from __future__ import annotations

from hand_gesture_primitives.keyframe_runner import (
    DriveMode,
    KeyframeRunner,
    KeyframeSegment,
    MotionProgram,
    resolve_t,
)


def _prog(n: int = 2) -> MotionProgram:
    segs = []
    for i in range(n):
        segs.append(
            KeyframeSegment(
                name=f"k{i}",
                target=tuple(float(i + 1) for _ in range(4)),
                duration=1.0,
            )
        )
    return MotionProgram(ref="pose/t", space="semantic_v3", unit="rad", segments=segs)


def test_resolve_t_override_priority() -> None:
    assert resolve_t(10.0, 1.0, 0.25) == 0.25
    assert resolve_t(0.5, 1.0, None) == 0.5
    assert resolve_t(5.0, 0.0, None) == 1.0
    assert resolve_t(0.0, 1.0, 1.5) == 1.0
    assert resolve_t(0.0, 1.0, -1.0) == 0.0


def test_progress_scrub_no_reanchor() -> None:
    runner = KeyframeRunner()
    start = [0.0, 0.0, 0.0, 0.0]
    runner.reset(_prog(1), start, drive=DriveMode.PROGRESS, progress=0.0)
    q0 = runner.compute(0.0)
    assert q0 == start
    runner.set_drive_progress(0.5)
    q1 = runner.compute(0.0)
    assert q1 == [0.5, 0.5, 0.5, 0.5]
    runner.set_drive_progress(0.25)
    q2 = runner.compute(0.0)
    # 同段不重锚：仍相对最初 start，不是相对 q1
    assert q2 == [0.25, 0.25, 0.25, 0.25]


def test_select_segment_reanchors() -> None:
    runner = KeyframeRunner()
    runner.reset(_prog(2), [0.0] * 4, drive=DriveMode.PROGRESS, progress=0.0)
    runner.select_segment("k1", [10.0] * 4, drive=DriveMode.PROGRESS, progress=0.5)
    q = runner.compute(0.0)
    # k1 target is all 2.0; from 10 toward 2 at 0.5 → 6
    assert q == [6.0, 6.0, 6.0, 6.0]
    assert runner.keyframe == "k1"


def test_auto_advances_segments() -> None:
    runner = KeyframeRunner()
    runner.reset(_prog(2), [0.0] * 4, drive=DriveMode.AUTO)
    q_mid = runner.compute(0.5)
    assert q_mid == [0.5, 0.5, 0.5, 0.5]
    assert runner.keyframe == "k0"
    q_next = runner.compute(1.5)
    # 大 dt：结转 0.5s 到 k1，从 1→2 的 50%
    assert runner.keyframe == "k1"
    assert q_next == [1.5, 1.5, 1.5, 1.5]
    q_end = runner.compute(2.0)
    assert q_end == [2.0, 2.0, 2.0, 2.0]
    assert runner.done


def test_auto_stop_after_segment() -> None:
    runner = KeyframeRunner()
    runner.reset(
        _prog(2),
        [0.0] * 4,
        drive=DriveMode.AUTO,
        segment_index=0,
        stop_after_segment=True,
    )
    q = runner.compute(1.0)
    assert q == [1.0, 1.0, 1.0, 1.0]
    assert runner.keyframe == "k0"
    assert runner.done
    # 不应前进到 k1
    q2 = runner.compute(2.0)
    assert runner.keyframe == "k0"
    assert q2 == [1.0, 1.0, 1.0, 1.0]


def test_progress_at_one_keeps_not_done() -> None:
    """PROGRESS 扫到 1.0 不得置 done，便于继续 scrub。"""
    runner = KeyframeRunner()
    runner.reset(_prog(1), [0.0] * 4, drive=DriveMode.PROGRESS, progress=0.0)
    runner.set_drive_progress(1.0)
    q = runner.compute(0.0)
    assert q == [1.0, 1.0, 1.0, 1.0]
    assert not runner.done
    runner.set_drive_progress(0.25)
    q2 = runner.compute(0.0)
    assert q2 == [0.25, 0.25, 0.25, 0.25]


def test_resolve_t_rejects_nan_override() -> None:
    assert resolve_t(0.5, 1.0, float("nan")) == 0.5
    assert resolve_t(0.25, 1.0, float("inf")) == 0.25
