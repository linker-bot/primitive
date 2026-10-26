"""run_gesture_cmd_suite 手型裁剪单元测试。"""

from __future__ import annotations

from hand_gesture_primitives.run_gesture_cmd_suite import (
    DEFAULT_SUITE,
    SuiteStep,
    _parse_suite,
    command_base,
    partition_by_hand_support,
)


def test_command_base_strips_stage() -> None:
    assert command_base("thumb_adduction_grip prep") == "thumb_adduction_grip"
    assert command_base("mode primitive") == "mode"
    assert command_base("open") == "open"


def test_l25_keeps_grasp_ready_and_twist() -> None:
    steps = _parse_suite(DEFAULT_SUITE)
    runnable, skipped = partition_by_hand_support(steps, "l25")
    names = {command_base(s.command) for s in runnable}
    assert "grasp_ready" in names
    assert "twist2digit" in names
    assert "large_wrap" not in names
    skipped_bases = {command_base(s.command) for s, _ in skipped}
    assert "large_wrap" in skipped_bases
    assert "hammer_grip" in skipped_bases


def test_o20_skips_l25_only_motion() -> None:
    steps = _parse_suite(DEFAULT_SUITE)
    runnable, skipped = partition_by_hand_support(steps, "o20")
    skipped_bases = {command_base(s.command) for s, _ in skipped}
    assert "grasp_ready" in skipped_bases
    assert "index_flick" in skipped_bases
    assert "twist5digit" in skipped_bases
    assert all(command_base(s.command) != "grasp_ready" for s in runnable)


def test_o6_keeps_large_wrap_skips_parallel_extension() -> None:
    steps = _parse_suite(DEFAULT_SUITE)
    runnable, skipped = partition_by_hand_support(steps, "o6")
    run_bases = {command_base(s.command) for s in runnable}
    skipped_bases = {command_base(s.command) for s, _ in skipped}
    assert "large_wrap" in run_bases
    assert "hammer_grip" in run_bases
    assert "parallel_extension" in skipped_bases
    assert "index_middle_adduction_grip" in skipped_bases
    assert "grasp_ready" in skipped_bases
    # mode 控制指令始终保留
    assert any(s.command == "mode primitive" for s in runnable)


def test_l6_skips_middle_pinch_and_hammer() -> None:
    steps = _parse_suite(DEFAULT_SUITE)
    runnable, skipped = partition_by_hand_support(steps, "l6")
    skipped_bases = {command_base(s.command) for s, _ in skipped}
    assert "middle_pinch" in skipped_bases
    assert "hammer_grip" in skipped_bases
    assert "parallel_extension_by_vision" in skipped_bases
    assert any(command_base(s.command) == "large_wrap" for s in runnable)


def test_include_unsupported_runs_all() -> None:
    steps = _parse_suite(DEFAULT_SUITE)
    runnable, skipped = partition_by_hand_support(steps, "o20", include_unsupported=True)
    assert not skipped
    assert len(runnable) == len(steps)


def test_skip_reason_is_readable() -> None:
    steps = [SuiteStep("grasp_ready", 1.0, "static")]
    runnable, skipped = partition_by_hand_support(steps, "o6")
    assert not runnable
    assert len(skipped) == 1
    assert "grasp_ready" in skipped[0][1]
    assert "不支持" in skipped[0][1] or "未启用" in skipped[0][1]
