"""Teach 面板关键帧摘要 / 删帧 / 细控回放（无 ROS）。"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import mock

from hand_gesture_primitives.pose_library import PoseKeyframe
from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM
from hand_gesture_primitives.teach_panel.controller import PanelController, PanelState

from hand_gesture_primitives import pose_library as pl


def test_list_pose_summaries_and_delete_keyframe(tmp_path: Path = None) -> None:
    tmp = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    with mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(tmp)}):
        err = pl.upsert_pose(
            "o20",
            "demo",
            [0.0] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="legacy255",
            keyframes=[
                PoseKeyframe("pre", [0.0] * SEMANTIC_DIM, 0.5),
                PoseKeyframe("grasp", [10.0] * SEMANTIC_DIM, 0.8),
            ],
        )
        assert err is None
        summaries = pl.list_pose_summaries("o20", hand_side="left")
        assert len(summaries) == 1
        assert summaries[0]["name"] == "demo"
        assert summaries[0]["keyframe_count"] == 2
        assert [k["name"] for k in summaries[0]["keyframes"]] == ["pre", "grasp"]
        assert "description" in summaries[0]

        err = pl.delete_pose_keyframe("o20", "demo", "pre", hand_side="left")
        assert err is None
        entry = pl.load_pose("o20", "demo", hand_side="left")
        assert entry is not None
        assert [kf.name for kf in entry.keyframes] == ["grasp"]


def test_rejected_mode_pending_pose_keeps_progress() -> None:
    cmds: list[str] = []
    ctrl = PanelController(
        6,
        send_cmd=lambda c: cmds.append(c),
        publish_joints=lambda _p: None,
    )
    ctrl.control_mode = "teach"
    ctrl.state = PanelState.TEACH_EDIT
    ctrl.begin_playback("demo", keyframe="grasp", progress=0.4)
    assert ctrl._playback_progress == 0.4  # noqa: SLF001
    cmds.clear()
    ctrl.on_status("rejected:teach_mode")
    assert "0.4" in ctrl._pending_pose_cmd  # noqa: SLF001
    assert "grasp" in ctrl._pending_pose_cmd  # noqa: SLF001
    ctrl.on_status("hold|mode=primitive")
    assert any(c == "pose demo grasp 0.4" for c in cmds)


def test_begin_playback_progress_mode_no_auto_finish() -> None:
    cmds = []
    ctrl = PanelController(
        6,
        send_cmd=lambda c: cmds.append(c),
        publish_joints=lambda _p: None,
    )
    ctrl.control_mode = "primitive"
    ctrl.state = PanelState.PRIMITIVE_IDLE
    ctrl.begin_playback("demo", keyframe="grasp", progress=0.0)
    assert ctrl.state == PanelState.PLAYBACK
    assert ctrl._playback_progress_mode is True  # noqa: SLF001
    assert any(c.startswith("pose demo grasp") for c in cmds)

    ctrl.on_result("EXECUTING: named_pose:demo | drive=progress|keyframe=grasp|progress=0.000")
    assert ctrl.motion_keyframe == "grasp"
    ctrl.on_result("DONE: named_pose:demo")
    # PROGRESS 不因 DONE 自动结束
    assert ctrl.state == PanelState.PLAYBACK

    ctrl.set_playback_progress(0.35)
    assert abs(ctrl.motion_progress - 0.35) < 1e-6
    # 无 publish_progress 时退回 pose 指令
    assert any("0.35" in c for c in cmds)

    ctrl.stop_playback(resume_teach=False)
    assert ctrl.state == PanelState.PRIMITIVE_IDLE
    assert "stop" in cmds


def test_scrub_to_same_segment_hot_update_then_switch() -> None:
    cmds: list[str] = []
    published: list[float] = []
    ctrl = PanelController(
        6,
        send_cmd=lambda c: cmds.append(c),
        publish_joints=lambda _p: None,
    )
    ctrl._publish_progress = lambda p: published.append(float(p))  # noqa: SLF001
    ctrl.control_mode = "primitive"
    ctrl.state = PanelState.PRIMITIVE_IDLE

    assert ctrl.scrub_to("demo", keyframe="pre", progress=0.0) is None
    assert ctrl.state == PanelState.PLAYBACK
    assert ctrl._playback_keyframe == "pre"  # noqa: SLF001
    assert any(c.startswith("pose demo pre") for c in cmds)

    # 尚未收到 EXECUTING 时不算同段，再次 scrub 会 begin 重发
    n_before = len(cmds)
    assert ctrl.scrub_to("demo", keyframe="pre", progress=0.2) is None
    assert len(cmds) > n_before

    ctrl.on_result("EXECUTING: named_pose:demo | drive=progress|keyframe=pre|progress=0.200")
    assert ctrl._playback_seen_named is True  # noqa: SLF001

    # 同段热更新：走 progress topic，不再发 pose
    n_cmds = len(cmds)
    assert ctrl.scrub_to("demo", keyframe="pre", progress=0.55) is None
    assert abs(ctrl.motion_progress - 0.55) < 1e-6
    assert published and abs(published[-1] - 0.55) < 1e-6
    assert len(cmds) == n_cmds

    # 换段：begin_playback 重锚
    assert ctrl.scrub_to("demo", keyframe="grasp", progress=0.1) is None
    assert ctrl._playback_keyframe == "grasp"  # noqa: SLF001
    assert any("pose demo grasp" in c for c in cmds)
    assert abs(ctrl.motion_progress - 0.1) < 1e-6


def test_map_global_progress_duration_weighted() -> None:
    """前端逻辑的纯 Python 对照：按时长加权跨段映射。"""

    def map_global(keyframes, g: float):
        segs = [
            (str(k["name"]).lower(), max(0.05, float(k.get("duration") or 0.6))) for k in keyframes
        ]
        total = sum(d for _, d in segs)
        t = max(0.0, min(1.0, g)) * total
        for i, (name, d) in enumerate(segs):
            last = i == len(segs) - 1
            if t <= d or last:
                local = max(0.0, min(1.0, t / d)) if d > 0 else 1.0
                return name, local
            t -= d
        return segs[-1][0], 1.0

    kfs = [
        {"name": "pre", "duration": 1.0},
        {"name": "grasp", "duration": 3.0},
    ]
    assert map_global(kfs, 0.0) == ("pre", 0.0)
    name, local = map_global(kfs, 0.25)  # 1s / 4s → 段末 pre
    assert name == "pre"
    assert abs(local - 1.0) < 1e-6
    name, local = map_global(kfs, 0.5)  # 2s → grasp 1/3
    assert name == "grasp"
    assert abs(local - (1.0 / 3.0)) < 1e-6
    name, local = map_global(kfs, 1.0)
    assert name == "grasp"
    assert abs(local - 1.0) < 1e-6


def test_begin_playback_auto_segment_sends_auto_token() -> None:
    cmds: list[str] = []
    ctrl = PanelController(
        6,
        send_cmd=lambda c: cmds.append(c),
        publish_joints=lambda _p: None,
    )
    ctrl.control_mode = "primitive"
    ctrl.state = PanelState.PRIMITIVE_IDLE
    ctrl.begin_playback("demo", keyframe="pre", auto_segment=True)
    assert ctrl.state == PanelState.PLAYBACK
    assert ctrl._playback_progress_mode is False  # noqa: SLF001
    assert ctrl._playback_auto_segment is True  # noqa: SLF001
    assert any(c == "pose demo pre auto" for c in cmds)


def test_move_and_rename_pose_keyframe(tmp_path: Path = None) -> None:
    tmp = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    with mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(tmp)}):
        err = pl.upsert_pose(
            "o20",
            "demo",
            [0.0] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="legacy255",
            keyframes=[
                PoseKeyframe("pre", [0.0] * SEMANTIC_DIM, 0.5),
                PoseKeyframe("grasp", [10.0] * SEMANTIC_DIM, 0.8),
            ],
        )
        assert err is None
        assert pl.move_pose_keyframe("o20", "demo", "grasp", delta=-1, hand_side="left") is None
        entry = pl.load_pose("o20", "demo", hand_side="left")
        assert entry is not None
        assert [kf.name for kf in entry.keyframes] == ["grasp", "pre"]
        assert pl.rename_pose_keyframe("o20", "demo", "grasp", "close", hand_side="left") is None
        entry = pl.load_pose("o20", "demo", hand_side="left")
        assert entry is not None
        assert [kf.name for kf in entry.keyframes] == ["close", "pre"]


def test_save_pose_uses_editing_keyframe_when_blank() -> None:
    """二次编辑：请求未带 keyframe 时覆盖 editing_keyframe，而非新建/拒绝。"""
    from hand_gesture_primitives.teach_panel.server import TeachPanelApp

    tmp = Path(tempfile.mkdtemp())
    with mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(tmp)}):
        err = pl.upsert_pose(
            "o20",
            "demo",
            [0.0] * SEMANTIC_DIM,
            hand_side="right",
            space="semantic_v3",
            unit="legacy255",
            keyframes=[
                PoseKeyframe("prepare", [1.0] * SEMANTIC_DIM, 0.6),
                PoseKeyframe("grasp", [2.0] * SEMANTIC_DIM, 0.6),
            ],
        )
        assert err is None
        ctrl = PanelController(
            20,
            send_cmd=lambda _c: None,
            publish_joints=lambda _p: None,
        )
        ctrl.state = PanelState.TEACH_EDIT
        ctrl.control_mode = "teach"
        ctrl.muted = False
        ctrl.editing_pose_name = "demo"
        ctrl.editing_keyframe = "prepare"
        ctrl.targets = [180.0] * 20
        app = TeachPanelApp(
            ctrl,
            hand_type="o20",
            hand_side="right",
            joint_names=[f"j{i}" for i in range(20)],
        )
        resp = app._save_pose(  # noqa: SLF001
            {
                "name": "demo",
                "keyframe": "",
                "duration": 0.6,
                "use_targets": True,
            }
        )
        assert resp.get("type") == "ok", resp
        assert resp.get("keyframe") == "prepare"
        entry = pl.load_pose("o20", "demo", hand_side="right")
        assert entry is not None
        assert [kf.name for kf in entry.keyframes] == ["prepare", "grasp"]
        # prepare 应被更新（不全是 1.0）
        assert any(abs(v - 1.0) > 1e-3 for v in entry.keyframes[0].positions)


def test_save_pose_multi_kf_blank_errors_without_editing() -> None:
    from hand_gesture_primitives.teach_panel.server import TeachPanelApp

    tmp = Path(tempfile.mkdtemp())
    with mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(tmp)}):
        err = pl.upsert_pose(
            "o20",
            "demo",
            [0.0] * SEMANTIC_DIM,
            hand_side="right",
            space="semantic_v3",
            unit="legacy255",
            keyframes=[
                PoseKeyframe("a", [1.0] * SEMANTIC_DIM, 0.6),
                PoseKeyframe("b", [2.0] * SEMANTIC_DIM, 0.6),
            ],
        )
        assert err is None
        ctrl = PanelController(
            20,
            send_cmd=lambda _c: None,
            publish_joints=lambda _p: None,
        )
        ctrl.state = PanelState.TEACH_EDIT
        ctrl.control_mode = "teach"
        ctrl.muted = False
        ctrl.targets = [100.0] * 20
        app = TeachPanelApp(
            ctrl,
            hand_type="o20",
            hand_side="right",
            joint_names=[f"j{i}" for i in range(20)],
        )
        resp = app._save_pose({"name": "demo", "keyframe": "", "use_targets": True})  # noqa: SLF001
        assert resp.get("type") == "error"
        assert "关键帧" in resp.get("message", "")
