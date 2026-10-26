"""Teach 面板 pose_bundle 下载/导入 HTTP API 单测。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("fastapi")

from hand_gesture_primitives.pose_library import pose_entry_to_hardware
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_IDX
from hand_gesture_primitives.teach_panel.controller import PanelController
from hand_gesture_primitives.teach_panel.server import TeachPanelApp

from hand_gesture_primitives import pose_library as pl


def _make_app(pose_dir: Path, hand_type: str = "o20", hand_side: str = "left") -> TeachPanelApp:
    ctrl = PanelController(
        20 if hand_type == "o20" else 6,
        publish_rate=10.0,
        status_timeout_sec=2.0,
        send_cmd=lambda _x: None,
        publish_joints=lambda _p: None,
    )
    names = [f"j{i}" for i in range(20 if hand_type == "o20" else 6)]
    return TeachPanelApp(
        ctrl,
        hand_type=hand_type,
        hand_side=hand_side,
        joint_names=names,
    )


class TestTeachPanelBundleApi(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pose_dir = Path(self._tmpdir.name)
        self._env = mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(self._pose_dir)})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmpdir.cleanup()

    def test_export_and_import_roundtrip(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")

        pl.upsert_pose(
            "o20",
            "panel_a",
            [float(i) for i in range(20)],
            hand_side="left",
            space="hardware",
        )
        app = _make_app(self._pose_dir)
        client = TestClient(app.app)

        res = client.get("/api/pose_bundle/export")
        self.assertEqual(res.status_code, 200)
        self.assertIn("attachment", res.headers.get("content-disposition", ""))
        payload = res.json()
        self.assertEqual(payload["kind"], "hand_gesture_pose_bundle")
        self.assertEqual(len(payload["entries"]), 1)

        pl.delete_pose("o20", "panel_a", hand_side="left")
        imp = client.post(
            "/api/pose_bundle/import?strategy=skip",
            content=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(imp.status_code, 200)
        body = imp.json()
        self.assertTrue(body["success"])
        self.assertGreaterEqual(body["imported"], 1)
        pose_names = [p["name"] for p in body["poses"]]
        self.assertIn("panel_a", pose_names)
        self.assertIsNotNone(pl.load_pose("o20", "panel_a", hand_side="left"))

        # 再次导入同名 skip → 应判定不成功（全 skip）
        again = client.post(
            "/api/pose_bundle/import?strategy=skip",
            content=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(again.status_code, 409)
        self.assertFalse(again.json()["success"])

    def test_import_factory(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")

        app = _make_app(self._pose_dir, hand_type="o20", hand_side="left")
        client = TestClient(app.app)
        res = client.post("/api/pose_bundle/import_factory?overwrite=true")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["success"])
        self.assertGreater(body["count"], 0)
        self.assertTrue(any(n.startswith("preset_") for n in body["names"]))

    def test_import_factory_l25_includes_calibrated_pinch(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")

        from hand_gesture_primitives.hand_config import HandConfig
        from hand_gesture_primitives.planable_primitives import PLANABLE_STATIC_PRIMITIVES
        from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM

        names = list(HandConfig("l25").joint_names)
        ctrl = PanelController(
            25,
            publish_rate=10.0,
            status_timeout_sec=2.0,
            send_cmd=lambda _x: None,
            publish_joints=lambda _p: None,
        )
        app = TeachPanelApp(
            ctrl,
            hand_type="l25",
            hand_side="left",
            joint_names=names,
            layout_hand_type="l25",
        )
        client = TestClient(app.app)
        res = client.post("/api/pose_bundle/import_factory?overwrite=true")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["success"])
        self.assertEqual(body["count"], len(PLANABLE_STATIC_PRIMITIVES))
        self.assertIn("preset_pinch", body["names"])
        self.assertIn("preset_fist", body["names"])
        pinch = pl.load_pose("l25", "preset_pinch", hand_side="left")
        self.assertIsNotNone(pinch)
        self.assertEqual(pinch.num_joints, SEMANTIC_DIM)
        self.assertEqual(float(pinch.positions[SEMANTIC_30_IDX["index_mcp_flex"]]), 255.0)
        hw = pose_entry_to_hardware(pinch, "l25")
        self.assertIsNotNone(hw)
        self.assertEqual(hw[1], 0.0)

    def test_import_factory_skip_all_not_success(self) -> None:
        try:
            from fastapi.testclient import TestClient
        except ImportError:
            self.skipTest("fastapi not installed")

        app = _make_app(self._pose_dir, hand_type="o20", hand_side="left")
        client = TestClient(app.app)
        first = client.post("/api/pose_bundle/import_factory?overwrite=true")
        self.assertEqual(first.status_code, 200)
        self.assertTrue(first.json()["success"])
        again = client.post("/api/pose_bundle/import_factory?overwrite=false")
        self.assertEqual(again.status_code, 409)
        body = again.json()
        self.assertFalse(body["success"])
        self.assertEqual(body["count"], 0)


class TestTeachPanelJointDisplay(unittest.TestCase):
    def test_joint_display_hw_only_without_converter(self) -> None:
        ctrl = PanelController(
            3,
            send_cmd=lambda _x: None,
            publish_joints=lambda _p: None,
        )
        app = TeachPanelApp(
            ctrl,
            hand_type="o20",
            hand_side="left",
            joint_names=["a", "b", "c"],
        )
        meta = app._joint_display_meta()  # noqa: SLF001
        # 无 Converter 仍提供 rad/deg（0~π/2 近似）
        self.assertEqual(meta["available"], ["hw255", "rad", "deg"])
        self.assertTrue(meta["approx"])
        self.assertFalse(meta["has_converter"])
        self.assertEqual(len(meta["scales"]), 3)
        self.assertGreater(meta["scales"][0]["upper"], meta["scales"][0]["lower"])
        snap = app._build_snapshot()  # noqa: SLF001
        self.assertIn("joint_display", snap)
        self.assertEqual(snap["joint_display"]["available"], ["hw255", "rad", "deg"])

    def test_joint_display_rad_deg_with_converter(self) -> None:
        class _FakeConv:
            joint_names = ["r_hand_thumb_base", "r_hand_index_base"]
            limits = [(0.0, 1.57), (-0.5, 0.5)]
            semantic_indices = [0, 5]
            inverted_indices = {5}

        ctrl = PanelController(
            2,
            send_cmd=lambda _x: None,
            publish_joints=lambda _p: None,
        )
        app = TeachPanelApp(
            ctrl,
            hand_type="o20",
            hand_side="left",
            joint_names=["thumb_base", "index_base", "orphan"],
            radian_converter=_FakeConv(),
        )
        meta = app._joint_display_meta()  # noqa: SLF001
        self.assertEqual(meta["available"], ["hw255", "rad", "deg"])
        self.assertEqual(meta["scales"][0]["upper"], 1.57)
        self.assertTrue(meta["scales"][1]["inverted"])
        # orphan 无匹配时回退 0~π/2 近似
        self.assertAlmostEqual(meta["scales"][2]["upper"], 1.5707963267948966)
        self.assertTrue(meta["approx"])

    def test_joint_display_prefix_strip_o20_names(self) -> None:
        from hand_gesture_primitives.hand_config import HandConfig
        from hand_gesture_primitives.joint_state_mapping import ros2_control_joint_names

        layout = HandConfig("o20").joint_names
        ctrl_names = ros2_control_joint_names("o20", "right")

        class _FakeConv:
            joint_names = ctrl_names
            limits = [(0.0, 1.0)] * len(ctrl_names)
            semantic_indices = list(range(len(ctrl_names)))
            inverted_indices = set()

        ctrl = PanelController(
            len(layout),
            send_cmd=lambda _x: None,
            publish_joints=lambda _p: None,
        )
        app = TeachPanelApp(
            ctrl,
            hand_type="o20",
            hand_side="right",
            joint_names=list(layout),
            radian_converter=_FakeConv(),
        )
        meta = app._joint_display_meta()  # noqa: SLF001
        self.assertIn("rad", meta["available"])
        matched = sum(1 for s in meta["scales"] if float(s["upper"]) > float(s["lower"]))
        self.assertGreaterEqual(matched, 15)


if __name__ == "__main__":
    unittest.main()
