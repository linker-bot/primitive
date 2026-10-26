"""gui_control presets 导入单测。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_gesture_primitives import import_gui_presets as ig
from hand_gesture_primitives import pose_library as pl


class TestImportGuiPresets(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pose_dir = Path(self._tmpdir.name)
        self._env_patcher = mock.patch.dict(
            os.environ, {"HAND_GESTURE_POSE_DIR": str(self._pose_dir)}
        )
        self._env_patcher.start()
        self._patcher = mock.patch.object(pl, "POSE_DIR", self._pose_dir)
        self._patcher.start()

    def tearDown(self) -> None:
        self._patcher.stop()
        self._env_patcher.stop()
        self._tmpdir.cleanup()

    def test_import_positions(self) -> None:
        src = Path(self._tmpdir.name) / "presets.json"
        src.write_text(
            json.dumps(
                {
                    "poses": {
                        "imported_a": {
                            "positions": [10.0] * 20,
                            "duration": 0.5,
                        },
                        "open": {"positions": [1.0] * 20},
                    }
                }
            ),
            encoding="utf-8",
        )
        imported, skipped, err = ig.import_gui_presets("o20", str(src), overwrite=False)
        self.assertIsNone(err)
        self.assertEqual(imported, 1)
        self.assertGreaterEqual(skipped, 1)
        self.assertEqual(pl.list_poses("o20"), ["imported_a"])

    def test_all_skipped_returns_error(self) -> None:
        src = Path(self._tmpdir.name) / "presets.json"
        src.write_text(
            json.dumps(
                {
                    "poses": {
                        "open": {"positions": [1.0] * 20},
                        "already": {"positions": [2.0] * 20},
                    }
                }
            ),
            encoding="utf-8",
        )
        pl.upsert_pose(
            "o20",
            "already",
            [2.0] * 20,
            hand_side="left",
            space="hardware",
        )
        imported, skipped, err = ig.import_gui_presets(
            "o20", str(src), overwrite=False, hand_side="left"
        )
        self.assertEqual(imported, 0)
        self.assertGreaterEqual(skipped, 2)
        self.assertIsNotNone(err)
        self.assertIn("skipped", err.lower())

    def test_overwrite_replaces_existing_keyframes(self) -> None:
        from hand_gesture_primitives.pose_library import PoseKeyframe
        from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM

        pl.upsert_pose(
            "o20",
            "multi",
            [0.2] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
            keyframes=[
                PoseKeyframe("pre", [0.1] * SEMANTIC_DIM, 0.4),
                PoseKeyframe("grasp", [0.2] * SEMANTIC_DIM, 0.5),
            ],
        )
        src = Path(self._tmpdir.name) / "presets.json"
        src.write_text(
            json.dumps({"poses": {"multi": {"positions": [9.0] * 20}}}),
            encoding="utf-8",
        )
        imported, skipped, err = ig.import_gui_presets(
            "o20", str(src), overwrite=True, hand_side="left"
        )
        self.assertIsNone(err)
        self.assertEqual(imported, 1)
        self.assertEqual(skipped, 0)
        entry = pl.load_pose("o20", "multi", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.keyframes, ())
        self.assertEqual(entry.space, "hardware")
        self.assertEqual(entry.positions[0], 9.0)


if __name__ == "__main__":
    unittest.main()
