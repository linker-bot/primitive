"""Pose bundle 导入导出 roundtrip / 策略 / 跨手型拒绝。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_gesture_primitives.export_primitive_poses import (
    export_primitive_poses,
    primitive_final_hardware,
)
from hand_gesture_primitives.planable_primitives import (
    is_planable_for_hand,
    is_planable_static,
    planable_reject_reason,
)

from hand_gesture_primitives import pose_bundle as pb
from hand_gesture_primitives import pose_library as pl


class TestPoseBundle(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pose_dir = Path(self._tmpdir.name) / "poses"
        self._pose_dir.mkdir()
        self._env = mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(self._pose_dir)})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmpdir.cleanup()

    def test_roundtrip_export_import_skip(self) -> None:
        pl.upsert_pose(
            "o20",
            "wave_a",
            [float(i) for i in range(20)],
            hand_side="left",
            space="hardware",
        )
        out = Path(self._tmpdir.name) / "out.pose_bundle.json"
        path, count = pb.export_library_to_file(
            out, hand_type="o20", hand_side="left", exporter="test"
        )
        self.assertEqual(count, 1)
        self.assertTrue(path.is_file())

        # 清空库再导入
        pl.delete_pose("o20", "wave_a", hand_side="left")
        self.assertEqual(pl.list_poses("o20", hand_side="left"), [])
        result = pb.import_bundle_file(out, strategy="skip")
        self.assertEqual(result.imported, 1)
        self.assertEqual(result.failed, 0)
        entry = pl.load_pose("o20", "wave_a", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.positions[0], 0.0)

    def test_skip_vs_overwrite(self) -> None:
        pl.upsert_pose("o6", "grip", [1.0] * 6, hand_side="left", space="hardware")
        out = Path(self._tmpdir.name) / "o6.pose_bundle.json"
        pb.export_library_to_file(out, hand_type="o6", hand_side="left")

        pl.upsert_pose("o6", "grip", [9.0] * 6, hand_side="left", space="hardware")
        skip = pb.import_bundle_file(out, strategy="skip")
        self.assertEqual(skip.skipped, 1)
        self.assertEqual(pl.load_pose("o6", "grip", hand_side="left").positions[0], 9.0)

        over = pb.import_bundle_file(out, strategy="overwrite")
        self.assertEqual(over.imported, 1)
        self.assertEqual(pl.load_pose("o6", "grip", hand_side="left").positions[0], 1.0)

    def test_rename_on_conflict(self) -> None:
        pl.upsert_pose("o6", "grip", [1.0] * 6, hand_side="right", space="hardware")
        out = Path(self._tmpdir.name) / "r.pose_bundle.json"
        pb.export_library_to_file(out, hand_type="o6", hand_side="right")
        pl.upsert_pose("o6", "grip", [2.0] * 6, hand_side="right", space="hardware")
        result = pb.import_bundle_file(out, strategy="rename")
        self.assertEqual(result.imported, 1)
        names = pl.list_poses("o6", hand_side="right")
        self.assertIn("grip", names)
        self.assertTrue(any(n.startswith("grip_imp") for n in names))

    def test_cross_hand_joint_mismatch_fails(self) -> None:
        # 手工构造错误 DOF 条目
        entry = pb.BundleEntry(
            canonical_hand_type="o6",
            hand_side="left",
            name="bad",
            positions=[1.0] * 20,
            num_joints=20,
            space="hardware",
        )
        result = pb.import_bundle(pb.build_bundle([entry]), strategy="overwrite")
        self.assertEqual(result.failed, 1)
        self.assertEqual(result.imported, 0)

    def test_checksum_mismatch(self) -> None:
        pl.upsert_pose("o20", "a", [1.0] * 20, hand_side="left", space="hardware")
        out = Path(self._tmpdir.name) / "chk.pose_bundle.json"
        pb.export_library_to_file(out, hand_type="o20", hand_side="left")
        import json

        raw = json.loads(out.read_text(encoding="utf-8"))
        raw["checksum"]["value"] = "0" * 64
        out.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaises(ValueError):
            pb.read_bundle(out, verify_checksum=True)

    def test_import_from_legacy_dir(self) -> None:
        legacy = Path(self._tmpdir.name) / "legacy"
        legacy.mkdir()
        with mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(legacy)}):
            pl.upsert_pose("o20", "legacy_a", [5.0] * 20, hand_side="left", space="hardware")
        # 当前库仍是 self._pose_dir
        result = pb.import_library_dir(legacy, strategy="skip", default_side="left")
        self.assertGreaterEqual(result.imported, 1)
        self.assertIsNotNone(pl.load_pose("o20", "legacy_a", hand_side="left"))

    def test_no_legacy_duplicate_on_full_hand_export(self) -> None:
        pl.upsert_pose("o20", "legacy_only", [1.0] * 20, space="hardware")
        pl.upsert_pose("o20", "side_only", [2.0] * 20, hand_side="left", space="hardware")
        # 侧文件首次写入会种子复制 legacy → legacy_only 只应出现一次
        entries = pb.collect_library_entries(hand_type="o20")
        names = [(e.name, e.hand_side) for e in entries]
        self.assertEqual(sum(1 for n, _ in names if n == "legacy_only"), 1)
        self.assertEqual(sum(1 for n, _ in names if n == "side_only"), 1)

    def test_single_side_falls_back_to_legacy(self) -> None:
        pl.upsert_pose("o20", "legacy_only", [3.0] * 20, space="hardware")
        entries = pb.collect_library_entries(hand_type="o20", hand_side="left")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0].hand_side, "left")

    def test_cross_side_import_not_success(self) -> None:
        entry = pb.BundleEntry(
            canonical_hand_type="o20",
            hand_side="right",
            name="wave",
            positions=[1.0] * 20,
            num_joints=20,
            space="hardware",
        )
        result = pb.import_bundle(
            pb.build_bundle([entry]),
            strategy="skip",
            hand_type="o20",
            hand_side="left",
        )
        self.assertEqual(result.imported, 0)
        self.assertFalse(result.success)
        self.assertTrue(any("no poses imported" in m for m in result.messages))

    def test_o20_bundle_upgrades_to_s30_on_import(self) -> None:
        entry = pb.BundleEntry(
            canonical_hand_type="o20",
            hand_side="left",
            name="legacy_o20",
            positions=[float(i) for i in range(20)],
            num_joints=20,
            space="semantic_o20",
            unit="legacy255",
        )
        result = pb.import_bundle(pb.build_bundle([entry]), strategy="overwrite")
        self.assertEqual(result.imported, 1)
        loaded = pl.load_pose("o20", "legacy_o20", hand_side="left")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.space, "semantic_v3")
        self.assertEqual(loaded.num_joints, 30)
        self.assertEqual(loaded.unit, "legacy255")


class TestPrimitiveExport(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pose_dir = Path(self._tmpdir.name)
        self._env = mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(self._pose_dir)})
        self._env.start()

    def tearDown(self) -> None:
        self._env.stop()
        self._tmpdir.cleanup()

    def test_planable_helper(self) -> None:
        self.assertTrue(is_planable_static("pinch"))
        self.assertFalse(is_planable_static("large_wrap_by_vision"))
        # O20 YAML 无 pinch gestures 段时不可对本手规划
        if not is_planable_for_hand("o20", "pinch"):
            self.assertIsNotNone(planable_reject_reason("o20", "pinch"))
        self.assertTrue(is_planable_for_hand("o6", "fist"))
        self.assertIsNone(planable_reject_reason("o6", "fist"))

    def test_fist_hardware_width(self) -> None:
        hw, duration = primitive_final_hardware("o6", "fist")
        self.assertEqual(len(hw), 6)
        self.assertGreater(duration, 0.0)
        self.assertTrue(all(0.0 <= v <= 255.0 for v in hw))

    def test_export_to_library_and_bundle(self) -> None:
        out = Path(self._tmpdir.name) / "factory.pose_bundle.json"
        count, path, names = export_primitive_poses(
            "o20",
            "left",
            primitives=["open", "fist"],
            write_library=True,
            output_path=str(out),
            overwrite=True,
        )
        self.assertEqual(count, 2)
        self.assertTrue(path.is_file())
        self.assertIn("preset_open", names)
        self.assertIn("preset_open", pl.list_poses("o20", hand_side="left"))

    def test_export_annotates_joint_names_import_ignores(self) -> None:
        """导出写旁路名字；带 joint_names 的 bundle 仍可导入（真相源 positions）。"""
        import json

        pl.upsert_pose(
            "o20",
            "wave_b",
            [float(i) for i in range(20)],
            hand_side="left",
            space="hardware",
        )
        out = Path(self._tmpdir.name) / "named.pose_bundle.json"
        pb.export_library_to_file(out, hand_type="o20", hand_side="left", exporter="test")
        payload = json.loads(out.read_text(encoding="utf-8"))
        entry = payload["entries"][0]
        self.assertIn("joint_names", entry)
        self.assertEqual(len(entry["joint_names"]), 20)
        self.assertEqual(entry["positions_named"]["thumb_base"], 0.0)
        self.assertEqual(entry["positions"][0], 0.0)

        pl.delete_pose("o20", "wave_b", hand_side="left")
        result = pb.import_bundle_file(out, strategy="skip")
        self.assertEqual(result.imported, 1)
        loaded = pl.load_pose("o20", "wave_b", hand_side="left")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.positions[0], 0.0)


if __name__ == "__main__":
    unittest.main()
