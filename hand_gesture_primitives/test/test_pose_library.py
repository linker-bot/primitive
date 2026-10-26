"""Pose Library 校验、L20→l25 分桶、按侧文件 + 旧文件回退。"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM

from hand_gesture_primitives import pose_library as pl


class TestPoseLibrary(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._pose_dir = Path(self._tmpdir.name)
        # Docker 常设 HAND_GESTURE_POSE_DIR，需一并指向临时目录，否则会写到真实配置仓
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

    def test_reject_reserved_name(self) -> None:
        err = pl.validate_pose_name("open")
        self.assertIsNotNone(err)
        err = pl.validate_pose_name("pose")
        self.assertIsNotNone(err)

    def test_reject_bad_length(self) -> None:
        err = pl.validate_positions([1.0, 2.0], 20)
        self.assertIsNotNone(err)

    def test_l20_writes_l25_bucket(self) -> None:
        positions = [128.0] * 25
        err = pl.upsert_pose(
            "L20", "my_open", positions, duration=0.5, hand_side="left", space="hardware"
        )
        self.assertIsNone(err)
        path = self._pose_dir / "l25_left.json"
        self.assertTrue(path.is_file())
        names = pl.list_poses("l25", hand_side="left")
        self.assertEqual(names, ["my_open"])
        entry = pl.load_pose("l20", "my_open", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.num_joints, 25)
        self.assertEqual(len(entry.positions), 25)

    def test_o20_roundtrip(self) -> None:
        positions = [float(i) for i in range(20)]
        err = pl.upsert_pose("o20", "wave_a", positions, hand_side="right", space="hardware")
        self.assertIsNone(err)
        entry = pl.load_pose("O20", "wave_a", hand_side="right")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.positions, positions)
        self.assertTrue((self._pose_dir / "o20_right.json").is_file())

    def test_legacy_fallback_read(self) -> None:
        """旧版无侧文件仍可读；写入侧文件后优先侧文件。"""
        positions = [10.0] * 20
        err = pl.upsert_pose("o20", "legacy_a", positions, space="hardware")
        self.assertIsNone(err)
        self.assertTrue((self._pose_dir / "o20.json").is_file())
        entry = pl.load_pose("o20", "legacy_a", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.positions[0], 10.0)

        err = pl.upsert_pose("o20", "side_only", [20.0] * 20, hand_side="left", space="hardware")
        self.assertIsNone(err)
        self.assertTrue((self._pose_dir / "o20_left.json").is_file())
        names = pl.list_poses("o20", hand_side="left")
        self.assertIn("side_only", names)
        self.assertIn("legacy_a", names)

    def test_left_right_isolated(self) -> None:
        pl.upsert_pose("o6", "grip", [1.0] * 6, hand_side="left", space="hardware")
        pl.upsert_pose("o6", "grip", [2.0] * 6, hand_side="right", space="hardware")
        left = pl.load_pose("o6", "grip", hand_side="left")
        right = pl.load_pose("o6", "grip", hand_side="right")
        self.assertEqual(left.positions[0], 1.0)
        self.assertEqual(right.positions[0], 2.0)

    def test_legacy_delete_seeds_side_file(self) -> None:
        pl.upsert_pose("o20", "legacy_del", [30.0] * 20, space="hardware")
        self.assertTrue(pl.delete_pose("o20", "legacy_del", hand_side="left"))
        self.assertIsNotNone(pl.load_pose("o20", "legacy_del", hand_side="right"))
        self.assertTrue((self._pose_dir / "o20_left.json").is_file())
        self.assertIsNone(pl.load_pose("o20", "legacy_del", hand_side="left"))
        self.assertIn("legacy_del", pl.list_poses("o20", hand_side="right"))

    def test_corrupt_json_quarantined_not_wiped_silently(self) -> None:
        path = self._pose_dir / "o20_left.json"
        path.write_text("{not-json", encoding="utf-8")
        self.assertEqual(pl.list_poses("o20", hand_side="left"), [])
        corrupt = list(self._pose_dir.glob("o20_left.json.corrupt.*"))
        self.assertEqual(len(corrupt), 1)
        err = pl.upsert_pose("o20", "recovered", [1.0] * 20, hand_side="left", space="hardware")
        self.assertIsNone(err)
        self.assertTrue(path.is_file())
        self.assertEqual(pl.list_poses("o20", hand_side="left"), ["recovered"])

    def test_reject_nan_and_inf(self) -> None:
        self.assertIsNotNone(pl.validate_positions([float("nan")] * 20, 20))
        self.assertIsNotNone(pl.validate_positions([float("inf")] * 20, 20))

    def test_corrupt_version_falls_back(self) -> None:
        path = self._pose_dir / "o20_left.json"
        path.write_text(
            '{"version": "bad", "hand_joint": "O20", "poses": {}}',
            encoding="utf-8",
        )
        data = pl._load_file("o20", "left")
        self.assertEqual(data["version"], pl.POSE_SCHEMA_VERSION)

    def test_pose_dir_env_override(self) -> None:
        other = Path(self._tmpdir.name) / "alt_poses"
        other.mkdir()
        with mock.patch.dict("os.environ", {"HAND_GESTURE_POSE_DIR": str(other)}):
            err = pl.upsert_pose("o6", "env_pose", [3.0] * 6, hand_side="left", space="hardware")
            self.assertIsNone(err)
            self.assertTrue((other / "o6_left.json").is_file())

    def test_list_poses_skips_corrupt_entries(self) -> None:
        pl.upsert_pose("o20", "good", [5.0] * 20, hand_side="left", space="hardware")
        path = self._pose_dir / "o20_left.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        data["poses"]["bad_nan"] = {
            "space": "hardware",
            "positions": [float("nan")] * 20,
            "num_joints": 20,
            "duration": 0.5,
        }
        data["poses"]["bad_len"] = {
            "space": "hardware",
            "positions": [1.0] * 3,
            "num_joints": 3,
            "duration": 0.5,
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        names = pl.list_poses("o20", hand_side="left")
        self.assertEqual(names, ["good"])
        self.assertIsNone(pl.load_pose("o20", "bad_nan", hand_side="left"))
        self.assertIsNone(pl.load_pose("o20", "bad_len", hand_side="left"))

    def test_default_save_space_is_semantic_v3(self) -> None:
        self.assertEqual(pl.DEFAULT_SAVE_SPACE, "semantic_v3")
        positions = [0.0] * SEMANTIC_DIM
        positions[5] = 128.0
        err = pl.upsert_pose("o20", "s30_wave", positions, hand_side="left", unit="legacy255")
        self.assertIsNone(err)
        entry = pl.load_pose("o20", "s30_wave", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.space, "semantic_v3")
        self.assertEqual(entry.num_joints, SEMANTIC_DIM)
        self.assertEqual(entry.unit, "legacy255")
        self.assertEqual(entry.positions[5], 128.0)

    def test_prepare_pose_for_save_from_hardware(self) -> None:
        from hand_gesture_primitives.hand_config import HandConfig

        hand = HandConfig("l25")
        hw = [float(i) for i in range(hand.num_joints)]
        prepared = pl.prepare_pose_for_save("l25", hw)
        self.assertIsInstance(prepared, tuple)
        positions, space, unit = prepared
        self.assertEqual(space, "semantic_v3")
        self.assertEqual(unit, "legacy255")
        self.assertEqual(len(positions), SEMANTIC_DIM)
        self.assertEqual(positions, hand.to_semantic(hw))

    def test_prepare_pose_for_save_accepts_semantic_v3(self) -> None:
        vec = [0.0] * SEMANTIC_DIM
        vec[5] = 99.0
        prepared = pl.prepare_pose_for_save("o20", vec, space="semantic_v3")
        positions, space, unit = prepared
        self.assertEqual(space, "semantic_v3")
        self.assertEqual(unit, "legacy255")
        self.assertEqual(positions[5], 99.0)

    def test_prepare_pose_for_save_hardware_space(self) -> None:
        hw = [128.0] * 20
        prepared = pl.prepare_pose_for_save("o20", hw, space="hardware")
        positions, space, unit = prepared
        self.assertEqual(space, "hardware")
        self.assertEqual(unit, "legacy255")
        self.assertEqual(positions, hw)

    def test_prepare_pose_for_save_invalid_space(self) -> None:
        err = pl.prepare_pose_for_save("o20", [0.0] * 20, space="invalid")
        self.assertIsInstance(err, str)
        self.assertIn("space", err)

    def test_prepare_pose_empty_space_defaults_semantic_v3(self) -> None:
        hw = [128.0] * 6
        prepared = pl.prepare_pose_for_save("o6", hw, space="")
        positions, space, unit = prepared
        self.assertEqual(space, "semantic_v3")
        self.assertEqual(len(positions), SEMANTIC_DIM)

    def test_semantic_v3_without_unit_defaults_legacy255(self) -> None:
        path = self._pose_dir / "o20_left.json"
        path.write_text(
            json.dumps(
                {
                    "version": 2,
                    "hand_joint": "O20",
                    "canonical_hand_type": "o20",
                    "hand_side": "left",
                    "poses": {
                        "legacy_s30": {
                            "space": "semantic_v3",
                            "positions": [128.0] * SEMANTIC_DIM,
                            "num_joints": SEMANTIC_DIM,
                            "duration": 0.6,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        entry = pl.load_pose("o20", "legacy_s30", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.unit, "legacy255")

    def test_keyframes_roundtrip_and_mirror_last(self) -> None:
        from hand_gesture_primitives.pose_library import PoseKeyframe

        pre = [0.1] * SEMANTIC_DIM
        grasp = [0.2] * SEMANTIC_DIM
        err = pl.upsert_pose(
            "o20",
            "my_grip",
            grasp,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
            keyframes=[
                PoseKeyframe("pre", pre, 0.4),
                PoseKeyframe("grasp", grasp, 0.5),
            ],
        )
        self.assertIsNone(err)
        entry = pl.load_pose("o20", "my_grip", hand_side="left")
        self.assertIsNotNone(entry)
        self.assertEqual(len(entry.keyframes), 2)
        self.assertEqual(entry.keyframes[0].name, "pre")
        self.assertEqual(entry.positions, grasp)
        self.assertEqual(entry.duration, 0.5)
        prog = pl.motion_program_from_entry(entry)
        self.assertEqual(prog.segment_names(), ["pre", "grasp"])

    def test_reject_hardware_keyframes(self) -> None:
        from hand_gesture_primitives.pose_library import PoseKeyframe

        err = pl.upsert_pose(
            "o20",
            "bad_hw",
            [1.0] * 20,
            hand_side="left",
            space="hardware",
            keyframes=[PoseKeyframe("pre", [1.0] * 20, 0.4)],
        )
        self.assertIsNotNone(err)
        self.assertIn("hardware", err.lower())

    def test_upsert_pose_keyframe_rejects_numeric_name(self) -> None:
        err = pl.upsert_pose_keyframe(
            "o20",
            "num_kf",
            "2",
            [0.1] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
        )
        self.assertIsNotNone(err)
        self.assertIn("纯数字", err or "")

    def test_upsert_pose_keyframe_migrates_single(self) -> None:
        pl.upsert_pose(
            "o20",
            "grip2",
            [0.1] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
        )
        err = pl.upsert_pose_keyframe(
            "o20",
            "grip2",
            "pre2",
            [0.3] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
        )
        self.assertIsNone(err)
        entry = pl.load_pose("o20", "grip2", hand_side="left")
        self.assertEqual([k.name for k in entry.keyframes], ["default", "pre2"])
        self.assertEqual(entry.positions[0], 0.3)

    def test_upsert_pose_refuses_wipe_keyframes(self) -> None:
        from hand_gesture_primitives.pose_library import PoseKeyframe

        err = pl.upsert_pose(
            "o20",
            "keep_kf",
            [0.2] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
            keyframes=[
                PoseKeyframe("pre", [0.1] * SEMANTIC_DIM, 0.4),
                PoseKeyframe("grasp", [0.2] * SEMANTIC_DIM, 0.5),
            ],
        )
        self.assertIsNone(err)
        err2 = pl.upsert_pose(
            "o20",
            "keep_kf",
            [0.9] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
        )
        self.assertIsNotNone(err2)
        self.assertIn("keyframe", err2.lower())
        entry = pl.load_pose("o20", "keep_kf", hand_side="left")
        self.assertEqual(len(entry.keyframes), 2)
        self.assertAlmostEqual(entry.positions[0], 0.2)

        err3 = pl.upsert_pose(
            "o20",
            "keep_kf",
            [0.9] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
            replace_multi_keyframe=True,
        )
        self.assertIsNone(err3)
        entry2 = pl.load_pose("o20", "keep_kf", hand_side="left")
        self.assertEqual(entry2.keyframes, ())
        self.assertAlmostEqual(entry2.positions[0], 0.9)

    def test_commit_saved_pose_keyframe_and_plain(self) -> None:
        err = pl.commit_saved_pose(
            "o20",
            "svc_pose",
            [0.1] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
            keyframe="pre",
        )
        self.assertIsNone(err)
        err2 = pl.commit_saved_pose(
            "o20",
            "svc_pose",
            [0.4] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
            keyframe="grasp",
        )
        self.assertIsNone(err2)
        entry = pl.load_pose("o20", "svc_pose", hand_side="left")
        self.assertEqual([k.name for k in entry.keyframes], ["pre", "grasp"])
        # 无 keyframe 再写会拒抹
        err3 = pl.commit_saved_pose(
            "o20",
            "svc_pose",
            [0.9] * SEMANTIC_DIM,
            hand_side="left",
            space="semantic_v3",
            unit="rad",
        )
        self.assertIsNotNone(err3)

    def test_upsert_writes_joint_names_and_positions_named(self) -> None:
        from hand_gesture_primitives.hand_config import HandConfig
        from hand_gesture_primitives.semantic_30 import SEMANTIC_30_NAMES

        err = pl.upsert_pose(
            "o20",
            "named_demo",
            [float(i) * 0.01 for i in range(SEMANTIC_DIM)],
            hand_side="right",
            space="semantic_v3",
            unit="rad",
        )
        self.assertIsNone(err)
        raw = json.loads((self._pose_dir / "o20_right.json").read_text(encoding="utf-8"))
        rec = raw["poses"]["named_demo"]
        self.assertEqual(rec["joint_names"], list(SEMANTIC_30_NAMES))
        self.assertEqual(rec["positions_named"]["thumb_mcp_flex"], rec["positions"][0])
        self.assertEqual(len(rec["positions_named"]), SEMANTIC_DIM)
        # 旁路字段不影响 load
        entry = pl.load_pose("o20", "named_demo", hand_side="right")
        self.assertIsNotNone(entry)
        self.assertEqual(entry.positions[0], rec["positions"][0])

        err = pl.upsert_pose(
            "o20",
            "hw_named",
            [10.0] * 20,
            hand_side="right",
            space="hardware",
        )
        self.assertIsNone(err)
        raw = json.loads((self._pose_dir / "o20_right.json").read_text(encoding="utf-8"))
        hw = raw["poses"]["hw_named"]
        self.assertEqual(hw["joint_names"], HandConfig("o20").joint_names)
        self.assertEqual(hw["positions_named"]["thumb_base"], 10.0)


if __name__ == "__main__":
    unittest.main()
