"""NamedPose hardware 直出：L25 完整 DOF 不被 from_hardware 裁掉。"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitives.named_pose import NamedPose
from hand_gesture_primitives.semantic_30 import SEMANTIC_DIM

from hand_gesture_primitives import pose_library as pl


class TestNamedPoseHardware(unittest.TestCase):
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

    def test_l25_hardware_preserves_tips(self) -> None:
        # tip 通道设为与 mid 不同，验证直出不经 from_hardware 丢失
        positions = [128.0] * 25
        for i in range(20, 25):
            positions[i] = 40.0 + (i - 20)
        err = pl.upsert_pose(
            "l25", "tip_pose", positions, duration=0.5, hand_side="left", space="hardware"
        )
        self.assertIsNone(err)

        prim = NamedPose("tip_pose")
        start_sem = [10.0] * 20
        prim.on_enter(start_sem)
        hw_fb = [200.0] * 25
        ctx = PrimitiveContext(
            hand_type="l25",
            hand_side="left",
            hand_hw_angles=hw_fb,
        )
        result = prim.compute(start_sem, elapsed=0.0, ctx=ctx)
        self.assertTrue(result.feasible)
        self.assertEqual(result.space, "hardware")
        self.assertEqual(len(result.target_angles), 25)
        # t=0 接近起点；终点应含 tip 目标
        result_end = prim.compute(start_sem, elapsed=0.5, ctx=ctx)
        self.assertTrue(result_end.feasible)
        self.assertEqual(result_end.space, "hardware")
        for i in range(20, 25):
            self.assertAlmostEqual(result_end.target_angles[i], positions[i], places=3)

    def test_semantic_v3_interpolates_in_30d(self) -> None:
        target = [float(i) for i in range(SEMANTIC_DIM)]
        err = pl.upsert_pose(
            "o20",
            "s30_pose",
            target,
            duration=0.5,
            hand_side="left",
            space="semantic_v3",
            unit="legacy255",
        )
        self.assertIsNone(err)

        prim = NamedPose("s30_pose")
        start = [0.0] * SEMANTIC_DIM
        prim.on_enter(start)
        ctx = PrimitiveContext(hand_type="o20", hand_side="left")
        mid = prim.compute(start, elapsed=0.25, ctx=ctx)
        self.assertTrue(mid.feasible)
        self.assertEqual(len(mid.target_angles), SEMANTIC_DIM)
        self.assertAlmostEqual(mid.target_angles[5], target[5] * 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
