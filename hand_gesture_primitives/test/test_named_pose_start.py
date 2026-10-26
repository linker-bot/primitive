"""NamedPose / hold：回放起点优先 commanded，避免反馈偏开导致先张开。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hand_gesture_primitives.primitive_base import PrimitiveContext
from hand_gesture_primitives.primitives.named_pose import NamedPose

from hand_gesture_primitives import pose_library as pl


class TestNamedPoseStartPrefersCommand(unittest.TestCase):
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

    def test_g20_start_uses_cmd_not_open_feedback(self) -> None:
        closed = (
            [40.0, 40.0, 255.0, 255.0, 255.0, 120.0]
            + [255.0] * 4
            + [100.0]
            + [40.0, 40.0]
            + [255.0] * 3
        )
        openish = [255.0] * 16
        err = pl.upsert_pose(
            "g20", "pinch_like", closed, duration=0.5, hand_side="left", space="hardware"
        )
        self.assertIsNone(err)

        prim = NamedPose("pinch_like")
        prim.on_enter([0.0] * 20)
        ctx = PrimitiveContext(
            hand_type="g20",
            hand_side="left",
            hand_hw_angles=openish,
            hand_cmd_angles=closed,
        )
        result0 = prim.compute([0.0] * 20, elapsed=0.0, ctx=ctx)
        self.assertTrue(result0.feasible)
        self.assertEqual(result0.space, "hardware")
        # t=0 应贴近 commanded（闭合），而非反馈的张开
        self.assertLess(result0.target_angles[0], 80.0)
        self.assertGreater(result0.target_angles[0], 20.0)

        result1 = prim.compute([0.0] * 20, elapsed=0.5, ctx=ctx)
        for i, v in enumerate(closed):
            self.assertAlmostEqual(result1.target_angles[i], v, places=2)

    def test_falls_back_to_feedback_when_no_cmd(self) -> None:
        target = [128.0] * 16
        fb = [10.0] * 16
        err = pl.upsert_pose(
            "g20", "mid", target, duration=0.4, hand_side="right", space="hardware"
        )
        self.assertIsNone(err)
        prim = NamedPose("mid")
        prim.on_enter([0.0] * 20)
        ctx = PrimitiveContext(hand_type="g20", hand_side="right", hand_hw_angles=fb)
        r0 = prim.compute([0.0] * 20, elapsed=0.0, ctx=ctx)
        self.assertAlmostEqual(r0.target_angles[0], 10.0, places=2)


@unittest.skipUnless(
    "geometry_msgs" in sys.modules
    or __import__("importlib").util.find_spec("geometry_msgs") is not None,
    "ROS msgs unavailable",
)
class TestExecutorHoldAndRadiansGuard(unittest.TestCase):
    def _bare_executor(self, hand_type: str = "g20"):
        from hand_gesture_primitives.executor import ControlMode, GestureExecutor
        from hand_gesture_primitives.hand_config import HandConfig
        from hand_gesture_primitives.hand_profile import load_hand_profile

        ex = GestureExecutor.__new__(GestureExecutor)
        ex._hand_type = hand_type
        ex._side = "left"
        ex._hand_config = HandConfig(hand_type)
        ex._profile = load_hand_profile(hand_type)
        ex._primitive_space_dim = 20
        n = ex._hand_config.num_joints
        ex._current_angles = [0.0] * 20
        ex._current_hw_angles = [255.0] * n
        ex._last_target = [0.0] * n
        ex._last_commanded_hw = [40.0] * n
        ex._hand_state_received = True
        ex._control_mode = ControlMode.TEACH
        ex._g20_l25_urdf = True
        ex._g20_l25_urdf_locked = True
        ex._cmd_backend = "ros2_control"
        ex._ros2_control_command_units = "radians"
        ex._radian_converter = None
        ex._current_semantic_rad = None
        ex._node = mock.Mock()
        ex._node.get_logger.return_value = mock.Mock()
        ex._active_primitive = None
        return ex

    def test_hold_prefers_commanded_over_open_feedback(self) -> None:
        from hand_gesture_primitives.executor import ControlMode

        ex = self._bare_executor("g20")
        hold = ex._hold_hw_for_resume()
        self.assertEqual(hold[0], 40.0)
        ex.set_control_mode(ControlMode.PRIMITIVE)
        self.assertEqual(ex._last_target[0], 40.0)
        self.assertEqual(ex.control_mode, ControlMode.PRIMITIVE)

    def test_radians_state_does_not_treat_radians_as_hardware(self) -> None:
        from sensor_msgs.msg import JointState

        ex = self._bare_executor("g20")
        ex._current_hw_angles = []
        ex._hand_state_received = False
        msg = JointState()
        msg.name = ["l_hand_thumb_cmc_roll"]
        msg.position = [0.5]
        ex._state_callback(msg)
        self.assertFalse(ex._hand_state_received)
        self.assertEqual(ex._current_hw_angles, [])


if __name__ == "__main__":
    unittest.main()
