from __future__ import annotations

import unittest
from unittest import mock

from hand_gesture_primitives.executor import GestureExecutor
from hand_gesture_primitives.gesture_params import load_static_gesture_params
from hand_gesture_primitives.hand_config import HandConfig
from hand_gesture_primitives.primitive_base import JOINT_NAMES, PrimitiveContext
from hand_gesture_primitives.primitives import index_pinch, middle_pinch, pinch
from hand_gesture_primitives.semantic_30 import SEMANTIC_30_NAMES

from hand_gesture_primitives import gesture_params, hand_config


class InitHandPoseTest(unittest.TestCase):
    def test_done_result_is_not_immediately_overwritten_by_idle(self) -> None:
        executor = GestureExecutor.__new__(GestureExecutor)
        executor._result_pub = mock.Mock()
        executor._last_result_msg = ""
        executor._result_msg_count = 0
        executor._result_period_ticks = 1
        executor._active_primitive = None

        with mock.patch(
            "hand_gesture_primitives.executor.time.monotonic",
            side_effect=(10.0, 10.5, 12.0),
        ):
            executor.publish_user_result("DONE: example")
            executor._publish_result()
            executor._publish_result()

        self.assertEqual(
            [call.args[0].data for call in executor._result_pub.publish.call_args_list],
            ["DONE: example", "IDLE"],
        )

    def test_new_primitive_clears_terminal_result_window(self) -> None:
        class Primitive:
            name = "example"
            phase = "full"

            def on_enter(self, _angles):
                return None

        executor = GestureExecutor.__new__(GestureExecutor)
        executor._active_primitive = None
        executor._terminal_result_until = 100.0
        executor._feedback_completion_reported = True
        executor._feedback_completion_diagnostic_logged = True
        executor._last_result_msg = "DONE: previous"
        executor._last_grasp_state = ""
        executor._infeasible_hold_logged = False
        executor._motion_logged = False
        executor._current_angles = [0.0] * 20
        executor._node = mock.Mock()
        executor._reach_strict = False
        executor._hand_type = "o6"
        executor._radian_converter = None
        executor._is_safe_primitive = mock.Mock(return_value=True)
        executor._log_primitive_context = mock.Mock()

        self.assertEqual(executor.set_primitive(Primitive()), "")
        self.assertEqual(executor._terminal_result_until, 0.0)

    def test_executor_passes_normalized_semantic_angles_directly_to_fk(self) -> None:
        executor = GestureExecutor.__new__(GestureExecutor)
        executor._fk_solver = mock.Mock()
        executor._fk_solver.compute_fingertips.return_value = None
        executor._current_angles = [float(index) for index in range(20)]

        executor._compute_fingertip_positions()

        executor._fk_solver.compute_fingertips.assert_called_once_with(executor._current_angles)

    def test_l25_init_uses_hand_profile_gesture_config(self) -> None:
        configured_thumb_base = 31
        configured_index_abduction = 129
        configured_gestures = {
            "init": {
                "angles": {
                    "space": "semantic_o20",
                    "target": {
                        "thumb_base": configured_thumb_base,
                        "index_abd": configured_index_abduction,
                    },
                },
                "motion": {"type": "static", "duration": 0.8},
            }
        }

        load_static_gesture_params.cache_clear()
        try:
            with mock.patch.object(
                gesture_params,
                "_load_raw_gestures",
                return_value=configured_gestures,
            ):
                semantic = load_static_gesture_params("l25", "init").target_angles

            self.assertEqual(
                semantic[JOINT_NAMES.index("thumb_base")],
                configured_thumb_base,
            )
            self.assertEqual(
                semantic[JOINT_NAMES.index("index_abd")],
                configured_index_abduction,
            )
        finally:
            load_static_gesture_params.cache_clear()

    def test_l25_init_differs_from_open(self) -> None:
        """L25 init 为半开待机，open 为全开，二者不得相同。"""
        load_static_gesture_params.cache_clear()
        try:
            init = load_static_gesture_params("l25", "init").target_angles
            open_ = load_static_gesture_params("l25", "open").target_angles
        finally:
            load_static_gesture_params.cache_clear()

        # target_angles 是 30 维 S30 向量，须用 SEMANTIC_30_NAMES 索引
        s30_thumb_flex = SEMANTIC_30_NAMES.index("thumb_mcp_flex")
        s30_thumb_init = init[s30_thumb_flex]
        self.assertNotEqual(init, open_)
        self.assertGreater(s30_thumb_init, open_[s30_thumb_flex])
        self.assertEqual(open_[s30_thumb_flex], 0.0)
        for s30_name in ("index_mcp_abd", "middle_mcp_abd", "ring_mcp_abd", "pinky_mcp_abd"):
            self.assertEqual(open_[SEMANTIC_30_NAMES.index(s30_name)], 128.0)

    def test_hand_profile_yaml_is_parsed_once_and_instances_are_isolated(self) -> None:
        hand_config._load_config_data.cache_clear()
        original_safe_load = hand_config.yaml.safe_load
        try:
            with mock.patch.object(
                hand_config.yaml,
                "safe_load",
                wraps=original_safe_load,
            ) as safe_load:
                first = HandConfig("l25")
                second = HandConfig("L25")

            self.assertEqual(safe_load.call_count, 1)
            self.assertIsNot(first._mapping, second._mapping)
            first._mapping[0]["to"] = -1
            self.assertNotEqual(first._mapping[0]["to"], second._mapping[0]["to"])
        finally:
            hand_config._load_config_data.cache_clear()

    def _settling_executor(self, model, measured, target):
        executor = GestureExecutor.__new__(GestureExecutor)
        executor._hand_type = model
        executor._profile = mock.Mock(hardware=HandConfig(model))
        executor._current_hw_angles = list(measured)
        executor._last_target = list(target)
        executor._active_primitive = mock.Mock(name="init")
        executor._settle_since = None
        executor._settle_status = "running"
        executor._settle_frames = 0
        executor._settle_feedback_at = 0.0
        executor._feedback_received_at = 2.0
        executor._trajectory_complete = False
        executor.publish_user_result = mock.Mock()
        return executor

    def test_o6_hold_primitive_reports_native_feedback_convergence(self) -> None:
        hardware = HandConfig("o6")
        semantic = load_static_gesture_params("o6", "init").target_angles
        native_target = hardware.from_semantic(semantic)
        executor = self._settling_executor("o6", native_target, native_target)
        with mock.patch("hand_gesture_primitives.executor.time.monotonic", return_value=2.0):
            self.assertFalse(executor._feedback_command_converged())
            executor._trajectory_complete = True
            for stamp in (1.9, 1.95):
                executor._feedback_received_at = stamp
                self.assertFalse(executor._settle_completed_motion())
                self.assertFalse(executor._feedback_command_converged())
            # Polling the same cached message cannot complete the trajectory.
            self.assertFalse(executor._settle_completed_motion())
            executor._feedback_received_at = 2.0
            self.assertTrue(executor._settle_completed_motion())
            self.assertTrue(executor._feedback_command_converged())

    def test_l25_completion_ignores_unmeasured_coupled_channels(self) -> None:
        hardware = HandConfig("l25")
        semantic = load_static_gesture_params("l25", "init").target_angles
        native_target = hardware.from_semantic(semantic)
        measured_native = [0.0] * 25
        for index in (0, 1, 2, 3, 4, 5, 10):
            measured_native[index] = max(0.0, native_target[index] - 3.0)
        executor = self._settling_executor("l25", measured_native, native_target)
        executor._trajectory_complete = True
        with mock.patch("hand_gesture_primitives.executor.time.monotonic", return_value=2.0):
            for stamp in (1.9, 1.95, 2.0):
                executor._feedback_received_at = stamp
                executor._settle_completed_motion()
            self.assertTrue(executor._feedback_command_converged())

        # A real measured channel outside tolerance still blocks completion.
        measured_native[0] = native_target[0] + (21.0 if native_target[0] < 230 else -21.0)
        executor = self._settling_executor("l25", measured_native, native_target)
        executor._trajectory_complete = True
        with mock.patch("hand_gesture_primitives.executor.time.monotonic", return_value=2.0):
            for stamp in (1.9, 1.95, 2.0):
                executor._feedback_received_at = stamp
                self.assertFalse(executor._settle_completed_motion())
            self.assertFalse(executor._feedback_command_converged())
        with mock.patch("hand_gesture_primitives.executor.time.monotonic", return_value=3.1):
            self.assertFalse(executor._settle_completed_motion())
            self.assertEqual(executor._settle_status, "settle_timeout")
            executor.publish_user_result.assert_called_once()

    def test_pinch_primitives_resolve_monitor_joints_once_per_activation(self) -> None:
        for module, primitive_type in (
            (pinch, pinch.Pinch),
            (index_pinch, index_pinch.IndexPinch),
            (middle_pinch, middle_pinch.MiddlePinch),
        ):
            with self.subTest(primitive=primitive_type.__name__):
                with mock.patch.object(
                    module,
                    "HandConfig",
                    wraps=module.HandConfig,
                ) as hand_config_factory:
                    primitive = primitive_type()
                    primitive.on_enter([0.0] * 20)
                    context = PrimitiveContext(hand_type="l25")
                    primitive.compute([0.0] * 20, 0.1, context)
                    primitive.compute([0.0] * 20, 0.2, context)

                self.assertEqual(hand_config_factory.call_count, 1)


if __name__ == "__main__":
    unittest.main()
