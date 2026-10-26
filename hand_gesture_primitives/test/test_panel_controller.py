"""PanelController 状态机单测（无 ROS）。"""

import unittest

from hand_gesture_primitives.teach_panel.controller import PanelController, PanelState


class TestPanelController(unittest.TestCase):
    def setUp(self) -> None:
        self.cmds = []
        self.pubs = []
        self.t = [100.0]

        def now():
            return self.t[0]

        self.c = PanelController(
            3,
            publish_rate=10.0,
            status_timeout_sec=2.0,
            send_cmd=lambda x: self.cmds.append(x),
            publish_joints=lambda p: self.pubs.append(list(p)),
            now=now,
        )

    def test_mute_until_teach(self) -> None:
        self.assertTrue(self.c.muted)
        err = self.c.set_joints([1, 2, 3])
        self.assertEqual(err, "muted_or_not_teach")
        # Mute 时仍保留本地 targets，避免 UI 被 snapshot 打回默认 128
        self.assertEqual(self.c.targets, [1.0, 2.0, 3.0])
        self.assertEqual(self.pubs, [])

    def test_enter_teach_waits_mode_ack(self) -> None:
        self.c.on_status("hold|mode=primitive", stamp=100.0)
        self.c.enter_teach()
        self.assertEqual(self.cmds[-1], "mode teach")
        self.assertEqual(self.c.state, PanelState.TEACH_EDIT)
        self.assertTrue(self.c.muted)
        # ACK 前不下发
        self.c.set_joints([10, 20, 30])
        self.c.tick(stamp=100.2)
        self.assertEqual(self.pubs, [])
        # mode=teach 确认后 unmute
        self.c.on_status("hold|mode=teach", stamp=100.3)
        self.assertFalse(self.c.muted)
        self.c.set_joints([10, 20, 30])
        self.c.tick(stamp=100.5)
        self.assertEqual(self.pubs[-1], [10.0, 20.0, 30.0])

    def test_ws_disconnect_exits_teach(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.assertEqual(self.c.state, PanelState.TEACH_EDIT)
        self.c.set_joints([1, 2, 3])
        self.cmds.clear()
        self.c.on_ws_disconnected()
        self.assertTrue(self.c.muted)
        self.assertIn("mode primitive", self.cmds)
        self.assertEqual(self.c.state, PanelState.PRIMITIVE_IDLE)
        self.t[0] = 100.5
        self.c.tick(stamp=100.5)
        self.assertEqual(self.pubs, [])

    def test_status_timeout_disconnect(self) -> None:
        self.c.on_status("hold|mode=primitive", stamp=100.0)
        self.t[0] = 103.0
        self.c.tick(stamp=103.0)
        self.assertEqual(self.c.state, PanelState.DISCONNECTED)
        self.assertTrue(self.c.muted)

    def test_playback_from_teach_returns_to_teach(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.begin_playback("my_open")
        self.assertTrue(self.c.muted)
        self.assertEqual(self.cmds[-2:], ["mode primitive", "pose my_open"])
        self.assertEqual(self.c.state, PanelState.PLAYBACK)
        # 切 mode 瞬间的 IDLE / hold 不得打断试播
        self.c.on_result("IDLE")
        self.assertEqual(self.c.state, PanelState.PLAYBACK)
        self.c.on_status("hold|mode=primitive", stamp=100.2)
        self.assertEqual(self.c.state, PanelState.PLAYBACK)
        self.c.on_status("named_pose:my_open|mode=primitive", stamp=100.4)
        self.cmds.clear()
        self.c.on_result("DONE: named_pose:my_open")
        # 默认试播后回示教
        self.assertEqual(self.c.state, PanelState.TEACH_EDIT)
        self.assertTrue(self.c.muted)
        self.assertIn("mode teach", self.cmds)
        self.c.on_status("hold|mode=teach", stamp=100.5)
        self.assertFalse(self.c.muted)

    def test_playback_from_teach_no_return_when_disabled(self) -> None:
        self.c.return_to_teach_after_playback = False
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.begin_playback("my_open")
        self.c.on_status("named_pose:my_open|mode=primitive", stamp=100.1)
        self.c.on_result("DONE: named_pose:my_open")
        self.assertEqual(self.c.state, PanelState.PRIMITIVE_IDLE)

    def test_playback_teach_reject_retries_after_mode_ack(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.begin_playback("my_open")
        self.cmds.clear()
        # 乐观 pose 若抢跑被拒，应保留并在 mode ACK 后补发
        self.c.on_status("rejected:teach_mode", stamp=100.1)
        self.assertEqual(self.c.state, PanelState.PLAYBACK)
        self.assertEqual(self.c._pending_pose_cmd, "my_open")
        self.c.on_status("hold|mode=primitive", stamp=100.2)
        self.assertIn("pose my_open", self.cmds)

    def test_playback_from_primitive_sends_pose_immediately(self) -> None:
        self.c.on_status("hold|mode=primitive", stamp=100.0)
        self.c.begin_playback("my_open")
        self.assertIn("pose my_open", self.cmds)
        self.assertEqual(self.c._pending_pose_cmd, "")
        # 立刻 hold 不得结束（尚未见 named_pose）
        self.c.on_status("hold|mode=primitive", stamp=100.1)
        self.assertEqual(self.c.state, PanelState.PLAYBACK)
        self.c.on_result("IDLE")
        self.assertEqual(self.c.state, PanelState.PLAYBACK)

    def test_playback_idle_after_named_pose_finishes(self) -> None:
        self.c.on_status("hold|mode=primitive", stamp=100.0)
        self.c.begin_playback("my_open")
        self.c.on_status("named_pose:my_open|mode=primitive", stamp=100.1)
        self.c.on_result("IDLE")
        # 非示教发起：结束后保持 primitive
        self.assertEqual(self.c.state, PanelState.PRIMITIVE_IDLE)

    def test_playback_reject_finishes_without_resume(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.begin_playback("missing")
        self.cmds.clear()
        self.c.on_status("rejected:pose_missing:missing", stamp=100.1)
        self.assertEqual(self.c.state, PanelState.PRIMITIVE_IDLE)
        self.assertTrue(self.c.last_error.startswith("rejected:"))
        self.assertEqual(self.c._pending_pose_cmd, "")
        self.assertNotIn("mode teach", self.cmds)

    def test_reject_status_mutes_in_teach(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.on_status("rejected:pose_missing:x", stamp=100.1)
        self.assertTrue(self.c.muted)
        self.assertTrue(self.c.last_error.startswith("rejected:"))

    def test_reinit_sends_init(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.reinit()
        self.assertIn("mode primitive", self.cmds)
        self.assertIn("init", self.cmds)
        self.assertEqual(self.c.state, PanelState.PRIMITIVE_IDLE)
        self.assertEqual(self.c.editing_pose_name, "")

    def test_load_pose_for_edit_waits_ack(self) -> None:
        err = self.c.load_pose_for_edit("my_a", [10, 20, 30])
        self.assertIsNone(err)
        self.assertEqual(self.c.editing_pose_name, "my_a")
        self.assertEqual(self.c.targets, [10.0, 20.0, 30.0])
        self.assertEqual(self.c.state, PanelState.TEACH_EDIT)
        self.assertIn("mode teach", self.cmds)
        self.assertEqual(self.pubs, [])
        self.c.on_status("hold|mode=teach", stamp=100.1)
        self.assertEqual(self.pubs[-1], [10.0, 20.0, 30.0])
        self.assertFalse(self.c.muted)

    def test_snapshot_includes_hint_and_pref(self) -> None:
        snap = self.c.snapshot()
        self.assertIn("hint", snap)
        self.assertTrue(snap["return_to_teach_after_playback"])

    def test_unmute_keeps_dirty_targets(self) -> None:
        """示教中改过滑条后，Mute 解除不得用反馈冲掉未保存目标。"""
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.assertFalse(self.c.muted)
        self.c.update_feedback([50, 50, 50])
        self.c.set_joints([200, 210, 220])
        self.assertEqual(self.c.targets, [200.0, 210.0, 220.0])
        # 模拟 reject / 外部切 primitive 再回 teach
        self.c.on_status("rejected:pose_missing:x", stamp=100.1)
        self.assertTrue(self.c.muted)
        self.c.update_feedback([50, 50, 50])
        self.c.on_status("hold|mode=teach", stamp=100.2)
        self.assertFalse(self.c.muted)
        self.assertEqual(self.c.targets, [200.0, 210.0, 220.0])

    def test_unmute_pulls_feedback_when_clean(self) -> None:
        self.c.on_status("hold|mode=teach", stamp=100.0)
        self.c.update_feedback([11, 22, 33])
        self.c.load_pose_for_edit("p", [40, 50, 60], enter_teach=True)
        # load 清 dirty；再 mute→unmute 应允许拉反馈
        self.c.on_status("rejected:x", stamp=100.1)
        self.c.update_feedback([77, 88, 99])
        self.c.on_status("hold|mode=teach", stamp=100.2)
        self.assertEqual(self.c.targets, [77.0, 88.0, 99.0])


if __name__ == "__main__":
    unittest.main()
