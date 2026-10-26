"""pub_gesture_cmd 查询结果解析 / 格式化。"""

import unittest

from hand_gesture_primitives.pub_gesture_cmd import format_pose_list, parse_poses_result


class TestPoseListCliHelpers(unittest.TestCase):
    def test_parse_poses_empty(self) -> None:
        self.assertEqual(parse_poses_result("POSES:"), [])
        self.assertEqual(parse_poses_result("POSES: "), [])

    def test_parse_poses_names(self) -> None:
        self.assertEqual(parse_poses_result("POSES:a,b"), ["a", "b"])
        self.assertEqual(parse_poses_result("POSES:my_open"), ["my_open"])

    def test_parse_poses_ignores_other(self) -> None:
        self.assertIsNone(parse_poses_result("IDLE"))
        self.assertIsNone(parse_poses_result("DONE: open"))
        self.assertIsNone(parse_poses_result(""))

    def test_format_empty(self) -> None:
        text = format_pose_list([], side="left", source="/left_hand/gesture_result")
        self.assertIn("(空)", text)
        self.assertIn("left", text)

    def test_format_names(self) -> None:
        text = format_pose_list(
            ["my_open", "wave"],
            side="right",
            source="local:/tmp/poses",
            hand_type="g20",
        )
        self.assertIn("g20/right", text)
        self.assertIn("- my_open", text)
        self.assertIn("- wave", text)
        self.assertIn("共 2 个", text)


if __name__ == "__main__":
    unittest.main()
