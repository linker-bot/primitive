"""L25 tip↔mid 耦合：只读映射与强制同步。"""

import unittest

from hand_gesture_primitives.joint_coupling import (
    apply_tip_mid_coupling,
    couple_meta,
    tip_mid_couple_map,
)
from hand_gesture_primitives.teach_panel.controller import PanelController, PanelState


class TestJointCoupling(unittest.TestCase):
    def test_l25_five_tips(self) -> None:
        cmap = tip_mid_couple_map("l25")
        self.assertEqual(len(cmap), 5)
        # mid 15-19 → tip 20-24
        for mid, tip in zip(range(15, 20), range(20, 25)):
            self.assertEqual(cmap[tip], mid)

    def test_o20_no_coupling(self) -> None:
        self.assertEqual(tip_mid_couple_map("o20"), {})

    def test_apply_forces_tip_to_mid(self) -> None:
        cmap = tip_mid_couple_map("l25")
        pos = [float(i) for i in range(25)]
        out = apply_tip_mid_coupling(pos, cmap)
        for tip, mid in cmap.items():
            self.assertEqual(out[tip], out[mid])

    def test_controller_set_joints_couples(self) -> None:
        cmap = tip_mid_couple_map("l25")
        c = PanelController(25, couple_map=cmap)
        c.state = PanelState.TEACH_EDIT
        c.muted = False
        c.control_mode = "teach"
        vals = [10.0] * 25
        vals[15] = 200.0  # thumb mid
        vals[20] = 1.0  # tip should be overwritten
        err = c.set_joints(vals)
        self.assertIsNone(err)
        self.assertEqual(c.targets[20], 200.0)

    def test_couple_meta_labels(self) -> None:
        meta = couple_meta("l25")
        self.assertEqual(len(meta), 5)
        self.assertTrue(all("耦合" in m["label"] for m in meta))


if __name__ == "__main__":
    unittest.main()
