"""semantic_30 投影往返与视图测试。"""

import pytest
from hand_gesture_primitives.semantic_30 import (
    _S30_UNMAPPED,
    ABD_INDICES_S30,
    SEMANTIC_30_IDX,
    SEMANTIC_30_NAMES,
    SemanticHand30,
    project_o20_to_s30,
    project_s30_to_o20,
)


class TestNameTable:
    def test_length_is_30(self):
        assert len(SEMANTIC_30_NAMES) == 30

    def test_idx_bijection(self):
        assert len(SEMANTIC_30_IDX) == 30
        for i, name in enumerate(SEMANTIC_30_NAMES):
            assert SEMANTIC_30_IDX[name] == i

    def test_finger_structure(self):
        for fi, finger in enumerate(["thumb", "index", "middle", "ring", "pinky"]):
            base_idx = fi * 5
            assert SEMANTIC_30_NAMES[base_idx].startswith(finger)
            assert "mcp_flex" in SEMANTIC_30_NAMES[base_idx]

    def test_block_b_starts_at_25(self):
        assert SEMANTIC_30_NAMES[25] == "wrist_flex"
        assert SEMANTIC_30_NAMES[29] == "reserved"


class TestProjection:
    def test_o20_to_s30_length(self):
        vec20 = list(range(20))
        vec30 = project_o20_to_s30(vec20)
        assert len(vec30) == 30

    def test_s30_to_o20_length(self):
        vec30 = list(range(30))
        vec20 = project_s30_to_o20(vec30)
        assert len(vec20) == 20

    def test_roundtrip_o20_s30_o20(self):
        """§3.5 契约：业务 16 轴 + rsv→pip 往返恒等。"""
        vec20 = [float(i * 10) for i in range(20)]
        vec30 = project_o20_to_s30(vec20)
        vec20_back = project_s30_to_o20(vec30)
        assert vec20_back == vec20

    def test_roundtrip_s30_o20_s30_mapped_channels(self):
        """映射通道 s30→o20→s30 恒等。"""
        vec30 = [float(i * 5) for i in range(30)]
        vec20 = project_s30_to_o20(vec30)
        vec30_back = project_o20_to_s30(vec20)
        for s30_idx in range(30):
            if s30_idx not in _S30_UNMAPPED:
                assert vec30_back[s30_idx] == vec30[s30_idx], (
                    f"s30[{s30_idx}] ({SEMANTIC_30_NAMES[s30_idx]}) 往返不等"
                )

    def test_unmapped_channels_zero_after_o20_roundtrip(self):
        """未映射通道经 o20 往返后归零。"""
        vec30 = [99.0] * 30
        vec20 = project_s30_to_o20(vec30)
        vec30_back = project_o20_to_s30(vec20)
        for s30_idx in _S30_UNMAPPED:
            assert vec30_back[s30_idx] == 0.0, (
                f"s30[{s30_idx}] ({SEMANTIC_30_NAMES[s30_idx]}) 应为 0"
            )

    def test_thumb_rot_mapping(self):
        """thumb_rot: O20[10] ↔ S30[4]。"""
        vec20 = [0.0] * 20
        vec20[10] = 200.0
        vec30 = project_o20_to_s30(vec20)
        assert vec30[4] == 200.0
        assert project_s30_to_o20(vec30)[10] == 200.0

    def test_reserved_to_pip_mapping(self):
        """O20 rsv[11-14] ↔ S30 四指 pip。"""
        vec20 = [0.0] * 20
        vec20[11] = 50.0  # index_pip
        vec20[12] = 60.0  # middle_pip
        vec20[13] = 70.0  # ring_pip
        vec20[14] = 80.0  # pinky_pip
        vec30 = project_o20_to_s30(vec20)
        assert vec30[7] == 50.0  # index_pip
        assert vec30[12] == 60.0  # middle_pip
        assert vec30[17] == 70.0  # ring_pip
        assert vec30[22] == 80.0  # pinky_pip

    def test_short_input_raises(self):
        with pytest.raises(ValueError):
            project_o20_to_s30([0.0] * 19)
        with pytest.raises(ValueError):
            project_s30_to_o20([0.0] * 29)

    def test_abd_indices_s30(self):
        """S30 abd 通道在正确位置。"""
        expected = [
            SEMANTIC_30_IDX["thumb_mcp_abd"],
            SEMANTIC_30_IDX["index_mcp_abd"],
            SEMANTIC_30_IDX["middle_mcp_abd"],
            SEMANTIC_30_IDX["ring_mcp_abd"],
            SEMANTIC_30_IDX["pinky_mcp_abd"],
        ]
        assert ABD_INDICES_S30 == expected


class TestSemanticHand30:
    def test_default_zero(self):
        h = SemanticHand30()
        assert h.as_vector() == [0.0] * 30

    def test_from_vector_roundtrip(self):
        data = [float(i) for i in range(30)]
        h = SemanticHand30.from_vector(data)
        assert h.as_vector() == data

    def test_thumb_view(self):
        h = SemanticHand30()
        h.thumb.mcp_flex = 100.0
        h.thumb.mcp_abd = 128.0
        h.thumb.dip = 80.0
        h.thumb.rot = 200.0
        vec = h.as_vector()
        assert vec[0] == 100.0
        assert vec[1] == 128.0
        assert vec[3] == 80.0
        assert vec[4] == 200.0

    def test_finger_view(self):
        h = SemanticHand30()
        h.index.mcp_flex = 50.0
        h.index.pip = 30.0
        h.index.dip = 70.0
        vec = h.as_vector()
        assert vec[5] == 50.0
        assert vec[7] == 30.0
        assert vec[8] == 70.0

    def test_finger_by_name(self):
        h = SemanticHand30()
        h.finger("ring").mcp_flex = 42.0
        assert h.as_vector()[15] == 42.0

    def test_finger_unknown_raises(self):
        h = SemanticHand30()
        with pytest.raises(KeyError):
            h.finger("unknown")

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            SemanticHand30([0.0] * 20)

    def test_indexing(self):
        h = SemanticHand30()
        h[7] = 99.0
        assert h[7] == 99.0
        assert len(h) == 30
