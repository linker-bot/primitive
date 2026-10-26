"""Embedding launch helpers carry explicit storage and speed to every host."""

import sys
from types import ModuleType

import pytest
from hand_gesture_primitives.launch_utils import gesture_nodes_for_side


def nodes(monkeypatch, **kwargs):
    actions = ModuleType("launch_ros.actions")
    actions.Node = lambda **values: values
    monkeypatch.setitem(sys.modules, "launch_ros.actions", actions)
    return gesture_nodes_for_side(
        "right",
        hand_type="o6",
        hand_joint="O6",
        with_gate=True,
        with_pose_server=True,
        with_teach_panel=True,
        **kwargs,
    )


def test_embedding_propagates_pose_directory_and_speed(monkeypatch, tmp_path):
    result = nodes(monkeypatch, pose_dir=str(tmp_path), primitive_speed=0.5)
    by_name = {item["executable"]: item for item in result}
    assert set(by_name) == {"gesture_node", "grasp_gate", "pose_library_server", "teach_panel"}
    for item in result:
        assert item["additional_env"]["HAND_GESTURE_POSE_DIR"] == str(tmp_path)
    for name in ("gesture_node", "teach_panel"):
        assert by_name[name]["parameters"][0]["primitive_speed"] == 0.5


def test_existing_launch_call_uses_environment_and_normal_speed(monkeypatch):
    result = nodes(monkeypatch)
    for item in result:
        assert "HAND_GESTURE_POSE_DIR" not in item.get("additional_env", {})
        if item["executable"] in {"gesture_node", "teach_panel"}:
            assert item["parameters"][0]["primitive_speed"] == 1.0


def test_invalid_speed_rejected_before_nodes_are_created(monkeypatch):
    with pytest.raises(ValueError):
        nodes(monkeypatch, primitive_speed=float("nan"))
