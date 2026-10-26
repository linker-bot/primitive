"""Keep every test away from operator pose libraries and production ROS domains."""

import pytest


@pytest.fixture(autouse=True)
def isolated_runtime(monkeypatch, tmp_path):
    monkeypatch.setenv("HAND_GESTURE_POSE_DIR", str(tmp_path / "poses"))
    monkeypatch.setenv("HAND_GESTURE_RUNTIME_STATUS_PATH", str(tmp_path / "runtime.json"))
    monkeypatch.setenv("ROS_DOMAIN_ID", "89")
