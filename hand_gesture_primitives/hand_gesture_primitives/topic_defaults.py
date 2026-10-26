"""Default ROS topic names for perception and side-scoped gesture APIs.

Dual-hand contract (side = left|right):
  /{side}_hand/gesture_cmd
  /{side}_hand/gesture_cmd_exec
  /{side}_hand/gesture_status
  /{side}_hand/gesture_result
  /{side}_hand/tcp_pose
  /{side}_hand/object_pose
  /{side}_hand/stream_target
  /{side}_hand/motion_progress
  /{side}_hand/grasp_gate/status
  /{side}_hand/grasp_gate/selected_label

Perception topics may stay shared (camera). Empty launch params resolve via
``resolve_side_topic`` / ``side_hand_topics``.
"""

from __future__ import annotations

from typing import Dict

DEFAULT_CAMERA_NS = "/camera_head"


def bboxes_3d_topic(camera_ns: str = DEFAULT_CAMERA_NS) -> str:
    return f"{camera_ns.rstrip('/')}/detection_bbox/bboxes_3d"


def labeled_poses_topic(camera_ns: str = DEFAULT_CAMERA_NS) -> str:
    return f"{camera_ns.rstrip('/')}/perception/labeled_poses"


def side_hand_prefix(side: str) -> str:
    s = (side or "left").strip().lower()
    if s not in ("left", "right"):
        raise ValueError(f"hand_side must be left|right, got {side!r}")
    return f"/{s}_hand"


def side_hand_topics(side: str) -> Dict[str, str]:
    """Return the dual-hand public topic map for one side."""
    p = side_hand_prefix(side)
    return {
        "gesture_cmd": f"{p}/gesture_cmd",
        "gesture_cmd_exec": f"{p}/gesture_cmd_exec",
        "gesture_status": f"{p}/gesture_status",
        "gesture_result": f"{p}/gesture_result",
        "tcp_pose": f"{p}/tcp_pose",
        "object_pose": f"{p}/object_pose",
        "gate_status": f"{p}/grasp_gate/status",
        "selected_label": f"{p}/grasp_gate/selected_label",
        "stream_target": f"{p}/stream_target",
        "motion_progress": f"{p}/motion_progress",
    }


def resolve_side_topic(value: str, auto_topic: str) -> str:
    """Use explicit non-empty value; otherwise the side-auto topic."""
    v = (value or "").strip()
    return v if v else auto_topic
