"""Default ROS topic names for perception integration (override via launch params)."""

DEFAULT_CAMERA_NS = "/camera_head"


def bboxes_3d_topic(camera_ns: str = DEFAULT_CAMERA_NS) -> str:
    return f"{camera_ns.rstrip('/')}/detection_bbox/bboxes_3d"


def labeled_poses_topic(camera_ns: str = DEFAULT_CAMERA_NS) -> str:
    return f"{camera_ns.rstrip('/')}/perception/labeled_poses"
