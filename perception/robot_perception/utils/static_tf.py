"""Publish static TF from T_world_cam so RViz can show camera-frame markers."""
import numpy as np
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation


def make_world_to_camera_transform(T_world_cam, world_frame, camera_frame, stamp):
    """Build TransformStamped: parent=world, child=camera (p_world = T @ p_cam)."""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = world_frame
    t.child_frame_id = camera_frame

    R = T_world_cam[:3, :3]
    trans = T_world_cam[:3, 3]
    q = Rotation.from_matrix(R).as_quat()
    t.transform.translation.x = float(trans[0])
    t.transform.translation.y = float(trans[1])
    t.transform.translation.z = float(trans[2])
    t.transform.rotation.x = float(q[0])
    t.transform.rotation.y = float(q[1])
    t.transform.rotation.z = float(q[2])
    t.transform.rotation.w = float(q[3])
    return t


def publish_static_tf_once(broadcaster, T_world_cam, world_frame, camera_frame, stamp):
    """Send a one-shot static transform (safe to call once per camera frame id)."""
    if T_world_cam is None or not camera_frame or not world_frame:
        return False
    msg = make_world_to_camera_transform(
        np.asarray(T_world_cam, dtype=np.float64),
        world_frame,
        camera_frame,
        stamp,
    )
    broadcaster.sendTransform(msg)
    return True
