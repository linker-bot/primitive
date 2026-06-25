"""World-frame ROI filters for 3D detection results."""
import numpy as np

from robot_perception.utils.workbench_plane import is_on_dynamic_plane


def _resolve_plane_state(args, plane_state):
    if plane_state is not None:
        return plane_state
    return getattr(args, 'workbench_plane_state', None)


def camera_forward_in_world(T_world_cam):
    """Camera optical +Z axis direction in world frame."""
    forward = T_world_cam[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)
    norm = np.linalg.norm(forward)
    if norm < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return forward / norm


def forward_distance_m(center_world, T_world_cam):
    """Signed distance (m) from camera along optical axis to center_world."""
    cam_pos = T_world_cam[:3, 3]
    forward = camera_forward_in_world(T_world_cam)
    return float(np.dot(center_world - cam_pos, forward))


def is_in_forward_roi(center_world, T_world_cam, max_forward_m):
    """True if object center is in front of camera within max_forward_m."""
    dist = forward_distance_m(center_world, T_world_cam)
    return 0.0 < dist <= max_forward_m


def is_on_workbench_surface(aabb_world, workbench_z, surface_tol_m, max_height_m,
                            plane_state=None):
    """True if AABB bottom sits on workbench plane (dynamic RANSAC or fixed z)."""
    if plane_state is not None and getattr(plane_state, 'valid', False):
        return is_on_dynamic_plane(
            aabb_world, plane_state, surface_tol_m, max_height_m)

    z_min = float(aabb_world['min'][2])
    z_max = float(aabb_world['max'][2])
    height = z_max - z_min
    bottom_on_plane = abs(z_min - workbench_z) <= surface_tol_m
    reasonable_height = height <= max_height_m
    above_plane = z_max >= workbench_z - surface_tol_m
    return bottom_on_plane and reasonable_height and above_plane


def get_effective_workbench_z(args, plane_state=None):
    """Return RANSAC plane height when valid, else static workbench_z prior."""
    plane_state = _resolve_plane_state(args, plane_state)
    if plane_state is not None and getattr(plane_state, 'valid', False):
        return float(plane_state.estimated_z)
    return float(args.workbench_z)


def roi_mode_label(args):
    mode = getattr(args, 'world_roi_mode', 'and')
    if mode == 'and':
        return 'forward AND on-surface'
    if mode == 'surface_only':
        return 'on-surface only'
    return 'forward OR on-surface'


def explain_world_roi(aabb_world, T_world_cam, args, plane_state=None):
    """Return human-readable ROI check details for logging."""
    plane_state = _resolve_plane_state(args, plane_state)
    center = np.asarray(aabb_world['center'], dtype=np.float64)
    fwd = forward_distance_m(center, T_world_cam)
    in_forward = is_in_forward_roi(center, T_world_cam, args.world_forward_max_m)
    effective_z = get_effective_workbench_z(args, plane_state)
    on_surface = is_on_workbench_surface(
        aabb_world, effective_z,
        args.workbench_surface_tol_m, args.workbench_max_height_m,
        plane_state=plane_state,
    )
    return {
        'forward_m': round(fwd, 3),
        'forward_ok': in_forward,
        'forward_max_m': float(args.world_forward_max_m),
        'on_surface_ok': on_surface,
        'z_min': round(float(aabb_world['min'][2]), 3),
        'z_max': round(float(aabb_world['max'][2]), 3),
        'plane_z': round(effective_z, 3),
        'mode': getattr(args, 'world_roi_mode', 'and'),
    }


def passes_world_roi(aabb_world, T_world_cam, args, plane_state=None):
    """Filter 3D boxes by configurable forward / workbench rules."""
    if aabb_world is None or T_world_cam is None:
        return False

    plane_state = _resolve_plane_state(args, plane_state)
    center = np.asarray(aabb_world['center'], dtype=np.float64)
    in_forward = is_in_forward_roi(center, T_world_cam, args.world_forward_max_m)
    effective_z = get_effective_workbench_z(args, plane_state)
    on_surface = is_on_workbench_surface(
        aabb_world,
        effective_z,
        args.workbench_surface_tol_m,
        args.workbench_max_height_m,
        plane_state=plane_state,
    )

    mode = getattr(args, 'world_roi_mode', 'and')
    if mode == 'and':
        return in_forward and on_surface
    if mode == 'surface_only':
        return on_surface
    return in_forward or on_surface


def apply_world_roi_to_aabbs(aabb_cam, aabb_work, T_world_cam, args, plane_state=None):
    """Drop published workbench 3D AABBs that fail ROI; camera-frame AABB is kept."""
    if not getattr(args, 'enable_world_roi_filter', True):
        return aabb_cam, aabb_work
    if aabb_work is None or T_world_cam is None:
        return aabb_cam, None
    if passes_world_roi(aabb_work, T_world_cam, args, plane_state=plane_state):
        return aabb_cam, aabb_work
    return aabb_cam, None
