"""RANSAC workbench plane extraction from depth and dynamic ROI helpers."""
import numpy as np

from robot_perception.utils.bbox3d_from_depth import transform_points


class WorkbenchPlaneState:
    """Estimated workbench plane in world frame: n·p = d (n unit, n[2] >= 0 preferred)."""

    __slots__ = (
        'valid', 'normal', 'd', 'centroid', 'inlier_count', 'estimated_z', 'tilt_deg',
    )

    def __init__(self):
        self.valid = False
        self.normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        self.d = 0.0
        self.centroid = np.zeros(3, dtype=np.float64)
        self.inlier_count = 0
        self.estimated_z = 0.0
        self.tilt_deg = 0.0


def plane_equivalent_z(normal, d):
    """Equivalent world Z height when the plane passes through x=y=0."""
    nz = float(normal[2])
    if abs(nz) < 1e-6:
        return float(d)
    return float(d / nz)


def sample_depth_points_world(depth, K, T_world_cam, args, exclude_mask=None):
    """全图均匀采样深度，投影到世界坐标。"""
    h, w = depth.shape[:2]
    stride = max(1, int(args.workbench_plane_sample_stride))

    vs = np.arange(0, h, stride, dtype=np.int32)
    us = np.arange(0, w, stride, dtype=np.int32)
    if len(vs) == 0 or len(us) == 0:
        return np.zeros((0, 3), dtype=np.float64)

    uu, vv = np.meshgrid(us, vs)
    z = depth[vv, uu].astype(np.float32)
    valid = (
        (z > args.depth_min_m)
        & (z < args.depth_max_m)
        & np.isfinite(z)
    )
    if exclude_mask is not None and exclude_mask.shape[:2] == (h, w):
        valid &= exclude_mask[vv, uu] == 0
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    u_flat = uu[valid].astype(np.float32)
    v_flat = vv[valid].astype(np.float32)
    z_flat = z[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_flat - cx) * z_flat / fx
    y = (v_flat - cy) * z_flat / fy
    pts_cam = np.stack([x, y, z_flat], axis=1).astype(np.float64)
    return transform_points(T_world_cam, pts_cam).astype(np.float64)


def _select_workbench_cluster(points, z_prior, z_tol, bin_size=0.02, min_points=100):
    """Z 高度直方图聚类，选出 z_prior 附近点数最多的水平层。

    先在 z_prior ± z_tol 范围内筛选候选点，再按 bin_size 分桶，
    选择峰值 bin 及相邻 ±1 bin 内的点作为桌面簇。
    """
    z_vals = points[:, 2]
    mask = np.abs(z_vals - z_prior) <= z_tol
    if mask.sum() < min_points:
        return points
    candidates = points[mask]

    z_cand = candidates[:, 2]
    z_min, z_max = float(z_cand.min()), float(z_cand.max())
    n_bins = max(1, int((z_max - z_min) / bin_size))
    if n_bins < 2:
        return candidates

    hist, edges = np.histogram(z_cand, bins=n_bins)
    peak_idx = int(np.argmax(hist))
    lo = edges[max(0, peak_idx - 1)]
    hi = edges[min(len(edges) - 1, peak_idx + 2)]
    cluster_mask = (z_cand >= lo) & (z_cand <= hi)

    cluster = candidates[cluster_mask]
    if len(cluster) < min_points:
        return candidates
    return cluster


def _fit_plane_from_triplet(p0, p1, p2):
    v1 = p1 - p0
    v2 = p2 - p0
    n = np.cross(v1, v2)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    if n[2] < 0.0:
        n = -n
    d = float(np.dot(n, p0))
    return n, d


def _refine_plane_svd(points, inlier_mask):
    pts = points[inlier_mask]
    if len(pts) < 3:
        return None
    centroid = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - centroid, full_matrices=False)
    n = vh[-1].astype(np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-9:
        return None
    n = n / norm
    if n[2] < 0.0:
        n = -n
    d = float(np.dot(n, centroid))
    return n, d, centroid


def _passes_plane_prior(normal, d, args, z_prior=None, z_tol=None):
    tilt_deg = float(np.degrees(np.arccos(np.clip(normal[2], -1.0, 1.0))))
    if tilt_deg > args.workbench_plane_normal_max_tilt_deg:
        return False, tilt_deg
    effective_z = z_prior if z_prior is not None else args.workbench_z
    effective_tol = z_tol if z_tol is not None else args.workbench_z_prior_tol_m
    z_eq = plane_equivalent_z(normal, d)
    if abs(z_eq - effective_z) > effective_tol:
        return False, tilt_deg
    return True, tilt_deg


def ransac_plane(points, args, z_prior=None, z_tol=None):
    """Fit a near-horizontal workbench plane with RANSAC + SVD refinement."""
    n_pts = len(points)
    min_inliers = int(args.workbench_plane_min_inliers)
    if n_pts < max(3, min_inliers):
        return None

    thresh = float(args.workbench_plane_inlier_thresh_m)
    n_iters = int(args.workbench_plane_ransac_iters)
    rng = np.random.default_rng()

    best_count = 0
    best_inliers = None
    best_plane = None

    for _ in range(n_iters):
        idx = rng.choice(n_pts, size=3, replace=False)
        fitted = _fit_plane_from_triplet(points[idx[0]], points[idx[1]], points[idx[2]])
        if fitted is None:
            continue
        n, d = fitted
        dists = np.abs(points @ n - d)
        inliers = dists < thresh
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_plane = (n, d)

    if best_plane is None or best_count < min_inliers:
        return None

    refined = _refine_plane_svd(points, best_inliers)
    if refined is None:
        return None
    normal, d, centroid = refined

    ok, tilt_deg = _passes_plane_prior(normal, d, args, z_prior=z_prior, z_tol=z_tol)
    if not ok:
        return None

    state = WorkbenchPlaneState()
    state.valid = True
    state.normal = normal
    state.d = d
    state.centroid = centroid
    state.inlier_count = best_count
    state.estimated_z = plane_equivalent_z(normal, d)
    state.tilt_deg = tilt_deg
    return state


class WorkbenchPlaneEstimator:
    """Temporal RANSAC plane estimator with EMA smoothing and adaptive Z prior."""

    FIRST_FIT_Z_TOL = 1.0  # relaxed tolerance (m) for initial calibration fit

    def __init__(self, args, T_world_cam, logger=None):
        self.args = args
        self.T_world_cam = T_world_cam
        self.logger = logger
        self.state = WorkbenchPlaneState()
        self._logged_ready = False
        self._last_update_frame = -10 ** 9
        self._z_calibrated = False
        self._z_override = None

    def update(self, depth, K, frame_count, exclude_mask=None):
        """Update plane estimate; returns current (possibly smoothed) state."""
        if not self.args.enable_ransac_workbench_plane or self.T_world_cam is None:
            return self.state

        interval = max(1, int(self.args.workbench_plane_update_interval))
        if frame_count - self._last_update_frame < interval and self.state.valid:
            return self.state
        self._last_update_frame = frame_count

        points = sample_depth_points_world(
            depth, K, self.T_world_cam, self.args, exclude_mask=exclude_mask)

        z_prior = self._z_override if self._z_calibrated else self.args.workbench_z
        z_tol = (self.args.workbench_z_prior_tol_m
                 if self._z_calibrated else self.FIRST_FIT_Z_TOL)

        if len(points) > 0:
            points = _select_workbench_cluster(points, z_prior, z_tol)

        measured = ransac_plane(
            points, self.args, z_prior=z_prior, z_tol=z_tol)

        if measured is None:
            return self.state

        if not self._z_calibrated:
            self._z_override = measured.estimated_z
            self._z_calibrated = True
            if self.logger is not None:
                self.logger.info(
                    f'[workbench_plane] Auto-calibrated Z prior: '
                    f'{measured.estimated_z:.3f} m (tilt={measured.tilt_deg:.1f} deg)')

        if not self.state.valid:
            self.state = measured
        else:
            self._smooth_into(measured)
            self._z_override = self.state.estimated_z

        if self.logger is not None and not self._logged_ready:
            self._logged_ready = True
            self.logger.info(
                f'[workbench_plane] RANSAC ready: z={self.state.estimated_z:.3f} m, '
                f'tilt={self.state.tilt_deg:.1f} deg, inliers={self.state.inlier_count}')
        return self.state

    def _smooth_into(self, measured):
        ema = float(self.args.workbench_plane_ema)
        new_n = measured.normal.astype(np.float64)
        new_d = float(measured.d)
        if np.dot(new_n, self.state.normal) < 0.0:
            new_n = -new_n
            new_d = -new_d

        old_n = self.state.normal
        blended_n = ema * old_n + (1.0 - ema) * new_n
        norm = np.linalg.norm(blended_n)
        if norm < 1e-9:
            blended_n = new_n
        else:
            blended_n = blended_n / norm
        if blended_n[2] < 0.0:
            blended_n = -blended_n

        self.state.normal = blended_n
        self.state.d = ema * self.state.d + (1.0 - ema) * new_d
        self.state.centroid = (
            ema * self.state.centroid + (1.0 - ema) * measured.centroid
        )
        self.state.inlier_count = measured.inlier_count
        self.state.estimated_z = plane_equivalent_z(self.state.normal, self.state.d)
        self.state.tilt_deg = float(
            np.degrees(np.arccos(np.clip(self.state.normal[2], -1.0, 1.0))))
        self.state.valid = True


def signed_plane_distances(points, normal, d):
    """Signed distances n·p - d for Nx3 points."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 3)
    return pts @ normal - d


def is_on_dynamic_plane(aabb_world, plane_state, surface_tol_m, max_height_m):
    """True if AABB sits on the estimated plane within tolerance."""
    if not plane_state.valid:
        return False

    corners = np.array([
        [aabb_world['min'][0], aabb_world['min'][1], aabb_world['min'][2]],
        [aabb_world['max'][0], aabb_world['min'][1], aabb_world['min'][2]],
        [aabb_world['max'][0], aabb_world['max'][1], aabb_world['min'][2]],
        [aabb_world['min'][0], aabb_world['max'][1], aabb_world['min'][2]],
        [aabb_world['min'][0], aabb_world['min'][1], aabb_world['max'][2]],
        [aabb_world['max'][0], aabb_world['min'][1], aabb_world['max'][2]],
        [aabb_world['max'][0], aabb_world['max'][1], aabb_world['max'][2]],
        [aabb_world['min'][0], aabb_world['max'][1], aabb_world['max'][2]],
    ], dtype=np.float64)

    signed = signed_plane_distances(corners, plane_state.normal, plane_state.d)
    bottom_dist = float(np.min(signed))
    height = float(np.max(signed) - np.min(signed))
    on_surface = abs(bottom_dist) <= surface_tol_m
    reasonable_height = height <= max_height_m
    above_plane = float(np.max(signed)) >= -surface_tol_m
    return on_surface and reasonable_height and above_plane
