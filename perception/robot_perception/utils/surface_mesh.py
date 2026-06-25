"""Hybrid observed-surface mesh: observed top + rim-to-bottom sides + bottom cap.

The mesh is NOT a full AABB shell. Visible top follows segmented depth points;
unseen underside is closed by vertical walls along the observed XY footprint rim
and a bottom cap at table height (optionally expanded to AABB footprint).
"""
import numpy as np

from robot_perception.utils.bbox3d_from_depth import transform_points, aabb_in_frame


def resolve_z_bottom(aabb, plane_state=None):
    """Bottom height: clamp to workbench plane — object bottom never goes below table."""
    z_bottom = float(aabb['min'][2])
    if plane_state is not None and getattr(plane_state, 'valid', False):
        n = np.asarray(plane_state.normal, dtype=np.float64)
        d = float(plane_state.d)
        cx = float(aabb['center'][0])
        cy = float(aabb['center'][1])
        nz = float(n[2])
        if abs(nz) > 0.2:
            z_plane = (d - n[0] * cx - n[1] * cy) / nz
            z_bottom = max(z_bottom, z_plane)
    return z_bottom


def _triangle_normal(p0, p1, p2):
    return np.cross(p1 - p0, p2 - p0)


def _max_edge_len(p0, p1, p2):
    return max(
        np.linalg.norm(p1 - p0),
        np.linalg.norm(p2 - p1),
        np.linalg.norm(p0 - p2),
    )


def _append_triangle(vertices, faces, p0, p1, p2):
    base = len(vertices)
    vertices.extend([np.asarray(p0, dtype=np.float64).tolist(),
                     np.asarray(p1, dtype=np.float64).tolist(),
                     np.asarray(p2, dtype=np.float64).tolist()])
    faces.append([base, base + 1, base + 2])


def _point_in_polygon_xy(point, polygon_xy):
    """Ray casting; polygon_xy is Nx2 ordered."""
    x, y = float(point[0]), float(point[1])
    n = len(polygon_xy)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        if ((yi > y) != (yj > y)) and (
                x < (xj - xi) * (y - yi) / max(yj - yi, 1e-12) + xi):
            inside = not inside
        j = i
    return inside


def _select_top_points(pts, z_bottom, args):
    """Points likely belonging to the visible upper surface."""
    z_vals = pts[:, 2]
    z_obs_max = float(np.max(z_vals))
    height = max(z_obs_max - z_bottom, 1e-4)

    top_frac = float(getattr(args, 'hybrid_surface_top_frac', 0.45))
    top_frac = min(max(top_frac, 0.05), 0.95)
    z_thresh = z_bottom + (1.0 - top_frac) * height
    top_pts = pts[z_vals >= z_thresh]
    if len(top_pts) < 3:
        top_pts = pts

    max_top = int(getattr(args, 'hybrid_surface_max_top_points', 600))
    if len(top_pts) > max_top:
        rng = np.random.default_rng(0)
        top_pts = top_pts[rng.choice(len(top_pts), max_top, replace=False)]
    return top_pts


def _ordered_observed_rim(pts, args):
    """Closed boundary polyline in 3D from observed XY convex hull + local max Z."""
    from scipy.spatial import ConvexHull

    hull = ConvexHull(pts[:, :2])
    hull_xy = pts[hull.vertices, :2]
    span = float(np.max(np.linalg.norm(hull_xy - hull_xy.mean(axis=0), axis=1)))
    search_r = max(
        float(getattr(args, 'hybrid_surface_rim_search_m', 0.012)),
        span * 0.08,
    )

    rim = []
    for vi in hull.vertices:
        xy = pts[vi, :2]
        dists = np.linalg.norm(pts[:, :2] - xy, axis=1)
        nearby = pts[dists <= search_r]
        z_rim = float(nearby[:, 2].max()) if len(nearby) else float(pts[vi, 2])
        rim.append([float(xy[0]), float(xy[1]), z_rim])
    return np.asarray(rim, dtype=np.float64), hull


def _build_top_delaunay(top_pts, pts_all, aabb, args):
    """Upper-surface triangles from observed depth (2.5D Delaunay, clipped to hull)."""
    from scipy.spatial import Delaunay

    vertices = []
    faces = []
    if len(top_pts) < 3:
        return vertices, faces

    _, hull = _ordered_observed_rim(pts_all, args)
    hull_xy = pts_all[hull.vertices, :2]

    max_edge = float(getattr(args, 'hybrid_surface_max_triangle_edge_m', 0.04))
    min_nz = float(getattr(args, 'hybrid_surface_min_triangle_normal_z', 0.15))
    x0, x1 = float(aabb['min'][0]), float(aabb['max'][0])
    y0, y1 = float(aabb['min'][1]), float(aabb['max'][1])
    pad = max_edge * 0.5

    try:
        tri = Delaunay(top_pts[:, :2])
    except Exception:
        return vertices, faces

    for simplex in tri.simplices:
        p0, p1, p2 = top_pts[simplex[0]], top_pts[simplex[1]], top_pts[simplex[2]]
        if _max_edge_len(p0, p1, p2) > max_edge:
            continue
        centroid = (p0 + p1 + p2) / 3.0
        if not (x0 - pad <= centroid[0] <= x1 + pad and y0 - pad <= centroid[1] <= y1 + pad):
            continue
        if not _point_in_polygon_xy(centroid[:2], hull_xy):
            continue
        normal = _triangle_normal(p0, p1, p2)
        norm = np.linalg.norm(normal)
        if norm < 1e-12:
            continue
        if normal[2] / norm < min_nz:
            continue
        _append_triangle(vertices, faces, p0, p1, p2)

    return vertices, faces


def _build_top_rim_cap(rim):
    """Fallback top: fan triangulation on observed rim polygon (uses rim Z)."""
    vertices = []
    faces = []
    n = len(rim)
    if n < 3:
        return vertices, faces
    center = rim.mean(axis=0)
    for i in range(n):
        p0 = rim[i]
        p1 = rim[(i + 1) % n]
        _append_triangle(vertices, faces, center, p0, p1)
    return vertices, faces


def _build_rim_side_walls(rim, z_bottom):
    """Vertical walls: observed top rim down to bottom plane (not AABB side shell)."""
    vertices = []
    faces = []
    n = len(rim)
    if n < 3:
        return vertices, faces

    zb = float(z_bottom)
    for i in range(n):
        p0 = rim[i].astype(np.float64)
        p1 = rim[(i + 1) % n].astype(np.float64)
        z0 = float(p0[2])
        z1 = float(p1[2])
        # Clamp rim to bottom so walls never fold through the table.
        p0_top = np.array([p0[0], p0[1], max(z0, zb)], dtype=np.float64)
        p1_top = np.array([p1[0], p1[1], max(z1, zb)], dtype=np.float64)
        b0 = np.array([p0_top[0], p0_top[1], zb], dtype=np.float64)
        b1 = np.array([p1_top[0], p1_top[1], zb], dtype=np.float64)
        _append_triangle(vertices, faces, p0_top, p1_top, b1)
        _append_triangle(vertices, faces, p0_top, b1, b0)
    return vertices, faces


def _build_bottom_hull_cap(rim, z_bottom):
    """Triangulate observed footprint at bottom height."""
    vertices = []
    faces = []
    n = len(rim)
    if n < 3:
        return vertices, faces

    zb = float(z_bottom)
    bottom = np.array([[p[0], p[1], zb] for p in rim], dtype=np.float64)
    center = bottom.mean(axis=0)
    for i in range(n):
        p0 = bottom[i]
        p1 = bottom[(i + 1) % n]
        _append_triangle(vertices, faces, center, p0, p1)
    return vertices, faces


def _build_aabb_bottom_flange(rim, aabb, z_bottom):
    """Bottom cap: AABB rectangle triangulated with rim footprint on the bottom plane."""
    from scipy.spatial import Delaunay

    vertices = []
    faces = []

    x0, y0 = float(aabb['min'][0]), float(aabb['min'][1])
    x1, y1 = float(aabb['max'][0]), float(aabb['max'][1])
    zb = float(z_bottom)

    rim2d = rim[:, :2]
    corners2d = np.array([
        [x0, y0], [x1, y0], [x1, y1], [x0, y1],
    ], dtype=np.float64)
    all2d = np.vstack([rim2d, corners2d])
    try:
        tri = Delaunay(all2d)
    except Exception:
        b0 = np.array([x0, y0, zb])
        b1 = np.array([x1, y0, zb])
        b2 = np.array([x1, y1, zb])
        b3 = np.array([x0, y1, zb])
        _append_triangle(vertices, faces, b0, b2, b1)
        _append_triangle(vertices, faces, b0, b3, b2)
        return vertices, faces

    for simplex in tri.simplices:
        pts = []
        for idx in simplex:
            x, y = all2d[idx]
            pts.append(np.array([x, y, zb], dtype=np.float64))
        _append_triangle(vertices, faces, pts[0], pts[1], pts[2])

    return vertices, faces


def _merge_mesh_parts(parts):
    vertices = []
    faces = []
    for part_v, part_f in parts:
        if not part_f:
            continue
        offset = len(vertices)
        vertices.extend(part_v)
        for f in part_f:
            faces.append([f[0] + offset, f[1] + offset, f[2] + offset])
    if not faces:
        return None
    return {
        'vertices': np.asarray(vertices, dtype=np.float64),
        'faces': np.asarray(faces, dtype=np.int32),
    }


def build_hybrid_surface_mesh(pts_world, aabb_world, plane_state=None, args=None):
    """Observed top + rim side walls + bottom cap (stitched, not a full AABB shell)."""
    if args is None or not getattr(args, 'enable_hybrid_surface_mesh', True):
        return None
    if aabb_world is None:
        return None

    min_pts = int(getattr(args, 'hybrid_surface_min_points', 30))
    pts = np.asarray(pts_world, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) < min_pts:
        return None

    if getattr(args, 'hybrid_surface_outlier_filter', True):
        from robot_perception.utils.point_filter import filter_point_cloud
        pts = filter_point_cloud(
            pts,
            iqr_k=float(getattr(args, 'hybrid_surface_iqr_k', 1.5)),
            sor_k=int(getattr(args, 'hybrid_surface_sor_k', 8)),
            sor_std=float(getattr(args, 'hybrid_surface_sor_std', 1.5)),
            min_points=min_pts,
        )
        if len(pts) < min_pts:
            return None

    z_bottom = resolve_z_bottom(aabb_world, plane_state)
    top_pts = _select_top_points(pts, z_bottom, args)
    rim, _ = _ordered_observed_rim(pts, args)

    top_v, top_f = _build_top_delaunay(top_pts, pts, aabb_world, args)
    if not top_f:
        top_v, top_f = _build_top_rim_cap(rim)

    side_v, side_f = _build_rim_side_walls(rim, z_bottom)

    use_aabb_bottom = bool(getattr(args, 'hybrid_surface_aabb_bottom', False))
    if use_aabb_bottom:
        bottom_v, bottom_f = _build_aabb_bottom_flange(rim, aabb_world, z_bottom)
    else:
        bottom_v, bottom_f = _build_bottom_hull_cap(rim, z_bottom)

    return _merge_mesh_parts([
        (top_v, top_f),
        (side_v, side_f),
        (bottom_v, bottom_f),
    ])


def build_hybrid_surface_from_cam(pts_cam, aabb_work, T_world_cam, plane_state=None,
                                  args=None, aabb_work_fallback=None):
    """Transform camera points to world and build hybrid surface mesh."""
    if pts_cam is None or T_world_cam is None:
        return None
    if len(pts_cam) == 0:
        return None
    pts_world = transform_points(T_world_cam, pts_cam)
    aabb = aabb_work if aabb_work is not None else aabb_work_fallback
    if aabb is None:
        aabb = aabb_in_frame(pts_world, T_frame_cam=None)
    return build_hybrid_surface_mesh(pts_world, aabb, plane_state, args)


def attach_hybrid_surface(result, T_world_cam, args, plane_state=None):
    """Compute and store hybrid surface mesh on a detection result dict."""
    if not getattr(args, 'enable_hybrid_surface_mesh', True):
        result['surface_mesh'] = None
        return
    if result.get('aabb_work') is None:
        result['surface_mesh'] = None
        return
    min_pts = int(getattr(args, 'hybrid_surface_min_points', 0))
    if min_pts <= 0:
        min_pts = int(getattr(args, 'min_depth_points', 50))
    pts = result.get('pts_cam')
    if pts is None or len(pts) < min_pts:
        result['surface_mesh'] = None
        return
    if plane_state is None:
        plane_state = getattr(args, 'workbench_plane_state', None)
    mesh = build_hybrid_surface_from_cam(
        pts,
        result.get('aabb_work'),
        T_world_cam,
        plane_state=plane_state,
        args=args,
    )
    result['surface_mesh'] = mesh
