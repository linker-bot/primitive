"""Compute 2D/3D bounding boxes, PCA orientation, and grasp hints from depth."""
import cv2
import numpy as np
from scipy.spatial.transform import Rotation


def refine_instance_mask(mask, min_pixels=50, open_kernel=3):
    """Keep the largest connected component; drop speckle that inflates 2D bbox.

    Cutie often leaves scattered pixels on the table/workbench.  Those pixels can
    be <2% of the image (mask_ratio looks fine) while the axis-aligned bbox spans
    most of the desk.  Morphological open + largest-component fixes that.
    """
    raw = (np.asarray(mask) > 0).astype(np.uint8)
    if raw.sum() == 0:
        return raw
    m = raw
    if open_kernel > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_kernel, open_kernel))
        opened = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        if opened.sum() >= max(10, min_pixels // 2):
            m = opened
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return m
    idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[idx, cv2.CC_STAT_AREA] < min_pixels:
        return m
    return (labels == idx).astype(np.uint8)


def bbox_mask_fill_ratio(bbox_xyxy, mask) -> float:
    """Share of bbox area covered by mask pixels (low => sparse scatter)."""
    if bbox_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    area = max(1.0, (x2 - x1 + 1.0) * (y2 - y1 + 1.0))
    return float((np.asarray(mask) > 0).sum()) / area


def bbox_area_xyxy(bbox_xyxy) -> float:
    if bbox_xyxy is None:
        return 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    return max(0.0, (x2 - x1 + 1.0) * (y2 - y1 + 1.0))


def bbox_size_fractions(bbox_xyxy, img_w, img_h):
    """Return (area_ratio, width_ratio, height_ratio) relative to image size."""
    if bbox_xyxy is None or img_w <= 0 or img_h <= 0:
        return 0.0, 0.0, 0.0
    x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
    bw = max(0.0, x2 - x1 + 1.0)
    bh = max(0.0, y2 - y1 + 1.0)
    img_area = float(max(1, int(img_w) * int(img_h)))
    return (bw * bh) / img_area, bw / float(img_w), bh / float(img_h)


def compute_result_geometry_stats(results, img_h, img_w):
    """Geometry stats for each published detection (mask vs 2D bbox vs image)."""
    img_area = max(1, int(img_h) * int(img_w))
    stats = []
    for r in results:
        bbox = r.get('bbox2d')
        mask = r.get('mask')
        mask_ratio = 0.0
        if mask is not None:
            mask_ratio = float((np.asarray(mask) > 0).sum()) / img_area
        bbox_ratio, bbox_w_frac, bbox_h_frac = bbox_size_fractions(
            bbox, img_w, img_h)
        fill = bbox_mask_fill_ratio(bbox, mask) if mask is not None else 0.0
        stats.append({
            'label': str(r.get('tag', '')),
            'instance_id': int(r.get('instance_id', 0)),
            'track_mode': str(r.get('track_mode', '')),
            'mask_ratio': mask_ratio,
            'bbox_ratio': bbox_ratio,
            'fill': fill,
            'bbox_w_pct': bbox_w_frac * 100.0,
            'bbox_h_pct': bbox_h_frac * 100.0,
        })
    return stats


def format_geometry_stats_log(stats):
    """Human-readable: label#id mask=M% box=B% fill=F% size=WxH%."""
    parts = []
    for s in stats:
        mode = s.get('track_mode') or '-'
        parts.append(
            f"{s['label']}#{s['instance_id']}({mode}):"
            f"mask={s['mask_ratio'] * 100:.1f}%:"
            f"box={s['bbox_ratio'] * 100:.1f}%:"
            f"fill={s['fill'] * 100:.0f}%:"
            f"size={s['bbox_w_pct']:.0f}x{s['bbox_h_pct']:.0f}%"
        )
    return ', '.join(parts)


def format_publish_draw_log(results, img_h, img_w):
    """Match annotated overlay: instance → viz color → label → bbox pixels."""
    if not results or img_h <= 0 or img_w <= 0:
        return ''
    parts = []
    for r in results:
        bbox = r.get('bbox2d')
        mask = r.get('mask')
        if bbox is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        bbox_ratio, bw, bh = bbox_size_fractions(bbox, img_w, img_h)
        mask_ratio = 0.0
        if mask is not None:
            mask_ratio = float((np.asarray(mask) > 0).sum()) / max(
                1, int(img_h) * int(img_w))
        prompt = str(r.get('prompt', '') or '')
        if len(prompt) > 24:
            prompt = prompt[:21] + '...'
        sam_pct = float(r.get('sam_mask_ratio', 0) or 0) * 100.0
        sam_part = f':sam_raw={sam_pct:.1f}%' if sam_pct > 0.1 else ''
        parts.append(
            f"#{r.get('instance_id', 0)}:{r.get('color_name', '?')}:"
            f"{r.get('tag', '')}:mode={r.get('track_mode', '')}:"
            f"prompt={prompt}:score={float(r.get('score', 0)):.2f}:"
            f"box={bbox_ratio * 100:.1f}%:mask={mask_ratio * 100:.1f}%:"
            f"xyxy=[{x1},{y1},{x2},{y2}]:size={bw * 100:.0f}x{bh * 100:.0f}%"
            f"{sam_part}"
        )
    return ', '.join(parts)


def mask_to_bbox_xyxy(mask, refine=True, min_component_pixels=50, open_kernel=3):
    """Extract axis-aligned 2D bbox [x1,y1,x2,y2] from a binary mask."""
    m = (
        refine_instance_mask(
            mask, min_pixels=min_component_pixels, open_kernel=open_kernel)
        if refine else (np.asarray(mask) > 0).astype(np.uint8)
    )
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)


def backproject_mask(depth, K, mask, depth_min=0.01, depth_max=3.0):
    """Back-project valid depth pixels inside mask to 3D camera-frame points."""
    h, w = depth.shape[:2]
    if mask.shape[:2] != (h, w):
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    valid = (mask > 0) & (depth > depth_min) & (depth < depth_max) & np.isfinite(depth)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    v, u = np.where(valid)
    z = depth[v, u].astype(np.float32)
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def aabb_from_points(pts):
    """Compute center, size, min, max for an axis-aligned 3D box."""
    if pts is None or len(pts) == 0:
        return None
    pmin = pts.min(axis=0)
    pmax = pts.max(axis=0)
    center = (pmin + pmax) * 0.5
    size = pmax - pmin
    return {
        'center': center.astype(np.float32),
        'size': size.astype(np.float32),
        'min': pmin.astype(np.float32),
        'max': pmax.astype(np.float32),
    }


def transform_points(T, pts):
    """Apply 4x4 transform to Nx3 points."""
    if len(pts) == 0:
        return pts
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts.astype(np.float64), ones])
    return (T @ pts_h.T).T[:, :3].astype(np.float32)


def aabb_in_frame(pts_cam, T_frame_cam=None):
    """Compute AABB in camera frame or after applying T_frame_cam (frame_from_cam)."""
    if T_frame_cam is not None:
        pts = transform_points(T_frame_cam, pts_cam)
    else:
        pts = pts_cam
    return aabb_from_points(pts)


def draw_aabb_projection(img, K, aabb_cam, color=(0, 255, 0), thickness=2):
    """Draw 3D AABB edges projected onto an RGB image (in-place)."""
    import cv2

    if aabb_cam is None:
        return img

    pmin = aabb_cam['min']
    pmax = aabb_cam['max']
    corners = np.array([
        [pmin[0], pmin[1], pmin[2]],
        [pmax[0], pmin[1], pmin[2]],
        [pmax[0], pmax[1], pmin[2]],
        [pmin[0], pmax[1], pmin[2]],
        [pmin[0], pmin[1], pmax[2]],
        [pmax[0], pmin[1], pmax[2]],
        [pmax[0], pmax[1], pmax[2]],
        [pmin[0], pmax[1], pmax[2]],
    ], dtype=np.float64)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = corners[:, 2]
    if np.any(z <= 1e-6):
        return img
    u = (corners[:, 0] * fx / z + cx).astype(np.int32)
    v = (corners[:, 1] * fy / z + cy).astype(np.int32)
    uv = np.stack([u, v], axis=1)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for i, j in edges:
        cv2.line(img, tuple(uv[i]), tuple(uv[j]), color, thickness, cv2.LINE_AA)
    return img


def aabb_projected_bbox_xyxy(aabb_cam, K):
    """Axis-aligned 2D bbox enclosing the projected 3D AABB corners (may be huge)."""
    if aabb_cam is None:
        return None
    pmin = aabb_cam['min']
    pmax = aabb_cam['max']
    corners = np.array([
        [pmin[0], pmin[1], pmin[2]],
        [pmax[0], pmin[1], pmin[2]],
        [pmax[0], pmax[1], pmin[2]],
        [pmin[0], pmax[1], pmin[2]],
        [pmin[0], pmin[1], pmax[2]],
        [pmax[0], pmin[1], pmax[2]],
        [pmax[0], pmax[1], pmax[2]],
        [pmin[0], pmax[1], pmax[2]],
    ], dtype=np.float64)
    z = corners[:, 2]
    if np.all(z <= 1e-6):
        return None
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    u = corners[:, 0] * fx / np.maximum(z, 1e-6) + cx
    v = corners[:, 1] * fy / np.maximum(z, 1e-6) + cy
    return np.array([u.min(), v.min(), u.max(), v.max()], dtype=np.float32)


def pca_orientation(pts, min_pts=10):
    """Compute principal-axis orientation quaternion (x,y,z,w) from Nx3 points.

    Returns identity quaternion if insufficient points.
    The axes are ordered so that the longest spread maps to X, middle to Y, shortest to Z.
    """
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    if pts is None or len(pts) < min_pts:
        return identity

    pts64 = np.asarray(pts, dtype=np.float64)
    centroid = pts64.mean(axis=0)
    centered = pts64 - centroid
    cov = (centered.T @ centered) / max(len(pts64) - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)

    order = np.argsort(eigvals)[::-1]
    axes = eigvecs[:, order].T

    R = np.stack(axes, axis=0)
    if np.linalg.det(R) < 0:
        R[2] = -R[2]

    q = Rotation.from_matrix(R).as_quat().astype(np.float32)
    return q


def compute_top_normal(pts, top_frac=0.3, min_pts=10):
    """Estimate average outward normal of the top surface (highest Z fraction).

    Returns unit normal pointing away from object center, or [0,0,1] if insufficient data.
    """
    default = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if pts is None or len(pts) < min_pts:
        return default

    pts64 = np.asarray(pts, dtype=np.float64)
    z_vals = pts64[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    height = z_max - z_min
    if height < 1e-4:
        return default

    z_thresh = z_max - top_frac * height
    top_pts = pts64[z_vals >= z_thresh]
    if len(top_pts) < 3:
        return default

    centroid = top_pts.mean(axis=0)
    centered = top_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float64)

    obj_center = pts64.mean(axis=0)
    if np.dot(normal, centroid - obj_center) < 0:
        normal = -normal

    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return default
    return (normal / norm).astype(np.float32)


def classify_grasp_type(size):
    """Classify grasp affordance from bounding box size [sx, sy, sz].

    Returns: 'precision' | 'lateral' | 'power' | ''
    """
    if size is None:
        return ''
    s = np.asarray(size, dtype=np.float64)
    max_dim = float(s.max())
    min_dim = float(s.min())

    if max_dim < 0.03:
        return 'precision'
    if max_dim < 0.08:
        if min_dim < 0.025:
            return 'lateral'
        return 'lateral'
    return 'power'
