"""IoU helpers and detection-to-track association."""
import numpy as np


def bbox_iou(a, b):
    """IoU for xyxy boxes."""
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 1e-9 else 0.0


def mask_iou(a, b):
    a = a.astype(bool)
    b = b.astype(bool)
    inter = np.logical_and(a, b).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union)


def bbox_center(box):
    """BBox xyxy center."""
    return np.array([
        (float(box[0]) + float(box[2])) * 0.5,
        (float(box[1]) + float(box[3])) * 0.5,
    ], dtype=np.float32)


def associate_detections_to_tracks(
        det_bboxes, track_bboxes, iou_min=0.3, compat_matrix=None,
        prefer_center_for_same_label=False, det_labels=None, track_labels=None):
    """Greedy IoU matching. Returns (det_idx->track_idx, unmatched_det, unmatched_trk).

    compat_matrix: optional bool (n_d, n_t); False entries are never matched.
    When prefer_center_for_same_label is True and labels are provided, ties among
    valid IoU matches prefer smaller bbox-center distance (helps duplicate labels).
    """
    n_d = len(det_bboxes)
    n_t = len(track_bboxes)
    if n_d == 0 or n_t == 0:
        return {}, list(range(n_d)), list(range(n_t))

    use_center = (
        prefer_center_for_same_label
        and det_labels is not None
        and track_labels is not None
        and len(det_labels) == n_d
        and len(track_labels) == n_t
    )
    if use_center:
        pairs = []
        for i in range(n_d):
            dc = bbox_center(det_bboxes[i])
            for j in range(n_t):
                if compat_matrix is not None and not compat_matrix[i, j]:
                    continue
                iou = bbox_iou(det_bboxes[i], track_bboxes[j])
                if iou < iou_min:
                    continue
                tc = bbox_center(track_bboxes[j])
                dist = float(np.linalg.norm(dc - tc))
                same_label = det_labels[i] == track_labels[j]
                pairs.append((iou, -dist if same_label else 0.0, i, j))
        pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        matched = {}
        used_d, used_t = set(), set()
        for iou, _prio, i, j in pairs:
            if i in used_d or j in used_t:
                continue
            matched[i] = j
            used_d.add(i)
            used_t.add(j)
        unmatched_d = [i for i in range(n_d) if i not in matched]
        unmatched_t = [j for j in range(n_t) if j not in matched]
        return matched, unmatched_d, unmatched_t

    iou_mat = np.zeros((n_d, n_t), dtype=np.float32)
    for i in range(n_d):
        for j in range(n_t):
            if compat_matrix is not None and not compat_matrix[i, j]:
                iou_mat[i, j] = -1.0
                continue
            iou_mat[i, j] = bbox_iou(det_bboxes[i], track_bboxes[j])

    matched = {}
    used_d, used_t = set(), set()
    while True:
        i, j = np.unravel_index(np.argmax(iou_mat), iou_mat.shape)
        if iou_mat[i, j] < iou_min:
            break
        if i in used_d or j in used_t:
            iou_mat[i, j] = -1.0
            continue
        matched[i] = j
        used_d.add(i)
        used_t.add(j)
        iou_mat[i, :] = -1.0
        iou_mat[:, j] = -1.0

    unmatched_d = [i for i in range(n_d) if i not in matched]
    unmatched_t = [j for j in range(n_t) if j not in matched]
    return matched, unmatched_d, unmatched_t
