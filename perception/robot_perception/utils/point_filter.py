"""点云离群点剔除：IQR + Statistical Outlier Removal (SOR)。"""
import numpy as np


def filter_iqr(points, k=1.5):
    """基于 IQR 的逐轴离群点剔除。

    对每个坐标轴计算 Q1, Q3, IQR = Q3-Q1，
    保留 [Q1 - k*IQR, Q3 + k*IQR] 范围内的点。
    """
    if len(points) < 4:
        return points
    mask = np.ones(len(points), dtype=bool)
    for axis in range(3):
        vals = points[:, axis]
        q1 = np.percentile(vals, 25)
        q3 = np.percentile(vals, 75)
        iqr = q3 - q1
        if iqr < 1e-6:
            continue
        lo = q1 - k * iqr
        hi = q3 + k * iqr
        mask &= (vals >= lo) & (vals <= hi)
    filtered = points[mask]
    if len(filtered) < 4:
        return points
    return filtered


def filter_sor(points, k=8, std_ratio=1.5):
    """Statistical Outlier Removal — 基于 k 近邻平均距离。

    计算每个点到 k 个最近邻的平均距离，
    剔除平均距离超过 μ + std_ratio * σ 的点。
    """
    from scipy.spatial import cKDTree

    n = len(points)
    if n <= k + 1:
        return points
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=k + 1)
    mean_dists = dists[:, 1:].mean(axis=1)

    mu = float(mean_dists.mean())
    sigma = float(mean_dists.std())
    if sigma < 1e-9:
        return points
    threshold = mu + std_ratio * sigma

    mask = mean_dists <= threshold
    filtered = points[mask]
    if len(filtered) < 4:
        return points
    return filtered


def filter_point_cloud(points, iqr_k=1.5, sor_k=8, sor_std=1.5, min_points=10):
    """两级过滤入口：先 IQR 后 SOR。

    Args:
        points: Nx3 numpy array
        iqr_k: IQR 系数，默认 1.5
        sor_k: SOR 近邻数，默认 8
        sor_std: SOR 标准差倍数，默认 1.5
        min_points: 过滤后最少保留点数，不足则回退
    """
    if points is None or len(points) < min_points:
        return points
    pts = filter_iqr(points, k=iqr_k)
    if len(pts) < min_points:
        return points
    pts = filter_sor(pts, k=sor_k, std_ratio=sor_std)
    if len(pts) < min_points:
        return points
    return pts
