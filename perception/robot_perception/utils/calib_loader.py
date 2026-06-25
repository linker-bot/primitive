"""Load camera-to-world calibration (T_world_cam) from npz or package txt."""
import os
import re

import numpy as np


def default_calib_file():
    """Prefer bundled package example calibration (replace for production)."""
    try:
        from ament_index_python.packages import get_package_share_directory
        pkg_share = get_package_share_directory('robot_perception')
        for name in ('full_calibration_result.npz', 'full_calibration_result.txt'):
            path = os.path.join(pkg_share, 'config', 'calib_results', name)
            if os.path.isfile(path):
                return path
    except Exception:
        pass
    env = os.environ.get('ROBOT_CALIB_FILE', '').strip()
    if env and os.path.isfile(os.path.expanduser(env)):
        return os.path.abspath(os.path.expanduser(env))
    return ''


def _parse_T_world_cam_txt(path):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    match = re.search(
        r'T_world_cam[^\[]*(\[\[.*?\]\])',
        text,
        re.DOTALL,
    )
    if not match:
        raise ValueError(f'T_world_cam matrix not found in {path}')

    nums = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', match.group(1))
    if len(nums) < 16:
        raise ValueError(f'Invalid T_world_cam matrix in {path}')
    T = np.array([float(x) for x in nums[:16]], dtype=np.float64).reshape(4, 4)
    return T


def load_T_world_cam(calib_file):
    """Load 4x4 T_world_cam from .npz or .txt. Returns None if unavailable."""
    if not calib_file or not os.path.isfile(calib_file):
        return None

    ext = os.path.splitext(calib_file)[1].lower()
    if ext == '.npz':
        data = np.load(calib_file, allow_pickle=True)
        if 'T_world_cam' not in data:
            raise ValueError(f'T_world_cam missing in {calib_file}')
        return data['T_world_cam'].astype(np.float64)

    if ext == '.txt':
        return _parse_T_world_cam_txt(calib_file)

    raise ValueError(f'Unsupported calib_file format: {calib_file}')
