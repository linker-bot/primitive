"""FoundationStereo TensorRT depth estimation helpers.

This module is optional — it requires:
  1. FoundationStereo's onnx_tensorrt/tensorrt_engine.py (place under FS_DIR or WEIGHTS_BASE)
  2. A pre-built TRT .plan engine file
  3. The `tensorrt` Python package

Set use_stereo_depth:=true and provide stereo_plan_path to enable.
"""
import os

import cv2
import numpy as np

from robot_perception.utils.paths import FS_DIR, WEIGHTS_BASE

STEREO_ENGINE_H = 448
STEREO_ENGINE_W = 672


def _find_tensorrt_engine_module():
    """Locate tensorrt_engine.py from FoundationStereo or weights directory."""
    candidates = [
        os.path.join(FS_DIR, 'onnx_tensorrt', 'tensorrt_engine.py'),
        os.path.join(WEIGHTS_BASE, 'foundation_stereo', 'tensorrt_engine.py'),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_stereo_engine(plan_path):
    """Load a TensorRT .plan/.engine file and return a tensorrt_engine.Engine instance."""
    import importlib.util
    import tensorrt as trt

    engine_module_path = _find_tensorrt_engine_module()
    if engine_module_path is None:
        raise FileNotFoundError(
            'tensorrt_engine.py not found. Place FoundationStereo under '
            f'{FS_DIR} or put tensorrt_engine.py in '
            f'{os.path.join(WEIGHTS_BASE, "foundation_stereo")}/')

    spec = importlib.util.spec_from_file_location('tensorrt_engine', engine_module_path)
    trt_engine_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trt_engine_mod)

    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_data)
    return trt_engine_mod.Engine(engine)


def run_stereo_trt(engine, left_ir, right_ir):
    """Run FoundationStereo TRT inference on a pair of IR images."""
    orig_h, orig_w = left_ir.shape[:2]

    left_resized = cv2.resize(left_ir, (STEREO_ENGINE_W, STEREO_ENGINE_H))
    right_resized = cv2.resize(right_ir, (STEREO_ENGINE_W, STEREO_ENGINE_H))

    if left_resized.ndim == 2:
        left_resized = np.stack([left_resized] * 3, axis=-1)
    if right_resized.ndim == 2:
        right_resized = np.stack([right_resized] * 3, axis=-1)

    left_t = np.ascontiguousarray(left_resized.astype(np.float32).transpose(2, 0, 1)[None])
    right_t = np.ascontiguousarray(right_resized.astype(np.float32).transpose(2, 0, 1)[None])

    results = engine.run([left_t, right_t])
    disp = results[0].reshape(STEREO_ENGINE_H, STEREO_ENGINE_W)
    return disp, (orig_h, orig_w)


def stereo_disp_to_aligned_depth(disp, K_ir_orig, baseline, orig_ir_shape,
                                  K_color, T_ir_to_color, color_shape):
    """Convert stereo disparity to depth aligned to the color camera frame."""
    engine_h, engine_w = disp.shape
    orig_h, orig_w = orig_ir_shape
    Hc, Wc = color_shape[:2]

    scale_x = orig_w / engine_w
    disp_up = cv2.resize(disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    disp_up *= scale_x

    K_ir = K_ir_orig
    depth_ir = np.zeros_like(disp_up)
    valid_disp = disp_up > 0.5
    depth_ir[valid_disp] = K_ir[0, 0] * baseline / disp_up[valid_disp]

    u, v = np.meshgrid(np.arange(orig_w, dtype=np.float32),
                       np.arange(orig_h, dtype=np.float32))
    fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
    cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]

    valid = depth_ir > 0.01
    if not np.any(valid):
        return np.zeros((Hc, Wc), dtype=np.float32)

    zv = depth_ir[valid]
    x_ir = (u[valid] - cx_ir) * zv / fx_ir
    y_ir = (v[valid] - cy_ir) * zv / fy_ir
    pts_ir = np.stack([x_ir, y_ir, zv], axis=1)

    R = T_ir_to_color[:3, :3]
    t = T_ir_to_color[:3, 3]
    pts_c = pts_ir @ R.T + t

    fx_c, fy_c = K_color[0, 0], K_color[1, 1]
    cx_c, cy_c = K_color[0, 2], K_color[1, 2]
    Z = pts_c[:, 2]
    front = Z > 1e-6

    u_c = np.round(pts_c[front, 0] * fx_c / Z[front] + cx_c).astype(np.int32)
    v_c = np.round(pts_c[front, 1] * fy_c / Z[front] + cy_c).astype(np.int32)
    z_c = Z[front].astype(np.float32)

    in_img = (u_c >= 0) & (u_c < Wc) & (v_c >= 0) & (v_c < Hc)
    u_c, v_c, z_c = u_c[in_img], v_c[in_img], z_c[in_img]

    order = np.argsort(-z_c)
    u_c, v_c, z_c = u_c[order], v_c[order], z_c[order]

    aligned = np.zeros((Hc, Wc), dtype=np.float32)
    aligned[v_c, u_c] = z_c

    hole_mask = (aligned == 0)
    if hole_mask.sum() < Hc * Wc * 0.95:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        aligned = cv2.morphologyEx(aligned, cv2.MORPH_CLOSE, kernel)

    return aligned
