"""Async Pixel3D mesh completion for tracked objects.

Manages a background worker that runs Pixel3D inference on cropped object images
and produces full 3D meshes aligned to world coordinates.

架构：
- L1 内存缓存 (_mesh_cache): 按 track_id 持久保存，每帧可取
- L2 Label 缓存 (_label_cache): 同类物体共享归一化 mesh
- L3 磁盘缓存: 跨会话持久化，节点重启后可加载
"""
import json
import math
import os
import sys
import threading
import queue
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from robot_perception.utils.paths import PIXEL3D_MODEL_PATH, resolve_weights_path


def _ensure_pixel3d_attn_backend(logger=None):
    """Pick attention backend before any pixal3d import (config reads env at import time)."""
    if os.environ.get('ATTN_BACKEND'):
        return
    try:
        import flash_attn  # noqa: F401
    except ImportError:
        os.environ['ATTN_BACKEND'] = 'sdpa'
        if logger:
            logger.warn(
                '[Pixel3D] flash_attn not installed, falling back to ATTN_BACKEND=sdpa')


@dataclass
class Pixel3DRequest:
    track_id: int
    label: str
    crop_rgba: object  # PIL Image RGBA
    fov_rad: float
    aabb_world: dict
    T_world_cam: np.ndarray


@dataclass
class Pixel3DResult:
    track_id: int
    mesh: Optional[dict] = None            # 已对齐到 aabb 的 mesh
    normalized_mesh: Optional[dict] = None  # 坐标变换后、未对齐的归一化 mesh
    error: Optional[str] = None


def fov_from_intrinsics(K, width):
    """Compute horizontal FOV in radians from intrinsics matrix."""
    fx = float(K[0, 0])
    return 2.0 * math.atan(width / (2.0 * fx))


def prepare_crop(color_rgb, mask, bbox2d, padding=20):
    """Crop object from RGB image using mask, return RGBA PIL Image.

    Args:
        color_rgb: HxW x3 uint8 numpy array (RGB)
        mask: HxW bool numpy array
        bbox2d: [x1, y1, x2, y2] pixel coords
        padding: extra pixels around bbox
    """
    from PIL import Image

    h, w = color_rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox2d]
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)

    crop = color_rgb[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]

    rgba = np.zeros((y2 - y1, x2 - x1, 4), dtype=np.uint8)
    rgba[:, :, :3] = crop
    rgba[:, :, 3] = (mask_crop * 255).astype(np.uint8)

    return Image.fromarray(rgba, mode='RGBA')


def align_mesh_to_world(vertices, aabb_world):
    """将归一化 mesh 按 per-axis scale 对齐到世界 AABB。"""
    aabb_min = np.asarray(aabb_world['min'], dtype=np.float64)
    aabb_max = np.asarray(aabb_world['max'], dtype=np.float64)
    aabb_size = aabb_max - aabb_min
    aabb_center = (aabb_min + aabb_max) / 2.0

    verts = np.asarray(vertices, dtype=np.float64)
    mesh_min = verts.min(axis=0)
    mesh_max = verts.max(axis=0)
    mesh_size = mesh_max - mesh_min
    mesh_center = (mesh_min + mesh_max) / 2.0

    scale = np.where(mesh_size > 1e-6, aabb_size / mesh_size, 1.0)
    verts = (verts - mesh_center) * scale + aabb_center
    return verts


class Pixel3DManager:
    """Manages async Pixel3D inference with multi-level caching."""

    def __init__(self, args, logger=None):
        self._args = args
        self._logger = logger
        self._pipeline = None
        self._request_queue = queue.Queue(maxsize=16)
        self._pending = set()
        self._completed = set()
        self._lock = threading.Lock()
        self._workers = []
        self._shutdown = False

        # L1: track_id → aligned mesh (每帧可取)
        self._mesh_cache = {}
        # L2: label → normalized mesh (跨 track 共享)
        self._label_cache = {}
        self._pipeline_lock = threading.Lock()
        # L3: 磁盘缓存目录
        self._cache_dir = str(getattr(args, 'pixel3d_cache_dir', '') or '')
        self._use_label_cache = bool(getattr(args, 'pixel3d_use_label_cache', True))

        self._load_all_disk_cache()

        max_workers = getattr(args, 'pixel3d_max_concurrent', 1)
        if max_workers <= 0:
            if self._logger:
                self._logger.warning(
                    '[Pixel3D] pixel3d_max_concurrent=0, no workers — inference disabled')
            max_workers = 0
        for _ in range(max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._workers.append(t)

        if self._logger:
            self._logger.info(
                f'[Pixel3D] Manager started, workers={max_workers}, '
                f'label_cache={self._use_label_cache}, '
                f'disk_cache={"ON" if self._cache_dir else "OFF"}')

    # --- Pipeline ---

    def _resolve_offline_config_path(self):
        path = str(getattr(self._args, 'pixel3d_offline_config', '') or '').strip()
        if path:
            return path
        try:
            from ament_index_python.packages import get_package_share_directory
            default = os.path.join(
                get_package_share_directory('robot_perception'),
                'config', 'pixel3d', 'pixel3d_offline.example.json')
            if os.path.isfile(default):
                return default
        except Exception:
            pass
        return ''

    def _load_offline_config(self):
        path = self._resolve_offline_config_path()
        if not path:
            return None
        try:
            with open(path, encoding='utf-8') as f:
                cfg = json.load(f)
            if self._logger:
                self._logger.info(f'[Pixel3D] Loaded offline config: {path}')
            return cfg
        except Exception as e:
            if self._logger:
                self._logger.warning(
                    f'[Pixel3D] Failed to load offline config {path}: {e}')
            return None

    def _build_image_cond_models(self, pipeline, offline_cfg, low_vram):
        """Build 4-stage DinoV3Proj models (required by Pixal3DImageTo3DPipeline.run)."""
        import torch
        from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
            DinoV3ProjFeatureExtractor,
        )

        image_cond = (offline_cfg or {}).get('image_cond_configs')
        if not image_cond:
            raise RuntimeError(
                'Pixel3D image_cond_models not configured — set pixel3d_offline_config '
                'or install config/pixel3d/pixel3d_offline.example.json')

        def _resolve_stage_cfg(stage_cfg):
            out = dict(stage_cfg)
            if 'model_name' in out:
                out['model_name'] = resolve_weights_path(out['model_name'])
            return out

        def _build(stage_cfg):
            model = DinoV3ProjFeatureExtractor(**_resolve_stage_cfg(stage_cfg))
            model.eval()
            return model

        pipeline.image_cond_model_ss = _build(image_cond['ss'])
        pipeline.image_cond_model_shape_512 = _build(image_cond['shape_512'])
        pipeline.image_cond_model_shape_1024 = _build(image_cond['shape_1024'])
        pipeline.image_cond_model_tex_1024 = _build(image_cond['tex_1024'])

        if low_vram:
            for attr in ('image_cond_model_ss', 'image_cond_model_shape_512',
                         'image_cond_model_shape_1024', 'image_cond_model_tex_1024'):
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, 'use_naf_upsample', False):
                    m._load_naf()
        else:
            pipeline.image_cond_model_ss.cuda()
            pipeline.image_cond_model_shape_512.cuda()
            pipeline.image_cond_model_shape_1024.cuda()
            pipeline.image_cond_model_tex_1024.cuda()
            for attr in ('image_cond_model_ss', 'image_cond_model_shape_512',
                         'image_cond_model_shape_1024', 'image_cond_model_tex_1024'):
                m = getattr(pipeline, attr, None)
                if m is not None and getattr(m, 'use_naf_upsample', False):
                    m._load_naf()

    def _init_pipeline(self):
        """Lazy-load Pixel3D pipeline on first use (in worker thread)."""
        if self._pipeline is not None:
            return

        with self._pipeline_lock:
            if self._pipeline is not None:
                return

            _ensure_pixel3d_attn_backend(self._logger)

            from robot_perception.utils.paths import PIXAL3D_DIR
            if os.path.isdir(PIXAL3D_DIR):
                if PIXAL3D_DIR not in sys.path:
                    sys.path.insert(0, PIXAL3D_DIR)
            elif self._logger:
                self._logger.warning(
                    f'[Pixel3D] PIXAL3D_DIR not found: {PIXAL3D_DIR} — '
                    f'set ROBOT_PERCEPTION_DIR env or check third_party/robot_perception/Pixal3D')

            import torch
            from pixal3d.pipelines import Pixal3DImageTo3DPipeline

            offline_cfg = self._load_offline_config()
            paths = (offline_cfg or {}).get('paths') or {}
            model_path = getattr(self._args, 'pixel3d_model_path',
                                 PIXEL3D_MODEL_PATH)
            if paths.get('pixal3d_t'):
                model_path = paths['pixal3d_t']
            low_vram = getattr(self._args, 'pixel3d_low_vram', True)

            if self._logger:
                self._logger.info(f'[Pixel3D] Loading pipeline from {model_path} '
                                  f'(low_vram={low_vram})...')

            pipeline = Pixal3DImageTo3DPipeline.from_pretrained(model_path)
            self._build_image_cond_models(pipeline, offline_cfg, low_vram)

            if low_vram:
                pipeline._device = torch.device('cuda')
                pipeline.low_vram = True
            else:
                pipeline.low_vram = False
                pipeline.cuda()

            self._pipeline = pipeline
            if self._logger:
                self._logger.info('[Pixel3D] Pipeline loaded.')

    # --- Worker ---

    def _worker_loop(self):
        """Background worker that processes Pixel3D requests."""
        while not self._shutdown:
            try:
                req = self._request_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            result = self._run_inference(req)
            with self._lock:
                self._pending.discard(req.track_id)
                self._completed.add(req.track_id)
                if result.mesh is not None:
                    self._mesh_cache[req.track_id] = result.mesh
                    if self._use_label_cache and req.label and result.normalized_mesh:
                        self._label_cache[req.label] = result.normalized_mesh
            if result.normalized_mesh and req.label:
                self._save_disk_cache(req.label, result.normalized_mesh)

    def _run_inference(self, req: Pixel3DRequest) -> Pixel3DResult:
        """Run Pixel3D inference for a single request."""
        try:
            import torch

            self._init_pipeline()

            # 释放其他模型推理残留的 CUDA 碎片，为 Pixel3D 腾出显存
            torch.cuda.empty_cache()

            image = req.crop_rgba
            fov = req.fov_rad

            image_preprocessed = self._pipeline.preprocess_image(image)

            grid_point = torch.tensor([-1.0, 0.0, 0.0])
            image_resolution = 512
            mesh_scale = 1.0
            extend_pixel = 0

            focal_length = 16.0 / torch.tan(torch.tensor(fov / 2.0))
            f_pixels = float((focal_length * image_resolution / 32.0).item())
            x_ndc = 0 - extend_pixel - image_resolution / 2.0
            rotation_matrix = torch.tensor(
                [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
            gp = grid_point.to(torch.float32) @ rotation_matrix.T
            gp = gp / mesh_scale / 2
            xw = gp[0].item()
            distance = f_pixels * xw / x_ndc - gp[1].item()

            camera_params = {
                'camera_angle_x': fov,
                'distance': distance,
                'mesh_scale': mesh_scale,
            }

            torch.manual_seed(42)
            mesh_list, _ = self._pipeline.run(
                image_preprocessed,
                camera_params=camera_params,
                seed=42,
                preprocess_image=False,
                return_latent=True,
                pipeline_type='1024_cascade',
            )

            mesh = mesh_list[0]
            vertices = np.asarray(mesh.vertices, dtype=np.float64)
            faces = np.asarray(mesh.faces, dtype=np.int32)

            rot = np.array([
                [-1, 0, 0],
                [0, 0, -1],
                [0, -1, 0],
            ], dtype=np.float64)
            vertices = vertices @ rot.T

            normalized_mesh = {
                'vertices': vertices.copy(),
                'faces': faces,
            }

            aligned_vertices = align_mesh_to_world(vertices, req.aabb_world)

            if self._logger:
                self._logger.info(
                    f'[Pixel3D] Inference done for track {req.track_id} '
                    f'(label={req.label}, verts={len(aligned_vertices)}, '
                    f'faces={len(faces)})')

            return Pixel3DResult(
                track_id=req.track_id,
                mesh={'vertices': aligned_vertices, 'faces': faces},
                normalized_mesh=normalized_mesh,
            )

        except Exception as e:
            if self._logger:
                self._logger.warning(f'[Pixel3D] Inference failed for track '
                                     f'{req.track_id}: {e}')
            return Pixel3DResult(track_id=req.track_id, error=str(e))

    # --- Public API ---

    def submit(self, track_id, label, crop_rgba, K, aabb_world, T_world_cam,
               img_width):
        """Submit a Pixel3D inference request."""
        with self._lock:
            if track_id in self._pending or track_id in self._completed:
                return False

        fov = fov_from_intrinsics(K, img_width)
        req = Pixel3DRequest(
            track_id=track_id,
            label=label,
            crop_rgba=crop_rgba,
            fov_rad=fov,
            aabb_world=aabb_world,
            T_world_cam=T_world_cam,
        )

        with self._lock:
            self._pending.add(track_id)

        try:
            self._request_queue.put_nowait(req)
        except queue.Full:
            with self._lock:
                self._pending.discard(track_id)
            return False
        return True

    def get_mesh(self, track_id):
        """获取 track_id 的缓存 mesh（持久保留，每帧可取）。"""
        with self._lock:
            return self._mesh_cache.get(track_id)

    def try_cached_mesh(self, track_id, label, aabb_world):
        """尝试从缓存获取 mesh 并对齐到当前 aabb。

        优先级: L1 track缓存 → L2 label缓存 → L3 磁盘缓存。
        返回 mesh dict 或 None。
        """
        with self._lock:
            cached = self._mesh_cache.get(track_id)
            if cached is not None:
                return cached

        if not self._use_label_cache or not label:
            return None

        with self._lock:
            normalized = self._label_cache.get(label)
        if normalized is None:
            return None

        aligned_verts = align_mesh_to_world(
            normalized['vertices'].copy(), aabb_world)
        mesh = {'vertices': aligned_verts, 'faces': normalized['faces']}
        with self._lock:
            self._mesh_cache[track_id] = mesh
        return mesh

    def is_pending(self, track_id):
        """Check if track_id is currently being processed or queued."""
        with self._lock:
            return track_id in self._pending

    def is_completed(self, track_id):
        """Check if track_id has already been processed (success or fail)."""
        with self._lock:
            return track_id in self._completed

    def evict_track(self, track_id):
        """Track 被移除时释放缓存。"""
        with self._lock:
            self._mesh_cache.pop(track_id, None)
            self._completed.discard(track_id)
            self._pending.discard(track_id)

    def shutdown(self):
        """Stop background workers."""
        self._shutdown = True
        for t in self._workers:
            t.join(timeout=5.0)

    # --- Disk Cache ---

    def _disk_cache_path(self, label):
        if not self._cache_dir:
            return None
        safe_label = label.replace('/', '_').replace('\\', '_')
        return os.path.join(self._cache_dir, f'{safe_label}.npz')

    def _save_disk_cache(self, label, normalized_mesh):
        path = self._disk_cache_path(label)
        if path is None:
            return
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez_compressed(
                path,
                vertices=normalized_mesh['vertices'],
                faces=normalized_mesh['faces'])
            if self._logger:
                self._logger.info(f'[Pixel3D] Disk cache saved: {path}')
        except Exception as e:
            if self._logger:
                self._logger.warning(f'[Pixel3D] Disk cache save failed: {e}')

    def _load_disk_cache(self, label):
        path = self._disk_cache_path(label)
        if path is None or not os.path.isfile(path):
            return None
        try:
            data = np.load(path)
            return {
                'vertices': data['vertices'].astype(np.float64),
                'faces': data['faces'].astype(np.int32),
            }
        except Exception:
            return None

    def _load_all_disk_cache(self):
        """启动时加载磁盘上所有已缓存的 mesh 到 label_cache。"""
        if not self._cache_dir or not os.path.isdir(self._cache_dir):
            return
        count = 0
        for fname in os.listdir(self._cache_dir):
            if not fname.endswith('.npz'):
                continue
            label = fname[:-4]
            mesh = self._load_disk_cache(label)
            if mesh is not None:
                self._label_cache[label] = mesh
                count += 1
        if count > 0 and self._logger:
            self._logger.info(
                f'[Pixel3D] Loaded {count} mesh(es) from disk cache: '
                f'{self._cache_dir}')
