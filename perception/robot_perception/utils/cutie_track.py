"""Cutie video object segmentation wrapper for multi-object bbox tracking."""
import os
import sys

import cv2
import numpy as np
import torch
from torchvision.transforms.functional import to_tensor as tv_to_tensor

from robot_perception.utils.paths import CUTIE_DIR


def _ensure_cutie_path():
    if CUTIE_DIR not in sys.path:
        sys.path.insert(0, CUTIE_DIR)


class MultiCutieTracker:
    """Single Cutie InferenceCore tracking multiple object IDs."""

    def __init__(self, seg_threshold=0.1, device='cuda'):
        _ensure_cutie_path()
        from hydra.core.global_hydra import GlobalHydra
        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model

        self.seg_threshold = seg_threshold
        self.device = device
        GlobalHydra.instance().clear()
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1
        self._active_obj_ids = set()

    def reset(self):
        """Drop all objects — recreate processor on next use."""
        _ensure_cutie_path()
        from hydra.core.global_hydra import GlobalHydra
        from cutie.inference.inference_core import InferenceCore
        from cutie.utils.get_default_model import get_default_model

        GlobalHydra.instance().clear()
        self.cutie = get_default_model()
        self.processor = InferenceCore(self.cutie, cfg=self.cutie.cfg)
        self.processor.max_internal_size = -1
        self._active_obj_ids = set()

    def _to_tensor(self, frame_rgb):
        t = tv_to_tensor(frame_rgb).to(self.device).float()
        return t

    def initialize_objects(self, frame_rgb, obj_masks):
        """Register or re-register objects. obj_masks: {obj_id: bool HxW mask}."""
        if not obj_masks:
            return {}

        h, w = frame_rgb.shape[:2]
        combined = np.zeros((h, w), dtype=np.int64)
        objects = []
        for obj_id, mask in obj_masks.items():
            m = mask.astype(bool)
            if m.sum() < 10:
                continue
            combined[m] = int(obj_id)
            objects.append(int(obj_id))

        if not objects:
            return {}

        with torch.no_grad():
            frame_tensor = self._to_tensor(frame_rgb)
            mask_tensor = torch.from_numpy(combined).to(self.device)
            output_prob = self.processor.step(
                frame_tensor, mask_tensor, objects=objects)
            out_mask = self.processor.output_prob_to_mask(
                output_prob, segment_threshold=self.seg_threshold)
            out_np = out_mask.cpu().numpy()

        self._active_obj_ids = set(objects)
        return {
            oid: (out_np == oid) for oid in objects
        }

    def track(self, frame_rgb):
        """Propagate all active objects to the next frame."""
        if not self._active_obj_ids:
            return {}

        with torch.no_grad():
            frame_tensor = self._to_tensor(frame_rgb)
            output_prob = self.processor.step(frame_tensor)
            out_mask = self.processor.output_prob_to_mask(
                output_prob, segment_threshold=self.seg_threshold)
            out_np = out_mask.cpu().numpy()

        result = {}
        for oid in list(self._active_obj_ids):
            mask = (out_np == oid)
            if mask.sum() >= 10:
                result[oid] = mask
        return result

    @staticmethod
    def mask_to_bbox_xyxy(mask):
        mask_u8 = mask.astype(np.uint8)
        ys, xs = np.where(mask_u8 > 0)
        if len(xs) == 0:
            return None
        return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)
