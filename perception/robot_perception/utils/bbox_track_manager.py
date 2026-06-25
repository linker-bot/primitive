"""Temporal multi-object tracking for detection_bbox (Cutie + layered GDINO)."""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import time
from typing import Dict, List, Optional

import numpy as np

from robot_perception.utils.bbox3d_from_depth import (
    aabb_in_frame,
    backproject_mask,
    bbox_area_xyxy,
    bbox_mask_fill_ratio,
    classify_grasp_type,
    compute_top_normal,
    mask_to_bbox_xyxy,
    pca_orientation,
    refine_instance_mask,
)
from robot_perception.utils.bbox_association import (
    associate_detections_to_tracks,
    bbox_iou,
    mask_iou,
)

# RGB colors on rgb8 annotated image; index = (instance_id - 1) % N
VIZ_COLOR_NAMES = (
    'green', 'red', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple',
)
from robot_perception.utils.cutie_track import MultiCutieTracker
from robot_perception.utils.gdino_sam import (
    expand_bbox_xyxy,
    gdino_detect,
    gdino_detect_crop,
    sam2_segment_boxes,
)
from robot_perception.utils.vlm_detector import vlm_detect_as_gdino, vlm_detect_multi_as_gdino
from robot_perception.utils.tag_mapping import (
    build_combined_caption,
    label_matches_target,
    match_phrase_to_label,
)
from robot_perception.utils.world_roi import apply_world_roi_to_aabbs, explain_world_roi, passes_world_roi


@dataclass
class BboxTrack:
    track_id: int
    cutie_obj_id: int
    label: str
    prompt: str
    score: float = 0.0
    last_mask: Optional[np.ndarray] = None
    last_bbox2d: Optional[np.ndarray] = None
    mode: str = 'init'
    lost_count: int = 0
    age: int = 0
    roi_fail_count: int = 0
    label_votes: deque = field(default_factory=deque)
    anchor_mask_pixels: int = 0


class BboxTrackManager:
    """Layer1 Cutie + Layer2 ROI refine + Layer4 global discovery/recovery."""

    def __init__(self, args, T_world_cam, object_colors, logger=None, vlm_detector=None,
                 device='cuda'):
        self.args = args
        self.T_world_cam = T_world_cam
        self.object_colors = object_colors
        self.logger = logger
        self.vlm_detector = vlm_detector
        self.device = device
        self.tracks: List[BboxTrack] = []
        self._next_track_id = 1
        self._next_cutie_obj_id = 1
        self._frame_idx = 0
        self._cutie: Optional[MultiCutieTracker] = None
        self.active_targets = []
        self.text_prompts = []
        self.prompt_to_label = {}
        self._last_layer_stats = {}
        self._last_discovery_eval: dict = {}
        self._discovery_batch_cursor: int = 0
        self._roi_blacklist: Dict[str, int] = {}
        self._spatial_blacklist: List[tuple] = []

    def reset(self):
        self.tracks = []
        self._next_track_id = 1
        self._next_cutie_obj_id = 1
        self._frame_idx = 0
        self._last_layer_stats = {}
        self._roi_blacklist = {}
        self._spatial_blacklist = []
        if self._cutie is not None:
            self._cutie.reset()

    def update_targets(self, active_targets, text_prompts, prompt_to_label):
        """Replace targets and reset all tracks (user-initiated prompt change)."""
        self.active_targets = list(active_targets)
        self.text_prompts = list(text_prompts)
        self.prompt_to_label = dict(prompt_to_label)
        self.reset()

    def sync_targets_preserve_tracks(self, active_targets, text_prompts, prompt_to_label):
        """Update caption/targets without resetting Cutie tracks.

        Drops tracks whose label is no longer in active_targets; keeps the rest.
        Used when scene understanding adds/removes objects incrementally.
        """
        self.active_targets = list(active_targets)
        self.text_prompts = list(text_prompts)
        self.prompt_to_label = dict(prompt_to_label)
        active_labels = {t['label'] for t in self.active_targets}
        before = len(self.tracks)
        self.tracks = [t for t in self.tracks if t.label in active_labels]
        if before != len(self.tracks) and self.logger:
            self.logger.info(
                f'[TrackManager] Dropped {before - len(self.tracks)} track(s) '
                f'after scene prompt sync (labels no longer active)')

    def extend_targets(self, active_targets, text_prompts, prompt_to_label):
        """Alias: update prompts without resetting existing tracks."""
        self.sync_targets_preserve_tracks(active_targets, text_prompts, prompt_to_label)

    def get_debug_info(self):
        return {
            'frame': self._frame_idx,
            'layer_stats': dict(self._last_layer_stats),
            'tracks': [
                {
                    'track_id': t.track_id,
                    'label': t.label,
                    'prompt': t.prompt,
                    'mode': t.mode,
                    'lost_count': t.lost_count,
                    'age': t.age,
                    'score': round(float(t.score), 3),
                    'label_votes': list(t.label_votes),
                    'mask_ratio': round(self._mask_area_ratio(t.last_mask), 3),
                }
                for t in self.tracks
            ],
        }

    def get_track_publish_stats(self):
        """Per-track mask/mode stats for frame logging."""
        stats = []
        h = w = 0
        if self.tracks and self.tracks[0].last_mask is not None:
            h, w = self.tracks[0].last_mask.shape[:2]
        img_area = max(1, h * w)
        for t in self.tracks:
            ratio = self._mask_area_ratio(t.last_mask)
            bbox_ratio = 0.0
            fill = 0.0
            if t.last_bbox2d is not None and img_area > 1:
                bbox_ratio = bbox_area_xyxy(t.last_bbox2d) / img_area
                fill = bbox_mask_fill_ratio(t.last_bbox2d, t.last_mask)
            _, cname = self._color_for_instance(t.track_id)
            stats.append({
                'id': t.track_id,
                'label': t.label,
                'mode': t.mode,
                'lost': t.lost_count,
                'mask_ratio': ratio,
                'bbox_ratio': bbox_ratio,
                'fill': fill,
                'color_name': cname,
                'prompt': t.prompt,
            })
        return stats

    def _geometry_mask(self, mask: np.ndarray) -> np.ndarray:
        min_px = int(getattr(self.args, 'min_mask_pixels', 100))
        open_k = int(getattr(self.args, 'track_mask_open_kernel', 3))
        return refine_instance_mask(mask, min_pixels=min_px, open_kernel=open_k)

    def _mask_refine_kwargs(self):
        return {
            'refine': True,
            'min_component_pixels': int(getattr(self.args, 'min_mask_pixels', 100)),
            'open_kernel': int(getattr(self.args, 'track_mask_open_kernel', 3)),
        }

    def _bbox_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        return mask_to_bbox_xyxy(mask.astype(np.uint8), **self._mask_refine_kwargs())

    def _bbox_geometry_ok(
            self, track: BboxTrack, bbox: np.ndarray, geo_mask: np.ndarray) -> bool:
        min_fill = float(getattr(self.args, 'track_bbox_min_fill_ratio', 0.10))
        fill = bbox_mask_fill_ratio(bbox, geo_mask)
        if fill < min_fill:
            self._log_reject(
                track.label, 'bbox too sparse (table speckle?)',
                f'(fill={fill:.2f} < {min_fill})')
            return False
        max_jump = float(getattr(self.args, 'track_bbox_area_jump_max', 2.5))
        if track.last_bbox2d is not None and track.lost_count == 0 and max_jump > 0:
            prev_a = bbox_area_xyxy(track.last_bbox2d)
            new_a = bbox_area_xyxy(bbox)
            if prev_a > 0 and new_a > prev_a * max_jump:
                self._log_reject(
                    track.label, 'bbox area jump',
                    f'({new_a:.0f} > {prev_a:.0f} * {max_jump})')
                return False
        return True

    def _smooth_bbox2d(self, track: BboxTrack, bbox: np.ndarray) -> np.ndarray:
        alpha = float(getattr(self.args, 'track_bbox_ema', 0.8))
        prev = track.last_bbox2d
        if prev is None or alpha <= 0 or alpha >= 1.0:
            return bbox
        return (alpha * prev + (1.0 - alpha) * bbox).astype(np.float32)

    def _ensure_cutie(self):
        if self._cutie is None and self.args.use_cutie_tracking:
            self._cutie = MultiCutieTracker(
                seg_threshold=self.args.cutie_seg_threshold,
                device=self.device,
            )

    def _new_label_votes(self):
        window = max(1, int(getattr(self.args, 'track_label_vote_window', 5)))
        return deque(maxlen=window)

    def _init_track_votes(self, track: BboxTrack, label: str):
        track.label_votes = self._new_label_votes()
        if label:
            track.label_votes.append(label)

    def _apply_label_vote(self, track: BboxTrack, candidate_label: Optional[str]):
        if not candidate_label:
            return
        if track.label_votes.maxlen != max(
                1, int(getattr(self.args, 'track_label_vote_window', 5))):
            track.label_votes = self._new_label_votes()
            if track.label:
                track.label_votes.append(track.label)
        track.label_votes.append(candidate_label)
        voted = Counter(track.label_votes).most_common(1)[0][0]
        if (
            not self.args.track_label_lock
            or track.age <= 1
            or track.mode in ('init', 'reinitialized')
        ):
            track.label = voted

    def _mask_area_ratio(self, mask: Optional[np.ndarray]) -> float:
        if mask is None:
            return 0.0
        h, w = mask.shape[:2]
        return float(mask.sum()) / max(1, h * w)

    def _mask_track_ok(self, track: BboxTrack, mask: np.ndarray) -> bool:
        if int(mask.sum()) < self.args.min_mask_pixels:
            return False
        max_ratio = float(getattr(self.args, 'track_max_mask_ratio', 0.25))
        if self._mask_area_ratio(mask) > max_ratio:
            return False
        if track.last_mask is None or track.age <= 1:
            return True
        iou = mask_iou(track.last_mask, mask)
        if iou < self.args.track_mask_iou_min:
            return False
        prev_area = max(int(track.last_mask.sum()), 1)
        area = int(mask.sum())
        ratio = area / prev_area
        if not (self.args.track_area_ratio_min <= ratio <= self.args.track_area_ratio_max):
            return False
        anchor = int(track.anchor_mask_pixels)
        if anchor > 0:
            anchor_max = float(getattr(self.args, 'track_anchor_area_ratio_max', 2.5))
            if area > anchor * anchor_max:
                return False
        return True

    def _mask_table_leak_ratio(self, mask: np.ndarray, depth: np.ndarray,
                               K) -> Optional[float]:
        """Fraction of mask pixels hugging the RANSAC workbench plane (table bleed)."""
        from robot_perception.utils.bbox3d_from_depth import transform_points
        from robot_perception.utils.workbench_plane import signed_plane_distances

        plane_state = getattr(self.args, 'workbench_plane_state', None)
        if (self.T_world_cam is None or plane_state is None
                or not getattr(plane_state, 'valid', False)):
            return None
        mask_u8 = (mask > 0).astype(np.uint8)
        h, w = depth.shape[:2]
        if mask_u8.shape[:2] != (h, w):
            return None
        ys, xs = np.where(mask_u8 > 0)
        if len(ys) < 30:
            return None
        step = max(1, len(ys) // 300)
        ys, xs = ys[::step], xs[::step]
        z = depth[ys, xs].astype(np.float64)
        valid = (
            (z > self.args.depth_min_m)
            & (z < self.args.depth_max_m)
            & np.isfinite(z)
        )
        if int(valid.sum()) < 20:
            return None
        ys, xs, z = ys[valid], xs[valid], z[valid]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)
        pts_world = transform_points(self.T_world_cam, pts_cam)
        dists = signed_plane_distances(pts_world, plane_state.normal, plane_state.d)
        margin = float(getattr(self.args, 'workbench_surface_tol_m', 0.04))
        return float((np.abs(dists) <= margin).mean())

    def _mask_depth_std(self, mask: np.ndarray, depth: np.ndarray) -> Optional[float]:
        mask_u8 = mask.astype(np.uint8)
        h, w = depth.shape[:2]
        if mask_u8.shape[:2] != (h, w):
            return None
        z = depth[mask_u8 > 0]
        valid = z[(z > self.args.depth_min_m) & (z < self.args.depth_max_m) & np.isfinite(z)]
        if len(valid) < self.args.min_depth_points:
            return None
        return float(np.std(valid))

    def _mask_depth_valid(self, mask: np.ndarray, depth: np.ndarray,
                          track: BboxTrack) -> bool:
        """Check mask region has meaningful depth variation (not just flat background.

        When an object is removed, Cutie may still propagate a ghost mask over the
        now-empty surface. The depth in that region becomes uniformly flat (table plane)
        with very low variance, unlike a real 3D object which has depth variation.
        """
        depth_std_min = getattr(self.args, 'track_depth_std_min', 0.003)
        if depth_std_min <= 0:
            return True
        std = self._mask_depth_std(mask, depth)
        if std is None:
            return False
        return std >= depth_std_min

    def _is_surface_like_mask(self, mask: np.ndarray, depth: np.ndarray,
                              label: str = '', K=None) -> bool:
        """Reject table/workbench-like segments before track association."""
        h, w = mask.shape[:2]
        mask_ratio = self._mask_area_ratio(mask)
        max_ratio = float(getattr(self.args, 'track_max_mask_ratio', 0.25))
        if mask_ratio > max_ratio:
            self._log_reject(
                label or '?', 'surface-like — mask too large',
                f'({mask_ratio:.2f} > {max_ratio})')
            return True
        depth_std_min = getattr(self.args, 'track_depth_std_min', 0.003)
        if depth_std_min > 0:
            std = self._mask_depth_std(mask, depth)
            if std is not None and std < depth_std_min:
                self._log_reject(
                    label or '?', 'surface-like — flat depth',
                    f'(std={std:.4f} < {depth_std_min})')
                return True
        if K is not None:
            leak = self._mask_table_leak_ratio(mask, depth, K)
            leak_min_ratio = float(
                getattr(self.args, 'track_table_leak_min_mask_ratio', 0.01))
            leak_max = float(getattr(self.args, 'track_table_leak_max', 0.72))
            if leak is not None and leak >= leak_max:
                if mask_ratio >= leak_min_ratio or leak >= 0.85:
                    self._log_reject(
                        label or '?', 'surface-like — table plane bleed',
                        f'(leak={leak:.2f}, mask={mask_ratio:.2f})')
                    return True
        return False

    def _add_spatial_blacklist(self, bbox2d: np.ndarray):
        frames = int(getattr(self.args, 'track_spatial_blacklist_frames', 100))
        self._spatial_blacklist.append((bbox2d.copy(), self._frame_idx + frames))

    def _is_spatial_blacklisted(self, bbox2d: np.ndarray) -> bool:
        iou_min = float(getattr(self.args, 'track_spatial_blacklist_iou', 0.3))
        kept = []
        blocked = False
        for bbox, expiry in self._spatial_blacklist:
            if self._frame_idx >= expiry:
                continue
            kept.append((bbox, expiry))
            if bbox_iou(bbox, bbox2d) >= iou_min:
                blocked = True
        self._spatial_blacklist = kept
        return blocked

    def _filter_discovery_records(self, det_records, depth, K):
        """Drop workbench-like / blacklisted detections before association."""
        filtered = []
        for rec in det_records:
            label = rec['label']
            if self._is_blacklisted(label):
                continue
            if self._is_spatial_blacklisted(rec['bbox2d']):
                continue
            if self._is_surface_like_mask(rec['mask'], depth, label, K):
                self._add_spatial_blacklist(rec['bbox2d'])
                continue
            filtered.append(rec)
        return filtered

    def _all_tracks_stable(self) -> bool:
        if not self.tracks:
            return False
        stable_modes = {'fast_track', 'refined', 'corrected', 'recovered'}
        min_age = int(getattr(self.args, 'track_stable_min_age', 3))
        for tr in self.tracks:
            if tr.lost_count > 0 or tr.mode == 'drift':
                return False
            if tr.age < min_age:
                return False
            if tr.mode not in stable_modes:
                return False
        return True

    def _track_covers_target(self, target_label: str) -> bool:
        """True if an active (non-lost) track matches this scene/target label."""
        for tr in self.tracks:
            if tr.lost_count > 0:
                continue
            if label_matches_target(tr.label, target_label):
                return True
        return False

    def _missing_discovery_prompts(self) -> list[str]:
        """Scene prompts with no active matching track — candidates for gap-fill GDINO."""
        missing = []
        seen = set()
        for t in self.active_targets:
            label = t['label']
            if self._track_covers_target(label):
                continue
            prompt = (t.get('prompt') or label).strip()
            key = prompt.lower().rstrip('.')
            if key in seen:
                continue
            seen.add(key)
            missing.append(prompt)
        return missing

    def _evaluate_global_discovery(self) -> tuple[bool, str, int, str]:
        """Return (need_run, reason, interval_frames, mode).

        mode: empty | unstable | stable | gap_fill | disabled
        """
        if not self.args.use_cutie_tracking:
            return False, 'cutie_off', 0, 'disabled'

        missing = self._missing_discovery_prompts()
        gap_fill = getattr(self.args, 'track_discovery_gap_fill', True)
        if (
            gap_fill
            and missing
            and len(self.text_prompts) > len(missing)
        ):
            gap_iv = max(
                1, int(getattr(self.args, 'track_gap_fill_discovery_interval', 5)))
            if self._frame_idx % gap_iv == 0:
                return True, f'gap_fill_{len(missing)}missing', gap_iv, 'gap_fill'
            next_in = gap_iv - (self._frame_idx % gap_iv)
            return False, f'gap_fill_wait_{next_in}f', gap_iv, 'gap_fill'

        if not self.tracks:
            interval = max(1, int(getattr(self.args, 'track_empty_discovery_interval', 5)))
            mode = 'empty'
        elif self._all_tracks_stable():
            stable_iv = int(getattr(self.args, 'track_stable_discovery_interval', 15))
            skip_stable = getattr(self.args, 'track_skip_discovery_when_stable', True)
            if stable_iv <= 0 and skip_stable:
                return False, 'stable_skip', 0, 'stable'
            interval = max(
                1,
                stable_iv if stable_iv > 0 else int(self.args.track_global_detect_interval),
            )
            mode = 'stable'
        else:
            interval = max(1, int(self.args.track_global_detect_interval))
            mode = 'unstable'

        if self._frame_idx % interval != 0:
            next_in = interval - (self._frame_idx % interval)
            return False, f'wait_{next_in}f', interval, mode

        return True, f'periodic_{mode}', interval, mode

    def _need_global_discovery(self) -> bool:
        need, reason, interval, mode = self._evaluate_global_discovery()
        self._last_discovery_eval = {
            'need': need,
            'reason': reason,
            'interval': interval,
            'mode': mode,
            'frame': self._frame_idx,
        }
        missing = self._missing_discovery_prompts()
        self._last_discovery_eval['missing_prompts'] = missing
        self._last_discovery_eval['missing_count'] = len(missing)
        if need and self.logger:
            if mode == 'gap_fill' and missing:
                cap = build_combined_caption(missing)
                self.logger.info(
                    f'[TrackManager] Global discovery: {reason} '
                    f'interval={interval}f mode={mode} frame={self._frame_idx} '
                    f'undetected={missing} caption="{cap}" '
                    f'tracks={len(self.tracks)}/{len(self.text_prompts)}',
                    throttle_duration_sec=1.0)
            else:
                cap = build_combined_caption(self.text_prompts)
                self.logger.info(
                    f'[TrackManager] Global discovery: {reason} '
                    f'interval={interval}f mode={mode} frame={self._frame_idx} '
                    f'caption="{cap}" tracks={len(self.tracks)} '
                    f'missing={len(missing)}',
                    throttle_duration_sec=1.0)
        return need

    def _should_apply_discovery_update(
            self, track: BboxTrack, mask: np.ndarray, depth, K) -> bool:
        """Decide whether global discovery should overwrite a track mask."""
        if self._is_surface_like_mask(mask, depth, track.label, K):
            return False
        if not self._mask_depth_valid(mask, depth, track):
            return False

        skip_iou = float(getattr(self.args, 'track_discovery_skip_reinit_iou', 0.65))
        if track.lost_count == 0 and track.last_mask is not None and skip_iou > 0:
            iou = mask_iou(track.last_mask, mask)
            if iou >= skip_iou:
                return False

        min_iou = float(getattr(self.args, 'track_discovery_update_iou_min', 0.2))
        if track.last_mask is not None and track.age > 1:
            iou = mask_iou(track.last_mask, mask)
            if iou < min_iou:
                self._log_reject(
                    track.label, 'discovery rejected — mask jump',
                    f'(iou={iou:.2f} < {min_iou})')
                return False

        tmp = BboxTrack(
            track_id=track.track_id,
            cutie_obj_id=track.cutie_obj_id,
            label=track.label,
            prompt=track.prompt,
            score=track.score,
            roi_fail_count=track.roi_fail_count,
        )
        if self._build_result(tmp, mask, depth, K) is None:
            track.roi_fail_count = tmp.roi_fail_count
            return False
        track.roi_fail_count = tmp.roi_fail_count
        return True

    def _commit_track_update(
            self, track: BboxTrack, mask, bbox2d, det, label,
            depth, K, results_by_id, init_masks, mode: str):
        track.lost_count = 0
        track.age += 1
        track.mode = mode
        if self._store_result(track, mask, depth, K, results_by_id):
            geo = results_by_id[track.track_id]['mask']
            init_masks[track.cutie_obj_id] = geo.copy()
            if track.anchor_mask_pixels <= 0:
                track.anchor_mask_pixels = int(geo.sum())

    def _color_for_instance(self, instance_id: int):
        """Stable viz color per track instance (#1=green, #2=red, ...)."""
        idx = (max(1, int(instance_id)) - 1) % len(self.object_colors)
        name = VIZ_COLOR_NAMES[idx % len(VIZ_COLOR_NAMES)]
        return self.object_colors[idx], name

    def _color_for_label(self, label: str):
        labels = [t['label'] for t in self.active_targets]
        idx = labels.index(label) if label in labels else 0
        return self.object_colors[idx % len(self.object_colors)]

    def _log_reject(self, label, reason, detail=''):
        if self.logger:
            msg = f'[TrackManager] [{label}] {reason}'
            if detail:
                msg += f' {detail}'
            self.logger.info(msg, throttle_duration_sec=2.0)

    def _plane_signed_dists_cam(self, pts_cam, plane_state):
        from robot_perception.utils.bbox3d_from_depth import transform_points
        from robot_perception.utils.workbench_plane import signed_plane_distances
        pts_world = transform_points(self.T_world_cam, pts_cam)
        return signed_plane_distances(pts_world, plane_state.normal, plane_state.d)

    def _clip_below_plane(self, pts_cam, plane_state):
        """Remove points that project below the workbench plane in world frame."""
        dists = self._plane_signed_dists_cam(pts_cam, plane_state)
        margin = float(getattr(self.args, 'workbench_surface_tol_m', 0.04))
        above = dists >= -margin
        if above.sum() < self.args.min_depth_points:
            return pts_cam
        return pts_cam[above]

    def _strip_table_plane_points(self, pts_cam, plane_state):
        """Drop on-table depth points so 3D AABB / mesh are not anchored to the desk."""
        if len(pts_cam) == 0:
            return pts_cam
        dists = self._plane_signed_dists_cam(pts_cam, plane_state)
        margin = float(getattr(self.args, 'workbench_surface_tol_m', 0.04))
        above = dists > margin
        if int(above.sum()) < self.args.min_depth_points:
            return pts_cam
        return pts_cam[above]

    def _strip_table_pixels_from_mask(self, mask, depth, K, plane_state):
        """Remove mask pixels hugging the workbench plane (annotated overlay / backproj)."""
        if self.T_world_cam is None or not getattr(plane_state, 'valid', False):
            return mask
        out = mask.astype(np.uint8).copy()
        ys, xs = np.where(out > 0)
        if len(ys) < 10:
            return out
        z = depth[ys, xs].astype(np.float64)
        valid = (
            (z > self.args.depth_min_m)
            & (z < self.args.depth_max_m)
            & np.isfinite(z)
        )
        if int(valid.sum()) < 10:
            return out
        ys, xs, z = ys[valid], xs[valid], z[valid]
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        x = (xs.astype(np.float64) - cx) * z / fx
        y = (ys.astype(np.float64) - cy) * z / fy
        pts_cam = np.stack([x, y, z], axis=1)
        dists = self._plane_signed_dists_cam(pts_cam, plane_state)
        margin = float(getattr(self.args, 'workbench_surface_tol_m', 0.04))
        on_table = dists <= margin
        if int(on_table.sum()) == 0:
            return out
        out[ys[on_table], xs[on_table]] = 0
        return out

    def _build_result(self, track: BboxTrack, mask: np.ndarray, depth, K):
        label = track.label
        raw_u8 = (np.asarray(mask) > 0).astype(np.uint8)
        h, w = raw_u8.shape[:2]
        raw_ratio = float(raw_u8.sum()) / max(1, h * w)
        geo_mask = self._geometry_mask(mask)
        if int(geo_mask.sum()) < self.args.min_mask_pixels:
            self._log_reject(label, 'mask too small', f'({int(geo_mask.sum())} px)')
            return None

        max_mask_ratio = float(getattr(self.args, 'track_max_mask_ratio', 0.25))
        h, w = geo_mask.shape[:2]
        mask_ratio = float(geo_mask.sum()) / max(1, h * w)
        if raw_ratio >= 0.08 and raw_ratio > mask_ratio * 1.8:
            _, cname = self._color_for_instance(track.track_id)
            self._log_reject(
                label,
                f'SAM→geo shrink [{cname} #{track.track_id}]',
                f'(sam={raw_ratio * 100:.1f}% → pub={mask_ratio * 100:.1f}%)')
        if mask_ratio > max_mask_ratio:
            self._log_reject(
                label, 'mask too large (likely table/background)',
                f'({mask_ratio:.2f} > {max_mask_ratio})')
            track.roi_fail_count += 1
            return None

        bbox2d = self._bbox_from_mask(geo_mask)
        if bbox2d is None:
            return None
        if not self._bbox_geometry_ok(track, bbox2d, geo_mask):
            track.roi_fail_count += 1
            return None
        bbox2d = self._smooth_bbox2d(track, bbox2d)

        plane_state = getattr(self.args, 'workbench_plane_state', None)
        if (self.T_world_cam is not None and plane_state is not None
                and getattr(plane_state, 'valid', False)):
            geo_mask = self._strip_table_pixels_from_mask(
                geo_mask, depth, K, plane_state)
            if int(geo_mask.sum()) < self.args.min_mask_pixels:
                self._log_reject(label, 'mask empty after table strip', '')
                track.roi_fail_count += 1
                return None
            bbox2d = self._bbox_from_mask(geo_mask)
            if bbox2d is None:
                return None
            bbox2d = self._smooth_bbox2d(track, bbox2d)

        pts_cam = backproject_mask(
            depth, K, geo_mask,
            depth_min=self.args.depth_min_m,
            depth_max=self.args.depth_max_m,
        )

        if (self.T_world_cam is not None and plane_state is not None
                and getattr(plane_state, 'valid', False) and len(pts_cam) > 0):
            pts_cam = self._clip_below_plane(pts_cam, plane_state)
            pts_cam = self._strip_table_plane_points(pts_cam, plane_state)

        aabb_cam = None
        aabb_work = None
        aabb_work_mesh = None
        n_pts = len(pts_cam)
        if n_pts >= self.args.min_depth_points:
            aabb_cam = aabb_in_frame(pts_cam, T_frame_cam=None)
            if self.T_world_cam is not None:
                aabb_work_mesh = aabb_in_frame(pts_cam, T_frame_cam=self.T_world_cam)
                aabb_cam, aabb_work = apply_world_roi_to_aabbs(
                    aabb_cam, aabb_work_mesh, self.T_world_cam, self.args,
                    plane_state=plane_state)
                if aabb_work is None and aabb_work_mesh is not None:
                    roi_info = explain_world_roi(
                        aabb_work_mesh, self.T_world_cam, self.args,
                        plane_state=plane_state)
                    self._log_reject(
                        label, '3D dropped — outside world ROI',
                        str(roi_info))
                    track.roi_fail_count += 1
                else:
                    track.roi_fail_count = 0
        else:
            self._log_reject(
                label, '2D only — too few depth points',
                f'({n_pts}<{self.args.min_depth_points})')

        orientation = pca_orientation(pts_cam)
        top_normal = compute_top_normal(pts_cam)
        orientation_world = None
        top_normal_world = None
        if self.T_world_cam is not None and n_pts >= 10:
            from robot_perception.utils.bbox3d_from_depth import transform_points
            pts_world = transform_points(self.T_world_cam, pts_cam)
            orientation_world = pca_orientation(pts_world)
            top_normal_world = compute_top_normal(pts_world)
        grasp_type = classify_grasp_type(
            aabb_work['size'] if aabb_work is not None
            else (aabb_cam['size'] if aabb_cam is not None else None))

        color, color_name = self._color_for_instance(track.track_id)
        result = {
            'tag': label,
            'prompt': track.prompt,
            'score': float(track.score),
            'instance_id': int(track.track_id),
            'track_mode': track.mode,
            'bbox2d': bbox2d,
            'mask': geo_mask,
            'aabb_cam': aabb_cam,
            'aabb_work': aabb_work,
            'aabb_work_mesh': aabb_work_mesh,
            'pts_cam': pts_cam if n_pts > 0 else None,
            'orientation': orientation,
            'orientation_world': orientation_world,
            'top_normal': top_normal,
            'top_normal_world': top_normal_world,
            'grasp_type': grasp_type,
            'color': color,
            'color_name': color_name,
            'sam_mask_ratio': raw_ratio,
        }
        return result

    def _update_score(self, track: BboxTrack, new_score: float):
        if track.age <= 0:
            track.score = float(new_score)
        else:
            ema = self.args.track_score_ema
            track.score = ema * track.score + (1.0 - ema) * float(new_score)

    def _store_result(self, track: BboxTrack, mask, depth, K, results_by_id):
        r = self._build_result(track, mask, depth, K)
        if r is not None:
            results_by_id[track.track_id] = r
            track.last_mask = r['mask']
            track.last_bbox2d = r['bbox2d']
        return r is not None

    def _cutie_reinit(self, color, init_masks: Dict[int, np.ndarray]):
        """Reset Cutie and register all active tracks with compact object IDs 1..N.

        Cutie InferenceCore accumulates object IDs permanently; partial reinit with
        monotonically increasing cutie_obj_id causes tensor shape mismatches after
        several global-discovery cycles.
        """
        if not self.args.use_cutie_tracking or not self.tracks:
            return
        self._ensure_cutie()
        if self._cutie is None:
            return

        track_masks = []
        for tr in self.tracks:
            mask = init_masks.get(tr.cutie_obj_id)
            if mask is None:
                mask = tr.last_mask
            if mask is None or int(mask.sum()) < self.args.min_mask_pixels:
                continue
            track_masks.append((tr, mask))

        if not track_masks:
            return

        self._cutie.reset()
        compact_masks: Dict[int, np.ndarray] = {}
        for new_id, (tr, mask) in enumerate(track_masks, start=1):
            tr.cutie_obj_id = new_id
            compact_masks[new_id] = mask

        self._next_cutie_obj_id = len(track_masks) + 1
        try:
            updated = self._cutie.initialize_objects(color, compact_masks)
        except RuntimeError as exc:
            if self.logger:
                self.logger.error(
                    f'[TrackManager] Cutie reinit failed ({exc}); resetting tracker')
            self._cutie.reset()
            return

        for tr, _ in track_masks:
            if tr.cutie_obj_id in updated:
                tr.last_mask = updated[tr.cutie_obj_id]
                tr.last_bbox2d = self._bbox_from_mask(tr.last_mask)

    def _try_holdover_result(self, tr: BboxTrack, depth, K, results_by_id) -> bool:
        """Re-publish last good mask when the current frame fails (reduces flicker)."""
        if tr.last_mask is None:
            return False
        return self._store_result(tr, tr.last_mask, depth, K, results_by_id)

    def _mark_drift(self, tr: BboxTrack, depth, K, results_by_id) -> int:
        tr.lost_count += 1
        tr.mode = 'drift'
        self._try_holdover_result(tr, depth, K, results_by_id)
        return 1

    def _layer1_cutie(
        self, color, depth, K, results_by_id,
    ) -> int:
        """Fast Cutie propagation. Returns count of tracks still drifting."""
        drift_count = 0
        if not (self.args.use_cutie_tracking and self.tracks):
            return drift_count

        self._ensure_cutie()
        if self._cutie is None:
            return len(self.tracks)

        cutie_masks = self._cutie.track(color)
        for tr in self.tracks:
            mask = cutie_masks.get(tr.cutie_obj_id)
            if mask is None:
                drift_count += self._mark_drift(tr, depth, K, results_by_id)
                continue
            cutie_ratio = self._mask_area_ratio(mask)
            if cutie_ratio > float(getattr(self.args, 'track_max_mask_ratio', 0.25)):
                _, cname = self._color_for_instance(tr.track_id)
                self._log_reject(
                    tr.label, f'Cutie mask too large [{cname} #{tr.track_id}]',
                    f'(area={cutie_ratio * 100:.1f}%)')
                drift_count += self._mark_drift(tr, depth, K, results_by_id)
                continue
            elif cutie_ratio >= 0.08:
                _, cname = self._color_for_instance(tr.track_id)
                if self.logger:
                    self.logger.info(
                        f'[TrackManager] [cutie/raw] #{tr.track_id}:{cname}:'
                        f'{tr.label} area={cutie_ratio * 100:.1f}% mode={tr.mode}',
                        throttle_duration_sec=1.0)
            if not self._mask_track_ok(tr, mask):
                drift_count += self._mark_drift(tr, depth, K, results_by_id)
                continue
            if self._is_surface_like_mask(mask, depth, tr.label, K):
                drift_count += self._mark_drift(tr, depth, K, results_by_id)
                continue
            if not self._mask_depth_valid(mask, depth, tr):
                drift_count += self._mark_drift(tr, depth, K, results_by_id)
                continue
            prev_mask = tr.last_mask
            prev_bbox = tr.last_bbox2d
            tr.age += 1
            if self._store_result(tr, mask, depth, K, results_by_id):
                tr.lost_count = 0
                tr.mode = 'fast_track'
                if tr.anchor_mask_pixels <= 0:
                    tr.anchor_mask_pixels = int(
                        results_by_id[tr.track_id]['mask'].sum())
            else:
                tr.last_mask = prev_mask
                tr.last_bbox2d = prev_bbox
                tr.age = max(0, tr.age - 1)
                tr.lost_count += 1
                tr.mode = 'drift'
                drift_count += 1
                if prev_mask is not None:
                    self._store_result(tr, prev_mask, depth, K, results_by_id)
        return drift_count

    def _track_ref_bbox(self, track: BboxTrack):
        if track.last_bbox2d is not None:
            return track.last_bbox2d
        if track.last_mask is not None:
            return mask_to_bbox_xyxy(track.last_mask.astype(np.uint8))
        return None

    def _use_vlm_detect(self) -> bool:
        return bool(
            getattr(self.args, 'use_vlm_detect', False) and self.vlm_detector is not None)

    def _detect_boxes(self, color, caption, grounding_model, device, crop_roi=None):
        """Run full-image or ROI-crop bbox detection (VLM or GDINO)."""
        if self._use_vlm_detect():
            nms_iou = getattr(self.args, 'vlm_nms_iou', 0.5)
            max_area_ratio = getattr(self.args, 'vlm_max_area_ratio', 0.5)
            if crop_roi is not None:
                x1, y1, x2, y2 = crop_roi
                crop = color[y1:y2, x1:x2]
                if crop.size == 0:
                    return []
                detections = vlm_detect_as_gdino(self.vlm_detector, crop, caption)
                offset = np.array([x1, y1, x1, y1], dtype=np.float32)
                for det in detections:
                    det['box_xyxy'] = det['box_xyxy'] + offset
                return detections
            prompts = [p for p in getattr(self, 'text_prompts', []) if p and str(p).strip()]
            seen = set()
            unique_prompts = []
            for p in prompts:
                key = p.strip().lower().rstrip('.')
                if key in seen:
                    continue
                seen.add(key)
                unique_prompts.append(p)
            if len(unique_prompts) > 1:
                return vlm_detect_multi_as_gdino(
                    self.vlm_detector, color, unique_prompts,
                    nms_iou=nms_iou, max_area_ratio=max_area_ratio)
            return vlm_detect_as_gdino(
                self.vlm_detector, color, unique_prompts[0] if unique_prompts else caption)

        if crop_roi is not None:
            x1, y1, x2, y2 = crop_roi
            box_thr = getattr(self.args, 'track_refine_box_threshold', 0.25)
            return gdino_detect_crop(
                grounding_model, color, x1, y1, x2, y2, caption,
                box_thr, self.args.text_threshold, device,
            )
        return gdino_detect(
            grounding_model, color, caption,
            self.args.box_threshold, self.args.text_threshold, device,
        )

    def _layer2_roi_refine(
        self, track, color, depth, K, grounding_model, sam2_predictor, device,
        results_by_id, init_masks,
    ) -> bool:
        """ROI-local GDINO + SAM to correct a drifting track."""
        ref_bbox = self._track_ref_bbox(track)
        ref_mask = track.last_mask
        if ref_bbox is None:
            return False

        h, w = color.shape[:2]
        roi = expand_bbox_xyxy(
            ref_bbox, h, w, getattr(self.args, 'track_roi_expand_ratio', 2.0))
        x1, y1, x2, y2 = roi
        caption = build_combined_caption([track.prompt])
        if not caption:
            caption = build_combined_caption([track.label])

        dets = self._detect_boxes(
            color, caption, grounding_model, device, crop_roi=roi)
        score_min = getattr(self.args, 'track_refine_score_min', 0.35)
        if not self._use_vlm_detect():
            dets = [d for d in dets if d['score'] >= score_min]
        if not dets:
            return False

        best_det = max(dets, key=lambda d: bbox_iou(d['box_xyxy'], ref_bbox))
        if bbox_iou(best_det['box_xyxy'], ref_bbox) < self.args.track_assoc_iou_min * 0.5:
            return False

        masks = sam2_segment_boxes(
            sam2_predictor, color, np.array([best_det['box_xyxy']]))
        new_mask = masks[0]
        iou = mask_iou(ref_mask, new_mask) if ref_mask is not None else 1.0

        refine_min = getattr(self.args, 'track_refine_iou_min', 0.5)
        hard_min = getattr(self.args, 'track_refine_iou_hard_min', 0.25)
        if iou >= refine_min:
            track.mode = 'refined'
        elif iou >= hard_min:
            track.mode = 'corrected'
        else:
            return False

        matched_label = match_phrase_to_label(
            best_det['phrase'], self.prompt_to_label, self.active_targets,
            accept_unmatched=self.args.accept_unmatched_detections)
        self._apply_label_vote(track, matched_label or track.label)
        track.prompt = best_det['phrase']
        self._update_score(track, best_det['score'])

        if not self._mask_depth_valid(new_mask, depth, track):
            return False
        if self._is_surface_like_mask(new_mask, depth, track.label, K):
            return False

        track.lost_count = 0
        track.age += 1
        if not self._store_result(track, new_mask, depth, K, results_by_id):
            return False
        geo = results_by_id[track.track_id]['mask']
        init_masks[track.cutie_obj_id] = geo.copy()
        return True

    def _label_compatible(self, track: BboxTrack, det_label: str) -> bool:
        """Reject cross-label IoU matches for mature locked tracks."""
        if not getattr(self.args, 'track_assoc_require_label_match', True):
            return True
        if det_label == track.label:
            return True
        if track.age <= 1 or track.mode in ('init', 'reinitialized'):
            return True
        if not self.args.track_label_lock:
            return True
        return False

    def _det_records_from_masks(self, detections, masks):
        det_records = []
        for det, mask in zip(detections, masks):
            label = match_phrase_to_label(
                det['phrase'], self.prompt_to_label, self.active_targets,
                accept_unmatched=self.args.accept_unmatched_detections)
            if label is None:
                continue
            bbox2d = mask_to_bbox_xyxy(mask.astype(np.uint8))
            if bbox2d is None:
                continue
            det_records.append({
                'det': det,
                'mask': mask,
                'label': label,
                'bbox2d': bbox2d,
            })
        return det_records

    def _associate_and_update(
        self, det_records, depth, K, results_by_id, init_masks,
    ):
        if not det_records:
            return

        track_bboxes = [
            tr.last_bbox2d for tr in self.tracks if tr.last_bbox2d is not None]
        track_indices = [
            i for i, tr in enumerate(self.tracks) if tr.last_bbox2d is not None]
        det_bboxes = [r['bbox2d'] for r in det_records]

        compat = None
        if getattr(self.args, 'track_assoc_require_label_match', True):
            compat = np.zeros((len(det_records), len(track_indices)), dtype=bool)
            for di, rec in enumerate(det_records):
                for tj, tr_idx in enumerate(track_indices):
                    compat[di, tj] = self._label_compatible(
                        self.tracks[tr_idx], rec['label'])

        matched, unmatched_d, _unmatched_t = associate_detections_to_tracks(
            det_bboxes, track_bboxes, self.args.track_assoc_iou_min,
            compat_matrix=compat,
            prefer_center_for_same_label=True,
            det_labels=[r['label'] for r in det_records],
            track_labels=[self.tracks[i].label for i in track_indices],
        )

        for det_i, tr_local_i in matched.items():
            tr = self.tracks[track_indices[tr_local_i]]
            rec = det_records[det_i]
            det = rec['det']
            mask = rec['mask']
            self._apply_label_vote(tr, rec['label'])
            tr.prompt = det['phrase']
            self._update_score(tr, det['score'])
            if not self._should_apply_discovery_update(tr, mask, depth, K):
                continue
            mode = 'refined' if tr.age > 1 else 'init'
            self._commit_track_update(
                tr, mask, rec['bbox2d'], det, rec['label'],
                depth, K, results_by_id, init_masks, mode)

        for det_i in unmatched_d:
            rec = det_records[det_i]
            det = rec['det']
            mask = rec['mask']
            if self._is_surface_like_mask(mask, depth, rec['label'], K):
                self._add_spatial_blacklist(rec['bbox2d'])
                continue
            if not self._mask_depth_valid(
                    mask, depth,
                    BboxTrack(track_id=0, cutie_obj_id=0, label=rec['label'], prompt='')):
                continue
            tr = BboxTrack(
                track_id=self._next_track_id,
                cutie_obj_id=self._next_cutie_obj_id,
                label=rec['label'],
                prompt=det['phrase'],
                score=float(det['score']),
                mode='reinitialized',
                age=1,
            )
            self._init_track_votes(tr, rec['label'])
            if not self._store_result(tr, mask, depth, K, results_by_id):
                continue
            geo = results_by_id[tr.track_id]['mask']
            tr.anchor_mask_pixels = int(geo.sum())
            self._next_track_id += 1
            self._next_cutie_obj_id += 1
            self.tracks.append(tr)
            init_masks[tr.cutie_obj_id] = geo.copy()

    def _log_raw_detections(self, detections, masks, shape_hw, stage='discovery'):
        """Log GDINO phrase + SAM raw mask size before geometry filter (diagnose red box)."""
        if not self.logger or not detections:
            return
        h, w = shape_hw[:2]
        img_area = max(1, h * w)
        for det, mask in zip(detections, masks):
            raw_u8 = (np.asarray(mask) > 0).astype(np.uint8)
            raw_pct = float(raw_u8.sum()) / img_area * 100.0
            gbox = det.get('box_xyxy')
            g_pct = 0.0
            xyxy_s = ''
            if gbox is not None:
                g_pct = bbox_area_xyxy(gbox) / img_area * 100.0
                xyxy_s = (
                    f'xyxy=[{int(gbox[0])},{int(gbox[1])},'
                    f'{int(gbox[2])},{int(gbox[3])}]')
            label = match_phrase_to_label(
                det.get('phrase', ''), self.prompt_to_label, self.active_targets,
                accept_unmatched=self.args.accept_unmatched_detections)
            self.logger.info(
                f'[TrackManager] [{stage}/raw] phrase="{det.get("phrase", "")}" '
                f'→ label={label!r} score={float(det.get("score", 0)):.2f} '
                f'gdino_box={g_pct:.1f}% sam_raw={raw_pct:.1f}% {xyxy_s}',
                throttle_duration_sec=1.0)

    def _discovery_prompt_batches(self, gap_fill_active: bool = False) -> list[list[str]]:
        """Split text_prompts into GDINO caption batches."""
        batch_size = max(1, int(getattr(self.args, 'track_discovery_batch_size', 2)))
        gap_only = getattr(self.args, 'track_discovery_gap_fill_only', True)
        missing = self._missing_discovery_prompts()
        if gap_fill_active and gap_only and missing:
            prompts = missing
        else:
            prompts = [p for p in self.text_prompts if p]
        if not prompts:
            return []
        if len(prompts) <= batch_size:
            return [prompts]

        batches = [
            prompts[i:i + batch_size]
            for i in range(0, len(prompts), batch_size)
        ]
        max_per_frame = int(getattr(self.args, 'track_discovery_max_batches_per_frame', 0))
        if max_per_frame <= 0 or max_per_frame >= len(batches):
            return batches

        start = self._discovery_batch_cursor % len(batches)
        self._discovery_batch_cursor += 1
        selected = []
        for i in range(max_per_frame):
            selected.append(batches[(start + i) % len(batches)])
        return selected

    def _run_discovery_batch(
        self, color, depth, K, caption, grounding_model, sam2_predictor, device,
        results_by_id, init_masks, batch_idx, batch_total,
    ) -> int:
        if not caption:
            return 0
        detections = self._detect_boxes(color, caption, grounding_model, device)
        if not self._use_vlm_detect():
            detections = [
                d for d in detections if d['score'] >= self.args.min_detection_score]
        if not detections:
            return 0

        boxes_xyxy = np.stack([d['box_xyxy'] for d in detections], axis=0)
        masks, sam_scores = sam2_segment_boxes(
            sam2_predictor, color, boxes_xyxy, return_scores=True)
        sam_min = getattr(self.args, 'sam2_score_min', 0.7)
        filtered = [
            (det, mask) for det, mask, sc in zip(detections, masks, sam_scores)
            if sc >= sam_min
        ]
        if not filtered:
            return 0
        f_dets, f_masks = zip(*filtered)
        stage = f'discovery/b{batch_idx + 1}of{batch_total}'
        self._log_raw_detections(list(f_dets), list(f_masks), color.shape[:2], stage)
        det_records = self._det_records_from_masks(list(f_dets), list(f_masks))
        before = len(det_records)
        det_records = self._filter_discovery_records(det_records, depth, K)
        if before > len(det_records) and self.logger:
            self.logger.info(
                f'[TrackManager] Global discovery batch {batch_idx + 1}/{batch_total}: '
                f'filtered {before - len(det_records)} detection(s)')
        if (
            det_records
            and self.logger
            and self._last_discovery_eval.get('mode') == 'gap_fill'
        ):
            labels = [r['label'] for r in det_records]
            self.logger.info(
                f'[TrackManager] Gap-fill batch {batch_idx + 1}/{batch_total}: '
                f'accepted {len(det_records)} detection(s) {labels}',
                throttle_duration_sec=0.5)
        self._associate_and_update(det_records, depth, K, results_by_id, init_masks)
        return len(det_records)

    def _layer4_global_discovery(
        self, color, depth, K, caption, grounding_model, sam2_predictor, device,
        results_by_id, init_masks,
    ) -> int:
        use_batches = getattr(self.args, 'track_discovery_use_batches', True)
        gap_fill_active = self._last_discovery_eval.get('mode') == 'gap_fill'
        if use_batches:
            batches = self._discovery_prompt_batches(gap_fill_active=gap_fill_active)
        else:
            missing = self._missing_discovery_prompts()
            gap_only = getattr(self.args, 'track_discovery_gap_fill_only', True)
            if gap_fill_active and gap_only and missing:
                batches = [missing]
            else:
                batches = [[p for p in self.text_prompts if p]]
        if not batches or not any(batches):
            return 0

        if self.logger and len(batches) > 1:
            sizes = [len(b) for b in batches]
            scope = 'gap_fill' if gap_fill_active else 'all'
            self.logger.info(
                f'[TrackManager] Discovery batched GDINO ({scope}): {len(batches)} batch(es) '
                f'prompts_per_batch={sizes} tracks_before={len(self.tracks)} '
                f'missing={len(self._missing_discovery_prompts())}')

        total = 0
        for bi, batch_prompts in enumerate(batches):
            cap = build_combined_caption(batch_prompts)
            if self.logger and len(batches) > 1:
                self.logger.info(
                    f'[TrackManager] Discovery batch {bi + 1}/{len(batches)} '
                    f'caption="{cap}"',
                    throttle_duration_sec=0.5)
            total += self._run_discovery_batch(
                color, depth, K, cap, grounding_model, sam2_predictor, device,
                results_by_id, init_masks, bi, len(batches),
            )
        return total

    def _layer4_recover_track(
        self, track, color, depth, K, grounding_model, sam2_predictor, device,
        results_by_id, init_masks,
    ) -> int:
        """Per-track full-image GDINO using track prompt (cheaper than global caption)."""
        ref_bbox = self._track_ref_bbox(track)
        caption = build_combined_caption([track.prompt or track.label])
        if not caption:
            return 0

        detections = self._detect_boxes(
            color, caption, grounding_model, device)
        if not self._use_vlm_detect():
            detections = [
                d for d in detections if d['score'] >= self.args.min_detection_score]
        if not detections:
            return 0

        if ref_bbox is not None:
            detections = sorted(
                detections,
                key=lambda d: bbox_iou(d['box_xyxy'], ref_bbox),
                reverse=True,
            )
            if bbox_iou(detections[0]['box_xyxy'], ref_bbox) < self.args.track_assoc_iou_min:
                return len(detections)

        best = detections[0]
        masks, sam_scores = sam2_segment_boxes(
            sam2_predictor, color, np.array([best['box_xyxy']]), return_scores=True)
        mask = masks[0]
        sam_min = getattr(self.args, 'sam2_score_min', 0.7)
        if sam_scores[0] < sam_min:
            return 0

        matched_label = match_phrase_to_label(
            best['phrase'], self.prompt_to_label, self.active_targets,
            accept_unmatched=self.args.accept_unmatched_detections)
        self._apply_label_vote(track, matched_label or track.label)
        track.prompt = best['phrase']
        self._update_score(track, best['score'])

        if not self._mask_depth_valid(mask, depth, track):
            return 0
        if self._is_surface_like_mask(mask, depth, track.label, K):
            return 0

        track.lost_count = 0
        track.age += 1
        track.mode = 'recovered'
        if not self._store_result(track, mask, depth, K, results_by_id):
            return 0
        geo = results_by_id[track.track_id]['mask']
        init_masks[track.cutie_obj_id] = geo.copy()
        return len(detections)

    def process_frame(self, color, depth, K, grounding_model, sam2_predictor, device):
        self._frame_idx += 1
        caption = build_combined_caption(self.text_prompts)
        if not caption:
            return [], 0

        log_timing = bool(getattr(self.args, 'log_frame_timing', True))
        t_frame = time.perf_counter()
        timings: Dict[str, float] = {}

        results_by_id: Dict[int, dict] = {}
        init_masks: Dict[int, np.ndarray] = {}
        raw_det_count = 0
        layer2_ok = 0

        t0 = time.perf_counter()
        drift_count = self._layer1_cutie(color, depth, K, results_by_id)
        timings['cutie_ms'] = (time.perf_counter() - t0) * 1000.0

        use_layer2 = getattr(self.args, 'use_layer2_roi_refine', True)
        t0 = time.perf_counter()
        if use_layer2 and self.tracks:
            for tr in list(self.tracks):
                if tr.lost_count > 0:
                    if self._layer2_roi_refine(
                        tr, color, depth, K,
                        grounding_model, sam2_predictor, device,
                        results_by_id, init_masks,
                    ):
                        layer2_ok += 1
        timings['layer2_ms'] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if init_masks:
            self._cutie_reinit(color, init_masks)
            init_masks = {}
        timings['reinit_ms'] = (time.perf_counter() - t0) * 1000.0

        need_discovery = self._need_global_discovery()
        lost_tracks = [t for t in self.tracks if t.lost_count > 0]
        recovery_count = 0

        t0 = time.perf_counter()
        if need_discovery:
            n = self._layer4_global_discovery(
                color, depth, K, caption,
                grounding_model, sam2_predictor, device,
                results_by_id, init_masks,
            )
            raw_det_count += n
        elif lost_tracks:
            for tr in lost_tracks:
                n = self._layer4_recover_track(
                    tr, color, depth, K,
                    grounding_model, sam2_predictor, device,
                    results_by_id, init_masks,
                )
                raw_det_count += n
                if tr.lost_count == 0:
                    recovery_count += 1
        timings['discovery_ms'] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        if init_masks:
            self._cutie_reinit(color, init_masks)
        timings['reinit_ms'] += (time.perf_counter() - t0) * 1000.0

        timings['total_ms'] = (time.perf_counter() - t_frame) * 1000.0

        self._last_layer_stats = {
            'drift': drift_count,
            'layer2_ok': layer2_ok,
            'global_discovery': need_discovery,
            'discovery_eval': dict(self._last_discovery_eval),
            'discovery_skipped_stable': (
                not need_discovery
                and self._last_discovery_eval.get('mode') == 'stable'
                and self._last_discovery_eval.get('reason', '').startswith('wait_')
            ),
            'recovery_ok': recovery_count,
            'tracks': len(self.tracks),
            'timing_ms': {k: round(v, 1) for k, v in timings.items()},
        }

        if log_timing and self.logger:
            self.logger.info(
                f'[TrackManager] Frame {self._frame_idx} timing: '
                f'cutie={timings["cutie_ms"]:.0f}ms '
                f'layer2={timings["layer2_ms"]:.0f}ms '
                f'discovery={timings["discovery_ms"]:.0f}ms '
                f'reinit={timings["reinit_ms"]:.0f}ms '
                f'total={timings["total_ms"]:.0f}ms '
                f'tracks={len(self.tracks)} '
                f'discovery={"yes" if need_discovery else "no"}'
                f'({self._last_discovery_eval.get("mode", "?")}'
                f'/{self._last_discovery_eval.get("interval", 0)}f'
                f':{self._last_discovery_eval.get("reason", "?")})',
                throttle_duration_sec=0.5)

        self._prune_tracks()
        return list(results_by_id.values()), raw_det_count

    def _prune_tracks(self):
        before = len(self.tracks)
        roi_max = int(getattr(self.args, 'track_roi_fail_max_frames', 15))
        blacklist_frames = int(getattr(self.args, 'track_roi_blacklist_frames', 100))
        pruned_labels = []
        kept = []
        for t in self.tracks:
            if t.lost_count >= self.args.track_lost_max_frames:
                pruned_labels.append(f'{t.label}(lost)')
            elif t.roi_fail_count >= roi_max:
                pruned_labels.append(f'{t.label}(roi_fail={t.roi_fail_count})')
                self._roi_blacklist[t.label] = self._frame_idx + blacklist_frames
            else:
                kept.append(t)
        self.tracks = kept
        if pruned_labels and self.logger:
            self.logger.info(
                f'[TrackManager] Pruned {before}->{len(self.tracks)}: {pruned_labels}')

    def _is_blacklisted(self, label: str) -> bool:
        """Check if a label was recently pruned for ROI failure."""
        expiry = self._roi_blacklist.get(label)
        if expiry is None:
            return False
        if self._frame_idx >= expiry:
            del self._roi_blacklist[label]
            return False
        return True