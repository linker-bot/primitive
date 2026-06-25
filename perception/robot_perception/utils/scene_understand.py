"""Scene understanding module: VLM-driven object discovery + change detection."""
from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor

import cv2
import numpy as np

from robot_perception.utils.vlm_detector import _is_background_surface

_NORMALIZE_RE = re.compile(
    r'\b(small|large|big|tiny|little|long|short|thin|thick|round|flat|'
    r'clear|transparent|opaque|shiny|matte|plastic|metal|wooden|glass|rubber|'
    r'red|blue|green|yellow|black|white|gray|grey|orange|purple|pink|brown|'
    r'silver|golden|dark|light|bright|colored)\b',
    re.IGNORECASE,
)


def _normalize_object_name(name: str) -> str:
    """提取核心名词，去除颜色/尺寸/材质形容词，用于跨次 VLM 结果去重。"""
    clean = name.strip().rstrip('.')
    simplified = _NORMALIZE_RE.sub('', clean)
    simplified = re.sub(r'\s+', ' ', simplified).strip()
    return simplified.lower() if simplified else clean.lower()


_VAGUE_SCENE_RE = re.compile(
    r'\b(?:electronics?|cables?|wires?|parts?|components?|items?|objects?|'
    r'stuff|things|devices?|equipment|accessories?|box(?:es)?|container?s?|'
    r'plastic|metal|glass|rubber|wood(?:en)?|stationery)\b',
    re.IGNORECASE,
)

# Single-word labels that are too generic for GDINO (multi-word names are kept).
_VAGUE_SINGLE_WORDS = frozenset({
    'electronics', 'electronic', 'cable', 'cables', 'wire', 'wires',
    'part', 'parts', 'component', 'components', 'item', 'items',
    'object', 'objects', 'stuff', 'thing', 'things', 'device', 'devices',
    'equipment', 'accessory', 'accessories', 'box', 'boxes',
    'container', 'containers', 'plastic', 'metal', 'glass', 'rubber',
    'wood', 'wooden', 'stationery',
})


def _is_vague_scene_object(name: str) -> bool:
    """Over-broad VLM labels that cause GDINO false positives on the desk."""
    clean = name.strip().rstrip('.').lower()
    if not clean:
        return True
    words = clean.split()
    if len(words) >= 2:
        # "cardboard box", "wire bundle", "metal bracket" are specific enough
        return False
    if clean in _VAGUE_SINGLE_WORDS:
        return True
    return bool(_VAGUE_SCENE_RE.fullmatch(clean))


def _label_matches_scene_object(label: str, scene_obj: str) -> bool:
    """Loose match between detection label and scene object name."""
    ln = _normalize_object_name(label)
    sn = _normalize_object_name(scene_obj)
    if not ln or not sn:
        return False
    if ln == sn or ln in sn or sn in ln:
        return True
    ln_words = set(ln.split())
    sn_words = set(sn.split())
    return bool(ln_words & sn_words)


class SceneUnderstandManager:
    """Manages VLM scene understanding with change-based re-triggering.

    Flow:
      1. First frame → describe_scene() → discover objects (blocking, additive merge)
      2. Change / empty-scene → re-run describe_scene() (async, replace reconcile)
      3. Merge discovered objects with user-specified prompts; re-runs can remove stale labels

    Stability features:
      - 触发后更新参考帧，避免正反馈环路
      - 首次增量合并；重跑时用 VLM 快照替换 scene 列表（可删已离场物体）
      - drift/lost track 的 mask 不参与变化检测遮挡
      - 长期无检出的 scene prompt 自动 prune
    """

    def __init__(self, vlm_detector, args, logger=None):
        self._vlm = vlm_detector
        self._args = args
        self._logger = logger
        self._scene_objects: list[str] = []
        self._normalized_set: set[str] = set()
        self._scene_understood = False
        self._last_bg_gray: np.ndarray | None = None
        self._frame_counter = 0
        self._process_frame_counter = 0
        self._last_trigger_time = 0.0
        self._cooldown_s = float(getattr(args, 'scene_understand_cooldown', 10.0))
        self._pending_future: Future | None = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._consecutive_no_change = 0
        self._last_vlm_raw: list[str] = []
        self._last_merge_stats: dict = {}
        self._last_run_elapsed_s: float = 0.0
        self._last_parse_ok: bool = False
        self._prior_hints_valid: bool = False
        self._label_last_seen: dict[str, float] = {}
        self._object_added_at: dict[str, float] = {}
        self._ever_detected_norms: set[str] = set()
        self._empty_scene_streak: int = 0
        self._last_trigger_eval: dict = {'trigger': False, 'reason': 'init'}
        self._last_change_stats: dict = {}
        self._last_vlm_meta: dict = {}
        self._last_vlm_mode: str = ''

    @property
    def scene_objects(self) -> list[str]:
        return list(self._scene_objects)

    @property
    def last_vlm_raw(self) -> list[str]:
        return list(self._last_vlm_raw)

    @property
    def has_run(self) -> bool:
        return self._scene_understood

    def get_debug_info(self) -> dict:
        """Snapshot for track_debug / diagnostics."""
        return {
            'understood': self._scene_understood,
            'cumulative_objects': list(self._scene_objects),
            'last_vlm_raw': list(self._last_vlm_raw),
            'last_merge': dict(self._last_merge_stats),
            'last_run_s': round(self._last_run_elapsed_s, 2),
            'last_parse_ok': self._last_parse_ok,
            'prior_hints_valid': self._prior_hints_valid,
            'empty_scene_streak': self._empty_scene_streak,
            'label_last_seen': dict(self._label_last_seen),
            'ever_detected': sorted(self._ever_detected_norms),
            'pending_async': self._pending_future is not None
            and not self._pending_future.done(),
            'prior_hints': self._prior_objects_for_vlm(),
            'last_trigger_eval': dict(self._last_trigger_eval),
            'last_change_stats': dict(self._last_change_stats),
            'last_vlm_mode': self._last_vlm_mode,
            'last_vlm_meta': self._vlm_meta_for_debug(),
        }

    def _should_log_scene_prompts(self) -> bool:
        return bool(getattr(self._args, 'log_scene_prompts', True))

    @staticmethod
    def _one_line(text: str, max_len: int = 480) -> str:
        s = ' '.join(str(text).split())
        if len(s) <= max_len:
            return s
        return s[: max_len - 3] + '...'

    def _vlm_meta_for_debug(self) -> dict:
        m = dict(self._last_vlm_meta)
        if not m:
            return {}
        out = {
            'first_run': m.get('first_run'),
            'prior_objects': list(m.get('prior_objects') or []),
            'parse_stage': m.get('parse_stage'),
            'error': m.get('error'),
            'prompt_snip': self._one_line(m.get('prompt', ''), 160),
            'raw_snip': self._one_line(m.get('raw', ''), 200),
        }
        if self._should_log_scene_prompts():
            out['prompt'] = m.get('prompt', '')
            out['raw'] = m.get('raw', '')
        return out

    def _log_vlm_exchange(
            self, mode: str, meta: dict, objects: list[str], parse_ok: bool,
            elapsed_s: float):
        if not self._logger or not self._should_log_scene_prompts():
            return
        self._last_vlm_mode = mode
        prior = meta.get('prior_objects') or []
        stage = meta.get('parse_stage', '?')
        err = meta.get('error')
        prompt_line = self._one_line(meta.get('prompt', ''), 600)
        self._logger.info(
            f'[SceneUnderstand] VLM prompt ({mode}) prior={prior} '
            f'elapsed={elapsed_s:.1f}s stage={stage}: {prompt_line}')
        raw = meta.get('raw', '')
        if raw:
            self._logger.info(
                f'[SceneUnderstand] VLM response ({mode}): '
                f'{self._one_line(raw, 500)}')
        elif err:
            self._logger.warn(
                f'[SceneUnderstand] VLM failed ({mode}): {err}')
        if parse_ok:
            self._logger.info(
                f'[SceneUnderstand] VLM parsed ({mode}): objects={objects}')
        elif raw and not err:
            self._logger.warn(
                f'[SceneUnderstand] VLM parse failed ({mode}) stage={stage}')

    def _use_prior_hints(self) -> bool:
        return bool(getattr(self._args, 'scene_use_prior_hints', True))

    def _prior_objects_for_vlm(self) -> list[str]:
        if not self._use_prior_hints():
            return []
        if not self._prior_hints_valid or not self._scene_objects:
            return []
        return list(self._scene_objects)

    def _filter_object_candidates(self, vlm_objects: list[str]) -> tuple[list[str], dict]:
        """Apply background/vague filters; return kept list + filter stats."""
        kept = []
        filtered_vague = []
        filtered_bg = []
        for obj in vlm_objects:
            obj_clean = obj.strip()
            if not obj_clean:
                continue
            if _is_background_surface(obj_clean):
                filtered_bg.append(obj_clean)
                continue
            if _is_vague_scene_object(obj_clean):
                filtered_vague.append(obj_clean)
                continue
            kept.append(obj_clean)
        return kept, {
            'filtered_vague': filtered_vague,
            'filtered_bg': filtered_bg,
        }

    def _dedupe_objects(self, objects: list[str]) -> list[str]:
        out = []
        seen = set()
        for obj in objects:
            norm = _normalize_object_name(obj)
            if norm in seen:
                continue
            seen.add(norm)
            out.append(obj)
        return out

    def _touch_labels(self, labels: list[str]):
        now = time.time()
        for lab in labels:
            for obj in self._scene_objects:
                if _label_matches_scene_object(lab, obj):
                    norm = _normalize_object_name(obj)
                    self._ever_detected_norms.add(norm)
                    self._label_last_seen[norm] = now

    def record_frame_outcome(
            self, result_labels: list[str], num_tracks: int, raw_det_count: int):
        """Call once per processed frame — updates stale tracking and empty streak."""
        self._process_frame_counter += 1
        self._touch_labels(result_labels)

        if num_tracks == 0 and raw_det_count == 0 and not result_labels:
            self._empty_scene_streak += 1
        else:
            self._empty_scene_streak = 0

    def prune_stale_objects(self) -> list[str]:
        """Drop scene objects with no matching detection for stale_sec.

        Objects never detected by GDINO get a longer grace (never_detected_grace_sec).
        """
        stale_sec = float(getattr(self._args, 'scene_prompt_stale_sec', 45.0))
        never_grace = float(
            getattr(self._args, 'scene_prompt_never_detected_grace_sec', 120.0))
        if stale_sec <= 0 or not self._scene_objects:
            return []

        now = time.time()
        removed = []
        kept = []
        kept_norms = set()
        for obj in self._scene_objects:
            norm = _normalize_object_name(obj)
            if norm not in self._ever_detected_norms:
                added_at = self._object_added_at.get(norm, self._last_trigger_time)
                if (now - added_at) <= never_grace:
                    kept.append(obj)
                    kept_norms.add(norm)
                else:
                    removed.append(obj)
                continue

            last = self._label_last_seen.get(norm, 0.0)
            if last > 0 and (now - last) <= stale_sec:
                kept.append(obj)
                kept_norms.add(norm)
            else:
                removed.append(obj)

        if not removed:
            return []

        self._scene_objects = kept
        self._normalized_set = kept_norms
        for norm in {_normalize_object_name(o) for o in removed}:
            self._object_added_at.pop(norm, None)
            self._ever_detected_norms.discard(norm)
            self._label_last_seen.pop(norm, None)
        if self._logger:
            self._logger.info(
                f'[SceneUnderstand] Pruned stale scene objects (>{stale_sec:.0f}s '
                f'no detection, never_grace={never_grace:.0f}s): {removed} → '
                f'remaining={self._scene_objects}')
        return removed

    def _register_scene_object(self, obj_clean: str, now: float):
        norm = _normalize_object_name(obj_clean)
        self._normalized_set.add(norm)
        self._scene_objects.append(obj_clean)
        self._object_added_at[norm] = now

    def _merge_objects_additive(self, vlm_objects: list[str]) -> tuple[list[str], list[str]]:
        """First-run merge: add new objects only."""
        filtered, fstats = self._filter_object_candidates(vlm_objects)
        new_objects = []
        dedup = []
        now = time.time()
        for obj_clean in filtered:
            norm = _normalize_object_name(obj_clean)
            if norm in self._normalized_set:
                dedup.append(obj_clean)
                continue
            self._register_scene_object(obj_clean, now)
            new_objects.append(obj_clean)

        self._last_merge_stats = {
            'mode': 'additive',
            'raw_count': len(vlm_objects),
            'new_count': len(new_objects),
            'removed_count': 0,
            'filtered_vague': fstats['filtered_vague'],
            'filtered_bg': fstats['filtered_bg'],
            'dedup': dedup,
            'cumulative_count': len(self._scene_objects),
        }
        if self._logger and (new_objects or fstats['filtered_vague'] or fstats['filtered_bg']):
            self._logger.info(
                f'[SceneUnderstand] merge (first) raw={len(vlm_objects)} → '
                f'new={len(new_objects)} '
                f'(vague={fstats["filtered_vague"]}, bg={fstats["filtered_bg"]}, '
                f'dedup={dedup}), total={len(self._scene_objects)}')
        return new_objects, []

    def _reconcile_objects_replace(self, vlm_objects: list[str]) -> tuple[list[str], list[str]]:
        """Re-run merge: VLM snapshot replaces scene list (objects may be removed)."""
        filtered, fstats = self._filter_object_candidates(vlm_objects)
        incoming = self._dedupe_objects(filtered)
        incoming_norms = {_normalize_object_name(o) for o in incoming}
        old_norm_to_obj = {_normalize_object_name(o): o for o in self._scene_objects}

        removed = [
            old_norm_to_obj[n] for n in old_norm_to_obj
            if n not in incoming_norms
        ]
        added = [
            o for o in incoming
            if _normalize_object_name(o) not in old_norm_to_obj
        ]

        now = time.time()
        self._scene_objects = list(incoming)
        self._normalized_set = set(incoming_norms)
        for norm in incoming_norms:
            if norm not in self._object_added_at:
                self._object_added_at[norm] = now

        self._last_merge_stats = {
            'mode': 'replace',
            'raw_count': len(vlm_objects),
            'new_count': len(added),
            'removed_count': len(removed),
            'filtered_vague': fstats['filtered_vague'],
            'filtered_bg': fstats['filtered_bg'],
            'dedup': [],
            'cumulative_count': len(self._scene_objects),
            'removed': removed,
            'added': added,
        }
        if self._logger:
            self._logger.info(
                f'[SceneUnderstand] reconcile raw={len(vlm_objects)} → '
                f'added={added}, removed={removed}, '
                f'total={self._scene_objects}')
        return added, removed

    def _apply_vlm_objects(self, vlm_objects: list[str], refresh: bool) -> dict:
        if refresh:
            added, removed = self._reconcile_objects_replace(vlm_objects)
        else:
            added, removed = self._merge_objects_additive(vlm_objects)
        return {'added': added, 'removed': removed}

    def _describe_scene(
            self, color: np.ndarray, mode: str = 'blocking',
    ) -> tuple[list[str], bool]:
        max_objects = int(getattr(self._args, 'scene_max_objects', 10))
        is_first = not self._scene_understood
        prior = [] if is_first else self._prior_objects_for_vlm()
        t0 = time.time()
        objects, parse_ok, meta = self._vlm.describe_scene(
            color,
            max_objects=max_objects,
            prior_objects=prior or None,
            first_run=is_first,
        )
        elapsed = time.time() - t0
        self._last_vlm_meta = meta
        vlm_mode = 'first_run' if is_first else f'{mode}_rerun'
        self._log_vlm_exchange(vlm_mode, meta, objects, parse_ok, elapsed)
        return objects, parse_ok

    def should_trigger(self, color: np.ndarray, stable_masks: list[np.ndarray | None]) -> bool:
        """Determine if scene understanding should run this frame."""
        trigger, reason, extra = self._evaluate_trigger(color, stable_masks)
        self._last_trigger_eval = {'trigger': trigger, 'reason': reason, **extra}
        if trigger and self._logger:
            self._logger.info(
                f'[SceneUnderstand] Trigger VLM: {reason} '
                f'scene={self._scene_objects}')
        elif self._logger and extra.get('log_skip'):
            self._logger.info(
                f'[SceneUnderstand] Skip VLM: {reason} '
                f'scene={self._scene_objects}',
                throttle_duration_sec=2.0)
        return trigger

    def _evaluate_trigger(
            self, color: np.ndarray, stable_masks: list[np.ndarray | None]
    ) -> tuple[bool, str, dict]:
        if not self._scene_understood:
            now = time.time()
            retry_s = float(getattr(self._args, 'scene_first_run_retry_sec', 2.0))
            if now - self._last_trigger_time < retry_s:
                return False, f'first_run_cooldown ({now - self._last_trigger_time:.1f}s<{retry_s}s)', {}
            return True, 'first_run', {}

        force_frames = int(getattr(self._args, 'scene_force_refresh_empty_frames', 20))
        if self._empty_scene_streak >= force_frames:
            if self._pending_future is None:
                return True, f'force_empty_streak={self._empty_scene_streak}', {}

        self._frame_counter += 1
        check_interval = int(getattr(self._args, 'scene_change_check_interval', 10))
        if self._frame_counter % max(1, check_interval) != 0:
            return False, f'check_interval (frame {self._frame_counter % check_interval}/{check_interval})', {}

        if self._pending_future is not None:
            return False, 'async_pending', {}

        now = time.time()
        since_last = now - self._last_trigger_time
        if since_last < self._cooldown_s:
            return False, f'cooldown ({since_last:.1f}s<{self._cooldown_s:.0f}s)', {'log_skip': True}

        if self._detect_change(color, stable_masks):
            stats = self._last_change_stats
            return True, (
                f'change_detected ratio={stats.get("change_ratio", 0):.3f}'
                f'>{stats.get("threshold", 0):.2f}'
            ), {}

        stats = self._last_change_stats
        return False, (
            f'no_change ratio={stats.get("change_ratio", 0):.3f}'
            f'<={stats.get("threshold", 0):.2f} '
            f'({stats.get("changed_pixels", 0)}/{stats.get("untracked_pixels", 0)} px)'
        ), {'log_skip': True}

    def check_async_result(self) -> dict | None:
        """Poll async scene understanding. Returns {added, removed} or None."""
        if self._pending_future is None:
            return None
        if not self._pending_future.done():
            return None
        try:
            changes = self._pending_future.result()
        except Exception as e:
            if self._logger:
                self._logger.warn(f'[SceneUnderstand] Async run failed: {e}')
            changes = {'added': [], 'removed': []}
        self._pending_future = None
        if not changes.get('added') and not changes.get('removed'):
            return None
        return changes

    def run(self, color: np.ndarray) -> dict:
        """Execute scene understanding. First call blocking; later calls async."""
        if not self._scene_understood:
            return self._run_blocking(color)
        self._run_async(color)
        return {'added': [], 'removed': []}

    def _run_blocking(self, color: np.ndarray) -> dict:
        t0 = time.time()
        objects, parse_ok = self._describe_scene(color)
        self._last_run_elapsed_s = time.time() - t0
        self._last_vlm_raw = list(objects)
        self._last_parse_ok = parse_ok

        if not parse_ok:
            self._last_trigger_time = time.time()
            if self._logger:
                stage = self._last_vlm_meta.get('parse_stage', 'unknown')
                err = self._last_vlm_meta.get('error')
                detail = f'{stage}: {err}' if err else stage
                self._logger.warn(
                    f'[SceneUnderstand] VLM failed in '
                    f'{self._last_run_elapsed_s:.1f}s ({detail}) — will retry')
            return {'added': [], 'removed': []}

        if self._logger:
            if objects:
                self._logger.info(
                    f'[SceneUnderstand] Discovered {len(objects)} raw objects in '
                    f'{self._last_run_elapsed_s:.1f}s: '
                    f'{objects[:8]}{"..." if len(objects) > 8 else ""}')
            else:
                self._logger.warn(
                    f'[SceneUnderstand] VLM returned empty object list in '
                    f'{self._last_run_elapsed_s:.1f}s')

        changes = self._apply_vlm_objects(objects, refresh=False)
        self._scene_understood = True
        self._prior_hints_valid = len(self._scene_objects) > 0
        self._last_trigger_time = time.time()
        self._empty_scene_streak = 0
        self._save_background(color, [])
        self._log_scene_state('blocking', changes)
        return changes

    def _run_async(self, color: np.ndarray):
        image_copy = color.copy()
        self._pending_future = self._executor.submit(self._async_worker, image_copy)
        self._last_trigger_time = time.time()

    def _async_worker(self, color: np.ndarray) -> dict:
        t0 = time.time()
        objects, parse_ok = self._describe_scene(color, mode='async')
        self._last_run_elapsed_s = time.time() - t0
        self._last_vlm_raw = list(objects)
        self._last_parse_ok = parse_ok

        if not parse_ok:
            if self._logger:
                stage = self._last_vlm_meta.get('parse_stage', 'unknown')
                err = self._last_vlm_meta.get('error')
                detail = f'{stage}: {err}' if err else stage
                self._logger.warn(
                    f'[SceneUnderstand] Async VLM failed in '
                    f'{self._last_run_elapsed_s:.1f}s ({detail}) — keeping scene list')
            return {'added': [], 'removed': []}

        if self._logger:
            if objects:
                self._logger.info(
                    f'[SceneUnderstand] Async raw {len(objects)} objects in '
                    f'{self._last_run_elapsed_s:.1f}s: '
                    f'{objects[:8]}{"..." if len(objects) > 8 else ""}')
            else:
                self._logger.info(
                    f'[SceneUnderstand] Async VLM: empty table in '
                    f'{self._last_run_elapsed_s:.1f}s — clearing scene objects')

        changes = self._apply_vlm_objects(objects, refresh=True)
        if self._scene_objects:
            self._prior_hints_valid = True
        else:
            self._prior_hints_valid = False
        self._empty_scene_streak = 0
        self._save_background(color, [])
        self._log_scene_state('async', changes)
        return changes

    def _log_scene_state(self, mode: str, changes: dict):
        if not self._logger:
            return
        stats = self._last_merge_stats
        merge_mode = stats.get('mode', '?')
        filtered_vague = stats.get('filtered_vague', [])
        filtered_bg = stats.get('filtered_bg', [])
        self._logger.info(
            f'[SceneUnderstand] state ({mode}): '
            f'vlm_raw={self._last_vlm_raw[:8]}'
            f'{"..." if len(self._last_vlm_raw) > 8 else ""}, '
            f'added={changes.get("added", [])}, '
            f'removed={changes.get("removed", [])}, '
            f'cumulative={self._scene_objects}, '
            f'merge={merge_mode}, '
            f'filtered_vague={filtered_vague}, filtered_bg={filtered_bg}')

    def get_merged_prompts(self, user_prompts: list[str]) -> list[str]:
        """Merge scene-discovered objects with user-specified prompts (deduped)."""
        if not self._scene_objects:
            return list(user_prompts)

        existing_lower = set()
        for p in user_prompts:
            if not p:
                continue
            normalized = p.rstrip('.').strip().lower()
            existing_lower.add(normalized)
            for word in normalized.split():
                existing_lower.add(word)

        merged = list(user_prompts)
        for obj in self._scene_objects:
            obj_clean = obj.strip()
            if not obj_clean:
                continue
            obj_lower = obj_clean.lower()
            if obj_lower in existing_lower:
                continue
            words = obj_lower.split()
            if any(w in existing_lower for w in words if len(w) > 3):
                continue
            prompt = obj_clean if obj_clean.endswith('.') else obj_clean + '.'
            merged.append(prompt)
            existing_lower.add(obj_lower)

        return merged

    def _detect_change(self, color: np.ndarray, stable_masks: list[np.ndarray | None]) -> bool:
        """Lightweight grayscale diff on regions not covered by stable tracks."""
        gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        untracked_mask = np.ones((h, w), dtype=np.uint8)
        for mask in stable_masks:
            if mask is None:
                continue
            m = mask.astype(np.uint8) if mask.dtype != np.uint8 else mask
            if m.shape[:2] != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            untracked_mask[m > 0] = 0

        untracked_pixels = int(untracked_mask.sum())
        stable_mask_px = int(h * w - untracked_pixels)
        change_threshold = float(getattr(self._args, 'scene_change_threshold', 0.15))
        if untracked_pixels < 100:
            self._last_change_stats = {
                'change_ratio': 0.0,
                'changed_pixels': 0,
                'untracked_pixels': untracked_pixels,
                'stable_mask_px': stable_mask_px,
                'threshold': change_threshold,
                'blocked': 'untracked_too_small',
            }
            return False

        if self._last_bg_gray is None:
            self._save_background(gray, stable_masks)
            self._last_change_stats = {
                'change_ratio': 0.0,
                'changed_pixels': 0,
                'untracked_pixels': untracked_pixels,
                'stable_mask_px': stable_mask_px,
                'threshold': change_threshold,
                'blocked': 'no_background',
            }
            return False

        if self._last_bg_gray.shape != gray.shape:
            self._save_background(gray, stable_masks)
            self._last_change_stats = {
                'change_ratio': 1.0,
                'changed_pixels': untracked_pixels,
                'untracked_pixels': untracked_pixels,
                'stable_mask_px': stable_mask_px,
                'threshold': change_threshold,
                'blocked': 'shape_changed',
            }
            return True

        diff = cv2.absdiff(gray, self._last_bg_gray)
        diff_masked = diff * untracked_mask
        threshold = int(getattr(self._args, 'scene_change_pixel_threshold', 30))
        changed_pixels = int((diff_masked > threshold).sum())

        change_ratio = changed_pixels / max(1, untracked_pixels)
        self._last_change_stats = {
            'change_ratio': round(change_ratio, 4),
            'changed_pixels': changed_pixels,
            'untracked_pixels': untracked_pixels,
            'stable_mask_px': stable_mask_px,
            'threshold': change_threshold,
        }

        if change_ratio > change_threshold:
            self._consecutive_no_change = 0
            self._save_background(gray, stable_masks)
            return True

        self._consecutive_no_change += 1
        self._save_background(gray, stable_masks)
        return False

    def _save_background(self, gray_or_color: np.ndarray, tracked_masks):
        if gray_or_color.ndim == 3:
            self._last_bg_gray = cv2.cvtColor(gray_or_color, cv2.COLOR_RGB2GRAY)
        else:
            self._last_bg_gray = gray_or_color.copy()

    def shutdown(self):
        self._executor.shutdown(wait=False)
