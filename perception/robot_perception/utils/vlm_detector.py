"""HTTP VLM bbox detector for detection_bbox (OpenAI-compatible, e.g. local MiniCPM serve)."""
from __future__ import annotations

import base64
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Response parsing utilities
# ---------------------------------------------------------------------------

_NEGATIVE_PATTERNS = re.compile(
    r'cannot\s+(?:find|see|detect|locate)|not\s+(?:visible|found|present|detected)'
    r'|no\s+(?:matching|such|visible)\s+object'
    r'|(?:未找到|没有|无法识别|图中没有|不存在)',
    re.IGNORECASE,
)


def _is_negative_response(text: str) -> bool:
    """Detect VLM responses indicating nothing was found."""
    clean = text.strip().lower()
    if clean in ('{}', '{"objects": []}', '{"objects":[]}', '[]'):
        return True
    if _NEGATIVE_PATTERNS.search(text):
        if not re.search(r'\[\s*\d', text):
            return True
    return False


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0].strip()
    return text


def _fix_space_separated_bbox_arrays(text: str) -> str:
    """Fix invalid JSON like [730 300 1000 600] -> [730, 300, 1000, 600]."""

    def repl(match):
        nums = re.findall(r'-?\d+', match.group(1))
        if len(nums) == 4:
            return '[' + ', '.join(nums) + ']'
        return match.group(0)

    return re.sub(r'\[(\s*-?\d+(?:\s+-?\d+){3})\s*\]', repl, text)


def _parse_vlm_objects(text: str):
    """Parse VLM response into a list of raw object entries; never raises."""
    text = _strip_markdown_fences(text)
    candidates = [text, _fix_space_separated_bbox_arrays(text)]

    brace_start = text.find('{')
    if brace_start >= 0:
        depth = 0
        for idx in range(brace_start, len(text)):
            ch = text[idx]
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    snippet = _fix_space_separated_bbox_arrays(text[brace_start:idx + 1])
                    candidates.append(snippet)
                    break

    seen = set()
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            objects = parsed.get('objects')
            if isinstance(objects, list):
                return objects
            if 'name' in parsed and ('bbox' in parsed or 'box' in parsed):
                return [parsed]
    return None


def _coerce_bbox_values(bbox):
    """Return four numeric coords or None."""
    if bbox is None:
        return None
    if isinstance(bbox, str):
        nums = re.findall(r'-?\d+(?:\.\d+)?', bbox)
        if len(nums) != 4:
            return None
        bbox = nums
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def _normalize_bbox_object(obj):
    """Convert one VLM object entry to {name, bbox} or None."""
    if isinstance(obj, str):
        return None
    if not isinstance(obj, dict):
        return None
    name = obj.get('name') or obj.get('label') or obj.get('class') or obj.get('phrase')
    bbox = _coerce_bbox_values(
        obj.get('bbox') or obj.get('box') or obj.get('box_xyxy'))
    if not name or bbox is None:
        return None
    return {'name': str(name).strip(), 'bbox': bbox}


def _parse_vlm_freeform(text: str):
    """Fallback parser: extract name + bbox from various freeform formats.

    Supports:
      - 'label 158 22 910 656'           (space-separated trailing bbox)
      - 'label [158, 22, 910, 656]'      (bracket bbox)
      - 'label: [158, 22, 910, 656]'     (colon separator)
      - Multi-line with one object per line
    """
    text = _strip_markdown_fences(text).strip()
    if not text or text.startswith('{'):
        return []

    results = []
    lines = text.split('\n')
    for line in lines:
        line = line.strip().lstrip('-•*').strip()
        if not line:
            continue
        obj = _parse_single_freeform_line(line)
        if obj:
            results.append(obj)

    if results:
        return results

    obj = _parse_single_freeform_line(text)
    return [obj] if obj else []


def _parse_single_freeform_line(line: str):
    """Parse a single freeform line into {name, bbox} or None."""
    bracket_match = re.search(r'[\[(]\s*(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)[,\s]+(-?\d+)\s*[\])]', line)
    if bracket_match:
        bbox = [float(bracket_match.group(i)) for i in range(1, 5)]
        prefix = line[:bracket_match.start()].strip().rstrip(':：-–—,').strip()
        name = prefix if prefix else 'object'
        return {'name': name, 'bbox': bbox}

    match = re.search(r'(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$', line)
    if match:
        bbox = [float(match.group(i)) for i in range(1, 5)]
        prefix = line[:match.start()].strip().rstrip(',').strip()
        if not prefix:
            return {'name': 'object', 'bbox': bbox}
        parts = [part.strip() for part in prefix.split(',') if part.strip()]
        label = parts[-1] if parts else prefix
        label = re.sub(r'\d+\s*$', '', label).strip() or label
        return {'name': label, 'bbox': bbox}

    return None


def _build_describe_scene_prompt(
        max_objects, prior_objects=None, first_run=False,
        prompt_config=None):
    """Build VLM scene-enumeration prompt."""
    if prompt_config is not None:
        return prompt_config.build_prompt(
            max_objects,
            prior_objects=prior_objects if not first_run else None,
            first_run=first_run,
        )

    # Fallback when no config object (tests / legacy)
    from robot_perception.utils.scene_prompt_config import ScenePromptConfig
    return ScenePromptConfig().build_prompt(
        max_objects,
        prior_objects=prior_objects if not first_run else None,
        first_run=first_run,
    )


def _scene_object_names_from_parsed(parsed_objects) -> list[str]:
    """Coerce _extract_vlm_objects output to plain name strings."""
    names = []
    for entry in parsed_objects:
        if isinstance(entry, str):
            name = entry.strip()
        elif isinstance(entry, dict):
            name = (
                entry.get('name')
                or entry.get('label')
                or entry.get('object')
                or ''
            )
            name = str(name).strip()
        else:
            continue
        if name:
            names.append(name)
    return names


def _parse_vlm_object_name_list(text: str):
    """Fallback for describe_scene: extract object names without bboxes.

    Handles common MiniCPM / VLM formats that ignore strict JSON instructions:
      - comma-separated:  a, b, c
      - mixed quotes:    large bottle "green screwdriver" "black cable"
      - bullet / numbered lists (one name per line)
    """
    text = _strip_markdown_fences(text).strip()
    if not text or text.startswith('{'):
        return None

    text = re.sub(r'^(?:objects?|items?)\s*[:\：]\s*', '', text, flags=re.IGNORECASE)

    if ',' in text:
        parts = [p.strip().strip('"\'') for p in text.split(',')]
        names = [p for p in parts if p and len(p) > 1]
        if names:
            return names

    quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', text)
    names = [q[0] or q[1] for q in quoted if q[0] or q[1]]
    if names:
        first_quote = len(text)
        for ch in ('"', "'"):
            pos = text.find(ch)
            if pos >= 0:
                first_quote = min(first_quote, pos)
        if first_quote > 0:
            prefix = text[:first_quote].strip().strip(',').strip()
            if prefix and len(prefix.split()) <= 8:
                names.insert(0, prefix)
        return names

    lines = []
    for line in text.split('\n'):
        line = re.sub(r'^\s*(?:[-•*]|\d+[.)])\s*', '', line.strip())
        line = line.strip().strip('"\'')
        if line and len(line) > 1:
            lines.append(line)
    if len(lines) >= 2:
        return lines

    return None


def _extract_vlm_objects(text: str):
    """Parse VLM response: negative check → JSON → freeform fallback."""
    if _is_negative_response(text):
        return []

    objects = _parse_vlm_objects(text)
    if objects is not None:
        if objects and all(isinstance(entry, str) for entry in objects):
            freeform = _parse_vlm_freeform(text)
            if freeform:
                print(
                    '[VLM] Parsed name-list + trailing bbox via freeform fallback',
                    file=sys.stderr,
                    flush=True,
                )
                return freeform
        return objects

    freeform = _parse_vlm_freeform(text)
    if freeform:
        print(
            '[VLM] Parsed non-JSON response via freeform fallback',
            file=sys.stderr,
            flush=True,
        )
        return freeform

    name_list = _parse_vlm_object_name_list(text)
    if name_list:
        print(
            f'[VLM] Parsed name-list via text fallback ({len(name_list)} objects)',
            file=sys.stderr,
            flush=True,
        )
        return name_list

    return None


# ---------------------------------------------------------------------------
# Scene understanding post-filter: remove active robot system from VLM results
# (individual parts like motors/servos on the table should NOT be filtered)
# ---------------------------------------------------------------------------

_ROBOT_SYSTEM_KEYWORDS = re.compile(
    r'\b(?:robot(?:ic)?|mechanical|dexterous|linker)\s*(?:hand|arm|gripper)\b'
    r'|\b(?:end[\s-]?effector|manipulator)\b'
    r'|\b(?:robot\s+base|robotic\s+arm)\b',
    re.IGNORECASE,
)

# 如果名称同时包含以下"零件"关键词，说明是台面上的独立零件，不应过滤
_PART_INDICATORS = re.compile(
    r'\b(?:motor|servo|actuator|joint|module|unit|assembly|part|component|spare'
    r'|cable|connector|bracket|housing|cover|shell|finger\s*tip)\b',
    re.IGNORECASE,
)


def _is_robot_part(name: str) -> bool:
    """Return True if name refers to the active robot system (not a loose part)."""
    if not _ROBOT_SYSTEM_KEYWORDS.search(name):
        return False
    if _PART_INDICATORS.search(name):
        return False
    return True


_BACKGROUND_SURFACE_KEYWORDS = re.compile(
    r'\b(?:table|desk|workbench|countertop|counter|surface|floor|ground|'
    r'wall|background|mat|pad|cloth|tabletop|desktop)\b',
    re.IGNORECASE,
)


def _is_background_surface(name: str) -> bool:
    """Return True if VLM name refers to table/workbench/background, not a graspable object."""
    clean = name.strip().rstrip('.').lower()
    if not clean:
        return True
    return bool(_BACKGROUND_SURFACE_KEYWORDS.search(clean))


# ---------------------------------------------------------------------------
# VLMDetector class
# ---------------------------------------------------------------------------

class VLMDetector:
    """Open-vocabulary bbox detection via a VLM chat-completions API."""

    def __init__(self, api_key, base_url, model, keepalive_interval=120, debug=False,
                 scene_prompt_config=None):
        from openai import OpenAI
        import httpx

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=httpx.Client(trust_env=False, timeout=60.0),
        )
        self.model = model
        self.debug = debug
        self.scene_prompt_config = scene_prompt_config
        self._executor = ThreadPoolExecutor(max_workers=4)

    def warmup(self):
        print('[VLM] Warming up endpoint...', file=sys.stderr, flush=True)
        t0 = time.time()
        try:
            self.client.chat.completions.create(
                model=self.model,
                messages=[{'role': 'user', 'content': [{'type': 'text', 'text': 'hello'}]}],
                max_tokens=1,
                temperature=0.0,
            )
            print(f'[VLM] Warmup done in {time.time() - t0:.1f}s', file=sys.stderr, flush=True)
        except Exception as e:
            print(
                f'[VLM] Warmup failed in {time.time() - t0:.1f}s: {e}',
                file=sys.stderr,
                flush=True,
            )

    def _encode_image(self, image_rgb, quality=85):
        bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        _, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buf).decode('utf-8')

    def _parse_and_scale(self, text, orig_w, orig_h):
        """Parse VLM response text and convert 0-1000 coords to pixel coords."""
        objects = _extract_vlm_objects(text)
        if objects is None:
            print(
                '[VLM] Failed to parse response — treating as no detections',
                file=sys.stderr,
                flush=True,
            )
            return []

        detections = []
        skipped_names = []
        for obj in objects:
            normalized = _normalize_bbox_object(obj)
            if normalized is None:
                if isinstance(obj, str):
                    skipped_names.append(obj)
                elif isinstance(obj, dict):
                    name = obj.get('name') or obj.get('label') or obj.get('class')
                    if name:
                        skipped_names.append(str(name))
                continue
            x1, y1, x2, y2 = normalized['bbox']
            pixel_bbox = [
                max(0, min(int(x1 / 1000 * orig_w), orig_w)),
                max(0, min(int(y1 / 1000 * orig_h), orig_h)),
                max(0, min(int(x2 / 1000 * orig_w), orig_w)),
                max(0, min(int(y2 / 1000 * orig_h), orig_h)),
            ]
            if pixel_bbox[2] <= pixel_bbox[0] + 1 or pixel_bbox[3] <= pixel_bbox[1] + 1:
                continue
            detections.append({
                'bbox': pixel_bbox,
                'label': normalized['name'],
            })

        if skipped_names:
            print(
                f'[VLM] Ignored {len(skipped_names)} name-only entries (no bbox): '
                f'{skipped_names[:6]}',
                file=sys.stderr,
                flush=True,
            )
        return detections

    def detect(self, image_rgb, text_prompt):
        """Return list of dicts: bbox [x1,y1,x2,y2] pixels, label str."""
        orig_h, orig_w = image_rgb.shape[:2]
        b64 = self._encode_image(image_rgb)

        prompt_text = (
            f'You are a robot vision detector. Find object(s) matching: {text_prompt}\n'
            'Output exactly one line of valid JSON. No markdown, no explanation, no thinking.\n'
            'Format: {{"objects": [{{"name": "<label>", "bbox": [x1, y1, x2, y2]}}]}}\n'
            'Rules:\n'
            '- bbox: integer coordinates normalized to 0-1000 (0=top-left, 1000=bottom-right)\n'
            '- bbox must fully enclose the entire object\n'
            '- If no match: {{"objects": []}}'
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                    {'type': 'text', 'text': prompt_text},
                ],
            }],
            max_tokens=1024,
            temperature=0.0,
        )

        text = response.choices[0].message.content.strip()
        if self.debug:
            print(f'[VLM DEBUG] Raw response: {text}', file=sys.stderr, flush=True)

        detections = self._parse_and_scale(text, orig_w, orig_h)
        if self.debug:
            print(
                f'[VLM DEBUG] Image size: {orig_w}x{orig_h}, '
                f'detections={len(detections)}',
                file=sys.stderr,
                flush=True,
            )
        return detections

    def detect_batch(self, image_rgb, object_names):
        """Detect multiple objects in one API call. Returns list of {bbox, label} dicts."""
        if not object_names:
            return []
        orig_h, orig_w = image_rgb.shape[:2]
        b64 = self._encode_image(image_rgb)

        names_str = ', '.join(n.strip().rstrip('.') for n in object_names if n.strip())
        prompt_text = (
            f'You are a robot vision detector. Find these objects in the image: {names_str}\n'
            'For each found object, output name and bbox. Skip objects not found.\n'
            'Output exactly one line of valid JSON. No markdown, no explanation, no thinking.\n'
            'Format: {{"objects": [{{"name": "<exact name from list>", "bbox": [x1, y1, x2, y2]}}]}}\n'
            'Rules:\n'
            '- name must exactly match one of the requested names (copy verbatim)\n'
            '- at most one bbox per object\n'
            '- bbox: integer coordinates normalized to 0-1000 (0=top-left, 1000=bottom-right)\n'
            '- bbox must fully enclose the entire object\n'
            '- If none found: {{"objects": []}}'
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        {'type': 'text', 'text': prompt_text},
                    ],
                }],
                max_tokens=2048,
                temperature=0.0,
            )
        except Exception as e:
            print(f'[VLM] Batch detect API error: {e}', file=sys.stderr, flush=True)
            return None

        text = response.choices[0].message.content.strip()
        if self.debug:
            print(f'[VLM DEBUG] Batch raw: {text}', file=sys.stderr, flush=True)

        detections = self._parse_and_scale(text, orig_w, orig_h)
        if self.debug:
            print(
                f'[VLM DEBUG] Batch result: {len(detections)} detections',
                file=sys.stderr,
                flush=True,
            )
        return detections

    def detect_parallel(self, image_rgb, prompts):
        futures = {}
        for prompt in prompts:
            futures[self._executor.submit(self.detect, image_rgb, prompt)] = prompt
        results = {}
        for future in as_completed(futures):
            prompt = futures[future]
            try:
                results[prompt] = future.result()
            except Exception as e:
                print(
                    f"[VLM] Parallel detect failed for '{prompt}': {e}",
                    file=sys.stderr,
                    flush=True,
                )
                results[prompt] = []
        return results

    def describe_scene(self, image_rgb, max_objects=10, prior_objects=None,
                       first_run=False):
        """Ask VLM to enumerate graspable objects on the workbench/table.

        Returns (object_names, parse_ok, meta).

        meta keys: prompt, raw, first_run, prior_objects, error, parse_stage
        """
        b64 = self._encode_image(image_rgb)
        prompt_text = _build_describe_scene_prompt(
            max_objects,
            prior_objects=prior_objects if not first_run else None,
            first_run=first_run,
            prompt_config=self.scene_prompt_config,
        )
        meta = {
            'prompt': prompt_text,
            'raw': '',
            'first_run': first_run,
            'prior_objects': list(prior_objects or []),
            'error': None,
            'parse_stage': 'ok',
        }

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{b64}'}},
                        {'type': 'text', 'text': prompt_text},
                    ],
                }],
                max_tokens=512,
                temperature=0.0,
            )
        except Exception as e:
            meta['error'] = str(e)
            meta['parse_stage'] = 'connection_error'
            print(f'[VLM] describe_scene failed: {e}', file=sys.stderr, flush=True)
            return [], False, meta

        text = response.choices[0].message.content.strip()
        meta['raw'] = text
        if self.debug:
            print(f'[VLM DEBUG] Scene describe raw: {text}', file=sys.stderr, flush=True)

        if not text:
            meta['parse_stage'] = 'empty_response'
            print('[VLM] describe_scene: empty response from model', file=sys.stderr, flush=True)
            return [], False, meta

        parsed = _extract_vlm_objects(text)
        if parsed is None:
            meta['parse_stage'] = 'parse_error'
            snippet = text.replace('\n', ' ')[:400]
            print(
                f'[VLM] describe_scene: failed to parse response '
                f'(first_run={first_run}, prior={bool(prior_objects)}): {snippet!r}',
                file=sys.stderr,
                flush=True,
            )
            return [], False, meta

        result = _scene_object_names_from_parsed(parsed)
        filtered = [o for o in result if _is_robot_part(o)]
        if filtered:
            print(f'[VLM] describe_scene: filtered robot parts: {filtered}',
                  file=sys.stderr, flush=True)
        result = [o for o in result if not _is_robot_part(o)]
        filtered_bg = [o for o in result if _is_background_surface(o)]
        if filtered_bg:
            print(f'[VLM] describe_scene: filtered background surfaces: {filtered_bg}',
                  file=sys.stderr, flush=True)
        result = [o for o in result if not _is_background_surface(o)]
        if not result and parsed:
            meta['parse_stage'] = 'all_filtered'
            print(f'[VLM] describe_scene: all {len(parsed)} objects filtered or empty',
                  file=sys.stderr, flush=True)
        return result[:max_objects], True, meta


# ---------------------------------------------------------------------------
# GDINO-compatible wrappers
# ---------------------------------------------------------------------------

def vlm_to_gdino_detections(vlm_detections, default_score=0.85):
    """Convert VLM bbox dicts to GDINO-style phrase/score/box_xyxy records."""
    out = []
    for det in vlm_detections:
        bbox = det.get('bbox')
        if not bbox or len(bbox) != 4:
            continue
        out.append({
            'phrase': det.get('label', 'object'),
            'score': float(default_score),
            'box_xyxy': np.array(bbox, dtype=np.float32),
        })
    return out


def vlm_detect_as_gdino(vlm_detector, image_rgb, caption, default_score=0.85):
    """Full-image VLM detect; returns GDINO-compatible detection list."""
    raw = vlm_detector.detect(image_rgb, caption)
    return vlm_to_gdino_detections(raw, default_score=default_score)


def _nms_detections(detections, iou_threshold=0.5):
    """Simple NMS across VLM detections to remove overlapping boxes."""
    if len(detections) <= 1:
        return detections
    boxes = np.array([d['box_xyxy'] for d in detections], dtype=np.float32)
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    order = np.argsort(-areas)
    keep = []
    suppressed = set()
    for i in order:
        if int(i) in suppressed:
            continue
        keep.append(int(i))
        for j in order:
            if int(j) in suppressed or int(j) == int(i):
                continue
            xx1 = max(boxes[i, 0], boxes[j, 0])
            yy1 = max(boxes[i, 1], boxes[j, 1])
            xx2 = min(boxes[i, 2], boxes[j, 2])
            yy2 = min(boxes[i, 3], boxes[j, 3])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[j] - inter + 1e-6)
            if iou > iou_threshold:
                suppressed.add(int(j))
    return [detections[i] for i in keep]


def _filter_vlm_boxes(detections, image_rgb, max_area_ratio=0.5, max_aspect=6.0):
    """Remove VLM boxes that are unreasonably large or have extreme aspect ratios."""
    if not detections:
        return detections
    img_h, img_w = image_rgb.shape[:2]
    img_area = img_h * img_w
    out = []
    for det in detections:
        x1, y1, x2, y2 = det['box_xyxy']
        w = x2 - x1
        h = y2 - y1
        if w < 2 or h < 2:
            continue
        area = w * h
        if area / img_area > max_area_ratio:
            continue
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect > max_aspect:
            continue
        out.append(det)
    return out


def vlm_detect_multi_as_gdino(vlm_detector, image_rgb, text_prompts, default_score=0.85,
                              nms_iou=0.5, max_area_ratio=0.5):
    """Detect multiple objects — batch first, parallel per-prompt as fallback."""
    prompts = [p for p in text_prompts if p and str(p).strip()]
    if not prompts:
        return []
    if len(prompts) == 1:
        dets = vlm_detect_as_gdino(vlm_detector, image_rgb, prompts[0], default_score=default_score)
        return _filter_vlm_boxes(dets, image_rgb, max_area_ratio=max_area_ratio)

    batch_dets = vlm_detector.detect_batch(image_rgb, prompts)
    if batch_dets is not None and len(batch_dets) > 0:
        out = vlm_to_gdino_detections(batch_dets, default_score=default_score)
        for det in out:
            det['phrase'] = det.get('phrase', 'object')
        out = _filter_vlm_boxes(out, image_rgb, max_area_ratio=max_area_ratio)
        out = _nms_detections(out, iou_threshold=nms_iou)
        if out:
            return out

    results_map = vlm_detector.detect_parallel(image_rgb, prompts)
    out = []
    for prompt, raw_dets in results_map.items():
        for det in vlm_to_gdino_detections(raw_dets, default_score=default_score):
            det['phrase'] = prompt
            out.append(det)
    out = _filter_vlm_boxes(out, image_rgb, max_area_ratio=max_area_ratio)
    out = _nms_detections(out, iou_threshold=nms_iou)
    return out
