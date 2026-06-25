"""Map Grounding DINO phrases to detection labels (tag or freeform prompt)."""
import re

from robot_perception.constants import LABEL_PROMPTS_MAP


# GDINO prompt 简化: 去掉对检测无帮助的形容词前缀
_COLOR_SIZE_PREFIX = re.compile(
    r'^(?:(?:white|black|red|blue|green|yellow|orange|purple|pink|grey|gray'
    r'|silver|golden|dark|light|bright|transparent|clear'
    r'|small|large|big|tiny|little|long|short|thin|thick|round|flat)\s+)+',
    re.IGNORECASE,
)

# GDINO caption 最大 prompt 数量 (超出截断，避免超长 caption 降低精度)
MAX_CAPTION_PROMPTS = 12


def simplify_for_gdino(name: str) -> str:
    """Strip color/size adjectives for GDINO — keep only core noun phrase.

    'black joint motor' → 'joint motor'
    'small red screwdriver' → 'screwdriver'
    'silver sheet metal bracket' → 'sheet metal bracket'
    """
    clean = name.strip().rstrip('.')
    simplified = _COLOR_SIZE_PREFIX.sub('', clean).strip()
    if not simplified:
        return clean
    return simplified


def normalize_prompt_key(prompt: str) -> str:
    """Normalize prompt/label for set comparison (case, trailing dot)."""
    return prompt.lower().strip().rstrip('.')


def prompts_for_tags(tags):
    """Return ordered (tag, prompt) pairs for active tags."""
    pairs = []
    for tag in tags:
        prompt = LABEL_PROMPTS_MAP.get(tag)
        if prompt is None:
            raise ValueError(f'Unknown tag "{tag}" — not in LABEL_PROMPTS_MAP')
        pairs.append((tag, prompt))
    return pairs


def build_combined_caption(text_prompts):
    """Build a single Grounding DINO caption from multiple prompts.

    Caps at MAX_CAPTION_PROMPTS to avoid degraded detection with overly long captions.
    """
    parts = []
    for prompt in text_prompts:
        p = prompt.lower().strip().rstrip('.')
        if p:
            parts.append(p)
    if not parts:
        return ''
    if len(parts) > MAX_CAPTION_PROMPTS:
        parts = parts[:MAX_CAPTION_PROMPTS]
    return '. '.join(parts) + '.'


def resolve_detection_targets(strings, allow_freeform=True):
    """Resolve a list of strings into detection targets.

    Each string is either a known tag (lookup in LABEL_PROMPTS_MAP) or a
    freeform text prompt when allow_freeform is True.

    Returns list of dicts: {label, prompt, is_tag}
    """
    targets = []
    for s in strings:
        s = s.strip()
        if not s:
            continue
        if s in LABEL_PROMPTS_MAP:
            targets.append({
                'label': s,
                'prompt': LABEL_PROMPTS_MAP[s],
                'is_tag': True,
            })
        elif allow_freeform:
            targets.append({
                'label': s,
                'prompt': s,
                'is_tag': False,
            })
        else:
            raise ValueError(f'Unknown tag "{s}" and freeform prompts are disabled')
    return targets


def build_prompt_to_label(targets):
    """Map normalized prompt text -> label for active targets."""
    mapping = {}
    for t in targets:
        key = _normalize_phrase(t['prompt'])
        if key in mapping and mapping[key] != t['label']:
            raise ValueError(
                f'Duplicate prompt "{key}" for labels {mapping[key]!r} and {t["label"]!r}')
        mapping[key] = t['label']
    return mapping


def _normalize_phrase(phrase):
    return phrase.lower().strip().rstrip('.')


def label_matches_target(label: str, target_name: str) -> bool:
    """Loose match between a track/detection label and a configured target name."""
    from robot_perception.utils.scene_understand import _label_matches_scene_object
    return _label_matches_scene_object(label, target_name)


def match_phrase_to_label(phrase, prompt_to_label, targets, accept_unmatched=False):
    """Match a GDINO phrase to a configured label.

    Returns label string, or None if unmatched and accept_unmatched is False.
    When accept_unmatched is True, returns the raw phrase as label.
    """
    norm = _normalize_phrase(phrase)
    if norm in prompt_to_label:
        return prompt_to_label[norm]

    labels = [t['label'] for t in targets]
    prompts = [_normalize_phrase(t['prompt']) for t in targets]

    best_label = None
    best_len = -1
    for label, prompt in zip(labels, prompts):
        if norm in prompt or prompt in norm:
            match_len = min(len(norm), len(prompt))
            if match_len > best_len:
                best_len = match_len
                best_label = label

    if best_label is not None:
        return best_label

    if accept_unmatched and norm:
        return phrase.strip()
    return None


def targets_from_tags(tags):
    """Build targets list from tag names only."""
    return [
        {'label': tag, 'prompt': prompt, 'is_tag': True}
        for tag, prompt in prompts_for_tags(tags)
    ]


def targets_from_text_prompts(text_prompts):
    """Build targets from arbitrary text prompts (label = prompt text)."""
    targets = []
    for prompt in text_prompts:
        p = prompt.strip()
        if not p:
            continue
        targets.append({'label': p, 'prompt': p, 'is_tag': False})
    return targets


def targets_from_scene_prompts(text_prompts, preserve_full_prompt=True):
    """Build targets from VLM scene-discovered prompts.

    label = full VLM description (display / track identity)
    prompt = GDINO caption phrase; by default keeps color/size (preserve_full_prompt)
    """
    targets = []
    seen_labels = set()
    for prompt in text_prompts:
        p = prompt.strip().rstrip('.')
        if not p:
            continue
        label_key = normalize_prompt_key(p)
        if label_key in seen_labels:
            continue
        seen_labels.add(label_key)
        if preserve_full_prompt:
            gdino_prompt = p
        else:
            gdino_prompt = simplify_for_gdino(p)
        targets.append({'label': p, 'prompt': gdino_prompt, 'is_tag': False})
    return targets
