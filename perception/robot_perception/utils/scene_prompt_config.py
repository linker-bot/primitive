"""Load user-editable VLM scene-understanding prompt templates from YAML."""
from __future__ import annotations

import os
from copy import deepcopy
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

# Built-in defaults (same as config/scene_understand_prompts.yaml).
_DEFAULT_SECTIONS: dict[str, str] = {
    'first_run_intro': (
        '观察机器人相机拍摄的工作台/桌面图像，忽略机械臂或机械手。\n'
        'Look at this workbench/table from a robot camera. '
        'Ignore any robotic arm or hand.\n'
    ),
    'rerun_intro': (
        '观察工作台/桌面图像，忽略画面中可见的机械臂或机械手。\n'
        'Look at this image of a workbench/table. '
        'Ignore any robotic arm or hand visible in the scene.\n'
    ),
    'rerun_body': (
        '列出当前桌面上所有可移动的物体。\n'
        'List ALL movable objects currently sitting on the table surface.\n'
    ),
    'naming_rules': (
        '命名规则 / Naming rules:\n'
        '- 列出夹爪可从桌面抓取的独立、可移动物体\n'
        '  List discrete, movable objects a gripper could pick up from the table\n'
        '- 每个物体用 2-4 个英文单词描述\n'
        '  Use specific 2-4 word ENGLISH phrases with color/size/material when visible\n'
        '  GOOD: "red screwdriver", "plastic bottle", "cardboard box", "sealed package"\n'
        '  BAD alone: part, item, thing, object, box, cable, component, stuff\n'
        '- 多个相似物体需区分 / distinguish similar items\n'
        '- 跳过桌面、固定线缆、显示器等 / Skip table surface, fixed cables, monitor\n'
    ),
    'json_rules': (
        '仅输出一行合法 JSON / Respond with ONLY valid JSON on one line.\n'
        '{"objects": ["red screwdriver", "clear plastic bottle"]}\n'
        'If empty: {"objects": []}\n'
        'IMPORTANT: every name in "objects" MUST be English (not Chinese).'
    ),
    'prior_block': (
        '\nContext — objects seen in a PREVIOUS analysis (some may have '
        'been removed): {prior_names}\n'
        '- List ONLY objects currently visible on the table NOW\n'
        '- Re-include previous objects only if still clearly visible\n'
        '- Do NOT list objects that were removed from the table\n'
        '- Add any newly appeared movable objects with specific names\n'
    ),
    'max_objects_line': 'Maximum {max_objects} objects.\n',
    'rerun_max_objects_line': '- At most {max_objects} objects\n',
}


def scene_gdino_prompt_phrase(label: str, preserve_full: bool = True) -> str:
    """GDINO phrase for a scene object label."""
    p = label.strip().rstrip('.')
    if preserve_full:
        return p
    return simplify_for_gdino(p)


def default_package_config_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(
            get_package_share_directory('robot_perception'),
            'config',
            'scene_understand_prompts.yaml',
        )
    except Exception:
        # Dev / non-ROS: config next to perception package root
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.abspath(
            os.path.join(here, '..', '..', 'config', 'scene_understand_prompts.yaml'))


def _normalize_yaml_sections(raw: dict[str, Any]) -> dict[str, str]:
    first = raw.get('first_run') or {}
    rerun = raw.get('rerun') or {}
    sections = {
        'first_run_intro': str(first.get('intro', '')).strip() + '\n',
        'rerun_intro': str(rerun.get('intro', '')).strip() + '\n',
        'rerun_body': str(rerun.get('body', '')).strip() + '\n',
        'naming_rules': str(raw.get('naming_rules', '')).strip() + '\n',
        'json_rules': str(raw.get('json_rules', '')).strip(),
        'prior_block': str(raw.get('prior_block', '')).strip() + '\n',
        'max_objects_line': str(raw.get('max_objects_line', '')).strip() + '\n',
        'rerun_max_objects_line': (
            str(raw.get('rerun_max_objects_line', '')).strip() + '\n'
        ),
    }
    for key, val in sections.items():
        if not val.strip():
            sections[key] = _DEFAULT_SECTIONS[key]
    return sections


class ScenePromptConfig:
    """Template sections for VLM describe_scene prompts."""

    def __init__(self, sections: dict[str, str] | None = None, source: str = 'builtin'):
        self.sections = deepcopy(_DEFAULT_SECTIONS)
        if sections:
            self.sections.update(sections)
        self.source = source

    def build_prompt(
            self, max_objects: int, prior_objects: list[str] | None = None,
            first_run: bool = False) -> str:
        s = self.sections
        naming = s['naming_rules']
        json_rules = s['json_rules']
        if first_run:
            return (
                f"{s['first_run_intro']}"
                f"{naming}"
                f"{s['max_objects_line'].format(max_objects=max_objects)}"
                f"{json_rules}"
            )

        base = f"{s['rerun_intro']}{s['rerun_body']}"
        prior_block = ''
        if prior_objects:
            cleaned = [
                str(o).strip().rstrip('.')
                for o in prior_objects
                if o and str(o).strip()
            ]
            if cleaned:
                names = ', '.join(cleaned[:max_objects])
                prior_block = s['prior_block'].format(prior_names=names)
                if not prior_block.startswith('\n'):
                    prior_block = '\n' + prior_block

        return (
            f"{base}"
            f"{prior_block}"
            f"{naming}"
            f"{s['rerun_max_objects_line'].format(max_objects=max_objects)}"
            f"{json_rules}"
        )


def load_scene_prompt_config(
        path: str | None = None, logger=None) -> ScenePromptConfig:
    """Load prompt templates from YAML; fall back to built-in defaults."""
    resolved = (path or '').strip()
    if not resolved:
        resolved = default_package_config_path()

    if not os.path.isfile(resolved):
        if logger and path:
            logger.warn(
                f'[ScenePrompt] Config not found: {resolved} — using built-in defaults')
        return ScenePromptConfig(source='builtin')

    if yaml is None:
        if logger:
            logger.warn('[ScenePrompt] PyYAML unavailable — using built-in defaults')
        return ScenePromptConfig(source='builtin')

    try:
        with open(resolved, encoding='utf-8') as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError('root must be a mapping')
        sections = _normalize_yaml_sections(raw)
        if logger:
            logger.info(f'[ScenePrompt] Loaded VLM prompt config: {resolved}')
        return ScenePromptConfig(sections=sections, source=resolved)
    except Exception as e:
        if logger:
            logger.warn(
                f'[ScenePrompt] Failed to load {resolved}: {e} — using built-in defaults')
        return ScenePromptConfig(source='builtin')
