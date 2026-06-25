"""Shared third-party model paths for perception nodes."""
import os
import sys


def _repo_root():
    current_file = os.path.realpath(__file__)
    d = os.path.dirname(current_file)
    for _ in range(9):
        if os.path.isdir(os.path.join(d, '.git')):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.abspath(os.path.join(os.path.dirname(current_file), '..', '..', '..'))


def resolve_weights_path(path: str) -> str:
    """Expand ${WEIGHTS_BASE} and repo-relative data/weights paths."""
    if not path:
        return path
    base = _resolve_weights_base()
    if '${WEIGHTS_BASE}' in path:
        return path.replace('${WEIGHTS_BASE}', base)
    if path.startswith('data/weights/'):
        return os.path.join(_repo_root(), path)
    return path


def _resolve_model_repo_base():
    env = os.environ.get('ROBOT_PERCEPTION_DIR')
    if env:
        env = os.path.abspath(os.path.expanduser(env))
        if os.path.isdir(env):
            return env

    current_file = os.path.realpath(__file__)
    candidates = []
    for depth in range(3, 9):
        root = os.path.abspath(os.path.join(current_file, *(['..'] * depth)))
        candidates.extend([
            os.path.join(root, 'src', 'third_party', 'robot_perception'),
            os.path.join(root, 'third_party', 'robot_perception'),
            os.path.join(root, '..', 'third_party', 'robot_perception'),
        ])

    seen = set()
    for path in candidates:
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        if os.path.isdir(path):
            return path

    return os.path.abspath(candidates[0])


def _resolve_weights_base():
    """Resolve unified weights base directory.

    Priority:
      1. Environment variable PERCEPTION_WEIGHTS_DIR
      2. data/weights/ at git repository root
    """
    env = os.environ.get('PERCEPTION_WEIGHTS_DIR')
    if env:
        env = os.path.abspath(os.path.expanduser(env))
        if os.path.isdir(env):
            return env

    return os.path.join(_repo_root(), 'data', 'weights')


_MODEL_REPO_BASE = _resolve_model_repo_base()
GSAM2_DIR = os.path.join(_MODEL_REPO_BASE, 'Grounded-SAM-2')
GSAM2_GDINO_DIR = os.path.join(GSAM2_DIR, 'grounding_dino')
FS_DIR = os.path.join(_MODEL_REPO_BASE, 'FoundationStereo')
CUTIE_DIR = os.path.join(_MODEL_REPO_BASE, 'Cutie')
PIXAL3D_DIR = os.path.join(_MODEL_REPO_BASE, 'Pixal3D')

# --- Unified weights paths ---
WEIGHTS_BASE = _resolve_weights_base()

SAM2_CHECKPOINT = os.path.join(WEIGHTS_BASE, 'sam2', 'sam2.1_hiera_tiny.pt')
GDINO_CONFIG = os.path.join(WEIGHTS_BASE, 'grounding_dino', 'GroundingDINO_SwinT_OGC.py')
GDINO_CHECKPOINT = os.path.join(WEIGHTS_BASE, 'grounding_dino', 'groundingdino_swint_ogc.pth')
STEREO_PLAN_PATH = os.path.join(WEIGHTS_BASE, 'foundation_stereo', 'foundation_stereo.plan')
CUTIE_WEIGHT_DIR = os.path.join(WEIGHTS_BASE, 'cutie')
PIXEL3D_MODEL_PATH = os.path.join(WEIGHTS_BASE, 'pixal3d-t')
MINICPM_WEIGHT_DIR = os.environ.get(
    'MINICPM_WEIGHT_DIR',
    os.path.join(WEIGHTS_BASE, 'minicpm-v-4.6-gptq'))
MINICPM_STAGING_DIR = os.path.join(_MODEL_REPO_BASE, 'models', 'minicpm-v-4.6-gptq')
if not os.path.isdir(MINICPM_WEIGHT_DIR) and os.path.isdir(MINICPM_STAGING_DIR):
    MINICPM_WEIGHT_DIR = MINICPM_STAGING_DIR


def default_vlm_serve_model_id(weight_dir=None):
    """Model id for transformers serve when loading weights from a local directory."""
    return os.path.abspath(os.path.expanduser(weight_dir or MINICPM_WEIGHT_DIR))


def model_repo_hint():
    return (
        f'ROBOT_PERCEPTION_DIR={_MODEL_REPO_BASE!r} '
        f'(exists={os.path.isdir(_MODEL_REPO_BASE)}), '
        f'GSAM2={GSAM2_DIR!r}, GDINO={GSAM2_GDINO_DIR!r}, '
        f'WEIGHTS_BASE={WEIGHTS_BASE!r}'
    )


def ensure_gsam2_path():
    """Add Grounded-SAM-2 paths for sam2 + groundingdino imports."""
    for path in (GSAM2_GDINO_DIR, GSAM2_DIR):
        if os.path.isdir(path) and path not in sys.path:
            sys.path.insert(0, path)
