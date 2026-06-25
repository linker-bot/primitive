#!/usr/bin/env bash
# Create or update the robot_perception_vlm conda environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

eval "$(conda shell.bash hook)"

if conda env list | grep -qw "^${VLM_ENV_NAME}"; then
    echo "[setup_vlm_env] Updating ${VLM_ENV_NAME} from environment_vlm.yml..."
    conda env update -n "${VLM_ENV_NAME}" -f "${PKG_DIR}/environment_vlm.yml" --prune
else
    echo "[setup_vlm_env] Creating ${VLM_ENV_NAME} from environment_vlm.yml..."
    conda env create -f "${PKG_DIR}/environment_vlm.yml"
fi

#!/usr/bin/env bash
# Create or update the robot_perception_vlm conda environment.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

eval "$(conda shell.bash hook)"

if conda env list | grep -qw "^${VLM_ENV_NAME}"; then
    echo "[setup_vlm_env] Updating ${VLM_ENV_NAME} from environment_vlm.yml..."
    conda env update -n "${VLM_ENV_NAME}" -f "${PKG_DIR}/environment_vlm.yml" --prune
else
    echo "[setup_vlm_env] Creating ${VLM_ENV_NAME} from environment_vlm.yml..."
    conda env create -f "${PKG_DIR}/environment_vlm.yml"
fi

echo "[setup_vlm_env] Verifying imports (GPTQ serve: optimum + gptqmodel + transformers serve)..."
conda run -n "${VLM_ENV_NAME}" python - <<'PY'
import importlib.metadata
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import fastapi
import uvicorn
print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
import transformers
print('transformers', transformers.__version__)
print('optimum', importlib.metadata.version('optimum'))
print('gptqmodel', importlib.metadata.version('gptqmodel'))
print('huggingface-hub', importlib.metadata.version('huggingface-hub'))
print('fastapi', fastapi.__version__, 'uvicorn', uvicorn.__version__)
PY

echo "[setup_vlm_env] Done. Activate with: conda activate ${VLM_ENV_NAME}"
