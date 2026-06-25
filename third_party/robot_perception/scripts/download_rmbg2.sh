#!/usr/bin/env bash
# Download briaai/RMBG-2.0 (Pixel3D rembg) into online/models staging.
# ModelScope: https://modelscope.cn/models/briaai/RMBG-2.0/files
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_OUT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)/models/RMBG-2.0"
OUT_DIR="${RMBG2_WEIGHT_DIR:-${DEFAULT_OUT}}"
MODEL_ID_MS="briaai/RMBG-2.0"
MODEL_ID_HF="briaai/RMBG-2.0"

mkdir -p "${OUT_DIR}"

download_hf() {
  echo "[download] Hugging Face (${MODEL_ID_HF}) -> ${OUT_DIR}"
  python3 - <<PY
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="${MODEL_ID_HF}",
    local_dir="${OUT_DIR}",
)
print("OK:", "${OUT_DIR}")
PY
}

download_ms() {
  echo "[download] ModelScope (${MODEL_ID_MS}) -> ${OUT_DIR}"
  python3 - <<PY
from modelscope import snapshot_download
snapshot_download("${MODEL_ID_MS}", local_dir="${OUT_DIR}")
print("OK:", "${OUT_DIR}")
PY
}

if python3 -c "import torch; assert hasattr(torch, '__version__')" 2>/dev/null; then
  if python3 -c "import modelscope" 2>/dev/null; then
    download_ms || download_hf
  else
    pip install -q modelscope
    download_ms || download_hf
  fi
else
  echo "[warn] PyTorch not installed (or broken namespace stub in base env)."
  echo "[warn] Create a small env, e.g.:"
  echo "       conda create -n ms_download python=3.11 -y && conda activate ms_download"
  echo "       pip install modelscope torch && bash $0"
  if python3 -c "import modelscope" 2>/dev/null; then
    download_ms || download_hf
  else
    pip install -q -U huggingface_hub
    download_hf
  fi
fi

echo "[done] Weights staged at: ${OUT_DIR}"
echo "[deploy] rsync -av ${OUT_DIR}/ ${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0/"
echo "[config] In pixal3d-t/pipeline.json set:"
echo '         "rembg_model": { "args": { "model_name": "${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0" } }'
ls -lh "${OUT_DIR}" 2>/dev/null | head -20
