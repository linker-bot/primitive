#!/usr/bin/env bash
# Download MiniCPM-V-4.6-GPTQ into third_party staging (no full robot_perception env).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT_DIR="${REPO_ROOT}/models/minicpm-v-4.6-gptq"
MODEL_ID_MS="OpenBMB/MiniCPM-V-4.6-GPTQ"
MODEL_ID_HF="openbmb/MiniCPM-V-4.6-GPTQ"

mkdir -p "${OUT_DIR}"

download_hf() {
  echo "[download] Hugging Face -> ${OUT_DIR}"
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
  echo "[download] ModelScope -> ${OUT_DIR}"
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
  echo "[warn] Using huggingface_hub only (ModelScope CLI needs torch)."
  pip install -q -U huggingface_hub
  download_hf
fi

echo "[done] Weights staged at: ${OUT_DIR}"
ls -lh "${OUT_DIR}" | head
