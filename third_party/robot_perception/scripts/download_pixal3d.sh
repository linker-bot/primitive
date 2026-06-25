#!/usr/bin/env bash
# Download Pixal3D-T weights into online/models staging (no full Pixal3D env needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Default: sibling repo online/models (override with PIXAL3D_WEIGHT_DIR)
DEFAULT_OUT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)/models/pixal3d-t"
OUT_DIR="${PIXAL3D_WEIGHT_DIR:-${DEFAULT_OUT}}"
MODEL_ID_MS="TencentARC/Pixal3D-T"
MODEL_ID_HF="TencentARC/Pixal3D"

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
  echo "[warn] Using huggingface_hub only (ModelScope SDK needs torch)."
  pip install -q -U huggingface_hub
  download_hf
fi

echo "[done] Weights staged at: ${OUT_DIR}"
echo "[info] Expected ~23GB under ckpts/ plus pipeline.json"
ls -lh "${OUT_DIR}" 2>/dev/null | head -20
