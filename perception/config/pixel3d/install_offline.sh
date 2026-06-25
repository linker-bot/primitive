#!/usr/bin/env bash
# Install Pixel3D offline configs (copy pipeline.offline.example.json into weight tree).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WEIGHTS_BASE="${PERCEPTION_WEIGHTS_DIR:-${REPO_ROOT}/data/weights}"
PIXAL3D_T="${PIXAL3D_MODEL_PATH:-${WEIGHTS_BASE}/pixal3d-t}"
RMBG="${RMBG_MODEL_PATH:-${WEIGHTS_BASE}/RMBG-2.0}"
DINOV3="${DINOV3_MODEL_PATH:-${WEIGHTS_BASE}/dinov3-vitl16-pretrain-lvd1689m}"

check_dir() {
  local path="$1" name="$2"
  if [[ ! -d "${path}" ]]; then
    echo "[error] Missing ${name}: ${path}"
    exit 1
  fi
}

check_dir "${PIXAL3D_T}" "pixal3d-t"
check_dir "${RMBG}" "RMBG-2.0"
check_dir "${DINOV3}" "dinov3-vitl16-pretrain-lvd1689m"
test -f "${PIXAL3D_T}/ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors" || {
  echo "[error] pixal3d-t ckpts incomplete under ${PIXAL3D_T}/ckpts/"
  exit 1
}

echo "[install] Backing up existing pipeline.json (if any)"
if [[ -f "${PIXAL3D_T}/pipeline.json" && ! -f "${PIXAL3D_T}/pipeline.json.bak" ]]; then
  cp -a "${PIXAL3D_T}/pipeline.json" "${PIXAL3D_T}/pipeline.json.bak"
fi

echo "[install] Installing offline pipeline.json -> ${PIXAL3D_T}/pipeline.json"
cp -f "${SCRIPT_DIR}/pipeline.offline.example.json" "${PIXAL3D_T}/pipeline.json"

echo "[done] Offline Pixel3D ready:"
echo "  pixal3d-t:  ${PIXAL3D_T}"
echo "  RMBG-2.0:   ${RMBG}"
echo "  DINOv3:     ${DINOV3}"
echo "  manifest:   ${SCRIPT_DIR}/pixel3d_offline.example.json"
echo ""
echo "Launch with:"
echo "  ros2 launch robot_perception detection_bbox.launch.py \\"
echo "    use_pixel3d:=true pixel3d_low_vram:=true \\"
echo "    pixel3d_offline_config:=${SCRIPT_DIR}/pixel3d_offline.example.json"
