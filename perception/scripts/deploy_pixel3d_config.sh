#!/bin/bash
# deploy_pixel3d_config.sh — install Pixel3D offline pipeline into PERCEPTION_WEIGHTS_DIR
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "${PKG_DIR}/.." && pwd)"
MODEL_BASE="${PERCEPTION_WEIGHTS_DIR:-${REPO_ROOT}/data/weights}"
PIXAL3D_MODEL_DIR="${MODEL_BASE}/pixal3d-t"
PIPELINE_SRC="${PKG_DIR}/config/pixel3d/pipeline.offline.example.json"
PIPELINE_DST="${PIXAL3D_MODEL_DIR}/pipeline.json"

echo "=== Pixel3D offline weight config ==="
echo "WEIGHTS_BASE=${MODEL_BASE}"
echo ""

check_dir() {
    if [ -d "$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [MISSING] $1"
        MISSING=1
    fi
}

check_file() {
    if [ -f "$1" ]; then
        echo "  [OK] $1"
    else
        echo "  [MISSING] $1"
        MISSING=1
    fi
}

MISSING=0
echo "Checking model directories..."
check_dir "${MODEL_BASE}/pixal3d-t"
check_dir "${MODEL_BASE}/pixal3d-t/ckpts"
check_dir "${MODEL_BASE}/RMBG-2.0"
check_dir "${MODEL_BASE}/dinov3-vitl16-pretrain-lvd1689m"
echo ""

echo "Checking Pixel3D sub-models..."
for ckpt in ss_dec_conv3d_16l8_fp16 \
            ss_flow_img_dit_1_3B_64_bf16 \
            slat_flow_img2shape_dit_1_3B_512_bf16 \
            slat_flow_img2shape_dit_1_3B_1024_bf16 \
            shape_dec_next_dc_f16c32_fp16 \
            slat_flow_imgshape2tex_dit_1_3B_1024_bf16 \
            tex_dec_next_dc_f16c32_fp16; do
    check_file "${PIXAL3D_MODEL_DIR}/ckpts/${ckpt}.safetensors"
done
echo ""

if [ "$MISSING" -eq 1 ]; then
    echo "[WARN] Some weights are missing. Deploying config files anyway."
    echo ""
fi

echo "Installing pipeline.json..."
if [ -f "$PIPELINE_SRC" ]; then
    mkdir -p "${PIXAL3D_MODEL_DIR}"
    cp "$PIPELINE_SRC" "$PIPELINE_DST"
    echo "  ${PIPELINE_SRC} -> ${PIPELINE_DST}"
else
    echo "  [ERROR] Missing source: ${PIPELINE_SRC}"
    exit 1
fi
echo ""

echo "Verify ament share offline manifest..."
SHARE_DIR=""
if command -v ros2 &>/dev/null; then
    SHARE_DIR=$(ros2 pkg prefix robot_perception 2>/dev/null || echo "")/share/robot_perception
fi
if [ -n "$SHARE_DIR" ] && [ -f "${SHARE_DIR}/config/pixel3d/pixel3d_offline.example.json" ]; then
    echo "  [OK] ${SHARE_DIR}/config/pixel3d/pixel3d_offline.example.json"
else
    echo "  [WARN] Run: colcon build --packages-select robot_perception"
fi
echo ""
echo "=== Done ==="
