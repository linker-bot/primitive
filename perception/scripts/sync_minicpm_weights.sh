#!/usr/bin/env bash
# Copy staged MiniCPM weights to PERCEPTION_WEIGHTS_DIR on deploy machine.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

SRC="${MINICPM_STAGING_DIR:-${ROBOT_PERCEPTION_DIR}/models/minicpm-v-4.6-gptq}"
DEST="${MINICPM_DEPLOY_DIR:-${MINICPM_WEIGHT_DIR}}"

if [ ! -d "${SRC}" ]; then
    echo "[sync_minicpm] ERROR: staging dir missing: ${SRC}" >&2
    exit 1
fi

mkdir -p "${DEST}"
rsync -av --delete "${SRC}/" "${DEST}/"
echo "[sync_minicpm] Synced ${SRC} -> ${DEST}"
