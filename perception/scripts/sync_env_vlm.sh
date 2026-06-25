#!/usr/bin/env bash
# Sync robot_perception_vlm conda env from environment_vlm.yml
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

eval "$(conda shell.bash hook)"

case "${1:-update}" in
    create)
        bash "${SCRIPT_DIR}/setup_vlm_env.sh"
        ;;
    update)
        if ! conda env list | grep -qw "^${VLM_ENV_NAME}"; then
            bash "${SCRIPT_DIR}/setup_vlm_env.sh"
        else
            conda env update -n "${VLM_ENV_NAME}" -f "${PKG_DIR}/environment_vlm.yml" --prune
        fi
        ;;
    *)
        echo "Usage: $0 [create|update]"
        exit 1
        ;;
esac

echo "[sync_env_vlm] ${VLM_ENV_NAME} ready."
