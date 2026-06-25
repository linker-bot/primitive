#!/usr/bin/env bash
# Start local MiniCPM-V OpenAI-compatible server (transformers serve).
# Run in robot_perception_vlm env, separate from GDINO/SAM2 stack.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

if [[ ! -d "${MINICPM_WEIGHT_DIR}" ]]; then
    echo "[start_vlm] ERROR: weights not found at ${MINICPM_WEIGHT_DIR}" >&2
    echo "[start_vlm] Run: bash scripts/sync_minicpm_weights.sh" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
if ! conda env list | grep -qw "^${VLM_ENV_NAME}"; then
    echo "[start_vlm] ERROR: conda env '${VLM_ENV_NAME}' missing. Run: bash scripts/setup_vlm_env.sh" >&2
    exit 1
fi

if ! conda run -n "${VLM_ENV_NAME}" python -c "import importlib.metadata; importlib.metadata.version('optimum')" 2>/dev/null; then
    echo "[start_vlm] ERROR: package 'optimum' missing (required for GPTQ weights)." >&2
    echo "[start_vlm] Fix: bash scripts/setup_vlm_env.sh" >&2
    echo "[start_vlm] Or:  bash scripts/sync_env_vlm.sh update" >&2
    exit 1
fi

echo "================================"
echo " MiniCPM-V local VLM server"
echo "================================"
echo " Env:     ${VLM_ENV_NAME}"
echo " Weights: ${MINICPM_WEIGHT_DIR}"
echo " URL:     ${VLM_BASE_URL}"
echo " Model:   ${VLM_MODEL}"
echo " Health:  ${VLM_HEALTH_URL}"
echo ""
echo " Note: /v1/models may return empty data[] for local weights — use Model id above."
echo ""

# Kill any leftover process occupying the VLM port
_existing_pid=$(lsof -ti "tcp:${VLM_PORT}" 2>/dev/null || true)
if [[ -n "${_existing_pid}" ]]; then
    echo "[start_vlm] Port ${VLM_PORT} occupied by PID ${_existing_pid}, killing..."
    kill ${_existing_pid} 2>/dev/null || true
    sleep 1
    # Force-kill if still alive
    if kill -0 ${_existing_pid} 2>/dev/null; then
        kill -9 ${_existing_pid} 2>/dev/null || true
        sleep 0.5
    fi
    echo "[start_vlm] Cleared."
fi

exec conda run --no-capture-output -n "${VLM_ENV_NAME}" \
    transformers serve "${MINICPM_WEIGHT_DIR}" \
    --host "${VLM_HOST}" \
    --port "${VLM_PORT}" \
    --continuous-batching \
    "$@"
