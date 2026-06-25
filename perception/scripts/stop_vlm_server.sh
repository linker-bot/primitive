#!/usr/bin/env bash
# Kill any running MiniCPM VLM server on the configured port.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/perception_env.sh"

_pids=$(lsof -ti "tcp:${VLM_PORT}" 2>/dev/null || true)
if [[ -z "${_pids}" ]]; then
    echo "[stop_vlm] No process found on port ${VLM_PORT}."
    exit 0
fi

echo "[stop_vlm] Killing process(es) on port ${VLM_PORT}: ${_pids}"
kill ${_pids} 2>/dev/null || true
sleep 1

# Force-kill survivors
_remaining=$(lsof -ti "tcp:${VLM_PORT}" 2>/dev/null || true)
if [[ -n "${_remaining}" ]]; then
    echo "[stop_vlm] Force-killing: ${_remaining}"
    kill -9 ${_remaining} 2>/dev/null || true
fi

echo "[stop_vlm] Done. Port ${VLM_PORT} is free."
