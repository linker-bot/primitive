#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

WITH_VLM=0
LAUNCH_ARGS=()
VLM_PID=""

cleanup() {
    if [[ -n "${VLM_PID}" ]] && kill -0 "${VLM_PID}" 2>/dev/null; then
        echo "[run_detection_bbox] Stopping VLM server (pid ${VLM_PID})..."
        kill "${VLM_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

while [[ $# -gt 0 ]]; do
    case "$1" in
        --with-vlm)
            WITH_VLM=1
            shift
            ;;
        *)
            LAUNCH_ARGS+=("$1")
            shift
            ;;
    esac
done

eval "$(conda shell.bash hook)"
conda activate "${PERCEPTION_ENV_NAME}"

if [[ -z "${ATTN_BACKEND:-}" ]]; then
    if ! python -c "import flash_attn" 2>/dev/null; then
        export ATTN_BACKEND=sdpa
        echo "[run_detection_bbox] flash_attn not found, using ATTN_BACKEND=sdpa"
    fi
fi

source /opt/ros/jazzy/setup.bash

if [[ -f "${WS_ROOT}/install/setup.bash" ]]; then
    source "${WS_ROOT}/install/setup.bash"
fi

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"
export ROS_AUTOMATIC_DISCOVERY_RANGE="${ROS_AUTOMATIC_DISCOVERY_RANGE:-LOCALHOST}"
export ROBOT_PERCEPTION_DIR

CONDA_PYTHON="$(conda run -n "${PERCEPTION_ENV_NAME}" which python3)"

echo "================================"
echo " detection_bbox"
echo "================================"
echo " Conda env: ${PERCEPTION_ENV_NAME}"
echo " Package:   ${PKG_DIR}"
echo " Python:    ${CONDA_PYTHON}"
echo " Third:     ${ROBOT_PERCEPTION_DIR}"
echo ""

if ! python3 -c "import catkin_pkg" 2>/dev/null || ! python3 -c "import em; assert int(em.__version__[0]) < 4" 2>/dev/null; then
    pip install --force-reinstall "empy<4" catkin-pkg lark
fi

COLCON_PATHS=()
if [[ -d "${PERCEPTION_MSGS_DIR}" ]]; then
    COLCON_PATHS+=("${PERCEPTION_MSGS_DIR}")
fi
COLCON_PATHS+=("${PKG_DIR}")

echo "[1/3] Building robot_perception_msgs (incremental)..."
colcon build --packages-select robot_perception_msgs --symlink-install \
    --paths "${PERCEPTION_MSGS_DIR}" 2>&1 | tail -3
source "${WS_ROOT}/install/setup.bash"

echo "[2/3] Building robot_perception (symlink-install)..."
PYTHON_INTERPRETTER="${CONDA_PYTHON}" colcon build --packages-select robot_perception \
    --symlink-install \
    --paths "${COLCON_PATHS[@]}" 2>&1 | tail -3
source "${WS_ROOT}/install/setup.bash"

echo "[2.5/3] Fixing entry point shebangs..."
find "${WS_ROOT}/install/robot_perception/lib/robot_perception" -type f -executable \
    -exec sed -i "1s|^#\!.*python.*|#\!${CONDA_PYTHON}|" {} \;

start_local_vlm_if_needed() {
    if curl -sf "${VLM_HEALTH_URL}" >/dev/null 2>&1; then
        echo "[run_detection_bbox] Local VLM already running at ${VLM_BASE_URL} (model=${VLM_MODEL})"
        return 0
    fi
    if [[ ! -d "${MINICPM_WEIGHT_DIR}" ]]; then
        echo "[run_detection_bbox] WARN: MiniCPM weights missing at ${MINICPM_WEIGHT_DIR}" >&2
        echo "[run_detection_bbox]       Run: bash scripts/sync_minicpm_weights.sh" >&2
        return 1
    fi
    echo "[run_detection_bbox] Starting local VLM server..."
    bash "${SCRIPT_DIR}/start_minicpm_vlm_server.sh" &
    VLM_PID=$!
    for _ in $(seq 1 120); do
        if curl -sf "${VLM_HEALTH_URL}" >/dev/null 2>&1; then
            echo "[run_detection_bbox] VLM ready at ${VLM_BASE_URL} (model=${VLM_MODEL})"
            return 0
        fi
        sleep 1
    done
    echo "[run_detection_bbox] ERROR: VLM server failed to start within 60s" >&2
    return 1
}

if [[ "${WITH_VLM}" -eq 1 ]]; then
    start_local_vlm_if_needed
    LAUNCH_ARGS+=(
        "use_vlm_detect:=true"
        "vlm_base_url:=${VLM_BASE_URL}"
        "vlm_api_key:=${VLM_API_KEY}"
        "vlm_model:=${VLM_MODEL}"
    )
fi

echo "[3/3] Launching detection_bbox..."
echo ""
ros2 launch robot_perception detection_bbox.launch.py "${LAUNCH_ARGS[@]}"
