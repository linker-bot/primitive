#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=perception_env.sh
source "${SCRIPT_DIR}/perception_env.sh"

eval "$(conda shell.bash hook)"
conda activate "${PERCEPTION_ENV_NAME}"

source /opt/ros/jazzy/setup.bash

if [[ -f "${WS_ROOT}/install/setup.bash" ]]; then
    source "${WS_ROOT}/install/setup.bash"
fi

CONDA_PYTHON="$(conda run -n "${PERCEPTION_ENV_NAME}" which python3)"

if ! python3 -c "import catkin_pkg" 2>/dev/null || ! python3 -c "import em; assert int(em.__version__[0]) < 4" 2>/dev/null; then
    pip install --force-reinstall "empy<4" catkin-pkg lark
fi

COLCON_PATHS=()
if [[ -d "${PERCEPTION_MSGS_DIR}" ]]; then
    COLCON_PATHS+=("${PERCEPTION_MSGS_DIR}")
fi
COLCON_PATHS+=("${PKG_DIR}")

echo "================================"
echo " Build Perception"
echo "================================"
echo " Python:   ${CONDA_PYTHON}"
echo " WS:       ${WS_ROOT}"
echo " Package:  ${PKG_DIR}"
echo " Third:    ${ROBOT_PERCEPTION_DIR}"
echo ""

echo "[1/2] Building robot_perception_msgs..."
colcon build --packages-select robot_perception_msgs --symlink-install \
    --paths "${PERCEPTION_MSGS_DIR}" 2>&1 | tail -3
source "${WS_ROOT}/install/setup.bash"

echo "[2/2] Building robot_perception..."
PYTHON_INTERPRETTER="${CONDA_PYTHON}" colcon build --packages-select robot_perception \
    --symlink-install \
    --paths "${COLCON_PATHS[@]}" 2>&1 | tail -3
source "${WS_ROOT}/install/setup.bash"

find "${WS_ROOT}/install/robot_perception/lib/robot_perception" -type f -executable \
    -exec sed -i "1s|^#\!.*python.*|#\!${CONDA_PYTHON}|" {} \;

echo ""
echo "Build complete."
echo "detection_bbox + VLM: bash scripts/setup_vlm_env.sh && bash scripts/run_detection_bbox.sh --with-vlm"
