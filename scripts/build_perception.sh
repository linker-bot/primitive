#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate conda robot_perception environment
eval "$(conda shell.bash hook)"
conda activate robot_perception

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source local workspace if exists
if [ -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    source "${WORKSPACE_DIR}/install/setup.bash"
fi

CONDA_PYTHON="$(conda run -n robot_perception which python3)"

# Ensure ROS build dependencies are available in the conda env
# catkin_pkg is needed by ament_cmake; empy must be 3.x (ROS 2 Jazzy is incompatible with empy 4.x)
if ! python3 -c "import catkin_pkg" 2>/dev/null || ! python3 -c "import em; assert int(em.__version__[0]) < 4" 2>/dev/null; then
    pip install --force-reinstall "empy<4" catkin-pkg lark
fi

echo "================================"
echo " Build Perception"
echo "================================"
echo " Python: ${CONDA_PYTHON}"
echo ""

cd "${WORKSPACE_DIR}"

echo "[1/3] Building robot_perception_msgs..."
colcon build --packages-select robot_perception_msgs --symlink-install 2>&1 | tail -3
source "${WORKSPACE_DIR}/install/setup.bash"

echo "[2/3] Building robot_perception..."
PYTHON_INTERPRETER="${CONDA_PYTHON}" colcon build --packages-select robot_perception \
    --symlink-install \
    --paths "${WORKSPACE_DIR}/perception" 2>&1 | tail -3
source "${WORKSPACE_DIR}/install/setup.bash"

echo "[3/3] Fixing shebang..."
# Fix shebang in entry points to use conda Python
find "${WORKSPACE_DIR}/install/robot_perception/lib/robot_perception" -type f -executable \
    -exec sed -i "1s|^#\!.*python.*|#\!${CONDA_PYTHON}|" {} \;

echo ""
echo "Build complete."
