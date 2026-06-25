#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Activate conda robot_perception environment
eval "$(conda shell.bash hook)"
conda activate robot_perception

# Source ROS2
source /opt/ros/jazzy/setup.bash

# Source local workspace (robot_perception_msgs etc.)
if [ -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    source "${WORKSPACE_DIR}/install/setup.bash"
fi

export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

echo "================================"
echo " Robot Perception Pipeline"
echo "================================"
echo ""
echo " Conda env: robot_perception"
echo " Package:   robot_perception"

CONDA_PYTHON="$(conda run -n robot_perception which python3)"
echo " Python:    ${CONDA_PYTHON}"
echo ""

# Ensure ROS build dependencies are available in the conda env
# catkin_pkg is needed by ament_cmake; empy must be 3.x (ROS 2 Jazzy is incompatible with empy 4.x)
if ! python3 -c "import catkin_pkg" 2>/dev/null || ! python3 -c "import em; assert int(em.__version__[0]) < 4" 2>/dev/null; then
    pip install --force-reinstall "empy<4" catkin-pkg lark
fi

echo "[1/3] Building robot_perception_msgs (incremental)..."
cd "${WORKSPACE_DIR}"
colcon build --packages-select robot_perception_msgs --symlink-install 2>&1 | tail -3
source "${WORKSPACE_DIR}/install/setup.bash"

echo "[2/3] Building robot_perception (symlink-install)..."
PYTHON_INTERPRETTER="${CONDA_PYTHON}" colcon build --packages-select robot_perception \
    --symlink-install \
    --paths "${WORKSPACE_DIR}/perception" 2>&1 | tail -3
source "${WORKSPACE_DIR}/install/setup.bash"

# Fix shebang in ament_python entry points to use conda Python
echo "[2.5/3] Fixing entry point shebangs..."
find "${WORKSPACE_DIR}/install/robot_perception/lib/robot_perception" -type f -executable \
    -exec sed -i "1s|^#\!.*python.*|#\!${CONDA_PYTHON}|" {} \;

echo "[3/3] Launching perception_on_demand..."
echo ""
ros2 launch robot_perception perception_on_demand.launch.py "$@"
