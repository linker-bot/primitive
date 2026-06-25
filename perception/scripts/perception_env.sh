# Source from build_perception.sh / run_perception.sh / start_minicpm_vlm_server.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"

if [ -z "${ROBOT_PERCEPTION_DIR:-}" ]; then
    for candidate in \
        "${REPO_ROOT}/third_party/robot_perception" \
        "${HOME}/ros_ws/src/hand_gesture_primitives/third_party/robot_perception" \
        "${HOME}/gitea_ws/src/hand_gesture_primitives/third_party/robot_perception"; do
        if [ -d "${candidate}" ]; then
            export ROBOT_PERCEPTION_DIR="${candidate}"
            break
        fi
    done
fi

export ROBOT_PERCEPTION_DIR="${ROBOT_PERCEPTION_DIR:-${REPO_ROOT}/third_party/robot_perception}"

# MiniCPM weights: staging in third_party, runtime under PERCEPTION_WEIGHTS_DIR
WEIGHTS_BASE="${PERCEPTION_WEIGHTS_DIR:-${REPO_ROOT}/data/weights}"
export MINICPM_WEIGHT_DIR="${MINICPM_WEIGHT_DIR:-${WEIGHTS_BASE}/minicpm-v-4.6-gptq}"
