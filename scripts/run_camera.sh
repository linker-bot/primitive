#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

source /opt/ros/jazzy/setup.bash

if [ -f "${WORKSPACE_DIR}/install/setup.bash" ]; then
    source "${WORKSPACE_DIR}/install/setup.bash"
fi

export ROS_DOMAIN_ID=42
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST

# 自动检测 Orbbec 相机 USB port
# Orbbec Vendor ID: 2bc5
detect_orbbec_ports() {
    local ports=()
    for d in /sys/bus/usb/devices/*/; do
        if [ -f "$d/idVendor" ] && grep -q "2bc5" "$d/idVendor" 2>/dev/null; then
            ports+=("$(basename "$d")")
        fi
    done
    echo "${ports[@]}"
}

# 参数: camera_head=USB_PORT camera_waist=USB_PORT
# 示例: ./run_camera.sh camera_head=2-9 camera_waist=2-6
# 如果不传参, 自动检测第一个 Orbbec 相机作为 cam_head
CAM_HEAD_PORT=""
CAM_WAIST_PORT=""

for arg in "$@"; do
    case "$arg" in
        camera_head=*)
            CAM_HEAD_PORT="${arg#*=}"
            ;;
        camera_waist=*)
            CAM_WAIST_PORT="${arg#*=}"
            ;;
        *)
            echo "Unknown argument: $arg"
            echo "Usage: $0 [camera_head=USB_PORT] [camera_waist=USB_PORT]"
            echo "Example: $0 camera_head=2-9 camera_waist=2-6"
            exit 1
            ;;
    esac
done

# 默认值: 不传参则自动检测 Orbbec 相机
if [ -z "$CAM_HEAD_PORT" ] && [ -z "$CAM_WAIST_PORT" ]; then
    DETECTED_PORTS=($(detect_orbbec_ports))
    if [ ${#DETECTED_PORTS[@]} -eq 0 ]; then
        echo "ERROR: No Orbbec camera detected on USB bus."
        echo "       Please connect a camera or specify port manually."
        exit 1
    fi
    CAM_HEAD_PORT="${DETECTED_PORTS[0]}"
    if [ ${#DETECTED_PORTS[@]} -ge 2 ]; then
        CAM_WAIST_PORT="${DETECTED_PORTS[1]}"
    fi
fi

ENABLE_HEAD="false"
ENABLE_WAIST="false"
[ -n "$CAM_HEAD_PORT" ] && ENABLE_HEAD="true"
[ -n "$CAM_WAIST_PORT" ] && ENABLE_WAIST="true"

echo "==========================="
echo " Orbbec Multi-Camera Launch"
echo "==========================="
echo ""
[ "$ENABLE_HEAD" = "true" ] && echo "  camera_head:  usb_port=${CAM_HEAD_PORT} (auto-detected)"
[ "$ENABLE_WAIST" = "true" ] && echo "  camera_waist: usb_port=${CAM_WAIST_PORT} (auto-detected)"
[ "$ENABLE_HEAD" = "false" ] && echo "  camera_head:  disabled"
[ "$ENABLE_WAIST" = "false" ] && echo "  camera_waist: disabled"
echo ""

pkill -f "ros2 launch.*orbbec_camera" 2>/dev/null || true
sleep 1

echo "[1/2] Building OrbbecSDK_ROS2 (incremental)..."
cd "${WORKSPACE_DIR}"
colcon build --packages-up-to orbbec_camera \
    --event-handlers console_direct+ \
    --cmake-args -DCMAKE_BUILD_TYPE=Release \
                 -DPython3_EXECUTABLE=/usr/bin/python3 \
                 -DPython3_FIND_STRATEGY=LOCATION \
                 -DPython3_FIND_REGISTRY=NEVER 2>&1 | tail -5
source "${WORKSPACE_DIR}/install/setup.bash"

echo "[2/2] Launching camera nodes (Ctrl+C to stop)..."
echo ""
ros2 launch orbbec_camera multi_camera.launch.py \
    cam_head_port:="${CAM_HEAD_PORT:-2-9}" \
    cam_waist_port:="${CAM_WAIST_PORT:-2-6}" \
    enable_cam_head:="${ENABLE_HEAD}" \
    enable_cam_waist:="${ENABLE_WAIST}"
