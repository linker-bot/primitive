#include <rclcpp/rclcpp.hpp>
#include <orbbec_camera/ob_camera_node_driver.h>
#include <orbbec_camera/ob_camera_node.h>
#include <magic_enum/magic_enum.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

using namespace orbbec_camera;

namespace {

struct CliArgs {
  bool help = false;
  std::string serial_number;
};

void printUsage() {
  std::cout
      << "Usage:\n"
      << "ros2 run orbbec_camera list_camera_profile_mode_node --\\\n"
      << "      [--serial_number SN]\n\n"
      << "Parameters:\n"
      << "  --serial_number SN  Select a specific camera by serial number.\n"
      << "  -h, --help          Show this help message.\n";
}

bool parseArgs(int argc, char **argv, CliArgs &args, std::string &error) {
  for (int i = 1; i < argc; ++i) {
    const std::string current = argv[i];
    if (current == "-h" || current == "--help") {
      args.help = true;
      return true;
    }

    if (current.rfind("--serial_number=", 0) == 0) {
      args.serial_number = current.substr(std::strlen("--serial_number="));
      if (args.serial_number.empty()) {
        error = "--serial_number requires a value";
        return false;
      }
      continue;
    }

    if (current == "--serial_number") {
      if (++i >= argc) {
        error = "--serial_number requires a value";
        return false;
      }
      args.serial_number = argv[i];
      continue;
    }

    error = "Unknown argument: " + current;
    return false;
  }

  return true;
}

std::shared_ptr<ob::Device> initializeDevice(const std::string& serial_number) {
  auto context = std::make_shared<ob::Context>();
  auto device_list = context->queryDeviceList();
  if (!device_list || device_list->getCount() == 0) {
    std::cout << "No device found" << std::endl;
    return nullptr;
  }

  if (!serial_number.empty()) {
    return device_list->getDeviceBySN(serial_number.c_str(), OB_DEVICE_DEFAULT_ACCESS);
  }

  return device_list->getDevice(0, OB_DEVICE_DEFAULT_ACCESS);
}

}  // namespace

void listSensorProfiles(const std::shared_ptr<ob::Device>& device) {
  auto sensor_list = device->getSensorList();
  auto pid = device->getDeviceInfo()->getPid();
  for (size_t i = 0; i < sensor_list->getCount(); i++) {
    auto sensor = sensor_list->getSensor(i);
    auto profile_list = sensor->getStreamProfileList();
    for (size_t j = 0; j < profile_list->getCount(); j++) {
      auto origin_profile = profile_list->getProfile(j);
      if ((sensor->getType() == OB_SENSOR_DEPTH || sensor->getType() == OB_SENSOR_IR_LEFT ||
           sensor->getType() == OB_SENSOR_IR_RIGHT) &&
          isGemini305SeriesPID(pid)) {
        // Gemini 305 series
        auto profile = origin_profile->as<ob::VideoStreamProfile>();
        std::cout << magic_enum::enum_name(sensor->getType()) << " profile: " << profile->getWidth()
                  << "x" << profile->getHeight() << " " << profile->getFps() << "fps "
                  << magic_enum::enum_name(profile->getFormat())
                  << " | width: " << profile->getDecimationConfig().originWidth
                  << " height: " << profile->getDecimationConfig().originHeight
                  << " downscale:" << profile->getDecimationConfig().factor << std::endl;
      } else if (sensor->getType() == OB_SENSOR_COLOR || sensor->getType() == OB_SENSOR_DEPTH ||
                 sensor->getType() == OB_SENSOR_IR || sensor->getType() == OB_SENSOR_IR_LEFT ||
                 sensor->getType() == OB_SENSOR_IR_RIGHT) {
        auto profile = origin_profile->as<ob::VideoStreamProfile>();
        std::cout << magic_enum::enum_name(sensor->getType()) << " profile: " << profile->getWidth()
                  << "x" << profile->getHeight() << " " << profile->getFps() << "fps "
                  << magic_enum::enum_name(profile->getFormat()) << std::endl;
      } else if (sensor->getType() == OB_SENSOR_ACCEL) {
        auto profile = origin_profile->as<ob::AccelStreamProfile>();
        std::cout << magic_enum::enum_name(sensor->getType())
                  << " profile: " << profile->getSampleRate() << "  full scale_range "
                  << profile->getFullScaleRange() << std::endl;
      } else if (sensor->getType() == OB_SENSOR_GYRO) {
        auto profile = origin_profile->as<ob::GyroStreamProfile>();
        std::cout << magic_enum::enum_name(sensor->getType())
                  << " profile: " << profile->getSampleRate() << "  full scale_range "
                  << profile->getFullScaleRange() << std::endl;
      } else if (sensor->getType() == OB_SENSOR_LIDAR) {
        auto profile = origin_profile->as<ob::LiDARStreamProfile>();
        std::cout << magic_enum::enum_name(sensor->getType())
                  << " scan rate: " << magic_enum::enum_name(profile->getScanRate())
                  << "  format:" << magic_enum::enum_name(profile->getFormat()) << std::endl;
      } else {
        std::cout << "Unknown profile: " << magic_enum::enum_name(sensor->getType()) << std::endl;
      }
    }
  }
}

void printDeviceProperties(const std::shared_ptr<ob::Device>& device) {
  if (!device->isPropertySupported(OB_STRUCT_CURRENT_DEPTH_ALG_MODE, OB_PERMISSION_READ_WRITE)) {
    std::cout << "Current device not support depth work mode!" << std::endl;
    return;
  }
  auto current_depth_mode = device->getCurrentDepthWorkMode();
  std::cout << "Current depth mode: " << current_depth_mode.name << std::endl;
  auto depth_mode_list = device->getDepthWorkModeList();
  std::cout << "Depth mode list: " << std::endl;
  for (uint32_t i = 0; i < depth_mode_list->getCount(); i++) {
    std::cout << "Depth_mode_list[" << i << "]: " << (*depth_mode_list)[i].name << std::endl;
  }
}

void printPreset(const std::shared_ptr<ob::Device>& device) {
  auto preset_list = device->getAvailablePresetList();
  if (!preset_list || preset_list->getCount() == 0) {
    return;
  }
  std::cout << "Preset list:" << std::endl;
  for (uint32_t i = 0; i < preset_list->getCount(); i++) {
    auto name = preset_list->getName(i);
    std::cout << "Preset list[" << i << "]: " << name << std::endl;
  }
}

int main(int argc, char **argv) {
  CliArgs args;
  std::string parse_error;
  if (!parseArgs(argc, argv, args, parse_error)) {
    std::cerr << parse_error << std::endl;
    printUsage();
    return 1;
  }
  if (args.help) {
    printUsage();
    return 0;
  }

  auto device = initializeDevice(args.serial_number);
  if (!device) {
    return -1;
  }
  listSensorProfiles(device);
  printDeviceProperties(device);
  printPreset(device);
  return 0;
}
