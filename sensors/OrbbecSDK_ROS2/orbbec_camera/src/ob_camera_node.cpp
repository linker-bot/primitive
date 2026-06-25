/*******************************************************************************
 * Copyright (c) 2023 Orbbec 3D Technology, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *******************************************************************************/

#include "orbbec_camera/ob_camera_node.h"
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <sstream>
#include <algorithm>

#include "orbbec_camera/utils.h"
#include <filesystem>
#include <fstream>
#include "diagnostic_msgs/msg/diagnostic_status.hpp"
#include "libobsensor/hpp/Utils.hpp"

#if defined(USE_RK_HW_DECODER)
#include "orbbec_camera/rk_mpp_decoder.h"
#elif defined(USE_NV_HW_DECODER)
#include "orbbec_camera/jetson_nv_decoder.h"
#endif

#include <malloc.h>

namespace orbbec_camera {
using namespace std::chrono_literals;

std::string OBCameraNode::normalizeDepthFilterName(const std::string &filter_name) {
  if (filter_name == "HardwareNoiseRemoval") {
    return "HardwareNoiseRemovalFilter";
  }
  if (filter_name == "SpatialFilter") {
    return "SpatialAdvancedFilter";
  }
  return filter_name;
}

namespace {

std::string getDepthFilterStatusName(const std::string &filter_name) {
  if (filter_name == "SpatialAdvancedFilter") {
    return "SpatialFilter";
  }
  if (filter_name == "DisparityTransform") {
    return "DisparityToDepth";
  }
  return filter_name;
}

std::string getDepthFilterStatusParamName(const std::string &filter_name,
                                          const std::string &param_name) {
  if (filter_name == "SpatialAdvancedFilter" && param_name == "disp_diff") {
    return "diff_threshold";
  }
  if (filter_name == "SpatialModerateFilter" && param_name == "disp_diff") {
    return "diff_threshold";
  }
  if (filter_name == "TemporalFilter" && param_name == "diff_scale") {
    return "diff_threshold";
  }
  if (filter_name == "DecimationFilter" && param_name == "decimate") {
    return "scale";
  }
  return param_name;
}

bool shouldExposeDepthFilterParams(const std::string &filter_name) {
  return filter_name != "MgcNoiseRemovalFilter" && filter_name != "LutNoiseRemovalFilter" &&
         filter_name != "DisparityTransform" && filter_name != "FalsePositiveFilter" &&
         filter_name != "EdgeNoiseRemovalFilter";
}

int64_t getSystemNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::system_clock::now().time_since_epoch())
      .count();
}

int64_t getSteadyNowUs() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

void OBCameraNode::appendDepthFilterParam(DepthFilterState &filter_state, const std::string &name,
                                          const std::string &value) {
  orbbec_camera_msgs::msg::DepthFilterParam param;
  param.name = name;
  param.value = value;
  filter_state.params.push_back(param);
}

DepthFilterState OBCameraNode::buildDepthFilterState(
    const std::string &filter_name, bool enabled, const std::shared_ptr<ob::Filter> &filter) const {
  const auto normalized_filter_name = normalizeDepthFilterName(filter_name);
  DepthFilterState filter_state;
  filter_state.filter_name = getDepthFilterStatusName(normalized_filter_name);
  filter_state.enabled = enabled;
  auto to_param_value = [](const auto &value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
  };

  if (normalized_filter_name == "NoiseRemovalFilter") {
    appendDepthFilterParam(filter_state, "min_diff",
                           to_param_value(noise_removal_filter_min_diff_));
    appendDepthFilterParam(filter_state, "max_size",
                           to_param_value(noise_removal_filter_max_size_));
  } else if (normalized_filter_name == "HardwareNoiseRemovalFilter") {
    appendDepthFilterParam(filter_state, "threshold",
                           to_param_value(hardware_noise_removal_filter_threshold_));
  }

  if (filter_state.params.empty() && filter &&
      shouldExposeDepthFilterParams(normalized_filter_name)) {
    auto format_filter_config_value = [](const OBFilterConfigSchemaItem &config_schema,
                                         double value) {
      switch (config_schema.type) {
        case OB_FILTER_CONFIG_VALUE_TYPE_INT: {
          return std::to_string(static_cast<long long>(value));
        }
        case OB_FILTER_CONFIG_VALUE_TYPE_BOOLEAN:
          return value != 0.0 ? std::string("true") : std::string("false");
        case OB_FILTER_CONFIG_VALUE_TYPE_FLOAT:
        default: {
          std::ostringstream ss;
          ss << value;
          return ss.str();
        }
      }
    };

    try {
      for (const auto &config_schema : filter->getConfigSchemaVec()) {
        if (config_schema.name == nullptr || config_schema.name[0] == '\0') {
          continue;
        }
        appendDepthFilterParam(
            filter_state, getDepthFilterStatusParamName(normalized_filter_name, config_schema.name),
            format_filter_config_value(config_schema, filter->getConfigValue(config_schema.name)));
      }
    } catch (const std::exception &) {
      // Keep the state without dynamic params if runtime querying fails.
    }
  }

  return filter_state;
}

void OBCameraNode::publishDepthFiltersStatus() {
  if (!depth_filters_status_pub_) {
    return;
  }

  std::vector<std::shared_ptr<ob::Filter>> depth_filters_snapshot;
  {
    std::lock_guard<std::mutex> depth_filter_lock(depth_filter_mutex_);
    depth_filters_snapshot = depth_filter_list_;
  }

  auto find_depth_filter = [&depth_filters_snapshot,
                            this](const std::string &filter_name) -> std::shared_ptr<ob::Filter> {
    const auto normalized_name = normalizeDepthFilterName(filter_name);
    auto it = std::find_if(depth_filters_snapshot.begin(), depth_filters_snapshot.end(),
                           [&normalized_name](const auto &filter) {
                             return normalizeDepthFilterName(filter->type()) == normalized_name ||
                                    normalizeDepthFilterName(filter->getName()) == normalized_name;
                           });
    if (it == depth_filters_snapshot.end()) {
      return nullptr;
    }
    return *it;
  };

  auto sync_filter_enabled = [&find_depth_filter](const std::string &filter_name,
                                                  bool &cached_state) {
    auto filter = find_depth_filter(filter_name);
    if (!filter) {
      return;
    }
    try {
      cached_state = filter->isEnabled();
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  };

  sync_filter_enabled("DecimationFilter", enable_decimation_filter_);
  sync_filter_enabled("HDRMerge", enable_hdr_merge_);
  sync_filter_enabled("SequenceIdFilter", enable_sequence_id_filter_);
  sync_filter_enabled("SpatialAdvancedFilter", enable_spatial_filter_);
  sync_filter_enabled("TemporalFilter", enable_temporal_filter_);
  sync_filter_enabled("HoleFillingFilter", enable_hole_filling_filter_);
  sync_filter_enabled("DisparityTransform", enable_disparity_to_depth_);
  sync_filter_enabled("ThresholdFilter", enable_threshold_filter_);
  sync_filter_enabled("SpatialFastFilter", enable_spatial_fast_filter_);
  sync_filter_enabled("SpatialModerateFilter", enable_spatial_moderate_filter_);
  sync_filter_enabled("FalsePositiveFilter", enable_false_positive_filter_);
  sync_filter_enabled("MgcNoiseRemovalFilter", enable_mgc_noise_removal_filter_);
  sync_filter_enabled("LutNoiseRemovalFilter", enable_lut_noise_removal_filter_);

  if (device_->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL, OB_PERMISSION_READ_WRITE)) {
    try {
      enable_noise_removal_filter_ = device_->getBoolProperty(OB_PROP_DEPTH_SOFT_FILTER_BOOL);
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (device_->isPropertySupported(OB_PROP_DEPTH_MAX_DIFF_INT, OB_PERMISSION_WRITE)) {
    try {
      noise_removal_filter_min_diff_ = device_->getIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT);
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (device_->isPropertySupported(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, OB_PERMISSION_WRITE)) {
    try {
      noise_removal_filter_max_size_ = device_->getIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT);
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                                   OB_PERMISSION_READ_WRITE)) {
    try {
      enable_hardware_noise_removal_filter_ =
          device_->getBoolProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL);
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                   OB_PERMISSION_READ_WRITE)) {
    try {
      hardware_noise_removal_filter_threshold_ =
          device_->getFloatProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT);
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }

  if (auto filter = find_depth_filter("DecimationFilter")) {
    try {
      decimation_filter_scale_ =
          static_cast<int>(filter->as<ob::DecimationFilter>()->getScaleValue());
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (auto filter = find_depth_filter("SequenceIdFilter")) {
    try {
      sequence_id_filter_id_ = filter->as<ob::SequenceIdFilter>()->getSelectSequenceId();
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (auto filter = find_depth_filter("ThresholdFilter")) {
    try {
      threshold_filter_min_ = static_cast<int>(filter->getConfigValue("min"));
      threshold_filter_max_ = static_cast<int>(filter->getConfigValue("max"));
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (auto filter = find_depth_filter("SpatialAdvancedFilter")) {
    try {
      auto params = filter->as<ob::SpatialAdvancedFilter>()->getFilterParams();
      spatial_filter_alpha_ = params.alpha;
      spatial_filter_diff_threshold_ = params.disp_diff;
      spatial_filter_magnitude_ = params.magnitude;
      spatial_filter_radius_ = params.radius;
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (auto filter = find_depth_filter("TemporalFilter")) {
    try {
      temporal_filter_diff_threshold_ = static_cast<float>(filter->getConfigValue("diff_scale"));
      temporal_filter_weight_ = static_cast<float>(filter->getConfigValue("weight"));
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (auto filter = find_depth_filter("SpatialFastFilter")) {
    try {
      auto params = filter->as<ob::SpatialFastFilter>()->getFilterParams();
      spatial_fast_filter_radius_ = params.radius;
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }
  if (auto filter = find_depth_filter("SpatialModerateFilter")) {
    try {
      auto params = filter->as<ob::SpatialModerateFilter>()->getFilterParams();
      spatial_moderate_filter_diff_threshold_ = params.disp_diff;
      spatial_moderate_filter_magnitude_ = params.magnitude;
      spatial_moderate_filter_radius_ = params.radius;
    } catch (const std::exception &) {
      // Keep the cached value if runtime querying fails.
    }
  }

  DepthFiltersStatus msg;
  msg.header.stamp = node_->now();
  msg.header.frame_id = camera_name_;

  const bool noise_removal_filter_supported =
      device_->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL, OB_PERMISSION_READ_WRITE) ||
      device_->isPropertySupported(OB_PROP_DEPTH_MAX_DIFF_INT, OB_PERMISSION_WRITE) ||
      device_->isPropertySupported(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, OB_PERMISSION_WRITE);
  const bool hardware_noise_removal_filter_supported =
      device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                                   OB_PERMISSION_READ_WRITE) ||
      device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                   OB_PERMISSION_READ_WRITE);

  std::vector<std::string> ordered_filter_names;
  ordered_filter_names.reserve(depth_filters_snapshot.size() + 2);
  auto append_unique_filter_name = [&ordered_filter_names](const std::string &filter_name) {
    if (std::find(ordered_filter_names.begin(), ordered_filter_names.end(), filter_name) ==
        ordered_filter_names.end()) {
      ordered_filter_names.push_back(filter_name);
    }
  };
  for (const auto &filter : depth_filters_snapshot) {
    if (!filter) {
      continue;
    }
    append_unique_filter_name(normalizeDepthFilterName(filter->type()));
  }
  if (noise_removal_filter_supported) {
    append_unique_filter_name("NoiseRemovalFilter");
  }
  if (hardware_noise_removal_filter_supported) {
    append_unique_filter_name("HardwareNoiseRemovalFilter");
  }

  msg.filters.reserve(ordered_filter_names.size());
  for (const auto &filter_name : ordered_filter_names) {
    bool enabled = false;
    auto filter = find_depth_filter(filter_name);
    if (filter_name == "NoiseRemovalFilter") {
      enabled = enable_noise_removal_filter_;
    } else if (filter_name == "HardwareNoiseRemovalFilter") {
      enabled = enable_hardware_noise_removal_filter_;
    }
    if (filter) {
      try {
        enabled = filter->isEnabled();
      } catch (const std::exception &) {
        // Keep default value when runtime querying fails.
      }
    }
    msg.filters.push_back(buildDepthFilterState(filter_name, enabled, filter));
  }
  depth_filters_status_pub_->publish(msg);
}

OBCameraNode::OBCameraNode(rclcpp::Node *node, std::shared_ptr<ob::Device> device,
                           std::shared_ptr<Parameters> parameters, bool use_intra_process)
    : node_(node),
      device_(std::move(device)),
      parameters_(std::move(parameters)),
      logger_(node->get_logger()),
      use_intra_process_(use_intra_process) {
  pid_ = device_->getDeviceInfo()->getPid();
  RCLCPP_INFO_STREAM(logger_,
                     "OBCameraNode: use_intra_process: " << (use_intra_process ? "ON" : "OFF"));
  is_running_.store(true);
  stream_name_[COLOR] = "color";
  stream_name_[COLOR_LEFT] = "left_color";
  stream_name_[COLOR_RIGHT] = "right_color";
  stream_name_[DEPTH] = "depth";
  stream_name_[INFRA0] = "ir";
  stream_name_[INFRA1] = "left_ir";
  stream_name_[INFRA2] = "right_ir";
  stream_name_[ACCEL] = "accel";
  stream_name_[GYRO] = "gyro";
  compression_params_.push_back(cv::IMWRITE_PNG_COMPRESSION);
  compression_params_.push_back(0);
  compression_params_.push_back(cv::IMWRITE_PNG_STRATEGY);
  compression_params_.push_back(cv::IMWRITE_PNG_STRATEGY_DEFAULT);
  setupDefaultImageFormat();
  setupTopics();

  if (enable_frame_timestamp_csv_) {
    if (frame_timestamp_csv_file_.empty()) {
      frame_timestamp_csv_file_ =
          (std::filesystem::current_path() / (camera_name_ + "_frame_timestamp_stats.csv"))
              .string();
    }
    frame_timestamp_csv_logger_ =
        std::make_unique<FrameTimestampCsvLogger>(true, frame_timestamp_csv_file_, logger_);
    if (!frame_timestamp_csv_logger_->enabled()) {
      frame_timestamp_csv_logger_.reset();
    }
  }

#if defined(USE_RK_HW_DECODER)
  if (enable_stream_[COLOR] && width_.count(COLOR) && height_.count(COLOR)) {
    jpeg_decoder_ = std::make_unique<RKJPEGDecoder>(width_[COLOR], height_[COLOR]);
  }
  if (enable_stream_[COLOR_LEFT] && width_.count(COLOR_LEFT) && height_.count(COLOR_LEFT)) {
    jpeg_decoder_left_ = std::make_unique<RKJPEGDecoder>(width_[COLOR_LEFT], height_[COLOR_LEFT]);
  }
  if (enable_stream_[COLOR_RIGHT] && width_.count(COLOR_RIGHT) && height_.count(COLOR_RIGHT)) {
    jpeg_decoder_right_ =
        std::make_unique<RKJPEGDecoder>(width_[COLOR_RIGHT], height_[COLOR_RIGHT]);
  }
#elif defined(USE_NV_HW_DECODER)
  if (enable_stream_[COLOR] && width_.count(COLOR) && height_.count(COLOR)) {
    jpeg_decoder_ = std::make_unique<JetsonNvJPEGDecoder>(width_[COLOR], height_[COLOR]);
  }
  if (enable_stream_[COLOR_LEFT] && width_.count(COLOR_LEFT) && height_.count(COLOR_LEFT)) {
    jpeg_decoder_left_ =
        std::make_unique<JetsonNvJPEGDecoder>(width_[COLOR_LEFT], height_[COLOR_LEFT]);
  }
  if (enable_stream_[COLOR_RIGHT] && width_.count(COLOR_RIGHT) && height_.count(COLOR_RIGHT)) {
    jpeg_decoder_right_ =
        std::make_unique<JetsonNvJPEGDecoder>(width_[COLOR_RIGHT], height_[COLOR_RIGHT]);
  }
#endif
  if (enable_d2c_viewer_) {
    auto rgb_qos = getRMWQosProfileFromString(image_qos_[COLOR]);
    auto depth_qos = getRMWQosProfileFromString(image_qos_[DEPTH]);
    d2c_viewer_ = std::make_unique<D2CViewer>(node_, rgb_qos, depth_qos, use_intra_process_);
  }
  if (enable_stream_[COLOR]) {
    rgb_buffer_ = new uint8_t[width_[COLOR] * height_[COLOR] * 4];
  }
  if (enable_stream_[COLOR_LEFT]) {
    rgb_buffer_left_ = new uint8_t[width_[COLOR_LEFT] * height_[COLOR_LEFT] * 4];
  }
  if (enable_stream_[COLOR_RIGHT]) {
    rgb_buffer_right_ = new uint8_t[width_[COLOR_RIGHT] * height_[COLOR_RIGHT] * 4];
  }
  if (enable_colored_point_cloud_ && enable_stream_[DEPTH] && enable_stream_[COLOR]) {
    rgb_point_cloud_buffer_size_ = width_[COLOR] * height_[COLOR] * sizeof(OBColorPoint);
    xy_table_data_size_ = width_[DEPTH] * height_[DEPTH] * 2;
  }
  is_camera_node_initialized_ = true;

  fps_counter_color_ = std::make_unique<FpsCounter>("Color", logger_, 1);
  fps_counter_depth_ = std::make_unique<FpsCounter>("Depth", logger_, 1);
  fps_counter_left_ir_ = std::make_unique<FpsCounter>("Left Ir", logger_, 1);
  fps_counter_right_ir_ = std::make_unique<FpsCounter>("Right Ir", logger_, 1);

  LogLevel log_level = LogLevel::DEBUG;
  if (show_fps_enable_) {
    log_level = LogLevel::INFO;
  }
  fps_counter_color_->setLogLevel(log_level);
  fps_counter_depth_->setLogLevel(log_level);
  fps_counter_left_ir_->setLogLevel(log_level);
  fps_counter_right_ir_->setLogLevel(log_level);

  fps_delay_status_color_ = std::make_unique<FpsDelayStatus>(logger_);
  fps_delay_status_depth_ = std::make_unique<FpsDelayStatus>(logger_);
}

template <class T>
void OBCameraNode::setAndGetNodeParameter(
    T &param, const std::string &param_name, const T &default_value,
    const rcl_interfaces::msg::ParameterDescriptor &parameter_descriptor) {
  try {
    param = parameters_
                ->setParam(param_name, rclcpp::ParameterValue(default_value),
                           std::function<void(const rclcpp::Parameter &)>(), parameter_descriptor)
                .get<T>();
  } catch (const rclcpp::ParameterTypeException &ex) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to set parameter: " << param_name << ". " << ex.what());
    throw;
  }
}

OBCameraNode::~OBCameraNode() noexcept { clean(); }

void OBCameraNode::rebootDevice() {
  RCLCPP_DEBUG_STREAM(logger_, "Cleaning before rebooting device");
  malloc_trim(0);
  clean();
  malloc_trim(0);
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  RCLCPP_INFO_STREAM(logger_, "Rebooting device");
  if (device_) {
    device_->reboot();
  }
  malloc_trim(0);
  RCLCPP_DEBUG_STREAM(logger_, "Reboot device complete");
}

void OBCameraNode::clean() noexcept {
  if (cleaning_.exchange(true)) {
    RCLCPP_DEBUG(logger_, "clean() already running, skip re-entry");
    return;
  }
  // Set running flag to false first to signal all operations to stop
  is_running_.store(false);

  try {
    if (frame_timestamp_csv_logger_) {
      frame_timestamp_csv_logger_->shutdown();
      frame_timestamp_csv_logger_.reset();
    }
  } catch (...) {
    RCLCPP_WARN_STREAM(logger_, "Exception while shutting down frame timestamp CSV logger");
  }

  // Stop diagnostic timer and updater first BEFORE acquiring device_lock to prevent deadlock
  try {
    if (diagnostic_timer_) {
      diagnostic_timer_->cancel();
      // Wait for any currently executing timer callbacks to complete
      {
        std::unique_lock<std::mutex> lk(diagnostic_mutex_);
        diagnostic_cv_.wait_for(lk, std::chrono::milliseconds(100),
                                [this]() { return !diagnostic_running_; });
      }
      diagnostic_timer_.reset();
    }
    if (software_trigger_timer_) {
      software_trigger_timer_->cancel();
      software_trigger_timer_.reset();
    }
    if (diagnostic_updater_) {
      diagnostic_updater_.reset();
    }
  } catch (...) {
    // Ignore exceptions during diagnostic cleanup
  }

  // Now acquire the device lock for the rest of the cleanup
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);
  RCLCPP_DEBUG_STREAM(logger_, "Do OBCameraNode clean");

  RCLCPP_DEBUG_STREAM(logger_, "Stop tf thread");
  try {
    if (tf_thread_ && tf_thread_->joinable()) {
      tf_cv_.notify_all();  // Wake up tf thread if it's waiting
      tf_thread_->join();
    }
  } catch (...) {
    RCLCPP_DEBUG_STREAM(logger_, "Exception while stopping tf thread");
  }

  RCLCPP_DEBUG_STREAM(logger_, "Stop color frame thread");
  try {
    if (colorFrameThread_ && colorFrameThread_->joinable()) {
      color_frame_queue_cv_.notify_all();
      colorFrameThread_->join();
    }
    if (leftColorFrameThread_ && leftColorFrameThread_->joinable()) {
      left_color_frame_queue_cv_.notify_all();
      leftColorFrameThread_->join();
    }
    if (rightColorFrameThread_ && rightColorFrameThread_->joinable()) {
      right_color_frame_queue_cv_.notify_all();
      rightColorFrameThread_->join();
    }
  } catch (...) {
    RCLCPP_DEBUG_STREAM(logger_, "Exception while stopping color frame thread");
  }

  RCLCPP_DEBUG_STREAM(logger_, "stop streams");
  try {
    stopStreams();
    stopIMU();
    {
      std::lock_guard<std::mutex> lk(frame_info_logged_mutex_);
      frame_info_logged_.clear();
    }
  } catch (...) {
    RCLCPP_DEBUG_STREAM(logger_, "Exception while stopping streams");
  }

  // Clean up d2c_viewer_ before cleaning buffers
  RCLCPP_DEBUG_STREAM(logger_, "Clean d2c_viewer");
  try {
    if (d2c_viewer_) {
      d2c_viewer_.reset();
    }
  } catch (...) {
    RCLCPP_DEBUG_STREAM(logger_, "Exception while cleaning up d2c_viewer");
  }

  RCLCPP_DEBUG_STREAM(logger_, "Clean up buffers");
  try {
    delete[] rgb_buffer_;
    rgb_buffer_ = nullptr;
    delete[] rgb_buffer_left_;
    rgb_buffer_left_ = nullptr;
    delete[] rgb_buffer_right_;
    rgb_buffer_right_ = nullptr;

    if (jpeg_decoder_) {
      jpeg_decoder_.reset();
    }
    if (jpeg_decoder_left_) {
      jpeg_decoder_left_.reset();
    }
    if (jpeg_decoder_right_) {
      jpeg_decoder_right_.reset();
    }
  } catch (...) {
    RCLCPP_DEBUG_STREAM(logger_, "Exception while cleaning up buffers");
  }

  RCLCPP_DEBUG_STREAM(logger_, "OBCameraNode cleanup complete");
  cleaning_.store(false);
}

void OBCameraNode::setupDevices() {
  if (!depth_work_mode_.empty() &&
      device_->isPropertySupported(OB_STRUCT_CURRENT_DEPTH_ALG_MODE, OB_PERMISSION_READ_WRITE)) {
    auto depthModeList = device_->getDepthWorkModeList();
    for (uint32_t i = 0; i < depthModeList->getCount(); i++) {
      RCLCPP_INFO_STREAM(logger_, "depthModeList[" << i << "]: " << (*depthModeList)[i].name);
    }
    TRY_EXECUTE_BLOCK(device_->switchDepthWorkMode(depth_work_mode_.c_str()));
    RCLCPP_INFO_STREAM(logger_, "Set device preset: " << depth_work_mode_);
  } else if (!device_preset_.empty()) {
    try {
      RCLCPP_DEBUG_STREAM(logger_, "Available presets:");
      auto preset_list = device_->getAvailablePresetList();
      for (uint32_t i = 0; i < preset_list->getCount(); i++) {
        RCLCPP_DEBUG_STREAM(logger_, "Preset " << i << ": " << preset_list->getName(i));
      }
      TRY_EXECUTE_BLOCK(device_->loadPreset(device_preset_.c_str()));
      RCLCPP_INFO_STREAM(logger_, "Loaded device preset: " << device_->getCurrentPresetName());
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(
          logger_, "Failed to load device preset: " << orbbec_camera::formatObErrorWithStatus(e));
    } catch (const std::exception &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to load device preset: " << e.what());
    } catch (...) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to load device preset");
    }
  }

  if (!preset_resolution_config_.empty()) {
    OBPresetResolutionConfig presetResolutionConfig;
    std::istringstream iss(preset_resolution_config_);
    std::string token;
    std::vector<int> values;
    values.reserve(4);
    while (std::getline(iss, token, ',')) {
      values.push_back(std::stoi(token));
    }

    if (values.size() >= 4) {
      presetResolutionConfig.width = values[0];
      presetResolutionConfig.height = values[1];
      presetResolutionConfig.irDecimationFactor = values[2];
      presetResolutionConfig.depthDecimationFactor = values[3];
    } else {
      RCLCPP_WARN_STREAM(
          logger_,
          "Invalid preset_resolution_config parameter. "
          "Expected format: width,height,ir_decimation_factor,depth_decimation_factor");
    }

    RCLCPP_INFO_STREAM(
        logger_, "Set preset resolution config: "
                     << "width=" << presetResolutionConfig.width
                     << ", height=" << presetResolutionConfig.height
                     << ", ir_decimation=" << presetResolutionConfig.irDecimationFactor
                     << ", depth_decimation=" << presetResolutionConfig.depthDecimationFactor);

    TRY_EXECUTE_BLOCK(device_->setStructuredData(OB_STRUCT_PRESET_RESOLUTION_CONFIG,
                                                 (uint8_t *)&presetResolutionConfig,
                                                 sizeof(presetResolutionConfig)));
  }

  auto sensor_list = device_->getSensorList();
  for (size_t i = 0; i < sensor_list->getCount(); i++) {
    auto sensor = sensor_list->getSensor(i);
    auto profiles = sensor->getStreamProfileList();
    for (size_t j = 0; j < profiles->getCount(); j++) {
      auto profile = profiles->getProfile(j);
      stream_index_pair sip{profile->getType(), 0};
      if (sensors_.find(sip) != sensors_.end()) {
        continue;
      }
      sensors_[sip] = sensor;
    }
  }

  for (const auto &[stream_index, enable] : enable_stream_) {
    if (enable && sensors_.find(stream_index) == sensors_.end()) {
      RCLCPP_DEBUG_STREAM(logger_, magic_enum::enum_name(stream_index.first)
                                       << " sensor not supported by current device, skipping");
      enable_stream_[stream_index] = false;
    }
  }
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info);

  if (retry_on_usb3_detection_failure_ &&
      device_->isPropertySupported(OB_PROP_DEVICE_USB3_REPEAT_IDENTIFY_BOOL,
                                   OB_PERMISSION_READ_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_DEVICE_USB3_REPEAT_IDENTIFY_BOOL,
                        retry_on_usb3_detection_failure_);
  }
  if (device_->isPropertySupported(OB_PROP_HEARTBEAT_BOOL, OB_PERMISSION_READ_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_HEARTBEAT_BOOL, enable_heartbeat_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current heartbeat: " << (device_->getBoolProperty(OB_PROP_HEARTBEAT_BOOL) ? "ON" : "OFF"));
  }
  device_->enableFirmwareLog(enable_firmware_log_);
  RCLCPP_INFO_STREAM(logger_, "Set firmware log to " << (enable_firmware_log_ ? "ON" : "OFF"));
  if (max_depth_limit_ > 0 &&
      device_->isPropertySupported(OB_PROP_MAX_DEPTH_INT, OB_PERMISSION_READ_WRITE)) {
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_MAX_DEPTH_INT, max_depth_limit_);
    RCLCPP_INFO_STREAM(
        logger_, "Current max depth limit: " << device_->getIntProperty(OB_PROP_MAX_DEPTH_INT));
  }
  if (min_depth_limit_ > 0 &&
      device_->isPropertySupported(OB_PROP_MIN_DEPTH_INT, OB_PERMISSION_READ_WRITE)) {
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_MIN_DEPTH_INT, min_depth_limit_);
    RCLCPP_INFO_STREAM(
        logger_, "Current min depth limit: " << device_->getIntProperty(OB_PROP_MIN_DEPTH_INT));
  }
  if (laser_energy_level_ != -1 &&
      device_->isPropertySupported(OB_PROP_LASER_ENERGY_LEVEL_INT, OB_PERMISSION_READ_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_LASER_ENERGY_LEVEL_INT);
    if (laser_energy_level_ < range.min || laser_energy_level_ > range.max) {
      RCLCPP_ERROR_STREAM(logger_,
                          "Laser energy level is out of range " << range.min << " - " << range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_LASER_ENERGY_LEVEL_INT, laser_energy_level_);
      auto new_laser_energy_level = device_->getIntProperty(OB_PROP_LASER_ENERGY_LEVEL_INT);
      RCLCPP_INFO_STREAM(logger_, "Current energy level: " << new_laser_energy_level);
    }
  }
  if (depth_registration_ && align_mode_ == "SW") {
    RCLCPP_DEBUG_STREAM(logger_, "Create align filter");
    align_filter_ = std::make_unique<ob::Align>(align_target_stream_);
  }
  if (sensors_.find(DEPTH) != sensors_.end() &&
      device_->isPropertySupported(OB_PROP_DISPARITY_TO_DEPTH_BOOL, OB_PERMISSION_READ_WRITE) &&
      device_->isPropertySupported(OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL, OB_PERMISSION_READ_WRITE)) {
    if (disparity_to_depth_mode_ == "HW") {
      device_->setBoolProperty(OB_PROP_DISPARITY_TO_DEPTH_BOOL, 1);
      device_->setBoolProperty(OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL, 0);
      RCLCPP_INFO_STREAM(logger_, "Disparity to depth mode: HW");
    } else if (disparity_to_depth_mode_ == "SW") {
      device_->setBoolProperty(OB_PROP_DISPARITY_TO_DEPTH_BOOL, 0);
      device_->setBoolProperty(OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL, 1);
      RCLCPP_INFO_STREAM(logger_, "Disparity to depth mode: SW");
    } else if (disparity_to_depth_mode_ == "disable") {
      device_->setBoolProperty(OB_PROP_DISPARITY_TO_DEPTH_BOOL, 0);
      device_->setBoolProperty(OB_PROP_SDK_DISPARITY_TO_DEPTH_BOOL, 0);
      RCLCPP_INFO_STREAM(logger_, "Disparity to depth mode: disabled");
    } else {
      RCLCPP_WARN_STREAM(logger_, "Unknown disparity to depth mode '"
                                      << disparity_to_depth_mode_ << "', keeping default settings");
    }
  }
  if (device_->isPropertySupported(OB_PROP_LDP_BOOL, OB_PERMISSION_READ_WRITE)) {
    if (device_->isPropertySupported(OB_PROP_LASER_CONTROL_INT, OB_PERMISSION_READ_WRITE)) {
      auto laser_enable = device_->getIntProperty(OB_PROP_LASER_CONTROL_INT);
      TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_LDP_BOOL, enable_ldp_);
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_LASER_CONTROL_INT, laser_enable);
    } else if (device_->isPropertySupported(OB_PROP_LASER_BOOL, OB_PERMISSION_READ_WRITE)) {
      if (!enable_ldp_) {
        auto laser_enable = device_->getIntProperty(OB_PROP_LASER_BOOL);
        TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_LDP_BOOL, enable_ldp_);
        std::this_thread::sleep_for(std::chrono::milliseconds(3));
        TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_LASER_BOOL, laser_enable);
      } else {
        TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_LDP_BOOL, enable_ldp_);
      }
    }
    RCLCPP_INFO_STREAM(
        logger_, "Current LDP: " << (device_->getBoolProperty(OB_PROP_LDP_BOOL) ? "ON" : "OFF"));
  }
  if (ldp_power_level_ != -1 &&
      device_->isPropertySupported(OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_LASER_POWER_LEVEL_CONTROL_INT);
    if (ldp_power_level_ < range.min || ldp_power_level_ > range.max) {
      RCLCPP_ERROR(logger_, "ldp power level value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_LASER_POWER_LEVEL_CONTROL_INT, ldp_power_level_);
      RCLCPP_INFO_STREAM(logger_, "Current lrm power level: " << device_->getIntProperty(
                                      OB_PROP_LASER_POWER_LEVEL_CONTROL_INT));
    }
  }
  if (device_->isPropertySupported(OB_PROP_LASER_CONTROL_INT, OB_PERMISSION_READ_WRITE)) {
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_LASER_CONTROL_INT, enable_laser_);
    RCLCPP_INFO_STREAM(logger_,
                       "Current G300 laser control: "
                           << (device_->getIntProperty(OB_PROP_LASER_CONTROL_INT) ? "ON" : "OFF"));
  }
  if (device_->isPropertySupported(OB_PROP_LASER_BOOL, OB_PERMISSION_READ_WRITE)) {
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_LASER_BOOL, enable_laser_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current laser control: " << (device_->getIntProperty(OB_PROP_LASER_BOOL) ? "ON" : "OFF"));
  }
  if (!sync_mode_str_.empty()) {
    auto sync_config = device_->getMultiDeviceSyncConfig();
    std::transform(sync_mode_str_.begin(), sync_mode_str_.end(), sync_mode_str_.begin(), ::toupper);
    sync_mode_ = OBSyncModeFromString(sync_mode_str_);
    sync_config.syncMode = sync_mode_;
    sync_config.depthDelayUs = depth_delay_us_;
    sync_config.colorDelayUs = color_delay_us_;
    sync_config.trigger2ImageDelayUs = trigger2image_delay_us_;
    sync_config.triggerOutDelayUs = trigger_out_delay_us_;
    sync_config.triggerOutEnable = trigger_out_enabled_;
    sync_config.framesPerTrigger = frames_per_trigger_;
    TRY_EXECUTE_BLOCK(device_->setMultiDeviceSyncConfig(sync_config));
    sync_config = device_->getMultiDeviceSyncConfig();
    RCLCPP_INFO_STREAM(logger_,
                       "Current sync mode: " << magic_enum::enum_name(sync_config.syncMode));
    if (sync_mode_ == OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING) {
      RCLCPP_INFO_STREAM(logger_, "Frames per trigger: " << sync_config.framesPerTrigger);
      RCLCPP_INFO_STREAM(logger_,
                         "Software trigger period " << software_trigger_period_.count() << " ms");
      software_trigger_timer_ = node_->create_wall_timer(software_trigger_period_, [this]() {
        if (software_trigger_enabled_) {
          TRY_EXECUTE_BLOCK(device_->triggerCapture());
        }
      });
    }
  }
  if (device_->isPropertySupported(OB_DEVICE_PTP_CLOCK_SYNC_ENABLE_BOOL,
                                   OB_PERMISSION_READ_WRITE)) {
    device_->setBoolProperty(OB_DEVICE_PTP_CLOCK_SYNC_ENABLE_BOOL, enable_ptp_config_);
    RCLCPP_INFO_STREAM(
        logger_, "Current PTP Config: "
                     << (device_->getBoolProperty(OB_DEVICE_PTP_CLOCK_SYNC_ENABLE_BOOL) ? "ON"
                                                                                        : "OFF"));
  }

  if (device_->isPropertySupported(OB_PROP_DEPTH_PRECISION_LEVEL_INT, OB_PERMISSION_READ_WRITE) &&
      !depth_precision_str_.empty()) {
    auto default_precision_level = device_->getIntProperty(OB_PROP_DEPTH_PRECISION_LEVEL_INT);
    if (default_precision_level != depth_precision_) {
      device_->setIntProperty(OB_PROP_DEPTH_PRECISION_LEVEL_INT, depth_precision_);
      const auto current_depth_precision =
          device_->getIntProperty(OB_PROP_DEPTH_PRECISION_LEVEL_INT);
      RCLCPP_INFO_STREAM(logger_, "Current depth precision: "
                                      << depthPrecisionLevelToString(current_depth_precision));
    }
  } else if (device_->isPropertySupported(OB_PROP_DEPTH_UNIT_FLEXIBLE_ADJUSTMENT_FLOAT,
                                          OB_PERMISSION_READ_WRITE) &&
             !depth_precision_str_.empty()) {
    auto depth_unit_flexible_adjustment = depthPrecisionFromString(depth_precision_str_);
    auto range = device_->getFloatPropertyRange(OB_PROP_DEPTH_UNIT_FLEXIBLE_ADJUSTMENT_FLOAT);
    RCLCPP_INFO_STREAM(logger_,
                       "Depth unit flexible adjustment range: " << range.min << " - " << range.max);
    if (depth_unit_flexible_adjustment < range.min || depth_unit_flexible_adjustment > range.max) {
      RCLCPP_ERROR_STREAM(
          logger_, "depth unit flexible adjustment value is out of range, please check the value");
    } else {
      TRY_TO_SET_PROPERTY(setFloatProperty, OB_PROP_DEPTH_UNIT_FLEXIBLE_ADJUSTMENT_FLOAT,
                          depth_unit_flexible_adjustment);
      RCLCPP_INFO_STREAM(
          logger_, "Current depth unit: "
                       << device_->getFloatProperty(OB_PROP_DEPTH_UNIT_FLEXIBLE_ADJUSTMENT_FLOAT)
                       << "mm");
    }
  }

  for (const auto &stream_index : IMAGE_STREAMS) {
    if (enable_stream_[stream_index]) {
      OBPropertyID mirrorPropertyID = OB_PROP_DEPTH_MIRROR_BOOL;
      if (stream_index == COLOR) {
        mirrorPropertyID = OB_PROP_COLOR_MIRROR_BOOL;
      } else if (stream_index == DEPTH) {
        mirrorPropertyID = OB_PROP_DEPTH_MIRROR_BOOL;
      } else if (stream_index == INFRA0) {
        mirrorPropertyID = OB_PROP_IR_MIRROR_BOOL;
      } else if (stream_index == INFRA1) {
        mirrorPropertyID = OB_PROP_IR_MIRROR_BOOL;
      } else if (stream_index == INFRA2) {
        mirrorPropertyID = OB_PROP_IR_RIGHT_MIRROR_BOOL;
      } else if (stream_index == COLOR_LEFT) {
        mirrorPropertyID = OB_PROP_COLOR_LEFT_MIRROR_BOOL;
      } else if (stream_index == COLOR_RIGHT) {
        mirrorPropertyID = OB_PROP_COLOR_RIGHT_MIRROR_BOOL;
      }
      if (device_->isPropertySupported(mirrorPropertyID, OB_PERMISSION_WRITE)) {
        TRY_TO_SET_PROPERTY(setBoolProperty, mirrorPropertyID, mirror_stream_[stream_index]);
        RCLCPP_INFO_STREAM(
            logger_, "Current " << stream_name_[stream_index] << " mirror: "
                                << (device_->getBoolProperty(mirrorPropertyID) ? "ON" : "OFF"));
      }
      OBPropertyID flipPropertyID = OB_PROP_DEPTH_FLIP_BOOL;
      if (stream_index == COLOR) {
        flipPropertyID = OB_PROP_COLOR_FLIP_BOOL;
      } else if (stream_index == DEPTH) {
        flipPropertyID = OB_PROP_DEPTH_FLIP_BOOL;
      } else if (stream_index == INFRA0) {
        flipPropertyID = OB_PROP_IR_FLIP_BOOL;
      } else if (stream_index == INFRA1) {
        flipPropertyID = OB_PROP_IR_FLIP_BOOL;
      } else if (stream_index == INFRA2) {
        flipPropertyID = OB_PROP_IR_RIGHT_FLIP_BOOL;
      } else if (stream_index == COLOR_LEFT) {
        flipPropertyID = OB_PROP_COLOR_LEFT_FLIP_BOOL;
      } else if (stream_index == COLOR_RIGHT) {
        flipPropertyID = OB_PROP_COLOR_RIGHT_FLIP_BOOL;
      }
      if (device_->isPropertySupported(flipPropertyID, OB_PERMISSION_WRITE)) {
        TRY_TO_SET_PROPERTY(setBoolProperty, flipPropertyID, flip_stream_[stream_index]);
        RCLCPP_INFO_STREAM(logger_,
                           "Current " << stream_name_[stream_index] << " flip: "
                                      << (device_->getBoolProperty(flipPropertyID) ? "ON" : "OFF"));
      }
      OBPropertyID rotationPropertyID = OB_PROP_DEPTH_ROTATE_INT;
      if (stream_index == COLOR) {
        rotationPropertyID = OB_PROP_COLOR_ROTATE_INT;
      } else if (stream_index == DEPTH) {
        rotationPropertyID = OB_PROP_DEPTH_ROTATE_INT;
      } else if (stream_index == INFRA0) {
        rotationPropertyID = OB_PROP_IR_ROTATE_INT;
      } else if (stream_index == INFRA1) {
        rotationPropertyID = OB_PROP_IR_ROTATE_INT;
      } else if (stream_index == INFRA2) {
        rotationPropertyID = OB_PROP_IR_RIGHT_ROTATE_INT;
      } else if (stream_index == COLOR_LEFT) {
        rotationPropertyID = OB_PROP_COLOR_LEFT_ROTATE_INT;
      } else if (stream_index == COLOR_RIGHT) {
        rotationPropertyID = OB_PROP_COLOR_RIGHT_ROTATE_INT;
      }
      if (rotation_stream_[stream_index] != -1 &&
          device_->isPropertySupported(rotationPropertyID, OB_PERMISSION_WRITE)) {
        TRY_TO_SET_PROPERTY(setIntProperty, rotationPropertyID, rotation_stream_[stream_index]);
        RCLCPP_INFO_STREAM(logger_, "Current " << stream_name_[stream_index] << " rotation: "
                                               << device_->getIntProperty(rotationPropertyID));
      }
    }
  }

  if (device_->isPropertySupported(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL,
                        enable_color_auto_white_balance_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current color auto white balance: "
            << (device_->getBoolProperty(OB_PROP_COLOR_AUTO_WHITE_BALANCE_BOOL) ? "ON" : "OFF"));
  }
  if (!color_preset_.empty() &&
      device_->isPropertySupported(OB_PROP_COLOR_PRESET_PRIORITY_INT, OB_PERMISSION_WRITE)) {
    std::string preset_key = color_preset_;
    std::transform(preset_key.begin(), preset_key.end(), preset_key.begin(), ::tolower);
    int preset_value = -1;
    if (preset_key == "default") {
      preset_value = 0;
    } else if (preset_key == "warm biased awb") {
      preset_value = 1;
    } else {
      RCLCPP_WARN_STREAM(
          logger_, "Unsupported color_preset: " << color_preset_
                                                << ". Supported values: Default, Warm Biased AWB");
    }
    if (preset_value >= 0) {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_PRESET_PRIORITY_INT, preset_value);
      RCLCPP_INFO_STREAM(logger_, "Current color preset: "
                                      << (device_->getIntProperty(OB_PROP_COLOR_PRESET_PRIORITY_INT)
                                              ? "Warm Biased AWB"
                                              : "Default"));
    }
  }
  if (color_exposure_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_EXPOSURE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_EXPOSURE_INT);
    if (color_exposure_ < range.min || color_exposure_ > range.max) {
      RCLCPP_ERROR(logger_, "color exposure value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_EXPOSURE_INT, color_exposure_);
      RCLCPP_INFO_STREAM(logger_, "Current color exposure: "
                                      << device_->getIntProperty(OB_PROP_COLOR_EXPOSURE_INT));
    }
  }
  if (color_gain_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_GAIN_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_GAIN_INT);
    if (color_gain_ < range.min || color_gain_ > range.max) {
      RCLCPP_ERROR(logger_, "color gain value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_GAIN_INT, color_gain_);
      RCLCPP_INFO_STREAM(logger_,
                         "Current color gain: " << device_->getIntProperty(OB_PROP_COLOR_GAIN_INT));
    }
  }
  if (device_->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT, OB_PERMISSION_WRITE)) {
    int set_enable_color_auto_exposure_priority = enable_color_auto_exposure_priority_ ? 1 : 0;
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT,
                        set_enable_color_auto_exposure_priority);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current color auto exposure priority: "
            << (device_->getIntProperty(OB_PROP_COLOR_AUTO_EXPOSURE_PRIORITY_INT) ? "ON" : "OFF"));
  }
  if (device_->isPropertySupported(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL,
                        enable_color_auto_exposure_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current color auto exposure: "
            << (device_->getBoolProperty(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL) ? "ON" : "OFF"));
  }
  if (color_white_balance_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_WHITE_BALANCE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_WHITE_BALANCE_INT);
    if (color_white_balance_ < range.min || color_white_balance_ > range.max) {
      RCLCPP_ERROR(logger_,
                   "color white balance value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_WHITE_BALANCE_INT, color_white_balance_);
      RCLCPP_INFO_STREAM(logger_, "Current color white balance: "
                                      << device_->getIntProperty(OB_PROP_COLOR_WHITE_BALANCE_INT));
    }
  }

  if (color_ae_max_exposure_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_AE_MAX_EXPOSURE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_AE_MAX_EXPOSURE_INT);
    if (color_ae_max_exposure_ < range.min || color_ae_max_exposure_ > range.max) {
      RCLCPP_ERROR(logger_,
                   "color AE max exposure value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_AE_MAX_EXPOSURE_INT,
                          color_ae_max_exposure_);
      RCLCPP_INFO_STREAM(logger_, "Current color AE max exposure: " << device_->getIntProperty(
                                      OB_PROP_COLOR_AE_MAX_EXPOSURE_INT));
    }
  }
  if (color_ae_max_gain_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_AE_MAX_GAIN_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_AE_MAX_GAIN_INT);
    if (color_ae_max_gain_ < range.min || color_ae_max_gain_ > range.max) {
      RCLCPP_ERROR(logger_,
                   "color AE max gain value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_AE_MAX_GAIN_INT, color_ae_max_gain_);
      RCLCPP_INFO_STREAM(logger_, "Current color AE max gain: "
                                      << device_->getIntProperty(OB_PROP_COLOR_AE_MAX_GAIN_INT));
    }
  }
  if (color_brightness_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_BRIGHTNESS_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_BRIGHTNESS_INT);
    if (color_brightness_ < range.min || color_brightness_ > range.max) {
      RCLCPP_ERROR(logger_, "color brightness value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_BRIGHTNESS_INT, color_brightness_);
      RCLCPP_INFO_STREAM(logger_, "Current color brightness: "
                                      << device_->getIntProperty(OB_PROP_COLOR_BRIGHTNESS_INT));
    }
  }
  if (color_roi_brightness_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_ROI_BRIGHTNESS_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_ROI_BRIGHTNESS_INT);
    if (color_roi_brightness_ < range.min || color_roi_brightness_ > range.max) {
      RCLCPP_ERROR(logger_,
                   "color roi brightness value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_ROI_BRIGHTNESS_INT, color_roi_brightness_);
      RCLCPP_INFO_STREAM(logger_, "Current color roi brightness: "
                                      << device_->getIntProperty(OB_PROP_COLOR_ROI_BRIGHTNESS_INT));
    }
  }
  if (color_sharpness_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_SHARPNESS_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_SHARPNESS_INT);
    if (color_sharpness_ < range.min || color_sharpness_ > range.max) {
      RCLCPP_ERROR(logger_, "color sharpness value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_SHARPNESS_INT, color_sharpness_);
      RCLCPP_INFO_STREAM(logger_, "Current color sharpness: "
                                      << device_->getIntProperty(OB_PROP_COLOR_SHARPNESS_INT));
    }
  }
  if (color_gamma_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_GAMMA_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_GAMMA_INT);
    if (color_gamma_ < range.min || color_gamma_ > range.max) {
      RCLCPP_ERROR(logger_, "color gamm value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_GAMMA_INT, color_gamma_);
      RCLCPP_INFO_STREAM(
          logger_, "Current color gamma: " << device_->getIntProperty(OB_PROP_COLOR_GAMMA_INT));
    }
  }
  if (color_saturation_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_SATURATION_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_SATURATION_INT);
    if (color_saturation_ < range.min || color_saturation_ > range.max) {
      RCLCPP_ERROR(logger_, "color saturation value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_SATURATION_INT, color_saturation_);
      RCLCPP_INFO_STREAM(logger_, "Current color saturation: "
                                      << device_->getIntProperty(OB_PROP_COLOR_SATURATION_INT));
    }
  }
  if (color_contrast_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_CONTRAST_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_CONTRAST_INT);
    if (color_contrast_ < range.min || color_contrast_ > range.max) {
      RCLCPP_ERROR(logger_, "color contrast value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_CONTRAST_INT, color_contrast_);
      RCLCPP_INFO_STREAM(logger_, "Current color contrast: "
                                      << device_->getIntProperty(OB_PROP_COLOR_CONTRAST_INT));
    }
  }
  if (color_hue_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_HUE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_COLOR_HUE_INT);
    if (color_hue_ < range.min || color_hue_ > range.max) {
      RCLCPP_ERROR(logger_, "color hue value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_HUE_INT, color_hue_);
      RCLCPP_INFO_STREAM(logger_,
                         "Current color hue: " << device_->getIntProperty(OB_PROP_COLOR_HUE_INT));
    }
  }
  if (color_backlight_compensation_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT,
                        color_backlight_compensation_);
    RCLCPP_INFO_STREAM(logger_, "Current color backlight compensation: " << device_->getIntProperty(
                                    OB_PROP_COLOR_BACKLIGHT_COMPENSATION_INT));
  }
  if (color_denoising_level_ != -1 &&
      device_->isPropertySupported(OB_PROP_COLOR_DENOISING_LEVEL_INT, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_DENOISING_LEVEL_INT, color_denoising_level_);
    RCLCPP_INFO_STREAM(logger_, "Current color denoising level: "
                                    << device_->getIntProperty(OB_PROP_COLOR_DENOISING_LEVEL_INT));
  }
  if (device_->isPropertySupported(OB_PROP_COLOR_ANTI_FLICKER_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_COLOR_ANTI_FLICKER_BOOL, color_anti_flicker_);
    RCLCPP_INFO_STREAM(
        logger_, "Current color anti-flicker to "
                     << (device_->getBoolProperty(OB_PROP_COLOR_ANTI_FLICKER_BOOL) ? "ON" : "OFF"));
  }
  if (!color_powerline_freq_.empty() &&
      device_->isPropertySupported(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, OB_PERMISSION_WRITE)) {
    if (color_powerline_freq_ == "disable") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 0);
    } else if (color_powerline_freq_ == "50hz") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 1);
    } else if (color_powerline_freq_ == "60hz") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 2);
    } else if (color_powerline_freq_ == "auto") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT, 3);
    }
    const auto current_color_powerline_freq =
        device_->getIntProperty(OB_PROP_COLOR_POWER_LINE_FREQUENCY_INT);
    RCLCPP_INFO_STREAM(logger_, "Current color powerline freq: " << colorPowerLineFrequencyToString(
                                    current_color_powerline_freq));
  }
  if (depth_exposure_ != -1 &&
      device_->isPropertySupported(OB_PROP_DEPTH_EXPOSURE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_DEPTH_EXPOSURE_INT);
    if (depth_exposure_ < range.min || depth_exposure_ > range.max) {
      RCLCPP_ERROR(logger_, "depth exposure value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DEPTH_EXPOSURE_INT, depth_exposure_);
      RCLCPP_INFO_STREAM(logger_, "Current depth exposure: "
                                      << device_->getIntProperty(OB_PROP_DEPTH_EXPOSURE_INT));
    }
  }
  if (depth_gain_ != -1 &&
      device_->isPropertySupported(OB_PROP_DEPTH_GAIN_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_DEPTH_GAIN_INT);
    if (depth_gain_ < range.min || depth_gain_ > range.max) {
      RCLCPP_ERROR(logger_, "depth gain value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DEPTH_GAIN_INT, depth_gain_);
      RCLCPP_INFO_STREAM(logger_,
                         "Current depth gain: " << device_->getIntProperty(OB_PROP_DEPTH_GAIN_INT));
    }
  }
  if (sensors_.find(DEPTH) != sensors_.end() &&
      device_->isPropertySupported(OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT, OB_PERMISSION_WRITE)) {
    int set_enable_depth_auto_exposure_priority = enable_depth_auto_exposure_priority_ ? 1 : 0;
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT,
                        set_enable_depth_auto_exposure_priority);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current depth auto exposure priority: "
            << (device_->getIntProperty(OB_PROP_DEPTH_AUTO_EXPOSURE_PRIORITY_INT) ? "ON" : "OFF"));
  }
  if (device_->isPropertySupported(OB_PROP_IR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_IR_AUTO_EXPOSURE_BOOL, enable_ir_auto_exposure_);
    RCLCPP_INFO_STREAM(
        logger_, "Current IR auto exposure: "
                     << (device_->getBoolProperty(OB_PROP_IR_AUTO_EXPOSURE_BOOL) ? "ON" : "OFF"));
  }
  if (mean_intensity_set_point_ != -1 &&
      device_->isPropertySupported(OB_PROP_IR_BRIGHTNESS_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_IR_BRIGHTNESS_INT);
    if (mean_intensity_set_point_ < range.min || mean_intensity_set_point_ > range.max) {
      RCLCPP_ERROR(logger_, "depth brightness value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_IR_BRIGHTNESS_INT, mean_intensity_set_point_);
      RCLCPP_INFO_STREAM(logger_, "Current depth brightness: "
                                      << device_->getIntProperty(OB_PROP_IR_BRIGHTNESS_INT));
    }
  }
  // ir ae max
  if (ir_ae_max_exposure_ != -1 &&
      device_->isPropertySupported(OB_PROP_IR_AE_MAX_EXPOSURE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_IR_AE_MAX_EXPOSURE_INT);
    if (ir_ae_max_exposure_ < range.min || ir_ae_max_exposure_ > range.max) {
      RCLCPP_ERROR(logger_,
                   "IR AE max exposure value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_IR_AE_MAX_EXPOSURE_INT, ir_ae_max_exposure_);
      RCLCPP_INFO_STREAM(logger_, "Current IR AE max exposure: "
                                      << device_->getIntProperty(OB_PROP_IR_AE_MAX_EXPOSURE_INT));
    }
  }
  // ir brightness
  if (ir_brightness_ != -1 &&
      device_->isPropertySupported(OB_PROP_IR_BRIGHTNESS_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_IR_BRIGHTNESS_INT);
    if (ir_brightness_ < range.min || ir_brightness_ > range.max) {
      RCLCPP_ERROR(logger_, "IR brightness value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_IR_BRIGHTNESS_INT, ir_brightness_);
      RCLCPP_INFO_STREAM(
          logger_, "Current IR brightness: " << device_->getIntProperty(OB_PROP_IR_BRIGHTNESS_INT));
    }
  }
  if (ir_exposure_ != -1 &&
      device_->isPropertySupported(OB_PROP_IR_EXPOSURE_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_IR_EXPOSURE_INT);
    if (ir_exposure_ < range.min || ir_exposure_ > range.max) {
      RCLCPP_ERROR(logger_, "ir exposure value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_IR_EXPOSURE_INT, ir_exposure_);
      RCLCPP_INFO_STREAM(
          logger_, "Current IR exposure: " << device_->getIntProperty(OB_PROP_IR_EXPOSURE_INT));
    }
  }
  if (ir_gain_ != -1 && device_->isPropertySupported(OB_PROP_IR_GAIN_INT, OB_PERMISSION_WRITE)) {
    auto range = device_->getIntPropertyRange(OB_PROP_IR_GAIN_INT);
    if (ir_gain_ < range.min || ir_gain_ > range.max) {
      RCLCPP_ERROR(logger_, "ir gain value is out of range[%d,%d], please check the value",
                   range.min, range.max);
    } else {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_IR_GAIN_INT, ir_gain_);
      RCLCPP_INFO_STREAM(logger_,
                         "Current IR gain: " << device_->getIntProperty(OB_PROP_IR_GAIN_INT));
    }
  }
  if (device_->isPropertySupported(OB_PROP_IR_LONG_EXPOSURE_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_IR_LONG_EXPOSURE_BOOL, enable_ir_long_exposure_);
    RCLCPP_INFO_STREAM(
        logger_, "Current IR long exposure: "
                     << (device_->getBoolProperty(OB_PROP_IR_LONG_EXPOSURE_BOOL) ? "ON" : "OFF"));
  }

  if (enable_noise_removal_filter_ && sensors_.find(DEPTH) != sensors_.end() &&
      device_->isPropertySupported(OB_PROP_DEPTH_MAX_DIFF_INT, OB_PERMISSION_WRITE)) {
    auto default_noise_removal_filter_min_diff =
        device_->getIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT);
    if (noise_removal_filter_min_diff_ != -1 &&
        default_noise_removal_filter_min_diff != noise_removal_filter_min_diff_) {
      auto range = device_->getIntPropertyRange(OB_PROP_DEPTH_MAX_DIFF_INT);
      if (noise_removal_filter_min_diff_ < range.min ||
          noise_removal_filter_min_diff_ > range.max) {
        RCLCPP_ERROR(logger_,
                     "noise removal filter min diff value is out of range[%d,%d], please check "
                     "the value",
                     range.min, range.max);
      } else {
        device_->setIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT, noise_removal_filter_min_diff_);
      }
    }
    RCLCPP_INFO_STREAM(logger_, "Current noise_removal_filter_min_diff: "
                                    << device_->getIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT));
  }

  if (enable_noise_removal_filter_ && sensors_.find(DEPTH) != sensors_.end() &&
      device_->isPropertySupported(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, OB_PERMISSION_WRITE)) {
    auto default_noise_removal_filter_max_size =
        device_->getIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT);
    if (noise_removal_filter_max_size_ != -1 &&
        default_noise_removal_filter_max_size != noise_removal_filter_max_size_) {
      auto range = device_->getIntPropertyRange(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT);
      if (noise_removal_filter_max_size_ < range.min ||
          noise_removal_filter_max_size_ > range.max) {
        RCLCPP_ERROR(logger_,
                     "noise removal filter max size value is out of range[%d,%d], please check "
                     "the value",
                     range.min, range.max);
      } else {
        device_->setIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, noise_removal_filter_max_size_);
      }
    }
    RCLCPP_INFO_STREAM(logger_, "Current noise_removal_filter_max_size: "
                                    << device_->getIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT));
  }
  if (sensors_.find(DEPTH) != sensors_.end() &&
      device_->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL, OB_PERMISSION_READ_WRITE)) {
    device_->setBoolProperty(OB_PROP_DEPTH_SOFT_FILTER_BOOL, enable_noise_removal_filter_);
    RCLCPP_INFO_STREAM(logger_, "Set noise removal filter to "
                                    << (enable_noise_removal_filter_ ? "true" : "false"));
  }
  if (disparity_range_mode_ != -1 &&
      device_->isPropertySupported(OB_PROP_DISP_SEARCH_RANGE_MODE_INT, OB_PERMISSION_WRITE)) {
    if (disparity_range_mode_ == 64) {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DISP_SEARCH_RANGE_MODE_INT, 0);
    } else if (disparity_range_mode_ == 128) {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DISP_SEARCH_RANGE_MODE_INT, 1);
    } else if (disparity_range_mode_ == 256) {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DISP_SEARCH_RANGE_MODE_INT, 2);
    } else {
      RCLCPP_ERROR(logger_, "disparity range mode does not support this setting");
    }
    const auto current_disparity_range_mode =
        device_->getIntProperty(OB_PROP_DISP_SEARCH_RANGE_MODE_INT);
    RCLCPP_INFO_STREAM(logger_, "Current disparity range mode: "
                                    << disparityRangeModeToString(current_disparity_range_mode));
  }
  if (device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                                   OB_PERMISSION_READ_WRITE)) {
    device_->setBoolProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                             enable_hardware_noise_removal_filter_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Set hardware noise removal filter to "
            << (device_->getBoolProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL) ? "true"
                                                                                     : "false"));
    if (device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                     OB_PERMISSION_READ_WRITE)) {
      if (hardware_noise_removal_filter_threshold_ != -1.0 &&
          enable_hardware_noise_removal_filter_) {
        device_->setFloatProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                  hardware_noise_removal_filter_threshold_);
        RCLCPP_INFO_STREAM(logger_, "Current hardware noise removal filter threshold: "
                                        << device_->getFloatProperty(
                                               OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT));
      }
    }
  }
  if (exposure_range_mode_ != "default" &&
      device_->isPropertySupported(OB_PROP_DEVICE_PERFORMANCE_MODE_INT, OB_PERMISSION_WRITE)) {
    if (exposure_range_mode_ == "ultimate") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DEVICE_PERFORMANCE_MODE_INT, 1);
    } else if (exposure_range_mode_ == "regular") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_DEVICE_PERFORMANCE_MODE_INT, 0);
    } else {
      RCLCPP_ERROR(logger_, "exposure range mode does not support this setting");
    }
    const auto current_exposure_range_mode =
        device_->getIntProperty(OB_PROP_DEVICE_PERFORMANCE_MODE_INT);
    RCLCPP_INFO_STREAM(logger_, "Current exposure range mode: "
                                    << exposureRangeModeToString(current_exposure_range_mode));
  }
  if (!load_config_json_file_path_.empty()) {
    device_->loadPresetFromJsonFile(load_config_json_file_path_.c_str());
    RCLCPP_INFO_STREAM(logger_, "Loaded config json file path : " << load_config_json_file_path_);
  }
  if (!export_config_json_file_path_.empty()) {
    device_->exportSettingsAsPresetJsonFile(export_config_json_file_path_.c_str());
    RCLCPP_INFO_STREAM(logger_,
                       "Exporting config json file path : " << export_config_json_file_path_);
  }
  if (device_->isPropertySupported(OB_PROP_SDK_ACCEL_FRAME_TRANSFORMED_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_SDK_ACCEL_FRAME_TRANSFORMED_BOOL,
                        enable_accel_data_correction_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current accel data correction: "
            << (device_->getBoolProperty(OB_PROP_SDK_ACCEL_FRAME_TRANSFORMED_BOOL) ? "ON" : "OFF"));
  }
  if (device_->isPropertySupported(OB_PROP_SDK_GYRO_FRAME_TRANSFORMED_BOOL, OB_PERMISSION_WRITE)) {
    TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_SDK_GYRO_FRAME_TRANSFORMED_BOOL,
                        enable_gyro_data_correction_);
    RCLCPP_INFO_STREAM(
        logger_,
        "Current gyro data correction: "
            << (device_->getBoolProperty(OB_PROP_SDK_GYRO_FRAME_TRANSFORMED_BOOL) ? "ON" : "OFF"));
  }
  if (isGemini335PID(pid_) && !intra_camera_sync_reference_.empty() &&
      (sync_mode_ == OB_MULTI_DEVICE_SYNC_MODE_SOFTWARE_TRIGGERING ||
       sync_mode_ == OB_MULTI_DEVICE_SYNC_MODE_HARDWARE_TRIGGERING) &&
      device_->isPropertySupported(OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT, OB_PERMISSION_WRITE)) {
    if (intra_camera_sync_reference_ == "Start") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT, 0);
    } else if (intra_camera_sync_reference_ == "Middle") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT, 1);
    } else if (intra_camera_sync_reference_ == "End") {
      TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT, 2);
    } else {
      RCLCPP_ERROR(logger_, "intra camera sync reference does not support this setting");
    }
    const auto current_intra_camera_sync_reference =
        device_->getIntProperty(OB_PROP_INTRA_CAMERA_SYNC_REFERENCE_INT);
    RCLCPP_INFO_STREAM(
        logger_, "Current intra camera sync reference: "
                     << intraCameraSyncReferenceToString(current_intra_camera_sync_reference));
  }
  if (device_->isPropertySupported(OB_PROP_DEVICE_AE_STRATEGY_INT, OB_PERMISSION_WRITE)) {
    device_->setIntProperty(OB_PROP_DEVICE_AE_STRATEGY_INT, (ae_strategy_ == "motion" ? 0 : 1));
    RCLCPP_INFO_STREAM(
        logger_, "Current Sports Mode: "
                     << (device_->getIntProperty(OB_PROP_DEVICE_AE_STRATEGY_INT) == 0 ? "ON"
                                                                                      : "OFF"));
  }

  if ((ae_reference_stream_ == "depth" || ae_reference_stream_ == "color") &&
      device_->isPropertySupported(OB_PROP_DEVICE_AE_REFERENCE_INT, OB_PERMISSION_WRITE)) {
    if (device_->isPropertySupported(OB_PROP_DEVICE_AE_REFERENCE_INT, OB_PERMISSION_WRITE)) {
      auto ae_reference = ae_reference_stream_ == "depth" ? 0 : 1;
      device_->setIntProperty(OB_PROP_DEVICE_AE_REFERENCE_INT, ae_reference);
      auto current_ae_reference = device_->getIntProperty(OB_PROP_DEVICE_AE_REFERENCE_INT);
      RCLCPP_INFO_STREAM(logger_, "Current AE Reference: "
                                      << (current_ae_reference == 0 ? "depthbased" : "colorbased"));
    }
  }
}
void OBCameraNode::setupColorPostProcessFilter() {
  try {
    auto color_sensor = device_->getSensor(OB_SENSOR_COLOR);
    if (color_sensor) {
      color_filter_list_ = color_sensor->createRecommendedFilters();
    }
  } catch (const std::exception &e) {
    RCLCPP_DEBUG_STREAM(logger_, "Main color sensor not found, trying left/right color sensors");
    auto left_color_sensor = device_->getSensor(OB_SENSOR_COLOR_LEFT);
    if (left_color_sensor) {
      left_color_filter_list_ = left_color_sensor->createRecommendedFilters();
    }
    auto right_color_sensor = device_->getSensor(OB_SENSOR_COLOR_RIGHT);
    if (right_color_sensor) {
      right_color_filter_list_ = right_color_sensor->createRecommendedFilters();
    }
  }
  if (color_filter_list_.empty() && left_color_filter_list_.empty() &&
      right_color_filter_list_.empty()) {
    RCLCPP_WARN_STREAM(logger_, "Failed to get any color sensor filter list");
  }
  for (size_t i = 0; i < color_filter_list_.size(); i++) {
    auto filter = color_filter_list_[i];
    std::map<std::string, bool> filter_params = {
        {"DecimationFilter", enable_color_decimation_filter_},
    };
    std::string filter_name = filter->type();
    RCLCPP_DEBUG_STREAM(logger_, "Configuring color filter: " << filter_name);
    if (filter_params.find(filter_name) != filter_params.end()) {
      const auto *value = filter_params[filter_name] ? "enabled" : "disabled";
      RCLCPP_INFO_STREAM(logger_, "Set color filter " << filter_name << " to " << value);
      filter->enable(filter_params[filter_name]);
    }
    if (filter_name == "DecimationFilter" && enable_color_decimation_filter_) {
      auto decimation_filter = filter->as<ob::DecimationFilter>();
      auto range = decimation_filter->getScaleRange();
      if (color_decimation_filter_scale_ != -1 && color_decimation_filter_scale_ <= range.max &&
          color_decimation_filter_scale_ >= range.min) {
        decimation_filter->setScaleValue(color_decimation_filter_scale_);
      }
      if (color_decimation_filter_scale_ != -1 && (color_decimation_filter_scale_ < range.min ||
                                                   color_decimation_filter_scale_ > range.max)) {
        RCLCPP_ERROR_STREAM(logger_, "Color Decimation filter scale value is out of range "
                                         << range.min << " - " << range.max);
      }
      RCLCPP_INFO_STREAM(logger_, "Current color decimation filter scale value: "
                                      << static_cast<int>(decimation_filter->getScaleValue()));
    }
  }

  for (size_t i = 0; i < left_color_filter_list_.size(); i++) {
    auto filter = left_color_filter_list_[i];
    std::map<std::string, bool> filter_params = {
        {"DecimationFilter", enable_left_color_decimation_filter_},
    };
    std::string filter_name = filter->type();
    RCLCPP_DEBUG_STREAM(logger_, "Configuring left color filter: " << filter_name);
    if (filter_params.find(filter_name) != filter_params.end()) {
      const auto *value = filter_params[filter_name] ? "enabled" : "disabled";
      RCLCPP_INFO_STREAM(logger_, "Set left color filter " << filter_name << " to " << value);
      filter->enable(filter_params[filter_name]);
    }
    if (filter_name == "DecimationFilter" && enable_left_color_decimation_filter_) {
      auto decimation_filter = filter->as<ob::DecimationFilter>();
      auto range = decimation_filter->getScaleRange();
      if (left_color_decimation_filter_scale_ != -1 &&
          left_color_decimation_filter_scale_ <= range.max &&
          left_color_decimation_filter_scale_ >= range.min) {
        decimation_filter->setScaleValue(left_color_decimation_filter_scale_);
      }
      if (left_color_decimation_filter_scale_ != -1 &&
          (left_color_decimation_filter_scale_ < range.min ||
           left_color_decimation_filter_scale_ > range.max)) {
        RCLCPP_ERROR_STREAM(logger_, "Left Color Decimation filter scale value is out of range "
                                         << range.min << " - " << range.max);
      }
      RCLCPP_INFO_STREAM(logger_, "Current left color decimation filter scale value: "
                                      << static_cast<int>(decimation_filter->getScaleValue()));
    }
  }

  for (size_t i = 0; i < right_color_filter_list_.size(); i++) {
    auto filter = right_color_filter_list_[i];
    std::map<std::string, bool> filter_params = {
        {"DecimationFilter", enable_right_color_decimation_filter_},
    };
    std::string filter_name = filter->type();
    RCLCPP_DEBUG_STREAM(logger_, "Configuring right color filter: " << filter_name);
    if (filter_params.find(filter_name) != filter_params.end()) {
      const auto *value = filter_params[filter_name] ? "enabled" : "disabled";
      RCLCPP_INFO_STREAM(logger_, "Set right color filter " << filter_name << " to " << value);
      filter->enable(filter_params[filter_name]);
    }
    if (filter_name == "DecimationFilter" && enable_right_color_decimation_filter_) {
      auto decimation_filter = filter->as<ob::DecimationFilter>();
      auto range = decimation_filter->getScaleRange();
      if (right_color_decimation_filter_scale_ != -1 &&
          right_color_decimation_filter_scale_ <= range.max &&
          right_color_decimation_filter_scale_ >= range.min) {
        decimation_filter->setScaleValue(right_color_decimation_filter_scale_);
      }
      if (right_color_decimation_filter_scale_ != -1 &&
          (right_color_decimation_filter_scale_ < range.min ||
           right_color_decimation_filter_scale_ > range.max)) {
        RCLCPP_ERROR_STREAM(logger_, "Right Color Decimation filter scale value is out of range "
                                         << range.min << " - " << range.max);
      }
      RCLCPP_INFO_STREAM(logger_, "Current right color decimation filter scale value: "
                                      << static_cast<int>(decimation_filter->getScaleValue()));
    }
  }
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info);
  if (pid_ == GEMINI2_PID || pid_ == GEMINI2L_PID) {
    if (enable_color_decimation_filter_) {
      auto decimation_filter = std::make_shared<ob::DecimationFilter>();
      decimation_filter->enable(true);
      color_filter_list_.push_back(decimation_filter);
      auto range = decimation_filter->getScaleRange();
      if (color_decimation_filter_scale_ != -1 && color_decimation_filter_scale_ <= range.max &&
          color_decimation_filter_scale_ >= range.min) {
        decimation_filter->setScaleValue(color_decimation_filter_scale_);
      }
      if (color_decimation_filter_scale_ != -1 && (color_decimation_filter_scale_ < range.min ||
                                                   color_decimation_filter_scale_ > range.max)) {
        RCLCPP_ERROR_STREAM(logger_, "Color Decimation filter scale value is out of range "
                                         << range.min << " - " << range.max);
      }
      RCLCPP_INFO_STREAM(logger_, "Current color decimation filter scale value: "
                                      << static_cast<int>(decimation_filter->getScaleValue()));
    }
  }
}
void OBCameraNode::setupLeftIrPostProcessFilter() {
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info);
  if (isGemini335PID(pid_)) {
    auto left_ir_sensor = device_->getSensor(OB_SENSOR_IR_LEFT);
    left_ir_filter_list_ = left_ir_sensor->createRecommendedFilters();
    if (left_ir_filter_list_.empty()) {
      RCLCPP_WARN_STREAM(logger_, "Failed to get left ir sensor filter list");
      return;
    }
    for (size_t i = 0; i < left_ir_filter_list_.size(); i++) {
      auto filter = left_ir_filter_list_[i];
      std::map<std::string, bool> filter_params = {
          {"SequenceIdFilter", enable_left_ir_sequence_id_filter_},
      };
      std::string filter_name = filter->type();
      RCLCPP_DEBUG_STREAM(logger_, "Configuring left IR filter: " << filter_name);
      if (filter_params.find(filter_name) != filter_params.end()) {
        const auto *value = filter_params[filter_name] ? "enabled" : "disabled";
        RCLCPP_INFO_STREAM(logger_, "Set left IR filter " << filter_name << " to " << value);
        filter->enable(filter_params[filter_name]);
      }
      if (filter_name == "SequenceIdFilter" && enable_left_ir_sequence_id_filter_) {
        auto sequenced_filter = filter->as<ob::SequenceIdFilter>();
        if (left_ir_sequence_id_filter_id_ != -1) {
          sequenced_filter->selectSequenceId(left_ir_sequence_id_filter_id_);
        }
        RCLCPP_INFO_STREAM(logger_, "Current left ir SequenceIdFilter ID: "
                                        << sequenced_filter->getSelectSequenceId());
      }
    }
  }
}

void OBCameraNode::setupRightIrPostProcessFilter() {
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info);
  if (isGemini335PID(pid_)) {
    auto right_ir_sensor = device_->getSensor(OB_SENSOR_IR_RIGHT);
    right_ir_filter_list_ = right_ir_sensor->createRecommendedFilters();
    if (right_ir_filter_list_.empty()) {
      RCLCPP_WARN_STREAM(logger_, "Failed to get right ir sensor filter list");
      return;
    }
    for (size_t i = 0; i < right_ir_filter_list_.size(); i++) {
      auto filter = right_ir_filter_list_[i];
      std::map<std::string, bool> filter_params = {
          {"SequenceIdFilter", enable_right_ir_sequence_id_filter_},
      };
      std::string filter_name = filter->type();
      RCLCPP_DEBUG_STREAM(logger_, "Configuring right IR filter: " << filter_name);
      if (filter_params.find(filter_name) != filter_params.end()) {
        const auto *value = filter_params[filter_name] ? "enabled" : "disabled";
        RCLCPP_INFO_STREAM(logger_, "Set right IR filter " << filter_name << " to " << value);
        filter->enable(filter_params[filter_name]);
      }
      if (filter_name == "SequenceIdFilter" && enable_right_ir_sequence_id_filter_) {
        auto sequenced_filter = filter->as<ob::SequenceIdFilter>();
        if (right_ir_sequence_id_filter_id_ != -1) {
          sequenced_filter->selectSequenceId(right_ir_sequence_id_filter_id_);
        }
        RCLCPP_INFO_STREAM(logger_, "Current right ir SequenceIdFilter ID: "
                                        << sequenced_filter->getSelectSequenceId());
      }
    }
  }
}
void OBCameraNode::setupDepthPostProcessFilter() {
  auto depth_sensor = device_->getSensor(OB_SENSOR_DEPTH);
  // set depth sensor to filter
  depth_filter_list_ = depth_sensor->createRecommendedFilters();
  if (depth_filter_list_.empty()) {
    RCLCPP_WARN_STREAM(logger_, "Failed to get depth sensor filter list");
    return;
  }
  std::map<std::string, bool> filter_params = {
      {"DecimationFilter", enable_decimation_filter_},
      {"HDRMerge", enable_hdr_merge_},
      {"SequenceIdFilter", enable_sequence_id_filter_},
      {"SpatialAdvancedFilter", enable_spatial_filter_},
      {"TemporalFilter", enable_temporal_filter_},
      {"HoleFillingFilter", enable_hole_filling_filter_},
      {"DisparityTransform", enable_disparity_to_depth_},
      {"ThresholdFilter", enable_threshold_filter_},
      {"SpatialFastFilter", enable_spatial_fast_filter_},
      {"SpatialModerateFilter", enable_spatial_moderate_filter_},
      {"FalsePositiveFilter", enable_false_positive_filter_},
      {"MgcNoiseRemovalFilter", enable_mgc_noise_removal_filter_},
      {"LutNoiseRemovalFilter", enable_lut_noise_removal_filter_},
  };

  for (size_t i = 0; i < depth_filter_list_.size(); i++) {
    auto filter = depth_filter_list_[i];
    std::string filter_name = filter->type();
    RCLCPP_DEBUG_STREAM(logger_, "Configuring depth filter: " << filter_name);
    if (filter_params.find(filter_name) != filter_params.end()) {
      const auto *value = filter_params[filter_name] ? "enabled" : "disabled";
      RCLCPP_INFO_STREAM(logger_, "Set depth filter " << filter_name << " to " << value);
      filter->enable(filter_params[filter_name]);
      filter_status_[filter_name] = filter_params[filter_name];
    }
    if (filter_name == "DecimationFilter" && enable_decimation_filter_) {
      auto decimation_filter = filter->as<ob::DecimationFilter>();
      auto range = decimation_filter->getScaleRange();
      if (decimation_filter_scale_ != -1 && decimation_filter_scale_ <= range.max &&
          decimation_filter_scale_ >= range.min) {
        decimation_filter->setScaleValue(decimation_filter_scale_);
      }
      if (decimation_filter_scale_ != -1 &&
          (decimation_filter_scale_ < range.min || decimation_filter_scale_ > range.max)) {
        RCLCPP_ERROR_STREAM(logger_, "Decimation filter scale value is out of range "
                                         << range.min << " - " << range.max);
      }
      RCLCPP_INFO_STREAM(logger_, "Current decimation filter scale value: "
                                      << static_cast<int>(decimation_filter->getScaleValue()));
    } else if (filter_name == "ThresholdFilter" && enable_threshold_filter_) {
      auto threshold_filter = filter->as<ob::ThresholdFilter>();
      if (threshold_filter_min_ != -1 && threshold_filter_max_ != -1) {
        threshold_filter->setValueRange(threshold_filter_min_, threshold_filter_max_);
      }
      RCLCPP_INFO_STREAM(logger_, "Current threshold filter value range: "
                                      << static_cast<int>(threshold_filter->getConfigValue("min"))
                                      << " - "
                                      << static_cast<int>(threshold_filter->getConfigValue("max")));
    } else if (filter_name == "SpatialAdvancedFilter" && enable_spatial_filter_) {
      auto spatial_filter = filter->as<ob::SpatialAdvancedFilter>();
      if (spatial_filter_alpha_ != -1.0 && spatial_filter_magnitude_ != -1 &&
          spatial_filter_radius_ != -1 && spatial_filter_diff_threshold_ != -1) {
        OBSpatialAdvancedFilterParams params{};
        params.alpha = spatial_filter_alpha_;
        params.magnitude = spatial_filter_magnitude_;
        params.radius = spatial_filter_radius_;
        params.disp_diff = spatial_filter_diff_threshold_;
        spatial_filter->setFilterParams(params);
      }
      auto current_params = spatial_filter->getFilterParams();
      RCLCPP_INFO_STREAM(logger_, "Current SpatialFilter params: "
                                      << "alpha=" << current_params.alpha << ", disp_diff="
                                      << current_params.disp_diff << ", magnitude="
                                      << static_cast<int>(current_params.magnitude)
                                      << ", radius=" << current_params.radius);
    } else if (filter_name == "TemporalFilter" && enable_temporal_filter_) {
      auto temporal_filter = filter->as<ob::TemporalFilter>();
      if (temporal_filter_diff_threshold_ != -1.0 && temporal_filter_weight_ != -1.0) {
        temporal_filter->setDiffScale(temporal_filter_diff_threshold_);
        temporal_filter->setWeight(temporal_filter_weight_);
      }
      RCLCPP_INFO_STREAM(
          logger_,
          "Current TemporalFilter params: "
              << "diff_scale=" << static_cast<float>(temporal_filter->getConfigValue("diff_scale"))
              << ", weight=" << static_cast<float>(temporal_filter->getConfigValue("weight")));
    } else if (filter_name == "HoleFillingFilter" && enable_hole_filling_filter_ &&
               !hole_filling_filter_mode_.empty()) {
      auto hole_filling_filter = filter->as<ob::HoleFillingFilter>();
      RCLCPP_INFO_STREAM(logger_,
                         "Default hole filling filter mode: " << hole_filling_filter_mode_);
      OBHoleFillingMode hole_filling_mode = holeFillingModeFromString(hole_filling_filter_mode_);
      hole_filling_filter->setFilterMode(hole_filling_mode);
      RCLCPP_INFO_STREAM(logger_, "Current HoleFillingFilter mode: "
                                      << static_cast<int>(hole_filling_filter->getFilterMode()));
    } else if (filter_name == "SequenceIdFilter" && enable_sequence_id_filter_) {
      auto sequenced_filter = filter->as<ob::SequenceIdFilter>();
      if (sequence_id_filter_id_ != -1) {
        sequenced_filter->selectSequenceId(sequence_id_filter_id_);
      }
      RCLCPP_INFO_STREAM(
          logger_, "Current SequenceIdFilter ID: " << sequenced_filter->getSelectSequenceId());
    } else if (filter_name == "HDRMerge" && enable_hdr_merge_) {
      if (hdr_merge_exposure_1_ != -1 && hdr_merge_gain_1_ != -1 && hdr_merge_exposure_2_ != -1 &&
          hdr_merge_gain_2_ != -1) {
        auto hdr_merge_filter = filter->as<ob::HdrMerge>();
        hdr_merge_filter->enable(true);
        auto config = OBHdrConfig();
        config.enable = true;
        config.exposure_1 = hdr_merge_exposure_1_;
        config.gain_1 = hdr_merge_gain_1_;
        config.exposure_2 = hdr_merge_exposure_2_;
        config.gain_2 = hdr_merge_gain_2_;
        device_->setStructuredData(OB_STRUCT_DEPTH_HDR_CONFIG,
                                   reinterpret_cast<const uint8_t *>(&config), sizeof(config));
        uint32_t config_size = sizeof(config);
        device_->getStructuredData(OB_STRUCT_DEPTH_HDR_CONFIG, reinterpret_cast<uint8_t *>(&config),
                                   &config_size);
        RCLCPP_INFO_STREAM(
            logger_, "Current HDRMerge params: "
                         << "exposure_1=" << config.exposure_1 << ", gain_1=" << config.gain_1
                         << ", exposure_2=" << config.exposure_2 << ", gain_2=" << config.gain_2);
      }
    } else if (filter_name == "SpatialFastFilter" && enable_spatial_fast_filter_) {
      auto spatial_fast_filter = filter->as<ob::SpatialFastFilter>();
      OBSpatialFastFilterParams params{};
      if (spatial_fast_filter_radius_ != -1) {
        params.radius = spatial_fast_filter_radius_;
        spatial_fast_filter->setFilterParams(params);
      }
      auto current_params = spatial_fast_filter->getFilterParams();
      RCLCPP_INFO_STREAM(
          logger_, "Current SpatialFastFilter radius: " << static_cast<int>(current_params.radius));
    } else if (filter_name == "SpatialModerateFilter" && enable_spatial_moderate_filter_) {
      auto spatial_moderate_filter = filter->as<ob::SpatialModerateFilter>();
      OBSpatialModerateFilterParams params{};
      if (spatial_moderate_filter_diff_threshold_ != -1 &&
          spatial_moderate_filter_magnitude_ != -1 && spatial_moderate_filter_radius_ != -1) {
        params.magnitude = spatial_moderate_filter_magnitude_;
        params.radius = spatial_moderate_filter_radius_;
        params.disp_diff = spatial_moderate_filter_diff_threshold_;
        spatial_moderate_filter->setFilterParams(params);
      }
      auto current_params = spatial_moderate_filter->getFilterParams();
      RCLCPP_INFO_STREAM(logger_, "Current SpatialModerateFilter params: "
                                      << "disp_diff=" << current_params.disp_diff << ", magnitude="
                                      << static_cast<int>(current_params.magnitude)
                                      << ", radius=" << static_cast<int>(current_params.radius));

    } else {
      RCLCPP_DEBUG_STREAM(logger_, "Skip setting filter: " << filter_name);
    }
  }
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info);
  if (pid_ == GEMINI2_PID || pid_ == GEMINI2L_PID) {
    if (enable_decimation_filter_) {
      auto decimation_filter = std::make_shared<ob::DecimationFilter>();
      decimation_filter->enable(true);
      depth_filter_list_.push_back(decimation_filter);
      auto range = decimation_filter->getScaleRange();
      if (decimation_filter_scale_ != -1 && decimation_filter_scale_ <= range.max &&
          decimation_filter_scale_ >= range.min) {
        decimation_filter->setScaleValue(decimation_filter_scale_);
      }
      if (decimation_filter_scale_ != -1 &&
          (decimation_filter_scale_ < range.min || decimation_filter_scale_ > range.max)) {
        RCLCPP_ERROR_STREAM(logger_, "Decimation filter scale value is out of range "
                                         << range.min << " - " << range.max);
      }
      RCLCPP_INFO_STREAM(logger_, "Current decimation filter scale value: "
                                      << static_cast<int>(decimation_filter->getScaleValue()));
    }
  }
  set_filter_srv_ = node_->create_service<SetFilter>(
      "set_filter", [this](const std::shared_ptr<SetFilter ::Request> request,
                           std::shared_ptr<SetFilter ::Response> response) {
        setFilterCallback(request, response);
      });
}

void OBCameraNode::selectBaseStream() {
  if (enable_stream_[DEPTH]) {
    base_stream_ = DEPTH;
  } else if (enable_stream_[INFRA0]) {
    base_stream_ = INFRA0;
  } else if (enable_stream_[INFRA1]) {
    base_stream_ = INFRA1;
  } else if (enable_stream_[INFRA2]) {
    base_stream_ = INFRA2;
  } else if (enable_stream_[COLOR_LEFT]) {
    base_stream_ = COLOR_LEFT;
  } else if (enable_stream_[COLOR_RIGHT]) {
    base_stream_ = COLOR_RIGHT;
  } else if (enable_stream_[COLOR]) {
    base_stream_ = COLOR;
  }
}

void OBCameraNode::printSensorProfiles(const std::shared_ptr<ob::Sensor> &sensor) {
  auto profiles = sensor->getStreamProfileList();
  for (size_t i = 0; i < profiles->getCount(); i++) {
    auto origin_profile = profiles->getProfile(i);
    if (sensor->getType() == OB_SENSOR_COLOR) {
      auto profile = origin_profile->as<ob::VideoStreamProfile>();
      RCLCPP_INFO_STREAM(
          logger_, "color profile: " << profile->getWidth() << "x" << profile->getHeight() << " "
                                     << profile->getFps() << "fps " << profile->getFormat());
    } else if (sensor->getType() == OB_SENSOR_DEPTH) {
      auto profile = origin_profile->as<ob::VideoStreamProfile>();
      RCLCPP_INFO_STREAM(
          logger_, "depth profile: " << profile->getWidth() << "x" << profile->getHeight() << " "
                                     << profile->getFps() << "fps " << profile->getFormat());
    } else if (sensor->getType() == OB_SENSOR_IR) {
      auto profile = origin_profile->as<ob::VideoStreamProfile>();
      RCLCPP_INFO_STREAM(logger_, "ir profile: " << profile->getWidth() << "x"
                                                 << profile->getHeight() << " " << profile->getFps()
                                                 << "fps " << profile->getFormat());
    } else if (sensor->getType() == OB_SENSOR_ACCEL) {
      auto profile = origin_profile->as<ob::AccelStreamProfile>();
      RCLCPP_INFO_STREAM(logger_, "accel profile: sampleRate " << profile->getSampleRate()
                                                               << "  full scale_range "
                                                               << profile->getFullScaleRange());
    } else if (sensor->getType() == OB_SENSOR_GYRO) {
      auto profile = origin_profile->as<ob::GyroStreamProfile>();
      RCLCPP_INFO_STREAM(logger_, "gyro profile: sampleRate " << profile->getSampleRate()
                                                              << "  full scale_range "
                                                              << profile->getFullScaleRange());
    } else {
      RCLCPP_INFO_STREAM(logger_, "unknown profile: " << magic_enum::enum_name(sensor->getType()));
    }
  }
}

void OBCameraNode::setupProfiles() {
  // Image stream
  for (const auto &elem : IMAGE_STREAMS) {
    if (enable_stream_[elem]) {
      const auto &sensor = sensors_[elem];
      CHECK_NOTNULL(sensor.get());
      auto profiles = sensor->getStreamProfileList();
      CHECK_NOTNULL(profiles.get());
      CHECK(profiles->getCount() > 0);
      for (size_t i = 0; i < profiles->getCount(); i++) {
        auto base_profile = profiles->getProfile(i)->as<ob::VideoStreamProfile>();
        if (base_profile == nullptr) {
          throw std::runtime_error("Failed to get profile " + std::to_string(i));
        }
        auto profile = base_profile->as<ob::VideoStreamProfile>();
        if (profile == nullptr) {
          throw std::runtime_error("Failed cast profile to VideoStreamProfile");
        }
        RCLCPP_DEBUG_STREAM(
            logger_, "Sensor profile: "
                         << "stream_type: " << magic_enum::enum_name(profile->getType())
                         << "Format: " << profile->getFormat() << ", Width: " << profile->getWidth()
                         << ", Height: " << profile->getHeight() << ", FPS: " << profile->getFps());
        supported_profiles_[elem].emplace_back(profile);
      }
      std::shared_ptr<ob::VideoStreamProfile> selected_profile;
      std::shared_ptr<ob::VideoStreamProfile> default_profile;
      try {
        if (width_[elem] == 0 && height_[elem] == 0 && fps_[elem] == 0 &&
            format_[elem] == OB_FORMAT_UNKNOWN) {
          selected_profile = profiles->getProfile(0)->as<ob::VideoStreamProfile>();
        } else {
          if (isGemini305SeriesPID(pid_) && elem == DEPTH) {
            OBHardwareDecimationConfig conf;
            conf.originWidth = width_[elem];
            conf.originHeight = height_[elem];
            conf.factor = depth_decimation_factor_;
            selected_profile = profiles->getVideoStreamProfile(conf, format_[elem], fps_[elem]);
          } else if (isGemini305SeriesPID(pid_) && elem == INFRA1) {
            OBHardwareDecimationConfig conf;
            conf.originWidth = width_[elem];
            conf.originHeight = height_[elem];
            conf.factor = left_ir_decimation_factor_;
            selected_profile = profiles->getVideoStreamProfile(conf, format_[elem], fps_[elem]);
          } else if (isGemini305SeriesPID(pid_) && elem == INFRA2) {
            OBHardwareDecimationConfig conf;
            conf.originWidth = width_[elem];
            conf.originHeight = height_[elem];
            conf.factor = right_ir_decimation_factor_;
            selected_profile = profiles->getVideoStreamProfile(conf, format_[elem], fps_[elem]);
          } else {
            selected_profile = profiles->getVideoStreamProfile(width_[elem], height_[elem],
                                                               format_[elem], fps_[elem]);
          }
        }

      } catch (const ob::Error &ex) {
        RCLCPP_ERROR_STREAM(
            logger_, "Failed to get " << stream_name_[elem] << "  profile: " << ex.getMessage());
        RCLCPP_ERROR_STREAM(
            logger_, "Stream: " << magic_enum::enum_name(elem.first)
                                << ", Stream Index: " << elem.second << ", Width: " << width_[elem]
                                << ", Height: " << height_[elem] << ", FPS: " << fps_[elem]
                                << ", Format: " << magic_enum::enum_name(format_[elem]));
        RCLCPP_ERROR(logger_,
                     "Error: The device might be connected via USB 2.0. Please verify your "
                     "configuration and try again. The current process will now exit.");
        RCLCPP_INFO_STREAM(logger_, "Available profiles:");
        printSensorProfiles(sensor);
        RCLCPP_ERROR(logger_, "Failed to configure the requested stream profile, exiting.");
        exit(-1);
      }

      if (!selected_profile) {
        RCLCPP_WARN_STREAM(logger_,
                           "Requested stream configuration is not supported by the device: "
                               << "stream=" << magic_enum::enum_name(elem.first)
                               << ", stream_index=" << elem.second << ", width=" << width_[elem]
                               << ", height=" << height_[elem] << ", fps=" << fps_[elem]
                               << ", format=" << magic_enum::enum_name(format_[elem]));
        if (default_profile) {
          RCLCPP_WARN_STREAM(logger_, "Using the default profile instead");
          RCLCPP_WARN_STREAM(logger_, "Default profile FPS: " << default_profile->getFps());
          selected_profile = default_profile;
        } else {
          RCLCPP_ERROR_STREAM(logger_, "No default profile found, disabling stream "
                                           << magic_enum::enum_name(elem.first));
          enable_stream_[elem] = false;
          continue;
        }
      }
      CHECK_NOTNULL(selected_profile);
      stream_profile_[elem] = selected_profile;
      height_[elem] = static_cast<int>(selected_profile->getHeight());
      width_[elem] = static_cast<int>(selected_profile->getWidth());
      fps_[elem] = static_cast<int>(selected_profile->getFps());
      format_[elem] = selected_profile->getFormat();
      updateImageConfig(elem);
      if (selected_profile->format() == OB_FORMAT_BGRA) {
        images_[elem] = cv::Mat(height_[elem], width_[elem], CV_8UC4, cv::Scalar(0, 0, 0, 0));
        encoding_[elem] = sensor_msgs::image_encodings::BGRA8;
        unit_step_size_[COLOR] = 4 * sizeof(uint8_t);
      } else if (selected_profile->format() == OB_FORMAT_RGBA) {
        images_[elem] = cv::Mat(height_[elem], width_[elem], CV_8UC4, cv::Scalar(0, 0, 0, 0));
        encoding_[elem] = sensor_msgs::image_encodings::RGBA8;
        unit_step_size_[COLOR] = 4 * sizeof(uint8_t);
      } else {
        images_[elem] =
            cv::Mat(height_[elem], width_[elem], image_format_[elem], cv::Scalar(0, 0, 0));
      }
    }
  }
  // IMU
  for (const auto &stream_index : HID_STREAMS) {
    if (!enable_stream_[stream_index]) {
      continue;
    }
    try {
      auto profile_list = sensors_[stream_index]->getStreamProfileList();
      if (stream_index == ACCEL) {
        auto full_scale_range = fullAccelScaleRangeFromString(imu_range_[stream_index]);
        auto sample_rate = sampleRateFromString(imu_rate_[stream_index]);
        auto profile = profile_list->getAccelStreamProfile(full_scale_range, sample_rate);
        stream_profile_[stream_index] = profile;
      } else if (stream_index == GYRO) {
        auto full_scale_range = fullGyroScaleRangeFromString(imu_range_[stream_index]);
        auto sample_rate = sampleRateFromString(imu_rate_[stream_index]);
        auto profile = profile_list->getGyroStreamProfile(full_scale_range, sample_rate);
        stream_profile_[stream_index] = profile;
      }
      RCLCPP_INFO_STREAM(logger_, "stream " << stream_name_[stream_index] << " full scale range "
                                            << imu_range_[stream_index] << " sample rate "
                                            << imu_rate_[stream_index]);
    } catch (const ob::Error &e) {
      RCLCPP_INFO_STREAM(logger_, "Failed to setup << "
                                      << stream_name_[stream_index]
                                      << " profile: " << orbbec_camera::formatObErrorWithStatus(e));
      enable_stream_[stream_index] = false;
      stream_profile_[stream_index] = nullptr;
    }
  }
}
void OBCameraNode::updateImageConfig(const stream_index_pair &stream_index) {
  if (format_[stream_index] == OB_FORMAT_Y8) {
    image_format_[stream_index] = CV_8UC1;
    encoding_[stream_index] = stream_index.first == OB_STREAM_DEPTH
                                  ? sensor_msgs::image_encodings::TYPE_8UC1
                                  : sensor_msgs::image_encodings::MONO8;
    unit_step_size_[stream_index] = sizeof(uint8_t);
  }
  if (format_[stream_index] == OB_FORMAT_MJPG) {
    if (stream_index.first == OB_STREAM_IR || stream_index.first == OB_STREAM_IR_LEFT ||
        stream_index.first == OB_STREAM_IR_RIGHT) {
      image_format_[stream_index] = CV_8UC1;
      encoding_[stream_index] = sensor_msgs::image_encodings::MONO8;
      unit_step_size_[stream_index] = sizeof(uint8_t);
    }
  }
  if (format_[stream_index] == OB_FORMAT_Y16 &&
      (stream_index == COLOR || stream_index == COLOR_LEFT || stream_index == COLOR_RIGHT)) {
    image_format_[stream_index] = CV_16UC1;
    encoding_[stream_index] = sensor_msgs::image_encodings::MONO16;
    unit_step_size_[stream_index] = sizeof(uint16_t);
  }
}
int OBCameraNode::init_interleave_hdr_param() {
  device_->setIntProperty(OB_PROP_FRAME_INTERLEAVE_CONFIG_INDEX_INT, 1);
  if (!isnotLaserDevices(pid_)) {
    device_->setIntProperty(OB_PROP_LASER_CONTROL_INT, hdr_index1_laser_control_);
  }
  device_->setIntProperty(OB_PROP_DEPTH_EXPOSURE_INT, hdr_index1_depth_exposure_);
  device_->setIntProperty(OB_PROP_IR_EXPOSURE_INT, hdr_index1_depth_exposure_);
  device_->setIntProperty(OB_PROP_DEPTH_GAIN_INT, hdr_index1_depth_gain_);
  device_->setIntProperty(OB_PROP_IR_BRIGHTNESS_INT, hdr_index1_ir_brightness_);
  device_->setIntProperty(OB_PROP_IR_AE_MAX_EXPOSURE_INT, hdr_index1_ir_ae_max_exposure_);

  // set interleaveae
  device_->setIntProperty(OB_PROP_FRAME_INTERLEAVE_CONFIG_INDEX_INT, 0);
  if (!isnotLaserDevices(pid_)) {
    device_->setIntProperty(OB_PROP_LASER_CONTROL_INT, hdr_index0_laser_control_);
  }
  device_->setIntProperty(OB_PROP_DEPTH_EXPOSURE_INT, hdr_index0_depth_exposure_);
  device_->setIntProperty(OB_PROP_IR_EXPOSURE_INT, hdr_index0_depth_exposure_);
  device_->setIntProperty(OB_PROP_DEPTH_GAIN_INT, hdr_index0_depth_gain_);
  device_->setIntProperty(OB_PROP_IR_BRIGHTNESS_INT, hdr_index0_ir_brightness_);
  device_->setIntProperty(OB_PROP_IR_AE_MAX_EXPOSURE_INT, hdr_index0_ir_ae_max_exposure_);
  return 0;
}

int OBCameraNode::init_interleave_laser_param() {
  device_->setIntProperty(OB_PROP_FRAME_INTERLEAVE_CONFIG_INDEX_INT, 1);
  device_->setIntProperty(OB_PROP_LASER_CONTROL_INT, laser_index1_laser_control_);
  device_->setIntProperty(OB_PROP_DEPTH_EXPOSURE_INT, laser_index1_depth_exposure_);
  device_->setIntProperty(OB_PROP_IR_EXPOSURE_INT, laser_index1_depth_exposure_);
  device_->setIntProperty(OB_PROP_DEPTH_GAIN_INT, laser_index1_depth_gain_);
  device_->setIntProperty(OB_PROP_IR_BRIGHTNESS_INT, laser_index1_ir_brightness_);
  device_->setIntProperty(OB_PROP_IR_AE_MAX_EXPOSURE_INT, laser_index1_ir_ae_max_exposure_);

  // set interleaveae
  device_->setIntProperty(OB_PROP_FRAME_INTERLEAVE_CONFIG_INDEX_INT, 0);
  device_->setIntProperty(OB_PROP_LASER_CONTROL_INT, laser_index0_laser_control_);
  device_->setIntProperty(OB_PROP_DEPTH_EXPOSURE_INT, laser_index0_depth_exposure_);
  device_->setIntProperty(OB_PROP_IR_EXPOSURE_INT, laser_index0_depth_exposure_);
  device_->setIntProperty(OB_PROP_DEPTH_GAIN_INT, laser_index0_depth_gain_);
  device_->setIntProperty(OB_PROP_IR_BRIGHTNESS_INT, laser_index0_ir_brightness_);
  device_->setIntProperty(OB_PROP_IR_AE_MAX_EXPOSURE_INT, laser_index0_ir_ae_max_exposure_);
  return 0;
}
void OBCameraNode::startStreams() {
  if (pipeline_ != nullptr) {
    pipeline_.reset();
  }
  pipeline_ = std::make_unique<ob::Pipeline>(device_);

  try {
    setupPipelineConfig();
    pipeline_->start(pipeline_config_, [this](const std::shared_ptr<ob::FrameSet> &frame_set) {
      onNewFrameSetCallback(frame_set);
    });
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(logger_,
                        "Failed to start pipeline: " << orbbec_camera::formatObErrorWithStatus(e));
    RCLCPP_INFO_STREAM(logger_, "try to disable ir stream and try again");
    enable_stream_[INFRA0] = false;
    setupPipelineConfig();
    pipeline_->start(pipeline_config_, [this](const std::shared_ptr<ob::FrameSet> &frame_set) {
      onNewFrameSetCallback(frame_set);
    });
  } catch (...) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to start pipeline");
    throw std::runtime_error("Failed to start pipeline");
  }
  if (enable_stream_[COLOR] && !colorFrameThread_) {
    colorFrameThread_ = std::make_shared<std::thread>([this]() { onNewColorFrameCallback(); });
  }
  if (enable_stream_[COLOR_LEFT] && !leftColorFrameThread_) {
    leftColorFrameThread_ =
        std::make_shared<std::thread>([this]() { onNewLeftColorFrameCallback(); });
  }
  if (enable_stream_[COLOR_RIGHT] && !rightColorFrameThread_) {
    rightColorFrameThread_ =
        std::make_shared<std::thread>([this]() { onNewRightColorFrameCallback(); });
  }
  if (enable_frame_sync_) {
    RCLCPP_INFO_STREAM(logger_, "Enable frame sync");
    TRY_EXECUTE_BLOCK(pipeline_->enableFrameSync());
  } else {
    RCLCPP_INFO_STREAM(logger_, "Disable frame sync");
    TRY_EXECUTE_BLOCK(pipeline_->disableFrameSync());
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // set interleave mode
  if (interleave_ae_mode_ == "hdr" && interleave_frame_enable_) {
    RCLCPP_INFO_STREAM(logger_, "Set interleave mode to hdr");
    device_->loadFrameInterleave("Depth from HDR");
    init_interleave_hdr_param();
  } else if (interleave_ae_mode_ == "laser" && interleave_frame_enable_) {
    RCLCPP_INFO_STREAM(logger_, "Set interleave mode to laser");
    device_->loadFrameInterleave("Laser On-Off");
    init_interleave_laser_param();
  } else {
    RCLCPP_DEBUG_STREAM(logger_, "Set interleave mode to nothing");
  }
  // enable interleave frame
  if ((interleave_ae_mode_ == "hdr") || (interleave_ae_mode_ == "laser")) {
    RCLCPP_INFO_STREAM(logger_, "current interleave_ae_mode_: " << interleave_ae_mode_);
    if (device_->isPropertySupported(OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL, OB_PERMISSION_WRITE)) {
      TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL,
                          interleave_frame_enable_);
      RCLCPP_INFO_STREAM(
          logger_,
          "Enable enable_interleave_depth_frame to "
              << (device_->getBoolProperty(OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL) ? "true"
                                                                                 : "false"));
    }
  }
  // set interleave larse PATTERN_SYNC_DELAY
  if ((interleave_ae_mode_ == "laser") && interleave_frame_enable_ &&
      device_->isPropertySupported(OB_PROP_FRAME_INTERLEAVE_LASER_PATTERN_SYNC_DELAY_INT,
                                   OB_PERMISSION_READ_WRITE) &&
      (sync_mode_str_ == "PRIMARY" || sync_mode_str_ == "SOFTWARE_TRIGGERING")) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    TRY_TO_SET_PROPERTY(setIntProperty, OB_PROP_FRAME_INTERLEAVE_LASER_PATTERN_SYNC_DELAY_INT, 0);
    RCLCPP_INFO_STREAM(logger_,
                       "Current interleave laser pattern sync delay: " << device_->getIntProperty(
                           OB_PROP_FRAME_INTERLEAVE_LASER_PATTERN_SYNC_DELAY_INT));
  }
  pipeline_started_.store(true);
}

void OBCameraNode::startIMUSyncStream() {
  if (imuPipeline_ != nullptr) {
    imuPipeline_.reset();
  }

  imuPipeline_ = std::make_unique<ob::Pipeline>(device_);
  if (imu_sync_output_start_) {
    return;
  }

  // ACCEL
  auto accelProfiles = imuPipeline_->getStreamProfileList(OB_SENSOR_ACCEL);
  auto accel_range = fullAccelScaleRangeFromString(imu_range_[ACCEL]);
  auto accel_rate = sampleRateFromString(imu_rate_[ACCEL]);
  auto accelProfile = accelProfiles->getAccelStreamProfile(accel_range, accel_rate);
  // GYRO
  auto gyroProfiles = imuPipeline_->getStreamProfileList(OB_SENSOR_GYRO);
  auto gyro_range = fullGyroScaleRangeFromString(imu_range_[GYRO]);
  auto gyro_rate = sampleRateFromString(imu_rate_[GYRO]);
  auto gyroProfile = gyroProfiles->getGyroStreamProfile(gyro_range, gyro_rate);
  std::shared_ptr<ob::Config> imuConfig = std::make_shared<ob::Config>();
  imuConfig->enableStream(accelProfile);
  imuConfig->enableStream(gyroProfile);
  TRY_EXECUTE_BLOCK(imuPipeline_->enableFrameSync());

  try {
    imuPipeline_->start(imuConfig, [&](std::shared_ptr<ob::Frame> frame) {
      auto frameSet = frame->as<ob::FrameSet>();
      auto aFrame = frameSet->getFrame(OB_FRAME_ACCEL);
      auto gFrame = frameSet->getFrame(OB_FRAME_GYRO);
      if (aFrame && gFrame) {
        onNewIMUFrameSyncOutputCallback(aFrame, gFrame);
      }
    });

    imu_sync_output_start_ = true;
    RCLCPP_INFO_STREAM(
        logger_, "start accel stream with range: " << fullAccelScaleRangeToString(accel_range)
                                                   << ",rate:" << sampleRateToString(accel_rate)
                                                   << ", and start gyro stream with range:"
                                                   << fullGyroScaleRangeToString(gyro_range)
                                                   << ",rate:" << sampleRateToString(gyro_rate));
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(
        logger_, "Failed to start IMU sync stream: " << orbbec_camera::formatObErrorWithStatus(e));
    imu_sync_output_start_ = false;
  } catch (...) {
    RCLCPP_ERROR_STREAM(
        logger_, "Failed to start IMU stream, please check the imu_rate and imu_range parameters.");
    imu_sync_output_start_ = false;
  }
}

void OBCameraNode::startIMU() {
  if (enable_sync_output_accel_gyro_) {
    startIMUSyncStream();
  } else {
    for (const auto &stream_index : HID_STREAMS) {
      if (enable_stream_[stream_index] && !imu_started_[stream_index]) {
        auto imu_profile = stream_profile_[stream_index];
        CHECK_NOTNULL(imu_profile);
        RCLCPP_INFO_STREAM(logger_, "start " << stream_name_[stream_index] << " stream");
        CHECK_NOTNULL(sensors_[stream_index]);
        sensors_[stream_index]->start(
            imu_profile, [this, stream_index](const std::shared_ptr<ob::Frame> &frame) {
              onNewIMUFrameCallback(frame, stream_index);
            });
      }
    }
  }
}

void OBCameraNode::stopStreams() {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);

  if (!pipeline_started_ || !pipeline_) {
    RCLCPP_DEBUG_STREAM(logger_, "pipeline not started or not exist, skip stop pipeline");
    return;
  }

  // Stop diagnostic timer first to prevent crashes during shutdown
  try {
    if (diagnostic_timer_) {
      diagnostic_timer_->cancel();
      diagnostic_timer_.reset();
    }
    if (diagnostic_updater_) {
      diagnostic_updater_.reset();
    }
  } catch (...) {
    // Ignore exceptions during diagnostic cleanup
  }

  // Mark pipeline as stopping to prevent new operations
  pipeline_started_.store(false);

  try {
    // Check if device is still valid before stopping pipeline
    if (device_ && pipeline_) {
      pipeline_->stop();

      // disable interleave frame only if device is still connected
      if ((interleave_ae_mode_ == "hdr") || (interleave_ae_mode_ == "laser")) {
        try {
          RCLCPP_DEBUG_STREAM(logger_, "Current interleave AE mode: " << interleave_ae_mode_);
          if (device_->isPropertySupported(OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL,
                                           OB_PERMISSION_WRITE)) {
            interleave_frame_enable_ = false;
            RCLCPP_DEBUG_STREAM(logger_, "Set enable_interleave_depth_frame to "
                                             << (interleave_frame_enable_ ? "true" : "false"));
            TRY_TO_SET_PROPERTY(setBoolProperty, OB_PROP_FRAME_INTERLEAVE_ENABLE_BOOL,
                                interleave_frame_enable_);
          }
        } catch (const ob::Error &e) {
          RCLCPP_WARN_STREAM(logger_, "Failed to disable interleave frame during shutdown: "
                                          << orbbec_camera::formatObErrorWithStatus(e));
        } catch (...) {
          RCLCPP_WARN_STREAM(logger_, "Failed to disable interleave frame during shutdown");
        }
      }
    } else {
      RCLCPP_WARN_STREAM(logger_,
                         "Device or pipeline not available during stop - likely disconnected");
    }
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(logger_,
                        "Failed to stop pipeline: " << orbbec_camera::formatObErrorWithStatus(e));
  } catch (...) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to stop pipeline");
  }
}

void OBCameraNode::stopIMU() {
  std::lock_guard<decltype(device_lock_)> lock(device_lock_);

  if (enable_sync_output_accel_gyro_) {
    if (!imu_sync_output_start_ || !imuPipeline_) {
      RCLCPP_DEBUG_STREAM(logger_,
                          "IMU pipeline not started or unavailable, skip stopping IMU pipeline");
      return;
    }
    try {
      imuPipeline_->stop();
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(
          logger_, "Failed to stop imu pipeline: " << orbbec_camera::formatObErrorWithStatus(e));
    } catch (...) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to stop imu pipeline");
    }
    imu_sync_output_start_.store(false);
  } else {
    for (const auto &stream_index : HID_STREAMS) {
      if (imu_started_[stream_index]) {
        CHECK(sensors_.count(stream_index));
        RCLCPP_DEBUG_STREAM(logger_, "Stop " << stream_name_[stream_index] << " stream");
        try {
          sensors_[stream_index]->stop();
        } catch (const ob::Error &e) {
          RCLCPP_ERROR_STREAM(logger_, "Failed to stop "
                                           << stream_name_[stream_index] << " stream: "
                                           << orbbec_camera::formatObErrorWithStatus(e));
        }
        imu_started_[stream_index] = false;
      }
    }
  }
}

// cs_param_t rd_par = {0, 0}, param = {1, 3000}; //30
int OBCameraNode::openSocSyncPwmTrigger(uint16_t fps) {
  const char *devicePath = DEVICE_PATH;
  const int TRIGGER_MODE_ENABLE = 1;
  const int TRIGGER_MODE_DISABLE = 0;

  int ret = -1;
  cs_param_t param = {TRIGGER_MODE_ENABLE, fps};
  cs_param_t rd_par = {TRIGGER_MODE_DISABLE, 0};

  if (access(devicePath, F_OK) != 0) {
    std::cerr << "Device node " << devicePath << " does not exist." << std::endl;
    return ret;
  }
  gmsl_trigger_fd_ = open(DEVICE_PATH, O_RDWR);
  if (gmsl_trigger_fd_ < 0) {
    perror("open device failed\n");
    return gmsl_trigger_fd_;
  }

  std::cout << "Written param mode=" << param.mode << ", fps=" << param.fps << std::endl;
  ret = write(gmsl_trigger_fd_, &param, sizeof(param));
  if (ret < 0) {
    perror("write device failed\n");
    close(gmsl_trigger_fd_);
    return ret;
  }

  ret = read(gmsl_trigger_fd_, &rd_par, sizeof(rd_par));
  if (ret < 0) {
    perror("read device failed\n");
    close(gmsl_trigger_fd_);
    return ret;
  }
  std::cout << "Read param mode=" << rd_par.mode << ", fps=" << rd_par.fps << std::endl;

  std::cout << "Start hardware triggering..." << std::endl;

  return 0;
}
int OBCameraNode::closeSocSyncPwmTrigger() {
  if (gmsl_trigger_fd_ >= 0) {
    close(gmsl_trigger_fd_);
    gmsl_trigger_fd_ = -1;  // Reset file descriptors
    std::cout << "close camSync success" << std::endl;
    return 0;
  }
  return -1;
}

void OBCameraNode::startGmslTrigger() {
  if (gmsl_trigger_fps_ > 0 && enable_gmsl_trigger_) {
    RCLCPP_WARN_STREAM(logger_, "Start HardwareTrigger by soc-trigger-source. gmsl_trigger_fps_: "
                                    << gmsl_trigger_fps_);
    openSocSyncPwmTrigger(gmsl_trigger_fps_);
  } else {
    RCLCPP_WARN_STREAM(logger_,
                       "Start HardwareTrigger by soc-trigger-source. gmsl_trigger_fps_ illegal: "
                           << gmsl_trigger_fps_);
  }
}
void OBCameraNode::stopGmslTrigger() { closeSocSyncPwmTrigger(); }

void OBCameraNode::setupDefaultImageFormat() {
  format_[DEPTH] = OB_FORMAT_Y16;
  format_str_[DEPTH] = "Y16";
  image_format_[DEPTH] = CV_16UC1;
  encoding_[DEPTH] = sensor_msgs::image_encodings::TYPE_16UC1;
  unit_step_size_[DEPTH] = sizeof(uint16_t);
  format_[INFRA0] = OB_FORMAT_Y16;
  format_str_[INFRA0] = "Y16";
  image_format_[INFRA0] = CV_16UC1;
  encoding_[INFRA0] = sensor_msgs::image_encodings::MONO16;
  unit_step_size_[INFRA0] = sizeof(uint16_t);

  format_[INFRA1] = OB_FORMAT_Y16;
  format_str_[INFRA1] = "Y16";
  image_format_[INFRA1] = CV_16UC1;
  encoding_[INFRA1] = sensor_msgs::image_encodings::MONO16;
  unit_step_size_[INFRA1] = sizeof(uint16_t);

  format_[INFRA2] = OB_FORMAT_Y16;
  format_str_[INFRA2] = "Y16";
  image_format_[INFRA2] = CV_16UC1;
  encoding_[INFRA2] = sensor_msgs::image_encodings::MONO16;
  unit_step_size_[INFRA2] = sizeof(uint16_t);

  image_format_[COLOR] = CV_8UC3;
  encoding_[COLOR] = sensor_msgs::image_encodings::RGB8;
  unit_step_size_[COLOR] = 3 * sizeof(uint8_t);

  image_format_[COLOR_LEFT] = CV_8UC3;
  encoding_[COLOR_LEFT] = sensor_msgs::image_encodings::RGB8;
  unit_step_size_[COLOR_LEFT] = 3 * sizeof(uint8_t);

  image_format_[COLOR_RIGHT] = CV_8UC3;
  encoding_[COLOR_RIGHT] = sensor_msgs::image_encodings::RGB8;
  unit_step_size_[COLOR_RIGHT] = 3 * sizeof(uint8_t);
}

void OBCameraNode::getParameters() {
  setAndGetNodeParameter<std::string>(camera_name_, "camera_name", "camera");
  camera_link_frame_id_ = camera_name_ + "_link";
  for (auto stream_index : IMAGE_STREAMS) {
    std::string param_name = stream_name_[stream_index] + "_width";
    setAndGetNodeParameter(width_[stream_index], param_name, 0);
    param_name = stream_name_[stream_index] + "_height";
    setAndGetNodeParameter(height_[stream_index], param_name, 0);
    param_name = stream_name_[stream_index] + "_fps";
    setAndGetNodeParameter(fps_[stream_index], param_name, 0);
    param_name = "enable_" + stream_name_[stream_index];
    if (stream_index == DEPTH) {
      setAndGetNodeParameter(enable_stream_[stream_index], param_name, true);
    } else {
      setAndGetNodeParameter(enable_stream_[stream_index], param_name, false);
    }
    param_name = stream_name_[stream_index] + "_flip";
    setAndGetNodeParameter<bool>(flip_stream_[stream_index], param_name, false);
    param_name = stream_name_[stream_index] + "_mirror";
    setAndGetNodeParameter<bool>(mirror_stream_[stream_index], param_name, false);
    param_name = stream_name_[stream_index] + "_rotation";
    setAndGetNodeParameter<int>(rotation_stream_[stream_index], param_name, -1);
    param_name = camera_name_ + "_" + stream_name_[stream_index] + "_frame_id";
    std::string default_frame_id = camera_name_ + "_" + stream_name_[stream_index] + "_frame";
    setAndGetNodeParameter(frame_id_[stream_index], param_name, default_frame_id);
    std::string default_optical_frame_id =
        camera_name_ + "_" + stream_name_[stream_index] + "_optical_frame";
    param_name = stream_name_[stream_index] + "_optical_frame_id";
    setAndGetNodeParameter(optical_frame_id_[stream_index], param_name, default_optical_frame_id);
    param_name = stream_name_[stream_index] + "_format";
    setAndGetNodeParameter(format_str_[stream_index], param_name, format_str_[stream_index]);
    format_[stream_index] = OBFormatFromString(format_str_[stream_index]);
    updateImageConfig(stream_index);
    param_name = stream_name_[stream_index] + "_qos";
    setAndGetNodeParameter<std::string>(image_qos_[stream_index], param_name, "default");
    param_name = stream_name_[stream_index] + "_camera_info_qos";
    setAndGetNodeParameter<std::string>(camera_info_qos_[stream_index], param_name, "default");
  }

  for (auto stream_index : IMAGE_STREAMS) {
    depth_aligned_frame_id_[stream_index] = optical_frame_id_[COLOR];
  }

  accel_gyro_frame_id_ = camera_name_ + "_accel_gyro_optical_frame";

  setAndGetNodeParameter<bool>(enable_sync_output_accel_gyro_, "enable_sync_output_accel_gyro",
                               false);
  for (const auto &stream_index : HID_STREAMS) {
    std::string param_name = stream_name_[stream_index] + "_qos";
    setAndGetNodeParameter<std::string>(imu_qos_[stream_index], param_name, "default");
    param_name = "enable_" + stream_name_[stream_index];
    setAndGetNodeParameter<bool>(enable_stream_[stream_index], param_name, false);
    if (enable_sync_output_accel_gyro_) {
      enable_stream_[stream_index] = true;
    }
    param_name = stream_name_[stream_index] + "_rate";
    setAndGetNodeParameter<std::string>(imu_rate_[stream_index], param_name, "");
    param_name = stream_name_[stream_index] + "_range";
    setAndGetNodeParameter<std::string>(imu_range_[stream_index], param_name, "");
    param_name = camera_name_ + "_" + stream_name_[stream_index] + "_frame_id";
    std::string default_frame_id = camera_name_ + "_" + stream_name_[stream_index] + "_frame";
    setAndGetNodeParameter(frame_id_[stream_index], param_name, default_frame_id);
    std::string default_optical_frame_id =
        camera_name_ + "_" + stream_name_[stream_index] + "_optical_frame";
    param_name = stream_name_[stream_index] + "_optical_frame_id";
    setAndGetNodeParameter(optical_frame_id_[stream_index], param_name, default_optical_frame_id);
    depth_aligned_frame_id_[stream_index] =
        camera_name_ + "_" + stream_name_[COLOR] + "_optical_frame";
  }
  setAndGetNodeParameter<bool>(publish_tf_, "publish_tf", true);
  setAndGetNodeParameter<double>(tf_publish_rate_, "tf_publish_rate", 0.0);
  setAndGetNodeParameter<bool>(depth_registration_, "depth_registration", false);
  setAndGetNodeParameter<bool>(enable_point_cloud_, "enable_point_cloud", false);
  setAndGetNodeParameter<std::string>(ir_info_url_, "ir_info_url", "");
  setAndGetNodeParameter<std::string>(color_info_url_, "color_info_url", "");
  setAndGetNodeParameter<bool>(enable_colored_point_cloud_, "enable_colored_point_cloud", false);
  setAndGetNodeParameter<bool>(enable_point_cloud_, "enable_point_cloud", false);
  setAndGetNodeParameter<int>(point_cloud_decimation_filter_factor_,
                              "point_cloud_decimation_filter_factor", 1);
  setAndGetNodeParameter<std::string>(point_cloud_qos_, "point_cloud_qos", "default");
  setAndGetNodeParameter<bool>(enable_d2c_viewer_, "enable_d2c_viewer", false);
  setAndGetNodeParameter<std::string>(disparity_to_depth_mode_, "disparity_to_depth_mode", "HW");
  setAndGetNodeParameter<std::string>(depth_filter_config_, "depth_filter_config", "");
  if (!depth_filter_config_.empty()) {
    enable_depth_filter_ = true;
  }
  setAndGetNodeParameter<bool>(enable_frame_sync_, "enable_frame_sync", false);
  setAndGetNodeParameter<bool>(enable_color_auto_exposure_priority_,
                               "enable_color_auto_exposure_priority", false);
  setAndGetNodeParameter<bool>(enable_color_auto_exposure_, "enable_color_auto_exposure", true);
  setAndGetNodeParameter<bool>(enable_color_auto_white_balance_, "enable_color_auto_white_balance",
                               true);
  setAndGetNodeParameter<int>(color_ae_roi_left_, "color_ae_roi_left", -1);
  setAndGetNodeParameter<int>(color_ae_roi_top_, "color_ae_roi_top", -1);
  setAndGetNodeParameter<int>(color_ae_roi_right_, "color_ae_roi_right", -1);
  setAndGetNodeParameter<int>(color_ae_roi_bottom_, "color_ae_roi_bottom", -1);
  setAndGetNodeParameter<int>(color_exposure_, "color_exposure", -1);
  setAndGetNodeParameter<int>(color_gain_, "color_gain", -1);
  setAndGetNodeParameter<int>(color_white_balance_, "color_white_balance", -1);
  setAndGetNodeParameter<int>(color_ae_max_exposure_, "color_ae_max_exposure", -1);
  setAndGetNodeParameter<int>(color_ae_max_gain_, "color_ae_max_gain", -1);
  setAndGetNodeParameter<int>(color_brightness_, "color_brightness", -1);
  setAndGetNodeParameter<int>(color_roi_brightness_, "color_roi_brightness", -1);
  setAndGetNodeParameter<int>(color_sharpness_, "color_sharpness", -1);
  setAndGetNodeParameter<int>(color_gamma_, "color_gamma", -1);
  setAndGetNodeParameter<int>(color_saturation_, "color_saturation", -1);
  setAndGetNodeParameter<int>(color_contrast_, "color_contrast", -1);
  setAndGetNodeParameter<int>(color_hue_, "color_hue", -1);
  setAndGetNodeParameter<int>(color_backlight_compensation_, "color_backlight_compensation", -1);
  setAndGetNodeParameter<bool>(color_anti_flicker_, "color_anti_flicker", false);
  setAndGetNodeParameter<int>(color_denoising_level_, "color_denoising_level", -1);
  setAndGetNodeParameter<std::string>(color_powerline_freq_, "color_powerline_freq", "");
  setAndGetNodeParameter<std::string>(color_preset_, "color_preset", "Default");
  setAndGetNodeParameter<bool>(enable_color_decimation_filter_, "enable_color_decimation_filter",
                               false);
  setAndGetNodeParameter<int>(color_decimation_filter_scale_, "color_decimation_filter_scale", -1);
  setAndGetNodeParameter<bool>(enable_left_color_decimation_filter_,
                               "enable_left_color_decimation_filter", false);
  setAndGetNodeParameter<int>(left_color_decimation_filter_scale_,
                              "left_color_decimation_filter_scale", -1);
  setAndGetNodeParameter<bool>(enable_right_color_decimation_filter_,
                               "enable_right_color_decimation_filter", false);
  setAndGetNodeParameter<int>(right_color_decimation_filter_scale_,
                              "right_color_decimation_filter_scale", -1);
  setAndGetNodeParameter<bool>(enable_depth_auto_exposure_priority_,
                               "enable_depth_auto_exposure_priority", false);
  setAndGetNodeParameter<int>(depth_ae_roi_left_, "depth_ae_roi_left", -1);
  setAndGetNodeParameter<int>(depth_ae_roi_top_, "depth_ae_roi_top", -1);
  setAndGetNodeParameter<int>(depth_ae_roi_right_, "depth_ae_roi_right", -1);
  setAndGetNodeParameter<int>(depth_ae_roi_bottom_, "depth_ae_roi_bottom", -1);
  setAndGetNodeParameter<int>(depth_exposure_, "depth_exposure", -1);
  setAndGetNodeParameter<int>(depth_gain_, "depth_gain", -1);
  setAndGetNodeParameter<int>(depth_brightness_, "depth_brightness", -1);
  setAndGetNodeParameter<int>(mean_intensity_set_point_, "mean_intensity_set_point",
                              depth_brightness_);
  setAndGetNodeParameter<std::string>(depth_precision_str_, "depth_precision", "");
  setAndGetNodeParameter<bool>(enable_ir_auto_exposure_, "enable_ir_auto_exposure", true);
  setAndGetNodeParameter<int>(ir_exposure_, "ir_exposure", -1);
  setAndGetNodeParameter<int>(ir_gain_, "ir_gain", -1);
  setAndGetNodeParameter<int>(ir_ae_max_exposure_, "ir_ae_max_exposure", -1);
  setAndGetNodeParameter<int>(ir_brightness_, "ir_brightness", -1);
  setAndGetNodeParameter<bool>(enable_ir_long_exposure_, "enable_ir_long_exposure", true);
  setAndGetNodeParameter<bool>(enable_right_ir_sequence_id_filter_,
                               "enable_right_ir_sequence_id_filter", false);
  setAndGetNodeParameter<int>(right_ir_sequence_id_filter_id_, "right_ir_sequence_id_filter_id",
                              -1);
  setAndGetNodeParameter<bool>(enable_left_ir_sequence_id_filter_,
                               "enable_left_ir_sequence_id_filter", false);
  setAndGetNodeParameter<int>(left_ir_sequence_id_filter_id_, "left_ir_sequence_id_filter_id", -1);
  setAndGetNodeParameter<std::string>(preset_resolution_config_, "preset_resolution_config", "");
  setAndGetNodeParameter<std::string>(sync_mode_str_, "sync_mode", "");
  setAndGetNodeParameter<int>(depth_delay_us_, "depth_delay_us", 0);
  setAndGetNodeParameter<int>(color_delay_us_, "color_delay_us", 0);
  setAndGetNodeParameter<int>(trigger2image_delay_us_, "trigger2image_delay_us", 0);
  setAndGetNodeParameter<int>(trigger_out_delay_us_, "trigger_out_delay_us", 0);
  setAndGetNodeParameter<bool>(trigger_out_enabled_, "trigger_out_enabled", true);
  setAndGetNodeParameter<bool>(software_trigger_enabled_, "software_trigger_enabled", true);
  setAndGetNodeParameter<bool>(enable_ptp_config_, "enable_ptp_config", false);
  setAndGetNodeParameter<std::string>(cloud_frame_id_, "cloud_frame_id", "");
  if (enable_colored_point_cloud_ || enable_d2c_viewer_) {
    depth_registration_ = true;
  }
  if (!enable_stream_[COLOR]) {
    enable_colored_point_cloud_ = false;
    depth_registration_ = false;
  }
  setAndGetNodeParameter<bool>(enable_ldp_, "enable_ldp", true);
  setAndGetNodeParameter<int>(ldp_power_level_, "ldp_power_level", -1);
  setAndGetNodeParameter<double>(linear_accel_cov_, "linear_accel_cov", 0.0003);
  setAndGetNodeParameter<double>(angular_vel_cov_, "angular_vel_cov", 0.02);
  setAndGetNodeParameter<bool>(ordered_pc_, "ordered_pc", false);
  setAndGetNodeParameter<int>(max_save_images_count_, "max_save_images_count", 10);
  setAndGetNodeParameter<bool>(enable_depth_scale_, "enable_depth_scale", true);
  setAndGetNodeParameter<int>(depth_decimation_factor_, "depth_decimation_factor", 1);
  setAndGetNodeParameter<int>(left_ir_decimation_factor_, "left_ir_decimation_factor", 1);
  setAndGetNodeParameter<int>(right_ir_decimation_factor_, "right_ir_decimation_factor", 1);
  setAndGetNodeParameter<std::string>(depth_work_mode_, "depth_work_mode", "");
  if (isDepthWorkModeDevices(device_->getDeviceInfo()->getPid())) {
    setAndGetNodeParameter<std::string>(depth_work_mode_, "device_preset", "");
  } else {
    setAndGetNodeParameter<std::string>(device_preset_, "device_preset", "");
  }
  setAndGetNodeParameter<bool>(enable_decimation_filter_, "enable_decimation_filter", false);
  setAndGetNodeParameter<bool>(enable_hdr_merge_, "enable_hdr_merge", false);
  setAndGetNodeParameter<bool>(enable_sequence_id_filter_, "enable_sequence_id_filter", false);
  setAndGetNodeParameter<bool>(enable_disparity_to_depth_, "enable_disparity_to_depth", true);
  setAndGetNodeParameter<bool>(enable_threshold_filter_, "enable_threshold_filter", false);
  setAndGetNodeParameter<bool>(enable_hardware_noise_removal_filter_,
                               "enable_hardware_noise_removal_filter", true);
  setAndGetNodeParameter<bool>(enable_noise_removal_filter_, "enable_noise_removal_filter", true);
  setAndGetNodeParameter<bool>(enable_spatial_filter_, "enable_spatial_filter", false);
  setAndGetNodeParameter<bool>(enable_temporal_filter_, "enable_temporal_filter", false);
  setAndGetNodeParameter<bool>(enable_hole_filling_filter_, "enable_hole_filling_filter", false);
  setAndGetNodeParameter<bool>(enable_spatial_fast_filter_, "enable_spatial_fast_filter", false);
  setAndGetNodeParameter<bool>(enable_spatial_moderate_filter_, "enable_spatial_moderate_filter",
                               false);
  setAndGetNodeParameter<bool>(enable_mgc_noise_removal_filter_, "enable_mgc_noise_removal_filter",
                               false);
  setAndGetNodeParameter<bool>(enable_lut_noise_removal_filter_, "enable_lut_noise_removal_filter",
                               false);
  setAndGetNodeParameter<int>(decimation_filter_scale_, "decimation_filter_scale", -1);
  setAndGetNodeParameter<int>(sequence_id_filter_id_, "sequence_id_filter_id", -1);
  setAndGetNodeParameter<int>(threshold_filter_max_, "threshold_filter_max", -1);
  setAndGetNodeParameter<int>(threshold_filter_min_, "threshold_filter_min", -1);
  setAndGetNodeParameter<float>(hardware_noise_removal_filter_threshold_,
                                "hardware_noise_removal_filter_threshold", -1.0);
  setAndGetNodeParameter<int>(noise_removal_filter_min_diff_, "noise_removal_filter_min_diff", 256);
  setAndGetNodeParameter<int>(noise_removal_filter_max_size_, "noise_removal_filter_max_size", 80);
  setAndGetNodeParameter<float>(spatial_filter_alpha_, "spatial_filter_alpha", -1.0);
  setAndGetNodeParameter<int>(spatial_filter_diff_threshold_, "spatial_filter_diff_threshold", -1);
  setAndGetNodeParameter<int>(spatial_filter_magnitude_, "spatial_filter_magnitude", -1);
  setAndGetNodeParameter<int>(spatial_filter_radius_, "spatial_filter_radius", -1);
  setAndGetNodeParameter<float>(temporal_filter_diff_threshold_, "temporal_filter_diff_threshold",
                                -1.0);
  setAndGetNodeParameter<float>(temporal_filter_weight_, "temporal_filter_weight", -1.0);
  setAndGetNodeParameter<std::string>(hole_filling_filter_mode_, "hole_filling_filter_mode", "");
  setAndGetNodeParameter<bool>(enable_false_positive_filter_, "enable_false_positive_filter",
                               false);
  setAndGetNodeParameter<int>(hdr_merge_exposure_1_, "hdr_merge_exposure_1", -1);
  setAndGetNodeParameter<int>(hdr_merge_gain_1_, "hdr_merge_gain_1", -1);
  setAndGetNodeParameter<int>(hdr_merge_exposure_2_, "hdr_merge_exposure_2", -1);
  setAndGetNodeParameter<int>(hdr_merge_gain_2_, "hdr_merge_gain_2", -1);
  setAndGetNodeParameter<std::string>(align_mode_, "align_mode", "HW");
  setAndGetNodeParameter<double>(diagnostic_period_, "diagnostic_period", 1.0);
  setAndGetNodeParameter<bool>(enable_laser_, "enable_laser", true);
  std::string align_target_stream_str_;
  setAndGetNodeParameter<std::string>(align_target_stream_str_, "align_target_stream", "COLOR");
  align_target_stream_ = obStreamTypeFromString(align_target_stream_str_);
  setAndGetNodeParameter<bool>(retry_on_usb3_detection_failure_, "retry_on_usb3_detection_failure",
                               false);
  setAndGetNodeParameter<int>(laser_energy_level_, "laser_energy_level", -1);
  setAndGetNodeParameter<int>(min_depth_limit_, "min_depth_limit", 0);
  setAndGetNodeParameter<int>(max_depth_limit_, "max_depth_limit", 0);
  setAndGetNodeParameter<bool>(enable_heartbeat_, "enable_heartbeat", false);
  setAndGetNodeParameter<bool>(enable_firmware_log_, "enable_firmware_log", false);
  setAndGetNodeParameter<bool>(enable_color_undistortion_, "enable_color_undistortion", false);
  setAndGetNodeParameter<std::string>(time_domain_, "time_domain", "global");
  setAndGetNodeParameter<bool>(enable_frame_timestamp_csv_, "enable_frame_timestamp_csv", false);
  setAndGetNodeParameter<std::string>(frame_timestamp_csv_file_, "frame_timestamp_csv_file", "");
  setAndGetNodeParameter<std::string>(exposure_range_mode_, "exposure_range_mode", "default");
  setAndGetNodeParameter<std::string>(load_config_json_file_path_, "load_config_json_file_path",
                                      "");
  setAndGetNodeParameter<std::string>(export_config_json_file_path_, "export_config_json_file_path",
                                      "");
  setAndGetNodeParameter<bool>(enable_accel_data_correction_, "enable_accel_data_correction", true);
  setAndGetNodeParameter<bool>(enable_gyro_data_correction_, "enable_gyro_data_correction", true);
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info.get());

  if (device_preset_ == "Dual Color Streams") {
    RCLCPP_INFO_STREAM(logger_,
                       "Using Double Color preset, only left and right color streams are enabled.");
    enable_stream_[COLOR] = false;
    enable_stream_[DEPTH] = false;
    enable_stream_[INFRA0] = false;
    enable_stream_[INFRA1] = false;
    enable_stream_[INFRA2] = false;
    enable_stream_[LIDAR] = false;
    enable_stream_[COLOR_LEFT] = true;
    enable_stream_[COLOR_RIGHT] = true;

    enable_point_cloud_ = false;
    enable_colored_point_cloud_ = false;
    depth_registration_ = false;
    enable_d2c_viewer_ = false;
    enable_depth_filter_ = false;
    enable_color_undistortion_ = false;
  }

  if (isOpenNIDevice(pid_)) {
    time_domain_ = "system";
  }
  if (time_domain_ == "global") {
    device_->enableGlobalTimestamp(true);
  }
  setAndGetNodeParameter<int>(frames_per_trigger_, "frames_per_trigger", 2);
  int software_trigger_period = 33;
  setAndGetNodeParameter<int>(software_trigger_period, "software_trigger_period", 33);
  software_trigger_period_ = std::chrono::milliseconds(software_trigger_period);
  setAndGetNodeParameter<int>(gmsl_trigger_fps_, "gmsl_trigger_fps", 3000);
  setAndGetNodeParameter<bool>(enable_gmsl_trigger_, "enable_gmsl_trigger", false);
  setAndGetNodeParameter<std::string>(interleave_ae_mode_, "interleave_ae_mode", "hdr");
  setAndGetNodeParameter<bool>(interleave_frame_enable_, "interleave_frame_enable", false);
  setAndGetNodeParameter<bool>(interleave_skip_enable_, "interleave_skip_enable", false);
  setAndGetNodeParameter<int>(interleave_skip_index_, "interleave_skip_index", 1);

  // hdr and laser interleave params
  setAndGetNodeParameter<int>(hdr_index1_laser_control_, "hdr_index1_laser_control", 1);
  setAndGetNodeParameter<int>(hdr_index1_depth_exposure_, "hdr_index1_depth_exposure", 1);
  setAndGetNodeParameter<int>(hdr_index1_depth_gain_, "hdr_index1_depth_gain", 16);
  setAndGetNodeParameter<int>(hdr_index1_ir_brightness_, "hdr_index1_ir_brightness", 30);
  setAndGetNodeParameter<int>(hdr_index1_ir_ae_max_exposure_, "hdr_index1_ir_ae_max_exposure",
                              30458);
  setAndGetNodeParameter<int>(hdr_index0_laser_control_, "hdr_index0_laser_control", 1);
  setAndGetNodeParameter<int>(hdr_index0_depth_exposure_, "hdr_index0_depth_exposure", 7500);
  setAndGetNodeParameter<int>(hdr_index0_depth_gain_, "hdr_index0_depth_gain", 16);
  setAndGetNodeParameter<int>(hdr_index0_ir_brightness_, "hdr_index0_ir_brightness", 90);
  setAndGetNodeParameter<int>(hdr_index0_ir_ae_max_exposure_, "hdr_index0_ir_ae_max_exposure",
                              30458);

  setAndGetNodeParameter<int>(laser_index1_laser_control_, "laser_index1_laser_control", 0);
  setAndGetNodeParameter<int>(laser_index1_depth_exposure_, "laser_index1_depth_exposure", 3000);
  setAndGetNodeParameter<int>(laser_index1_depth_gain_, "laser_index1_depth_gain", 16);
  setAndGetNodeParameter<int>(laser_index1_ir_brightness_, "laser_index1_ir_brightness", 60);
  setAndGetNodeParameter<int>(laser_index1_ir_ae_max_exposure_, "laser_index1_ir_ae_max_exposure",
                              7000);
  setAndGetNodeParameter<int>(laser_index0_laser_control_, "laser_index0_laser_control", 1);
  setAndGetNodeParameter<int>(laser_index0_depth_exposure_, "laser_index0_depth_exposure", 3000);
  setAndGetNodeParameter<int>(laser_index0_depth_gain_, "laser_index0_depth_gain", 16);
  setAndGetNodeParameter<int>(laser_index0_ir_brightness_, "laser_index0_ir_brightness", 60);
  setAndGetNodeParameter<int>(laser_index0_ir_ae_max_exposure_, "laser_index0_ir_ae_max_exposure",
                              17000);
  setAndGetNodeParameter<int>(disparity_range_mode_, "disparity_range_mode", -1);
  setAndGetNodeParameter<int>(disparity_search_offset_, "disparity_search_offset", -1);
  setAndGetNodeParameter<bool>(disparity_offset_config_, "disparity_offset_config", false);
  setAndGetNodeParameter<int>(offset_index0_, "offset_index0", -1);
  setAndGetNodeParameter<int>(offset_index1_, "offset_index1", -1);

  setAndGetNodeParameter<std::string>(frame_aggregate_mode_, "frame_aggregate_mode", "ANY");

  setAndGetNodeParameter<bool>(show_fps_enable_, "show_fps_enable", false);
  setAndGetNodeParameter<bool>(enable_publish_extrinsic_, "enable_publish_extrinsic", false);
  setAndGetNodeParameter<std::string>(intra_camera_sync_reference_, "intra_camera_sync_reference",
                                      "Middle");
  setAndGetNodeParameter<std::string>(ae_reference_stream_, "ae_reference_stream", "depth");
  setAndGetNodeParameter<std::string>(ae_strategy_, "ae_strategy", "motion");

  RCLCPP_INFO_STREAM(logger_, "Current time domain: " << time_domain_);
  RCLCPP_DEBUG_STREAM(logger_, "hdr_index1_laser_control_ "
                                   << hdr_index1_laser_control_ << " hdr_index1_depth_exposure_ "
                                   << hdr_index1_depth_exposure_ << " hdr_index1_depth_gain_ "
                                   << hdr_index1_depth_gain_ << " hdr_index1_ir_brightness_ "
                                   << hdr_index1_ir_brightness_
                                   << " hdr_index1_ir_ae_max_exposure_ "
                                   << hdr_index1_ir_ae_max_exposure_ << "\n");
  RCLCPP_DEBUG_STREAM(logger_, "hdr_index0_laser_control_ "
                                   << hdr_index0_laser_control_ << " hdr_index0_depth_exposure_ "
                                   << hdr_index0_depth_exposure_ << " hdr_index0_depth_gain_ "
                                   << hdr_index0_depth_gain_ << " hdr_index0_ir_brightness_ "
                                   << hdr_index0_ir_brightness_
                                   << " hdr_index0_ir_ae_max_exposure_ "
                                   << hdr_index0_ir_ae_max_exposure_ << "\n");
  RCLCPP_DEBUG_STREAM(logger_,
                      "laser_index1_laser_control_ "
                          << laser_index1_laser_control_ << " laser_index1_depth_exposure_ "
                          << laser_index1_depth_exposure_ << " laser_index1_depth_gain_ "
                          << laser_index1_depth_gain_ << " laser_index1_ir_brightness_ "
                          << laser_index1_ir_brightness_ << " laser_index1_ir_ae_max_exposure_ "
                          << laser_index1_ir_ae_max_exposure_ << "\n");
  RCLCPP_DEBUG_STREAM(logger_,
                      "laser_index0_laser_control_ "
                          << laser_index0_laser_control_ << " laser_index0_depth_exposure_ "
                          << laser_index0_depth_exposure_ << " laser_index0_depth_gain_ "
                          << laser_index0_depth_gain_ << " laser_index0_ir_brightness_ "
                          << laser_index0_ir_brightness_ << " laser_index0_ir_ae_max_exposure_ "
                          << laser_index0_ir_ae_max_exposure_ << "\n");
}

void OBCameraNode::setupTopics() {
  try {
    getParameters();
    setupDevices();
    if (enable_stream_[DEPTH]) {
      setupDepthPostProcessFilter();
    }
    if (enable_stream_[COLOR] || enable_stream_[COLOR_LEFT] || enable_stream_[COLOR_RIGHT]) {
      setupColorPostProcessFilter();
    }
    if (enable_stream_[INFRA2]) {
      setupRightIrPostProcessFilter();
    }
    if (enable_stream_[INFRA1]) {
      setupLeftIrPostProcessFilter();
    }
    setupProfiles();
    setupCameraInfo();
    selectBaseStream();
    setupCameraCtrlServices();
    setupPublishers();
    setupDiagnosticUpdater();
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(logger_,
                        "Failed to setup topics: " << orbbec_camera::formatObErrorWithStatus(e));
    throw std::runtime_error(orbbec_camera::formatObErrorWithStatus(e));
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to setup topics: " << e.what());
    throw std::runtime_error(e.what());
  } catch (...) {
    RCLCPP_ERROR(logger_, "Failed to setup topics");
    throw std::runtime_error("Failed to setup topics");
  }
}

void OBCameraNode::onTemperatureUpdate(diagnostic_updater::DiagnosticStatusWrapper &status) {
  try {
    // Check to ensure we're not shutting down and device is valid
    if (!is_running_.load() || !is_camera_node_initialized_.load()) {
      status.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE,
                     "Device disconnected or shutting down");
      return;
    }

    // Try to acquire device lock with timeout to avoid blocking during shutdown
    std::unique_lock<decltype(device_lock_)> lock(device_lock_, std::try_to_lock);
    if (!lock.owns_lock()) {
      status.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE, "Device busy or shutting down");
      return;
    }

    if (!device_) {
      status.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE, "Device not available");
      return;
    }

    // Additional safety check - verify device is actually accessible
    try {
      auto device_info = device_->getDeviceInfo();
      if (!device_info) {
        status.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE, "Device info not available");
        return;
      }
    } catch (...) {
      status.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE, "Device not accessible");
      return;
    }

    OBDeviceTemperature temperature;
    uint32_t data_size = sizeof(OBDeviceTemperature);
    device_->getStructuredData(OB_STRUCT_DEVICE_TEMPERATURE,
                               reinterpret_cast<uint8_t *>(&temperature), &data_size);
    status.add("CPU Temperature", temperature.cpuTemp);
    status.add("IR Temperature", temperature.irTemp);
    status.add("LDM Temperature", temperature.ldmTemp);
    status.add("MainBoard Temperature", temperature.mainBoardTemp);
    status.add("TEC Temperature", temperature.tecTemp);
    status.add("IMU Temperature", temperature.imuTemp);
    status.add("RGB Temperature", temperature.rgbTemp);
    status.add("Left IR Temperature", temperature.irLeftTemp);
    status.add("Right IR Temperature", temperature.irRightTemp);
    status.add("Chip Top Temperature", temperature.chipTopTemp);
    status.add("Chip Bottom Temperature", temperature.chipBottomTemp);
    status.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Temperature is normal");
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(
        logger_, "Failed to TemperatureUpdate1: " << orbbec_camera::formatObErrorWithStatus(e));
    status.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR,
                   orbbec_camera::formatObErrorWithStatus(e));
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to TemperatureUpdate2: " << e.what());
    status.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, e.what());
  } catch (...) {
    RCLCPP_ERROR(logger_, "Failed to TemperatureUpdate3: Device is deactivated/disconnected!");
    status.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "Unknown error");
  }
}

void OBCameraNode::setupDiagnosticUpdater() {
  if (diagnostic_period_ <= 0.0) {
    return;
  }
  try {
    RCLCPP_INFO_STREAM(logger_, "Publish diagnostics every " << diagnostic_period_ << " seconds");
    auto info = device_->getDeviceInfo();
    std::string serial_number = info->getSerialNumber();
    diagnostic_updater_ = std::make_unique<diagnostic_updater::Updater>(node_, 10000.0);
    diagnostic_updater_->setHardwareID(serial_number);
    diagnostic_updater_->add("Temperatures", this, &OBCameraNode::onTemperatureUpdate);
    diagnostic_timer_ =
        node_->create_wall_timer(std::chrono::seconds(int(diagnostic_period_)), [this]() {
          try {
            // Check if we're still running and all components are valid
            if (!is_running_.load() || !diagnostic_updater_ ||
                !is_camera_node_initialized_.load() || !device_) {
              return;
            }

            // Try to acquire device lock with timeout to avoid blocking during shutdown
            std::unique_lock<decltype(device_lock_)> lock(device_lock_, std::try_to_lock);
            if (!lock.owns_lock()) {
              // Device is busy or shutting down, skip this update
              return;
            }
            // Mark diagnostic as running to prevent concurrent reset/publish races
            {
              std::lock_guard<std::mutex> lk(diagnostic_mutex_);
              diagnostic_running_ = true;
            }
            try {
              diagnostic_updater_->force_update();
            } catch (...) {
              std::lock_guard<std::mutex> lk(diagnostic_mutex_);
              diagnostic_running_ = false;
              diagnostic_cv_.notify_all();
              throw;
            }
            {
              std::lock_guard<std::mutex> lk(diagnostic_mutex_);
              diagnostic_running_ = false;
            }
            diagnostic_cv_.notify_all();
          } catch (const ob::Error &e) {
            RCLCPP_WARN_STREAM(
                logger_, "Diagnostic update failed: " << orbbec_camera::formatObErrorWithStatus(e)
                                                      << " - Device may be disconnected");
            // Stop the diagnostic timer if device is having issues
            try {
              if (diagnostic_timer_) {
                diagnostic_timer_->cancel();
                diagnostic_timer_.reset();
              }
            } catch (...) {
              // Ignore cleanup exceptions
            }
          } catch (const std::exception &e) {
            RCLCPP_WARN_STREAM(logger_, "Diagnostic update failed: " << e.what());
          } catch (...) {
            RCLCPP_WARN(logger_, "Diagnostic update failed: Unknown error");
          }
        });
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to setup diagnostic updater: "
                                     << orbbec_camera::formatObErrorWithStatus(e));
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to setup diagnostic updater: " << e.what());
  } catch (...) {
    RCLCPP_ERROR(logger_, "Failed to TemperatureUpdate");
  }
}

void OBCameraNode::setupPipelineConfig() {
  if (pipeline_config_) {
    pipeline_config_.reset();
  }
  pipeline_config_ = std::make_shared<ob::Config>();
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info.get());
  if (depth_registration_ && enable_stream_[COLOR] && enable_stream_[DEPTH] &&
      align_mode_ == "HW") {
    OBAlignMode align_mode = ALIGN_D2C_HW_MODE;
    RCLCPP_INFO_STREAM(logger_, "set align mode to " << magic_enum::enum_name(align_mode));
    pipeline_config_->setAlignMode(align_mode);
    RCLCPP_INFO_STREAM(logger_, "enable depth scale " << (enable_depth_scale_ ? "ON" : "OFF"));
    pipeline_config_->setDepthScaleRequire(enable_depth_scale_);
  }
  for (const auto &stream_index : IMAGE_STREAMS) {
    if (enable_stream_[stream_index]) {
      RCLCPP_DEBUG_STREAM(logger_, "Enable " << stream_name_[stream_index] << " stream");
      auto profile = stream_profile_[stream_index]->as<ob::VideoStreamProfile>();

      if (stream_index == COLOR && align_target_stream_ == OB_STREAM_COLOR && align_filter_) {
        auto video_profile = profile;
        align_filter_->setAlignToStreamProfile(video_profile);
      }
      if (stream_index == DEPTH && align_target_stream_ == OB_STREAM_DEPTH && align_filter_) {
        auto video_profile = profile;
        align_filter_->setAlignToStreamProfile(video_profile);
      }
      pipeline_config_->enableStream(stream_profile_[stream_index]);
    }
  }

  if (frame_aggregate_mode_ == "full_frame") {
    pipeline_config_->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_FULL_FRAME_REQUIRE);
  } else if (frame_aggregate_mode_ == "color_frame") {
    pipeline_config_->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_COLOR_FRAME_REQUIRE);
  } else if (frame_aggregate_mode_ == "disable") {
    pipeline_config_->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_DISABLE);
  } else {
    pipeline_config_->setFrameAggregateOutputMode(OB_FRAME_AGGREGATE_OUTPUT_ANY_SITUATION);
  }
}

void OBCameraNode::setupCameraInfo() {
  std::string color_camera_name = camera_name_ + "_color";
  if (!color_info_url_.empty()) {
    color_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(
        node_, color_camera_name, color_info_url_);
  }
  std::string ir_camera_name = camera_name_ + "_ir";
  if (!ir_info_url_.empty()) {
    ir_info_manager_ = std::make_unique<camera_info_manager::CameraInfoManager>(
        node_, ir_camera_name, ir_info_url_);
  }
}

void OBCameraNode::setupPublishers() {
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using CameraInfo = sensor_msgs::msg::CameraInfo;
  auto point_cloud_qos_profile = getRMWQosProfileFromString(point_cloud_qos_);
  if (use_intra_process_) {
    point_cloud_qos_profile = rmw_qos_profile_default;
  }
  if (enable_colored_point_cloud_) {
    depth_registration_cloud_pub_ = node_->create_publisher<PointCloud2>(
        "depth_registered/points",
        rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(point_cloud_qos_profile),
                    point_cloud_qos_profile));
  }
  if (enable_point_cloud_) {
    depth_cloud_pub_ = node_->create_publisher<PointCloud2>(
        "depth/points", rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(point_cloud_qos_profile),
                                    point_cloud_qos_profile));
  }
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info.get());
  for (const auto &stream_index : IMAGE_STREAMS) {
    if (!enable_stream_[stream_index]) {
      continue;
    }
    std::string name = stream_name_[stream_index];
    std::string topic = name + "/image_raw";
    auto image_qos = image_qos_[stream_index];
    auto image_qos_profile = getRMWQosProfileFromString(image_qos);
    if (use_intra_process_) {
      image_qos_profile = rmw_qos_profile_default;
    }
    if (use_intra_process_) {
      image_publishers_[stream_index] =
          std::make_shared<image_rcl_publisher>(*node_, topic, image_qos_profile);
    } else {
      image_publishers_[stream_index] =
          std::make_shared<image_transport_publisher>(*node_, topic, image_qos_profile);
    }

    topic = name + "/camera_info";
    auto camera_info_qos = camera_info_qos_[stream_index];
    auto camera_info_qos_profile = getRMWQosProfileFromString(camera_info_qos);
    if (use_intra_process_) {
      camera_info_qos_profile = rmw_qos_profile_default;
    }
    camera_info_publishers_[stream_index] = node_->create_publisher<CameraInfo>(
        topic, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(camera_info_qos_profile),
                           camera_info_qos_profile));
    if (isPublishMetaData(pid_)) {
      metadata_publishers_[stream_index] =
          node_->create_publisher<orbbec_camera_msgs::msg::Metadata>(
              name + "/metadata",
              rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(camera_info_qos_profile),
                          camera_info_qos_profile));
    }
    if (stream_index == COLOR && enable_color_undistortion_) {
      if (use_intra_process_) {
        color_undistortion_publisher_ = std::make_shared<image_rcl_publisher>(
            *node_, "color/image_undistorted", image_qos_profile);
      } else {
        color_undistortion_publisher_ = std::make_shared<image_transport_publisher>(
            *node_, "color/image_undistorted", image_qos_profile);
      }
    }
  }

  if (depth_registration_ && align_mode_ == "SW") {
    auto depth_image_qos_profile = getRMWQosProfileFromString(image_qos_[DEPTH]);
    if (use_intra_process_) {
      depth_image_qos_profile = rmw_qos_profile_default;
    }
    if (use_intra_process_) {
      depth_unaligned_publisher_ = std::make_shared<image_rcl_publisher>(
          *node_, "depth/image_unaligned", depth_image_qos_profile);
    } else {
      depth_unaligned_publisher_ = std::make_shared<image_transport_publisher>(
          *node_, "depth/image_unaligned", depth_image_qos_profile);
    }
  }

  if (enable_sync_output_accel_gyro_) {
    std::string topic_name = stream_name_[GYRO] + "_" + stream_name_[ACCEL] + "/sample";
    auto data_qos = getRMWQosProfileFromString(imu_qos_[GYRO]);
    if (use_intra_process_) {
      data_qos = rmw_qos_profile_default;
    }
    imu_gyro_accel_publisher_ = node_->create_publisher<sensor_msgs::msg::Imu>(
        topic_name, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(data_qos), data_qos));
    topic_name = stream_name_[GYRO] + "/imu_info";
    imu_info_publishers_[GYRO] = node_->create_publisher<orbbec_camera_msgs::msg::IMUInfo>(
        topic_name, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(data_qos), data_qos));
    topic_name = stream_name_[ACCEL] + "/imu_info";
    imu_info_publishers_[ACCEL] = node_->create_publisher<orbbec_camera_msgs::msg::IMUInfo>(
        topic_name, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(data_qos), data_qos));
  } else {
    for (const auto &stream_index : HID_STREAMS) {
      if (!enable_stream_[stream_index]) {
        continue;
      }
      std::string data_topic_name = stream_name_[stream_index] + "/sample";
      auto data_qos = getRMWQosProfileFromString(imu_qos_[stream_index]);
      if (use_intra_process_) {
        data_qos = rmw_qos_profile_default;
      }
      imu_publishers_[stream_index] = node_->create_publisher<sensor_msgs::msg::Imu>(
          data_topic_name, rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(data_qos), data_qos));
      data_topic_name = stream_name_[stream_index] + "/imu_info";
      imu_info_publishers_[stream_index] =
          node_->create_publisher<orbbec_camera_msgs::msg::IMUInfo>(
              data_topic_name,
              rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(data_qos), data_qos));
    }
  }
  auto extrinsics_qos = rclcpp::QoS(1).transient_local();
  if (use_intra_process_) {
    extrinsics_qos = rclcpp::QoS(1);
  }
  if (enable_stream_[DEPTH] && enable_stream_[INFRA0] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[INFRA0] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("depth_to_ir", extrinsics_qos);
  }
  if (enable_stream_[DEPTH] && enable_stream_[COLOR] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[COLOR] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("depth_to_color",
                                                                     extrinsics_qos);
  }
  if (enable_stream_[DEPTH] && enable_stream_[INFRA1] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[INFRA1] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("depth_to_left_ir",
                                                                     extrinsics_qos);
  }
  if (enable_stream_[DEPTH] && enable_stream_[INFRA2] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[INFRA2] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("depth_to_right_ir",
                                                                     extrinsics_qos);
  }
  if (enable_stream_[DEPTH] && enable_stream_[ACCEL] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[ACCEL] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("depth_to_accel",
                                                                     extrinsics_qos);
  }
  if (enable_stream_[DEPTH] && enable_stream_[GYRO] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[GYRO] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("depth_to_gyro",
                                                                     extrinsics_qos);
  }
  if (enable_stream_[COLOR_LEFT] && enable_stream_[COLOR_RIGHT] && enable_publish_extrinsic_) {
    depth_to_other_extrinsics_publishers_[COLOR_LEFT] =
        node_->create_publisher<orbbec_camera_msgs::msg::Extrinsics>("left_color_to_right_color",
                                                                     extrinsics_qos);
  }
  filter_status_pub_ =
      node_->create_publisher<std_msgs::msg::String>("depth_filter_status", extrinsics_qos);
  std_msgs::msg::String msg;
  msg.data = filter_status_.dump(2);
  filter_status_pub_->publish(msg);
  depth_filters_status_pub_ =
      node_->create_publisher<DepthFiltersStatus>("depth_filters/status", extrinsics_qos);
  publishDepthFiltersStatus();
}

void OBCameraNode::publishPointCloud(const std::shared_ptr<ob::FrameSet> &frame_set) {
  try {
    if (depth_registration_ || enable_colored_point_cloud_) {
      if (frame_set->depthFrame() != nullptr && frame_set->colorFrame() != nullptr) {
        publishColoredPointCloud(frame_set);
      }
    }

    if (enable_point_cloud_ && frame_set->depthFrame() != nullptr) {
      publishDepthPointCloud(frame_set);
    }

  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(logger_, orbbec_camera::formatObErrorWithStatus(e));
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(logger_, e.what());
  } catch (...) {
    RCLCPP_ERROR_STREAM(logger_, "publishPointCloud with unknown error");
  }
}

void OBCameraNode::publishRawDepthImage(const std::shared_ptr<ob::Frame> &depth_frame) {
  if (!depth_frame || !depth_unaligned_publisher_ || !depth_registration_ ||
      depth_unaligned_publisher_->get_subscription_count() == 0) {
    return;
  }

  auto video_frame = depth_frame->as<ob::DepthFrame>();
  if (!video_frame) {
    return;
  }

  int width = static_cast<int>(video_frame->getWidth());
  int height = static_cast<int>(video_frame->getHeight());
  auto frame_timestamp = getFrameTimestampUs(depth_frame);
  auto timestamp = fromUsToROSTime(frame_timestamp);

  std::string frame_id = optical_frame_id_[DEPTH];

  cv::Mat depth_image(height, width, image_format_[DEPTH]);
  memcpy(depth_image.data, video_frame->getData(), video_frame->getDataSize());

  auto depth_scale = video_frame->getValueScale();
  depth_image.convertTo(depth_image, depth_image.type(), depth_scale);

  sensor_msgs::msg::Image::UniquePtr image_msg(new sensor_msgs::msg::Image());
  cv_bridge::CvImage(std_msgs::msg::Header(), encoding_[DEPTH], depth_image).toImageMsg(*image_msg);
  image_msg->header.stamp = timestamp;
  image_msg->is_bigendian = false;
  image_msg->step = width * unit_step_size_[DEPTH];
  image_msg->header.frame_id = frame_id;

  depth_unaligned_publisher_->publish(std::move(image_msg));
}

void OBCameraNode::publishDepthPointCloud(const std::shared_ptr<ob::FrameSet> &frame_set) {
  if (!depth_cloud_pub_ || depth_cloud_pub_->get_subscription_count() == 0 ||
      !enable_point_cloud_) {
    return;
  }
  std::lock_guard<decltype(point_cloud_mutex_)> point_cloud_msg_lock(point_cloud_mutex_);
  auto depth_frame = frame_set->depthFrame();
  if (!depth_frame) {
    RCLCPP_ERROR_STREAM(logger_, "depth frame is null");
    return;
  }
  if (!pipeline_) {
    RCLCPP_ERROR_STREAM(logger_, "pipeline is null in publishDepthPointCloud");
    return;
  }
  auto camera_params = pipeline_->getCameraParam();
  if (!device_) {
    RCLCPP_ERROR_STREAM(logger_, "device is null in publishDepthPointCloud");
    return;
  }
  auto device_info = device_->getDeviceInfo();
  if (!device_info || !device_info.get()) {
    RCLCPP_ERROR_STREAM(logger_, "device_info is null in publishDepthPointCloud");
    return;
  }
  if (depth_registration_ || pid_ == DABAI_MAX_PID) {
    camera_params.depthIntrinsic = camera_params.rgbIntrinsic;
  }
  depth_point_cloud_filter_.setCameraParam(camera_params);
  float depth_scale = depth_frame->getValueScale();
  depth_point_cloud_filter_.setPositionDataScaled(depth_scale);
  depth_point_cloud_filter_.setCreatePointFormat(OB_FORMAT_POINT);
  depth_point_cloud_filter_.setDecimationFactor(point_cloud_decimation_filter_factor_);
  auto result_frame = depth_point_cloud_filter_.process(depth_frame);
  if (!result_frame) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to process depth frame");
    return;
  }
  auto point_size = result_frame->dataSize() / sizeof(OBPoint);
  auto *points = static_cast<OBPoint *>(result_frame->data());
  auto width = depth_frame->width() / point_cloud_decimation_filter_factor_;
  auto height = depth_frame->height() / point_cloud_decimation_filter_factor_;
  auto point_cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
  sensor_msgs::PointCloud2Modifier modifier(*point_cloud_msg);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(width * height);
  point_cloud_msg->width = width;
  point_cloud_msg->height = height;
  point_cloud_msg->row_step = point_cloud_msg->width * point_cloud_msg->point_step;
  point_cloud_msg->data.resize(point_cloud_msg->height * point_cloud_msg->row_step);
  sensor_msgs::PointCloud2Iterator<float> iter_x(*point_cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*point_cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*point_cloud_msg, "z");
  const static float MIN_DISTANCE = 20.0;     // 2cm
  const static float MAX_DISTANCE = 10000.0;  // 10m
  const static float min_depth = MIN_DISTANCE / depth_scale;
  const static float max_depth = MAX_DISTANCE / depth_scale;
  size_t valid_count = 0;
  for (size_t i = 0; i < point_size; i++) {
    bool valid_point = points[i].z >= min_depth && points[i].z <= max_depth;
    if (valid_point || ordered_pc_) {
      *iter_x = static_cast<float>(points[i].x / 1000.0);
      *iter_y = static_cast<float>(points[i].y / 1000.0);
      *iter_z = static_cast<float>(points[i].z / 1000.0);
      ++iter_x, ++iter_y, ++iter_z;
      valid_count++;
    }
  }
  if (valid_count == 0) {
    RCLCPP_WARN(logger_, "No valid point in point cloud");
    return;
  }
  if (!ordered_pc_) {
    point_cloud_msg->is_dense = true;
    point_cloud_msg->width = valid_count;
    point_cloud_msg->height = 1;
    modifier.resize(valid_count);
    point_cloud_msg->row_step = point_cloud_msg->width * point_cloud_msg->point_step;
  }
  auto frame_timestamp = getFrameTimestampUs(depth_frame);
  auto timestamp = fromUsToROSTime(frame_timestamp);
  std::string frame_id = depth_registration_ ? optical_frame_id_[COLOR] : optical_frame_id_[DEPTH];
  if (!cloud_frame_id_.empty()) {
    frame_id = cloud_frame_id_;
  }
  point_cloud_msg->header.stamp = timestamp;
  point_cloud_msg->header.frame_id = frame_id;
  if (save_point_cloud_) {
    save_point_cloud_ = false;
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    auto current_path = std::filesystem::current_path().string();
    std::string filename = current_path + "/point_cloud/points_" + ss.str() + ".ply";
    if (!std::filesystem::exists(current_path + "/point_cloud")) {
      std::filesystem::create_directory(current_path + "/point_cloud");
    }
    RCLCPP_INFO_STREAM(logger_, "Saving point cloud to " << filename);
    try {
      saveDepthPointsToPly(point_cloud_msg, filename);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to save point cloud: " << e.what());
    }
  }
  depth_cloud_pub_->publish(std::move(point_cloud_msg));
}

void OBCameraNode::publishColoredPointCloud(const std::shared_ptr<ob::FrameSet> &frame_set) {
  if (!depth_registration_cloud_pub_ ||
      depth_registration_cloud_pub_->get_subscription_count() == 0 ||
      !enable_colored_point_cloud_ || !frame_set) {
    return;
  }
  std::lock_guard<decltype(point_cloud_mutex_)> point_cloud_msg_lock(point_cloud_mutex_);

  auto depth_frame = frame_set->depthFrame();
  auto color_frame = frame_set->colorFrame();

  if (!depth_frame || !color_frame) {
    return;
  }

  auto depth_width = depth_frame->getWidth();
  auto depth_height = depth_frame->getHeight();
  auto color_width = color_frame->getWidth();
  auto color_height = color_frame->getHeight();
  if (depth_width != color_width || depth_height != color_height) {
    RCLCPP_DEBUG(logger_, "Depth (%d x %d) and color (%d x %d) frame size mismatch", depth_width,
                 depth_height, color_width, color_height);
    return;
  }

  if (!pipeline_) {
    RCLCPP_ERROR_STREAM(logger_, "pipeline is null in publishColoredPointCloud");
    return;
  }
  auto camera_params = pipeline_->getCameraParam();
  if (!device_) {
    RCLCPP_ERROR_STREAM(logger_, "device is null in publishColoredPointCloud");
    return;
  }
  auto device_info = device_->getDeviceInfo();
  if (!device_info || !device_info.get()) {
    RCLCPP_ERROR_STREAM(logger_, "device_info is null in publishColoredPointCloud");
    return;
  }
  if (depth_registration_ || pid_ == DABAI_MAX_PID) {
    camera_params.depthIntrinsic = camera_params.rgbIntrinsic;
  }

  color_point_cloud_filter_.setCameraParam(camera_params);
  auto depth_scale = depth_frame->getValueScale();
  color_point_cloud_filter_.setPositionDataScaled(depth_scale);
  color_point_cloud_filter_.setCreatePointFormat(OB_FORMAT_RGB_POINT);
  color_point_cloud_filter_.setDecimationFactor(point_cloud_decimation_filter_factor_);
  auto result_frame = color_point_cloud_filter_.process(frame_set);
  if (!result_frame) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to process depth frame");
    return;
  }
  auto point_size = result_frame->dataSize() / sizeof(OBColorPoint);
  auto *point_cloud = static_cast<OBColorPoint *>(result_frame->data());
  auto width = color_frame->getWidth() / point_cloud_decimation_filter_factor_;
  auto height = color_frame->getHeight() / point_cloud_decimation_filter_factor_;
  auto point_cloud_msg = std::make_unique<sensor_msgs::msg::PointCloud2>();
  sensor_msgs::PointCloud2Modifier modifier(*point_cloud_msg);
  modifier.setPointCloud2FieldsByString(1, "xyz");
  modifier.resize(width * height);
  point_cloud_msg->width = width;
  point_cloud_msg->height = height;
  std::string format_str = "rgb";
  point_cloud_msg->point_step =
      addPointField(*point_cloud_msg, format_str, 1, sensor_msgs::msg::PointField::FLOAT32,
                    static_cast<int>(point_cloud_msg->point_step));
  point_cloud_msg->row_step = point_cloud_msg->width * point_cloud_msg->point_step;
  point_cloud_msg->data.resize(point_cloud_msg->height * point_cloud_msg->row_step);
  sensor_msgs::PointCloud2Iterator<float> iter_x(*point_cloud_msg, "x");
  sensor_msgs::PointCloud2Iterator<float> iter_y(*point_cloud_msg, "y");
  sensor_msgs::PointCloud2Iterator<float> iter_z(*point_cloud_msg, "z");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*point_cloud_msg, "r");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*point_cloud_msg, "g");
  sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*point_cloud_msg, "b");
  size_t valid_count = 0;
  static const float MIN_DISTANCE = 20.0;
  static const float MAX_DISTANCE = 10000.0;
  static float min_depth = MIN_DISTANCE / depth_scale;
  static float max_depth = MAX_DISTANCE / depth_scale;
  for (size_t i = 0; i < point_size; i++) {
    bool valid_point = point_cloud[i].z >= min_depth && point_cloud[i].z <= max_depth;
    if (valid_point || ordered_pc_) {
      *iter_x = static_cast<float>(point_cloud[i].x / 1000.0);
      *iter_y = static_cast<float>(point_cloud[i].y / 1000.0);
      *iter_z = static_cast<float>(point_cloud[i].z / 1000.0);
      *iter_r = static_cast<uint8_t>(point_cloud[i].r);
      *iter_g = static_cast<uint8_t>(point_cloud[i].g);
      *iter_b = static_cast<uint8_t>(point_cloud[i].b);
      ++iter_x, ++iter_y, ++iter_z, ++iter_r, ++iter_g, ++iter_b;
      ++valid_count;
    }
  }

  if (valid_count == 0) {
    RCLCPP_WARN(logger_, "No valid points in point cloud");
    return;
  }
  if (!ordered_pc_) {
    point_cloud_msg->is_dense = true;
    point_cloud_msg->width = valid_count;
    point_cloud_msg->height = 1;
    modifier.resize(valid_count);
    point_cloud_msg->row_step = point_cloud_msg->width * point_cloud_msg->point_step;
  }
  auto frame_timestamp = getFrameTimestampUs(depth_frame);
  std::string frame_id = optical_frame_id_[COLOR];
  if (!cloud_frame_id_.empty()) {
    frame_id = cloud_frame_id_;
  }
  auto timestamp = fromUsToROSTime(frame_timestamp);
  point_cloud_msg->header.stamp = timestamp;
  point_cloud_msg->header.frame_id = frame_id;
  if (save_colored_point_cloud_) {
    save_colored_point_cloud_ = false;
    auto now = std::time(nullptr);
    std::stringstream ss;
    ss << std::put_time(std::localtime(&now), "%Y%m%d_%H%M%S");
    auto current_path = std::filesystem::current_path().string();
    std::string filename = current_path + "/point_cloud/colored_points_" + ss.str() + ".ply";
    if (!std::filesystem::exists(current_path + "/point_cloud")) {
      std::filesystem::create_directory(current_path + "/point_cloud");
    }
    RCLCPP_INFO_STREAM(logger_, "Saving point cloud to " << filename);
    try {
      saveRGBPointCloudMsgToPly(point_cloud_msg, filename);
    } catch (const std::exception &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to save point cloud: " << e.what());
    } catch (...) {
      RCLCPP_ERROR(logger_, "Failed to save point cloud");
    }
  }

  depth_registration_cloud_pub_->publish(std::move(point_cloud_msg));
}
std::shared_ptr<ob::Frame> OBCameraNode::processRightIrFrameFilter(
    std::shared_ptr<ob::Frame> &frame) {
  if (frame == nullptr || frame->getType() != OB_FRAME_IR_RIGHT) {
    return nullptr;
  }
  for (size_t i = 0; i < right_ir_filter_list_.size(); i++) {
    auto filter = right_ir_filter_list_[i];
    CHECK_NOTNULL(filter.get());
    if (filter->isEnabled() && frame != nullptr) {
      frame = filter->process(frame);
      if (frame == nullptr) {
        RCLCPP_ERROR_STREAM(logger_, "Right Ir filter process failed");
        break;
      }
    }
  }
  return frame;
}
std::shared_ptr<ob::Frame> OBCameraNode::processLeftIrFrameFilter(
    std::shared_ptr<ob::Frame> &frame) {
  if (frame == nullptr || frame->getType() != OB_FRAME_IR_LEFT) {
    return nullptr;
  }
  for (size_t i = 0; i < left_ir_filter_list_.size(); i++) {
    auto filter = left_ir_filter_list_[i];
    CHECK_NOTNULL(filter.get());
    if (filter->isEnabled() && frame != nullptr) {
      frame = filter->process(frame);
      if (frame == nullptr) {
        RCLCPP_ERROR_STREAM(logger_, "Left Ir filter process failed");
        break;
      }
    }
  }
  return frame;
}
std::shared_ptr<ob::Frame> OBCameraNode::processColorFrameFilter(
    std::shared_ptr<ob::Frame> &frame) {
  if (frame == nullptr) {
    return nullptr;
  }
  auto frame_type = frame->getType();
  if (frame_type == OB_FRAME_COLOR) {
    for (size_t i = 0; i < color_filter_list_.size(); i++) {
      auto filter = color_filter_list_[i];
      CHECK_NOTNULL(filter.get());
      if (filter->isEnabled() && frame != nullptr) {
        frame = filter->process(frame);
        if (frame == nullptr) {
          RCLCPP_ERROR_STREAM(logger_, "Color filter process failed");
          break;
        }
      }
    }
    return frame;
  } else if (frame_type == OB_FRAME_COLOR_LEFT) {
    for (size_t i = 0; i < left_color_filter_list_.size(); i++) {
      auto filter = left_color_filter_list_[i];
      CHECK_NOTNULL(filter.get());
      if (filter->isEnabled() && frame != nullptr) {
        frame = filter->process(frame);
        if (frame == nullptr) {
          RCLCPP_ERROR_STREAM(logger_, "Left color filter process failed");
          break;
        }
      }
    }
    return frame;
  } else if (frame_type == OB_FRAME_COLOR_RIGHT) {
    for (size_t i = 0; i < right_color_filter_list_.size(); i++) {
      auto filter = right_color_filter_list_[i];
      CHECK_NOTNULL(filter.get());
      if (filter->isEnabled() && frame != nullptr) {
        frame = filter->process(frame);
        if (frame == nullptr) {
          RCLCPP_ERROR_STREAM(logger_, "Right color filter process failed");
          break;
        }
      }
    }
    return frame;
  }
  return nullptr;
}
std::shared_ptr<ob::Frame> OBCameraNode::processDepthFrameFilter(
    std::shared_ptr<ob::Frame> &frame) {
  if (frame == nullptr || frame->getType() != OB_FRAME_DEPTH) {
    return nullptr;
  }
  std::lock_guard<std::mutex> depth_filter_lock(depth_filter_mutex_);
  for (size_t i = 0; i < depth_filter_list_.size(); i++) {
    auto filter = depth_filter_list_[i];
    CHECK_NOTNULL(filter.get());
    if (filter->isEnabled() && frame != nullptr) {
      frame = filter->process(frame);
      if (frame == nullptr) {
        RCLCPP_WARN_STREAM(logger_, "Depth filter process failed, frame is null");
        break;
      }
    }
  }
  return frame;
}
void OBCameraNode::setDisparitySearchOffset() {
  static bool has_run = false;
  auto config = OBDispOffsetConfig();
  if (has_run) {
    return;
  }
  if (device_->isPropertySupported(OB_PROP_DISP_SEARCH_OFFSET_INT, OB_PERMISSION_WRITE)) {
    if (disparity_search_offset_ >= 0 && disparity_search_offset_ <= 127) {
      device_->setIntProperty(OB_PROP_DISP_SEARCH_OFFSET_INT, disparity_search_offset_);
      RCLCPP_INFO_STREAM(logger_, "Set disparity search offset to " << disparity_search_offset_);
    }
    if (offset_index0_ >= 0 && offset_index0_ <= 127 && offset_index1_ >= 0 &&
        offset_index1_ <= 127) {
      config.enable = disparity_offset_config_;
      config.offset0 = offset_index0_;
      config.offset1 = offset_index1_;
      config.reserved = 0;

      device_->setStructuredData(OB_STRUCT_DISP_OFFSET_CONFIG,
                                 reinterpret_cast<const uint8_t *>(&config), sizeof(config));
      RCLCPP_INFO_STREAM(logger_, "disparity_offset_config: "
                                      << disparity_offset_config_ << "  offset_index0:"
                                      << offset_index0_ << "  offset_index1:" << offset_index1_);
    }
  }
  has_run = true;
}

void OBCameraNode::setDepthAutoExposureROI() {
  static bool depth_roi_has_run = false;
  if (depth_roi_has_run) {
    return;
  }
  if (isGemini305SeriesPID(pid_) && ae_reference_stream_ == "color") {
    RCLCPP_WARN_STREAM(logger_, "Skip setting depth AE ROI because AE Reference Stream is color");
    depth_roi_has_run = true;
    return;
  }
  if (device_->isPropertySupported(OB_STRUCT_DEPTH_AE_ROI, OB_PERMISSION_READ_WRITE)) {
    auto config = OBRegionOfInterest();
    uint32_t data_size = sizeof(config);
    device_->getStructuredData(OB_STRUCT_DEPTH_AE_ROI, reinterpret_cast<uint8_t *>(&config),
                               &data_size);
    if (depth_ae_roi_left_ != -1) {
      config.x0_left = (depth_ae_roi_left_ < 0) ? 0 : depth_ae_roi_left_;
      config.x0_left =
          (depth_ae_roi_left_ > width_[DEPTH] - 1) ? width_[DEPTH] - 1 : config.x0_left;
    }
    if (depth_ae_roi_top_ != -1) {
      config.y0_top = (depth_ae_roi_top_ < 0) ? 0 : depth_ae_roi_top_;
      config.y0_top = (depth_ae_roi_top_ > height_[DEPTH] - 1) ? height_[DEPTH] - 1 : config.y0_top;
    }
    if (depth_ae_roi_right_ != -1) {
      config.x1_right = (depth_ae_roi_right_ < 0) ? 0 : depth_ae_roi_right_;
      config.x1_right =
          (depth_ae_roi_right_ > width_[DEPTH] - 1) ? width_[DEPTH] - 1 : config.x1_right;
    }
    if (depth_ae_roi_bottom_ != -1) {
      config.y1_bottom = (depth_ae_roi_bottom_ < 0) ? 0 : depth_ae_roi_bottom_;
      config.y1_bottom =
          (depth_ae_roi_bottom_ > height_[DEPTH] - 1) ? height_[DEPTH] - 1 : config.y1_bottom;
    }
    device_->setStructuredData(OB_STRUCT_DEPTH_AE_ROI, reinterpret_cast<const uint8_t *>(&config),
                               sizeof(config));
    device_->getStructuredData(OB_STRUCT_DEPTH_AE_ROI, reinterpret_cast<uint8_t *>(&config),
                               &data_size);
    RCLCPP_INFO_STREAM(logger_, "Set depth AE ROI to " << config.x0_left << ", " << config.x1_right
                                                       << ", " << config.y0_top << ", "
                                                       << config.y1_bottom);
  }
  depth_roi_has_run = true;
}

void OBCameraNode::setColorAutoExposureROI() {
  static bool color_roi_has_run = false;
  if (color_roi_has_run) {
    return;
  }
  if (isGemini305SeriesPID(pid_) && ae_reference_stream_ == "depth") {
    RCLCPP_WARN_STREAM(logger_, "Skip setting color AE ROI because AE Reference Stream is depth");
    color_roi_has_run = true;
    return;
  }
  if (device_->isPropertySupported(OB_STRUCT_COLOR_AE_ROI, OB_PERMISSION_READ_WRITE)) {
    auto config = OBRegionOfInterest();
    uint32_t data_size = sizeof(config);
    device_->getStructuredData(OB_STRUCT_COLOR_AE_ROI, reinterpret_cast<uint8_t *>(&config),
                               &data_size);
    if (color_ae_roi_left_ != -1) {
      config.x0_left = (color_ae_roi_left_ < 0) ? 0 : color_ae_roi_left_;
      config.x0_left =
          (color_ae_roi_left_ > width_[COLOR] - 1) ? width_[COLOR] - 1 : config.x0_left;
    }
    if (color_ae_roi_top_ != -1) {
      config.y0_top = (color_ae_roi_top_ < 0) ? 0 : color_ae_roi_top_;
      config.y0_top = (color_ae_roi_top_ > height_[COLOR] - 1) ? height_[COLOR] - 1 : config.y0_top;
    }
    if (color_ae_roi_right_ != -1) {
      config.x1_right = (color_ae_roi_right_ < 0) ? 0 : color_ae_roi_right_;
      config.x1_right =
          (color_ae_roi_right_ > width_[COLOR] - 1) ? width_[COLOR] - 1 : config.x1_right;
    }
    if (color_ae_roi_bottom_ != -1) {
      config.y1_bottom = (color_ae_roi_bottom_ < 0) ? 0 : color_ae_roi_bottom_;
      config.y1_bottom =
          (color_ae_roi_bottom_ > height_[COLOR] - 1) ? height_[COLOR] - 1 : config.y1_bottom;
    }
    device_->setStructuredData(OB_STRUCT_COLOR_AE_ROI, reinterpret_cast<const uint8_t *>(&config),
                               sizeof(config));
    device_->getStructuredData(OB_STRUCT_COLOR_AE_ROI, reinterpret_cast<uint8_t *>(&config),
                               &data_size);
    RCLCPP_INFO_STREAM(logger_, "Set color AE ROI to " << config.x0_left << ", " << config.x1_right
                                                       << ", " << config.y0_top << ", "
                                                       << config.y1_bottom);
  }
  color_roi_has_run = true;
}

uint64_t OBCameraNode::getFrameTimestampUs(const std::shared_ptr<ob::Frame> &frame) {
  if (frame == nullptr) {
    RCLCPP_WARN(logger_, "getFrameTimestampUs: frame is nullptr, return 0");
    return 0;
  }
  if (time_domain_ == "device") {
    return frame->getTimeStampUs();
  } else if (time_domain_ == "global") {
    return frame->getGlobalTimeStampUs();
  } else {
    return frame->getSystemTimeStampUs();
  }
}
void OBCameraNode::onNewFrameSetCallback(std::shared_ptr<ob::FrameSet> frame_set) {
  if (!is_running_.load()) {
    return;
  }
  if (!is_camera_node_initialized_.load()) {
    return;
  }
  if (frame_set == nullptr) {
    return;
  }
  const auto frame_set_arrival_system_us = getSystemNowUs();
  const auto frame_set_arrival_steady_us = getSteadyNowUs();
  try {
    if (!tf_published_) {
      publishStaticTransforms();
      tf_published_ = true;
    }
    auto device_info = device_->getDeviceInfo();
    CHECK_NOTNULL(device_info);
    auto depth_frame = frame_set->getFrame(OB_FRAME_DEPTH);
    auto color_frame = frame_set->getFrame(OB_FRAME_COLOR);
    auto left_ir_frame = frame_set->getFrame(OB_FRAME_IR_LEFT);
    auto right_ir_frame = frame_set->getFrame(OB_FRAME_IR_RIGHT);
    auto left_color_frame = frame_set->getFrame(OB_FRAME_COLOR_LEFT);
    auto right_color_frame = frame_set->getFrame(OB_FRAME_COLOR_RIGHT);
    auto ir_frame = frame_set->getFrame(OB_FRAME_IR);
    if (depth_frame) {
      setDisparitySearchOffset();
      setDepthAutoExposureROI();
      depth_frame = processDepthFrameFilter(depth_frame);
      if (depth_frame) {
        frame_set->pushFrame(depth_frame);
        fps_counter_depth_->tick();
      }
    }
    if (color_frame) {
      setColorAutoExposureROI();
      color_frame = processColorFrameFilter(color_frame);
      frame_set->pushFrame(color_frame);
      fps_counter_color_->tick();
    }
    if (left_color_frame) {
      setColorAutoExposureROI();
      left_color_frame = processColorFrameFilter(left_color_frame);
      frame_set->pushFrame(left_color_frame);
    }
    if (right_color_frame) {
      right_color_frame = processColorFrameFilter(right_color_frame);
      frame_set->pushFrame(right_color_frame);
    }
    if (left_ir_frame) {
      left_ir_frame = processLeftIrFrameFilter(left_ir_frame);
      frame_set->pushFrame(left_ir_frame);
      fps_counter_left_ir_->tick();
    }
    if (right_ir_frame) {
      right_ir_frame = processRightIrFrameFilter(right_ir_frame);
      frame_set->pushFrame(right_ir_frame);
      fps_counter_right_ir_->tick();
    }
    if (depth_registration_ && align_filter_ && depth_frame) {
      publishRawDepthImage(depth_frame);
      if (auto new_frame = align_filter_->process(frame_set)) {
        auto new_frame_set = new_frame->as<ob::FrameSet>();
        CHECK_NOTNULL(new_frame_set.get());
        frame_set = new_frame_set;
      } else {
        RCLCPP_ERROR(logger_, "Failed to align depth frame to color frame");
        return;
      }
    } else {
      RCLCPP_DEBUG_ONCE(logger_,
                        "Depth registration is disabled or align filter is null or depth frame is "
                        "null or color frame is null");
    }

    auto final_color_frame = frame_set->getFrame(OB_FRAME_COLOR);
    auto final_depth_frame = frame_set->getFrame(OB_FRAME_DEPTH);
    if (frame_timestamp_csv_logger_ && frame_timestamp_csv_logger_->enabled()) {
      const bool track_color = enable_stream_[COLOR] && static_cast<bool>(final_color_frame);
      const bool track_depth = enable_stream_[DEPTH] && static_cast<bool>(final_depth_frame);
      const bool color_publish_expected = track_color;
      const bool depth_publish_expected = track_depth;
      frame_timestamp_csv_logger_->recordFrameSet(
          final_color_frame, final_depth_frame, frame_set_arrival_system_us,
          frame_set_arrival_steady_us, track_color, track_depth, color_publish_expected,
          depth_publish_expected);
    }

    // Refresh frame from current frameset before logging to reflect post-filter/alignment output.
    for (const auto &stream_index : IMAGE_STREAMS) {
      if (!enable_stream_[stream_index]) {
        continue;
      }
      auto frame_type = STREAM_TYPE_TO_FRAME_TYPE.at(stream_index.first);
      auto updated_frame = frame_set->getFrame(frame_type);
      if (!updated_frame || !updated_frame->is<ob::VideoFrame>()) {
        continue;
      }
      auto updated_video = updated_frame->as<ob::VideoFrame>();

      // For D2C, avoid logging an early unaligned depth frame before align target is available.
      if (stream_index == DEPTH && depth_registration_) {
        auto align_target_frame_type = STREAM_TYPE_TO_FRAME_TYPE.at(align_target_stream_);
        auto align_target_frame = frame_set->getFrame(align_target_frame_type);
        if (!align_target_frame || !align_target_frame->is<ob::VideoFrame>()) {
          continue;
        }
        auto target_video = align_target_frame->as<ob::VideoFrame>();
        if (updated_video->getWidth() != target_video->getWidth() ||
            updated_video->getHeight() != target_video->getHeight()) {
          continue;
        }
      }

      logFrameInfoOnce(stream_index, updated_video);
    }

    if (enable_stream_[COLOR] && color_frame) {
      std::unique_lock<std::mutex> lock(color_frame_queue_lock_);
      color_frame_queue_.push(frame_set);
      color_frame_queue_cv_.notify_all();
    } else {
      publishPointCloud(frame_set);
    }

    if (enable_stream_[COLOR_LEFT] && left_color_frame) {
      std::unique_lock<std::mutex> lock(left_color_frame_queue_lock_);
      left_color_frame_queue_.push(frame_set);
      left_color_frame_queue_cv_.notify_all();
    }
    if (enable_stream_[COLOR_RIGHT] && right_color_frame) {
      std::unique_lock<std::mutex> lock(right_color_frame_queue_lock_);
      right_color_frame_queue_.push(frame_set);
      right_color_frame_queue_cv_.notify_all();
    }

    for (const auto &stream_index : IMAGE_STREAMS) {
      if (enable_stream_[stream_index]) {
        auto frame_type = STREAM_TYPE_TO_FRAME_TYPE.at(stream_index.first);
        if (frame_type == OB_FRAME_COLOR || frame_type == OB_FRAME_COLOR_LEFT ||
            frame_type == OB_FRAME_COLOR_RIGHT) {
          continue;
        }

        auto frame = frame_set->getFrame(frame_type);
        if (frame == nullptr) {
          continue;
        }
        onNewFrameCallback(frame, stream_index);
      }
    }
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(
        logger_, "onNewFrameSetCallback error: " << orbbec_camera::formatObErrorWithStatus(e));
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(logger_, "onNewFrameSetCallback error: " << e.what());
  } catch (...) {
    RCLCPP_ERROR_STREAM(logger_, "onNewFrameSetCallback error: unknown error");
  }
}

void OBCameraNode::logFrameInfoOnce(const stream_index_pair &stream_index,
                                    const std::shared_ptr<ob::VideoFrame> &video_frame) {
  if (!video_frame) {
    return;
  }

  {
    std::lock_guard<std::mutex> lock(frame_info_logged_mutex_);
    auto iter = frame_info_logged_.find(stream_index);
    if (iter != frame_info_logged_.end() && iter->second) {
      return;
    }
    frame_info_logged_[stream_index] = true;
  }

  RCLCPP_INFO_STREAM(logger_, stream_name_[stream_index]
                                  << " Frame - Width: " << video_frame->getWidth() << " Height: "
                                  << video_frame->getHeight() << " fps: " << fps_[stream_index]
                                  << " Format: " << video_frame->getFormat());
}

void OBCameraNode::onNewColorFrameCallback() {
  while (enable_stream_[COLOR] && rclcpp::ok() && is_running_.load()) {
    std::unique_lock<std::mutex> lock(color_frame_queue_lock_);
    color_frame_queue_cv_.wait(
        lock, [this]() { return !color_frame_queue_.empty() || !(is_running_.load()); });

    if (!rclcpp::ok() || !is_running_.load()) {
      break;
    }
    std::shared_ptr<ob::FrameSet> frameSet = color_frame_queue_.front();
    is_color_frame_decoded_ = decodeColorFrameToBuffer(frameSet->colorFrame(), rgb_buffer_);
    onNewFrameCallback(frameSet->colorFrame(), COLOR);
    publishPointCloud(frameSet);
    color_frame_queue_.pop();
  }

  RCLCPP_DEBUG_STREAM(logger_, "Color frame thread exited");
}

void OBCameraNode::onNewLeftColorFrameCallback() {
  while (enable_stream_[COLOR_LEFT] && rclcpp::ok() && is_running_.load()) {
    std::unique_lock<std::mutex> lock(left_color_frame_queue_lock_);
    left_color_frame_queue_cv_.wait(
        lock, [this]() { return !left_color_frame_queue_.empty() || !(is_running_.load()); });

    if (!rclcpp::ok() || !is_running_.load()) {
      break;
    }
    std::shared_ptr<ob::FrameSet> frameSet = left_color_frame_queue_.front();
    is_left_color_frame_decoded_ =
        decodeColorFrameToBuffer(frameSet->getFrame(OB_FRAME_COLOR_LEFT), rgb_buffer_left_);
    onNewFrameCallback(frameSet->getFrame(OB_FRAME_COLOR_LEFT), COLOR_LEFT);
    left_color_frame_queue_.pop();
  }
  RCLCPP_DEBUG_STREAM(logger_, "Left color frame thread exited");
}

void OBCameraNode::onNewRightColorFrameCallback() {
  while (enable_stream_[COLOR_RIGHT] && rclcpp::ok() && is_running_.load()) {
    std::unique_lock<std::mutex> lock(right_color_frame_queue_lock_);
    right_color_frame_queue_cv_.wait(
        lock, [this]() { return !right_color_frame_queue_.empty() || !(is_running_.load()); });

    if (!rclcpp::ok() || !is_running_.load()) {
      break;
    }
    std::shared_ptr<ob::FrameSet> frameSet = right_color_frame_queue_.front();
    is_right_color_frame_decoded_ =
        decodeColorFrameToBuffer(frameSet->getFrame(OB_FRAME_COLOR_RIGHT), rgb_buffer_right_);
    onNewFrameCallback(frameSet->getFrame(OB_FRAME_COLOR_RIGHT), COLOR_RIGHT);
    right_color_frame_queue_.pop();
  }
  RCLCPP_DEBUG_STREAM(logger_, "Right color frame thread exited");
}

std::shared_ptr<ob::Frame> OBCameraNode::softwareDecodeColorFrame(
    const std::shared_ptr<ob::Frame> &frame, const stream_index_pair &stream_index) {
  if (frame == nullptr) {
    return nullptr;
  }
  if (frame->getFormat() == OB_FORMAT_RGB || frame->getFormat() == OB_FORMAT_BGR) {
    return frame;
  }
  if (frame->getFormat() == OB_FORMAT_RGBA || frame->getFormat() == OB_FORMAT_BGRA) {
    return frame;
  }
  if (frame->getFormat() == OB_FORMAT_Y16 || frame->getFormat() == OB_FORMAT_Y8) {
    return frame;
  }

  ob::FormatConvertFilter *filter = &format_convert_filter_;
  if (stream_index == COLOR_LEFT) {
    filter = &format_convert_filter_left_;
  } else if (stream_index == COLOR_RIGHT) {
    filter = &format_convert_filter_right_;
  }

  if (!setupFormatConvertType(frame->getFormat(), *filter)) {
    RCLCPP_ERROR(logger_, "Unsupported color format: %d", frame->getFormat());
    return nullptr;
  }

  std::shared_ptr<ob::Frame> color_frame;
  try {
    color_frame = filter->process(frame);
  } catch (const ob::Error &e) {
    RCLCPP_ERROR_STREAM(logger_,
                        "Format convert failed: " << orbbec_camera::formatObErrorWithStatus(e));
    return nullptr;
  } catch (const std::exception &e) {
    RCLCPP_ERROR_STREAM(logger_, "Format convert failed: " << e.what());
    return nullptr;
  } catch (...) {
    RCLCPP_ERROR(logger_, "Format convert failed: unknown error");
    return nullptr;
  }
  if (color_frame == nullptr) {
    RCLCPP_ERROR_SKIPFIRST_THROTTLE(logger_, *(node_->get_clock()), 1000,
                                    "Failed to convert frame to RGB format");
    return nullptr;
  }
  return color_frame;
}

bool OBCameraNode::decodeColorFrameToBuffer(const std::shared_ptr<ob::Frame> &frame,
                                            uint8_t *buffer) {
  if (frame == nullptr) {
    return false;
  }
  if (!buffer) {
    return false;
  }

  stream_index_pair stream_index = COLOR;
  switch (frame->getType()) {
    case OB_FRAME_COLOR:
      stream_index = COLOR;
      break;
    case OB_FRAME_COLOR_LEFT:
      stream_index = COLOR_LEFT;
      break;
    case OB_FRAME_COLOR_RIGHT:
      stream_index = COLOR_RIGHT;
      break;
    default:
      stream_index = COLOR;
      break;
  }

  bool has_subscriber = false;
  if (image_publishers_.count(stream_index) && image_publishers_[stream_index]) {
    has_subscriber = image_publishers_[stream_index]->get_subscription_count() > 0;
  }
  if (stream_index == COLOR && enable_color_undistortion_ && color_undistortion_publisher_) {
    has_subscriber = true;
  }

  if (frame->getType() == OB_FRAME_COLOR && enable_colored_point_cloud_ &&
      depth_registration_cloud_pub_ &&
      depth_registration_cloud_pub_->get_subscription_count() > 0) {
    has_subscriber = true;
  }

  if (metadata_publishers_.count(stream_index) && metadata_publishers_[stream_index] &&
      metadata_publishers_[stream_index]->get_subscription_count() > 0) {
    has_subscriber = true;
  }
  if (camera_info_publishers_.count(stream_index) && camera_info_publishers_[stream_index] &&
      camera_info_publishers_[stream_index]->get_subscription_count() > 0) {
    has_subscriber = true;
  }

  if (!has_subscriber) {
    return false;
  }

  std::shared_ptr<JPEGDecoder> decoder;
  if (stream_index == COLOR_LEFT) {
    decoder = jpeg_decoder_left_;
  } else if (stream_index == COLOR_RIGHT) {
    decoder = jpeg_decoder_right_;
  } else {
    decoder = jpeg_decoder_;
  }
  bool is_decoded = false;
  if (!frame) {
    return false;
  }

#if defined(USE_RK_HW_DECODER) || defined(USE_NV_HW_DECODER)
  if (frame && frame->getFormat() != OB_FORMAT_RGB888) {
    if (frame->getFormat() == OB_FORMAT_MJPG && decoder) {
      CHECK_NOTNULL(decoder.get());
      CHECK_NOTNULL(buffer);
      auto video_frame = frame->as<ob::ColorFrame>();
      bool ret = false;
      if (video_frame && width_.count(stream_index) && height_.count(stream_index) &&
          static_cast<int>(video_frame->getWidth()) == width_[stream_index] &&
          static_cast<int>(video_frame->getHeight()) == height_[stream_index]) {
        ret = decoder->decode(video_frame, buffer);
      }
      if (!ret) {
        RCLCPP_ERROR_STREAM(logger_, "Decode frame failed");
        is_decoded = false;

      } else {
        is_decoded = true;
      }
    }
  }
#endif
  if (!is_decoded) {
    auto video_frame = softwareDecodeColorFrame(frame, stream_index);
    if (!video_frame) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to convert frame to video frame");
      return false;
    }
    CHECK_NOTNULL(buffer);
    memcpy(buffer, video_frame->getData(), video_frame->getDataSize());
    return true;
  }
  return true;
}

std::shared_ptr<ob::Frame> OBCameraNode::decodeIRMJPGFrame(
    const std::shared_ptr<ob::Frame> &frame) {
  if (frame == nullptr) {
    return nullptr;
  }
  if (frame->getFormat() == OB_FORMAT_MJPEG &&
      (frame->getType() == OB_FRAME_IR || frame->getType() == OB_FRAME_IR_LEFT ||
       frame->getType() == OB_FRAME_IR_RIGHT)) {
    auto video_frame = frame->as<ob::IRFrame>();

    cv::Mat mjpgMat(1, video_frame->getDataSize(), CV_8UC1, video_frame->getData());
    cv::Mat irRawMat = cv::imdecode(mjpgMat, cv::IMREAD_GRAYSCALE);

    std::shared_ptr<ob::Frame> irFrame =
        ob::FrameFactory::createVideoFrame(video_frame->getType(), video_frame->getFormat(),
                                           video_frame->getWidth(), video_frame->getHeight(), 0);

    uint32_t buffer_size = irRawMat.rows * irRawMat.cols * irRawMat.channels();

    if (buffer_size > irFrame->getDataSize()) {
      RCLCPP_ERROR_STREAM(logger_,
                          "Insufficient buffer size allocation,failed to decode ir mjpg frame!");
      return nullptr;
    }

    memcpy(irFrame->getData(), irRawMat.data, buffer_size);
    ob::FrameHelper::setFrameDeviceTimestamp(irFrame, video_frame->getTimeStampUs());
    ob::FrameHelper::setFrameDeviceTimestampUs(irFrame, video_frame->getTimeStampUs());
    ob::FrameHelper::setFrameSystemTimestamp(irFrame, video_frame->getSystemTimeStampUs());
    return irFrame;
  }

  return frame;
}

void OBCameraNode::onNewFrameCallback(const std::shared_ptr<ob::Frame> &frame,
                                      const stream_index_pair &stream_index) {
  if (frame == nullptr) {
    return;
  }
  CHECK_NOTNULL(image_publishers_[stream_index]);
  const bool has_raw_image_subscriber =
      image_publishers_[stream_index]->get_subscription_count() > 0;
  const bool enable_undistortion_publish =
      (stream_index == COLOR && enable_color_undistortion_ && color_undistortion_publisher_);
  bool has_subscriber = has_raw_image_subscriber || enable_undistortion_publish;
  has_subscriber =
      has_subscriber || camera_info_publishers_[stream_index]->get_subscription_count() > 0;
  has_subscriber =
      has_subscriber || (metadata_publishers_.count(stream_index) &&
                         metadata_publishers_[stream_index]->get_subscription_count() > 0);
  if (!has_subscriber) {
    return;
  }
  std::shared_ptr<ob::VideoFrame> video_frame;
  if (frame->getType() == OB_FRAME_COLOR || frame->getType() == OB_FRAME_COLOR_LEFT ||
      frame->getType() == OB_FRAME_COLOR_RIGHT) {
    video_frame = frame->as<ob::ColorFrame>();
  } else if (frame->getType() == OB_FRAME_DEPTH) {
    video_frame = frame->as<ob::DepthFrame>();
  } else if (frame->getType() == OB_FRAME_IR || frame->getType() == OB_FRAME_IR_LEFT ||
             frame->getType() == OB_FRAME_IR_RIGHT) {
    video_frame = frame->as<ob::IRFrame>();

    // interleave filter speckle or flood ir
    if (interleave_frame_enable_ && interleave_skip_enable_) {
      RCLCPP_DEBUG(logger_, "interleave filter skip interleave_skip_index_: %d",
                   interleave_skip_index_);
      if (video_frame->getMetadataValue(OB_FRAME_METADATA_TYPE_HDR_SEQUENCE_INDEX) ==
          interleave_skip_index_) {
        RCLCPP_DEBUG(logger_, "interleave filter skip frame type: %d", frame->getType());
        return;
      }
    }
  } else {
    RCLCPP_ERROR(logger_, "Unsupported frame type: %d", frame->getType());
    return;
  }
  if (!video_frame) {
    RCLCPP_ERROR(logger_, "Failed to convert frame to video frame");
    return;
  }
  int width = static_cast<int>(video_frame->getWidth());
  int height = static_cast<int>(video_frame->getHeight());
  auto frame_timestamp = getFrameTimestampUs(frame);
  auto timestamp = fromUsToROSTime(frame_timestamp);
  if (!device_) {
    RCLCPP_ERROR_STREAM(logger_, "device is null in onNewFrameCallback");
    return;
  }
  auto device_info = device_->getDeviceInfo();
  if (!device_info || !device_info.get()) {
    RCLCPP_ERROR_STREAM(logger_, "device_info is null in onNewFrameCallback");
    return;
  }
  OBCameraIntrinsic intrinsic;
  OBCameraDistortion distortion;
  auto stream_profile = frame->getStreamProfile();
  CHECK_NOTNULL(stream_profile);
  auto video_stream_profile = stream_profile->as<ob::VideoStreamProfile>();
  CHECK_NOTNULL(video_stream_profile);
  intrinsic = video_stream_profile->getIntrinsic();
  distortion = video_stream_profile->getDistortion();
  if (pid_ == DABAI_MAX_PID) {
    auto camera_params = pipeline_->getCameraParam();
    // use color param
    intrinsic = camera_params.rgbIntrinsic;
    distortion = camera_params.rgbDistortion;
  }
  std::string frame_id = optical_frame_id_[stream_index];
  if (depth_registration_ && stream_index == DEPTH) {
    frame_id = depth_aligned_frame_id_[stream_index];
  }
  sensor_msgs::msg::CameraInfo camera_info{};
  if (color_info_manager_ && color_info_manager_->isCalibrated() && stream_index == COLOR) {
    camera_info = color_info_manager_->getCameraInfo();
    camera_info.header.stamp = timestamp;
    camera_info.header.frame_id = frame_id;
    camera_info.width = width;
    camera_info.height = height;
  } else if (ir_info_manager_ && ir_info_manager_->isCalibrated() &&
             (stream_index == INFRA1 || stream_index == INFRA2 || stream_index == DEPTH)) {
    camera_info = ir_info_manager_->getCameraInfo();
    camera_info.header.stamp = timestamp;
    camera_info.header.frame_id = frame_id;
    camera_info.width = width;
    camera_info.height = height;
  } else {
    camera_info = convertToCameraInfo(intrinsic, distortion, width);
    camera_info.header.stamp = timestamp;
    camera_info.header.frame_id = frame_id;
    camera_info.width = width;
    camera_info.height = height;
  }
  auto &image = images_[stream_index];
  if (frame->getType() == OB_FRAME_IR_RIGHT && enable_stream_[INFRA1]) {
    auto stream_profile = frame->getStreamProfile();
    CHECK_NOTNULL(stream_profile);
    auto video_stream_profile = stream_profile->as<ob::VideoStreamProfile>();
    CHECK_NOTNULL(video_stream_profile);
    auto left_video_profile = stream_profile_[INFRA1]->as<ob::VideoStreamProfile>();
    CHECK_NOTNULL(left_video_profile);
    auto ex = video_stream_profile->getExtrinsicTo(left_video_profile);
    float fx = camera_info.k.at(0);
    float fy = camera_info.k.at(4);
    camera_info.p.at(3) = -fx * ex.trans[0] / 1000.0 + 0.0;
    camera_info.p.at(7) = -fy * ex.trans[1] / 1000.0 + 0.0;
  }
  CHECK_NOTNULL(image_publishers_[stream_index]);
  if (image.empty() || image.cols != width || image.rows != height) {
    image.create(height, width, image_format_[stream_index]);
  }
  if (frame->getType() == OB_FRAME_COLOR && !is_color_frame_decoded_) {
    RCLCPP_ERROR(logger_, "color frame is not decoded");
    return;
  }
  if (frame->getType() == OB_FRAME_COLOR_LEFT && !is_left_color_frame_decoded_) {
    RCLCPP_ERROR(logger_, "left color frame is not decoded");
    return;
  }
  if (frame->getType() == OB_FRAME_COLOR_RIGHT && !is_right_color_frame_decoded_) {
    RCLCPP_ERROR(logger_, "right color frame is not decoded");
    return;
  }
  if (frame->getType() == OB_FRAME_COLOR && frame->format() != OB_FORMAT_Y8 &&
      frame->format() != OB_FORMAT_Y16 && frame->format() != OB_FORMAT_BGRA &&
      frame->format() != OB_FORMAT_RGBA && has_subscriber) {
    memcpy(image.data, rgb_buffer_, video_frame->getWidth() * video_frame->getHeight() * 3);
  } else if (frame->getType() == OB_FRAME_COLOR_LEFT && frame->format() != OB_FORMAT_Y8 &&
             frame->format() != OB_FORMAT_Y16 && frame->format() != OB_FORMAT_BGRA &&
             frame->format() != OB_FORMAT_RGBA && has_subscriber) {
    memcpy(image.data, rgb_buffer_left_, video_frame->getWidth() * video_frame->getHeight() * 3);
  } else if (frame->getType() == OB_FRAME_COLOR_RIGHT && frame->format() != OB_FORMAT_Y8 &&
             frame->format() != OB_FORMAT_Y16 && frame->format() != OB_FORMAT_BGRA &&
             frame->format() != OB_FORMAT_RGBA && has_subscriber) {
    memcpy(image.data, rgb_buffer_right_, video_frame->getWidth() * video_frame->getHeight() * 3);
  } else {
    memcpy(image.data, video_frame->getData(), video_frame->getDataSize());
  }

  if (enable_undistortion_publish) {
    auto undistort_result = undistortImage(image, intrinsic, distortion);
    sensor_msgs::msg::Image::UniquePtr undistorted_image_msg(new sensor_msgs::msg::Image());
    cv_bridge::CvImage(std_msgs::msg::Header(), encoding_[stream_index], undistort_result.image)
        .toImageMsg(*undistorted_image_msg);
    CHECK_NOTNULL(undistorted_image_msg.get());
    undistorted_image_msg->header.stamp = timestamp;
    undistorted_image_msg->is_bigendian = false;
    undistorted_image_msg->step = width * unit_step_size_[stream_index];
    undistorted_image_msg->header.frame_id = frame_id;
    color_undistortion_publisher_->publish(std::move(undistorted_image_msg));
    // Update intrinsic with the new camera matrix from undistortion
    camera_info.p.at(0) = undistort_result.new_intrinsic.fx;
    camera_info.p.at(5) = undistort_result.new_intrinsic.fy;
    camera_info.p.at(2) = undistort_result.new_intrinsic.cx;
    camera_info.p.at(6) = undistort_result.new_intrinsic.cy;
  }

  CHECK(camera_info_publishers_.count(stream_index) > 0);
  camera_info_publishers_[stream_index]->publish(camera_info);
  publishMetadata(frame, stream_index, camera_info.header);

  if (stream_index == DEPTH) {
    auto depth_scale = video_frame->as<ob::DepthFrame>()->getValueScale();
    image = image * depth_scale;
  }
  sensor_msgs::msg::Image::UniquePtr image_msg(new sensor_msgs::msg::Image());

  cv_bridge::CvImage(std_msgs::msg::Header(), encoding_[stream_index], image)
      .toImageMsg(*image_msg);
  CHECK_NOTNULL(image_msg.get());
  image_msg->header.stamp = timestamp;
  image_msg->is_bigendian = false;
  image_msg->step = width * unit_step_size_[stream_index];
  image_msg->header.frame_id = frame_id;
  CHECK(image_publishers_.count(stream_index) > 0);
  saveImageToFile(stream_index, image, *image_msg);
  if (stream_index == COLOR) {
    fps_delay_status_color_->tick(frame_timestamp);
  } else if (stream_index == DEPTH) {
    fps_delay_status_depth_->tick(frame_timestamp);
  }
  if (has_raw_image_subscriber) {
    if (frame_timestamp_csv_logger_ && frame_timestamp_csv_logger_->enabled() &&
        (stream_index == COLOR || stream_index == DEPTH)) {
      frame_timestamp_csv_logger_->recordPreImagePublish(stream_index.first, frame,
                                                         getSystemNowUs(), getSteadyNowUs());
    }
    image_publishers_[stream_index]->publish(std::move(image_msg));
  }
}

void OBCameraNode::publishMetadata(const std::shared_ptr<ob::Frame> &frame,
                                   const stream_index_pair &stream_index,
                                   const std_msgs::msg::Header &header) {
  if (metadata_publishers_.count(stream_index) == 0) {
    return;
  }
  auto metadata_publisher = metadata_publishers_[stream_index];
  if (metadata_publisher->get_subscription_count() == 0) {
    return;
  }
  orbbec_camera_msgs::msg::Metadata metadata_msg;
  metadata_msg.header = header;
  nlohmann::json json_data;

  for (int i = 0; i < OB_FRAME_METADATA_TYPE_COUNT; i++) {
    auto meta_data_type = static_cast<OBFrameMetadataType>(i);
    std::string field_name = metaDataTypeToString(meta_data_type);
    if (!frame->hasMetadata(meta_data_type)) {
      continue;
    }
    int64_t value = frame->getMetadataValue(meta_data_type);
    json_data[field_name] = value;
  }
  metadata_msg.json_data = json_data.dump(2);
  metadata_publisher->publish(metadata_msg);
}

void OBCameraNode::saveImageToFile(const stream_index_pair &stream_index, const cv::Mat &image,
                                   const sensor_msgs::msg::Image &image_msg) {
  if (save_images_[stream_index]) {
    auto now = std::chrono::system_clock::now();
    auto in_time_t = std::chrono::system_clock::to_time_t(now);
    auto us =
        std::chrono::duration_cast<std::chrono::microseconds>(now.time_since_epoch()) % 1000000;

    std::stringstream ss;
    ss << std::put_time(std::localtime(&in_time_t), "%Y%m%d_%H%M%S");
    ss << "_" << std::setw(6) << std::setfill('0') << us.count();
    auto current_path = std::filesystem::current_path().string();
    auto fps = fps_[stream_index];
    int index = save_images_count_[stream_index];
    std::string file_suffix = stream_index == COLOR ? ".png" : ".raw";
    std::string filename = current_path + "/image/" + stream_name_[stream_index] + "_" +
                           std::to_string(image_msg.width) + "x" +
                           std::to_string(image_msg.height) + "_" + std::to_string(fps) + "hz_" +
                           ss.str() + "_" + std::to_string(index) + file_suffix;
    if (!std::filesystem::exists(current_path + "/image")) {
      std::filesystem::create_directory(current_path + "/image");
    }
    RCLCPP_INFO_STREAM(logger_, "Saving image to " << filename);
    if (stream_index.first == OB_STREAM_COLOR) {
      auto image_to_save =
          cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8)->image;
      cv::imwrite(filename, image_to_save);
    } else if (stream_index.first == OB_STREAM_IR || stream_index.first == OB_STREAM_IR_LEFT ||
               stream_index.first == OB_STREAM_IR_RIGHT || stream_index.first == OB_STREAM_DEPTH) {
      std::ofstream ofs(filename, std::ios::out | std::ios::binary);
      if (!ofs.is_open()) {
        RCLCPP_ERROR_STREAM(logger_, "Failed to open file: " << filename);
        return;
      }
      if (image.isContinuous()) {
        ofs.write(reinterpret_cast<const char *>(image.data), image.total() * image.elemSize());
      } else {
        int rows = image.rows;
        int cols = image.cols * image.channels();
        for (int r = 0; r < rows; ++r) {
          ofs.write(reinterpret_cast<const char *>(image.ptr<uchar>(r)), cols);
        }
      }
      ofs.close();
    } else {
      RCLCPP_ERROR_STREAM(logger_, "Unsupported stream type: " << stream_index.first);
    }
    if (++save_images_count_[stream_index] >= max_save_images_count_) {
      save_images_[stream_index] = false;
    }
  }
}

void OBCameraNode::onNewIMUFrameSyncOutputCallback(const std::shared_ptr<ob::Frame> &accelframe,
                                                   const std::shared_ptr<ob::Frame> &gryoframe) {
  if (!is_camera_node_initialized_) {
    return;
  }
  if (!imu_gyro_accel_publisher_) {
    RCLCPP_ERROR_STREAM(logger_, "stream Accel Gryo publisher not initialized");
    return;
  }
  bool has_subscriber = imu_gyro_accel_publisher_->get_subscription_count() > 0;
  has_subscriber = has_subscriber || imu_info_publishers_[GYRO]->get_subscription_count() > 0;
  has_subscriber = has_subscriber || imu_info_publishers_[ACCEL]->get_subscription_count() > 0;
  if (!has_subscriber) {
    return;
  }
  auto imu_msg = sensor_msgs::msg::Imu();
  setDefaultIMUMessage(imu_msg);

  imu_msg.header.frame_id = optical_frame_id_[GYRO];
  auto frame_timestamp = getFrameTimestampUs(accelframe);
  auto timestamp = fromUsToROSTime(frame_timestamp);
  imu_msg.header.stamp = timestamp;

  auto gyro_info = createIMUInfo(GYRO);
  gyro_info.header = imu_msg.header;
  imu_info_publishers_[GYRO]->publish(gyro_info);

  auto accel_info = createIMUInfo(ACCEL);
  imu_msg.header.frame_id = optical_frame_id_[ACCEL];
  accel_info.header = imu_msg.header;
  imu_info_publishers_[ACCEL]->publish(accel_info);

  imu_msg.header.frame_id = accel_gyro_frame_id_;

  auto gyro_frame = gryoframe->as<ob::GyroFrame>();
  auto gyroData = gyro_frame->getValue();
  imu_msg.angular_velocity.x = gyroData.x - gyro_info.bias[0];
  imu_msg.angular_velocity.y = gyroData.y - gyro_info.bias[1];
  imu_msg.angular_velocity.z = gyroData.z - gyro_info.bias[2];

  auto accel_frame = accelframe->as<ob::AccelFrame>();
  auto accelData = accel_frame->getValue();
  imu_msg.linear_acceleration.x = accelData.x - accel_info.bias[0];
  imu_msg.linear_acceleration.y = accelData.y - accel_info.bias[1];
  imu_msg.linear_acceleration.z = accelData.z - accel_info.bias[2];

  imu_gyro_accel_publisher_->publish(imu_msg);
}

void OBCameraNode::onNewIMUFrameCallback(const std::shared_ptr<ob::Frame> &frame,
                                         const stream_index_pair &stream_index) {
  if (!is_camera_node_initialized_) {
    return;
  }
  if (!imu_publishers_.count(stream_index)) {
    RCLCPP_ERROR_STREAM(logger_,
                        "stream " << stream_name_[stream_index] << " publisher not initialized");
    return;
  }
  bool has_subscriber = imu_publishers_[stream_index]->get_subscription_count() > 0;
  has_subscriber =
      has_subscriber || imu_info_publishers_[stream_index]->get_subscription_count() > 0;
  if (!has_subscriber) {
    return;
  }
  auto imu_msg = sensor_msgs::msg::Imu();
  setDefaultIMUMessage(imu_msg);

  imu_msg.header.frame_id = optical_frame_id_[stream_index];
  auto timestamp = fromUsToROSTime(frame->getTimeStampUs());
  imu_msg.header.stamp = timestamp;

  auto imu_info = createIMUInfo(stream_index);
  imu_info.header = imu_msg.header;
  imu_info_publishers_[stream_index]->publish(imu_info);

  if (frame->getType() == OB_FRAME_GYRO) {
    auto gyro_frame = frame->as<ob::GyroFrame>();
    auto data = gyro_frame->getValue();
    imu_msg.angular_velocity.x = data.x - imu_info.bias[0];
    imu_msg.angular_velocity.y = data.y - imu_info.bias[1];
    imu_msg.angular_velocity.z = data.z - imu_info.bias[2];
  } else if (frame->getType() == OB_FRAME_ACCEL) {
    auto accel_frame = frame->as<ob::AccelFrame>();
    auto data = accel_frame->getValue();
    imu_msg.linear_acceleration.x = data.x - imu_info.bias[0];
    imu_msg.linear_acceleration.y = data.y - imu_info.bias[1];
    imu_msg.linear_acceleration.z = data.z - imu_info.bias[2];
  } else {
    RCLCPP_ERROR(logger_, "Unsupported IMU frame type");
    return;
  }
  imu_publishers_[stream_index]->publish(imu_msg);
}

void OBCameraNode::setDefaultIMUMessage(sensor_msgs::msg::Imu &imu_msg) {
  imu_msg.header.frame_id = "imu_link";
  imu_msg.orientation.x = 0.0;
  imu_msg.orientation.y = 0.0;
  imu_msg.orientation.z = 0.0;
  imu_msg.orientation.w = 1.0;

  imu_msg.orientation_covariance = {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  imu_msg.linear_acceleration_covariance = {
      linear_accel_cov_, 0.0, 0.0, 0.0, linear_accel_cov_, 0.0, 0.0, 0.0, linear_accel_cov_};
  imu_msg.angular_velocity_covariance = {
      angular_vel_cov_, 0.0, 0.0, 0.0, angular_vel_cov_, 0.0, 0.0, 0.0, angular_vel_cov_};
}

sensor_msgs::msg::Imu OBCameraNode::createUnitIMUMessage(const IMUData &accel_data,
                                                         const IMUData &gyro_data) {
  sensor_msgs::msg::Imu imu_msg;
  rclcpp::Time timestamp(gyro_data.timestamp_);
  imu_msg.header.stamp = timestamp;
  imu_msg.angular_velocity.x = gyro_data.data_.x();
  imu_msg.angular_velocity.y = gyro_data.data_.y();
  imu_msg.angular_velocity.z = gyro_data.data_.z();

  imu_msg.linear_acceleration.x = accel_data.data_.x();
  imu_msg.linear_acceleration.y = accel_data.data_.y();
  imu_msg.linear_acceleration.z = accel_data.data_.z();
  return imu_msg;
}

std::optional<OBCameraParam> OBCameraNode::findDefaultCameraParam() {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int depth_w = param.depthIntrinsic.width;
    int depth_h = param.depthIntrinsic.height;
    int color_w = param.rgbIntrinsic.width;
    int color_h = param.rgbIntrinsic.height;
    if ((depth_w * height_[DEPTH] == depth_h * width_[DEPTH]) &&
        (color_w * height_[COLOR] == color_h * width_[COLOR])) {
      return param;
    }
  }
  return {};
}

std::optional<OBCameraParam> OBCameraNode::getDepthCameraParam() {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int depth_w = param.depthIntrinsic.width;
    int depth_h = param.depthIntrinsic.height;
    if (depth_w == width_[DEPTH] && depth_h == height_[DEPTH]) {
      RCLCPP_INFO_STREAM(logger_, "getCameraDepthParam w: " << depth_w << ",h:" << depth_h);
      return param;
    }
  }

  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int depth_w = param.depthIntrinsic.width;
    int depth_h = param.depthIntrinsic.height;
    if (depth_w * height_[DEPTH] == depth_h * width_[DEPTH]) {
      RCLCPP_INFO_STREAM(logger_, "getCameraDepthParam w: " << depth_w << ",h:" << depth_h);
      return param;
    }
  }
  return {};
}

std::optional<OBCameraParam> OBCameraNode::getColorCameraParam() {
  auto camera_params = device_->getCalibrationCameraParamList();
  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int color_w = param.rgbIntrinsic.width;
    int color_h = param.rgbIntrinsic.height;
    if (color_w == width_[COLOR] && color_h == height_[COLOR]) {
      RCLCPP_INFO_STREAM(logger_, "getColorCameraParam w: " << color_w << ",h:" << color_h);
      return param;
    }
  }

  for (size_t i = 0; i < camera_params->count(); i++) {
    auto param = camera_params->getCameraParam(i);
    int color_w = param.rgbIntrinsic.width;
    int color_h = param.rgbIntrinsic.height;
    if (color_w * height_[COLOR] == color_h * width_[COLOR]) {
      RCLCPP_INFO_STREAM(logger_, "getColorCameraParam w: " << color_w << ",h:" << color_h);
      return param;
    }
  }
  return {};
}

void OBCameraNode::publishStaticTF(const rclcpp::Time &t, const tf2::Vector3 &trans,
                                   const tf2::Quaternion &q, const std::string &from,
                                   const std::string &to) {
  geometry_msgs::msg::TransformStamped msg;
  msg.header.stamp = t;
  msg.header.frame_id = from;
  msg.child_frame_id = to;
  msg.transform.translation.x = trans[2] / 1000.0;
  msg.transform.translation.y = -trans[0] / 1000.0;
  msg.transform.translation.z = -trans[1] / 1000.0;
  msg.transform.rotation.x = q.getX();
  msg.transform.rotation.y = q.getY();
  msg.transform.rotation.z = q.getZ();
  msg.transform.rotation.w = q.getW();
  static_tf_msgs_.push_back(msg);
}

void OBCameraNode::calcAndPublishStaticTransform() {
  tf2::Quaternion quaternion_optical, zero_rot;
  zero_rot.setRPY(0.0, 0.0, 0.0);
  quaternion_optical.setRPY(-M_PI / 2, 0.0, -M_PI / 2);
  tf2::Vector3 zero_trans(0, 0, 0);
  auto base_stream_profile = stream_profile_[base_stream_];
  auto device_info = device_->getDeviceInfo();
  CHECK_NOTNULL(device_info);
  if (!base_stream_profile) {
    RCLCPP_ERROR_STREAM(logger_, "Failed to get base stream profile");
    return;
  }
  CHECK_NOTNULL(base_stream_profile.get());
  for (const auto &item : stream_profile_) {
    auto stream_index = item.first;

    auto stream_profile = item.second;
    if (!stream_profile) {
      continue;
    }
    OBExtrinsic ex;
    try {
      ex = stream_profile->getExtrinsicTo(base_stream_profile);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << stream_name_[stream_index] << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }

    auto Q = rotationMatrixToQuaternion(ex.rot);
    Q = quaternion_optical * Q * quaternion_optical.inverse();
    tf2::Vector3 trans(ex.trans[0], ex.trans[1], ex.trans[2]);
    auto timestamp = node_->now();
    if (stream_index.first != base_stream_.first) {
      if (stream_index.first == OB_STREAM_IR_RIGHT && base_stream_.first == OB_STREAM_DEPTH) {
        trans[0] = std::abs(trans[0]);  // because left and right ir calibration is error
      }
      publishStaticTF(timestamp, trans, Q, frame_id_[base_stream_], frame_id_[stream_index]);
    }
    publishStaticTF(timestamp, zero_trans, quaternion_optical, frame_id_[stream_index],
                    optical_frame_id_[stream_index]);
    RCLCPP_DEBUG_STREAM(logger_, "Publishing static transform from " << stream_name_[stream_index]
                                                                     << " to "
                                                                     << stream_name_[base_stream_]);
    RCLCPP_DEBUG_STREAM(logger_,
                        "Translation " << trans[0] << ", " << trans[1] << ", " << trans[2]);
    RCLCPP_DEBUG_STREAM(logger_, "Rotation " << Q.getX() << ", " << Q.getY() << ", " << Q.getZ()
                                             << ", " << Q.getW());
  }

  if ((pid_ == FEMTO_BOLT_PID || pid_ == FEMTO_MEGA_PID) && enable_stream_[DEPTH] &&
      enable_stream_[COLOR]) {
    // calc depth to color

    CHECK_NOTNULL(stream_profile_[COLOR]);
    auto depth_to_color_extrinsics = base_stream_profile->getExtrinsicTo(stream_profile_[COLOR]);
    auto Q = rotationMatrixToQuaternion(depth_to_color_extrinsics.rot);
    Q = quaternion_optical * Q * quaternion_optical.inverse();
    publishStaticTF(node_->now(), zero_trans, Q, camera_link_frame_id_, frame_id_[base_stream_]);
  } else {
    publishStaticTF(node_->now(), zero_trans, zero_rot, camera_link_frame_id_,
                    frame_id_[base_stream_]);
  }

  if (enable_stream_[DEPTH] && enable_stream_[COLOR] && enable_publish_extrinsic_) {
    static const char *frame_id = "depth_to_color_extrinsics";
    OBExtrinsic ex;
    try {
      ex = base_stream_profile->getExtrinsicTo(stream_profile_[COLOR]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    depth_to_other_extrinsics_[COLOR] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[COLOR]);
    depth_to_other_extrinsics_publishers_[COLOR]->publish(ex_msg);
  }

  if (enable_stream_[DEPTH] && enable_stream_[INFRA0] && enable_publish_extrinsic_) {
    static const char *frame_id = "depth_to_ir_extrinsics";
    OBExtrinsic ex;
    try {
      ex = base_stream_profile->getExtrinsicTo(stream_profile_[INFRA0]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    depth_to_other_extrinsics_[INFRA0] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[INFRA0]);
    depth_to_other_extrinsics_publishers_[INFRA0]->publish(ex_msg);
  }
  if (enable_stream_[DEPTH] && enable_stream_[INFRA1] && enable_publish_extrinsic_) {
    static const char *frame_id = "depth_to_left_ir_extrinsics";
    OBExtrinsic ex;
    try {
      ex = base_stream_profile->getExtrinsicTo(stream_profile_[INFRA1]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    depth_to_other_extrinsics_[INFRA1] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[INFRA1]);
    depth_to_other_extrinsics_publishers_[INFRA1]->publish(ex_msg);
  }
  if (enable_stream_[DEPTH] && enable_stream_[INFRA2] && enable_publish_extrinsic_) {
    static const char *frame_id = "depth_to_right_ir_extrinsics";
    OBExtrinsic ex;
    try {
      ex = base_stream_profile->getExtrinsicTo(stream_profile_[INFRA2]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    ex.trans[0] = -std::abs(ex.trans[0]);
    depth_to_other_extrinsics_[INFRA2] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[INFRA2]);
    depth_to_other_extrinsics_publishers_[INFRA2]->publish(ex_msg);
  }
  if (enable_stream_[DEPTH] && enable_stream_[ACCEL] && enable_publish_extrinsic_) {
    static const char *frame_id = "depth_to_accel_extrinsics";
    OBExtrinsic ex;
    try {
      ex = base_stream_profile->getExtrinsicTo(stream_profile_[ACCEL]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    depth_to_other_extrinsics_[ACCEL] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[ACCEL]);
    depth_to_other_extrinsics_publishers_[ACCEL]->publish(ex_msg);
  }
  if (enable_stream_[DEPTH] && enable_stream_[GYRO] && enable_publish_extrinsic_) {
    static const char *frame_id = "depth_to_gyro_extrinsics";
    OBExtrinsic ex;
    try {
      ex = base_stream_profile->getExtrinsicTo(stream_profile_[GYRO]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    depth_to_other_extrinsics_[GYRO] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[GYRO]);
    depth_to_other_extrinsics_publishers_[GYRO]->publish(ex_msg);
  }
  if (enable_stream_[COLOR_LEFT] && enable_stream_[COLOR_RIGHT] && enable_publish_extrinsic_) {
    static const char *frame_id = "left_color_to_right_color_extrinsics";
    OBExtrinsic ex;
    try {
      ex = stream_profile_[COLOR_LEFT]->getExtrinsicTo(stream_profile_[COLOR_RIGHT]);
    } catch (const ob::Error &e) {
      RCLCPP_ERROR_STREAM(logger_, "Failed to get " << frame_id << " extrinsic: "
                                                    << orbbec_camera::formatObErrorWithStatus(e));
      ex = OBExtrinsic({{1, 0, 0, 0, 1, 0, 0, 0, 1}, {0, 0, 0}});
    }
    depth_to_other_extrinsics_[COLOR_LEFT] = ex;
    auto ex_msg = obExtrinsicsToMsg(ex, frame_id);
    CHECK_NOTNULL(depth_to_other_extrinsics_publishers_[COLOR_LEFT]);
    depth_to_other_extrinsics_publishers_[COLOR_LEFT]->publish(ex_msg);
  }
  if (enable_sync_output_accel_gyro_) {
    tf2::Quaternion zero_rot;
    zero_rot.setRPY(0.0, 0.0, 0.0);
    tf2::Vector3 zero_trans(0, 0, 0);
    publishStaticTF(node_->now(), zero_trans, zero_rot, optical_frame_id_[GYRO],
                    accel_gyro_frame_id_);
  }
}

void OBCameraNode::publishStaticTransforms() {
  if (!publish_tf_) {
    return;
  }
  static_tf_broadcaster_ = std::make_shared<tf2_ros::StaticTransformBroadcaster>(node_);
  dynamic_tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(node_);
  calcAndPublishStaticTransform();
  if (tf_publish_rate_ > 0) {
    tf_thread_ = std::make_shared<std::thread>([this]() { publishDynamicTransforms(); });
  } else {
    static_tf_broadcaster_->sendTransform(static_tf_msgs_);
  }
}

void OBCameraNode::publishDynamicTransforms() {
  RCLCPP_WARN(logger_, "Publishing dynamic camera transforms (/tf) at %g Hz", tf_publish_rate_);
  std::mutex mu;
  std::unique_lock<std::mutex> lock(mu);
  while (rclcpp::ok() && is_running_) {
    tf_cv_.wait_for(lock, std::chrono::milliseconds((int)(1000.0 / tf_publish_rate_)),
                    [this] { return (!(is_running_)); });
    {
      rclcpp::Time t = node_->now();
      for (auto &msg : static_tf_msgs_) {
        msg.header.stamp = t;
      }
      dynamic_tf_broadcaster_->sendTransform(static_tf_msgs_);
    }
  }
}

template <typename T>
T lerp(const T &a, const T &b, const double t) {
  return a * (1.0 - t) + b * t;
}

void OBCameraNode::FillImuDataLinearInterpolation(const IMUData &imu_data,
                                                  std::deque<sensor_msgs::msg::Imu> &imu_msgs) {
  imu_history_.push_back(imu_data);
  stream_index_pair steam_index(imu_data.stream_);
  imu_msgs.clear();
  std::deque<IMUData> gyros_data;
  IMUData accel0, accel1, current_imu;
  while (!imu_history_.empty()) {
    current_imu = imu_history_.front();
    imu_history_.pop_front();
    if (accel0.isSet() && current_imu.stream_ == ACCEL) {
      accel0 = current_imu;
    } else if (accel0.isSet() && current_imu.stream_ == ACCEL) {
      accel1 = current_imu;
      const double dt = accel1.timestamp_ - accel0.timestamp_;
      while (!gyros_data.empty()) {
        auto current_gyro = gyros_data.front();
        gyros_data.pop_front();
        const double alpha = (current_gyro.timestamp_ - accel0.timestamp_) / dt;
        IMUData current_accel(ACCEL, lerp(accel0.data_, accel1.data_, alpha),
                              current_gyro.timestamp_);
        imu_msgs.push_back((createUnitIMUMessage(current_accel, current_gyro)));
      }
      accel0 = accel1;
    } else if (accel0.isSet() && current_imu.timestamp_ >= accel0.timestamp_ &&
               current_imu.stream_ == GYRO) {
      gyros_data.push_back(current_imu);
    }
  }
  imu_history_.push_back(current_imu);
}

void OBCameraNode::FillImuDataCopy(const IMUData &imu_data,
                                   std::deque<sensor_msgs::msg::Imu> &imu_msgs) {
  stream_index_pair steam_index(imu_data.stream_);
  if (steam_index == ACCEL) {
    accel_data_ = imu_data;
    return;
  }
  if (accel_data_.isSet()) {
    return;
  }
  imu_msgs.push_back(createUnitIMUMessage(accel_data_, imu_data));
}

bool OBCameraNode::setupFormatConvertType(OBFormat format) {
  return setupFormatConvertType(format, format_convert_filter_);
}

bool OBCameraNode::setupFormatConvertType(OBFormat format, ob::FormatConvertFilter &filter) {
  switch (format) {
    case OB_FORMAT_RGB888:
      return true;
    case OB_FORMAT_I420:
      filter.setFormatConvertType(FORMAT_I420_TO_RGB888);
      break;
    case OB_FORMAT_MJPG:
      filter.setFormatConvertType(FORMAT_MJPEG_TO_RGB888);
      break;
    case OB_FORMAT_YUYV:
      filter.setFormatConvertType(FORMAT_YUYV_TO_RGB888);
      break;
    case OB_FORMAT_NV21:
      filter.setFormatConvertType(FORMAT_NV21_TO_RGB888);
      break;
    case OB_FORMAT_NV12:
      filter.setFormatConvertType(FORMAT_NV12_TO_RGB888);
      break;
    case OB_FORMAT_UYVY:
      filter.setFormatConvertType(FORMAT_UYVY_TO_RGB888);
      break;
    default:
      return false;
  }
  return true;
}

bool OBCameraNode::isGemini335PID(uint32_t pid) {
  return pid == GEMINI_335_PID || pid == GEMINI_330_PID || pid == GEMINI_336_PID ||
         pid == GEMINI_335L_PID || pid == GEMINI_330L_PID || pid == GEMINI_336L_PID ||
         pid == GEMINI_335LG_PID || pid == GEMINI_336LG_PID || pid == GEMINI_335LE_PID ||
         pid == GEMINI_336LE_PID || pid == CUSTOM_ADVANTECH_GEMINI_336_PID ||
         pid == CUSTOM_ADVANTECH_GEMINI_336L_PID || pid == GEMINI_338_PID ||
         pid == GEMINI_338L_PID || pid == GEMINI_338LE_PID || pid == GEMINI_338LG_PID;
}

bool OBCameraNode::isGemini435LePID(uint32_t pid) { return pid == GEMINI_435Le_PID; }
bool OBCameraNode::isPublishMetaData(uint32_t pid) {
  return isGemini335PID(pid) || isGemini435LePID(pid) || isGemini305SeriesPID(pid);
}

bool OBCameraNode::isDepthWorkModeDevices(uint32_t pid) { return pid == GEMINI_435Le_PID; }

bool OBCameraNode::isnotLaserDevices(uint32_t pid) { return isGemini305SeriesPID(pid); }

orbbec_camera_msgs::msg::IMUInfo OBCameraNode::createIMUInfo(
    const stream_index_pair &stream_index) {
  orbbec_camera_msgs::msg::IMUInfo imu_info;
  imu_info.header.frame_id = optical_frame_id_[stream_index];
  imu_info.header.stamp = node_->now();

  if (stream_index == GYRO) {
    auto gyro_profile = stream_profile_[stream_index]->as<ob::GyroStreamProfile>();
    auto gyro_intrinsics = gyro_profile->getIntrinsic();
    imu_info.noise_density = gyro_intrinsics.noiseDensity;
    imu_info.random_walk = gyro_intrinsics.randomWalk;
    imu_info.reference_temperature = gyro_intrinsics.referenceTemp;
    imu_info.bias = {gyro_intrinsics.bias[0], gyro_intrinsics.bias[1], gyro_intrinsics.bias[2]};
    imu_info.scale_misalignment = {
        gyro_intrinsics.scaleMisalignment[0], gyro_intrinsics.scaleMisalignment[1],
        gyro_intrinsics.scaleMisalignment[2], gyro_intrinsics.scaleMisalignment[3],
        gyro_intrinsics.scaleMisalignment[4], gyro_intrinsics.scaleMisalignment[5],
        gyro_intrinsics.scaleMisalignment[6], gyro_intrinsics.scaleMisalignment[7],
        gyro_intrinsics.scaleMisalignment[8]};
    imu_info.temperature_slope = {
        gyro_intrinsics.tempSlope[0], gyro_intrinsics.tempSlope[1], gyro_intrinsics.tempSlope[2],
        gyro_intrinsics.tempSlope[3], gyro_intrinsics.tempSlope[4], gyro_intrinsics.tempSlope[5],
        gyro_intrinsics.tempSlope[6], gyro_intrinsics.tempSlope[7], gyro_intrinsics.tempSlope[8]};
  } else if (stream_index == ACCEL) {
    auto accel_profile = stream_profile_[stream_index]->as<ob::AccelStreamProfile>();
    auto accel_intrinsics = accel_profile->getIntrinsic();
    imu_info.noise_density = accel_intrinsics.noiseDensity;
    imu_info.random_walk = accel_intrinsics.randomWalk;
    imu_info.reference_temperature = accel_intrinsics.referenceTemp;
    imu_info.bias = {accel_intrinsics.bias[0], accel_intrinsics.bias[1], accel_intrinsics.bias[2]};
    imu_info.gravity = {accel_intrinsics.gravity[0], accel_intrinsics.gravity[1],
                        accel_intrinsics.gravity[2]};
    imu_info.scale_misalignment = {
        accel_intrinsics.scaleMisalignment[0], accel_intrinsics.scaleMisalignment[1],
        accel_intrinsics.scaleMisalignment[2], accel_intrinsics.scaleMisalignment[3],
        accel_intrinsics.scaleMisalignment[4], accel_intrinsics.scaleMisalignment[5],
        accel_intrinsics.scaleMisalignment[6], accel_intrinsics.scaleMisalignment[7],
        accel_intrinsics.scaleMisalignment[8]};
    imu_info.temperature_slope = {accel_intrinsics.tempSlope[0], accel_intrinsics.tempSlope[1],
                                  accel_intrinsics.tempSlope[2], accel_intrinsics.tempSlope[3],
                                  accel_intrinsics.tempSlope[4], accel_intrinsics.tempSlope[5],
                                  accel_intrinsics.tempSlope[6], accel_intrinsics.tempSlope[7],
                                  accel_intrinsics.tempSlope[8]};
  }

  return imu_info;
}
void OBCameraNode::setFilterCallback(const std::shared_ptr<SetFilter ::Request> &request,
                                     std::shared_ptr<SetFilter ::Response> &response) {
  try {
    response->success = false;
    response->message.clear();
    auto fail = [&response](const std::string &msg) {
      response->success = false;
      response->message = msg;
    };
    const auto normalized_request_filter_name = normalizeDepthFilterName(request->filter_name);
    const bool is_noise_removal_filter = normalized_request_filter_name == "NoiseRemovalFilter";
    const bool is_hardware_noise_removal_filter =
        normalized_request_filter_name == "HardwareNoiseRemovalFilter";
    bool is_supported_by_property = false;
    if (is_noise_removal_filter) {
      is_supported_by_property =
          device_->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL, OB_PERMISSION_READ_WRITE) ||
          device_->isPropertySupported(OB_PROP_DEPTH_MAX_DIFF_INT, OB_PERMISSION_WRITE) ||
          device_->isPropertySupported(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, OB_PERMISSION_WRITE);
    } else if (is_hardware_noise_removal_filter) {
      is_supported_by_property =
          device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                                       OB_PERMISSION_READ_WRITE) ||
          device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                       OB_PERMISSION_READ_WRITE);
    }

    RCLCPP_INFO_STREAM(logger_, "filter_name: " << request->filter_name << "  filter_enable: "
                                                << (request->filter_enable ? "true" : "false"));
    if (is_noise_removal_filter || is_hardware_noise_removal_filter) {
      if (!is_supported_by_property) {
        fail("Filter '" + normalized_request_filter_name + "' is not supported by this device");
        return;
      }
      if (is_noise_removal_filter) {
        if (device_->isPropertySupported(OB_PROP_DEPTH_SOFT_FILTER_BOOL,
                                         OB_PERMISSION_READ_WRITE)) {
          device_->setBoolProperty(OB_PROP_DEPTH_SOFT_FILTER_BOOL, request->filter_enable);
          RCLCPP_INFO_STREAM(logger_, "enable_noise_removal_filter:" << request->filter_enable);
        }
        if (request->filter_param.size() > 1) {
          if (device_->isPropertySupported(OB_PROP_DEPTH_MAX_DIFF_INT, OB_PERMISSION_WRITE)) {
            device_->setIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT, request->filter_param[0]);
            auto new_noise_removal_filter_min_diff =
                device_->getIntProperty(OB_PROP_DEPTH_MAX_DIFF_INT);
            RCLCPP_INFO_STREAM(logger_, "Set noise_removal_filter_min_diff: "
                                            << new_noise_removal_filter_min_diff);
            noise_removal_filter_min_diff_ = request->filter_param[0];
          }
          if (device_->isPropertySupported(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT,
                                           OB_PERMISSION_WRITE)) {
            device_->setIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, request->filter_param[1]);
            auto new_noise_removal_filter_max_size =
                device_->getIntProperty(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT);
            RCLCPP_INFO_STREAM(logger_, "Set noise_removal_filter_max_size: "
                                            << new_noise_removal_filter_max_size);
            noise_removal_filter_max_size_ = request->filter_param[1];
          }
        }
        enable_noise_removal_filter_ = request->filter_enable;
      } else if (is_hardware_noise_removal_filter) {
        if (device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                                         OB_PERMISSION_READ_WRITE)) {
          device_->setBoolProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_ENABLE_BOOL,
                                   request->filter_enable);
          RCLCPP_INFO_STREAM(logger_,
                             "Set hardware_noise_removal_filter:" << request->filter_enable);
          if (request->filter_param.size() > 0 &&
              device_->isPropertySupported(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                           OB_PERMISSION_READ_WRITE)) {
            if (request->filter_enable) {
              device_->setFloatProperty(OB_PROP_HW_NOISE_REMOVE_FILTER_THRESHOLD_FLOAT,
                                        request->filter_param[0]);
              RCLCPP_INFO_STREAM(logger_, "Set hardware_noise_removal_filter_threshold :"
                                              << request->filter_param[0]);
              hardware_noise_removal_filter_threshold_ = request->filter_param[0];
            }
          } else {
            fail("The filter switch setting is successful, but the filter parameter setting fails");
            return;
          }
        }
        enable_hardware_noise_removal_filter_ = request->filter_enable;
      }
    } else {
      std::unique_lock<std::mutex> depth_filter_lock(depth_filter_mutex_);
      auto is_same_filter =
          [&normalized_request_filter_name](const std::shared_ptr<ob::Filter> &filter) {
            if (!filter) {
              return false;
            }
            return normalizeDepthFilterName(filter->getName()) == normalized_request_filter_name ||
                   normalizeDepthFilterName(filter->type()) == normalized_request_filter_name;
          };

      auto first_match_it =
          std::find_if(depth_filter_list_.begin(), depth_filter_list_.end(),
                       [&is_same_filter](const auto &filter) { return is_same_filter(filter); });
      if (first_match_it == depth_filter_list_.end()) {
        fail("Filter '" + normalized_request_filter_name + "' is not supported by this device");
        return;
      }
      const auto &existing_filter = *first_match_it;
      if (!existing_filter) {
        fail("Filter '" + normalized_request_filter_name + "' is not supported by this device");
        return;
      }

      if (normalized_request_filter_name == "DecimationFilter") {
        auto decimation_filter = existing_filter->as<ob::DecimationFilter>();
        decimation_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 0) {
          auto range = decimation_filter->getScaleRange();
          auto decimation_filter_scale = request->filter_param[0];
          if (decimation_filter_scale <= range.max && decimation_filter_scale >= range.min) {
            RCLCPP_INFO_STREAM(logger_,
                               "Set decimation filter scale value to " << decimation_filter_scale);
            decimation_filter->setScaleValue(decimation_filter_scale);
          }
          if (decimation_filter_scale != -1 &&
              (decimation_filter_scale < range.min || decimation_filter_scale > range.max)) {
            RCLCPP_ERROR_STREAM(logger_, "Decimation filter scale value is out of range "
                                             << range.min << " - " << range.max);
            fail("Decimation filter scale value is out of range");
            return;
          }
          if (decimation_filter_scale <= range.max && decimation_filter_scale >= range.min) {
            decimation_filter_scale_ = decimation_filter_scale;
          }
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_decimation_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "HDRMerge") {
        auto hdr_merge_filter = existing_filter->as<ob::HdrMerge>();
        hdr_merge_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 3) {
          auto config = OBHdrConfig();
          config.enable = true;
          config.exposure_1 = request->filter_param[0];
          config.gain_1 = request->filter_param[1];
          config.exposure_2 = request->filter_param[2];
          config.gain_2 = request->filter_param[3];
          device_->setStructuredData(OB_STRUCT_DEPTH_HDR_CONFIG,
                                     reinterpret_cast<const uint8_t *>(&config), sizeof(config));
          RCLCPP_INFO_STREAM(logger_, "Set HDR merge filter params: "
                                          << "\nexposure_1: " << request->filter_param[0]
                                          << "\ngain_1: " << request->filter_param[1]
                                          << "\nexposure_2: " << request->filter_param[2]
                                          << "\ngain_2: " << request->filter_param[3]);
          hdr_merge_exposure_1_ = request->filter_param[0];
          hdr_merge_gain_1_ = request->filter_param[1];
          hdr_merge_exposure_2_ = request->filter_param[2];
          hdr_merge_gain_2_ = request->filter_param[3];
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_hdr_merge_ = request->filter_enable;
      } else if (normalized_request_filter_name == "SequenceIdFilter") {
        auto sequenced_filter = existing_filter->as<ob::SequenceIdFilter>();
        sequenced_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 0) {
          sequenced_filter->selectSequenceId(request->filter_param[0]);
          RCLCPP_INFO_STREAM(logger_, "Set sequenced filter selectSequenceId value to "
                                          << request->filter_param[0]);
          sequence_id_filter_id_ = request->filter_param[0];
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_sequence_id_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "ThresholdFilter") {
        auto threshold_filter = existing_filter->as<ob::ThresholdFilter>();
        threshold_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 1) {
          auto threshold_filter_min = request->filter_param[0];
          auto threshold_filter_max = request->filter_param[1];
          threshold_filter->setValueRange(threshold_filter_min, threshold_filter_max);
          RCLCPP_INFO_STREAM(logger_, "Set threshold filter value range to "
                                          << threshold_filter_min << " - " << threshold_filter_max);
          threshold_filter_min_ = threshold_filter_min;
          threshold_filter_max_ = threshold_filter_max;
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_threshold_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "SpatialAdvancedFilter") {
        auto spatial_filter = existing_filter->as<ob::SpatialAdvancedFilter>();
        spatial_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 3) {
          OBSpatialAdvancedFilterParams params{};
          params.alpha = request->filter_param[0];
          params.disp_diff = request->filter_param[1];
          params.magnitude = request->filter_param[2];
          params.radius = request->filter_param[3];
          spatial_filter->setFilterParams(params);
          RCLCPP_INFO_STREAM(logger_, "Set SpatialFilter params: "
                                          << "\nalpha:" << params.alpha
                                          << "\ndisp_diff:" << params.disp_diff
                                          << "\nmagnitude:" << static_cast<int>(params.magnitude)
                                          << "\nradius:" << params.radius);
          spatial_filter_alpha_ = params.alpha;
          spatial_filter_diff_threshold_ = params.disp_diff;
          spatial_filter_magnitude_ = params.magnitude;
          spatial_filter_radius_ = params.radius;
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_spatial_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "TemporalFilter") {
        auto temporal_filter = existing_filter->as<ob::TemporalFilter>();
        temporal_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 1) {
          temporal_filter->setDiffScale(request->filter_param[0]);
          temporal_filter->setWeight(request->filter_param[1]);
          RCLCPP_INFO_STREAM(logger_, "Set TemporalFilter params: "
                                          << "\ndiff_scale:" << request->filter_param[0]
                                          << "\nweight:" << request->filter_param[1]);
          temporal_filter_diff_threshold_ = request->filter_param[0];
          temporal_filter_weight_ = request->filter_param[1];
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_temporal_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "SpatialFastFilter") {
        auto spatial_fast_filter = existing_filter->as<ob::SpatialFastFilter>();
        spatial_fast_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 0) {
          OBSpatialFastFilterParams params{};
          params.radius = request->filter_param[0];
          spatial_fast_filter->setFilterParams(params);
          RCLCPP_INFO_STREAM(logger_,
                             "Set SpatialFastFilter radius to " << static_cast<int>(params.radius));
          spatial_fast_filter_radius_ = params.radius;
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_spatial_fast_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "SpatialModerateFilter") {
        auto spatial_moderate_filter = existing_filter->as<ob::SpatialModerateFilter>();
        spatial_moderate_filter->enable(request->filter_enable);
        if (request->filter_param.size() > 2) {
          OBSpatialModerateFilterParams params{};
          params.disp_diff = request->filter_param[0];
          params.magnitude = request->filter_param[1];
          params.radius = request->filter_param[2];
          spatial_moderate_filter->setFilterParams(params);
          RCLCPP_INFO_STREAM(logger_, "Set SpatialModerateFilter params: "
                                          << "\ndisp_diff:" << params.disp_diff
                                          << "\nmagnitude:" << static_cast<int>(params.magnitude)
                                          << "\nradius:" << static_cast<int>(params.radius));
          spatial_moderate_filter_diff_threshold_ = params.disp_diff;
          spatial_moderate_filter_magnitude_ = params.magnitude;
          spatial_moderate_filter_radius_ = params.radius;
        } else {
          fail("The filter switch setting is successful, but the filter parameter setting fails");
          return;
        }
        enable_spatial_moderate_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "FalsePositiveFilter") {
        auto false_positive_filter = existing_filter->as<ob::FalsePositiveFilter>();
        false_positive_filter->enable(request->filter_enable);
        enable_false_positive_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "MgcNoiseRemovalFilter") {
        auto mgc_filter = existing_filter->as<ob::MgcNoiseRemovalFilter>();
        mgc_filter->enable(request->filter_enable);
        enable_mgc_noise_removal_filter_ = request->filter_enable;
      } else if (normalized_request_filter_name == "LutNoiseRemovalFilter") {
        auto lut_filter = existing_filter->as<ob::LutNoiseRemovalFilter>();
        lut_filter->enable(request->filter_enable);
        enable_lut_noise_removal_filter_ = request->filter_enable;
      } else {
        fail(normalized_request_filter_name + " cannot be set");
        return;
      }
    }
    filter_status_[normalized_request_filter_name] = static_cast<bool>(request->filter_enable);
    if (filter_status_pub_) {
      std_msgs::msg::String msg;
      msg.data = filter_status_.dump(2);
      filter_status_pub_->publish(msg);
    }
    publishDepthFiltersStatus();
    response->success = true;
  } catch (const ob::Error &e) {
    response->success = false;
    response->message = "Failed to set filter: " + orbbec_camera::formatObErrorWithStatus(e);
    RCLCPP_ERROR_STREAM(logger_,
                        "Failed to set filter: " << orbbec_camera::formatObErrorWithStatus(e));
  } catch (const std::exception &e) {
    response->success = false;
    response->message = std::string("Failed to set filter: ") + e.what();
    RCLCPP_ERROR_STREAM(logger_, "Failed to set filter: " << e.what());
  } catch (...) {
    response->success = false;
    response->message = "unknown error";
    RCLCPP_ERROR_STREAM(logger_, "unknown error");
  }
}
bool OBCameraNode::isWriteCustomerDataSuccess() const {
  return write_customer_data_success_.load();
}
}  // namespace orbbec_camera
