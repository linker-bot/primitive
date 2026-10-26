cmake_minimum_required(VERSION 3.8)
project(hand_gesture_primitives_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SavePose.srv"
  "srv/ListPoses.srv"
  "srv/DeletePose.srv"
  "srv/ImportGuiPresets.srv"
  "srv/PlanGesture.srv"
  "srv/ExportPoseBundle.srv"
  "srv/ImportPoseBundle.srv"
  "srv/ExportPrimitivePoses.srv"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
