cmake_minimum_required(VERSION 3.8)
project(robot_perception_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/LabeledPose.msg"
  "msg/LabeledPoseArray.msg"
  "msg/LabeledBBox2D.msg"
  "msg/LabeledBBox3D.msg"
  "msg/LabeledBBox2DArray.msg"
  "msg/LabeledBBox3DArray.msg"
  "msg/SurfaceMesh.msg"
  "msg/SurfaceMeshArray.msg"
  "msg/WorkbenchPlane.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
