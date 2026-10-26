# ROS 与姿态数据接口（0.2.2）

本文说明本包实际提供的接口。环境与完整启动命令见[安装指南](../../docs/getting-started.md)，
网页操作见[用户手册](../../docs/teach-panel-user-guide.md)。下文 `{side}` 为 `left` 或 `right`。

## 节点与启动前提

| 可执行文件 | 职责 |
| --- | --- |
| `gesture_node` | 接收手势命令、执行原语、发布控制目标和状态；提供 `PlanGesture` |
| `grasp_gate` | 按目标几何条件过滤需要门控的指令 |
| `teach_panel` | 独立 Mock 或连接 ROS 原语节点的网页面板 |
| `pose_library_server` | 可选的姿态库 ROS 服务，网页直接访问共享姿态库时不必启动 |
| `joint_state_bridge` | 将反馈转换为 mesh URDF 的关节状态，用于可视化 |
| `mock_perception` | 模拟目标几何；不提供手驱动或关节反馈 |

真实 ROS 接入需先准备设备驱动、新鲜反馈及对应控制器的 `robot_description`。
默认后端使用弧度目标；省略实际 URDF 的启动命令不能作为通用真机部署方式。
`gesture_node` 初始化会选择 `init`，完整启动顺序与停止确认步骤见安装指南。

## gesture_node 的主要参数

下表为节点声明的默认值；launch 可覆盖它们。`hand_joint` 非空时覆盖 `hand_type`。

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `hand_side` | `left` | 左右侧 |
| `hand_type` | `o20` | 支持 o20、o6、l6、l25、o30、g20；l20 使用 l25 配置 |
| `hand_joint` | 空 | 兼容的手型覆盖参数 |
| `state_topic` | `/joint_states` | 实际关节反馈 |
| `cmd_backend` | 空 | 当前六种配置手型自动选择 `ros2_control`；也可显式选择 `legacy_cb` |
| `ros2_control_command_units` | `radians` | `radians` 或 `hardware`；`legacy_cb` 必须配合 `hardware` |
| `robot_description` | 空 | 控制器关节顺序与限位 URDF XML；弧度控制和 ROS 工作台需要 |
| `g20_joint_order` | 空 | G20 的 native / l25_urdf 关节排列选择 |
| `control_rate` | `10.0` | 启动时控制频率，范围 1–200 Hz；运行中不能改此参数 |
| `primitive_speed` | `1.0` | 运行速度倍率，范围 0.25–3.0，可动态设置 |
| `cmd_topic` | 空 | 自动使用 `/{side}_hand/gesture_cmd`；启用 gate 的 launch 改为 `gesture_cmd_exec` |
| `hand_cmd_topic` | 空 | 按后端选择下表中的控制输出 |
| `tcp_pose_topic` | 空 | 自动使用 `/{side}_hand/tcp_pose` |
| `object_pose_topic` | 空 | 自动使用 `/{side}_hand/object_pose` |
| `bboxes_3d_topic` | `/camera_head/detection_bbox/bboxes_3d` | 几何与 grasp_type 输入 |
| `labeled_pose_topic` | `/camera_head/perception/labeled_poses` | 可选的其他感知输入 |
| `motion_progress_topic` | 空 | 自动使用 `/{side}_hand/motion_progress` |
| `stream_target_topic` | 空 | 自动使用 `/{side}_hand/stream_target` |
| `stream_watchdog_sec` | `0.5` | STREAM 断流后保持当前目标 |
| `urdf_path` | 空 | FK 几何模型路径，与控制器 `robot_description` 的用途不同 |
| `publish_fingertips` | `true` | 有效 FK 时发布指尖位置与 marker |
| `contact_config_path` | 空 | 接触阈值 YAML；空时读取随包配置 |

接触、过载阈值由 `config/contact_thresholds.yaml`、手型配置及相应 ROS 参数共同决定。
应按设备反馈量纲核对，不能把电流 mA、力矩百分比和触觉值互换。

## 主要话题

### 输入

| 默认话题 | 类型 | 语义 |
| --- | --- | --- |
| `/{side}_hand/gesture_cmd` | `std_msgs/String` | 空格分隔的命令文本，不是 JSON |
| `/{side}_hand/gesture_cmd_exec` | `std_msgs/String` | gate 过滤后的执行入口，仅供启用门控的节点使用 |
| `/joint_states` | `sensor_msgs/JointState` | 驱动反馈，按手型和关节名提取 |
| `/{side}_hand/motion_progress` | `std_msgs/Float32` | 更新当前支持进度控制的段，范围 0–1 |
| `/{side}_hand/stream_target` | `sensor_msgs/JointState` | STREAM 目标；量纲与排列见下节 |
| `/{side}_hand/tcp_pose` | `geometry_msgs/PoseStamped` | 集成方提供的 TCP 位姿 |
| `/{side}_hand/object_pose` | `geometry_msgs/PoseStamped` | 目标位姿；可由 gate 或其他感知组件提供 |
| `/camera_head/detection_bbox/bboxes_3d` | `robot_perception_msgs/LabeledBBox3DArray` | 目标位置、尺寸、类别与抓取类型 |
| `/camera_head/perception/labeled_poses` | `robot_perception_msgs/LabeledPoseArray` | 可选的带标签位姿数组 |

力反馈按设备接入 `/cb_{side}_hand_info`（`std_msgs/String`）、
`/cb_{side}_hand_force`（`hand_force_msg/HandForce`）、
`/cb_{side}_hand_normal_force` 和 `/cb_{side}_hand_motor_torque`（`std_msgs/Float32MultiArray`），
或 `/cb_{side}_hand_matrix_touch_mass`（`std_msgs/String`）。不是所有手型都提供这些输入。
代码对部分可选消息导入有降级处理，但完整部署仍应按安装指南编译消息包。

### 输出

| 默认话题 | 类型 | 语义 |
| --- | --- | --- |
| `/{side}_hand_controller/commands` | `std_msgs/Float64MultiArray` | ros2_control 目标，按 `control_rate` 下发 |
| `/cb_{side}_hand_control_cmd` | `sensor_msgs/JointState` | legacy_cb 的产品关节 0–255 目标 |
| `/{side}_hand/gesture_status` | `std_msgs/String` | 每 0.5 秒发布，即约 2 Hz |
| `/{side}_hand/gesture_result` | `std_msgs/String` | 执行结果；周期检查约 1 Hz，通常只在结果改变时发送，另有即时结果 |
| `/cb_{side}_hand_fingertip_positions` | `std_msgs/Float32MultiArray` | 有效 FK 时的五指 xyz，单位米 |
| `/cb_{side}_hand_fingertip_markers` | `visualization_msgs/MarkerArray` | FK 指尖可视化 |

工作台还使用 `/{side}_hand/gesture_catalog`、`workbench_command`、`workbench_result`
三个 `std_msgs/String` JSON 话题，协议版本为 v2。执行身份和控制语义见
[工作台协议](../../docs/workbench.md#ros-控制与进度协议)。它们与原始 `gesture_cmd` 文本协议不同。

常规 `gesture_status` 形如：

```text
<active>|mode=<mode>|speed=<倍率>|control_rate=<频率>|settle=<状态>
```

后面可追加 motion、grasp、type、suggest 等字段；拒绝或查询结果可临时覆盖整条状态。
`gesture_result` 包括 `IDLE`、`EXECUTING: <name>`、`DONE: <name>`、`POSES:a,b` 等文本。
停止确认和 `DONE` 的软件状态都不能代替设备物理停稳检查。

## 命令文本与进度控制

以下指令发往公开的 `/{side}_hand/gesture_cmd`。手型支持范围以 [README](../../README.md)
和 `hand_profile` 为准；规范原语名见[原语使用说明](primitives.md)。

| 指令 | 作用 |
| --- | --- |
| `open`、`fist`、`pinch` 等 | 执行当前手型支持的注册原语 |
| `stop` | 停止当前动作并保持目标 |
| `mode primitive` / `mode teach` / `mode stream` | 切换控制模式；teach/stream 模式拒绝普通原语执行 |
| `pose my_grip` | 回放当前手型、当前侧保存的自定义手势 |
| `pose my_grip grasp 0.35` | 对指定关键帧段设置进度；只给帧名时进入 0% hold |
| `pose_list` | 查询当前姿态库，结果为 `POSES:...` |
| `tripod prep` / `tripod close` | 执行支持分阶段命令的原语相位 |
| `tripod prep 0.3` / `large_wrap close 0.5` | 对支持的定位相位设置进度 |
| `target screwdriver` / `target_id 3` | 设置目标筛选；`target_id 0` 清除 ID 筛选 |
| `auto` / `auto screwdriver` | 按当前目标 grasp_type 与手型配置选择原语 |

`named_pose` 是注册模板名；ROS 文本入口使用 `pose <name>`，不能用工作台菜单 ID
`named_pose:<name>` 代替。`PlanGesture` 请求接受的姿态别名也不等同于命令文本。

示例要求原语节点和驱动已按安装指南就绪；每次选择一条操作：

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right open
ros2 run hand_gesture_primitives pub_gesture_cmd -- right stop
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "pose my_grip"
ros2 run hand_gesture_primitives pub_gesture_cmd -- right pose_list
```

进度输入只更新已建立的段，不自动开启新动作。内置命令级 progress 仅支持 `tripod`
和 `large_wrap` 的 `prep` / `close`；`auto/full` 以及力控相位不能用百分比替代反馈闭环。
自定义关键帧的同段连续更新不重新取起点，换段会重新建立预览，详见[网页说明](../../docs/teach-panel-user-guide.md)。

### STREAM 的量纲与排列

STREAM 只在 `mode stream` 下接收 `JointState.position`；按配置读取所需数量的前 N 项，
不能仅靠修改 `JointState.name` 重新排列目标。

- `ros2_control_command_units=radians` 且转换器有效：使用**实际控制器顺序的弧度**，
  N 等于转换器关节数。例如 L25 通常为 16 个主动关节，不能发送 25 项硬件值冒充弧度。
- `hardware` 模式：使用产品 DOF 顺序的 0–255 值；`legacy_cb` 要求此模式。
- `stream_watchdog_sec` 超时后保持当前目标；退出 STREAM 不会自动恢复旧动作。

## 姿态库与迁移

必须设置 `HAND_GESTURE_POSE_DIR`；文件按规范手型与侧别存储，例如 `o6_right.json`。
新姿态使用 `semantic_v3`、30 维 S30 和 `unit=rad`；文件 schema 与传输 bundle 的版本分别管理。
旧 hardware、semantic_o20、legacy255 数据保留读取兼容。详细量纲见[工作台数据约定](../../docs/workbench.md#数据与公共接口)。

传输包的 `kind` 是 `hand_gesture_pose_bundle`，当前 **`schema_version=2`**；兼容读取
v1 hardware/legacy255。导出的 v2 包保存每个条目的 `space`、`unit`、关键帧和说明，
不意味着每条旧数据都会自动变成新格式。checksum 校验条目内容；导入策略为 `skip`、`overwrite`、`rename`。

以下 CLI 只处理姿态数据，不发送动作：

```bash
export HAND_GESTURE_POSE_DIR="$HOME/hand-gesture-data/poses"
ros2 run hand_gesture_primitives pose_bundle_export -- \
  --hand-type o6 --hand-side right --output /tmp/right_o6.pose_bundle.json
ros2 run hand_gesture_primitives pose_bundle_import -- \
  --input /tmp/right_o6.pose_bundle.json --strategy skip --hand-type o6 --hand-side right
```

目标机器应先设置自己的持久化目录。旧目录迁移可使用
`pose_bundle_import --from-dir /path/to/old/poses --default-side right`，不会自动扫描用户目录。

### ROS 服务

可选的 `pose_library_server` 提供以下服务，类型均属于 `hand_gesture_primitives_msgs/srv`：

| `/{side}_hand/` 下的服务名 | 类型 |
| --- | --- |
| `save_pose` | `SavePose` |
| `list_poses` | `ListPoses` |
| `delete_pose` | `DeletePose` |
| `import_gui_presets` | `ImportGuiPresets` |
| `export_pose_bundle` | `ExportPoseBundle` |
| `import_pose_bundle` | `ImportPoseBundle` |
| `export_primitive_poses` | `ExportPrimitivePoses` |

`SavePose.positions` 为空时取最近反馈；非空 `keyframe` 用于新增/替换指定帧。
准确请求字段见仓库 [srv 定义](../../messages/hand_gesture_primitives_msgs/srv)。
网页直接访问共享姿态库，不要求启动该独立服务。

`gesture_node` 提供 `/{side}_hand/plan_gesture`（`PlanGesture`），只生成名义轨迹，不发送控制命令。
请求包含 `primitive` 和 `sample_period`；可规划的静态原语由 `PLANABLE_STATIC_PRIMITIVES` 决定。
自定义姿态请求接受 `pose <name>`、`pose:<name>`、`pose/<name>` 或 `named_pose:<name>`；
响应包含控制器关节名、采样时间、展平的位置数组和可选 `segments_json`。视觉/力控闭环不属于名义规划。

## 网页面板与 HTTP

独立 Mock 使用 `teach_panel --mock`；ROS 模式使用安装指南里的节点与 URDF 配置。
直接启动 ROS 面板默认 `http_port=8001`，不因右手自动变为 8002；多实例需显式设置不同端口。
`teach_panel.launch.py` 默认 `launch_gesture_node=false`、`launch_pose_server=true`，仅需网页时
可设 `launch_pose_server:=false`。面板必须和原语使用同一姿态目录与手型/侧别配置。

| 入口 | 用途 |
| --- | --- |
| `/`、`/legacy`、`/workbench` | 默认示教页、兼容入口、扩展工作台 |
| `/ws?side=right` | 默认页的 WebSocket 交互 |
| `GET /api/workbench/state` | 后端、默认侧、双手状态、速度及编辑暂停状态 |
| `GET /api/workbench/catalog?side=right` | 原语支持与可用条件 |
| `GET /api/workbench/library?side=right` | 当前侧姿态库 |
| `GET /api/pose_bundle/export?side=right` | 导出所选侧的整个已保存库 |
| `POST /api/pose_bundle/import` | 姿态包导入，支持 side、strategy、verify_checksum 查询参数 |
| `POST /api/pose_bundle/import_factory` | 静态终态预设，支持 side、overwrite 查询参数 |

默认页的滑条按产品布局显示 0–255，并可切换显示单位；工作台 `PoseDraft.positions`
使用控制器弧度。因此不能把两种页面的数值数组直接互换。

## GraspGate

`launch_gate:=true` 时，用户命令仍发到 `gesture_cmd`，gate 转发到 `gesture_cmd_exec`。
目录中 11 个 `gated` 视觉原语执行 H/XY/Fwd 条件检查；`index_ring_by_vision` 和
`large_wrap_by_vision` 为目录明确的直通例外。直通不代表不需要感知或反馈。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `palm_z` / `tip_z` | `0.06` / `0.18` m | 手在 TCP 前方的可达范围 |
| `half_w` / `half_h` | `0.20` / `0.15` m | 掌面半宽与半高 |
| `height_tol` | `0.05` m | 高度容差 |
| `data_timeout` | `2.0` s | 输入新鲜度 |
| `de_bounce` | `3` | 连续通过检查次数 |

H 检查物体与手的世界 Z 范围相交；XY 检查目标在掌面范围；Fwd 要求手坐标系 z>0。
数据还必须采用一致坐标系。需要门控的原语名单由 `primitive_catalog.gated_primitive_names()` 给出。

## joint_state_bridge 与模型显示

此节点输出 mesh 模型关节状态，**默认输出 `/mesh_joint_states`**，不能当成驱动反馈源。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `hand_side` | `left` | 侧别 |
| `hand_joint` | `L25` | 模型与手型 |
| `state_source` | `auto` | 当前六种配置手型默认读取 ros2_control 的关节反馈 |
| `state_topic` | 空 | joint_states 模式为 `/joint_states`，legacy_cb 模式为 `/cb_{side}_hand_state` |
| `output_topic` | `/mesh_joint_states` | 可视化关节状态 |
| `robot_description` | 空 | 控制器 URDF，供弧度反馈映射 |
| `urdf_path` | 空 | FK/可视化几何模型路径 |

所需几何模型见[模型说明](../../data/urdf/README.md)。查看随包 O6 右手模型的示例：

```bash
ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
  mode:=gui hand_model:=O6 hand_joint:=O6 hand_side:=right
```

此示例启动可视化滑条与模型，不启动机械手驱动。其他手型需要另外提供对应 URDF 和 mesh。
