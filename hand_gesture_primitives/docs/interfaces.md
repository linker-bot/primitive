# 接口文档

本文档描述 `hand_gesture_primitives` 包对外暴露的 ROS2 接口：Topic、参数、消息格式与节点关系。

---

## 节点概览

| 节点名 | 可执行文件 | 功能 |
|--------|-----------|------|
| `hand_gesture_node` | `gesture_node` | 指令解析 + 10Hz 原语执行器 |
| `grasp_gate` | `grasp_gate` | 抓取可行性三条件门控（可选） |
| `joint_state_bridge` | `joint_state_bridge` | 硬件反馈 → URDF 弧度转换（RViz 可视化） |
| `mock_perception` | `mock_perception` | 模拟感知输出（测试用） |

---

## 数据流

```
                    ┌─────────────────────────────┐
                    │  外部感知 (robot_perception)  │
                    │  bboxes_3d / labeled_poses   │
                    └──────────────┬──────────────┘
                                   │
/hand_gesture_cmd ──┐              │
                    ▼              ▼
              ┌──────────┐   ┌──────────────────────────┐
              │GraspGate │──▶│ hand_gesture_node        │
              │(可选门控) │   │  GestureNode + Executor  │
              └──────────┘   └────────────┬─────────────┘
                                          │
                    ┌─────────────────────┼─────────────────────────┐
                    ▼                     ▼                         ▼
    /cb_{side}_hand_control_cmd   /hand_gesture_status   /hand_gesture/result
    (JointState, 10Hz)            (String, 0.5Hz)        (String, ~1Hz)
```

---

## hand_gesture_node

### 订阅 Topic

| Topic | 消息类型 | 频率 | 说明 |
|-------|---------|------|------|
| `/hand_gesture_cmd_exec` | `std_msgs/String` | 按需 | 手势指令输入（可通过参数 `cmd_topic` 自定义） |
| `/cb_{side}_hand_state` | `sensor_msgs/JointState` | ~50Hz | 手部关节角度反馈（0–255） |
| `/cb_{side}_hand_info` | `std_msgs/String` | ~10Hz | 手部信息 JSON（含 `current_current` 电流数组） |
| `/tcp_pose` | `geometry_msgs/PoseStamped` | ~30Hz | 机械臂末端 TCP 位姿 |
| `/object_pose` | `geometry_msgs/PoseStamped` | ~5Hz | 目标物体位姿（**world frame**；默认由 GraspGate 从 `bboxes_3d.center` 填充） |
| `bboxes_3d` | `robot_perception_msgs/LabeledBBox3DArray` | ~5Hz | 感知 3D 包围盒（含 size/grasp_type） |
| `labeled_poses` | `robot_perception_msgs/LabeledPoseArray` | 可选 | 感知物体位姿数组（world frame）；**detection_bbox 未发布**，需外部节点或未来扩展 |
| `/cb_{side}_hand_force` | `std_msgs/Float32MultiArray` | ~10Hz | 单点触觉压感（前 5 元素为各指法向力） |
| `/cb_{side}_hand_normal_force` | `std_msgs/Float32MultiArray` | ~10Hz | SDK 法向力（每指一个值） |
| `/cb_{side}_hand_matrix_touch_mass` | `std_msgs/String` | ~10Hz | 矩阵压感 JSON（每指总压力质量 g） |
| `/cb_{side}_hand_motor_torque` | `std_msgs/Float32MultiArray` | ~10Hz | 关节力矩反馈 |
| `/grasp_gate/selected_label` | `std_msgs/String` | 按需 | 门控选中的物体 label |

> `{side}` = `left` 或 `right`，由参数 `hand_side` 决定。
> `robot_perception_msgs` 为可选依赖，缺失时自动降级（仅使用 `/object_pose`）。

### 发布 Topic

| Topic | 消息类型 | 频率 | 说明 |
|-------|---------|------|------|
| `/cb_{side}_hand_control_cmd` | `sensor_msgs/JointState` | 10Hz | 目标关节角度（0–255，始终发送） |
| `/hand_gesture_status` | `std_msgs/String` | 0.5Hz | 当前状态：`原语名\|grasp=状态\|type=...\|suggest=...` |
| `/hand_gesture/result` | `std_msgs/String` | ~1Hz | 执行状态文本：`IDLE` / `EXECUTING: name` / `DONE: name` |
| `/cb_{side}_hand_fingertip_positions` | `std_msgs/Float32MultiArray` | 10Hz | FK 指尖位置 [5×3] xyz (m)，手基坐标系 |
| `/cb_{side}_hand_fingertip_markers` | `visualization_msgs/MarkerArray` | 10Hz | RViz 指尖球体 marker（5 个彩色球体） |

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hand_side` | string | `"left"` | 手侧：`left` / `right` |
| `hand_type` | string | `"o20"` | 手型：`o20`（20-DOF）/ `l25`（25-DOF） |
| `cmd_topic` | string | `"/hand_gesture_cmd_exec"` | 指令输入 topic |
| `tcp_pose_topic` | string | `"/tcp_pose"` | TCP 位姿 topic |
| `object_pose_topic` | string | `"/object_pose"` | 物体位姿 topic（world frame，通常由 GraspGate 发布） |
| `bboxes_3d_topic` | string | `"/camera_head/detection_bbox/bboxes_3d"` | 3D 包围盒 topic |
| `labeled_pose_topic` | string | `"/camera_head/perception/labeled_poses"` | 感知位姿数组 topic（可选；detection_bbox 未发布） |
| `target_object_label` | string | `""` | 目标物体 label（空=自动选最高 score；非空须与 `bboxes_3d.label` 一致） |
| `result_topic` | string | `"/hand_gesture/result"` | 执行结果 topic |
| `selected_label_topic` | string | `"/grasp_gate/selected_label"` | 门控选中 label topic |
| `reach_check_strict` | bool | `false` | 严格模式：不可达时拒绝执行 |
| `reach_max_distance` | float | `0.15` | TCP→目标最大允许距离 (m) |
| `palm_offset_x` | float | `0.0` | 掌心相对 TCP 偏移 X (m) |
| `palm_offset_y` | float | `0.0` | 掌心相对 TCP 偏移 Y (m) |
| `palm_offset_z` | float | `-0.05` | 掌心相对 TCP 偏移 Z (m) |
| `publish_fingertips` | bool | `true` | 是否发布 FK 指尖位置 |
| `fingertip_frame` | string | `""` | 指尖 marker frame_id（空=自动） |
| `urdf_path` | string | `""` | URDF 路径（空=自动搜索） |
| `contact_config_path` | string | `""` | 接触/电流阈值 YAML（空=包内 `config/contact_thresholds.yaml`） |
| `contact_pressure_threshold` | float | `20.0` | 触觉压感接触阈值 (0–255) |
| `contact_mass_threshold` | float | `2.0` | 矩阵触觉质量阈值 (g) |
| `contact_current_delta` | float | `270` | 无触觉时电流增量接触阈值 (mA) |
| `contact_current_delta_narrow` | float | `250` | 窄物体力控原语电流增量 (mA) |
| `contact_current_settle_frames` | int | `10` | 电流基线稳定帧数 (@10Hz) |
| `hold_safe_current` | float | `800` | 力控持握安全电流上限 (mA) |
| `overload_current_threshold` | float | `1000` | 全局过载保护阈值 (mA) |
| `overload_duration_sec` | float | `2.0` | 过载保护持续时间 (s) |

默认阈值见安装目录 `share/hand_gesture_primitives/config/contact_thresholds.yaml`，可按现场负载修改 YAML 或通过 launch 参数覆盖。

---

## grasp_gate

### 订阅 Topic

| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/hand_gesture_cmd` | `std_msgs/String` | 用户原始手势指令 |
| `/tcp_pose` | `geometry_msgs/PoseStamped` | TCP 位姿 |
| `bbox_topic` | `robot_perception_msgs/LabeledBBox3DArray` | 3D 包围盒 |
| `labeled_pose_topic` | `robot_perception_msgs/LabeledPoseArray` | 感知位姿（备选） |

### 发布 Topic

| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/hand_gesture_cmd_exec` | `std_msgs/String` | 过滤后的指令（转发给 gesture_node） |
| `/grasp_gate/status` | `std_msgs/String` | 门控判定结果：`OK: cmd, 'label' fwd=Xm` / `REJECT: cmd, reason` |
| `/object_pose` | `geometry_msgs/PoseStamped` | 选中物体位姿（world frame，供 gesture_node 可达性检查） |

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tcp_pose_topic` | string | `"/tcp_pose"` | TCP 位姿 topic |
| `bbox_topic` | string | `"/camera_head/detection_bbox/bboxes_3d"` | 3D 包围盒 topic |
| `labeled_pose_topic` | string | `"/camera_head/perception/labeled_poses"` | 感知位姿 topic（可选 fallback；detection_bbox 未发布） |
| `input_cmd_topic` | string | `"/hand_gesture_cmd"` | 原始指令输入 |
| `output_cmd_topic` | string | `"/hand_gesture_cmd_exec"` | 过滤后指令输出 |
| `status_topic` | string | `"/grasp_gate/status"` | 状态输出 topic |
| `selected_pose_topic` | string | `"/object_pose"` | 选中物体位姿输出 |
| `palm_z` | float | `0.06` | TCP → 掌心 Z 偏移 (m) |
| `tip_z` | float | `0.18` | TCP → 指尖 Z 偏移 (m) |
| `half_w` | float | `0.20` | 掌面左右半宽 (m) |
| `half_h` | float | `0.15` | 掌面上下半高 (m) |
| `height_tol` | float | `0.05` | 高度 Z 容差 (m) |
| `data_timeout` | float | `2.0` | 数据过期时间 (s) |
| `de_bounce` | int | `3` | 连续通过帧数阈值 |

### 门控逻辑

对 **11 个**包络/定位原语执行三条件判定，其余指令直接转发：

`ring_by_vision` `small_warp_by_vision` `no_index_warp_by_vision` `middle_ring_by_vision` `hook_by_vision` `tripod_by_vision` `palmar_by_vision` `parallel_extension_by_vision` `index_pinch_by_vision` `middle_pinch_by_vision` `disk_by_vision`

| 条件 | 判定方式 | 通过条件 |
|------|---------|---------|
| **H** (高度) | 物体世界 Z 范围 ∩ 手世界 Z 可达范围 | 有交叠 |
| **XY** (掌面) | 物体中心在手坐标系掌面矩形内 | \|cx\| < half_w 且 \|cy\| < half_h |
| **Fwd** (前方) | 物体在手坐标系中的 Z 坐标 | cz > 0 |

坐标系转换：世界 → TCP → R_z(90°) → 手坐标系

```
hand +X = -TCP_Y (掌面左右)
hand +Y = +TCP_X (掌面上下)
hand +Z = +TCP_Z (手指前方)
```

---

## joint_state_bridge

将手部硬件 O20/L25 格式反馈转为 URDF 弧度，供 `robot_state_publisher` 驱动 RViz 模型。

### 订阅 Topic

| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/cb_{side}_hand_state` | `sensor_msgs/JointState` | 硬件关节反馈 |

### 发布 Topic

| Topic | 消息类型 | 说明 |
|-------|---------|------|
| `/joint_states` | `sensor_msgs/JointState` | URDF 弧度关节状态 |

### 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `hand_side` | string | `"left"` | 手侧 |
| `hand_joint` | string | `"L25"` | 手型标识 |
| `urdf_path` | string | `""` | URDF 路径（空=自动搜索） |

---

## 指令协议

通过 `/hand_gesture_cmd`（或 `cmd_topic`）接收 `std_msgs/String`，格式：

```
<command> [args...]
```

### 指令列表

| 指令 | 格式 | 说明 |
|------|------|------|
| `<原语名>` | `pinch` / `open` / `fist` / `tripod_by_vision` ... | 执行对应原语（共 24 个） |
| `<原语名> <label>` | `index_ring_by_vision screwdriver` | 执行原语 + 设置目标 label 过滤 |
| `<原语名> prep` | `thumb_adduction_grip prep` | 仅执行预备阶段（侧向夹持原语） |
| `<原语名> close` | `thumb_adduction_grip close` | 执行闭合阶段 |
| `auto` | `auto` / `auto [label]` / `auto [label] prep` | 根据感知 grasp_type 自动选择原语 |
| `target <label>` | `target screwdriver` | 设置目标物体 label 过滤（不执行） |
| `target_id <id>` | `target_id 3` | 按 instance_id 过滤（0=清除） |
| `stop` | `stop` | 停止当前原语，保持最终姿态 |

### 可用原语名（24 个）

| 类别 | 指令 |
|------|------|
| 安全/复位 | `init` `open` `release` `relax_grip` |
| 固定手势 | `fist` `pinch` `point` `ok_sign` `v_sign` |
| 精细捏取 | `index_pinch_by_vision` `middle_pinch_by_vision` |
| 感知自适应 | `index_ring_by_vision` `large_wrap_by_vision` |
| 侧向夹持 | `thumb_adduction_grip` `index_middle_adduction_grip` |
| 力控包络 | `middle_ring_by_vision` `ring_by_vision` `small_warp_by_vision` `no_index_warp_by_vision` `hook_by_vision` `tripod_by_vision` `palmar_by_vision` |
| 定位捏取/包络 | `index_pinch_by_vision` `middle_pinch_by_vision` `parallel_extension_by_vision` `disk_by_vision` |

### 状态反馈格式

`/hand_gesture_status` 发布格式（`|` 分隔）：

```
<active_primitive>|grasp=<grasp_state>
<active_primitive>|type=<grasp_type>|suggest=<suggested_primitive>
rejected:<reason>
```

示例：
```
index_ring_by_vision|grasp=progressive
hold|type=precision|suggest=index_ring_by_vision
rejected:too_far:0.312m>(max:0.150m)
```

`/hand_gesture/result` 发布格式：

```
IDLE                              # 无活跃原语
EXECUTING: <name>                 # 正在执行
EXECUTING: <name>, obj=<label>    # 执行中，有目标物体
DONE: <name>                      # 原语完成
```

---

## 消息格式详解

### JointState 控制指令

发布到 `/cb_{side}_hand_control_cmd`：

```yaml
header:
  stamp: <当前时间>
name: [thumb_base, index_base, middle_base, ring_base, pinky_base,
       thumb_abd, index_abd, middle_abd, ring_abd, pinky_abd,
       thumb_rot, reserved_11, reserved_12, reserved_13, reserved_14,
       thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
position: [0.0 ~ 255.0] × 20    # O20: 0=伸直, 255=弯曲 (L25 反转)
velocity: [0.0] × 20
effort: [0.0] × 20
```

### hand_info JSON 格式

订阅 `/cb_{side}_hand_info`（驱动端发布）：

```json
{
  "current_current": [120, 85, 92, 78, 65, 150, 0, 0, 0, 0, 200, 0, 0, 0, 0, 95, 88, 72, 60, 55],
  "temperature": [35, 34, 36, ...]
}
```

`current_current` 数组长度 = num_joints，单位 mA。

### Float32MultiArray 指尖位置

发布到 `/cb_{side}_hand_fingertip_positions`：

```yaml
layout:
  dim:
    - label: "finger"
      size: 5
      stride: 15
    - label: "xyz"
      size: 3
      stride: 3
data: [x0, y0, z0, x1, y1, z1, ..., x4, y4, z4]  # 15 floats
# 顺序: thumb, index, middle, ring, pinky
# 坐标系: hand_base_link, 单位: 米
```

### matrix_touch_mass JSON 格式

订阅 `/cb_{side}_hand_matrix_touch_mass`：

```json
{
  "thumb_mass": 12.5,
  "index_mass": 8.3,
  "middle_mass": 0.0,
  "ring_mass": 0.0,
  "little_mass": 0.0
}
```

单位：克 (g)。作为触觉信号的 fallback 来源（仅在 `hand_force` 无数据时使用）。

---

## Launch 使用

### 基础启动

```bash
ros2 launch hand_gesture_primitives gesture_node.launch.py \
  hand_side:=left hand_type:=o20
```

### 启用门控

```bash
ros2 launch hand_gesture_primitives gesture_node.launch.py \
  hand_side:=left launch_gate:=true
```

### 带 Mock 感知测试

```bash
ros2 launch hand_gesture_primitives test_with_mock.launch.py \
  hand_side:=left shape:=cylinder
```

### RViz 手模型可视化

```bash
ros2 launch hand_gesture_primitives view_hand_urdf.launch.py \
  hand_side:=left hand_joint:=L25
```

---

## 典型集成拓扑

### 最小配置（无感知）

```
LinkerHand 驱动
  ├── /cb_left_hand_state ──────────────▶ hand_gesture_node
  ◀── /cb_left_hand_control_cmd ────────┘
```

仅支持安全/固定类原语（`init`、`open`、`fist`、`pinch` 等）。

### 标准配置（感知 + 门控）

```
robot_perception                    机械臂驱动
  ├── bboxes_3d ──▶ grasp_gate ◀── /tcp_pose
  │                     │
  │                     ├── /object_pose (world)
  │                     └── /hand_gesture_cmd_exec
  │              hand_gesture_node
  │                     │
  │                     ▼
  │              LinkerHand 驱动
  │
  └── bboxes_3d ──▶ hand_gesture_node (object_size / grasp_type)
```

支持全部 24 个原语：自适应闭合、力控包络、auto 映射等。

### 最小感知配置（无门控）

```bash
# 无门控时，cmd_topic 直接订阅用户 topic
ros2 launch hand_gesture_primitives gesture_node.launch.py \
  hand_side:=left cmd_topic:=/hand_gesture_cmd
```

跳过门控，所有指令直达 gesture_node。包络原语不经可行性过滤。

---

## 保护机制

| 机制 | 触发条件 | 行为 |
|------|---------|------|
| 电流过载保护 | 任一关节电流 > `overload_current_threshold` 持续 > `overload_duration_sec` | 自动执行 `init`（全手复位） |
| 可达性检查 | TCP→目标距离 > `reach_max_distance` | 严格模式拒绝 / 非严格模式警告 |
| infeasible 保持 | 原语返回 `feasible=false` | 保持上帧目标继续 10Hz 发送 |
| 感知超时 | bbox 连续 15 帧无更新 (~3s) | 清除物体缓存（不影响当前原语执行） |
| 门控去抖 | 连续 `de_bounce` 帧三条件通过 | 才转发指令（防抖动，仅 11 个门控原语） |

---

## 与 LinkerHand 驱动的对接

本包依赖 LinkerHand ROS2 SDK 提供的以下 Topic：

| 驱动 Topic | 方向 | 说明 |
|-----------|------|------|
| `/cb_{side}_hand_control_cmd` | 本包 → 驱动 | 20/25-DOF 目标角度 |
| `/cb_{side}_hand_state` | 驱动 → 本包 | 关节角度反馈 |
| `/cb_{side}_hand_info` | 驱动 → 本包 | 电流/温度 JSON |
| `/cb_{side}_hand_force` | 驱动 → 本包 | 触觉压感 |
| `/cb_{side}_hand_normal_force` | 驱动 → 本包 | 法向力 |
| `/cb_{side}_hand_matrix_touch_mass` | 驱动 → 本包 | 矩阵压感合值 |
| `/cb_{side}_hand_motor_torque` | 驱动 → 本包 | 关节力矩 |

兼容 LinkerHand O20 和 L25 驱动，通过 `hand_type` 参数自动适配关节数量与角度方向。
