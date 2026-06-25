# Perception 外发 Topic 说明

本文档描述 `robot_perception` 包各节点对外发布的 ROS2 Topic、消息格式、坐标系约定及下游集成建议。

**默认相机命名空间：** `{cam_ns} = /camera_head`（可通过 launch 参数 `cam_ns` 修改）

**消息包：** `robot_perception_msgs`（源码位于仓库 `messages/robot_perception_msgs/`）

---

## Topic 总表

以下示例以默认 `{cam_ns}=/camera_head` 为准。

**发布状态图例：** ✅ 始终 = 有检测结果时每帧发布（可能为空数组）；✅ 默认开 = 默认 launch 配置下发布；⚠️ = 有条件或默认无内容；❌ = 需显式开启 launch 参数。

### detection_bbox

| 完整 Topic 示例 | 消息类型 | 默认是否发布 | 说明 |
|----------------|----------|-------------|------|
| `/camera_head/detection_bbox/bboxes_2d` | `robot_perception_msgs/LabeledBBox2DArray` | ✅ 始终 | 2D 检测框：`bbox_xyxy` + `label` + `prompt` + `score` + `instance_id`；`header.frame_id` 为相机 optical frame |
| `/camera_head/detection_bbox/bboxes_3d` | `robot_perception_msgs/LabeledBBox3DArray` | ✅ 始终 | 3D AABB：`center`/`size`/`min`/`max` + `orientation` + `top_normal` + `grasp_type` + `track_mode`；默认仅 world 系（`publish_workbench_bbox:=true`）；**无标定时 boxes 为空** |
| `/camera_head/detection_bbox/surface_meshes` | `robot_perception_msgs/SurfaceMeshArray` | ✅ 默认开 | 结构化表面 mesh（`vertices`/`faces`）；默认 hybrid 顶面重建；`use_pixel3d:=true` 成功后替换为完整 mesh；`frame_id=world` |
| `/camera_head/detection_bbox/workbench_plane` | `robot_perception_msgs/WorkbenchPlane` | ✅ 默认开 | RANSAC 台面平面：`normal`/`d`/`centroid`/`estimated_z`/`tilt_deg`；需 `enable_ransac_workbench_plane:=true`（默认）+ 标定 |
| `/camera_head/detection_bbox/annotated` | `sensor_msgs/Image` | ✅ 默认开 | 彩色图叠加 2D 框、mask、可选 3D 投影；`always_publish_annotated:=true`（默认）或有订阅者时发布 |
| `/camera_head/detection_bbox/markers_workbench` | `visualization_msgs/MarkerArray` | ✅ 默认开 | RViz 用，world 系 3D 线框/立方体；需 `publish_rviz_markers:=true`（默认） |
| `/camera_head/detection_bbox/markers_surface` | `visualization_msgs/MarkerArray` | ✅ 默认开 | RViz 用，物体表面 mesh（hybrid 或 Pixel3D）；需 `publish_hybrid_surface_marker:=true`（默认） |
| `/camera_head/detection_bbox/markers_workbench_plane` | `visualization_msgs/MarkerArray` | ✅ 默认开 | RViz 用，RANSAC 台面半透明平面；需 `publish_workbench_plane_marker:=true`（默认） |
| `/camera_head/detection_bbox/markers_camera` | `visualization_msgs/MarkerArray` | ⚠️ 默认空 | RViz 用，相机系 3D 框；publisher 默认创建，**有内容需** `publish_camera_bbox:=true`（默认 false） |
| `/camera_head/detection_bbox/object_points` | `sensor_msgs/PointCloud2` | ❌ 需显式开启 | 分割深度点云，字段含 `x/y/z` + `instance_id`（相机系）；`publish_object_pointcloud:=true` |
| `/camera_head/detection_bbox/mask` | `sensor_msgs/Image` | ❌ 需显式开启 | 实例分割 mask 图像；`publish_mask_image:=true` |
| `/camera_head/detection_bbox/track_debug` | `std_msgs/String` | ❌ 需显式开启 | JSON 跟踪状态；`publish_track_debug:=true` |

**TF（非 Topic）：** `publish_static_tf:=true`（默认）且有标定时，广播 `world → camera_head_color_optical_frame`。

---

## 1. 节点与外发关系

> 完整 Topic 列表见上文 **[Topic 总表](#topic-总表)**。

| 节点 | 是否直接外发 | 说明 |
|------|-------------|------|
| `detection_bbox` | ✅ | 2D/3D AABB、表面 mesh、RViz markers |

节点输入：

| 输入 | 类型 | 说明 |
|------|------|------|
| `{cam_ns}/color/image_raw` | `sensor_msgs/Image` | 彩色图 |
| `{cam_ns}/depth/image_raw` | `sensor_msgs/Image` | 深度图 |
| `{cam_ns}/color/camera_info` | `sensor_msgs/CameraInfo` | 相机内参 |
| `/robot_perception/detection_prompts` | `ros_gz_interfaces/StringVec` | 动态检测目标（tag 或自由文本 prompt） |

---

## 2. detection_bbox 补充说明

**Topic 前缀：** `{cam_ns}/detection_bbox/`

### 2.1 bboxes_3d 发布逻辑

受 launch 参数控制：

| 参数 | 默认值 | 效果 |
|------|--------|------|
| `publish_workbench_bbox` | `true` | 向 `bboxes_3d` 追加 **world 系** AABB（`aabb_work`） |
| `publish_camera_bbox` | `false` | 是否追加 **相机系** AABB（`aabb_cam`） |

**默认行为：** `bboxes_3d` 仅含 world 系 3D 框。无手眼标定（`calib_file`）时 3D 为空，仅有 2D 结果。

**注意：** 当 `publish_camera_bbox` 与 `publish_workbench_bbox` 同时为 `true` 时，同一 `LabeledBBox3DArray` 内可能包含不同 `frame_id` 的条目。下游须读取每个 `LabeledBBox3D.frame_id`，不能仅依赖 `header.frame_id`。

`header.frame_id` 规则：

- `publish_workbench_bbox=true` → `header.frame_id = workbench_frame_id`（默认 `world`）
- 否则 → `header.frame_id = 相机 optical frame`

### 2.2 TF 广播（非 Topic）

当 `publish_static_tf:=true` 且加载手眼标定（`calib_file`）后，节点广播静态变换：

```
{workbench_frame_id}  →  {cam_frame}
默认: world → camera_head_color_optical_frame
```

RViz Fixed Frame 应设为 `world`（或 `workbench_frame_id`）。

---

## 3. 消息字段定义

### 3.1 LabeledBBox2D / LabeledBBox2DArray

```
# LabeledBBox2D
string label
string prompt
float32 score
float32[4] bbox_xyxy    # 像素坐标 [x1, y1, x2, y2]
uint32 instance_id      # 跟踪 ID，跨帧稳定（启用 temporal tracking 时）

# LabeledBBox2DArray
std_msgs/Header header
LabeledBBox2D[] boxes
```

### 3.2 LabeledBBox3D / LabeledBBox3DArray

```
# LabeledBBox3D
string label
string prompt
float32 score
uint32 instance_id
string frame_id
float32[3] center       # AABB 中心
float32[3] size         # AABB 尺寸 (dx, dy, dz)
float32[3] min          # AABB 最小角
float32[3] max          # AABB 最大角
float32[4] orientation  # PCA 主轴四元数 (x, y, z, w)；不可用时为单位四元数
float32[3] top_normal   # 顶面法向量（抓取接近方向参考）
string grasp_type       # 抓取类型提示: precision / lateral / power / ""
string track_mode       # 跟踪模式: fast_track / refined / recovered / reinitialized / ""

# LabeledBBox3DArray
std_msgs/Header header
LabeledBBox3D[] boxes
```

### 3.3 SurfaceMesh / SurfaceMeshArray

```
# SurfaceMesh
string label
uint32 instance_id
string frame_id
uint32 num_vertices
uint32 num_faces
float32[] vertices      # N×3 展平: [x0,y0,z0, x1,y1,z1, ...]
uint32[] faces          # M×3 展平: [i0,j0,k0, i1,j1,k1, ...]

# SurfaceMeshArray
std_msgs/Header header
SurfaceMesh[] meshes
```

**mesh 来源：**

- 默认：hybrid surface mesh（由 mask 深度点重建的可见顶面，非完整补全）
- 启用 Pixel3D（`use_pixel3d:=true`）且推理成功后：替换为完整补全 mesh
- 触发条件：track 稳定（`pixel3d_trigger_age` 帧）、`aabb_work` 非空且在 world ROI 内

### 3.4 WorkbenchPlane

```
std_msgs/Header header
bool valid
float64[3] normal       # 平面法向量（world 系）
float64 d               # 平面方程 normal·x + d = 0
float64[3] centroid     # 内点质心
float64 estimated_z     # 估计台面高度
float64 tilt_deg        # 相对水平面倾角
uint32 inlier_count     # RANSAC 内点数
```

### 3.5 object_points（PointCloud2 字段）

| 字段 | 类型 | 说明 |
|------|------|------|
| `x`, `y`, `z` | float32 | 相机系 3D 坐标 |
| `instance_id` | uint32 | 与 `LabeledBBox2D.instance_id` 对应 |

---

## 4. 下游集成建议

| 需求 | 推荐 Topic | 备注 |
|------|-----------|------|
| 开放词汇检测 + 跨帧跟踪 | `bboxes_2d` + `bboxes_3d` | 用 `instance_id` 关联同一物体 |
| 粗抓取 / 放置规划 | `bboxes_3d` | 关注 `top_normal`、`grasp_type`、`center` |
| 碰撞检测 / 精细 mesh | `surface_meshes` | 等待 Pixel3D 完成后再消费；检查 `num_faces > 0` |
| 台面约束 / 放置区域 | `workbench_plane` | 配合 `enable_world_roi_filter` 使用 |
| RViz 调试 | `markers_*`、`annotated` | Fixed Frame 设为 `world` |

**实例关联键：** 优先使用 `instance_id`（tracking 生命周期内不变）

---

## 5. 数据流概览

```
相机 Color + Depth
        │
        ▼
  detection_bbox ──► bboxes_2d / bboxes_3d / surface_meshes / workbench_plane
        │              markers_* / annotated（可视化）
        │
        ▼（可选 Pixel3D 异步）
  surface_meshes 更新为完整 mesh
```

---

## 6. 常用调试命令

```bash
# 列出 detection_bbox 外发 topic
ros2 topic list | grep detection_bbox

# 查看 3D 检测框
ros2 topic echo /camera_head/detection_bbox/bboxes_3d --once

# 查看 surface mesh
ros2 topic echo /camera_head/detection_bbox/surface_meshes --once

# 查看 topic 类型与频率
ros2 topic info /camera_head/detection_bbox/bboxes_2d -v
```

---

## 7. 相关 Launch 参数速查

| 参数 | 默认值 | 影响的输出 |
|------|--------|-----------|
| `cam_ns` | `['/camera_head']` | 所有 topic 前缀 |
| `workbench_frame_id` | `world` | 3D bbox / mesh / plane 坐标系 |
| `publish_workbench_bbox` | `true` | `bboxes_3d` 是否含 world 系框 |
| `publish_camera_bbox` | `false` | `bboxes_3d` / `markers_camera` 是否含相机系框 |
| `publish_surface_mesh_topic` | `true` | 是否发布 `surface_meshes` |
| `publish_rviz_markers` | `true` | 是否发布 `markers_*` |
| `publish_object_pointcloud` | `false` | 是否发布 `object_points` |
| `publish_mask_image` | `false` | 是否发布 `mask` |
| `publish_track_debug` | `false` | 是否发布 `track_debug` |
| `enable_ransac_workbench_plane` | `true` | 是否发布 `workbench_plane` + `markers_workbench_plane` |
| `use_pixel3d` | `false` | 是否异步生成完整 mesh 并更新 `surface_meshes` |
| `calib_file` | 自动查找 | 无标定时, 3D / world 输出不可用 |

完整参数说明见 [readme.md](../readme.md)。
