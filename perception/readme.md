# robot_perception

可选的 ROS 2 Jazzy 感知包。`detection_bbox` 接收 RGB、深度和相机内参，输出带标签的
2D/3D 包围盒、跟踪结果、表面网格和可视化信息。普通手势与网页示教不需要安装此包。

## 文档入口

- [环境、模型下载、编译与启动](../docs/perception-setup.md)：唯一的安装步骤入口。
- [输入/输出话题与消息字段](docs/output_topics.md)：下游集成接口。
- [手势 ROS 接口](../hand_gesture_primitives/docs/interfaces.md)：如何提供目标、TCP 和反馈。

`environment.yml`、`environment_vlm.yml` 是环境快照，不能直接视为任意机器的安装锁。
仓内部分旧脚本依赖未定义的环境变量；请使用部署指南中的明确路径与解释器命令。

## 功能组合

| 组合 | 作用 | 额外条件 |
| --- | --- | --- |
| GDINO + SAM2 | 默认的文本目标检测与分割 | GDINO/BERT/SAM2 权重与 CUDA 扩展 |
| Cutie 跟踪 | 传播已有目标 mask，配合周期检测更新 | 跟踪权重与相应缓存；默认开启 |
| VLM 检测 + SAM2 | 用 VLM 替代 GDINO 发现目标 | 独立、兼容的 VLM HTTP 服务 |
| VLM 场景理解 | 生成/更新场景 prompt，可与 GDINO 或 VLM 检测组合 | VLM 服务；不要求 `use_vlm_detect=true` |
| hybrid surface | 从可见深度点构造表面 mesh | 有效深度、标定及相应发布配置 |
| Pixal3D | 可选的完整 mesh 补全 | 额外源码、权重、GPU 扩展与稳定 track |
| FoundationStereo | 可选的立体深度后端 | 额外推理实现、匹配设备的 TensorRT engine 和 IR 输入 |

显存需求取决于分辨率、模型组合、量化后端与同卡并发任务；`pixel3d_low_vram=true`
不保证任意 GPU 都能运行。异步推理仍会竞争 GPU 资源，不保证检测帧率完全不受影响。

## 路径与输入

- `ROBOT_PERCEPTION_DIR`：第三方源码根，包含随仓库提供的 `Grounded-SAM-2/` 和 `Cutie/`。
  可选 `Pixal3D/` 源码需按部署指南另行获取。
- `PERCEPTION_WEIGHTS_DIR`：模型权重根。应设为已存在的绝对路径，区别于源码目录。
- `calib_file`：本机真实标定文件；仓库中的样例结果不能替代设备标定。
- 默认相机前缀为 `/camera_head`，输入为 `color/image_raw`、`depth/image_raw`、`color/camera_info`。
- 相机、感知、原语和 TCP 发布者需要可互相发现的 ROS 配置以及一致的坐标系与时间基准。

权重下载的默认暂存位置因脚本而异。按部署指南显式指定目标位置，避免将脚本的下载
暂存目录误认为运行时权重目录。

## 启动与目标选择

先完成部署指南中的主感知环境、模型、编译和 ROS overlay 准备。在该环境中指定实际标定：

```bash
export GESTURE_CALIB_FILE=/absolute/path/to/your/full_calibration_result.txt
ros2 launch robot_perception detection_bbox.launch.py \
  cam_ns:="['/camera_head']" calib_file:="$GESTURE_CALIB_FILE" \
  use_vlm_detect:=false use_scene_understand:=false \
  use_pixel3d:=false use_stereo_depth:=false launch_rviz:=false
```

默认启用产线物体 prompt。自定义目标时，在上述命令中增加：

```text
auto_industry_scene_prompts:=false text_prompts:="['blue box.']"
```

也可使用 `tags`，取值以源码 `constants.py` 的标签映射为准。
`/robot_perception/detection_prompts` 接收 `ros_gz_interfaces/StringVec`，支持动态更新 prompt。
启用 VLM 或 Pixal3D 前完成部署指南里的额外步骤，不是只改一个 launch 参数。

## 常用 launch 参数

以下为 `launch/detection_bbox.launch.py` 的默认值，节点直接启动时应另外核对参数。

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `fps` | `5.0` | 检测频率上限 |
| `box_threshold` / `text_threshold` | `0.35` / `0.25` | GDINO 阈值 |
| `min_detection_score` | `0.45` | 检测结果过滤 |
| `sam2_score_min` | `0.7` | SAM2 分割分数阈值 |
| `use_temporal_tracking` / `use_cutie_tracking` | `true` / `true` | 时序与 Cutie 跟踪 |
| `track_lost_max_frames` | `5` | 丢失目标的保留帧数 |
| `use_vlm_detect` / `use_scene_understand` | `false` / `false` | VLM 检测与场景理解，分别控制 |
| `vlm_max_area_ratio` | `0.25` | VLM 检测框面积占比过滤 |
| `scene_change_threshold` | `0.15` | 场景变化比例阈值 |
| `use_pixel3d` / `use_stereo_depth` | `false` / `false` | 可选 mesh / 深度后端 |
| `pixel3d_low_vram` | `true` | 按阶段管理模型显存 |
| `publish_workbench_bbox` / `publish_camera_bbox` | `true` / `false` | world / 相机系包围盒 |
| `enable_world_roi_filter` | `true` | 世界坐标 ROI 过滤 |
| `workbench_frame_id` | `world` | 输出世界坐标系名 |
| `launch_rviz` | `false` | 是否启动可视化 |

输出开关和消息字段见[话题说明](docs/output_topics.md)。完整参数可在已安装环境查询：

```bash
ros2 launch robot_perception detection_bbox.launch.py --show-args
```

## 下游集成与排查

原语默认订阅 `/camera_head/detection_bbox/bboxes_3d`。感知目标可以左右手共享；TCP、
控制命令与手反馈按侧配置。`detection_bbox` 不发布 `/camera_head/perception/labeled_poses`，
这个兼容输入需由其他组件提供。

3D 输出是轴对齐包围盒；orientation 和 top_normal 是附加几何信息，不能把 center/size
直接解释成 OBB。默认只发布 world 系框，缺少标定或有效深度时，3D 数组可能为空。

| 现象 | 核对项 |
| --- | --- |
| 能检测 2D，3D 为空 | 标定、深度、相机内参、有效点数、world ROI |
| 找不到模型或扩展 | 源码与权重目录、当前 Python、GDINO `_C`、GPU/CUDA 兼容性 |
| VLM 无响应 | 服务 URL、模型标识、服务日志、显存及实际图像推理 |
| 目标 ID 经常变化 | 遮挡、检测质量与跟踪阈值；减小丢失保留帧数会更早删除 track |
| 没有 Pixal3D mesh | 依赖/权重、稳定 track、有效 world 几何、ROI 及推理错误日志 |
| RViz 无显示 | 实际消息、TF 以及 Fixed Frame 是否与 `workbench_frame_id` 一致 |

模型就绪或检测框出现，不代表机械手的 FK、接触反馈和门控已就绪。相关条件须在原语侧
继续核对，软件导入检查也不能替代目标设备上的模型推理验证。
