# robot_perception

ROS2 (Jazzy) 感知功能包，提供基于深度学习的物体检测、分割与位姿估计能力。

## 功能概览

本包提供一个核心节点：

| 节点 | 功能 | 输出 | 适用场景 |
|------|------|------|----------|
| `detection_bbox` | GDINO/VLM + SAM2 + 深度反投影 | 2D/3D AABB + 时序跟踪 | 开放词汇检测、产线工装识别 |

## 目录结构

```
perception/
├── docs/
│   └── output_topics.md        # 外发 Topic 说明文档
├── robot_perception/           # Python 源码
│   ├── detection_bbox_node.py  # 2D/3D AABB 检测节点
│   ├── constants.py            # 标签映射、默认 prompt 列表
│   └── utils/                  # 工具模块
│       ├── gdino_sam.py        # Grounding DINO + SAM2 封装
│       ├── bbox3d_from_depth.py # 深度反投影、AABB、PCA 方向
│       ├── bbox_track_manager.py # Cutie 时序跟踪管理器
│       ├── bbox_association.py # IoU 匹配
│       ├── stereo_depth.py     # FoundationStereo TRT 推理
│       ├── world_roi.py        # 世界坐标 ROI 过滤
│       ├── workbench_plane.py  # RANSAC 台面拟合
│       ├── surface_mesh.py     # 混合表面网格
│       ├── pixel3d_mesh.py     # Pixel3D 异步 mesh 补全
│       ├── vlm_detector.py     # MiniCPM VLM 检测器
│       ├── cutie_track.py      # Cutie mask 传播封装
│       ├── calib_loader.py     # 标定文件加载
│       ├── tag_mapping.py      # prompt/tag 映射
│       ├── rviz_markers.py     # RViz MarkerArray 构建
│       └── ...
├── launch/                     # Launch 文件
├── config/calib_results/       # 手眼标定结果
├── rviz/                       # RViz 配置
├── scripts/                    # 编译/启动脚本
├── environment.yml             # Conda 主环境
└── environment_vlm.yml         # VLM 独立环境
```

## 环境依赖

### 系统要求

- ROS2 Jazzy
- CUDA GPU（基础检测 >=8GB；VLM serve 另需约 8–10GB；启用 Pixel3D 需 >=24GB）
- Conda（`robot_perception` + 可选 `robot_perception_vlm` 双环境）

### 第三方模型路径（必配）

节点启动时需要找到 `third_party/robot_perception/` 下的源码目录（通过 `ROBOT_PERCEPTION_DIR` 解析）：

```bash
export ROBOT_PERCEPTION_DIR=/path/to/third_party/robot_perception
# 例: export ROBOT_PERCEPTION_DIR=~/ros_ws/src/hand_gesture_primitives/third_party/robot_perception
```

**目录结构（与 GDINO/Cutie 同级）：**

```
third_party/robot_perception/
├── Grounded-SAM-2/      # GDINO + SAM2 源码（sys.path）
├── Cutie/               # 时序 mask 跟踪
├── Pixal3D/             # Pixel3D 源码（含 pixal3d/ Python 包，启用 use_pixel3d 时必配）
└── scripts/
    ├── download_pixal3d.sh   # 仅下载 pixal3d-t 主权重
    ├── download_rmbg2.sh     # 下载 RMBG-2.0 去背景辅助权重
    └── download_dinov3_vitl16.sh  # 下载 DINOv3 图像条件辅助权重
```

> **注意：** `Pixal3D/` 是**源码仓库**（需 git clone），与下方**模型权重**是两套路径。只下载权重、不 clone 源码会报 `No module named 'pixal3d'`。

**模型权重默认路径：**

| 模型 | 路径 |
|------|------|
| Grounding DINO | `data/weights/grounding_dino/` |
| SAM2 | `data/weights/sam2/` |
| Cutie | `data/weights/cutie/` |
| FoundationStereo（可选） | `data/weights/foundation_stereo/foundation_stereo.plan` |
| MiniCPM VLM | `data/weights/minicpm-v-4.6-gptq/` |
| Pixel3D 主权重 | `data/weights/pixal3d-t` |
| RMBG-2.0（Pixel3D rembg） | `${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0`（staging: `data/weights/RMBG-2.0`） |
| DINOv3 ViT-L/16（Pixel3D cond） | `${PERCEPTION_WEIGHTS_DIR:-data/weights}/dinov3-vitl16-pretrain-lvd1689m`（staging: `data/weights/dinov3-vitl16-pretrain-lvd1689m`） |

**Staging → 部署流程**（开发机下载，测试/真机 rsync 到 `${PERCEPTION_WEIGHTS_DIR:-data/weights}/`）：

```bash
# 1. 在可访问 ModelScope 的机器上下载到 data/weights/
bash third_party/robot_perception/scripts/download_pixal3d.sh
bash third_party/robot_perception/scripts/download_rmbg2.sh
bash third_party/robot_perception/scripts/download_dinov3_vitl16.sh

# 2. 同步到目标机
rsync -av data/weights/pixal3d-t/  ${PERCEPTION_WEIGHTS_DIR:-data/weights}/pixal3d-t/
rsync -av data/weights/RMBG-2.0/   ${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0/
rsync -av data/weights/dinov3-vitl16-pretrain-lvd1689m/ \
      ${PERCEPTION_WEIGHTS_DIR:-data/weights}/dinov3-vitl16-pretrain-lvd1689m/
```

下载 Pixel3D 主权重（约 23GB）：

```bash
bash third_party/robot_perception/scripts/download_pixal3d.sh
# 自定义输出: PIXAL3D_WEIGHT_DIR=/path/to/pixal3d-t bash ...
```

下载 RMBG-2.0 去背景模型（Pixel3D pipeline 辅助权重，离线机必下）：

```bash
# ModelScope: https://modelscope.cn/models/briaai/RMBG-2.0/files
bash third_party/robot_perception/scripts/download_rmbg2.sh
# 默认输出: data/weights/RMBG-2.0
# 或: RMBG2_WEIGHT_DIR=${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0 bash ...
```

下载 DINOv3 图像条件模型（Pixel3D pipeline 辅助权重，离线机必下）：

```bash
# ModelScope: https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m
bash third_party/robot_perception/scripts/download_dinov3_vitl16.sh
# 默认输出: data/weights/dinov3-vitl16-pretrain-lvd1689m
```

部署后修改 `pixal3d-t/pipeline.json`，将辅助模型指向本地路径（避免访问 huggingface.co）：

```json
"image_cond_model": {
  "name": "DinoV3FeatureExtractor",
  "args": {
    "model_name": "${PERCEPTION_WEIGHTS_DIR:-data/weights}/dinov3-vitl16-pretrain-lvd1689m"
  }
},
"rembg_model": {
  "name": "BiRefNet",
  "args": {
    "model_name": "${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0"
  }
}
```

Clone Pixel3D 源码（启用 `use_pixel3d` 时必做）：

```bash
cd $ROBOT_PERCEPTION_DIR
git clone https://github.com/TencentARC/Pixal3D.git
# 验证
test -f Pixal3D/pixal3d/pipelines/__init__.py && echo OK
```

### Python 依赖

`environment.yml` 已与测试机可运行环境同步（2026-06-24），包含 detection_bbox 主链路 + Pixel3D/TRELLIS 扩展 + ROS colcon 编译依赖（`empy==3.3.4`、`catkin-pkg`、`lark`）。

```bash
# 主环境 (robot_perception conda)
conda env create -f environment.yml
# 已有环境时同步
conda env update -n robot_perception -f environment.yml --prune
# 或使用脚本
bash sync_env.sh update

# VLM 独立环境 (MiniCPM serve，与 detection_bbox 主环境分离)
bash scripts/setup_vlm_env.sh
# 或手动
conda env create -f environment_vlm.yml
conda env update -n robot_perception_vlm -f environment_vlm.yml --prune
```

#### VLM 独立环境 (`robot_perception_vlm`)

`environment_vlm.yml` 已与测试机可运行环境同步（2026-06-24），包含 MiniCPM-V GPTQ 推理 + `transformers serve` HTTP 服务全栈（fastapi/uvicorn/gptqmodel 等）。

**必须与 `robot_perception` 分离：** 主环境锁定 `transformers==4.57.3`（GDINO/SAM2），VLM 需要 `transformers==5.12.1`，二者不可合并。

| 包 | 测试机版本 |
|----|-----------|
| Python | 3.12.13 |
| torch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| transformers | 5.12.1 |
| tokenizers | 0.22.2 |
| gptqmodel | 7.1.0 |
| optimum | 2.2.0 |
| huggingface-hub | 1.20.1 |
| accelerate | 1.14.0 |
| safetensors | 0.8.0 |
| fastapi / uvicorn | 0.136.3 / 0.49.0 |
| openai (client) | 2.43.0 |

```bash
# 1. 下载权重（staging）
bash third_party/robot_perception/scripts/download_minicpm_v4_6_gptq.sh

# 2. 部署到运行路径（默认 ${PERCEPTION_WEIGHTS_DIR:-data/weights}/minicpm-v-4.6-gptq/）
bash scripts/sync_minicpm_weights.sh

# 3. 创建 VLM conda 环境
bash scripts/setup_vlm_env.sh

# 4. 验证 serve 可启动
bash scripts/start_minicpm_vlm_server.sh
curl -sf http://127.0.0.1:8000/health && echo OK
```

**显存参考：** MiniCPM-V-4.6-GPTQ 本地 serve 约需 **8–10 GB** 显存（与 `robot_perception` 主节点分 GPU 或分时运行时可避免 OOM）。

**启用 Pixel3D 时仍需额外步骤**（源码与部分 Pixal3D 专属依赖不在 yml 内）：

```bash
conda activate robot_perception

# 1. Clone Pixal3D 源码（见上文「Clone Pixel3D 源码」）
# 2. Pixal3D 剩余 Python 依赖（diffusers 等，yml 已含 easydict/cumesh/natten 等）
pip install -r $ROBOT_PERCEPTION_DIR/Pixal3D/requirements.txt

# 3. Pixal3D README 额外依赖
pip install https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl

# 4. 若 yml 中 TRELLIS wheel 与当前 GPU/torch ABI 不兼容，可重装（见下方方式 A/B）
#    测试机已验证: cumesh==1.0 flex-gemm==0.0.1 o-voxel==0.0.1 natten==0.21.0 zstandard==0.25.0

# 5. 验证
python3 -c "
import sys; sys.path.insert(0, '$ROBOT_PERCEPTION_DIR/Pixal3D')
from transformers import DINOv3ViTModel
import cumesh, flex_gemm, o_voxel, nvdiffrast
from pixal3d.pipelines import Pixal3DImageTo3DPipeline
print('Pixel3D deps OK')
"
```

<details>
<summary>TRELLIS.2 wheel 手动重装（仅当 conda create 后 import 失败时）</summary>

```bash
BASE=https://raw.githubusercontent.com/visualbruno/ComfyUI-Trellis2/main/wheels/Linux/Torch270
pip install ${BASE}/cumesh-1.0-cp312-cp312-linux_x86_64.whl
pip install ${BASE}/flex_gemm-0.0.1-cp312-cp312-linux_x86_64.whl
pip install ${BASE}/nvdiffrast-0.4.0-cp312-cp312-linux_x86_64.whl
pip install --no-deps ${BASE}/o_voxel-0.0.1-cp312-cp312-linux_x86_64.whl
pip install zstandard
# 或从源码编译（需 CUDA_HOME 与 PyTorch CUDA 版本一致）
```
</details>

> 修改 msg 定义（如 `SurfaceMesh`）后须**先**编译 `robot_perception_msgs` 再编译 `robot_perception`。

## 编译

```bash
source /opt/ros/jazzy/setup.bash
eval "$(conda shell.bash hook)" && conda activate robot_perception

# 编译消息包 + 感知包
colcon build --packages-select robot_perception_msgs robot_perception \
  --symlink-install \
  --paths messages/robot_perception_msgs perception

source install/setup.bash
```

或使用脚本一键编译：

```bash
bash scripts/build_perception.sh
```

## 快速启动

> **前置步骤（每次新终端都需要执行）：**
>
> ```bash
> # 1. 激活 conda 环境
> eval "$(conda shell.bash hook)"
> conda activate robot_perception
>
> # 2. Source ROS2 和 workspace
> source /opt/ros/jazzy/setup.bash
> source install/setup.bash
>
> # 3. 设置第三方模型路径（如未写入 .bashrc）
> export ROBOT_PERCEPTION_DIR=/path/to/third_party/robot_perception
>
> # 4. 设置 DDS（与其他节点保持一致）
> export ROS_DOMAIN_ID=42
> export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
> ```
>
> 或直接使用 `scripts/run_detection_bbox.sh`，该脚本会自动完成上述所有步骤。

### detection_bbox（推荐日常使用）

detection_bbox 支持两种检测后端，按需选择：

| 后端 | 模型 | 优势 | 劣势 |
|------|------|------|------|
| **GDINO** (默认) | Grounding DINO + SAM2 | 快速、离线、无需额外服务 | 仅英文 prompt、需精确 caption |
| **VLM** | MiniCPM-V + SAM2 | 支持自然语言描述、多模态理解 | 需启动独立推理服务、推理略慢 |

#### 链路 A: GDINO 检测（默认）

无需额外服务，直接启动：

```bash
# 默认产线工装检测 (机械手/电机/钣金/五金工具等 19 类) + 时序跟踪
ros2 launch robot_perception detection_bbox.launch.py launch_rviz:=true

# 桌面乐高积木
ros2 launch robot_perception detection_bbox.launch.py \
  auto_industry_scene_prompts:=false auto_lego_scene_prompts:=true

# 指定 tag（精确匹配 LABEL_PROMPTS_MAP 中定义的物体）
ros2 launch robot_perception detection_bbox.launch.py \
  auto_industry_scene_prompts:=false \
  tags:="['2_4_blue_lego','2_10_red_bridge_lego']"

# 自定义文本 prompt（开放词汇）
ros2 launch robot_perception detection_bbox.launch.py \
  auto_industry_scene_prompts:=false \
  text_prompts:="['blue lego.', 'red lego.']"
```

GDINO 后端加载 `groundingdino_swint_ogc.pth` + `sam2.1_hiera_tiny.pt`，全部在 GPU 本地推理。

#### 链路 B: VLM 检测

VLM 链路使用 MiniCPM-V 替代 GDINO 做物体发现，适合需要更灵活语义理解的场景。需要**两步启动**：

**Step 1: 启动 VLM 推理服务（独立终端）**

```bash
bash perception/scripts/start_minicpm_vlm_server.sh
```

该脚本会：
- 激活 `robot_perception_vlm` conda 环境
- 校验 MiniCPM 权重路径（默认 `${PERCEPTION_WEIGHTS_DIR:-data/weights}/minicpm-v-4.6-gptq/`）
- 启动 `transformers serve` OpenAI 兼容 API，监听 `http://127.0.0.1:8000`
- 服务就绪后可通过 `curl http://127.0.0.1:8000/health` 验证

**Step 2: 启动 detection_bbox 节点（指定 VLM 后端）**

```bash
# 方式 1: 手动 launch（VLM 服务已在独立终端运行）
ros2 launch robot_perception detection_bbox.launch.py \
  use_vlm_detect:=true \
  vlm_base_url:=http://127.0.0.1:8000/v1

# 方式 2: 一键脚本（自动检测/启动 VLM 服务）
bash perception/scripts/run_detection_bbox.sh --with-vlm launch_rviz:=true
```

`--with-vlm` 模式下脚本会自动检测 VLM 服务是否在线，未在线则后台拉起并等待就绪。

**VLM 链路差异：**
- 不加载 Grounding DINO 权重（节省约 1GB 显存）
- Layer2 ROI refine 和 Layer4 全图发现均走 VLM API
- 仍使用 SAM2 做分割、Cutie 做时序跟踪
- VLM 检测结果经过跨 prompt NMS 去重 + 面积/宽高比过滤
- SAM2 分割置信度低于 `sam2_score_min` 的结果会被丢弃

**VLM 环境首次部署：**

```bash
# 1. 下载并同步权重
bash third_party/robot_perception/scripts/download_minicpm_v4_6_gptq.sh
bash scripts/sync_minicpm_weights.sh

# 2. 创建/同步 VLM conda 环境（environment_vlm.yml，测试机 2026-06-24）
bash scripts/setup_vlm_env.sh

# 3. 验证服务可启动
bash scripts/start_minicpm_vlm_server.sh
# 看到 "Uvicorn running on http://0.0.0.0:8000" 即成功
curl -sf http://127.0.0.1:8000/health
```

**VLM 常见问题：**

| 报错 | 处理 |
|------|------|
| `conda env 'robot_perception_vlm' missing` | `bash scripts/setup_vlm_env.sh` |
| `PackageNotFoundError: optimum` / `gptqmodel` | `bash scripts/sync_env_vlm.sh update` |
| `/v1/models` 返回空 `data: []` | 正常，本地权重 serve 不注册到 model list，使用权重目录路径即可 |
| VLM 超时无响应 | 检查 GPU 显存是否被 `robot_perception` 占满；VLM 与 GDINO 建议分 GPU 或先停主感知 |
| `weights not found` | `bash scripts/sync_minicpm_weights.sh` |
| 端口 8000 被占用 | `bash scripts/stop_vlm_server.sh` 后重启 |

#### 链路 C: 场景理解模式（VLM 自动发现物体）

当不确定场景中有什么物体时，启用场景理解模式让 VLM 自主发现：

```bash
# 无需预设 prompt，VLM 自动分析场景中有哪些可操作物体
ros2 launch robot_perception detection_bbox.launch.py \
  use_vlm_detect:=true \
  use_scene_understand:=true \
  auto_industry_scene_prompts:=false

# 也可与预设 prompt 配合使用（补充模式）
ros2 launch robot_perception detection_bbox.launch.py \
  use_vlm_detect:=true \
  use_scene_understand:=true \
  text_prompts:="['screwdriver.']"
```

**场景理解工作流程：**
1. 首帧：VLM 分析图像，列出所有可操作物体（如 "red screwdriver", "blue lego brick"）
2. 发现结果自动合并为检测 prompt，后续链路正常运行
3. 每 N 帧检测未跟踪区域的像素变化（轻量帧差法）
4. 检测到新物体出现（如放置新物体到工作台）→ 自动重新理解场景并更新检测目标

**注意：** 场景理解需要 VLM 服务运行（`use_vlm_detect:=true`）。

#### 链路 D: Pixel3D 3D Mesh 补全

对分割出的物体生成完整 3D 网格（含未观测面），替代默认的混合表面网格（仅观测面）：

```bash
# 配合 GDINO 检测使用
ros2 launch robot_perception detection_bbox.launch.py \
  use_pixel3d:=true \
  pixel3d_low_vram:=true

# 配合 VLM 检测使用
ros2 launch robot_perception detection_bbox.launch.py \
  use_vlm_detect:=true \
  use_pixel3d:=true
```

**前置条件（缺一不可）：**

| 项 | 说明 |
|----|------|
| `$ROBOT_PERCEPTION_DIR/Pixal3D/` | Pixal3D **源码**（含 `pixal3d/` 包） |
| `pixel3d_model_path` | **权重**目录（含 `pipeline.json`、`ckpts/`） |
| RMBG-2.0 本地路径 | `${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0`（见离线配置） |
| DINOv3 本地路径 | `${PERCEPTION_WEIGHTS_DIR:-data/weights}/dinov3-vitl16-pretrain-lvd1689m`（见离线配置） |
| 离线配置 | `config/pixel3d/pixel3d_offline.offline.example.json` + `pipeline.offline.example.json` |
| Python 依赖 | 见上文「启用 Pixel3D 时需额外安装」 |
| `aabb_work` 在 ROI 内 | 仅 world 系 3D bbox 通过 ROI 的目标会触发 Pixel3D |

**工作原理：**
1. 正常检测+跟踪流程运行（不受影响）
2. 当某物体 track 稳定后（连续 N 帧无丢失，且 `aabb_work` 非空），自动提交 Pixel3D 推理请求
3. 后台线程裁剪物体 RGB 图像 → Pixel3D 生成完整 mesh → 对齐到 world 坐标
4. 结果就绪后替换该物体的 `surface_mesh`，发布到 `surface_meshes` topic 与 RViz `markers_surface`

**特点：**
- 懒加载：Pixel3D 模型在首次推理请求时才加载，不拖慢节点启动
- 异步不阻塞：推理在后台线程运行，实时检测帧率不受影响
- 每 track 仅一次：同一 track 只触发一次；**推理失败后也会标记完成，需重启节点才能重试**
- low_vram 模式：各 stage 模型按需加载/卸载，峰值 ~10-12 GB
- 未启用 Pixel3D 时仍发布 **hybrid surface mesh**（深度点重建的可见面，非完整补全）

**硬件要求：**
- 建议 ≥ 24 GB VRAM（如 RTX 4090），与检测模型同 GPU 运行
- low_vram=true 峰值 ~17-19 GB（检测模型 ~7GB + Pixel3D 单阶段 ~10-12GB）
- 8 GB 显卡不支持同 GPU 运行 Pixel3D

**模型权重：**
- 主权重默认路径：`${PERCEPTION_WEIGHTS_DIR:-data/weights}/pixal3d-t`
- 主权重下载：`bash third_party/robot_perception/scripts/download_pixal3d.sh`
- rembg 辅助权重：[RMBG-2.0 @ ModelScope](https://modelscope.cn/models/briaai/RMBG-2.0/files) → `download_rmbg2.sh`
- DINOv3 辅助权重：[dinov3-vitl16 @ ModelScope](https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m) → `download_dinov3_vitl16.sh`
- 可通过 launch 参数 `pixel3d_model_path` 自定义主权重路径

**离线部署（data/weights）：**

将权重 rsync 到开发机后，安装离线 `pipeline.json`（把 HF 在线路径改为本地）：

```bash
# 权重目录（开发机）
${PERCEPTION_WEIGHTS_DIR:-data/weights}/pixal3d-t/
${PERCEPTION_WEIGHTS_DIR:-data/weights}/RMBG-2.0/
${PERCEPTION_WEIGHTS_DIR:-data/weights}/dinov3-vitl16-pretrain-lvd1689m/

# 安装离线 pipeline.json（在 colcon 安装后的 share 目录，或源码 config 目录）
bash $(ros2 pkg prefix robot_perception)/share/robot_perception/config/pixel3d/install_offline.sh
# 或源码路径：
bash perception/config/pixel3d/install_offline.sh

# 启动（自动加载 pixel3d_offline.offline.example.json）
ros2 launch robot_perception detection_bbox.launch.py \
  use_pixel3d:=true pixel3d_low_vram:=true
```

配置文件说明：

| 文件 | 用途 |
|------|------|
| `config/pixel3d/pipeline.offline.example.json` | 覆盖 `pixal3d-t/pipeline.json`，rembg/dinov3 指向 `${WEIGHTS_BASE}/...` |
| `config/pixel3d/pixel3d_offline.offline.example.json` | 4 阶段 image_cond 本地路径 + 模型根目录 |
| `config/pixel3d/install_offline.sh` | 一键拷贝 pipeline 配置到权重目录 |

自定义路径可编辑 `pixel3d_offline.offline.example.json`，或通过 `pixel3d_offline_config:=/path/to/your.json` 指定。


```text
[Pixel3D] Loading pipeline from ... (low_vram=True)...
[Pixel3D] Pipeline loaded.
[Pixel3D] Inference done for track N (label=..., verts=..., faces=...)
```

### perception_pipeline（已移除）

> 6DOF FoundationPose 链路已移除，仅保留 detection_bbox。

## 运行时 prompt 切换

所有节点均监听 `/robot_perception/detection_prompts` (类型: `ros_gz_interfaces/msg/StringVec`)：

```bash
# 发送 tag 名称（查 LABEL_PROMPTS_MAP）
ros2 topic pub /robot_perception/detection_prompts ros_gz_interfaces/msg/StringVec \
  "{data: ['2_4_blue_lego', '2_10_red_bridge_lego']}" --once

# 发送自由文本 prompt（detection_bbox 支持，allow_freeform_prompts=true）
ros2 topic pub /robot_perception/detection_prompts ros_gz_interfaces/msg/StringVec \
  "{data: ['cup.', 'bottle.', 'phone.']}" --once
```

## 发布 Topic

完整外发 Topic、消息字段、坐标系约定及下游集成说明见：

**[docs/output_topics.md](docs/output_topics.md)**

自定义消息定义位于 `messages/robot_perception_msgs/`。

## 时序跟踪系统（detection_bbox）

`use_temporal_tracking:=true`（默认）启用多层跟踪：

```
Layer 1 (Cutie):  mask 传播，维持稳定 ID
Layer 2 (ROI):    漂移检测时在局部 ROI 内精细 GDINO+SAM
Layer 4 (Global): 周期全图发现新物体 / 恢复丢失 track
```

- `instance_id` 跨帧稳定（track 生命周期内不变）
- `track_label_lock=true` 防止 label 抖动
- `track_global_detect_interval=10` 每 10 帧全图检测

关闭跟踪使用逐帧独立检测：`use_temporal_tracking:=false`

## 3D AABB 链路

```
Color + Depth → GDINO 检测 → SAM2 分割 → Mask
                                            ↓
Depth + K + Mask → backproject_mask() → 3D 点云 (camera frame)
                                            ↓
T_world_cam → transform_points() → 3D 点云 (world frame) → aabb_from_points()
                                            ↓
World ROI Filter (forward + surface) → 发布 / 丢弃
```

### 深度源优先级

1. **Hardware depth**（默认）— 相机自带深度图
2. **FoundationStereo**（可选，`use_stereo_depth:=true` + IR 双目就绪 + TF 可用 + TRT engine）

### ROI 过滤模式 (world_roi_mode)

| 模式 | 条件 |
|------|------|
| `and` (默认) | 前方 ≤1m **且** 在台面上 |
| `or` | 前方 ≤1m **或** 在台面上 |
| `surface_only` | 仅在台面上 |

台面高度通过 RANSAC 动态拟合（`enable_ransac_workbench_plane=true`），或使用静态先验 `workbench_z`。

## 标定

手眼标定结果存放于 `config/calib_results/`：

| 文件 | 用途 |
|------|------|
| `full_calibration_result.txt` | **示例占位**（单位阵），仅验证文件格式；**不可用于生产** |
| `full_calibration_result.npz` | 可选 `.npz` 格式（用户自行标定后生成） |

通过 launch 参数 `calib_file` 指定实机标定文件（`.txt` 或 `.npz`）。也可设置环境变量 `ROBOT_CALIB_FILE`。

**无有效标定时：** world 系 3D bbox / mesh / 台面平面不可用，`bboxes_3d` 中 world 条目为空，仅保留 2D 检测与相机系可视化。

## 重要参数速查

### 检测参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `fps` | 5.0 | 最大检测帧率 |
| `box_threshold` | 0.35 | GDINO box 阈值 |
| `text_threshold` | 0.25 | GDINO text 阈值 |
| `min_detection_score` | 0.45 | 最终置信度过滤 |
| `min_mask_pixels` | 100 | 最小 mask 面积 |
| `min_depth_points` | 50 | 3D bbox 最少深度点 |
| `depth_min_m` / `depth_max_m` | 0.01 / 1.5 | 有效深度范围 |
| `sam2_score_min` | 0.7 | SAM2 分割置信度阈值（低于此值丢弃） |
| `vlm_nms_iou` | 0.5 | VLM 多 prompt 结果 NMS IoU 阈值 |
| `vlm_max_area_ratio` | 0.5 | VLM 检测框最大面积占比（过滤大框噪声） |

### 场景 prompt 选择

| 参数 | 默认 | 说明 |
|------|------|------|
| `auto_industry_scene_prompts` | **true** | 产线 19 类 (机械手/电机/钣金/五金工具等) |
| `auto_lego_scene_prompts` | false | 桌面乐高 10 类 |
| `auto_open_scene_prompts` | false | 开放场景 32 类 |

### 场景理解参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_scene_understand` | false | 启用 VLM 场景理解（需 use_vlm_detect=true） |
| `scene_change_check_interval` | 10 | 每 N 帧检测一次场景变化 |
| `scene_change_threshold` | 0.05 | 未跟踪区域变化像素占比阈值 |
| `scene_change_pixel_threshold` | 30 | 像素灰度差判定为"变化"的阈值 |
| `scene_max_objects` | 10 | 单次场景理解最多返回物体数 |

### Pixel3D 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_pixel3d` | false | 启用 Pixel3D 完整 mesh 补全 |
| `pixel3d_model_path` | `${PERCEPTION_WEIGHTS_DIR:-data/weights}/pixal3d-t` | 模型权重路径（非源码目录） |
| `pixel3d_low_vram` | true | 低显存模式（按需加载各阶段模型） |
| `pixel3d_trigger_age` | 5 | track 稳定帧数后触发推理 |
| `pixel3d_max_concurrent` | 1 | 最大并发推理任务数 |
| `pixel3d_cache_dir` | `""` | Pixel3D 磁盘 mesh 缓存目录（空=关闭） |
| `pixel3d_use_label_cache` | true | 同 label 复用 mesh |
| `publish_surface_mesh_topic` | true | 发布 `surface_meshes` topic |
| `enable_hybrid_surface_mesh` | true | 深度点 hybrid 表面（Pixel3D 未就绪时的 fallback） |

### 跟踪参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `use_temporal_tracking` | true | 启用时序跟踪 |
| `track_global_detect_interval` | 10 | 全图检测周期(帧) |
| `track_lost_max_frames` | 5 | 丢失 N 帧后删除 track |
| `track_label_lock` | true | label 注册后锁定 |
| `track_assoc_iou_min` | 0.3 | 关联最小 IoU |

### 3D/世界坐标参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `workbench_frame_id` | `world` | 世界坐标系名称 |
| `enable_world_roi_filter` | true | 启用 ROI 过滤 |
| `world_roi_mode` | `and` | ROI 模式 |
| `world_forward_max_m` | 1.0 | 最大前方距离 |
| `workbench_z` | -1.0 | 台面高度先验 (m) |
| `enable_ransac_workbench_plane` | true | RANSAC 动态台面 |
| `publish_static_tf` | true | 发布 world→camera TF |

## 脚本说明

| 脚本 | 作用 |
|------|------|
| `scripts/build_perception.sh` | 编译 msgs + robot_perception |
| `scripts/run_detection_bbox.sh` | 增量编译 + 启动 `detection_bbox`；`--with-vlm` 启 VLM |
| `scripts/perception_env.sh` | 公共路径变量 |
| `scripts/setup_vlm_env.sh` | 从 `environment_vlm.yml` 创建/更新 `robot_perception_vlm` |
| `scripts/sync_env_vlm.sh` | 同步 VLM conda 环境（`create` / `update`） |
| `scripts/start_minicpm_vlm_server.sh` | 启动本地 MiniCPM VLM 推理服务（自动清理残留端口占用） |
| `scripts/stop_vlm_server.sh` | 停止 VLM 推理服务并释放端口 |
| `scripts/sync_minicpm_weights.sh` | 同步 VLM 模型权重 |
| `third_party/.../download_pixal3d.sh` | 下载 Pixel3D-T 主权重 |
| `third_party/.../download_rmbg2.sh` | 下载 RMBG-2.0 rembg 辅助权重 |
| `third_party/.../download_dinov3_vitl16.sh` | 下载 DINOv3 图像条件辅助权重 |

## RViz 可视化

启动时附加 `launch_rviz:=true`，预配置显示：

- **Hybrid Surface / Pixel3D Mesh**: 表面 mesh（Pixel3D 成功后为完整补全 mesh）
- **Surface Meshes**: `surface_meshes` topic（结构化 mesh 数据）
- **Workbench Plane**: RANSAC 台面（半透明绿板）
- **AABB Wireframes**: 3D 包围盒线框（默认关闭）
- **Annotated Image**: 2D 检测结果叠加图

Fixed Frame 须设为 `world`。

## 故障排查

| 现象 | 可能原因 | 处理 |
|------|----------|------|
| `No module named groundingdino.util` | ROBOT_PERCEPTION_DIR 未设置 | 设置环境变量指向含 `Grounded-SAM-2/` 的目录 |
| `cannot import name 'SurfaceMesh'` | 消息包未重建 | `colcon build --packages-select robot_perception_msgs robot_perception`，再 `source install/setup.bash` |
| `No module named 'pixal3d'` | 缺 Pixal3D **源码** | clone 到 `$ROBOT_PERCEPTION_DIR/Pixal3D/`，确认 `pixal3d/` 子目录存在 |
| `No module named 'easydict'` | Pixal3D 依赖未装 | `pip install -r $ROBOT_PERCEPTION_DIR/Pixal3D/requirements.txt` |
| `cannot import name 'DINOv3ViTModel'` | transformers 过旧 | `pip install "transformers==4.57.3"`（需 ≥4.56） |
| `No module named 'cumesh'` / `flex_gemm` / `o_voxel` | TRELLIS.2 CUDA 扩展未装 | 见「启用 Pixel3D 时需额外安装」第 4 步；`o_voxel` 须 `pip install --no-deps` |
| `ResolutionImpossible` cumesh 与 o-voxel 冲突 | 同条 pip 装四个 wheel | 先装 cumesh/flex_gemm/nvdiffrast，再 `pip install --no-deps` 装 o_voxel |
| `Network is unreachable` + `briaai/RMBG-2.0` | 离线机 pipeline 拉 HF rembg | 预下载 RMBG-2.0，`pipeline.json` 改本地路径 |
| `Network is unreachable` + `dinov3-vitl16` | 离线机 pipeline 拉 HF DINOv3 | 预下载 DINOv3，`pipeline.json` 改本地路径 |
| `No module named 'zstandard'` | `o_voxel` 用 `--no-deps` 跳过了依赖 | `pip install zstandard` |
| `[Pixel3D] Inference failed` 后不再重试 | track 已标记 completed | 修复依赖后**重启** detection_bbox 节点 |
| 无 Pixel3D mesh，仅有 hybrid 面 | 推理未成功或 track 无 `aabb_work` | 查 log；放宽 world ROI；确认权重路径与 GPU 显存 |
| 无检测结果 | prompt 不在 caption 中 | 检查 text_prompts 或 tag 拼写 |
| 无 3D bbox | 深度点不足 / 无标定 / ROI 过滤 | 检查 calib_file、降低 min_depth_points；log 中 `3D dropped — outside world ROI` |
| OOM | 与 pipeline 同时运行 | 二选一或降低 fps、`pixel3d_low_vram:=true` |
| TF lookup failed (color ↔ IR) | IR frame 与 color 不在同一 TF 树 | 确认驱动发布 IR optical frame；或检查 launch 中 `ir_optical_frame` 参数 |
| instance_id 频繁变化 | track 频繁丢失 | 降低 track_lost_max_frames 或检查遮挡 |
| RViz 不显示 | Fixed Frame 不是 world | 切换 Fixed Frame 为 `world` |
| stereo 回退 hardware | IR/TF 未就绪 | 启动后约数帧内常见；持久出现则查 IR topic 和 TF |
| VLM `Failed to parse response` | VLM 返回非标准 JSON | 部分帧无检测，跟踪可维持；可忽略或调 prompt |

## 与 hand_gesture_primitives 集成

**推荐链路（启用 GraspGate）：**

```
detection_bbox → /camera_head/detection_bbox/bboxes_3d
                        ↓
              grasp_gate (launch_gate:=true)
                        ↓
         /object_pose (world) + /hand_gesture_cmd_exec
                        ↓
              hand_gesture_node (gesture_node)
```

`grasp_gate` 从 `bboxes_3d` 读取选中物体的 `center`，发布 `/object_pose`（`frame_id=world`），并转发门控后的手势指令。无需额外桥接节点。

**无门控链路：** 仅订阅 `bboxes_3d`，靠 `object_size` / `grasp_type` 做自适应闭合；不依赖 `/object_pose`。

**可选：** `/camera_head/perception/labeled_poses`（`LabeledPoseArray`）可由外部节点发布；当前 `detection_bbox` 未内置该 topic。

## 已知限制

- GDINO **只能检测 caption 中出现的类别**，不是通用物体检测器
- 3D 输出为轴对齐 AABB，非紧凑 OBB
- `LABEL_PROMPTS_MAP` 中部分 prompt 为中文，GDINO 英文模型无法识别
