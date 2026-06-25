# 第三方感知依赖：GitHub 源码与权重下载

本文档汇总 `third_party/robot_perception/` 下各算法库的 **源码仓库** 与 **模型权重** 获取方式，对应 `perception` 包（`detection_bbox` 节点）的实际使用路径。

> **权重根目录**（优先级）：`PERCEPTION_WEIGHTS_DIR` → 项目 `data/weights/` → `${PERCEPTION_WEIGHTS_DIR:-data/weights}/`  
> **源码根目录**：`ROBOT_PERCEPTION_DIR`（默认 `third_party/robot_perception/`）

### Git 子模块（MiniCPM-V / Pixal3D）

`MiniCPM-V` 与 `Pixal3D` 为 **git submodule**（VLM 参考文档、Pixel3D mesh 补全源码）。Clone 后需初始化：

```bash
git submodule update --init third_party/robot_perception/MiniCPM-V third_party/robot_perception/Pixal3D
```

或 clone 时：`git clone --recurse-submodules <repo-url>`

> **Cutie**、**Grounded-SAM-2** 已 vendored 在仓库内（见各目录 `LICENSE` / `README.md`）。  
> **物体 mesh**（力控原语可选）：`data/mesh_model/{label}/model.obj`，见 `hand_gesture_primitives/docs/primitives.md`。

---

## 一、detection_bbox 主链路（必装）

| 组件 | 用途 | 本地权重路径 | 源码 GitHub | 权重下载 |
|------|------|-------------|-------------|----------|
| **Grounded-SAM-2** | GDINO + SAM2 检测分割 | — | [IDEA-Research/Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2) | 见下表子项 |
| **Grounding DINO SwinT** | 开放词汇 2D 检测 | `grounding_dino/groundingdino_swint_ogc.pth` | [IDEA-Research/GroundingDINO](https://github.com/IDEA-Research/GroundingDINO) | [GitHub Release](https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth) · [HuggingFace](https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth) |
| **Grounding DINO 配置** | 模型结构定义 | `grounding_dino/GroundingDINO_SwinT_OGC.py` | 同上（`grounding_dino/config/`） | 随 Grounded-SAM-2 仓库 |
| **SAM 2.1 Hiera-Tiny** | 实例 mask 分割 | `sam2/sam2.1_hiera_tiny.pt` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) | [HuggingFace `facebook/sam2.1-hiera-tiny`](https://huggingface.co/facebook/sam2.1-hiera-tiny)（文件 `sam2.1_hiera_tiny.pt`） |
| **Cutie** | 时序 mask 跟踪 | `cutie/cutie-base-mega.pth` | [hkchengrex/Cutie](https://github.com/hkchengrex/Cutie) | [GitHub Release v1.0](https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth)（首次运行自动下载） |

### Grounded-SAM-2 官方下载脚本

仓库内（若已 clone 完整 Grounded-SAM-2）：

```bash
cd $ROBOT_PERCEPTION_DIR/Grounded-SAM-2/checkpoints && bash download_ckpts.sh   # SAM2
cd $ROBOT_PERCEPTION_DIR/Grounded-SAM-2/gdino_checkpoints && bash download_ckpts.sh  # GDINO
```

---

## 二、FoundationStereo 高质量深度（可选，`use_stereo_depth:=true`）

> 已从 `third_party/` 移除以精简开源 license。如需使用，放回源码或仅提供 `tensorrt_engine.py` + `.plan` 文件。

| 组件 | 用途 | 本地权重路径 | 源码 GitHub | 权重下载 |
|------|------|-------------|-------------|----------|
| **FoundationStereo** | 双目 IR → 深度（TRT） | `foundation_stereo/foundation_stereo.plan` | [NVlabs/FoundationStereo](https://github.com/NVlabs/FoundationStereo) | 见下方说明 |

### 启用方式

1. 将 FoundationStereo 源码放到 `$ROBOT_PERCEPTION_DIR/FoundationStereo/`，或将 `onnx_tensorrt/tensorrt_engine.py` 放到 `data/weights/foundation_stereo/`
2. 放置 TRT engine: `data/weights/foundation_stereo/foundation_stereo.plan`
3. Launch 时加 `use_stereo_depth:=true`

### 权重获取

1. **PyTorch 预训练**（Google Drive）：  
   - [23-51-11 Vit-L（推荐）](https://drive.google.com/drive/folders/1VhPebc_mMxWKccrv7pdQLTvXYVcLYpsf?usp=sharing)  
   - 商业版：[NVIDIA NGC TAO](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/foundationstereo)

2. **导出 ONNX / TensorRT**（本仓库使用 `.plan`）：  
   按 [FoundationStereo readme](https://github.com/NVlabs/FoundationStereo) 的 ONNX/TRT 章节导出后放置为：  
   `foundation_stereo/foundation_stereo.plan`

---

## 三、VLM 检测后端（可选）

| 组件 | 用途 | 本地权重路径 | 源码 GitHub | 权重下载 |
|------|------|-------------|-------------|----------|
| **MiniCPM-V 4.6 GPTQ** | 自然语言检测（HTTP serve） | `minicpm-v-4.6-gptq/` | [OpenBMB/MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) | 见下方脚本 |

**Conda 环境：** `robot_perception_vlm`（与主感知环境分离，配置见 `perception/environment_vlm.yml`，测试机同步 2026-06-24）

| 包 | 版本 |
|----|------|
| transformers | 5.12.1 |
| gptqmodel | 7.1.0 |
| optimum | 2.2.0 |
| torch | 2.8.0+cu128 |

```bash
# 下载到 third_party/robot_perception/models/minicpm-v-4.6-gptq/
bash third_party/robot_perception/scripts/download_minicpm_v4_6_gptq.sh

# 部署到运行目录
bash perception/scripts/sync_minicpm_weights.sh

# 创建 VLM conda 环境
bash perception/scripts/setup_vlm_env.sh
```

| 平台 | Model ID |
|------|----------|
| HuggingFace | [openbmb/MiniCPM-V-4.6-GPTQ](https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ) |
| ModelScope | [OpenBMB/MiniCPM-V-4.6-GPTQ](https://modelscope.cn/models/OpenBMB/MiniCPM-V-4.6-GPTQ) |

启动：`bash perception/scripts/start_minicpm_vlm_server.sh`

---

## 四、Pixel3D 3D Mesh（可选，`use_pixel3d:=true`）

| 组件 | 用途 | 本地路径 | 源码 GitHub | 权重下载 |
|------|------|---------|-------------|----------|
| **Pixal3D 源码** | Python 包 `pixal3d` | `$ROBOT_PERCEPTION_DIR/Pixal3D/` | [TencentARC/Pixal3D](https://github.com/TencentARC/Pixal3D) | `git clone`（**非**权重） |
| **Pixal3D-T 主权重** | 8 个子模型 pipeline | `pixal3d-t/`（~23GB） | 同上 | 见下方脚本 |
| **RMBG-2.0** | 去背景（rembg） | `RMBG-2.0/` | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | 见下方脚本 |
| **DINOv3 ViT-L/16** | 图像条件 backbone | `dinov3-vitl16-pretrain-lvd1689m/` | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | 见下方脚本 |

### 一键下载脚本（项目内）

```bash
# 默认 staging: <workspace>/models/
bash third_party/robot_perception/scripts/download_pixal3d.sh      # ~23GB
bash third_party/robot_perception/scripts/download_rmbg2.sh
bash third_party/robot_perception/scripts/download_dinov3_vitl16.sh
```

| 权重 | HuggingFace | ModelScope |
|------|-------------|------------|
| Pixal3D-T | [TencentARC/Pixal3D](https://huggingface.co/TencentARC/Pixal3D) | [TencentARC/Pixal3D-T](https://modelscope.cn/models/TencentARC/Pixal3D-T) |
| RMBG-2.0 | [briaai/RMBG-2.0](https://huggingface.co/briaai/RMBG-2.0) | [briaai/RMBG-2.0](https://modelscope.cn/models/briaai/RMBG-2.0/files) |
| DINOv3 ViT-L/16 | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) | [facebook/dinov3-vitl16-pretrain-lvd1689m](https://modelscope.cn/models/facebook/dinov3-vitl16-pretrain-lvd1689m) |

Clone 源码：

```bash
cd $ROBOT_PERCEPTION_DIR
git clone https://github.com/TencentARC/Pixal3D.git
```

离线部署需拷贝 `perception/config/pixel3d/pipeline.offline.example.json` → `pixal3d-t/pipeline.json`，并将 rembg / DINOv3 指向本地路径（详见 `perception/readme.md`）。

### Pixel3D 额外依赖（非权重）

| 依赖 | GitHub / 下载 |
|------|---------------|
| utils3d wheel | [LDYang694/Storages Release](https://github.com/LDYang694/Storages/releases/download/20260430/utils3d-0.0.2-py3-none-any.whl) |
| CuMesh（源码编译备选） | [JeffreyXiang/CuMesh](https://github.com/JeffreyXiang/CuMesh) |
| TRELLIS.2 CUDA wheels | [ComfyUI-Trellis2 wheels](https://github.com/visualbruno/ComfyUI-Trellis2/tree/main/wheels/Linux) |

---

## 五、完整权重目录结构（部署参考）

```
data/weights/                           # 或 PERCEPTION_WEIGHTS_DIR / ${PERCEPTION_WEIGHTS_DIR:-data/weights}
├── grounding_dino/
│   ├── GroundingDINO_SwinT_OGC.py
│   └── groundingdino_swint_ogc.pth
├── sam2/
│   └── sam2.1_hiera_tiny.pt
├── cutie/
│   └── cutie-base-mega.pth
├── foundation_stereo/                   # 可选
│   ├── foundation_stereo.plan
│   └── tensorrt_engine.py              # 从 FoundationStereo/onnx_tensorrt/ 复制
├── minicpm-v-4.6-gptq/                 # VLM 可选
├── pixal3d-t/                           # Pixel3D 可选 (~23GB)
│   ├── pipeline.json
│   └── ckpts/*.safetensors
├── RMBG-2.0/                            # Pixel3D 辅助
└── dinov3-vitl16-pretrain-lvd1689m/     # Pixel3D 辅助
```

配置清单见：`perception/config/models_offline.offline.example.yaml`

---

## 六、环境变量速查

```bash
export ROBOT_PERCEPTION_DIR=/path/to/third_party/robot_perception
export PERCEPTION_WEIGHTS_DIR=/path/to/data/weights   # 可选，覆盖默认权重根目录
export MINICPM_WEIGHT_DIR=/path/to/minicpm-v-4.6-gptq  # 可选
```

---

## 七、相关文档

- 感知包详细说明：[perception/readme.md](../perception/readme.md)
- 离线权重配置：[perception/config/models_offline.offline.example.yaml](../perception/config/models_offline.offline.example.yaml)
- 路径解析代码：[perception/robot_perception/utils/paths.py](../perception/robot_perception/utils/paths.py)
