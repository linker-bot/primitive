# 可选感知部署与权重准备（0.2.2）

基础原语、自定义多阶段、速度调节、进度控制、示教和姿态库**不需要 AI 模型权重**。
力控原语需要实际电流、力矩或触觉反馈，也不因此需要视觉模型。
先用[安装与启动指南](getting-started.md)运行核心功能即可。

视觉原语使用外部 ROS 的目标几何、TCP 位姿和关节/接触反馈；可以接你自己的感知服务。
只有选用仓库中的 `perception/` 示例，才需要本文的检测/分割模型。
网页提供原语入口，不会替你下载权重、启动相机或生成机械臂 TCP。

以下步骤为可选感知部署参考，需按目标机器的 GPU、CUDA、相机及标定配置验证。

## 1. 软件分层与目录

| 环境 | 用途 | 与其他环境的关系 |
| --- | --- | --- |
| 原语独立 ROS venv | `gesture_node`、`teach_panel`、姿态库 | 系统 Python 3.12，保持 apt NumPy/SciPy；见安装指南 |
| 主感知 Conda 环境 | 默认 GDINO、SAM2、Cutie | 按第 3 节创建最小环境；NumPy 1.26.4，与 ROS `cv_bridge` 配合 |
| VLM Conda 环境（可选） | MiniCPM-V GPTQ HTTP 服务 | 按第 6 节准备；与主感知分离，不加载 ROS `cv_bridge` |
| Pixal3D 扩展（可选） | 完整 mesh | 第 7 节另行适配 CUDA 扩展，不是默认检测的安装前置 |

不要在原语 ROS venv 中安装感知的 CUDA/PyTorch 依赖。
`perception/environment.yml` 和 `environment_vlm.yml` 仅是**既有站点环境快照/版本参考**，
不是新机器的一键安装配方。直接 `conda env create -f` 存在已知问题：

- `torch==2.8.0+cu128` 必须从 PyTorch 对应 CUDA wheel 源安装，快照没有配置该源。
- 主快照同时列出三种 OpenCV 4.13 分发包和 NumPy 1.26.4；在 Python 3.12 下
  OpenCV 4.12 起的这些 wheel 要求 NumPy 2，不能直接按这个组合求解。
- `cumesh`、`flex-gemm`、`o-voxel`、`pytorch3d` 等属于额外平台扩展，需要匹配的
  GPU/PyTorch/CUDA wheel 或编译工具链，默认检测不需要全部安装。
- VLM 快照的 TorchAO 0.17 与 Torch 2.8 不在作者兼容表内，不能直接复制这组版本。

已有站点复现时，应使用站点保存的完整 wheelhouse/环境锁和实际 GPU 信息核对快照；
不要对正在使用的环境直接执行 `conda env update --prune`。
下面列出默认检测所需依赖；请在目标 GPU 环境中完成导入和推理验证。

下面从**完整检出的仓库根目录**开始；路径按你的克隆位置修改。每个感知终端都需设置：

```bash
export GESTURE_SOURCE="$HOME/primitive"
export ROBOT_PERCEPTION_DIR="$GESTURE_SOURCE/third_party/robot_perception"
export PERCEPTION_WEIGHTS_DIR="$HOME/hand-gesture-data/perception-weights"
export PERCEPTION_WS="$HOME/gesture-perception-ws-0.2.2"
export TORCH_HOME="$HOME/hand-gesture-data/torch-cache"
export HF_HOME="$HOME/hand-gesture-data/huggingface-cache"
cd "$GESTURE_SOURCE"
test -f perception/package.xml
test -f perception/environment.yml
mkdir -p "$PERCEPTION_WEIGHTS_DIR" "$PERCEPTION_WS" "$TORCH_HOME" "$HF_HOME"
```

如果 `test` 失败，先核对仓库路径与稀疏检出规则，不要改指向另一份旧源码。
`PERCEPTION_WEIGHTS_DIR` 必须在启动前存在；所有感知终端使用同一绝对路径。
这些目录与 `HAND_GESTURE_POSE_DIR` 的用户姿态数据互相独立。

## 2. 需要哪些源码和权重

以下路径相对 `$PERCEPTION_WEIGHTS_DIR`。下载入口来自模型作者的官方仓库/发布页；
具体模型修订和文件清单以所选版本的发布说明为准。

| 功能 | 权重/配置及落点 | 官方入口 |
| --- | --- | --- |
| 默认 GDINO 检测 | `grounding_dino/groundingdino_swint_ogc.pth`、同目录 `GroundingDINO_SwinT_OGC.py` | [Grounding DINO Release](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha) |
| GDINO 文本编码器 | `bert-base-uncased/`；配置中的 `text_encoder_type` 指向这里 | [Google BERT](https://huggingface.co/google-bert/bert-base-uncased) |
| SAM2 分割 | `sam2/sam2.1_hiera_tiny.pt` | [Meta SAM 2.1 Hiera-Tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/main) |
| Cutie 时序跟踪 | `cutie/cutie-base-mega.pth`；当前下载器还检查 `coco_lvis_h18_itermask.pth` | [Cutie v1.0](https://github.com/hkchengrex/Cutie/releases/tag/v1.0) |
| VLM 检测/场景理解，可选 | `minicpm-v-4.6-gptq/` | [OpenBMB MiniCPM-V-4.6-GPTQ](https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ) |
| Pixal3D 完整 mesh，可选 | `pixal3d-t/`，包含 `pipeline.json`、`ckpts/` | [TencentARC Pixal3D](https://huggingface.co/TencentARC/Pixal3D) |
| Pixal3D 辅助模型 | `RMBG-2.0/`、`dinov3-vitl16-pretrain-lvd1689m/` | [BRIA RMBG](https://huggingface.co/briaai/RMBG-2.0)、[Meta DINOv3](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m) |
| FoundationStereo 深度，可选 | `foundation_stereo/foundation_stereo.plan` 与对应推理实现 | [NVlabs FoundationStereo](https://github.com/NVlabs/FoundationStereo) |

默认检测先准备 GDINO、BERT、SAM2；启用默认时序跟踪还要准备 Cutie 及其缓存。
VLM 检测使用 SAM2，启用跟踪时仍用 Cutie，不加载 GDINO 检测权重。
仅增加 VLM 场景理解、仍由 GDINO 检测时，两套模型都需要。

`Grounded-SAM-2/`、`Cutie/` 已随仓库带入。Pixal3D 源码与其权重是两套文件。
0.2.2 的 `.gitmodules` 虽列了 Pixal3D/MiniCPM-V，但 Git 树未登记对应 gitlink，
不能依靠 `git submodule update` 获取它们。按第 7 节单独 clone 可选 Pixal3D。
MiniCPM-V 源码仓库仅作参考，不是这里启动 Transformers 服务的必需目录。

## 3. 主感知环境与基础模型下载

本节适用于 Ubuntu 24.04、Linux x86_64、Python 3.12 和支持 CUDA 12.8 的 NVIDIA 环境。
ARM、其他 CUDA/GPU 组合需要选择对应官方 wheel，并同步调整编译目标。
先安装 Conda、ROS 2 Jazzy、目标 GPU 驱动和用于第 4 节编译的 CUDA Toolkit/C++ 工具链，
再在独立终端创建空环境，不读取大型站点快照：

```bash
cd "$GESTURE_SOURCE"
eval "$(conda shell.bash hook)"
conda create -n gesture-perception-0.2.2 python=3.12 pip -y
conda activate gesture-perception-0.2.2
cat > "$PERCEPTION_WS/perception-constraints.txt" <<'CONSTRAINTS'
numpy==1.26.4
opencv-python==4.11.0.86
torch==2.8.0+cu128
torchvision==0.23.0+cu128
CONSTRAINTS
python -m pip install "numpy==1.26.4"
python -m pip install --constraint "$PERCEPTION_WS/perception-constraints.txt" \
  torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --constraint "$PERCEPTION_WS/perception-constraints.txt" \
  "opencv-python==4.11.0.86" "scipy==1.17.1" "transformers==4.57.3" \
  "huggingface-hub==0.36.2" "supervision==0.28.0" "timm==1.0.27" \
  "hydra-core==1.3.2" "omegaconf==2.3.0" "iopath==0.1.10" \
  "addict==2.4.0" "yapf==0.43.0" "pycocotools==2.0.11" \
  pillow tqdm requests pyyaml httpx "openai==2.38.0" \
  "setuptools==69.5.1" wheel ninja "empy==3.3.4" catkin-pkg lark colcon-common-extensions
python -m pip check
python - <<'PY_CHECK'
import sys, numpy, scipy, cv2, torch, torchvision
import hydra, omegaconf, iopath, addict, yapf, pycocotools
import transformers, supervision, timm, openai, httpx, colcon_core
assert sys.version_info[:2] == (3, 12)
assert numpy.__version__ == "1.26.4"
print("numpy/scipy/opencv:", numpy.__version__, scipy.__version__, cv2.__version__)
print("torch/vision/CUDA:", torch.__version__, torchvision.__version__, torch.version.cuda)
assert torch.cuda.is_available(), "先解决 GPU 驱动/CUDA 可见性，再编译模型扩展"
PY_CHECK
```

使用了 [PyTorch 2.8 官方 CUDA 12.8 安装源](https://pytorch.org/get-started/previous-versions/)。
默认链路不需要 `torchaudio`、Pixal3D/TRELLIS 扩展或训练依赖。
上述依赖来自 vendored GDINO 的 `requirements.txt`、SAM2 的 `setup.py`、Cutie 推理源码及
`perception/robot_perception` 的 imports；除兼容 OpenCV 外，主要固定版本沿用本仓快照。
这里仅安装一种 `opencv-python`，不要同时安装 headless/contrib 分发包来覆盖同一个 `cv2`。

NumPy/OpenCV约束有明确包元数据依据：
[OpenCV 4.11.0.86](https://pypi.org/pypi/opencv-python/4.11.0.86/json) 在 Python 3.12 接受 NumPy 1.26，
[SciPy 1.17.1](https://pypi.org/pypi/scipy/1.17.1/json) 要求 NumPy ≥1.26.4，
[Supervision 0.28](https://pypi.org/pypi/supervision/0.28.0/json) 接受这组 OpenCV/NumPy。
`openai`/`httpx` 是感知代码导入的客户端库，即使不开 VLM 也保留；不因此启动 VLM 服务。
后续向这个环境加包时继续传入同一 constraints 文件，避免依赖解析升级 NumPy 或替换 CUDA wheel。

这是首次创建命令；已有同名环境时先确认其来源，不覆盖其他部署。
`pip check`、上述导入和后面的 `_C` 检查都通过后，再继续 ROS 构建及模型推理验证。

以下下载只写模型目录；使用此感知环境中已有的 `huggingface_hub`。
配置文件从本仓库复制，BERT 路径改在复制品里，不修改 vendored 配置：

```bash
mkdir -p "$PERCEPTION_WEIGHTS_DIR/grounding_dino" "$PERCEPTION_WEIGHTS_DIR/sam2"
curl -fL --retry 3 \
  https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth \
  -o "$PERCEPTION_WEIGHTS_DIR/grounding_dino/groundingdino_swint_ogc.pth"
python - <<'PY'
import os, re
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
root = Path(os.environ["PERCEPTION_WEIGHTS_DIR"])
hf_hub_download("facebook/sam2.1-hiera-tiny", "sam2.1_hiera_tiny.pt",
                local_dir=str(root / "sam2"))
snapshot_download("google-bert/bert-base-uncased", local_dir=str(root / "bert-base-uncased"),
                  allow_patterns=["config.json", "model.safetensors", "tokenizer.json",
                                  "tokenizer_config.json", "vocab.txt"])
source = Path(os.environ["ROBOT_PERCEPTION_DIR"]) / "Grounded-SAM-2/grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py"
text, changed = re.subn(r'^text_encoder_type\s*=.*$',
                        lambda _: "text_encoder_type = " + repr(str(root / "bert-base-uncased")),
                        source.read_text(), flags=re.MULTILINE)
assert changed == 1, "Grounding DINO 配置结构变化，请核对 text_encoder_type"
(root / "grounding_dino/GroundingDINO_SwinT_OGC.py").write_text(text)
PY
```

配置原来使用相对路径 `gdino_checkpoints/bert-base-uncased`；仅下载 `.pth` 不足以离线启动。
下载模型前按对应作者页面处理访问许可。RMBG/DINOv3 的官方 Hugging Face 页面要求访问授权；
授权失败时应完成账号授权，不能把错误页当成模型文件。

启用 Cutie 跟踪时，预先执行仓库自己的下载器，并准备其 ResNet 缓存：

```bash
python "$ROBOT_PERCEPTION_DIR/Cutie/cutie/utils/download_models.py"
python - <<'PY'
import torch
for name in ("resnet18-5c106cde.pth", "resnet50-19c8e357.pth"):
    torch.hub.load_state_dict_from_url("https://download.pytorch.org/models/" + name,
                                        map_location="cpu", check_hash=True)
PY
```

Cutie 下载器检查两份 `.pth` 的 MD5，缺失/不匹配时重新下载；只放 `cutie-base-mega.pth`
仍可能触发另一个下载。离线迁移还要带上同一 `TORCH_HOME` 缓存，不能只复制主权重。
如果暂不使用时序跟踪，可在 launch 中明确设置 `use_temporal_tracking:=false use_cutie_tracking:=false`。

## 4. 编译模型扩展和 ROS 感知包

GDINO 在 CUDA 上需要其自定义 C++/CUDA 算子。工具链须与当前 PyTorch CUDA 和 GPU 架构兼容。
以下命令在已激活的主感知环境编译扩展，`CUDA_HOME` 按实际 Toolkit 设置：

```bash
# 先确认 nvcc 与 PyTorch CUDA 版本；CUDA_HOME 需指向实际 Toolkit。
nvcc --version
python -c "from torch.utils.cpp_extension import CUDA_HOME; print(CUDA_HOME)"
(
  cd "$ROBOT_PERCEPTION_DIR/Grounded-SAM-2/grounding_dino"
  python setup.py build_ext --inplace
)
PYTHONPATH="$ROBOT_PERCEPTION_DIR/Grounded-SAM-2:$ROBOT_PERCEPTION_DIR/Grounded-SAM-2/grounding_dino${PYTHONPATH:+:$PYTHONPATH}" \
  python -c "from grounding_dino.groundingdino import _C; print(_C.__file__)"
```

导入 `_C` 失败时不能继续声称 GPU 检测已安装完成。仓库中的 `setup.py` 含特定架构编译选项；
其他 GPU/Toolkit 组合需按扩展实际报错处理，不要通过卸载原语 venv 的系统数值库解决。

只构建感知消息与感知包，不把原语/手驱动纳入这个 Conda 构建：

```bash
source /opt/ros/jazzy/setup.bash
rosdep install --from-paths "$GESTURE_SOURCE/perception" \
  "$GESTURE_SOURCE/messages/robot_perception_msgs" --ignore-src -y
python -c "import rclpy, cv_bridge, colcon_core, catkin_pkg, em, lark; print('ROS/build imports OK')"
python "$(command -v colcon)" --log-base "$PERCEPTION_WS/log" build \
  --base-paths "$GESTURE_SOURCE/perception" "$GESTURE_SOURCE/messages/robot_perception_msgs" \
  --build-base "$PERCEPTION_WS/build" --install-base "$PERCEPTION_WS/install" \
  --packages-select robot_perception_msgs robot_perception \
  --cmake-args "-DPython3_EXECUTABLE=$(command -v python)" "-DPYTHON_EXECUTABLE=$(command -v python)"
source "$PERCEPTION_WS/install/local_setup.bash"
head -n 1 "$PERCEPTION_WS/install/robot_perception/lib/robot_perception/detection_bbox"
python -c "from robot_perception_msgs.msg import LabeledBBox3DArray; import robot_perception; print(robot_perception.__file__)"
```

入口 shebang 应指向刚才的感知 Conda Python，包路径应是这个新工作空间。
`rosdep` 需在宿主按安装指南准备；`colcon` 的 Python 模块在第 3 节装入感知 Conda 环境，
构建时明确使用当前 Python，避免系统脚本的 shebang 切回系统解释器。这里不创建相机或硬件驱动。

**为什么这里使用直接命令：** 0.2.2 的 `perception/scripts/perception_env.sh` 没有定义
旧构建脚本所需的 `PERCEPTION_ENV_NAME`、`WS_ROOT`、`PERCEPTION_MSGS_DIR`、`PKG_DIR`，
也没有定义 VLM 脚本所需的 `VLM_ENV_NAME`、`VLM_HOST`、`VLM_PORT` 等。
因此直接运行旧 `build_perception.sh` / `setup_vlm_env.sh` / `start_minicpm_vlm_server.sh`
可能失败。本文使用明确的环境、解释器和路径完成相应步骤。

## 5. 启动默认检测与接入原语

新终端先重设第 1 节的变量，再激活主感知环境和 ROS overlay：

```bash
eval "$(conda shell.bash hook)"
conda activate gesture-perception-0.2.2
source /opt/ros/jazzy/setup.bash
source "$PERCEPTION_WS/install/local_setup.bash"
# 此处 77 是示例，必须与相机、原语和 TCP 发布者的实际域一致。
export ROS_DOMAIN_ID=77
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
```

`LOCALHOST` 只允许本机发现；跨机器部署需所有参与节点采用能互相发现的配置。
启动前准备相机话题、相机内参以及**当前机器的真实手眼标定**。
仓库自带 `full_calibration_result.txt` 不能作为任意机器的标定结果。

```bash
export GESTURE_CALIB_FILE=/absolute/path/to/your/full_calibration_result.txt
test -f "$GESTURE_CALIB_FILE"
ros2 launch robot_perception detection_bbox.launch.py \
  cam_ns:="['/camera_head']" calib_file:="$GESTURE_CALIB_FILE" \
  use_vlm_detect:=false use_scene_understand:=false \
  use_pixel3d:=false use_stereo_depth:=false launch_rviz:=false
```

此命令只启动感知，不启动手驱动或原语节点。默认采用已有产线物体 prompt；
自定义物体可额外指定 `auto_industry_scene_prompts:=false text_prompts:="['blue box.']"`。

| 接入数据 | 类型/要求 |
| --- | --- |
| `/camera_head/color/image_raw`、`/camera_head/depth/image_raw` | RGB/深度 `sensor_msgs/Image` |
| `/camera_head/color/camera_info` | `sensor_msgs/CameraInfo`，须匹配实际图像 |
| `/camera_head/detection_bbox/bboxes_3d` | `robot_perception_msgs/LabeledBBox3DArray`，原语默认订阅的几何输出 |
| `/{side}_hand/tcp_pose` | 上层臂/TF桥发布的 `geometry_msgs/PoseStamped`，本包不生成 |
| `/joint_states` 与该侧接触反馈 | 实际关节状态，以及对应手型的电流/力矩/触觉，须持续新鲜 |
| FK 与门控 | 对应真实手型/侧别的 URDF、Pinocchio；需要门控的原语还需同侧 GraspGate |

目标、TCP 和标定必须采用一致的坐标系、米/弧度单位和时间基准。
检测框出现不代表原语已经满足门控或力反馈条件；缺少标定时 3D 框可能为空。
原语也可接其他感知的 `labeled_poses`，默认名为 `/camera_head/perception/labeled_poses`；
本检测示例输出 `detection_bbox/bboxes_3d`，不发布前一话题。

只读检查示例：

```bash
ros2 topic info /camera_head/detection_bbox/bboxes_3d
ros2 topic echo --once /camera_head/detection_bbox/bboxes_3d
ros2 topic info /right_hand/tcp_pose
```

原语节点/网页仍在原语自己的终端按[启动指南](getting-started.md)运行。
感知原语需要门控时，按[ROS 接口文档](../hand_gesture_primitives/docs/interfaces.md)
配置 `launch_gate:=true`；命令发往 `/{side}_hand/gesture_cmd`，不要绕过门控直接发执行话题。
本文不提供直接驱动真机的测试动作。

## 6. 可选：MiniCPM-V 检测或场景理解

下载脚本可能安装/升级下载 SDK，因此用独立下载 venv 获取权重，不修改主感知或原语环境：

```bash
cd "$GESTURE_SOURCE"
/usr/bin/python3 -m venv "$HOME/gesture-model-download-0.2.2"
source "$HOME/gesture-model-download-0.2.2/bin/activate"
bash third_party/robot_perception/scripts/download_minicpm_v4_6_gptq.sh
export MINICPM_WEIGHT_DIR="$PERCEPTION_WEIGHTS_DIR/minicpm-v-4.6-gptq"
bash perception/scripts/sync_minicpm_weights.sh
deactivate
```

下载脚本固定暂存到 `third_party/robot_perception/models/minicpm-v-4.6-gptq`；
同步脚本从该目录复制到 `MINICPM_WEIGHT_DIR`，并用 `rsync --delete` 对齐此模型目录。
不要把 `MINICPM_WEIGHT_DIR` 设为混放其他文件的根目录。

VLM 需要自己的 NumPy 2/Transformers 5/GPTQ 环境，不能复用第 3 节的 ROS 感知环境。
重设第 1 节变量和上述 `MINICPM_WEIGHT_DIR`，在新终端创建独立环境。
下面是 **Linux x86_64 / CUDA 12.8 的安装参考组合**。已核对版本元数据与上游兼容说明，
尚未验证此组合在目标 GPU 上的模型加载和推理：

- [模型作者说明](https://huggingface.co/openbmb/MiniCPM-V-4.6-GPTQ#inference-with-transformers)
  要求 Transformers ≥5.7，服务依赖使用 `serving` extra；此处保留快照中的 5.12.1。
- [GPTQModel 7.1 元数据](https://pypi.org/pypi/gptqmodel/7.1.0/json)
  要求 Torch ≥2.8、Transformers ≥5.4、Python 3.12 下 NumPy 2.2.6、TorchAO ≥0.16。
- 为避免快照的 Torch 2.8/TorchAO 0.17 错配，选择 Torch 2.10.0、Torchvision 0.25.0 和
  TorchAO 0.16.0：前两者有 [PyTorch 官方 cu128 配方](https://pytorch.org/get-started/previous-versions/)，
  后者在 [TorchAO 官方兼容表](https://github.com/pytorch/ao/issues/2919) 对应 Torch 2.10，
  [官方 cu128 索引](https://download.pytorch.org/whl/cu128/torchao/) 提供 Linux x86_64 的 ABI3 wheel。

```bash
eval "$(conda shell.bash hook)"
conda create -n gesture-perception-vlm-0.2.2 python=3.12 pip -y
conda activate gesture-perception-vlm-0.2.2
cat > "$PERCEPTION_WS/vlm-constraints.txt" <<'CONSTRAINTS'
torch==2.10.0+cu128
torchvision==0.25.0+cu128
torchao==0.16.0+cu128
numpy==2.2.6
transformers==5.12.1
gptqmodel==7.1.0
optimum==2.2.0
CONSTRAINTS
python -m pip install "numpy==2.2.6"
python -m pip install --constraint "$PERCEPTION_WS/vlm-constraints.txt" \
  torch==2.10.0 torchvision==0.25.0 torchao==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install --constraint "$PERCEPTION_WS/vlm-constraints.txt" \
  "transformers[serving]==5.12.1" "optimum==2.2.0" "gptqmodel==7.1.0"
python -m pip check
python - <<'PY_CHECK'
import importlib.metadata as md
import torch, torchvision, torchao, gptqmodel
from transformers import AutoModelForImageTextToText, AutoProcessor
import fastapi, uvicorn
for name in ("torch", "torchvision", "torchao", "transformers", "gptqmodel", "optimum", "numpy"):
    print(name, md.version(name))
assert torch.cuda.is_available(), "VLM 环境无法使用 GPU"
PY_CHECK
transformers serve --help
```

GPTQModel 7.1 在 PyPI 发布的是源码分发包，量化后端还涉及本机原生/JIT kernel；
元数据兼容不证明所有 GPU 或模型 kernel 都能运行。若编译或上面的导入失败，
保留错误信息，按 [GPTQModel v7.1 安装说明](https://github.com/ModelCloud/GPTQModel/tree/v7.1.0#install)
与目标 GPU 工具链适配；不要删掉约束强装，也不要向主感知环境加入 NumPy 2。
可继续使用第 5 节的默认 GDINO 路线，或接入已有经过验证的 VLM HTTP 服务。

所有检查通过后才启动服务：

```bash
transformers serve "$MINICPM_WEIGHT_DIR" --host 127.0.0.1 --port 8000 --continuous-batching
```

端口 8000 必须可用；直接命令不会杀掉占用端口的其他服务。
在另一个终端检查 `curl -f http://127.0.0.1:8000/health`。
健康检查只证明服务就绪，模型加载与实际图像推理还需查看服务日志。
`/v1/models` 可能不登记本地权重，检测端 `vlm_model` 使用同一绝对权重目录。

回到第 5 节的主感知环境，替换 launch 的对应参数：

```bash
ros2 launch robot_perception detection_bbox.launch.py \
  calib_file:="$GESTURE_CALIB_FILE" use_vlm_detect:=true \
  vlm_base_url:=http://127.0.0.1:8000/v1 \
  vlm_model:="$PERCEPTION_WEIGHTS_DIR/minicpm-v-4.6-gptq" \
  use_scene_understand:=false use_pixel3d:=false launch_rviz:=false
```

仅增加场景理解时，使用 `use_vlm_detect:=false use_scene_understand:=true`，
同时保留 GDINO/SAM2 模型和 VLM 服务。若环境中的 Transformers CLI 不接受所示参数，
以这个 VLM 环境的 `transformers serve --help` 为准，并核对上述依赖版本和安装日志。

## 7. 可选：Pixal3D 完整 mesh

这条路径需要更多显存、额外源码和 CUDA 扩展。普通 3D bbox 不要求开启 Pixal3D。
在主感知环境中准备源码；目标目录已有内容时先核对来源，不覆盖：

```bash
cd "$GESTURE_SOURCE"
git clone https://github.com/TencentARC/Pixal3D.git "$ROBOT_PERCEPTION_DIR/Pixal3D"
git -C "$ROBOT_PERCEPTION_DIR/Pixal3D" rev-parse HEAD
```

记录此次实际源码提交；0.2.2 并未锁定这份外部源码。根据其
[官方安装说明](https://github.com/TencentARC/Pixal3D#installation)
安装对应 GPU/PyTorch 的依赖和扩展，在主感知环境确认 `pixal3d.pipelines` 能导入后继续。
外部 CUDA wheel 必须匹配实际 PyTorch、CUDA 和 GPU 架构，不应直接套用其他环境的二进制包。

三个下载脚本的默认位置是**仓库父目录下的 `models/`**；以下在独立下载 venv 中显式指定运行目录：

```bash
/usr/bin/python3 -m venv "$HOME/gesture-model-download-0.2.2"
source "$HOME/gesture-model-download-0.2.2/bin/activate"
PIXAL3D_WEIGHT_DIR="$PERCEPTION_WEIGHTS_DIR/pixal3d-t" \
  bash third_party/robot_perception/scripts/download_pixal3d.sh
RMBG2_WEIGHT_DIR="$PERCEPTION_WEIGHTS_DIR/RMBG-2.0" \
  bash third_party/robot_perception/scripts/download_rmbg2.sh
DINOV3_VITL16_WEIGHT_DIR="$PERCEPTION_WEIGHTS_DIR/dinov3-vitl16-pretrain-lvd1689m" \
  bash third_party/robot_perception/scripts/download_dinov3_vitl16.sh
bash perception/config/pixel3d/install_offline.sh
deactivate
```

下载脚本有 PyTorch 时优先尝试 ModelScope，可回退 Hugging Face；这个轻量下载 venv 无 PyTorch，
会使用 Hugging Face 分支，并将下载 SDK 安装在下载 venv 内。
访问授权以对应模型页面为准。`install_offline.sh` 检查三个模型目录和主 checkpoint，
首次备份现有 `pipeline.json` 为 `.bak`，然后安装本仓库的离线路径配置。

准备完整后，在第 5 节检测命令上增加：

```text
use_pixel3d:=true pixel3d_low_vram:=true
pixel3d_model_path:=<PERCEPTION_WEIGHTS_DIR的实际路径>/pixal3d-t
pixel3d_offline_config:=<GESTURE_SOURCE的实际路径>/perception/config/pixel3d/pixel3d_offline.example.json
```

这段是参数清单，不是一条可独立执行的 shell 命令。`low_vram` 不能保证目标 GPU 一定装得下。
FoundationStereo 是另一项可选深度后端，当前树不带其源码或 TensorRT engine；
启用 `use_stereo_depth:=true` 前需按作者说明生成与目标机器匹配的 `.plan`，
并提供推理实现和 IR 相机输入，不能仅改一个参数完成部署。

## 8. 问题定位与离线迁移

| 现象 | 先检查 |
| --- | --- |
| 找不到模型/配置 | 绝对 `PERCEPTION_WEIGHTS_DIR`、对应文件名；GDINO 还需配置与 BERT |
| `_C`、CUDA 扩展导入失败 | 主感知解释器、编译产物、`CUDA_HOME`、PyTorch CUDA、GPU 架构 |
| 首次运行仍访问网络 | Cutie 第二份权重、ResNet/Torch缓存、BERT、Pixal3D辅助模型是否齐全 |
| 2D正常但没有 world 3D框 | 当前标定、深度、相机内参、世界坐标 ROI与frame；不是网页问题 |
| 网页显示视觉/FK/反馈不可用 | 同侧 TCP、对应URDF/FK、目标新鲜度、实际接触反馈与门控 |
| `VLM_ENV_NAME` 等未定义 | 使用本文的直接环境/构建/启动命令，勿混用旧一键脚本 |
| 模型就绪但无法发现ROS话题 | 同一domain、发现范围、网络与实际话题名 |

跨机复制时保留权重目录、BERT路径配置、所需缓存及外部源码提交，重新配置目标机的绝对路径。
GDINO配置中的 `text_encoder_type` 也要指向目标机路径；源机器的 CUDA/TensorRT 产物不保证跨机适配。
复制模型文件不等于迁移手眼标定或手驱动配置。

详细消息字段见[感知输出话题](../perception/docs/output_topics.md)，
原语所需接口见[ROS 接口文档](../hand_gesture_primitives/docs/interfaces.md)。

示例生成的 `perception-constraints.txt`、`vlm-constraints.txt` 分别约束对应的参考环境。
完成部署后记录实际包版本、GPU/驱动/CUDA、模型修订和外部源码提交，供后续复现。
