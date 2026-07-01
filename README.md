# hand_gesture_primitives

# 1\. 项目简介

[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hand: O20](https://img.shields.io/badge/Hand-O20-orange)](https://github.com/linker-bot/linkerhand-o20-ros2)
[![Hand: O6](https://img.shields.io/badge/Hand-O6-blue)](https://github.com/linker-bot/linkerhand-o6-ros2)

当前绝大多数自动化产线、小型机器人平台仍普遍采用简易二指爪夹作为末端执行器，反观五指灵巧手虽具备高自由度与丰富接触形态，但在真实产线、实验室和桌面操作场景中，**从「能动手」到「稳定抓得准」** 之间仍隔着一道工程鸿沟：关节空间复杂、感知噪声大、抓取策略难以复用。本项目由灵心巧手算法团队研发推出，针对灵巧手落地难、控制繁琐的工程痛点开展优化迭代。

**hand_gesture_primitives** 的目标，是把灵巧手控制从「逐关节调参」升级为 **可组合、可感知、可门控的手势原语（Gesture Primitives）**，让开发者用一条字符串命令触发经过验证的抓取/释放动作，并与视觉感知、机械臂 TCP 位姿、触觉/电流反馈协同工作——**加速灵巧手在真实场景中的部署与迭代**。

本项目基于 ROS2 Jazzy 版本开发，适配LinkerBot灵巧手，提供标准化手势原语控制，基于触觉与力矩的实时反馈实现自适应抓取。


项目详细文档:
[手势原语发布文档](https://alidocs.dingtalk.com/i/nodes/R4GpnMqJzGzNeyYDCkm0qyox8Ke0xjE3)

# 2\. 快速开始
## 2\.1 运行环境

基础运行依赖环境，部署前需提前配置完成：

- ROS 2 Jazzy

- Python 3 \+ numpy \+ scipy

- LinkerHand O20 硬件驱动预先正常运行

## 2\.2 权重与源码下载
此项目提供一个感知示例供调试使用，使用方可根据需求使用自己的感知模块。

下载视觉感知所需预训练权重、第三方模型及源码，执行以下命令：

```bash
# SAM 2.1 Hiera-Tiny 权重
https://huggingface.co/facebook/sam2.1-hiera-tiny

# Grounding DINO SwinT 权重
https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

# Cutie 权重
https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth

# Pixel3D 全家桶权重
bash third_party/robot_perception/scripts/download_pixal3d.sh
bash third_party/robot_perception/scripts/download_rmbg2.sh
bash third_party/robot_perception/scripts/download_dinov3_vitl16.sh

# VLM 量化模型权重
bash third_party/robot_perception/scripts/download_minicpm_v4_6_gptq.sh

# Pixal3D 源码克隆
git clone https://github.com/TencentARC/Pixal3D.git $ROBOT_PERCEPTION_DIR/Pixal3D
```

## 2\.3 项目编译

进入工作空间，依次编译机械手驱动、手势原语核心包与感知功能包：

```bash
cd ~/your_ws

# 编译机械手驱动包
colcon build --packages-select linker_hand_o20_ros2

# 编译手势原语核心功能包
colcon build --packages-select hand_gesture_primitives
source install/setup.bash

# 编译全套感知功能包
bash src/hand_gesture_primitives/perception/scripts/build_perception.sh
```

## 2\.4 节点启动流程

按照「感知节点 → 手势控制节点」顺序启动，支持纯手势、门控抓取两种模式：

```bash
# 1. 启动视觉感知节点 （此项目提供一个感知示例供调试使用，使用方可根据需求使用自己的感知模块）
bash src/perception/scripts/start_minicpm_vlm_server.sh
ros2 launch robot_perception detection_bbox.launch.py use_vlm_detect:=false use_scene_understand:=true auto_industry_scene_prompts:=false use_pixel3d:=false pixel3d_low_vram:=false

# 2. 启动手势节点 + GraspGate抓取门控
ros2 launch hand_gesture_primitives gesture_node.launch.py \
  hand_side:=left launch_gate:=true
```

## 2\.5 手势指令发送

通过ROS2话题发布字符串指令，控制灵巧手执行对应动作：

```bash
# 无接触手势原语
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'open'"
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'init'"

# 视觉力控有接触手势原语
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'index_ring_by_vision'"

# 力控有接触手势原语
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'thumb_adduction_grip'"
```


# 3\. 手势原语清单


## 3\.1 规则说明
- **支持型号**：与 `config/o20.yaml`、`config/o6.yaml` 中 `primitives.supported` allowlist 一致（**已实机验证**）；L25 等配置预留型号尚未验证，不在下表列出。

- **无接触手势原语**： 仅输入「话题字符串指令 \+ 当前关节状态」，无需视觉、TCP位姿。任意时刻可执行，不经过抓取门控。

- **力控有接触手势原语**：预成型 → 手指力控闭合 → 拇指力控闭合 → 自适应保持；持握 \>800mA 自动卸力；过载保护：单关节 \>1000mA 持续 \>2s 自动归位。任意时刻可执行，不经过抓取门控。

- **视觉力控有接触手势原语**：输入包含「话题字符串指令 \+ 当前关节状态 \+ TCP位姿 \+ 视觉3D bounding box」，包络类原语需经过 GraspGate 三判定 \+ 3帧防抖校验。 经过门控，避免过早闭合导致抓取失败。

## 3\.2 无接触手势原语

|序号|指令名称|功能说明|支持型号|输入信息|
|---|---|---|---|---|
|1|open|五指完全张开，复位基础姿态|O20, O6|话题字符串指令、当前关节状态|
|2|init|手部复位至安全初始化中立位置|O20, O6|话题字符串指令、当前关节状态|
|3|fist|五指完全握拳固定姿态|O20, O6|话题字符串指令、当前关节状态|
|4|pinch|拇指\+食指标准精细对捏|O20, O6|话题字符串指令、当前关节状态|
|5|point|食指单独伸直指向，其余手指弯曲收拢|O20, O6|话题字符串指令、当前关节状态|
|6|ok\_sign|OK手势：拇指食指对接成环，剩余三指伸直|O20, O6|话题字符串指令、当前关节状态|
|7|v\_sign|V字手势：食指、中指伸直，无名指小指弯曲|O20, O6|话题字符串指令、当前关节状态|
|8|relax\_grip|缓慢卸力，放松当前握持姿态|O20, O6|话题字符串指令、当前关节状态|
|9|release|完全释放抓取物体，五指舒展打开|O20, O6|话题字符串指令、当前关节状态|

## 3\.3 力控有接触手势原语

|序号|指令名称|功能说明|支持型号|输入信息|
|---|---|---|---|---|
|1|thumb\_adduction\_grip|拇指侧向夹持，支持 prep预就位 / close 闭合两阶段执行|O20, O6|话题字符串指令、当前关节状态|
|2|index\_middle\_adduction\_grip|食指\+中指侧向对夹夹持，夹取香烟、笔杆、细棒等细长物体|O20|话题字符串指令、当前关节状态|
|3|ring|食中指环形包络，拇指与食指\+中指同时力控闭合自适应抓取|O20, O6|话题字符串指令、当前关节状态|
|4|middle\_ring|中指环形包络，拇指与中指上下错开力控闭合自适应抓取|O20, O6|话题字符串指令、当前关节状态|
|5|tripod|三指捏取，拇指与食指\+中指指尖对捏形成三角支撑|O20, O6|话题字符串指令、当前关节状态|
|6|index\_pinch|食指捏取，拇指与食指指尖精确对捏，拾取螺丝/针/薄片等小物体|O20, O6|话题字符串指令、当前关节状态|
|7|middle\_pinch|中指捏取，拇指与中指指尖对捏，食指保持自由避让|O20, O6|话题字符串指令、当前关节状态|

## 3\.4 视觉力控有接触手势原语

|序号|指令名称|功能说明|支持型号|输入信息|
|---|---|---|---|---|
|1|index\_ring\_by\_vision|感知自适应精细环握，适配小型薄物体，auto映射precision模式|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|2|large\_wrap\_by\_vision|全包络强力抓取，适配大尺寸多面体，auto映射power模式|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|3|middle\_ring\_by\_vision|拇指\+中指单指对夹，高精度薄物体精细抓取|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|4|ring\_by\_vision|拇指\+食指\+中指环形包络，适配中等圆柱/扁平件|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|5|small\_warp\_by\_vision|拇指配合全部四指小型全包络，小物体高力量抓取|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|6|no\_index\_warp\_by\_vision|无食指参与包络，仅拇指\+中/无名/小指，保留食指自由作业|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|7|hook\_by\_vision|四指钩握、无拇指参与，适配把手、挂钩、环形物件|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|8|index\_pinch\_by\_vision|拇指食指超精细窄幅对捏，微小物体拾取|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|9|middle\_pinch\_by\_vision|拇指中指单独精细对捏，避开食指干涉|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|10|tripod\_by\_vision|三脚架三指夹持（拇指\+食指\+中指），三点稳定支撑抓取|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|11|palmar\_by\_vision|掌面全包络大面积贴合，适配大尺寸平整物体|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|12|parallel\_extension\_by\_vision|手指平行伸展对夹，适配薄板、片状工件|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|13|disk\_by\_vision|圆盘环形环绕抓取，适配圆形盘状、环状物体|O20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
