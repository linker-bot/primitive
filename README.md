# hand_gesture_primitives
## 0. 项目简介
[![ROS2 Jazzy](https://img.shields.io/badge/ROS2-Jazzy-blue)](https://docs.ros.org/en/jazzy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Hand: O20](https://img.shields.io/badge/Hand-O20-orange)](https://github.com/linker-bot/linkerhand-o20-ros2)
[![Hand: O6](https://img.shields.io/badge/Hand-O6-blue)](https://github.com/linker-bot/linkerhand-o6-ros2)

当前绝大多数自动化产线、小型机器人平台仍普遍采用简易二指爪夹作为末端执行器，反观五指灵巧手虽具备高自由度与丰富接触形态，但在真实产线、实验室和桌面操作场景中，**从「能动手」到「稳定抓得准」** 之间仍隔着一道工程鸿沟：关节空间复杂、感知噪声大、抓取策略难以复用。本项目由灵心巧手算法团队研发推出，针对灵巧手落地难、控制繁琐的工程痛点开展优化迭代。

**hand_gesture_primitives** 的目标，是把灵巧手控制从「逐关节调参」升级为 **可组合、可感知、可门控的手势原语（Gesture Primitives）**，让开发者用一条字符串命令触发经过验证的抓取/释放动作，并与视觉感知、机械臂 TCP 位姿、触觉/电流反馈协同工作——**加速灵巧手在真实场景中的部署与迭代**。

本项目基于 ROS2 Jazzy 版本开发，适配LinkerBot灵巧手，提供标准化手势原语控制，基于触觉与力矩的实时反馈实现自适应抓取。

项目详细文档:
[手势原语发布文档](https://alidocs.dingtalk.com/i/nodes/R4GpnMqJzGzNeyYDCkm0qyox8Ke0xjE3)

## 1. 文档导航与使用范围

| 你要做什么 | 从哪里开始 |
| --- | --- |
| 下载源码、准备环境、安装编译、验证安装 | [安装与启动指南](docs/getting-started.md) |
| 远程打开模拟网页、不使用机械手 | [安装与启动指南：Mock与SSH访问](docs/getting-started.md) |
| 接入已有ROS驱动，运行单手或双手 | [安装与启动指南：ROS部署](docs/getting-started.md) |
| 理解网页每个按钮、编辑多帧、速度与进度、导入导出 | [网页功能与操作手册](docs/teach-panel-user-guide.md) |
| 下载感知权重与模型源码，部署可选视觉/VLM示例 | [可选感知环境与权重部署](docs/perception-setup.md) |
| 查看公共接口、控制约束、数据格式和开发测试 | [工作台技术说明](docs/workbench.md) |
| 查看版本功能和已知限制 | [0.2.2发布说明](docs/releases/0.2.2.md) |

先选择需要的运行范围：

| 范围 | 环境 | AI权重 | 额外条件 |
| --- | --- | --- | --- |
| Mock网页、自定义多帧、速度/进度 | Python3.12独立venv | 不需要 | 模拟反馈，不启动ROS或驱动 |
| 真实基础原语、自定义示教 | Ubuntu24.04、ROS2 Jazzy、系统Python3.12 venv | 不需要 | 已有机械手驱动、关节反馈、实际控制器URDF |
| 纯力控原语 | 上述ROS环境 | 不需要 | 对应手型的有效电流/力矩/触觉反馈 |
| 感知原语 | 上述ROS环境与感知输入 | 可由外部感知系统提供结果 | 几何目标、TCP、FK以及对应门控/反馈条件 |
| 仓库自带检测/分割/VLM/Pixal3D示例 | 单独的感知Conda/CUDA环境 | 按启用模块下载 | 相机、模型文件与ROS话题/坐标系配置 |

Pinocchio用于FK。基础与自定义动作不要求安装视觉模型；依赖FK的感知执行需要
Pinocchio和真实手型URDF。安装包自带O6右手模型，左手及其他缺失模型按
[URDF说明](data/urdf/README.md)提供。网页真实关节换算所用的`robot_description`
必须匹配驱动，这与FK模型是否可选是两个不同条件。

## 2. 快速开始

### 2.1 获取源码

通过 HTTPS 克隆源码；AI权重按所需感知模块另行下载。

```bash
git clone https://github.com/linker-bot/primitive.git
cd primitive
export GESTURE_SOURCE="$(pwd)"
```

使用 `git rev-parse HEAD` 记录所用源码版本。
完整下载、目录约定与固定提交的方法见[安装指南](docs/getting-started.md)。

### 2.2 先运行独立Mock网页

以下命令从仓库根目录执行，使用Python3.12；首次使用venv的Ubuntu机器先安装
`python3-venv`。姿态和运行状态使用专门的Mock目录。

```bash
/usr/bin/python3 -m venv "$HOME/hand-gesture-mock-0.2.2"
source "$HOME/hand-gesture-mock-0.2.2/bin/activate"
python -m pip install './hand_gesture_primitives[teach]'
export HAND_GESTURE_POSE_DIR="$HOME/hand-gesture-data/mock-poses"
export HAND_GESTURE_RUNTIME_STATUS_PATH="$HOME/hand-gesture-data/mock-runtime.json"
teach_panel --mock --pose-dir "$HAND_GESTURE_POSE_DIR" \
  --hand left:o6 --hand right:o6 --host 127.0.0.1 --port 18002
```

浏览器打开 `http://127.0.0.1:18002/`。首页是手掌滑条、内置原语与多帧姿态库；
`/workbench` 是按住推进、菜单与草稿编辑器。SSH远程访问方式见安装指南。

### 2.3 ROS环境配置与编译

0.2.2网页依赖**Pydantic2**；仅执行裸`colcon build`不会安装它和Web依赖。
使用[安装指南](docs/getting-started.md)中的系统依赖准备步骤，再运行：

```bash
source /opt/ros/jazzy/setup.bash
export HAND_GESTURE_INSTALL="$HOME/hand-gesture-ros-0.2.2"
/usr/bin/python3 hand_gesture_primitives/scripts/validate_ros_install.py \
  --output "$HAND_GESTURE_INSTALL"
source "$HAND_GESTURE_INSTALL/venv/bin/activate"
source "$HAND_GESTURE_INSTALL/workspace/install/setup.bash"
```

输出目录必须是新的、位于源码目录外。脚本构建4个原语/消息包并用Mock验证安装，
保持apt NumPy/SciPy；机械手驱动、相机和感知包另行准备。真实节点启动可能执行
`init`，请按安装指南完成控制器状态、URDF、侧别和话题准备后再启动。

### 2.4 权重与感知源码

只做手势示教、速度或进度控制，可直接使用上述步骤。
需要仓库检测示例时，按[可选感知部署](docs/perception-setup.md)下载SAM2.1、
GroundingDINO、Cutie等基础模型；VLM和Pixal3D相关权重按需配置。
该指南逐项列出官方来源、仓内脚本、实际目标目录、环境和编译命令。

### 2.5 网页与指令入口

[网页操作手册](docs/teach-panel-user-guide.md)详细说明“回放／定点／载入”、
添加中间态、上下排序、速度、进度、只读、共享编辑暂停，以及导出/导入/出厂预设。
注意同一面板实例的速度倍率作用于已配置的左右手；进入编辑也可能暂停两侧动作。

ROS节点与驱动已按指南启动后，可发送以下命令（会控制对应侧机械手）：

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right open
ros2 run hand_gesture_primitives pub_gesture_cmd -- right stop
```

左手将`right`改为`left`。公开指令入口为`/{side}_hand/gesture_cmd`；启用GraspGate时
仍发公开入口，由门控转发。自定义、进度和旧接口见
[接口说明](hand_gesture_primitives/docs/interfaces.md)。

原语实现及公司独立开发的工作台代码采用 [MIT 许可证](LICENSE)。O6 模型资源和
`hand_force_msg` 保留原有 [Apache-2.0 许可](LICENSE.apache-2.0)，其他组件保留各自许可；
适用范围见 [NOTICE](NOTICE)。

# 3\. 手势原语清单


## 3\.1 规则说明
- **支持型号**：与[手型配置](hand_gesture_primitives/hand_gesture_primitives/config)中各型号的 `primitives.supported` allowlist 一致。**L20 与 L25 为同协议两个版本**：配置 `hand_joint:=L20` / `hand_type:=l20` 时按 L25 档案运行，凡表中支持 L25 的原语均同时支持 L20。

|手型|支持原语数|
|---|---:|
|O6|32|
|O20|32|
|L25|38|
|L20|38|
|L6|28|
|O30|38|
|G20|37|

> 当前共注册 40 个原语。上表由 `hand_profile.supported_primitives()` 导出，
> 并由[原语清单检查](hand_gesture_primitives/test/test_readme_matches_profiles.py)持续校验，避免与代码脱节。

- **无接触手势原语**： 仅输入「话题字符串指令 \+ 当前关节状态」，无需视觉、TCP位姿。在关节反馈有效、当前控制权允许时执行，不经过抓取门控。

- **力控原语**：需要对应手型的有效电流/力矩/触觉反馈及控制权，接触与保护阈值以当前配置和节点参数为准。下方抓取/动态原语表按用途汇总，网页的“默认/感知”分类以运行时目录为准，并非表内每项都依赖力反馈。

- **视觉感知原语**：输入包含当前关节状态、TCP位姿、视觉目标几何及对应反馈/FK条件；需要门控的条目由同侧GraspGate校验后执行。网页会显示缺失或过期的输入条件，具体接口见感知指南。

## 3\.2 无接触手势原语

|序号|指令名称|功能说明|支持型号|输入信息|
|---|---|---|---|---|
|1|open|五指完全张开，复位基础姿态|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|2|init|手部复位至安全初始化中立位置|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|3|fist|五指完全握拳固定姿态|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|4|pinch|拇指\+食指标准精细对捏|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|5|point|食指单独伸直指向，其余手指弯曲收拢|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|6|ok\_sign|OK手势：拇指食指对接成环，剩余三指伸直|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|7|v\_sign|V字手势：食指、中指伸直，无名指小指弯曲|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|8|relax\_grip|缓慢卸力，放松当前握持姿态|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|9|release|完全释放抓取物体，五指舒展打开|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|10|named\_pose|回放示教保存的命名姿态，支持多关键帧中间态|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、姿态库|
|11|grasp\_ready|抓取前预成型，手指张开至待抓取就位姿态|L25, L20, O30|话题字符串指令、当前关节状态|

## 3\.3 抓取、动态与力控手势原语

|序号|指令名称|功能说明|支持型号|输入信息|
|---|---|---|---|---|
|1|thumb\_adduction\_grip|拇指侧向夹持，支持 prep预就位 / close 闭合两阶段执行|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|2|index\_middle\_adduction\_grip|食指\+中指侧向对夹夹持，夹取香烟、笔杆、细棒等细长物体|O20, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|3|ring|食中指环形包络，拇指与食指\+中指同时力控闭合自适应抓取|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|4|middle\_ring|中指环形包络，拇指与中指上下错开力控闭合自适应抓取|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|5|tripod|三指捏取，拇指与食指\+中指指尖对捏形成三角支撑，支持 prep / close / auto|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|6|tripod\_by\_force|三指捏取的力控版本，按电流/力矩/触觉反馈逐指停止闭合|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|7|index\_pinch|食指捏取，拇指与食指指尖精确对捏，拾取螺丝/针/薄片等小物体|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|8|middle\_pinch|中指捏取，拇指与中指指尖对捏，食指保持自由避让|O20, O6, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|9|parallel\_extension|拇指与四指平行伸展对捏，固定姿态夹持薄板、卡片等物体|O20, L25, L20, O30, G20|话题字符串指令、当前关节状态|
|10|large\_wrap|大尺寸物体全包络握持，支持 prep / close / auto 分阶段|O6, L6|话题字符串指令、当前关节状态|
|11|hammer\_grip|锤握，拇指压合于四指之上，适配锤柄等杆状工具|O6|话题字符串指令、当前关节状态|
|12|index\_flick|食指伸出圆周摆动拨动|L25, L20, O30, G20|话题字符串指令、当前关节状态|
|13|index\_middle\_flick|食指\+中指圆周摆动拨动|L25, L20, O30, G20|话题字符串指令、当前关节状态|
|14|twist2digit|拇指食指捏取后反复扭动|L25, L20, O30, G20|话题字符串指令、当前关节状态|
|15|twist3digit|拇指\+食指\+中指三指捏取后反复扭动|L25, L20, O30, G20|话题字符串指令、当前关节状态|
|16|twist5digit|五指捏取后反复扭动|L25, L20, O30, G20|话题字符串指令、当前关节状态|

## 3\.4 视觉力控有接触手势原语

|序号|指令名称|功能说明|支持型号|输入信息|
|---|---|---|---|---|
|1|index\_ring\_by\_vision|感知自适应精细环握，适配小型薄物体，auto映射precision模式|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|2|large\_wrap\_by\_vision|全包络强力抓取，适配大尺寸多面体，auto映射power模式|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|3|middle\_ring\_by\_vision|拇指\+中指单指对夹，高精度薄物体精细抓取|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|4|ring\_by\_vision|拇指\+食指\+中指环形包络，适配中等圆柱/扁平件|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|5|small\_warp\_by\_vision|拇指配合全部四指小型全包络，小物体高力量抓取|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|6|no\_index\_warp\_by\_vision|无食指参与包络，仅拇指\+中/无名/小指，保留食指自由作业|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|7|hook\_by\_vision|四指钩握、无拇指参与，适配把手、挂钩、环形物件|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|8|index\_pinch\_by\_vision|拇指食指超精细窄幅对捏，微小物体拾取|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|9|middle\_pinch\_by\_vision|拇指中指单独精细对捏，避开食指干涉|O20, O6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|10|tripod\_by\_vision|三脚架三指夹持（拇指\+食指\+中指），三点稳定支撑抓取|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|11|palmar\_by\_vision|掌面全包络大面积贴合，适配大尺寸平整物体|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|12|parallel\_extension\_by\_vision|手指平行伸展对夹，适配薄板、片状工件|O20, O6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
|13|disk\_by\_vision|圆盘环形环绕抓取，适配圆形盘状、环状物体|O20, O6, L6, L25, L20, O30, G20|话题字符串指令、当前关节状态、TCP位姿、3D bounding box|
