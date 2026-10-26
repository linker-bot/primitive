# 下载、环境配置、安装编译与启动（0.2.2）

本指南把独立Mock、ROS原语与可选感知环境分开说明。普通手势、自定义多阶段、
速度/进度和网页姿态编辑不需要下载AI权重。使用仓库感知示例时，再阅读
[感知环境与权重部署](perception-setup.md)。网页操作见[用户手册](teach-panel-user-guide.md)。

## 1. 下载源码与选择版本

在希望存放源码的目录运行；公开仓库可直接通过 HTTPS 读取：

```bash
git clone https://github.com/linker-bot/primitive.git
cd primitive
export GESTURE_SOURCE="$(pwd)"
git rev-parse HEAD
```

`main` 是公开维护主线。请保存最后一条命令输出的完整提交 SHA，作为此次安装的
源码版本；审阅尚未合并的 PR 时，先按该 PR 的说明检出其分支，再记录 SHA。
需要复现时，在干净的新克隆中执行 `git checkout --detach PUBLIC_COMMIT_SHA`，
其中 `PUBLIC_COMMIT_SHA` 替换为已记录的公开提交 SHA。

`0.2.2` 是包版本号；Git 标签以仓库实际提供的版本为准。文档和源码应从同一个
公开提交读取。

完整 Git 源码包含原语、消息、感知示例、相机源码及已跟踪的第三方代码；模型权重
另行获取。Python wheel 和 sdist 只交付原语包及其资源，不能替代完整的感知和相机
工程。本文的源码安装流程使用上述完整克隆。

| 目录 | 用途 |
| --- | --- |
| `hand_gesture_primitives/` | ROS/Python原语包、网页、launch与安装验证脚本 |
| `messages/` | 原语、感知和触觉相关消息包 |
| `data/urdf/` | 仓库自带O6右手URDF及mesh |
| `perception/` | 可选ROS感知包及环境/配置 |
| `third_party/robot_perception/` | 感知模型适配与下载脚本；额外源码按感知指南获取 |
| `sensors/` | 相机ROS工程，按实际设备选用 |

若你自己的工作区使用了稀疏检出，目录可能没有展开；以`git ls-tree HEAD`核对。
普通新克隆不需要为本指南开启稀疏检出。

## 2. 环境选择

| 用途 | 使用的解释器/环境 | 是否需要ROS/设备 |
| --- | --- | --- |
| 独立Mock、纯Python/Web功能 | Python3.12普通venv | 不需要 |
| 原语ROS节点与真机面板 | Ubuntu24.04、ROS2 Jazzy、系统Python3.12创建的专用venv | ROS必需；动作需要外部驱动 |
| 仓库检测/分割/VLM/Pixal3D | 按对应环境文件创建的独立Conda/CUDA环境 | 见感知指南 |

ROS2 Jazzy本身的安装以[ROS官方Ubuntu安装文档](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)为准。
本指南假定ROS已安装到`/opt/ros/jazzy`。ROS网页需要Pydantic2；Ubuntu24.04的apt
Pydantic1不能替代它。专用安装脚本保留apt NumPy/SciPy与ROS扩展，只在venv提供
Web/构建依赖。感知CUDA依赖应留在其独立环境。

## 3. 独立Mock安装与启动

Ubuntu首次使用venv时准备工具；已准备好的机器可跳过apt步骤：

```bash
sudo apt update
sudo apt install python3-venv python3-pip
/usr/bin/python3 --version
```

以下从仓库根目录运行。使用新的venv路径，避免覆盖现有环境：

```bash
cd "$GESTURE_SOURCE"
/usr/bin/python3 -m venv "$HOME/hand-gesture-mock-0.2.2"
source "$HOME/hand-gesture-mock-0.2.2/bin/activate"
python -m pip install './hand_gesture_primitives[teach]'
python -c 'from importlib.metadata import version; print(version("hand_gesture_primitives"))'
```

应输出`0.2.2`。`teach`提供网页依赖；`test`只在开发回归时需要。
普通venv内也可以安装发布wheel并启用对应extra，例如：

```bash
python -m pip install '/path/to/hand_gesture_primitives-0.2.2-py3-none-any.whl[teach]'
```

上面的`/path/to/`须替换为实际文件位置。离线安装还需准备所有依赖wheel，
单独一个原语wheel不等于完整Python环境。

启动双手模拟网页：

```bash
export HAND_GESTURE_POSE_DIR="$HOME/hand-gesture-data/mock-poses"
export HAND_GESTURE_RUNTIME_STATUS_PATH="$HOME/hand-gesture-data/mock-runtime.json"
teach_panel --mock --pose-dir "$HAND_GESTURE_POSE_DIR" \
  --hand left:o6 --hand right:o6 --speed 1.0 --control-rate 30 \
  --host 127.0.0.1 --port 18002
```

确认终端出现本地HTTP地址，浏览器打开`http://127.0.0.1:18002/`。
`/api/workbench/state`的`backend`应为`mock`。该模式只在内存模拟反馈，不创建ROS节点
或硬件驱动；姿态文件仍会持久化，因此示例使用独立的Mock目录。

- 只配置一只手：只保留一个`--hand right:o6`。
- 更换手型：修改`--hand`中的型号，重复的侧别会被拒绝。
- 只读模拟：追加`--observe-only`；可编辑保存，动作入口禁用。
- 在此终端按`Ctrl+C`关闭服务；下次使用同一姿态目录可继续读取已保存数据。

## 4. 远程浏览器访问

网页默认监听远端机器的`127.0.0.1`。使用远程桌面时，直接在远端浏览器打开即可。
只通过SSH连接时，在自己的电脑另开终端，替换`USER@HOST`后运行：

```bash
ssh -N -L 18002:127.0.0.1:18002 USER@HOST
```

随后在自己电脑的浏览器访问`http://127.0.0.1:18002/`。保持SSH转发进程运行。
真实面板示例使用8002时，将转发的两处端口都换成8002。

## 5. ROS环境准备与编译安装

在新的终端准备系统依赖，避免将感知或策略Conda环境作为此处的Python解释器：

```bash
source /opt/ros/jazzy/setup.bash
sudo apt install python3-venv python3-numpy python3-scipy python3-yaml \
  python3-filelock python3-colcon-common-extensions python3-rosdep
```

首次使用rosdep的机器需先执行`sudo rosdep init`；已初始化的机器不要重复初始化。
从仓库根目录解析包依赖：

```bash
export GESTURE_SOURCE="$HOME/primitive"  # 改为实际源码目录
cd "$GESTURE_SOURCE"
rosdep update
rosdep install --from-paths hand_gesture_primitives messages --ignore-src \
  --skip-keys python3-pydantic -y
```

Pydantic由下述venv提供。FK所需Pinocchio应使用与ROS发行版一致的apt包；本机若需
补装可用`sudo apt install ros-jazzy-pinocchio`，不在该ROS环境中另装pip的`pin`。

执行独立安装：

```bash
export HAND_GESTURE_INSTALL="$HOME/hand-gesture-ros-0.2.2"
/usr/bin/python3 hand_gesture_primitives/scripts/validate_ros_install.py \
  --output "$HAND_GESTURE_INSTALL"
```

`HAND_GESTURE_INSTALL`必须是**源码目录外尚不存在的新目录**。已有同名环境时，
选择另一个目录，例如`hand-gesture-ros-0.2.2-build2`。脚本会完成：

1. 创建系统Python3.12的`--system-site-packages` venv并安装Pydantic2/Web依赖。
2. 复制原语、消息和已有URDF资源到独立source目录。
3. 用该venv解释器调用colcon，编译`hand_force_msg`、`robot_perception_msgs`、
   `hand_gesture_primitives_msgs`、`hand_gesture_primitives`四个包。
4. 检查实际解释器、消息导入、apt数值库路径，并启动双手Mock检查网页入口。
5. 生成`acceptance.json`以及`pip.log`、`colcon.log`、`mock.log`等日志。

该流程不构建机械手驱动、相机或感知包。拥有Web/构建依赖wheelhouse时可追加
`--wheelhouse /path/to/wheelhouse`使用离线依赖；ROS和apt依赖仍须预先准备。

每个使用原语的ROS终端都需加载环境和同一个真实姿态目录：

```bash
source /opt/ros/jazzy/setup.bash
export HAND_GESTURE_INSTALL="$HOME/hand-gesture-ros-0.2.2"
source "$HAND_GESTURE_INSTALL/venv/bin/activate"
source "$HAND_GESTURE_INSTALL/workspace/install/setup.bash"
export PYTHONNOUSERSITE=1
export HAND_GESTURE_POSE_DIR="$HOME/hand-gesture-data/poses"
```

如果你选择了其他安装目录，同步修改这里的变量。多个ROS overlay共存时，让此处的
原语安装overlay最后生效。快速确认实际加载位置：

```bash
ros2 pkg prefix hand_gesture_primitives
python -c 'import hand_gesture_primitives,pydantic; print(hand_gesture_primitives.__file__); print(pydantic.__version__)'
```

包路径应在所选安装目录，Pydantic应为2.x。独立安装是源码快照：以后修改Git工作区
不会自动更新安装产物；更新运行代码时需要新的安装目录。

## 6. 真实ROS接入前提

本仓库通过已有ROS驱动控制手，不包含通用CAN/SDK驱动的一键安装。
驱动按设备厂商提供的部署说明安装。普通动作至少确认：

- 驱动提供新鲜关节反馈，默认`/joint_states`，关节名称、顺序与手型匹配。
- 控制后端与目标话题一致。下例使用`ros2_control`的弧度目标；已有`legacy_cb`
  接入方式需显式配置 `cmd_backend:=legacy_cb ros2_control_command_units:=hardware`；
  其话题和关节顺序按实际驱动设置。
- 所有相关终端使用与驱动相同的ROS domain与发现范围。
- `robot_description`提供**实际控制器使用的关节顺序和限位URDF XML**。
- 需要FK时另提供真实几何模型。控制器限位URDF不能自动代替完整FK模型，
  包内O6右模型也不能拿来当左手模型。

**启动`gesture_node`会选择`init`，因此可能产生运动。**
对于ros2_control部署，按驱动启动流程先停用动作控制器、保留状态广播；启动原语和
面板后发送stop并确认idle，再激活所选手的动作控制器，由操作者明确执行动作。
驱动/控制器名称因部署而异，状态检查方式见[工作台播放排查](workbench.md)。

以下示例要求先完成上述驱动和控制器配置。同一侧只运行一个原语节点；已有节点时
只连接面板。

## 7. 单手与双手使用示例

### 单右手：原语节点与面板分别启动

先在各终端完成上一节的ROS环境加载。将`/path/to/`改为真实文件路径：

```bash
export HAND_CONTROLLER_URDF=/path/to/right_o6_controller.urdf
test -f "$HAND_CONTROLLER_URDF"
ros2 launch hand_gesture_primitives gesture_node.launch.py \
  hand_type:=o6 hand_side:=right cmd_backend:=ros2_control \
  ros2_control_command_units:=radians control_rate:=30.0 primitive_speed:=0.25 \
  launch_gate:=false pose_dir:="$HAND_GESTURE_POSE_DIR" \
  robot_description:="$(cat "$HAND_CONTROLLER_URDF")"
```

在另一个已加载相同环境的终端，使用同一URDF文件开启面板：

```bash
export HAND_CONTROLLER_URDF=/path/to/right_o6_controller.urdf
ros2 run hand_gesture_primitives teach_panel --ros-args \
  -p hand_type:=o6 -p hand_side:=right -p cmd_backend:=ros2_control \
  -p ros2_control_command_units:=radians -p primitive_speed:=0.25 -p control_rate:=30.0 \
  -p "robot_description:=$(cat "$HAND_CONTROLLER_URDF")" \
  -p http_host:=127.0.0.1 -p http_port:=8002
```

也可用`teach_panel.launch.py`，但要明确`launch_gesture_node`和`launch_pose_server`；
已有原语节点时保持`launch_gesture_node:=false`。本指南的网页直接使用共享姿态库，
不另行启动`pose_library_server`；组合launch可显式设置`launch_pose_server:=false`。
已有其他客户端需要独立姿态服务时，按其部署配置保留该服务。

### 双手：一个面板管理两个原语节点

两侧型号相同、话题采用默认命名时，可使用以下启动文件。合并后的URDF必须包含
**左右两侧**真实控制器关节与限位，公共link去重；不能只传右手URDF给双手：

```bash
export HAND_CONTROLLER_URDF=/path/to/dual_o6_controller.urdf
ros2 launch hand_gesture_primitives bimanual_gesture.launch.py \
  hand_type:=o6 cmd_backend:=ros2_control ros2_control_command_units:=radians \
  control_rate:=30.0 primitive_speed:=0.25 launch_gate:=false \
  pose_dir:="$HAND_GESTURE_POSE_DIR" \
  robot_description:="$(cat "$HAND_CONTROLLER_URDF")"
```

另一个终端加载相同环境、设置相同URDF路径后：

```bash
export HAND_CONTROLLER_URDF=/path/to/dual_o6_controller.urdf
ros2 run hand_gesture_primitives teach_panel --ros-args \
  -p hand_type:=o6 -p hand_side:=right -p "workbench_hands:=['left:o6','right:o6']" \
  -p cmd_backend:=ros2_control -p ros2_control_command_units:=radians \
  -p primitive_speed:=0.25 -p control_rate:=30.0 \
  -p "robot_description:=$(cat "$HAND_CONTROLLER_URDF")" \
  -p http_host:=127.0.0.1 -p http_port:=8002
```

浏览器打开`http://127.0.0.1:8002/`，默认右手；`?side=left`可显式选左手。
两侧型号不同或自定义话题时，分别启动各侧`gesture_node.launch.py`，并按实际手型
配置面板的`workbench_hands`。同一面板的速度与编辑暂停作用范围见网页手册。

### 只读连接与命令发送

只观察现有节点时，在上述**面板**命令中追加`-p observe_only:=true`。
它约束面板发送动作的行为，不会停止外部执行者，也不会改变另行启动的原语节点行为。

驱动和原语已按前述流程就绪后，可在加载ROS环境的命令终端发送以下独立示例。
每次选择一条执行，等待完成或明确停止后再发送下一条：

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right stop
ros2 run hand_gesture_primitives pub_gesture_cmd -- right open
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "pose my_grip"
```

`my_grip`必须已保存在当前O6右手库中。左手替换侧别为`left`。
命令是执行入口；`open`和命名姿态会运动。启用GraspGate仍使用公开指令入口，
感知条件与模型启动见[感知指南](perception-setup.md)。

## 8. 数据、升级与常见问题

- 姿态目录必须显式指定，且路径不得经过软链接；旧HOME目录不会自动迁移。
  多进程应使用同一真实姿态目录，Mock测试使用另一目录。
- 库文件按手型与侧别保存，例如`o6_left.json`、`o6_right.json`。跨机迁移使用
  页面姿态包；它不携带驱动配置、模型权重或当前运行速度倍率。
- `Pydantic >=2 required`：检查是否激活所选ROS venv、overlay顺序和实际Python路径。
- `robot_description`/关节限位错误：检查实际控制器URDF、型号、侧别；不要套用另一只手的模型。
- 页面有进度但手不动：检查当前侧、对应动作控制器是否active；新鲜反馈不等于动作控制器已激活。
- 感知原语不可用：检查目录显示的具体原因，以及视觉/TCP/FK/力反馈/gate；安装权重本身不等于感知数据已接通。
- 端口被占用：选择另一HTTP端口，并同步调整浏览器地址与SSH转发。
- 更新已发布版本：保留原数据和旧安装，在新目录安装；核对版本、路径和Mock后再按设备部署流程切换。

当前版本的功能和限制见[发布说明](releases/0.2.2.md)。
