# 工作台接口与运行约定（0.2.2）

本仓库提供原语运行时、姿态库和独立网页工作台。设备通过已有 ROS 驱动接入，
Mock 模式在内存中模拟反馈，不创建 ROS 节点或硬件驱动。

- 下载、环境准备、单手与双手启动：[安装与启动指南](getting-started.md)。
- 示教、编辑、回放及姿态包操作：[网页操作指南](teach-panel-user-guide.md)。
- 视觉模型与感知接入：[感知配置](perception-setup.md)。

## 运行配置

- 默认示教页为 `/`，`/legacy` 是兼容入口，`/workbench` 提供扩展编辑器和按住推进。
- ROS 面板通过 `workbench_hands` 配置多手，默认侧由 `hand_side` 指定；URL 的
  `?side=left` 或 `?side=right` 可选择已配置的一侧。双手 Mock 默认右侧。
- `robot_description` 必须包含实际驱动使用的关节顺序和限位；双手配置需包含两侧关节。
  面板通过原语节点发送动作，与节点使用相同的手型、侧别及姿态目录。
- `observe_only:=true` 不发送动作、速度或停止命令，仍允许编辑保存姿态。
- 安装包自带 O6 右手 URDF 与 meshes。其他模型通过 `LINKERHAND_URDF_DIR` 提供，
  详见[模型说明](../data/urdf/README.md)。FK 依赖 Pinocchio；ROS 环境使用发行版提供的包，
  独立 Python 环境可安装 `fk` extra。

## 真机播放排查

内置原语与已保存姿态分别显示；内置原语不需要先导入出厂预设。出厂预设只是
可规划原语的终态快照，并不代表全部支持的原语。姿态目录按手型、左右侧分库，
新姿态目录为空时不会自动读取其他目录或另一只手的数据。

页面有新鲜关节反馈，只说明状态广播器正在工作，不能据此判定运动控制器已激活。
若轨迹进度前进但手不动，先核对页眉侧别，再只读检查对应控制器：

```bash
export HAND_CONTROLLER_MANAGER=/your_controller_manager
ros2 control list_controllers -c "$HAND_CONTROLLER_MANAGER"
```

如果对应的 forward controller 为 `inactive`，不要直接激活：执行器可能仍保留
先前的目标。先停止该侧原语，等待工作台状态确认 `active=null`、
`execution_kind=idle`、`command_pending=false`、`state_confirmed=true`，
再按设备启动流程激活控制器，由操作者重新明确点击播放。将示例中的
`/your_controller_manager` 替换为设备实际使用的 controller manager 名称。

## 数据与公共接口

- `HAND_GESTURE_POSE_DIR` 必须显式设置，且路径不得经过软链接；没有隐式 HOME 回退。
- 运行状态路径使用 `HAND_GESTURE_RUNTIME_STATUS_PATH`；默认写入
  `~/.local/state/hand_gesture_primitives/runtime-status.json`。
- `PoseDraft` 编辑位置为控制器顺序的弧度；落盘保留 `semantic_v3/rad/30`。
  既有 hardware、semantic_o20 与 legacy255 姿态仍可读取。
- 创建使用空 revision；更新/删除携带读取到的 revision，冲突拒绝覆盖。
  左右姿态分桶，菜单自定义 ID 不复用。
- 公共模块包括 `workbench_core`、`workbench_runner`、`workbench_client`、
  `workbench_edit` 和 `workbench_catalog`。宿主提供 JointAdapter、反馈及运行调度；
  runner 返回关节目标，设备写入仍由宿主唯一发送者负责。
- 原有 ROS 字符串命令和姿态 CLI 保留；示教面板为默认首页，`/legacy` 兼容入口
  与 bundle 导入导出继续可用。两种页面动作均经公共控制路径处理。


### 位置与量纲

| 接口 / 字段 | 位置表示 |
| --- | --- |
| `PoseDraft.positions` | 实际控制器关节顺序的弧度，长度为该手型有效自由度 |
| 新建姿态文件 | `space=semantic_v3`、`unit=rad`，固定 30 维 S30 语义布局 |
| `PrimitiveResult.space=semantic_rad` | S30 语义布局中的弧度，依赖真实 URDF 转换器 |
| `PrimitiveResult.space=semantic` | 无转换器时兼容的 S30 语义 0–255 表示 |
| `PrimitiveResult.space=hardware` | 产品关节顺序的 0–255 表示，用于兼容硬件边界 |

旧 hardware、semantic_o20 和 legacy255 数据在读取边界转换；不要将它们直接当作
控制器弧度目标。接口与 ROS 消息字段详见[接口文档](../hand_gesture_primitives/docs/interfaces.md)。

## 交互和控制语义

默认示教页与扩展工作台共用控制路径，但保存时机不同。默认页“保存”“＋ 添加”、
关键帧排序/重命名/删除立即写入姿态库；工作台编辑器的帧调整先留在草稿，点击
“保存原语”才落盘。“载入”或切帧不自动将整帧目标下发给手，但进入编辑可能停止
既有动作。逐关节拖动在示教或实时跟随启用时驱动所编辑关节；捕获、保存不执行整帧。

编辑暂停作用于同一个面板实例管理的所有手。首次进入编辑时，先停止所有侧的普通动作，
再允许编辑者跟随或试播；只读观察不发送这些命令。关闭工作台编辑器后需显式恢复控制；
默认页“退出示教”在没有其他编辑占用时恢复输入。恢复输入不自动续播动作。

有限轨迹支持按住推进、松开暂停、再次按住继续。到达终点后持续按住不循环；
重新执行由当前姿态起步。0.25–3.0 倍速仅改变后续运动时间，不改变硬件限矩，
不缩放接触保护时间，也不使进度跳变。速度由本面板实例配置的左右手共用，
仅对本次运行生效；选择侧别不会把倍率修改限定为单侧。

默认页“定点”进入所选段的 progress=0；该起点取建立这次段预览时的当前手姿，
不是保存帧的目标。全局条按各帧时长加权，跨段时建立新段预览；同段连续拖动不重新取起点。
工作台执行状态条只显示进度；编辑器的可拖动进度必须先建立本页试播。

编辑 token 的租约为 10 秒，单手同一时刻只有一个实时编辑者。
实时跟随 0.5 秒未续约即停止。停止序号使晚到的旧更新失效。
切帧、试播、离开页面和连接中断清除跟随；恢复后需要新的明确操作。

轨迹结束与反馈到位分别显示。连续 3 条不同的新鲜反馈确认到位；
等待超过 1 秒报告未到位并保持终点。进程内阈值为 0.03 rad；
ROS executor 使用 3.0 硬件单位并只比较可观测通道。
停止 ACK 表示执行器已停止处理该动作，不能作为电机物理停稳的证明。

## ROS 控制与进度协议

ROS 工作台协议使用 v2。`progress_controls: 1` 能力用于绑定执行身份的
进度控制：preview 分配 `control_id`，可选 `initial_progress` 在首个控制周期前
原子进入进度模式；后续 `kind=progress` 携带相同 `control_id` 和 `progress`。
缺少能力的旧节点会明确提示升级，不退回未关联执行的进度命令。
旧 Float32 `motion_progress` 话题继续供既有客户端使用。

同一姿态内容 revision、关键帧和编辑租约的连续 scrub 只改变进度，不重新取当前
手姿作为起点。换段或换版本时重新锚定一次。停止、失权和节点重启使旧请求失效，
迟到的确认不能恢复动作。窗口失焦、页面隐藏和指针捕获丢失都会释放按住推进；
恢复焦点后需要重新按下。

Python 原语结果提供可选 `PrimitiveResult.trajectory_finished`，缺省 `None` 保留
已有 done/engine 判定兼容。明确的终点标记用于统一普通与按住执行；名义时长只
显示进度，不能替代终点或新鲜反馈。持续力控不因静止或经过足够时间而自动结束。

运行状态中的 `completed_control_id` / `completed_kind` 标识本执行自然完成后的
终态，`done` 与 `settle_status` 说明反馈确认结果。旧页面仅根据匹配身份的普通
AUTO 完成处理返回示教，PROGRESS 和 held 不会自动退出。显式不返回的停止请求
以及当前关闭的返回示教偏好始终得到遵守。

内部控制世代同时覆盖旧字符串和 v2 控制入口，阻止停止后迟到的门控批准复活
动作；无效请求不撤销原有合法动作。O6 的 hand_info 与 motor_torque 使用独立
时间戳，优先新鲜 hand_info，再选择新鲜 topic，读取或切换来源不刷新样本时间。

## 开发检查

先按安装指南准备 Python 环境；运行 ROS 测试时还需编译消息包并加载对应 overlay。
从仓库根目录进入原语包运行已有测试：

```bash
cd hand_gesture_primitives
python -m pip install '.[teach,test]'
python -m pytest test
```

测试使用临时姿态目录与隔离的 ROS domain。缺少外部模型或 ROS 环境的测试会明确跳过。
浏览器用例位于 `hand_gesture_primitives/test/browser/`，针对独立 Mock 服务验证界面操作。
安装验证脚本检查消息导入、命令入口及双手 Mock；硬件驱动和感知部署需在目标设备上验证。

## 数据升级

升级前保留姿态目录和旧安装，在新环境安装并确认路径、手型与侧别。跨机或跨目录迁移
使用姿态包导入导出；版本切换不自动改写用户姿态文件。
