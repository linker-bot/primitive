# 手势原语使用与扩展（0.2.2）

规范名称来自 `PRIMITIVE_REGISTRY`，能力分类来自 `primitive_catalog.py`，各手型允许执行的
集合来自[手型配置](../hand_gesture_primitives/config)。完整手型支持矩阵见 [README](../../README.md)。
“注册”“当前手型支持”和“反馈等条件就绪”是三个不同判断；网页会说明不可用原因。

## 规范名称与分类

当前注册 40 项。下表中的 `safe` 是目录分类，不表示动作无运动风险或能绕过控制权检查。
`named_pose` 是模板，实际自定义手势以其保存名称出现在网页中。

| 目录分类 | 规范原语名 |
| --- | --- |
| `safe`（5） | `init`、`open`、`release`、`relax_grip`、`named_pose` |
| `gesture`（17） | `fist`、`pinch`、`point`、`ok_sign`、`v_sign`、`grasp_ready`、`hammer_grip`、`large_wrap`、`tripod`、`index_pinch`、`middle_pinch`、`parallel_extension`、`index_flick`、`index_middle_flick`、`twist2digit`、`twist3digit`、`twist5digit` |
| `force`（5） | `thumb_adduction_grip`、`index_middle_adduction_grip`、`tripod_by_force`、`ring`、`middle_ring` |
| `vision`（13） | `index_ring_by_vision`、`large_wrap_by_vision`、`middle_ring_by_vision`、`ring_by_vision`、`small_warp_by_vision`、`no_index_warp_by_vision`、`hook_by_vision`、`index_pinch_by_vision`、`middle_pinch_by_vision`、`tripod_by_vision`、`palmar_by_vision`、`parallel_extension_by_vision`、`disk_by_vision` |

网页的“默认原语”合并 safe/gesture；“感知原语”合并 force/vision。各手型的支持项不同，
例如 `hammer_grip` 仅 O6，`large_wrap` 为 O6/L6，L20 复用 L25 配置。
原始指令应使用完整注册名，例如 `disk_by_vision`；不要省略 `_by_vision` 猜测旧别名。

## 输入条件

- 基础与自定义手势不需要视觉模型；ROS 执行仍需有效关节反馈、控制权和匹配的驱动/URDF 配置。
- force 分类使用手型配置的触觉、电流或力矩反馈。不同手型的反馈源和阈值不同，
  应以配置与运行时提示为准，不能把无反馈时的动作当作已具备接触保护。
- vision 分类需要相应目标、TCP、FK 和反馈。11 个原语由 GraspGate 执行几何门控；
  `index_ring_by_vision`、`large_wrap_by_vision` 为门控直通例外。
- 软件目录中的硬件能力描述不等于传感器已经接通；完整接入见[感知指南](../../docs/perception-setup.md)。

## 常用执行方式

先按[安装指南](../../docs/getting-started.md)启动节点与驱动，然后每次选择一条指令执行：

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right open
ros2 run hand_gesture_primitives pub_gesture_cmd -- right fist
ros2 run hand_gesture_primitives pub_gesture_cmd -- right stop
```

`init` 是配置的初始化姿态，`open` 是张开目标，`release` 和 `relax_grip` 的动作由手型参数实现。
它们都会产生控制目标，不替代硬件急停。实际目标和时长以对应 YAML 配置为准。

### 分阶段与进度

以下六项支持命令级相位：`tripod`、`large_wrap`、`hammer_grip`、`tripod_by_force`、
`thumb_adduction_grip`、`index_middle_adduction_grip`。该集合由
`command_phased_primitive_names()` 定义；原语自身有多个阶段不等于接受 `prep/close` 参数。

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "tripod prep"
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "tripod close"
```

`tripod`、`tripod_by_force` 和 `large_wrap` 支持自动执行形式，实际阶段/停顿依手型实现。
只有 `tripod`、`large_wrap` 的显式 `prep/close` 支持命令级 progress：

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "tripod prep 0.3"
```

`auto/full` 不接受 progress；其他原语不能通过添加一个百分比获得进度控制。
工作台的“按住推进”是另一控制入口，只在运行时声明支持的有限轨迹上显示。

### 自定义手势

通过网页保存 `my_grip` 后，ROS 命令为：

```bash
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "pose my_grip"
ros2 run hand_gesture_primitives pub_gesture_cmd -- right "pose my_grip grasp 0.5"
```

`grasp` 是已经保存的关键帧名。`named_pose:<name>` 是工作台菜单/部分服务使用的标识，
不能替代上述 ROS 文本。关键帧时长、单段回放与进度含义见[网页手册](../../docs/teach-panel-user-guide.md)。

### 感知自动选择

`auto` 根据当前目标的 grasp_type 和手型配置中的 `grasp_type_map` 选择原语，
不是“运行全部原语”。无有效目标/映射时会拒绝。需要指定目标时可使用 `target <label>`
或 `target_id <id>`；话题与门控规则见[接口文档](interfaces.md)。

## 位置与量纲

有真实 URDF 转换器时，原语执行结果使用 `semantic_rad`（30 维 S30 语义槽、弧度）。
无转换器的兼容路径及旧 hardware/legacy255 数据仍存在，不能将它们与控制器弧度数组混用。
完整的数据边界见[工作台数据约定](../../docs/workbench.md#数据与公共接口)。

目标、接触阈值、阶段时长和关节耦合应通过对应手型配置核对。不同型号即使 DOF 相同，
硬件方向和主动关节顺序也不一定相同，不应复制其他手型的数值直接控制。

## 添加原语

1. 在包内 `primitives/` 的公共模块中实现具体的 `HandGesturePrimitive` 子类，提供唯一名称
   和 `compute()`，返回 `PrimitiveResult`。以现有实现为模板；以 `_` 开头的私有模块不参与发现。
2. 在 `primitive_catalog.py` 声明类别、所需能力、门控及相位策略；在目标手型配置的
   `primitives.supported` 中加入名称，必要时补充 `gestures` 参数。
3. 需要名义轨迹规划时，按 `planable_primitives.py` 的支持范围接入；视觉/力控闭环
   不应伪装成静态规划。修改注册名称时同步 README 支持表。
4. 在独立环境执行原有检查，从包目录运行：

```bash
python -m pytest test/test_primitive_discovery.py \
  test/test_capability_matrix_consistency.py test/test_readme_matches_profiles.py
```

原语源码热更新只在执行器空闲时应用；部署安装使用源码快照，修改工作区并不会自动更新
已经安装的副本。ROS 启动、目录配置与更新方式统一见安装指南。
