# 手势原语库

本文档介绍 **24 个**预定义手势原语的分类、行为、参数与使用方式。注册表见 `primitives/__init__.py` 中 `PRIMITIVE_REGISTRY`。

---

## 原语总览

| # | 指令名 | 类别 | 参与手指 | 感知输入 | 力控 | 分阶段 |
|---|--------|------|---------|---------|------|--------|
| 1 | `init` | 安全 | 全部 | — | — | — |
| 2 | `open` | 安全 | 全部 | — | — | — |
| 3 | `release` | 安全 | 全部 | — | — | — |
| 4 | `relax_grip` | 安全 | 全部 | — | — | — |
| 5 | `fist` | 固定 | 全部 | — | — | — |
| 6 | `pinch` | 固定 | 拇+食 | — | — | — |
| 7 | `point` | 固定 | 全部 | — | — | — |
| 8 | `ok_sign` | 固定 | 全部 | — | — | — |
| 9 | `v_sign` | 固定 | 全部 | — | — | — |
| 10 | `index_ring_by_vision` | 自适应 | 拇+食 | object_size | 触觉 | — |
| 11 | `large_wrap_by_vision` | 自适应 | 全部 | object_size | 触觉 | — |
| 12 | `thumb_adduction_grip` | 侧向夹持 | 拇+(食中无小) | object_size | 触觉 | prep/close |
| 13 | `index_middle_adduction_grip` | 侧向夹持 | 拇+食+中 | object_size | 触觉 | prep/close |
| 14 | `middle_ring_by_vision` | 力控包络 | 拇+中 | object_size | 电流 | — |
| 15 | `ring_by_vision` | 力控包络 | 拇+食+中 | object_size | 电流 | — |
| 16 | `small_warp_by_vision` | 力控包络 | 拇+食+中+无+小 | object_size | 电流 | — |
| 17 | `no_index_warp_by_vision` | 力控包络 | 拇+中+无+小 | object_size | 电流 | — |
| 18 | `hook_by_vision` | 力控包络 | 食+中+无+小 | object_size | 电流 | — |
| 19 | `tripod_by_vision` | 力控包络 | 拇+食+中 | object_size | 电流 | — |
| 20 | `palmar_by_vision` | 力控包络 | 全部（掌心支撑） | object_size | 电流 | — |
| 21 | `index_pinch_by_vision` | 定位捏取 | 拇+食 | tcp_pose / object_pose | — | — |
| 22 | `middle_pinch_by_vision` | 定位捏取 | 拇+中（食避让） | tcp_pose / object_pose | — | — |
| 23 | `parallel_extension_by_vision` | 定位包络 | 拇 vs 食+中+无+小 | tcp_pose / object_pose | — | — |
| 24 | `disk_by_vision` | 定位包络 | 全部 C 形 | tcp_pose / object_pose | — | 内部 P1–P3 |

> **GraspGate 门控（11 个）**：`ring_by_vision` `small_warp_by_vision` `no_index_warp_by_vision` `middle_ring_by_vision` `hook_by_vision` `tripod_by_vision` `palmar_by_vision` `parallel_extension_by_vision` `index_pinch_by_vision` `middle_pinch_by_vision` `disk_by_vision` — 须经 H/XY/Fwd 三判定后才转发至 Executor。  
> **Executor 可达性**：除 4 个安全原语外，所有原语激活时检查 TCP–物体距离（默认 15 cm）；定位类原语在运行中还会根据 `object_pose` 做掌心前方 2–15 cm 保持判定。

---

## 类别说明

### 安全类（4 个）

不受可达性检查限制，任何时刻均可调用。用于复位、释放、紧急保护。

| 指令 | 行为 | 过渡时间 |
|------|------|---------|
| `init` | 半张开自然姿态（所有原语 fallback 目标） | 0.8s |
| `open` | 五指完全伸直 | 0.6s |
| `release` | 从当前位置相对张开 | 0.6s |
| `relax_grip` | 先放松再回中（重置抓取状态） | 0.6s |

### 固定类（5 个）

预设目标角度 + 平滑插值（lerp），不依赖感知输入，不经 GraspGate。

| 指令 | 手形描述 | 运动特征 |
|------|---------|---------|
| `fist` | 五指完全握紧 | 4 阶段避免拇指碰撞：四指闭 → 拇指侧摆 → 旋转 → 下压 |
| `pinch` | 拇食对捏，其余张开 | 单阶段 lerp |
| `point` | 食指伸出，其余握拢 | 单阶段 lerp |
| `ok_sign` | 拇食成圆，其余伸直 | 单阶段 lerp |
| `v_sign` | 食中伸开，其余握拢 | 单阶段 lerp |

### 感知自适应类（2 个）

根据 `PrimitiveContext.object_size` 实时调整闭合量，支持触觉或**关节电流**闭环（无触觉传感器时自动 fallback）。

| 指令 | 手形 | 自适应逻辑 | 闭环策略 |
|------|------|-----------|---------|
| `index_ring_by_vision` | 拇+食环形包络 | 物体截面直径 → 调拇/食弯曲量 | progressive + 触觉/电流冻结 |
| `large_wrap_by_vision` | 五指包络 | 物体最大截面 → 调五指弯曲 | progressive + 触觉/电流冻结 |

**三阶段执行流程：**

```
P1 (0.4s): 非参与手指先张开
P2 (0.5s): 参与手指闭合到视觉目标
P3 (持续): 无接触 → 渐进闭合 (25–30/s)；接触后冻结（触觉或电流增量，阈值见 `contact_thresholds.yaml`）
```

**电流/触觉阈值**：可在 `config/contact_thresholds.yaml` 或 `gesture_node` ROS 参数中调节（如 `contact_current_delta`、`hold_safe_current`）。无触觉时默认以相对基线 **270 mA** 增量判定接触。

**尺寸自适应**：物体尺寸变化超过 2mm 时触发目标更新，经 250ms blend 平滑过渡。

### 侧向夹持类（2 个）

拇指或食中指侧面并拢夹持，**支持 prep/close 两阶段控制**（`PHASED_GRASP_PRIMITIVES`）。

| 指令 | 夹持方式 | 典型场景 |
|------|---------|---------|
| `thumb_adduction_grip` | 拇指侧面下压贴紧食指侧面 | 卡片、钥匙、薄片 |
| `index_middle_adduction_grip` | 食指与中指侧面并拢 | 笔杆、香烟、细棒 |

**分阶段命令格式：**

```bash
# 完整执行（默认 full）
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'thumb_adduction_grip'"

# 分步执行
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'thumb_adduction_grip prep'"
# ... 机械臂就位 ...
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'thumb_adduction_grip close'"
```

**`thumb_adduction_grip` 执行流程（full 模式）：**

```
P1 (0.35s): 四指弯曲收拢形成承接面
P2 (0.40s): 拇指侧摆外展 (abd)
P3 (0.35s): 拇指旋转就位 (rot)
P4 (0.65s): 拇指根部+指尖下压闭合
P5 (持续):  渐进夹紧 → 触觉冻结
```

`prep` 在 P3 结束后暂停（`grasp_state = "ready"`），等待 `close` 命令继续 P4+。

### 力控包络类（8 个）

基于 `object_size`（及可选 mesh 预选型）预成型，逐指电流检测接触后停止。**经 GraspGate 门控后执行**。

**Mesh 目录（可选，用于 thumb_rot 预选型）：**

```
data/mesh_model/{label}/model.obj
```

- 默认根目录：仓库内 `data/mesh_model/`
- 覆盖路径：`export MESH_MODEL_DIR=/path/to/meshes`
- `{label}` 须与感知 `bboxes_3d.label` 一致（或通过 launch 参数 `target_object_label` 指定；空=自动选最高 score）
- 无 mesh 时回退到 `object_size` 估算宽度

| 指令 | 对抗手指 | 适用场景 |
|------|---------|---------|
| `middle_ring_by_vision` | 拇 vs 中 | 中等物体对捏 |
| `ring_by_vision` | 拇 vs 食+中 | 柱形/瓶状物体 |
| `small_warp_by_vision` | 拇 vs 食+中+无+小 | 小型多面体全手抓取 |
| `no_index_warp_by_vision` | 拇 vs 中+无+小 | 食指需保持伸出的场景 |
| `hook_by_vision` | 无拇指，食+中+无+小 | 把手/挂钩/环形物体 |
| `tripod_by_vision` | 拇 vs 食+中 | 三指三角支撑（笔、螺丝刀柄） |
| `palmar_by_vision` | 掌心 vs 食+中+无+小 | 强力掌心包络、球体/多面体 |

**统一两阶段算法（middle_ring_by_vision / ring_by_vision / small_warp_by_vision / no_index_warp_by_vision / hook_by_vision / tripod_by_vision / palmar_by_vision）：**

```
Phase 1 — 预成型 (0.5s):
  根据 object_size（或 mesh 宽度）选择 thumb_rot 等预备角
  快速 lerp 到预备姿态（手指半张开）

Phase 2 — 力控闭合 (持续):
  各手指独立以固定速率闭合
  逐指监测电流增量 > `contact_current_delta`（或窄物体原语 `contact_current_delta_narrow`）→ 该指停止
  全部参与指停止 → grasp_state = "contact"
  持握安全电流上限: `hold_safe_current`（默认 800 mA）
```

运行中需 `tcp_pose`；有 `object_pose` 时物体须在掌心前方 2–15 cm，否则 `_hold()` 保持。

### 定位捏取 / 定位包络类（4 个）

固定目标姿态 + lerp，**不经 object_size 自适应、不用电流力控**，但须经 **GraspGate** 并在运行中校验 TCP/物体相对位姿。

| 指令 | 手形 | 典型场景 | 过渡 |
|------|------|---------|------|
| `index_pinch_by_vision` | 拇+食指尖对捏，其余伸直 | 螺丝、针、薄片 | 0.5s lerp |
| `middle_pinch_by_vision` | 拇+中对捏，食指外展避让 | 需保持食指自由的捏取 | 0.5s lerp |
| `parallel_extension_by_vision` | 五指伸直，拇 vs 四指平行夹持 | 书本、卡片、平板 | 0.6s lerp |
| `disk_by_vision` | 四指 C 弧 + 拇指侧摆合拢 | 盘、碟、扁圆物体 | 内部三阶段 1.4s |

**`disk_by_vision` 内部三阶段（非 prep/close 命令）：**

```
P1 (0.4s): 拇指 abd 侧摆至中指对齐位
P2 (0.5s): 四指弯成 C 弧，拇指 flex 关节保持
P3 (0.5s): 拇指 base/rot/tip 合拢握住
```

**运行约束（与力控包络类相同）：**

- 无 `tcp_pose` → 保持当前姿态
- 有 `object_pose` → 掌心前方 2–15 cm，否则 hold
- 无 `object_pose` 时跳过前方距离检查（GraspGate 已做 H/XY/Fwd）

---

## auto 命令映射

发送 `auto` 时，根据感知输出的 `grasp_type` 自动选择原语（`grasp_mapping.py`）：

```
感知 grasp_type    →    原语
─────────────────────────────
precision          →    index_ring_by_vision
lateral            →    index_middle_adduction_grip
power              →    large_wrap_by_vision
```

无 `grasp_type` 或未知类型时不执行任何动作。新增定位/力控原语暂未纳入 auto 映射。

---

## 原语生命周期

```
cmd "pinch" 到达
       │
       ▼
┌─ set_primitive() ─────────────────────────────┐
│  1. 可达性检查 (非安全原语)                      │
│  2. 旧原语 on_exit()                           │
│  3. 新原语 on_enter(current_angles)            │
│  4. 记录 start_time                           │
└───────────────────────────────────────────────┘
       │
       ▼ (每 100ms)
┌─ _tick() ─────────────────────────────────────┐
│  ctx = PrimitiveContext(tcp, bbox, current...) │
│  result = primitive.compute(angles, elapsed, ctx)│
│  if result.feasible:                           │
│      target = result.target_angles             │
│  else:                                         │
│      target = last_target  (hold)              │
│  publish(target)                               │
└───────────────────────────────────────────────┘
       │
       ▼ (新命令到达 或 primitive.done)
  on_exit() → 切换到下一个原语
```

---

## grasp_state 状态机

力控和自适应原语通过 `grasp_state` 属性暴露内部进度：

| 状态 | 含义 | 后续行为 |
|------|------|---------|
| `approaching` | 正在向目标位置运动 | 继续 lerp |
| `ready` | prep 阶段完成，等待 close | 保持当前姿态 |
| `progressive` | 已到视觉目标，渐进闭合中 | 缓慢收紧 |
| `contact` | 检测到接触，冻结 | 保持直到新指令 |
| `closing` | 力控闭合中 | 监测电流 |

Executor 在状态变化时打印日志，方便调试。

---

## 添加新原语

1. 创建 `primitives/my_gesture.py`：

```python
from ..primitive_base import HandGesturePrimitive, PrimitiveContext, PrimitiveResult, lerp_angles

class MyGesture(HandGesturePrimitive):
    @property
    def name(self) -> str:
        return "my_gesture"

    def on_enter(self, current_angles):
        super().on_enter(current_angles)
        # 初始化内部状态

    def compute(self, current_angles, elapsed, ctx) -> PrimitiveResult:
        # 计算目标角度
        target = [...]  # 20-DOF (O20) 或 25-DOF (L25)
        t = min(1.0, elapsed / 0.5)
        angles = lerp_angles(self._start_angles, target, t)
        return self._move(angles)
```

2. 注册到 `primitives/__init__.py`：

```python
from .my_gesture import MyGesture
PRIMITIVE_REGISTRY["my_gesture"] = MyGesture
```

3. 若需 GraspGate，加入 `grasp_gate.py` 中 `GATED_PRIMITIVES`。

4. 测试：

```bash
ros2 topic pub --once /hand_gesture_cmd std_msgs/String "data: 'my_gesture'"
```

---

## 关节布局参考（O20 20-DOF）

| 索引 | 名称 | 含义 | 值域说明 |
|------|------|------|---------|
| 0 | thumb_base | 拇指根部弯曲 | 0=伸直, 255=弯曲120° |
| 1 | index_base | 食指根部弯曲 | 0=伸直, 255=弯曲180° |
| 2 | middle_base | 中指根部弯曲 | 0=伸直, 255=弯曲180° |
| 3 | ring_base | 无名指根部弯曲 | 0=伸直, 255=弯曲180° |
| 4 | pinky_base | 小指根部弯曲 | 0=伸直, 255=弯曲180° |
| 5 | thumb_abd | 拇指侧摆 | 0=0°, 255=180° |
| 6 | index_abd | 食指侧摆 | 128=中立, 0/-30°, 255/+30° |
| 7 | middle_abd | 中指侧摆 | 128=中立 |
| 8 | ring_abd | 无名指侧摆 | 128=中立 |
| 9 | pinky_abd | 小指侧摆 | 128=中立 |
| 10 | thumb_rot | 拇指旋转 | 0=0°, 255=130° |
| 11–14 | reserved | 预留位 | 恒为 0 |
| 15 | thumb_tip | 拇指指尖弯曲 | 0=伸直, 255=弯曲150° |
| 16 | index_tip | 食指指尖弯曲 | 0=伸直, 255=弯曲180° |
| 17 | middle_tip | 中指指尖弯曲 | 0=伸直, 255=弯曲180° |
| 18 | ring_tip | 无名指指尖弯曲 | 0=伸直, 255=弯曲180° |
| 19 | pinky_tip | 小指指尖弯曲 | 0=伸直, 255=弯曲180° |

> L25 (25-DOF) 新增 thumb_root2[15]、middle_root2[17]、thumb_tip→[20]、各指 tip→[21–24]，且角度方向反转（0=弯曲, 255=伸直）。详见 `primitive_base.py` 中 `HAND_CONFIGS["l25"]`。
