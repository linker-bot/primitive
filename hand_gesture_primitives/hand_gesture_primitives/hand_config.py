"""手部型号配置加载 — 根据 hand_joint 参数加载 YAML，提供语义↔硬件映射。

路径：
- ``from_semantic`` / ``to_semantic``：semantic_30 通道名（**P6 主路径**；
  YAML 未声明时由 ``from_o20`` 自动合成）
- ``from_o20`` / ``to_hardware`` / ``from_hardware``：O20 20 维（兼容别名，供旧工具链）
"""

import copy
import os
from functools import lru_cache
from typing import Dict, List, Optional, Set

import yaml

from .semantic_30 import (
    SEMANTIC_30_IDX,
    SEMANTIC_30_NAMES,
    SEMANTIC_DIM,
    o20_joint_index_to_s30,
    project_o20_to_s30,
)

# 默认配置目录 (source tree, 开发时使用)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")

# 用户对外型号 → 实际 config 基名（同协议不同版本）。
# L20 / L25 为同一硬件协议的两个版本；控制与原语统一按 l25 档案运行。
# G20 与 L20 外观结构一致，但 SDK 为独立 16 维协议，使用独立 g20.yaml（不别名到 l25）。
HAND_TYPE_ALIASES = {
    "l20": "l25",
}

# 旧 capabilities 功能 token → 默认启用的 semantic_30 通道
_CAPABILITY_CHANNEL_EXPAND: Dict[str, List[str]] = {
    "thumb_abd": ["thumb_mcp_abd"],
    "finger_abd": [
        "index_mcp_abd",
        "middle_mcp_abd",
        "ring_mcp_abd",
        "pinky_mcp_abd",
    ],
    "thumb_rot": ["thumb_rot"],
    "thumb_tip": ["thumb_dip"],
    "finger_tips": ["index_dip", "middle_dip", "ring_dip", "pinky_dip"],
    "finger_root2": ["index_pip", "middle_pip", "ring_pip", "pinky_pip"],
}

# 有映射时始终启用的屈伸/基础通道（不依赖旧 token）
_BASE_FLEX_CHANNELS: List[str] = [
    "thumb_mcp_flex",
    "index_mcp_flex",
    "middle_mcp_flex",
    "ring_mcp_flex",
    "pinky_mcp_flex",
]


def resolve_hand_config_key(hand_joint: str) -> str:
    """将 hand_type / hand_joint 归一为 config 文件基名（小写）。"""
    key = hand_joint.strip().lower()
    return HAND_TYPE_ALIASES.get(key, key)


def _find_config_path(hand_joint: str) -> str:
    """查找配置文件路径: 先 ament share, 再 source tree。"""
    filename = f"{resolve_hand_config_key(hand_joint)}.yaml"
    # 尝试 ament 安装路径
    try:
        from ament_index_python.packages import get_package_share_directory

        share = get_package_share_directory("hand_gesture_primitives")
        path = os.path.join(share, "config", filename)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    # Fallback: source tree
    path = os.path.join(_CONFIG_DIR, filename)
    if os.path.isfile(path):
        return path
    raise FileNotFoundError(f"找不到手部配置: {filename} (搜索: ament share + {_CONFIG_DIR})")


@lru_cache(maxsize=None)
def _load_config_data(path: str) -> dict:
    """Parse an installed hand profile once for the lifetime of the process."""

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"手部配置必须是 YAML 对象: {path}")
    return data


def _expand_channels_from_capabilities(caps: dict) -> Dict[str, bool]:
    """由 capabilities 功能 token 展开默认通道 mask。"""
    enabled: Set[str] = set(_BASE_FLEX_CHANNELS)
    for token, channels in _CAPABILITY_CHANNEL_EXPAND.items():
        if caps.get(token):
            enabled.update(channels)
    return {name: (name in enabled) for name in SEMANTIC_30_NAMES}


def _synthesize_semantic_map_from_o20(
    mapping: Dict[int, dict],
) -> Dict[str, dict]:
    """由 ``from_o20``（o20_idx → {to, invert, also}）合成通道名映射。

    P6：YAML 未声明 ``from_semantic`` 时自动启用，使 executor 主路径
    始终走通道名，而不再依赖运行时 ``project_s30_to_o20`` + ``to_hardware``。
    """
    out: Dict[str, dict] = {}
    for o20_idx, spec in mapping.items():
        s30_idx = o20_joint_index_to_s30(int(o20_idx))
        if s30_idx < 0:
            continue
        ch = SEMANTIC_30_NAMES[s30_idx]
        also_raw = spec.get("also") or []
        out[ch] = {
            "to": int(spec["to"]),
            "invert": bool(spec.get("invert", False)),
            "also": [int(x) for x in also_raw],
        }
    return out


class HandConfig:
    """手部型号配置，负责语义空间与硬件之间的转换。

    用法:
        cfg = HandConfig("O30")
        hw = cfg.from_semantic(vec30)           # P6 主路径
        vec30 = cfg.to_semantic(hw)
        hw = cfg.to_hardware(o20_angles)       # 兼容 from_o20
        o20 = cfg.from_hardware(hw)

    **注意**：与 ``primitive_base.HandConfig`` / ``SemanticPoseConfig``（S30
    关节下标表）是不同类。新代码可读别名 ``HardwareHandConfig``。
    """

    def __init__(self, hand_joint: str):
        path = _find_config_path(hand_joint)
        # Callers historically treat configuration lists and mappings as
        # mutable.  Keep instances isolated while removing YAML parsing from
        # control loops.
        data = copy.deepcopy(_load_config_data(path))

        self._hand_joint: str = data["hand_joint"]
        self._num_joints: int = data["num_joints"]
        self._joint_names: List[str] = data["joint_names"]
        self._name_to_hw: Dict[str, int] = {name: i for i, name in enumerate(self._joint_names)}

        # 保留位
        rsv = data.get("reserved", {})
        self._reserved_indices: List[int] = rsv.get("indices", [])
        self._reserved_default: float = float(rsv.get("default", 0))

        # 映射表: o20_idx → {to, invert, also}
        self._mapping: Dict[int, dict] = {}
        for k, v in data.get("from_o20", {}).items():
            self._mapping[int(k)] = v

        # 反向映射表: hw_idx → {from_o20_idx, invert}
        self._reverse_map: Dict[int, dict] = {}
        for o20_idx, spec in self._mapping.items():
            hw_idx = spec["to"]
            self._reverse_map[hw_idx] = {
                "from": o20_idx,
                "invert": spec.get("invert", False),
            }

        # 校准偏移: hw_idx → offset
        cal_data = data.get("calibration") or {}
        self._calibration: Dict[int, float] = {int(k): float(v) for k, v in cal_data.items()}

        # 原语超参覆盖: prim_name → {param_name: value}
        # 值以硬件约定存储，get_primitive_params() 时自动转为 O20
        self._primitive_params: dict = data.get("primitive_params", {})

        # 身份映射快捷标志 (O20 无需转换)
        self._is_identity = (
            self._num_joints == 20
            and all(
                spec["to"] == idx and not spec.get("invert", False)
                for idx, spec in self._mapping.items()
            )
            and not self._calibration
        )

        # --- P5: from_semantic / channels / coupling ---
        caps = data.get("capabilities") or {}
        channels = _expand_channels_from_capabilities(caps)
        for name, enabled in (data.get("channels") or {}).items():
            if name in SEMANTIC_30_IDX:
                channels[str(name)] = bool(enabled)
        self._channels: Dict[str, bool] = channels

        self._coupling: Dict[str, dict] = {
            str(k): dict(v) for k, v in (data.get("coupling") or {}).items()
        }

        # 可选：收敛比较通道（默认 = channels 启用集）
        conv_raw = data.get("convergence")
        self._convergence_names: Optional[List[str]] = None
        if isinstance(conv_raw, list):
            names: List[str] = []
            for item in conv_raw:
                ch = str(item)
                if ch not in SEMANTIC_30_IDX:
                    raise ValueError(f"{self._hand_joint} convergence 未知通道: {ch}")
                names.append(ch)
            self._convergence_names = names

        # channel → {to: hw_idx, invert, also: [hw_idx,...]}
        self._semantic_map: Dict[str, dict] = {}
        self._semantic_map_source: str = "none"  # yaml | synthesized | none
        for ch_name, spec in (data.get("from_semantic") or {}).items():
            ch = str(ch_name)
            if ch not in SEMANTIC_30_IDX:
                raise ValueError(f"{self._hand_joint} from_semantic 未知通道: {ch}")
            joint = spec.get("joint")
            if joint is None:
                raise ValueError(f"{self._hand_joint} from_semantic[{ch}] 缺少 joint")
            joint_name = str(joint)
            if joint_name not in self._name_to_hw:
                raise ValueError(f"{self._hand_joint} from_semantic[{ch}] 未知关节: {joint_name}")
            also_names = [str(x) for x in (spec.get("also") or [])]
            also_idx = []
            for an in also_names:
                if an not in self._name_to_hw:
                    raise ValueError(f"{self._hand_joint} from_semantic[{ch}].also 未知关节: {an}")
                also_idx.append(self._name_to_hw[an])
            self._semantic_map[ch] = {
                "to": self._name_to_hw[joint_name],
                "invert": bool(spec.get("invert", False)),
                "also": also_idx,
            }
        if self._semantic_map:
            self._semantic_map_source = "yaml"
        elif self._mapping:
            # P6：无显式 from_semantic 时由 from_o20 合成（兼容别名退居次要）
            self._semantic_map = _synthesize_semantic_map_from_o20(self._mapping)
            self._semantic_map_source = "synthesized"

        # hw_idx → [(channel, invert), ...] 用于 to_semantic（also 不反向）
        self._semantic_reverse: Dict[int, List[tuple]] = {}
        for ch, spec in self._semantic_map.items():
            self._semantic_reverse.setdefault(spec["to"], []).append((ch, spec["invert"]))

    @property
    def hand_joint(self) -> str:
        return self._hand_joint

    @property
    def num_joints(self) -> int:
        return self._num_joints

    @property
    def joint_names(self) -> List[str]:
        return self._joint_names

    @property
    def reserved_indices(self) -> List[int]:
        return self._reserved_indices

    @property
    def is_identity(self) -> bool:
        return self._is_identity

    @property
    def has_from_semantic(self) -> bool:
        """是否具备 semantic_30→硬件映射（YAML 显式或由 from_o20 合成）。"""
        return bool(self._semantic_map)

    @property
    def semantic_map_source(self) -> str:
        """``yaml`` | ``synthesized`` | ``none``。"""
        return self._semantic_map_source

    @property
    def channels(self) -> Dict[str, bool]:
        return dict(self._channels)

    def channel_enabled(self, name: str) -> bool:
        return bool(self._channels.get(name, False))

    def enabled_channel_names(self) -> List[str]:
        return [n for n, on in self._channels.items() if on]

    def enabled_channel_indices(self) -> List[int]:
        """启用通道在 semantic_30 中的下标（收敛 / mask 用）。"""
        return [SEMANTIC_30_IDX[n] for n in SEMANTIC_30_NAMES if self._channels.get(n)]

    def convergence_channel_indices(self) -> List[int]:
        """DONE 收敛比较的 semantic_30 下标。

        若 YAML 声明 ``convergence:`` 列表则用之（如 L25 排除耦合 tip）；
        否则回退 ``enabled_channel_indices()``。
        """
        if self._convergence_names is not None:
            return [SEMANTIC_30_IDX[n] for n in self._convergence_names]
        return self.enabled_channel_indices()

    def get_primitive_params(self, name: str) -> dict:
        """返回当前手型下指定原语的超参覆盖 (已转为 O20 内部表示)。

        无覆盖时返回空 dict，原语使用自己的 O20 默认值。
        """
        raw = self._primitive_params.get(name, {})
        if not raw:
            return {}
        result = {}
        for key, value in raw.items():
            if (
                key.endswith("_angles")
                and isinstance(value, list)
                and len(value) == self._num_joints
            ):
                result[key] = self.from_hardware(value)
            else:
                result[key] = value
        return result

    def to_hardware(self, o20_angles: List[float]) -> List[float]:
        """O20 20-DOF 内部表示 → 硬件指令。"""
        if self._is_identity:
            return list(o20_angles)

        out = [self._reserved_default] * self._num_joints
        for o20_idx, spec in self._mapping.items():
            val = float(o20_angles[o20_idx])
            if spec.get("invert", False):
                val = 255.0 - val
            hw_idx = spec["to"]
            out[hw_idx] = val
            for also_idx in spec.get("also", []):
                out[also_idx] = val

        # 校准偏移
        for hw_idx, offset in self._calibration.items():
            out[hw_idx] = max(0.0, min(255.0, out[hw_idx] + offset))

        return out

    def from_hardware(self, hw_angles: List[float]) -> List[float]:
        """硬件状态反馈 → O20 20-DOF 内部表示。"""
        if self._is_identity:
            return list(hw_angles[:20])

        out = [0.0] * 20
        for hw_idx, spec in self._reverse_map.items():
            if hw_idx >= len(hw_angles):
                continue
            val = float(hw_angles[hw_idx])
            # 反向校准
            if hw_idx in self._calibration:
                val -= self._calibration[hw_idx]
            if spec["invert"]:
                val = 255.0 - val
            o20_idx = spec["from"]
            out[o20_idx] = max(0.0, min(255.0, val))

        # 保留位清零
        for i in self._reserved_indices:
            out[i] = 0.0
        return out

    def _apply_coupling(self, vec30: List[float]) -> List[float]:
        """按 coupling 段改写语义向量（ignore / copy_from / fixed / scale）。"""
        out = list(vec30)
        for ch, spec in self._coupling.items():
            idx = SEMANTIC_30_IDX.get(ch)
            if idx is None:
                continue
            mode = str(spec.get("mode", "ignore")).lower()
            if mode == "ignore":
                out[idx] = 0.0
            elif mode == "fixed":
                out[idx] = float(spec.get("value", 0.0))
            elif mode == "copy_from":
                src = str(spec.get("source", ""))
                src_idx = SEMANTIC_30_IDX.get(src)
                if src_idx is not None:
                    out[idx] = out[src_idx]
            elif mode == "scale":
                src = str(spec.get("source", ""))
                src_idx = SEMANTIC_30_IDX.get(src)
                if src_idx is not None:
                    out[idx] = out[src_idx] * float(spec.get("factor", 1.0))
        return out

    def from_semantic(self, vec30: List[float]) -> List[float]:
        """semantic_30 → 硬件指令（P6 主路径）。

        无通道映射时回退：``project_s30_to_o20`` + ``to_hardware``。
        """
        if len(vec30) < SEMANTIC_DIM:
            raise ValueError(f"from_semantic: 需要 {SEMANTIC_DIM} 维，收到 {len(vec30)}")
        if not self._semantic_map:
            from .semantic_30 import project_s30_to_o20

            return self.to_hardware(project_s30_to_o20(vec30))

        coupled = self._apply_coupling([float(v) for v in vec30[:SEMANTIC_DIM]])
        out = [self._reserved_default] * self._num_joints
        for ch, spec in self._semantic_map.items():
            if not self._channels.get(ch, True):
                # 显式关闭的通道不写硬件
                continue
            idx = SEMANTIC_30_IDX[ch]
            val = coupled[idx]
            if spec["invert"]:
                val = 255.0 - val
            hw_idx = spec["to"]
            out[hw_idx] = val
            for also_idx in spec["also"]:
                out[also_idx] = val

        for hw_idx, offset in self._calibration.items():
            out[hw_idx] = max(0.0, min(255.0, out[hw_idx] + offset))
        return [max(0.0, min(255.0, v)) for v in out]

    def to_semantic(self, hw_angles: List[float]) -> List[float]:
        """硬件反馈 → semantic_30。

        无通道映射时回退：``from_hardware`` + ``project_o20_to_s30``。
        """
        if not self._semantic_map:
            return project_o20_to_s30(self.from_hardware(hw_angles))

        out = [0.0] * SEMANTIC_DIM
        for hw_idx, entries in self._semantic_reverse.items():
            if hw_idx >= len(hw_angles):
                continue
            val = float(hw_angles[hw_idx])
            if hw_idx in self._calibration:
                val -= self._calibration[hw_idx]
            for ch, invert in entries:
                v = 255.0 - val if invert else val
                out[SEMANTIC_30_IDX[ch]] = max(0.0, min(255.0, v))

        # coupling：无硬件轴的通道从源通道回填
        for ch, spec in self._coupling.items():
            idx = SEMANTIC_30_IDX.get(ch)
            if idx is None:
                continue
            mode = str(spec.get("mode", "")).lower()
            if mode == "copy_from":
                src_idx = SEMANTIC_30_IDX.get(str(spec.get("source", "")))
                if src_idx is not None:
                    out[idx] = out[src_idx]
            elif mode == "fixed":
                out[idx] = float(spec.get("value", 0.0))
        return out


def semantic_legacy255_to_hardware(
    hand: HandConfig,
    semantic,
    *,
    clamp: bool = True,
) -> List[float]:
    """S30 legacy255 语义向量 → 硬件 0~255（与 executor 命令路径一致）。"""
    from .semantic_30 import as_semantic_30, project_s30_to_o20

    vec = as_semantic_30(semantic)
    if hand.has_from_semantic:
        hw = hand.from_semantic(vec)
    else:
        hw = hand.to_hardware(project_s30_to_o20(vec))
    out = [float(v) for v in hw]
    if clamp:
        return [max(0.0, min(255.0, v)) for v in out]
    return out


# 缓存 L25 HandConfig 实例，避免重复加载 YAML
_l25_hand = None


def _get_l25_hand():
    global _l25_hand
    if _l25_hand is None:
        _l25_hand = HandConfig("l25")
    return _l25_hand


def l25_hw_to_o20(hw_angles):
    """L25 25-DOF 硬件空间 → O20 20-DOF 语义空间。"""
    return _get_l25_hand().from_hardware(hw_angles)


def l25_legacy_angles_to_o20(angles_25):
    """旧 L25 ANGLES 常量 → O20 20-DOF 语义空间。

    旧常量为旧 blanket invert (255-val 全关节) 设计：
    先还原为真正的 L25 硬件值 (255 - old)，再经 from_hardware 转为 O20。
    """
    true_hw = [255.0 - v for v in angles_25[:25]]
    return _get_l25_hand().from_hardware(true_hw)


# 清晰别名：避免与 ``primitive_base.SemanticPoseConfig`` 混淆。
HardwareHandConfig = HandConfig
