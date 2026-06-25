"""手部型号配置加载 — 根据 hand_joint 参数加载 YAML，提供 O20↔硬件映射。"""

import os
from typing import Dict, List, Optional

import yaml


# 默认配置目录 (source tree, 开发时使用)
_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")


def _find_config_path(hand_joint: str) -> str:
    """查找配置文件路径: 先 ament share, 再 source tree。"""
    filename = f"{hand_joint.lower()}.yaml"
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
    raise FileNotFoundError(
        f"找不到手部配置: {filename} (搜索: ament share + {_CONFIG_DIR})")


class HandConfig:
    """手部型号配置，负责 O20 内部表示与硬件之间的转换。

    用法:
        cfg = HandConfig("L25")
        hw_angles = cfg.to_hardware(o20_angles)   # 原语输出 → 驱动指令
        o20_angles = cfg.from_hardware(hw_angles)  # 驱动反馈 → 原语输入
    """

    def __init__(self, hand_joint: str):
        path = _find_config_path(hand_joint)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._hand_joint: str = data["hand_joint"]
        self._num_joints: int = data["num_joints"]
        self._joint_names: List[str] = data["joint_names"]

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
        self._calibration: Dict[int, float] = {
            int(k): float(v)
            for k, v in cal_data.items()
        }

        # 身份映射快捷标志 (O20 无需转换)
        self._is_identity = (
            self._num_joints == 20
            and all(
                spec["to"] == idx and not spec.get("invert", False)
                for idx, spec in self._mapping.items()
            )
            and not self._calibration
        )

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
