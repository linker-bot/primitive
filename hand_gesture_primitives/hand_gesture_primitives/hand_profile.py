"""手型配置档案 — 关节映射 + 原语支持策略（由 config/{hand}.yaml 驱动）。"""

from __future__ import annotations

import glob
import os
from typing import Dict, List, Optional, Set

import yaml

from .hand_config import HandConfig, _find_config_path, _CONFIG_DIR
from .primitive_catalog import (
    CAP_CURRENT,
    CAP_FINGER_ABD,
    CAP_FINGER_ROOT2,
    CAP_FINGER_TIPS,
    CAP_MOTOR_TORQUE,
    CAP_THUMB_ABD,
    CAP_THUMB_ROT,
    CAP_THUMB_TIP,
    CAP_VISION_FK,
    CAP_TOUCH,
    missing_capabilities,
    catalog_entry,
)
from .primitives import PRIMITIVE_REGISTRY

_CONTACT_CONFIG = "contact_thresholds.yaml"

# 能力 token → 用户可读说明
CAPABILITY_LABELS: Dict[str, str] = {
    CAP_FINGER_ABD: "食/中/无/小指侧摆",
    CAP_THUMB_ABD: "拇指侧摆",
    CAP_THUMB_ROT: "拇指旋转",
    CAP_FINGER_TIPS: "独立指尖弯曲",
    CAP_THUMB_TIP: "拇指指尖弯曲",
    CAP_FINGER_ROOT2: "指根第二关节",
    CAP_VISION_FK: "视觉正运动学",
    CAP_TOUCH: "触觉/压感",
    CAP_CURRENT: "关节电流反馈",
    CAP_MOTOR_TORQUE: "关节力矩反馈",
}

# 手型 × 原语 → 推荐替代（用于拒绝提示）
_PRIMITIVE_ALTERNATIVES: Dict[str, Dict[str, str]] = {
    "o6": {
        "index_middle_adduction_grip": "thumb_adduction_grip",
    },
}


class HandProfile:
    """单手机械手的完整运行档案。"""

    def __init__(self, hand_joint: str):
        path = _find_config_path(hand_joint)
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self._hand_joint = str(raw.get("hand_joint", hand_joint))
        self._hand_type = self._hand_joint.lower()
        self._config_path = path
        self._hardware = HandConfig(hand_joint)

        prim = raw.get("primitives") or {}
        self._mode: str = str(prim.get("mode", "allowlist")).lower()
        self._supported_explicit: List[str] = list(prim.get("supported") or [])
        self._unsupported: Set[str] = set(prim.get("unsupported") or [])
        self._grasp_type_map: Dict[str, str] = dict(prim.get("grasp_type_map") or {})
        self._aliases: Dict[str, str] = {
            str(k): str(v) for k, v in (prim.get("aliases") or {}).items()
        }
        self._gestures: dict = dict(raw.get("gestures") or {})

        caps = raw.get("capabilities") or {}
        self._capabilities: Set[str] = {
            k for k, v in caps.items() if v and k != "description"
        }

        self._supported_set: Optional[Set[str]] = None
        self._validate_config()

    @property
    def hand_joint(self) -> str:
        return self._hand_joint

    @property
    def hand_type(self) -> str:
        return self._hand_type

    @property
    def hardware(self) -> HandConfig:
        return self._hardware

    @property
    def capabilities(self) -> frozenset:
        return frozenset(self._capabilities)

    @property
    def config_path(self) -> str:
        return self._config_path

    def _validate_config(self) -> None:
        unknown = [
            n for n in self._supported_explicit
            if n not in PRIMITIVE_REGISTRY
        ]
        if unknown:
            raise ValueError(
                f"{self._hand_joint} config 含未注册原语: {unknown}")

        for alias_target in self._aliases.values():
            if alias_target not in PRIMITIVE_REGISTRY:
                raise ValueError(
                    f"{self._hand_joint} alias 目标未注册: {alias_target}")

        for grasp, prim in self._grasp_type_map.items():
            if prim and prim not in PRIMITIVE_REGISTRY:
                raise ValueError(
                    f"{self._hand_joint} grasp_type_map[{grasp!r}] "
                    f"未注册: {prim}")

    def _compute_supported(self) -> Set[str]:
        if self._supported_set is not None:
            return self._supported_set

        if self._mode == "capability":
            supported = {
                name for name in PRIMITIVE_REGISTRY
                if not missing_capabilities(name, self.capabilities)
            }
        else:
            if self._supported_explicit:
                supported = set(self._supported_explicit)
            else:
                supported = set(PRIMITIVE_REGISTRY.keys())

        supported -= self._unsupported
        self._supported_set = supported
        return supported

    def resolve_primitive_name(self, name: str) -> str:
        """解析别名 → 注册名。"""
        return self._aliases.get(name, name)

    def is_primitive_supported(self, name: str) -> bool:
        name = self.resolve_primitive_name(name)
        if name not in PRIMITIVE_REGISTRY:
            return False
        return name in self._compute_supported()

    def reject_reason(self, name: str) -> str:
        """不支持时返回可读原因，支持则返回空串。"""
        resolved = self.resolve_primitive_name(name)
        if resolved not in PRIMITIVE_REGISTRY:
            return f"unknown_primitive:{name}"
        if resolved in self._unsupported:
            return f"unsupported_on_{self._hand_type}:{resolved}"
        if resolved not in self._compute_supported():
            missing = missing_capabilities(resolved, self.capabilities)
            if missing:
                return (
                    f"capability_missing:{resolved}:"
                    f"{','.join(sorted(missing))}")
            return f"not_in_allowlist:{resolved}"
        cap_missing = missing_capabilities(resolved, self.capabilities)
        if cap_missing:
            return (
                f"capability_warning:{resolved}:"
                f"{','.join(sorted(cap_missing))}")
        return ""

    def reject_user_message(self, name: str) -> str:
        """不支持时返回用户可读中文说明，支持则返回空串。"""
        reason = self.reject_reason(name)
        if not reason:
            return ""
        return format_reject_message(
            self.hand_joint,
            self.hand_type,
            reason,
            num_joints=self._hardware.num_joints,
        )

    def supported_primitives(self) -> List[str]:
        return sorted(self._compute_supported())

    def gesture_config(self, name: str) -> dict:
        """返回 gestures 段中某原语的原始 YAML 配置。"""
        return dict(self._gestures.get(name) or {})

    def primitive_for_grasp_type(self, grasp_type: str) -> Optional[str]:
        if not grasp_type:
            return None
        key = grasp_type.lower().strip()
        prim = self._grasp_type_map.get(key)
        if prim is None:
            return None
        if prim == "" or prim.lower() == "none":
            return None
        if not self.is_primitive_supported(prim):
            return None
        return prim

    def summary_lines(self) -> List[str]:
        """启动日志用摘要。"""
        supported = self.supported_primitives()
        lines = [
            f"手型档案: {self._hand_joint} ({self._hardware.num_joints} DOF)",
            f"配置: {self._config_path}",
            f"原语策略: {self._mode}, 支持 {len(supported)}/{len(PRIMITIVE_REGISTRY)}",
            f"capabilities: {sorted(self._capabilities)}",
        ]
        if self._gestures:
            lines.append(f"gestures: {sorted(self._gestures.keys())}")
        by_cat: Dict[str, List[str]] = {}
        for n in supported:
            entry = catalog_entry(n)
            cat = entry.category if entry else "gesture"
            by_cat.setdefault(cat, []).append(n)
        for cat in ("safe", "gesture", "force", "vision"):
            if cat in by_cat:
                lines.append(f"  [{cat}] {', '.join(by_cat[cat])}")
        return lines


def list_configured_hand_joints() -> List[str]:
    """列出 config/ 下所有手型 YAML（排除 contact_thresholds）。"""
    names: Set[str] = set()
    for pattern in (
        os.path.join(_CONFIG_DIR, "*.yaml"),
        os.path.join(_CONFIG_DIR, "..", "..", "config", "*.yaml"),
    ):
        for path in glob.glob(pattern):
            base = os.path.basename(path)
            if base == _CONTACT_CONFIG:
                continue
            names.add(os.path.splitext(base)[0])
    try:
        from ament_index_python.packages import get_package_share_directory
        share_cfg = os.path.join(
            get_package_share_directory("hand_gesture_primitives"), "config")
        for path in glob.glob(os.path.join(share_cfg, "*.yaml")):
            base = os.path.basename(path)
            if base == _CONTACT_CONFIG:
                continue
            names.add(os.path.splitext(base)[0])
    except Exception:
        pass
    return sorted(names, key=str.lower)


def format_reject_message(
    hand_joint: str,
    hand_type: str,
    reason: str,
    *,
    num_joints: Optional[int] = None,
) -> str:
    """将 reject_reason 机器码转为用户可读中文提示。"""
    if not reason:
        return ""

    hand_label = hand_joint.upper() if hand_joint else hand_type.upper()
    dof_hint = f" ({num_joints} DOF)" if num_joints is not None else ""

    if reason.startswith("capability_missing:"):
        parts = reason.split(":", 2)
        prim = parts[1] if len(parts) > 1 else "?"
        caps = parts[2].split(",") if len(parts) > 2 and parts[2] else []
        cap_text = "、".join(CAPABILITY_LABELS.get(c, c) for c in caps if c)
        msg = (
            f"{hand_label}{dof_hint} 不支持手势原语 '{prim}'："
            f"缺少硬件能力 [{cap_text}]，关节自由度不足。"
        )
        if prim == "index_middle_adduction_grip":
            msg += (
                " 该原语依赖食/中指侧摆 (abd) 张开再并拢，"
                "O6 无指侧摆关节，无法实现「筷子式」侧夹。"
            )
        elif prim == "ring":
            msg += (
                " 该原语依赖独立指尖与指侧摆力控包络，"
                "O6 为 6 DOF 欠驱动手，无法实现 O20 版 ring。"
            )
        alt = _PRIMITIVE_ALTERNATIVES.get(hand_type.lower(), {}).get(prim)
        if alt:
            msg += f" 建议使用替代原语: {alt}。"
        return msg

    if reason.startswith("not_in_allowlist:"):
        prim = reason.split(":", 1)[1]
        alt = _PRIMITIVE_ALTERNATIVES.get(hand_type.lower(), {}).get(prim)
        msg = (
            f"{hand_label}{dof_hint} 未启用原语 '{prim}'"
            f"（不在本手型 allowlist 中）。"
        )
        if alt:
            msg += f" 建议使用: {alt}。"
        return msg

    if reason.startswith("unsupported_on_"):
        rest = reason.split(":", 1)[1]
        hand, _, prim = rest.partition(":")
        msg = f"{hand_label}{dof_hint} 配置禁用原语 '{prim or rest}'。"
        alt = _PRIMITIVE_ALTERNATIVES.get(hand_type.lower(), {}).get(prim or rest)
        if alt:
            msg += f" 建议使用: {alt}。"
        return msg

    if reason.startswith("unknown_primitive:"):
        prim = reason.split(":", 1)[1]
        return f"未知手势原语 '{prim}'，请检查指令拼写。"

    if reason.startswith("capability_warning:"):
        parts = reason.split(":", 2)
        prim = parts[1] if len(parts) > 1 else "?"
        caps = parts[2].split(",") if len(parts) > 2 and parts[2] else []
        cap_text = "、".join(CAPABILITY_LABELS.get(c, c) for c in caps if c)
        return (
            f"{hand_label}{dof_hint} 原语 '{prim}' 将以 MCP 近似姿态运行"
            f"（本手型无 [{cap_text}]），可继续执行。"
        )

    return reason


def load_hand_profile(hand_type_or_joint: str) -> HandProfile:
    """按 hand_type / hand_joint 加载档案（大小写不敏感）。"""
    key = hand_type_or_joint.strip()
    if not key:
        raise ValueError("hand_type / hand_joint 不能为空")
    available = {n.lower(): n for n in list_configured_hand_joints()}
    normalized = key.lower()
    if normalized not in available:
        raise FileNotFoundError(
            f"未找到手型配置 {key!r}，可用: {sorted(available.values())}")
    return HandProfile(available[normalized])
