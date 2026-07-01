"""Load contact / current thresholds from YAML and ROS parameters."""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
_DEFAULT_FILENAME = "contact_thresholds.yaml"


@dataclass
class ContactThresholds:
    """Tactile and joint-current thresholds for grasp contact detection."""

    pressure_threshold: float = 20.0
    mass_threshold: float = 2.0
    current_delta: float = 270.0
    current_delta_narrow: float = 250.0
    current_settle_frames: int = 10
    hold_safe_current: float = 800.0
    # O6 hand.torque：0~100 无量纲%（100≈1657.5mA）
    torque_delta_pct: float = 15.0
    hold_safe_torque_pct: float = 55.0
    pinch_torque_stop_pct: float = 15.0
    # 静态 pinch 堵转：绝对力矩超过此值才停（空载 lerp 不误触）
    pinch_stall_torque_pct: float = 50.0
    overload_threshold: float = 1000.0
    overload_duration_sec: float = 2.0

    def contact_delta(self, narrow: bool = False) -> float:
        return self.current_delta_narrow if narrow else self.current_delta


def _resolve_config_path(config_path: Optional[str] = None) -> Optional[str]:
    if config_path and os.path.isfile(config_path):
        return config_path
    try:
        from ament_index_python.packages import get_package_share_directory
        share = get_package_share_directory("hand_gesture_primitives")
        path = os.path.join(share, "config", _DEFAULT_FILENAME)
        if os.path.isfile(path):
            return path
    except Exception:
        pass
    path = os.path.join(_CONFIG_DIR, _DEFAULT_FILENAME)
    if os.path.isfile(path):
        return path
    return None


def _section_from_yaml(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not data:
        return {}
    if "contact" in data:
        return data["contact"] or {}
    return data


def load_contact_thresholds(config_path: Optional[str] = None) -> ContactThresholds:
    """Load thresholds from YAML; missing keys fall back to dataclass defaults."""
    defaults = ContactThresholds()
    path = _resolve_config_path(config_path)
    section: Dict[str, Any] = {}
    if path:
        with open(path, "r", encoding="utf-8") as f:
            section = _section_from_yaml(yaml.safe_load(f))

    def _get(key: str, fallback: Any) -> Any:
        return section.get(key, fallback)

    return ContactThresholds(
        pressure_threshold=float(_get("pressure_threshold", defaults.pressure_threshold)),
        mass_threshold=float(_get("mass_threshold", defaults.mass_threshold)),
        current_delta=float(_get("current_delta", defaults.current_delta)),
        current_delta_narrow=float(_get("current_delta_narrow", defaults.current_delta_narrow)),
        current_settle_frames=int(_get("current_settle_frames", defaults.current_settle_frames)),
        hold_safe_current=float(_get("hold_safe_current", defaults.hold_safe_current)),
        torque_delta_pct=float(_get("torque_delta_pct", defaults.torque_delta_pct)),
        hold_safe_torque_pct=float(
            _get("hold_safe_torque_pct", defaults.hold_safe_torque_pct)),
        pinch_torque_stop_pct=float(
            _get("pinch_torque_stop_pct", defaults.pinch_torque_stop_pct)),
        pinch_stall_torque_pct=float(
            _get("pinch_stall_torque_pct", defaults.pinch_stall_torque_pct)),
        overload_threshold=float(_get("overload_threshold", defaults.overload_threshold)),
        overload_duration_sec=float(_get("overload_duration_sec", defaults.overload_duration_sec)),
    )
