"""Load contact / current thresholds from YAML and ROS parameters."""

import os
from dataclasses import dataclass, replace
from typing import Any, Dict, Optional

import yaml

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "config")
_DEFAULT_FILENAME = "contact_thresholds.yaml"

# YAML / 代码里按 tick 计数的阈值均按此基准标定；实际 control_rate 变化时再缩放。
REFERENCE_CONTROL_RATE_HZ = 10.0
DEFAULT_CONTROL_RATE_HZ = 10.0
MIN_CONTROL_RATE_HZ = 1.0
MAX_CONTROL_RATE_HZ = 200.0


def clamp_control_rate_hz(rate: float) -> float:
    """Validate and clamp gesture command loop rate (Hz)."""
    value = float(rate)
    if value < MIN_CONTROL_RATE_HZ or value > MAX_CONTROL_RATE_HZ:
        raise ValueError(
            f"control_rate 必须在 {MIN_CONTROL_RATE_HZ:g}~{MAX_CONTROL_RATE_HZ:g} Hz，"
            f"当前为: {rate!r}"
        )
    return value


def scale_tick_count(
    frames: int,
    control_rate_hz: float,
    reference_hz: float = REFERENCE_CONTROL_RATE_HZ,
) -> int:
    """把按 reference_hz 标定的 tick 数换算到当前 control_rate。"""
    n = int(frames)
    if n <= 0:
        return 0
    rate = float(control_rate_hz)
    ref = float(reference_hz)
    if rate <= 0.0 or ref <= 0.0:
        raise ValueError(
            f"control_rate/reference 必须为正，收到 control_rate={control_rate_hz!r}, "
            f"reference={reference_hz!r}"
        )
    if abs(rate - ref) < 1e-9:
        return n
    return max(1, int(round(n * rate / ref)))


def scale_per_tick_delta(
    delta: float,
    control_rate_hz: float,
    reference_hz: float = REFERENCE_CONTROL_RATE_HZ,
) -> float:
    """把按 reference_hz 每 tick 标定的位置增量换算到当前 control_rate。

    例如 finger_step=8 @10Hz → 100Hz 时每 tick 仅 +0.8，墙钟闭合速度不变。
    """
    value = float(delta)
    if value == 0.0:
        return 0.0
    rate = float(control_rate_hz)
    ref = float(reference_hz)
    if rate <= 0.0 or ref <= 0.0:
        raise ValueError(
            f"control_rate/reference 必须为正，收到 control_rate={control_rate_hz!r}, "
            f"reference={reference_hz!r}"
        )
    if abs(rate - ref) < 1e-9:
        return value
    return value * ref / rate


def scaled_hold_retreat_step(
    over: float,
    control_rate_hz: float,
    divisor: float = 30.0,
) -> float:
    """HOLD 过流回退步长：按 10Hz 标定后随 control_rate 缩放。"""
    nominal = max(1.0, float(over) / float(divisor))
    return scale_per_tick_delta(nominal, control_rate_hz)


@dataclass
class ContactThresholds:
    """Tactile and joint-current thresholds for grasp contact detection."""

    pressure_threshold: float = 20.0
    mass_threshold: float = 2.0
    normal_force_threshold_n: float = 2.0
    current_delta: float = 270.0
    current_delta_narrow: float = 250.0
    # 按 REFERENCE_CONTROL_RATE_HZ 标定；启动时由 scaled_for_control_rate 换算
    current_settle_frames: int = 10
    hold_safe_current: float = 800.0
    # O6 hand.torque：0~100 无量纲%（100≈1657.5mA）
    torque_delta_pct: float = 30.0
    hold_safe_torque_pct: float = 80.0
    pinch_torque_stop_pct: float = 15.0
    # 静态 pinch 堵转：绝对力矩超过此值才停（空载 lerp 不误触）
    pinch_stall_torque_pct: float = 50.0
    overload_threshold: float = 1000.0
    overload_duration_sec: float = 2.0

    def contact_delta(self, narrow: bool = False) -> float:
        return self.current_delta_narrow if narrow else self.current_delta

    def scaled_for_control_rate(self, control_rate_hz: float) -> "ContactThresholds":
        """返回 settle 帧数已换算到当前下发频率的副本。"""
        return replace(
            self,
            current_settle_frames=scale_tick_count(
                self.current_settle_frames, control_rate_hz
            ),
        )


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
        normal_force_threshold_n=float(
            _get("normal_force_threshold_n", defaults.normal_force_threshold_n)
        ),
        current_delta=float(_get("current_delta", defaults.current_delta)),
        current_delta_narrow=float(_get("current_delta_narrow", defaults.current_delta_narrow)),
        current_settle_frames=int(_get("current_settle_frames", defaults.current_settle_frames)),
        hold_safe_current=float(_get("hold_safe_current", defaults.hold_safe_current)),
        torque_delta_pct=float(_get("torque_delta_pct", defaults.torque_delta_pct)),
        hold_safe_torque_pct=float(_get("hold_safe_torque_pct", defaults.hold_safe_torque_pct)),
        pinch_torque_stop_pct=float(_get("pinch_torque_stop_pct", defaults.pinch_torque_stop_pct)),
        pinch_stall_torque_pct=float(
            _get("pinch_stall_torque_pct", defaults.pinch_stall_torque_pct)
        ),
        overload_threshold=float(_get("overload_threshold", defaults.overload_threshold)),
        overload_duration_sec=float(_get("overload_duration_sec", defaults.overload_duration_sec)),
    )
