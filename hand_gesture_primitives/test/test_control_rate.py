"""Tests for configurable gesture command rate helpers."""

import pytest
from hand_gesture_primitives.contact_config import (
    ContactThresholds,
    clamp_control_rate_hz,
    scale_per_tick_delta,
    scale_tick_count,
)


def test_scale_tick_count_keeps_wall_clock() -> None:
    assert scale_tick_count(10, 10.0) == 10
    assert scale_tick_count(10, 100.0) == 100
    assert scale_tick_count(3, 100.0) == 30
    assert scale_tick_count(0, 100.0) == 0


def test_scale_per_tick_delta_keeps_wall_clock_speed() -> None:
    # finger_step=8 @10Hz → 80 units/s；100Hz 时每 tick 应为 0.8
    assert scale_per_tick_delta(8.0, 10.0) == pytest.approx(8.0)
    assert scale_per_tick_delta(8.0, 100.0) == pytest.approx(0.8)
    assert scale_per_tick_delta(8.0, 50.0) == pytest.approx(1.6)
    assert scale_per_tick_delta(0.0, 100.0) == 0.0


def test_scaled_hold_retreat_step() -> None:
    from hand_gesture_primitives.contact_config import scaled_hold_retreat_step

    # over=30 → nominal 1 @10Hz；100Hz → 0.1
    assert scaled_hold_retreat_step(30.0, 10.0) == pytest.approx(1.0)
    assert scaled_hold_retreat_step(30.0, 100.0) == pytest.approx(0.1)
    assert scaled_hold_retreat_step(60.0, 10.0) == pytest.approx(2.0)
    assert scaled_hold_retreat_step(60.0, 100.0) == pytest.approx(0.2)


def test_clamp_control_rate_accepts_100hz() -> None:
    assert clamp_control_rate_hz(100) == 100.0
    assert clamp_control_rate_hz(10.0) == 10.0


def test_clamp_control_rate_rejects_out_of_range() -> None:
    with pytest.raises(ValueError):
        clamp_control_rate_hz(0.5)
    with pytest.raises(ValueError):
        clamp_control_rate_hz(500)


def test_contact_thresholds_scale_settle_frames() -> None:
    base = ContactThresholds(current_settle_frames=10)
    scaled = base.scaled_for_control_rate(100.0)
    assert scaled.current_settle_frames == 100
    assert base.current_settle_frames == 10
