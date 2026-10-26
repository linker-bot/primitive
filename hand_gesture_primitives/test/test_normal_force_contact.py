from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from hand_gesture_primitives.contact_config import ContactThresholds
from hand_gesture_primitives.contact_detection import tactile_contact
from hand_gesture_primitives.executor import GestureExecutor
from hand_gesture_primitives.primitive_base import PrimitiveContext


def _context(mode: str, value: float) -> PrimitiveContext:
    return PrimitiveContext(
        tactile_mode=mode,
        tactile_pressure=np.asarray([value, 0.0, 0.0, 0.0, 0.0]),
        contact_thresholds=ContactThresholds(
            pressure_threshold=20.0,
            mass_threshold=2.0,
            normal_force_threshold_n=2.0,
        ),
    )


def test_normal_force_uses_newton_threshold() -> None:
    assert not tactile_contact(_context("normal_force_n", 2.0), [0])
    assert tactile_contact(_context("normal_force_n", 2.01), [0])


def test_pressure_threshold_is_unchanged() -> None:
    assert not tactile_contact(_context("pressure", 3.0), [0])
    assert tactile_contact(_context("pressure", 20.01), [0])


def test_mass_threshold_is_unchanged() -> None:
    assert not tactile_contact(_context("mass", 2.0), [0])
    assert tactile_contact(_context("mass", 2.01), [0])


def test_executor_keeps_structured_pressure_and_normal_force_units_separate() -> None:
    class Message:
        data = [3.0, 0.0, 0.0, 0.0, 0.0]

    hand_force = SimpleNamespace(
        finger_names=["thumb", "index", "middle", "ring", "little"],
        finger_force=SimpleNamespace(
            data=[3.0, 0.0, 0.0, 0.0, 0.0],
            layout=SimpleNamespace(dim=[SimpleNamespace(size=size) for size in (5, 1, 1)]),
        ),
        palm_force=SimpleNamespace(data=[]),
    )
    executor = GestureExecutor.__new__(GestureExecutor)
    executor._contact_thresholds = ContactThresholds(
        pressure_threshold=20.0,
        normal_force_threshold_n=2.0,
    )

    executor._force_callback(hand_force)
    assert executor._tactile_mode == "pressure"
    assert not executor._contact_detected

    executor._normal_force_callback(Message())
    assert executor._tactile_mode == "normal_force_n"
    assert executor._contact_detected


def test_executor_reads_structured_finger_and_palm_pressure() -> None:
    def matrix(data, *sizes):
        return SimpleNamespace(
            data=data,
            layout=SimpleNamespace(dim=[SimpleNamespace(size=size) for size in sizes]),
        )

    message = SimpleNamespace(
        finger_names=["thumb", "index", "middle", "ring", "little"],
        finger_force=matrix(
            [0.0, 21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            5,
            1,
            2,
        ),
        palm_force=matrix([0.0, 25.0], 1, 2),
    )
    executor = GestureExecutor.__new__(GestureExecutor)
    executor._contact_thresholds = ContactThresholds(
        pressure_threshold=20.0,
        normal_force_threshold_n=2.0,
    )

    executor._force_callback(message)

    assert executor._contact_detected
    assert executor._tactile_mode == "pressure"
    assert np.array_equal(executor._tactile_pressure, [21.0, 0.0, 0.0, 0.0, 0.0])
    assert executor._palm_pressure == 25.0
