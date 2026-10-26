# SPDX-License-Identifier: MIT
# Independently developed LinkerHand workbench code; see LICENSE and NOTICE.
"""Thread-safe "latest message" holders shared by ROS 2 backends.

Transport callbacks deliver messages on their thread; the host control loop reads them on
its own thread. ``LatestCache`` keeps only the most recent value plus the wall-clock and
message stamps so readers can decide whether data is fresh enough to act on.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Generic, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class Sample(Generic[T]):
    """One cached message with reception metadata."""

    value: T
    recv_monotonic: float  # time.monotonic() at reception
    stamp_ns: int  # header.stamp if available, else wall clock at reception
    seq: int  # monotonically increasing per cache
    receive_time_ns: int = 0
    source_stamp_ns: int | None = None

    def metadata(self, source: str) -> dict:
        return {
            "sample_id": f"{source}:{self.seq}",
            "source": source,
            "seq": self.seq,
            "source_stamp_ns": self.source_stamp_ns,
            "receive_time_ns": self.receive_time_ns,
            "receive_monotonic_ns": int(self.recv_monotonic * 1e9),
            "clock_domain": "ros_header_unverified"
            if self.source_stamp_ns is not None
            else "unavailable",
        }


class LatestCache(Generic[T]):
    """Lock-protected single-slot cache.

    ``put`` is called from rclpy callbacks; ``get``/``age_s``/``wait`` from the control loop.
    ``wait`` blocks until at least one sample newer than ``after_seq`` arrives.
    """

    def __init__(self, name: str = "") -> None:
        self.name = name
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._sample: Sample[T] | None = None
        self._seq = 0

    def put(
        self, value: T, stamp_ns: int | None = None, *, received_ns=None, received_monotonic=None
    ) -> Sample[T]:
        now = time.monotonic() if received_monotonic is None else received_monotonic
        received_ns = time.time_ns() if received_ns is None else received_ns
        with self._cond:
            self._seq += 1
            sample = Sample(
                value,
                now,
                stamp_ns if stamp_ns is not None else received_ns,
                self._seq,
                received_ns,
                stamp_ns,
            )
            self._sample = sample
            self._cond.notify_all()
        return sample

    def get(self) -> Sample[T] | None:
        with self._lock:
            return self._sample

    @property
    def has_value(self) -> bool:
        return self.get() is not None

    def age_s(self) -> float:
        """Seconds since the last reception (``inf`` when nothing was received yet)."""
        sample = self.get()
        return float("inf") if sample is None else time.monotonic() - sample.recv_monotonic

    def is_fresh(self, max_age_s: float) -> bool:
        return self.age_s() <= max_age_s

    def wait(self, timeout_s: float, after_seq: int = 0) -> Sample[T] | None:
        """Block until a sample with ``seq > after_seq`` exists or ``timeout_s`` elapses."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._sample is None or self._sample.seq <= after_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cond.wait(remaining)
            return self._sample

    def map(self, fn: Callable[[T], T]) -> None:
        """Apply ``fn`` to the cached value in place (used by tests/mocks)."""
        with self._cond:
            if self._sample is not None:
                self._sample = replace(self._sample, value=fn(self._sample.value))
