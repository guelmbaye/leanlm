"""Time sources.

Two distinct clocks are used deliberately:
  * ``wall_now`` for timestamps written into artifacts (traceability);
  * ``monotonic_ns`` for durations (measurement, never affected by NTP jumps).
"""
from __future__ import annotations

import datetime as _dt
import time


def wall_now() -> _dt.datetime:
    """UTC wall clock, timezone aware."""
    return _dt.datetime.now(_dt.timezone.utc)


def iso_now() -> str:
    """ISO-8601 UTC timestamp with second precision (stable formatting)."""
    return wall_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def monotonic_ns() -> int:
    return time.monotonic_ns()


class Stopwatch:
    """Monotonic stopwatch used by every telemetry span."""

    __slots__ = ("_start", "_stop")

    def __init__(self) -> None:
        self._start = monotonic_ns()
        self._stop: int | None = None

    def stop(self) -> float:
        self._stop = monotonic_ns()
        return self.elapsed_ms

    @property
    def elapsed_ms(self) -> float:
        end = self._stop if self._stop is not None else monotonic_ns()
        return round((end - self._start) / 1_000_000.0, 3)
