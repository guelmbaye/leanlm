"""Telemetry contract (MIB s.14).

Each capability publishes, at minimum: Started, Completed, Duration, Memory,
CPU, Warnings, Errors. A span is the unit of that contract.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

from .clock import Stopwatch, iso_now
from .errors import ErrorCategory, ErrorRecord, LeanLMError, Severity
from .events import EventBus, EventName


@dataclass
class StageRecord:
    """One executed pipeline stage. Immutable once the span closes."""

    stage: str
    capability: str
    started_at: str
    duration_ms: float = 0.0
    rss_mb: float = 0.0
    cpu_percent: float = 0.0
    status: str = "completed"
    warnings: list[str] = field(default_factory=list)
    errors: list[ErrorRecord] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "capability": self.capability,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "rss_mb": self.rss_mb,
            "cpu_percent": self.cpu_percent,
            "status": self.status,
            "warnings": list(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "fields": dict(self.fields),
        }


class Telemetry:
    """Emits capability spans onto the event bus and records stage metrics."""

    def __init__(self, bus: EventBus, session_id: str, resource_probe=None) -> None:
        self._bus = bus
        self._session_id = session_id
        self._probe = resource_probe
        self.records: list[StageRecord] = []

    @contextmanager
    def span(self, stage: str, capability: str, **fields: Any) -> Iterator[StageRecord]:
        record = StageRecord(stage=stage, capability=capability, started_at=iso_now(),
                             fields=dict(fields))
        self._bus.emit(EventName.CAPABILITY_STARTED, self._session_id,
                       capability=capability, stage=stage)
        watch = Stopwatch()
        try:
            yield record
        except LeanLMError as exc:
            record.status = "failed"
            record.errors.append(exc.record)
            record.duration_ms = watch.stop()
            self._finalize(record)
            self._bus.emit(EventName.CAPABILITY_FAILED, self._session_id,
                           capability=capability, stage=stage, code=exc.record.code)
            raise
        except Exception as exc:  # unexpected failures are still transparent
            record.status = "failed"
            record.errors.append(
                ErrorRecord(
                    code="RT-999",
                    category=ErrorCategory.RUNTIME,
                    severity=Severity.CRITICAL,
                    message=f"{type(exc).__name__}: {exc}",
                    capability=capability,
                    stage=stage,
                    recommended_action="inspect logs and re-run with --log-level debug",
                )
            )
            record.duration_ms = watch.stop()
            self._finalize(record)
            self._bus.emit(EventName.CAPABILITY_FAILED, self._session_id,
                           capability=capability, stage=stage, code="RT-999")
            raise
        else:
            record.duration_ms = watch.stop()
            self._finalize(record)
            self._bus.emit(EventName.CAPABILITY_COMPLETED, self._session_id,
                           capability=capability, stage=stage,
                           duration_ms=record.duration_ms)

    def _finalize(self, record: StageRecord) -> None:
        if self._probe is not None:
            try:
                sample = self._probe()
                record.rss_mb = sample.process_rss_mb
                record.cpu_percent = sample.cpu_percent
            except Exception:
                pass
        self.records.append(record)

    def total_ms(self) -> float:
        return round(sum(r.duration_ms for r in self.records), 3)

    def stage_durations(self) -> dict[str, float]:
        return {r.stage: r.duration_ms for r in self.records}
