"""Runtime and capability events (Runtime Blueprint s.14, ICIB s.10).

Events carry metadata only -- never document content, never user text. They are
the substrate of observability: dashboard, logs, benchmark reports.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable

from .clock import iso_now


class EventName(str, enum.Enum):
    SESSION_STARTED = "SESSION_STARTED"
    MODEL_LOADED = "MODEL_LOADED"
    RESOURCES_ASSESSED = "RESOURCES_ASSESSED"
    CAPABILITY_STARTED = "CAPABILITY_STARTED"
    CAPABILITY_COMPLETED = "CAPABILITY_COMPLETED"
    CAPABILITY_FAILED = "CAPABILITY_FAILED"
    DOCUMENT_LOADED = "DOCUMENT_LOADED"
    CONTEXT_LOADED = "CONTEXT_LOADED"
    CONTEXT_OPTIMIZED = "CONTEXT_OPTIMIZED"
    EVIDENCE_RETRIEVED = "EVIDENCE_RETRIEVED"
    PROMPT_BUILT = "PROMPT_BUILT"
    INFERENCE_STARTED = "INFERENCE_STARTED"
    FIRST_TOKEN = "FIRST_TOKEN"
    INFERENCE_FINISHED = "INFERENCE_FINISHED"
    VALIDATION_COMPLETED = "VALIDATION_COMPLETED"
    PIPELINE_COMPLETED = "PIPELINE_COMPLETED"
    RESOURCE_WARNING = "RESOURCE_WARNING"
    BENCHMARK_COMPLETED = "BENCHMARK_COMPLETED"
    SUBMISSION_GENERATED = "SUBMISSION_GENERATED"
    SESSION_CLOSED = "SESSION_CLOSED"


@dataclass(frozen=True, slots=True)
class Event:
    name: EventName
    session_id: str
    timestamp: str = field(default_factory=iso_now)
    capability: str = "runtime"
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name.value,
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "capability": self.capability,
            "payload": dict(self.payload),
        }


Subscriber = Callable[[Event], None]


class EventBus:
    """Synchronous, in-process, allocation-light event bus.

    Synchronous on purpose: an 8 GB laptop does not need a thread pool to move
    a few hundred metadata dictionaries, and synchronous delivery keeps the
    execution order deterministic (IC-04).
    """

    def __init__(self, *, keep_last: int = 2000) -> None:
        self._subscribers: list[Subscriber] = []
        self._history: list[Event] = []
        self._keep_last = keep_last

    def subscribe(self, subscriber: Subscriber) -> Callable[[], None]:
        self._subscribers.append(subscriber)
        return lambda: self._subscribers.remove(subscriber)

    def publish(self, event: Event) -> Event:
        self._history.append(event)
        if len(self._history) > self._keep_last:
            del self._history[: len(self._history) - self._keep_last]
        for subscriber in list(self._subscribers):
            try:
                subscriber(event)
            except Exception:  # a broken listener must never break the pipeline
                continue
        return event

    def emit(self, name: EventName, session_id: str, *, capability: str = "runtime",
             **payload: Any) -> Event:
        return self.publish(Event(name=name, session_id=session_id,
                                  capability=capability, payload=payload))

    def history(self, session_id: str | None = None) -> list[Event]:
        if session_id is None:
            return list(self._history)
        return [e for e in self._history if e.session_id == session_id]

    def clear(self) -> None:
        self._history.clear()
