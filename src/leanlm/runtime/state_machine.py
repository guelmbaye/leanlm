"""Runtime state machine (Runtime Blueprint s.13).

Every transition is observable. An illegal transition is a bug, not a warning:
it means a stage ran out of order, which is exactly what DIC-01 forbids.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field

from ..shared.clock import iso_now
from ..shared.errors import contract_error


class RuntimeState(str, enum.Enum):
    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    OPTIMIZING = "optimizing"
    RETRIEVING = "retrieving"
    PROMPT_READY = "prompt_ready"
    INFERENCING = "inferencing"
    VALIDATING = "validating"
    COMPLETED = "completed"
    FAILED = "failed"


_ALLOWED: dict[RuntimeState, tuple[RuntimeState, ...]] = {
    RuntimeState.IDLE: (RuntimeState.LOADING, RuntimeState.FAILED),
    RuntimeState.LOADING: (RuntimeState.READY, RuntimeState.FAILED),
    # READY -> LOADING is legitimate: reloading the model is how a cold start
    # is measured (scenario S5) and how a model swap happens mid-session.
    RuntimeState.READY: (RuntimeState.OPTIMIZING, RuntimeState.LOADING,
                         RuntimeState.IDLE, RuntimeState.FAILED),
    RuntimeState.OPTIMIZING: (RuntimeState.RETRIEVING, RuntimeState.FAILED),
    RuntimeState.RETRIEVING: (RuntimeState.PROMPT_READY, RuntimeState.FAILED),
    RuntimeState.PROMPT_READY: (RuntimeState.INFERENCING, RuntimeState.FAILED),
    RuntimeState.INFERENCING: (RuntimeState.VALIDATING, RuntimeState.FAILED),
    RuntimeState.VALIDATING: (RuntimeState.COMPLETED, RuntimeState.FAILED),
    RuntimeState.COMPLETED: (RuntimeState.READY, RuntimeState.IDLE, RuntimeState.FAILED),
    RuntimeState.FAILED: (RuntimeState.READY, RuntimeState.IDLE),
}


@dataclass
class StateMachine:
    state: RuntimeState = RuntimeState.IDLE
    history: list[tuple[str, str]] = field(default_factory=list)

    def transition(self, target: RuntimeState) -> RuntimeState:
        if target not in _ALLOWED[self.state]:
            raise contract_error(
                "RTS-001",
                f"illegal runtime transition {self.state.value} -> {target.value}",
                capability="runtime",
                recommended_action="a stage ran out of order; DIC-01 fixes the lifecycle",
            )
        self.history.append((iso_now(), f"{self.state.value}->{target.value}"))
        self.state = target
        return self.state

    def reset(self) -> None:
        self.state = RuntimeState.IDLE
        self.history.clear()
