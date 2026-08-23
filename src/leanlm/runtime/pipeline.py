"""Inference pipeline orchestrator (IPB).

The orchestrator owns the stage sequence and nothing else. It does not know what
a context budget is or how BM25 scores a passage; it knows the order, the
telemetry, the state machine and the error contract. That is what keeps the DIC
verifiable: the order lives in exactly one place.
"""
from __future__ import annotations

from typing import Callable

from ..contracts.capability import CapabilityRegistry
from ..contracts.iec import STAGE_ORDER, InferenceExecutionContext, PipelineStage
from ..shared.errors import LeanLMError
from ..shared.events import EventBus, EventName
from ..shared.telemetry import Telemetry
from .state_machine import RuntimeState, StateMachine

# Stages the runtime performs itself: they belong to no business capability.
_RUNTIME_STAGES = {
    PipelineStage.SESSION_CREATION,
    PipelineStage.RESOURCE_ASSESSMENT,
    PipelineStage.DOCUMENT_DISCOVERY,
    PipelineStage.SESSION_CLEANUP,
}

_STATE_BY_STAGE = {
    PipelineStage.CONTEXT_OPTIMIZATION: RuntimeState.OPTIMIZING,
    PipelineStage.EVIDENCE_RETRIEVAL: RuntimeState.RETRIEVING,
    PipelineStage.PROMPT_ASSEMBLY: RuntimeState.PROMPT_READY,
    PipelineStage.MODEL_INFERENCE: RuntimeState.INFERENCING,
    PipelineStage.RESPONSE_VALIDATION: RuntimeState.VALIDATING,
}

RuntimeStageHandler = Callable[[InferenceExecutionContext, PipelineStage],
                               InferenceExecutionContext]


class InferencePipeline:
    def __init__(self, registry: CapabilityRegistry, bus: EventBus,
                 runtime_handlers: dict[PipelineStage, RuntimeStageHandler],
                 *, resource_probe=None) -> None:
        self.registry = registry
        self.bus = bus
        self.runtime_handlers = runtime_handlers
        self.resource_probe = resource_probe

    def run(self, iec: InferenceExecutionContext, machine: StateMachine,
            *, stop_after: PipelineStage | None = None
            ) -> InferenceExecutionContext:
        telemetry = Telemetry(self.bus, iec.session_id, self.resource_probe)
        for stage in STAGE_ORDER:
            handler = self._handler_for(stage)
            if handler is None:
                continue
            capability_name = self._capability_name(stage)
            target_state = _STATE_BY_STAGE.get(stage)
            if target_state is not None:
                machine.transition(target_state)
            try:
                with telemetry.span(stage.value, capability_name) as record:
                    iec = handler(iec, stage)
                    record.fields.update(self._stage_fields(iec, stage))
            except LeanLMError as error:
                machine.transition(RuntimeState.FAILED)
                iec = iec.with_stage(telemetry.records[-1]).with_error(error.record)
                iec = self._finalize_on_failure(iec, telemetry)
                self.bus.emit(EventName.CAPABILITY_FAILED, iec.session_id,
                              capability=capability_name, stage=stage.value,
                              code=error.record.code)
                return iec
            iec = iec.with_stage(telemetry.records[-1])
            if stop_after is not None and stage is stop_after:
                # Deliberate early exit: the caller wants the artifacts built so
                # far, not an answer. Used to hand someone the exact prompt a
                # backend would receive.
                return iec
        machine.transition(RuntimeState.COMPLETED)
        self.bus.emit(EventName.PIPELINE_COMPLETED, iec.session_id,
                      stages=len(iec.stages), total_ms=telemetry.total_ms())
        return iec

    # -- internals ----------------------------------------------------------
    def _handler_for(self, stage: PipelineStage) -> RuntimeStageHandler | None:
        if stage in _RUNTIME_STAGES:
            return self.runtime_handlers.get(stage)
        capability = self.registry.for_stage(stage)
        if capability is None:
            return None
        return lambda iec, s: capability.execute(iec, s)

    def _capability_name(self, stage: PipelineStage) -> str:
        if stage in _RUNTIME_STAGES:
            return "runtime"
        capability = self.registry.for_stage(stage)
        return capability.spec.name.lower().replace(" ", "_") if capability else "runtime"

    @staticmethod
    def _stage_fields(iec: InferenceExecutionContext, stage: PipelineStage) -> dict:
        if stage is PipelineStage.DOCUMENT_DISCOVERY and iec.context_package:
            return {"units": iec.context_package.total_units,
                    "tokens": iec.context_package.total_tokens}
        if stage is PipelineStage.CONTEXT_OPTIMIZATION and iec.optimized_context:
            return {"kept_units": len(iec.optimized_context.units),
                    "compression_ratio": iec.optimized_context.compression_ratio,
                    "budget_tokens": iec.optimized_context.budget.evidence_tokens
                    if iec.optimized_context.budget else 0}
        if stage is PipelineStage.EVIDENCE_RETRIEVAL:
            return {"passages": len(iec.evidence)}
        if stage is PipelineStage.PROMPT_ASSEMBLY and iec.prompt:
            return {"prompt_tokens": iec.prompt.prompt_tokens}
        if stage is PipelineStage.MODEL_INFERENCE and iec.response:
            return {"generated_tokens": iec.response.generated_tokens,
                    "tokens_per_second": iec.response.tokens_per_second}
        if stage is PipelineStage.RESPONSE_VALIDATION and iec.validation:
            return {"passed": iec.validation.passed,
                    "grounding_rate": iec.validation.grounding_rate}
        return {}

    def _finalize_on_failure(self, iec: InferenceExecutionContext,
                             telemetry: Telemetry) -> InferenceExecutionContext:
        """A failed run still produces metrics -- an incident is a measurement."""
        capability = self.registry.for_stage(PipelineStage.METRICS_FINALIZATION)
        if capability is None:
            return iec
        try:
            return capability.execute(iec, PipelineStage.METRICS_FINALIZATION)
        except Exception:
            return iec
