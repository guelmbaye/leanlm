"""Inference Execution Context (IPB s.4).

The IEC is the single object that travels through the pipeline. Every capability
receives a version of it and returns an *enriched* version -- never a mutated
one. Immutable-by-design is what makes DIC-01 (fixed lifecycle) and DIC-04
(traceability) checkable rather than merely claimed: the stage history is part
of the object, and a stage can only ever append to it.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

from ..shared.clock import iso_now
from ..shared.errors import ErrorRecord
from ..shared.telemetry import StageRecord
from .dto import (
    ContextPackage, IntentClassification, MetricsSnapshot, ModelResponse,
    OptimizedContext, PromptPackage, ResourceObservation, SelectedEvidence,
    ValidationReport,
)


class PipelineStage(str, enum.Enum):
    """The ten phases of the Deterministic Inference Contract.

    Reconciles the nine Runtime Blueprint phases with the Inference Pipeline
    Blueprint ordering (evidence retrieval promoted to its own phase). See
    ADR-006.
    """

    SESSION_CREATION = "session_creation"
    RESOURCE_ASSESSMENT = "resource_assessment"
    DOCUMENT_DISCOVERY = "document_discovery"
    CONTEXT_OPTIMIZATION = "context_optimization"
    EVIDENCE_RETRIEVAL = "evidence_retrieval"
    PROMPT_ASSEMBLY = "prompt_assembly"
    MODEL_INFERENCE = "model_inference"
    RESPONSE_VALIDATION = "response_validation"
    METRICS_FINALIZATION = "metrics_finalization"
    SESSION_CLEANUP = "session_cleanup"


STAGE_ORDER: tuple[PipelineStage, ...] = tuple(PipelineStage)


@dataclass(frozen=True, slots=True)
class InferenceExecutionContext:
    """Complete state of one inference. Enriched, never rewritten."""

    session_id: str
    request_id: str
    question: str
    profile_id: str
    created_at: str = field(default_factory=iso_now)
    revision: int = 1

    resources: ResourceObservation | None = None
    intent: IntentClassification | None = None
    context_package: ContextPackage | None = None
    optimized_context: OptimizedContext | None = None
    evidence: tuple[SelectedEvidence, ...] = ()
    prompt: PromptPackage | None = None
    response: ModelResponse | None = None
    validation: ValidationReport | None = None
    metrics: MetricsSnapshot | None = None

    stages: tuple[StageRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()
    trace: dict[str, Any] = field(default_factory=dict)

    # -- evolution ----------------------------------------------------------
    def evolve(self, **changes: Any) -> "InferenceExecutionContext":
        """Return the next revision. Existing validated data is never dropped."""
        protected = {"session_id", "request_id", "question", "created_at"}
        illegal = protected & set(changes)
        if illegal:
            raise ValueError(f"IEC identity fields are immutable: {sorted(illegal)}")
        return replace(self, revision=self.revision + 1, **changes)

    def with_stage(self, record: StageRecord, **changes: Any) -> "InferenceExecutionContext":
        return self.evolve(stages=self.stages + (record,), **changes)

    def with_warning(self, message: str) -> "InferenceExecutionContext":
        return self.evolve(warnings=self.warnings + (message,))

    def with_error(self, error: ErrorRecord) -> "InferenceExecutionContext":
        return self.evolve(errors=self.errors + (error,))

    # -- inspection ---------------------------------------------------------
    @property
    def completed_stages(self) -> tuple[str, ...]:
        return tuple(record.stage for record in self.stages)

    @property
    def failed(self) -> bool:
        return any(record.status == "failed" for record in self.stages)

    def stage_durations(self) -> dict[str, float]:
        return {record.stage: record.duration_ms for record in self.stages}

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        """Serialize the IEC.

        ``include_text=False`` strips user content so an IEC can be archived
        for diagnostics without persisting document text (TL-03).
        """
        def maybe(obj):
            return obj.to_dict() if obj is not None else None

        payload: dict[str, Any] = {
            "session_id": self.session_id,
            "request_id": self.request_id,
            "profile_id": self.profile_id,
            "created_at": self.created_at,
            "revision": self.revision,
            "question": self.question if include_text else "<redacted>",
            "resources": maybe(self.resources),
            "intent": maybe(self.intent),
            "optimized_context": maybe(self.optimized_context) if include_text else None,
            "evidence": [
                e.to_dict() if include_text else
                {"label": e.label, "unit_id": e.unit_id, "score": e.score}
                for e in self.evidence
            ],
            "prompt": maybe(self.prompt) if include_text else None,
            "response": maybe(self.response) if include_text else None,
            "validation": maybe(self.validation),
            "metrics": maybe(self.metrics),
            "stages": [s.to_dict() for s in self.stages],
            "warnings": list(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
            "trace": dict(self.trace),
        }
        return payload
