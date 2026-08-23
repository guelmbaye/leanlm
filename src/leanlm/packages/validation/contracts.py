"""CAP-006 public contract."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_INFERENCE, Capability, CapabilitySpec,
)
from ...contracts.iec import InferenceExecutionContext, PipelineStage
from ...shared.errors import contract_error
from ...shared.events import EventBus, EventName
from .policies import ValidationPolicy
from .services import ResponseValidator
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-006",
    name="Response Validation",
    package="packages/validation",
    pipeline=PIPELINE_INFERENCE,
    stages=(PipelineStage.RESPONSE_VALIDATION.value,),
    requirement_ids=("FR-06",),
    adtc_criteria=("Accuracy",),
    estimated_memory_mb=12.0,
    expected_duration_ms=15.0,
    produces_metrics=METRICS,
)


class ResponseValidationCapability(Capability):
    """Annotates the answer. Never edits it."""

    def __init__(self, policy: ValidationPolicy | None = None,
                 bus: EventBus | None = None) -> None:
        super().__init__(SPEC)
        self.validator = ResponseValidator(policy)
        self.bus = bus

    def check_preconditions(self, iec: InferenceExecutionContext,
                            stage: PipelineStage) -> None:
        if iec.response is None:
            raise contract_error(
                "VAL-001", "validation requires a model response",
                capability=self.spec.name, stage=stage.value,
                recommended_action="run the model_inference stage first",
            )

    def run(self, iec: InferenceExecutionContext,
            stage: PipelineStage) -> InferenceExecutionContext:
        report = self.validator.validate(iec.response, iec.evidence, iec.question)
        trace = dict(iec.trace)
        trace["validation"] = {
            "passed": report.passed,
            "confidence": report.confidence.value,
            "grounding_rate": report.grounding_rate,
            "evidence_coverage": report.evidence_coverage,
            "cited": list(report.cited_labels),
            "checks": dict(report.checks),
        }
        result = iec.evolve(validation=report, trace=trace)
        for warning in report.warnings:
            result = result.with_warning(warning)
        if self.bus:
            self.bus.emit(EventName.VALIDATION_COMPLETED, iec.session_id,
                          capability="validation", passed=report.passed,
                          confidence=report.confidence.value)
        return result
