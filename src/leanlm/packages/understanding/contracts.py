"""CAP-002 public contract."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_DOCUMENT, CapabilitySpec, DocumentCapability,
)
from ...contracts.ddp import DocumentPipelineState, DocumentState
from ...contracts.iec import PipelineStage
from ...shared.errors import input_error
from ...shared.text import TokenCounter
from .policies import SegmentationPolicy
from .services import build_structured_document
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-002",
    name="Document Understanding",
    package="packages/understanding",
    pipeline=PIPELINE_DOCUMENT,
    stages=("structure_detection", "segmentation", "metadata_extraction"),
    requirement_ids=("FR-02",),
    adtc_criteria=("Accuracy",),
    estimated_memory_mb=48.0,
    expected_duration_ms=120.0,
    produces_metrics=METRICS,
)


class DocumentUnderstandingCapability(DocumentCapability):
    """Structure detection, segmentation, metadata extraction."""

    def __init__(self, policy: SegmentationPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        super().__init__(SPEC)
        self.policy = policy or SegmentationPolicy()
        self.counter = counter

    def run(self, state: DocumentPipelineState) -> DocumentPipelineState:
        if state.descriptor is None or not state.normalized_text:
            raise input_error(
                "UND-001", "understanding requires a normalized document",
                capability="understanding", stage="structure_detection",
                recommended_action="run CAP-001 first",
            )
        structured = build_structured_document(
            state.normalized_text,
            filename=state.descriptor.filename,
            descriptor_id=state.descriptor.artifact_id,
            policy=self.policy,
            counter=self.counter,
        )
        if not structured.units:
            raise input_error(
                "UND-002", f"no semantic unit produced for {state.descriptor.filename}",
                capability="understanding", stage="segmentation",
                recommended_action="check the document content",
            )
        return state.evolve(structured=structured, state=DocumentState.SEGMENTED)
