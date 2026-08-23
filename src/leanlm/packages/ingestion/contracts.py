"""CAP-001 public contract."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_DOCUMENT, CapabilitySpec, DocumentCapability,
)
from ...contracts.ddp import DocumentPipelineState, DocumentState
from ...contracts.iec import PipelineStage
from .policies import IngestionPolicy
from .services import DocumentLoader
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-001",
    name="Document Ingestion",
    package="packages/ingestion",
    pipeline=PIPELINE_DOCUMENT,
    stages=("import", "validation", "normalization"),
    requirement_ids=("FR-01",),
    adtc_criteria=("Accuracy",),
    estimated_memory_mb=32.0,
    expected_duration_ms=60.0,
    produces_metrics=METRICS,
)


class DocumentIngestionCapability(DocumentCapability):
    """Import -> validate -> normalize. Produces the normalized text."""

    def __init__(self, policy: IngestionPolicy | None = None) -> None:
        super().__init__(SPEC)
        self.loader = DocumentLoader(policy)

    def run(self, state: DocumentPipelineState) -> DocumentPipelineState:
        descriptor = self.loader.describe(state.source_path)
        raw = self.loader.load(descriptor)
        normalized = self.loader.normalize(raw)
        return state.evolve(
            descriptor=descriptor,
            raw=raw,
            normalized_text=normalized,
            state=DocumentState.NORMALIZED,
        )
