"""Document pipeline contract (DDPB).

The document pipeline is a *separate* pipeline from the inference pipeline: it
runs at corpus build time, not per request. It therefore has its own immutable
state object, but the same discipline -- fixed stage order, enrichment only,
full traceability. See ADR-007.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any

from ..shared.clock import iso_now
from ..shared.errors import ErrorRecord
from ..shared.telemetry import StageRecord
from .dto import DocumentDescriptor, RawDocument, StructuredDocument


class DocumentStage(str, enum.Enum):
    IMPORT = "import"
    VALIDATION = "validation"
    NORMALIZATION = "normalization"
    STRUCTURE_DETECTION = "structure_detection"
    SEGMENTATION = "segmentation"
    METADATA_EXTRACTION = "metadata_extraction"
    EVIDENCE_PREPARATION = "evidence_preparation"
    CONTEXT_EXPORT = "context_export"


DOCUMENT_STAGE_ORDER: tuple[DocumentStage, ...] = tuple(DocumentStage)


class DocumentState(str, enum.Enum):
    RAW = "raw"
    VALIDATED = "validated"
    NORMALIZED = "normalized"
    STRUCTURED = "structured"
    SEGMENTED = "segmented"
    PREPARED = "prepared"
    EXPORTED = "exported"


DOCUMENT_STATE_ORDER: tuple[DocumentState, ...] = tuple(DocumentState)


@dataclass(frozen=True, slots=True)
class DocumentPipelineState:
    """State of one document travelling through the DDPB."""

    source_path: str
    created_at: str = field(default_factory=iso_now)
    revision: int = 1
    state: DocumentState = DocumentState.RAW

    descriptor: DocumentDescriptor | None = None
    raw: RawDocument | None = None
    normalized_text: str = ""
    structured: StructuredDocument | None = None

    stages: tuple[StageRecord, ...] = ()
    warnings: tuple[str, ...] = ()
    errors: tuple[ErrorRecord, ...] = ()

    def evolve(self, **changes: Any) -> "DocumentPipelineState":
        if "source_path" in changes:
            raise ValueError("source_path is immutable")
        return replace(self, revision=self.revision + 1, **changes)

    def advance(self, state: DocumentState, **changes: Any) -> "DocumentPipelineState":
        """Move to the next state. States are never skipped (DDPB s.15)."""
        current = DOCUMENT_STATE_ORDER.index(self.state)
        target = DOCUMENT_STATE_ORDER.index(state)
        if target != current + 1 and target != current:
            raise ValueError(
                f"illegal document state transition {self.state.value} -> {state.value}"
            )
        return self.evolve(state=state, **changes)

    def with_stage(self, record: StageRecord, **changes: Any) -> "DocumentPipelineState":
        return self.evolve(stages=self.stages + (record,), **changes)

    def with_warning(self, message: str) -> "DocumentPipelineState":
        return self.evolve(warnings=self.warnings + (message,))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "created_at": self.created_at,
            "revision": self.revision,
            "state": self.state.value,
            "descriptor": self.descriptor.to_dict() if self.descriptor else None,
            "structured": self.structured.to_dict() if self.structured else None,
            "stages": [s.to_dict() for s in self.stages],
            "warnings": list(self.warnings),
            "errors": [e.to_dict() for e in self.errors],
        }
