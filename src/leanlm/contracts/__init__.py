"""Internal contracts (ICIB). The only language spoken between capabilities."""

from .capability import (
    Capability, CapabilityRegistry, CapabilitySpec, DocumentCapability,
)
from .ddp import (
    DOCUMENT_STAGE_ORDER, DOCUMENT_STATE_ORDER, DocumentPipelineState,
    DocumentStage, DocumentState,
)
from .dto import (
    ContextBudget, ContextPackage, DocumentDescriptor, EvidenceCandidate, IntentClassification,
    MetricsSnapshot, ModelResponse, OptimizedContext, PromptPackage, RawDocument,
    ResourceObservation, SelectedEvidence, SemanticUnit, StructuredDocument,
    ValidationReport,
)
from .iec import InferenceExecutionContext, PipelineStage, STAGE_ORDER
from .versions import CONTRACT_VERSIONS

__all__ = [
    "Capability", "CapabilityRegistry", "CapabilitySpec", "ContextPackage",
    "DocumentDescriptor", "EvidenceCandidate", "IntentClassification",
    "InferenceExecutionContext", "MetricsSnapshot", "ModelResponse",
    "OptimizedContext", "PipelineStage", "PromptPackage", "RawDocument",
    "ResourceObservation", "STAGE_ORDER", "SelectedEvidence", "SemanticUnit",
    "StructuredDocument", "ValidationReport", "CONTRACT_VERSIONS", "DocumentCapability",
    "DocumentPipelineState", "DocumentStage", "DocumentState",
    "DOCUMENT_STAGE_ORDER", "DOCUMENT_STATE_ORDER", "ContextBudget",
]
