"""Contract versions (ICIB s.12).

Every DTO is versioned independently. A breaking change requires a new version
here, an ADR, and an entry in the Engineering Decision Book.
"""
from __future__ import annotations

CONTRACT_VERSIONS: dict[str, str] = {
    "RawDocument": "1.0.0",
    "DocumentDescriptor": "1.0.0",
    "StructuredDocument": "1.0.0",
    "SemanticUnit": "1.0.0",
    "ContextPackage": "1.0.0",
    "OptimizedContext": "1.0.0",
    "EvidenceCandidate": "1.0.0",
    "SelectedEvidence": "1.0.0",
    "PromptPackage": "1.0.0",
    "ModelResponse": "1.0.0",
    "ValidationReport": "1.0.0",
    "MetricsSnapshot": "1.0.0",
    "InferenceExecutionContext": "1.0.0",
    "BenchmarkEvidence": "1.0.0",
    "RuntimeProfile": "1.0.0",
}

CAPABILITY_CONTRACT_VERSION = "1.0.0"
