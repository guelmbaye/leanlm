"""Canonical DTOs (DMSB s.4, ICIB s.4).

All immutable, all deterministic, all derived from CanonicalObject. These are
the only objects allowed to cross a capability boundary.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from ..shared.com import CanonicalObject
from ..shared.serialization import to_plain

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


class DocumentFormat(str, enum.Enum):
    TXT = "txt"
    MARKDOWN = "md"
    PDF = "pdf"


@dataclass(frozen=True)
class DocumentDescriptor(CanonicalObject):
    """A document as found on disk. The original file is never modified."""

    SCHEMA_VERSION = "1.0.0"

    path: str = ""
    filename: str = ""
    doc_format: DocumentFormat = DocumentFormat.TXT
    size_bytes: int = 0
    file_checksum: str = ""
    modified_at: str = ""

    def identity(self) -> tuple[str, ...]:
        return (self.file_checksum, self.filename)


@dataclass(frozen=True)
class RawDocument(CanonicalObject):
    """Extracted, not yet normalized text."""

    SCHEMA_VERSION = "1.0.0"

    descriptor_id: str = ""
    filename: str = ""
    text: str = ""
    extraction_method: str = ""
    page_count: int = 0

    def identity(self) -> tuple[str, ...]:
        return (self.descriptor_id, self.extraction_method)


# ---------------------------------------------------------------------------
# Understanding
# ---------------------------------------------------------------------------


class UnitKind(str, enum.Enum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"


@dataclass(frozen=True)
class SemanticUnit(CanonicalObject):
    """A coherent fragment of a document, traceable to its source (FR-04)."""

    SCHEMA_VERSION = "1.0.0"

    unit_id: str = ""
    document_id: str = ""
    document_name: str = ""
    kind: UnitKind = UnitKind.PARAGRAPH
    text: str = ""
    section_path: tuple[str, ...] = ()
    order: int = 0
    char_start: int = 0
    char_end: int = 0
    token_estimate: int = 0
    heading_level: int = 0

    def identity(self) -> tuple[str, ...]:
        return (self.document_id, str(self.order), self.text[:120])

    @property
    def citation(self) -> str:
        section = " > ".join(self.section_path) if self.section_path else "-"
        return f"{self.document_name} :: {section}"


@dataclass(frozen=True)
class StructuredDocument(CanonicalObject):
    """Normalized document plus its detected logical structure."""

    SCHEMA_VERSION = "1.0.0"

    document_id: str = ""
    filename: str = ""
    title: str = ""
    units: tuple[SemanticUnit, ...] = ()
    section_titles: tuple[str, ...] = ()
    token_estimate: int = 0
    language_hint: str = "und"

    def identity(self) -> tuple[str, ...]:
        return (self.document_id, str(len(self.units)))


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


class IntentType(str, enum.Enum):
    FACTUAL = "factual"
    COMPARISON = "comparison"
    SUMMARY = "summary"
    EXTRACTION = "extraction"
    EXPLANATION = "explanation"
    DRAFTING = "drafting"


@dataclass(frozen=True)
class IntentClassification(CanonicalObject):
    SCHEMA_VERSION = "1.0.0"

    intent: IntentType = IntentType.FACTUAL
    confidence: str = "medium"
    signals: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()

    def identity(self) -> tuple[str, ...]:
        return (self.intent.value, ",".join(self.keywords))


@dataclass(frozen=True)
class ResourceObservation(CanonicalObject):
    """What the Resource Manager saw just before a decision was taken."""

    SCHEMA_VERSION = "1.0.0"

    total_ram_mb: float = 0.0
    available_ram_mb: float = 0.0
    process_rss_mb: float = 0.0
    cpu_percent: float = 0.0
    cpu_count: int = 1
    temperature_c: float | None = None
    thermal_source: str = "unavailable"
    disk_free_mb: float = 0.0
    sampled_at: str = ""

    def identity(self) -> tuple[str, ...]:
        return (self.sampled_at, f"{self.available_ram_mb:.0f}")


@dataclass(frozen=True)
class ContextPackage(CanonicalObject):
    """Interface between the document pipeline (DDPB) and the inference pipeline."""

    SCHEMA_VERSION = "1.0.0"

    units: tuple[SemanticUnit, ...] = ()
    document_ids: tuple[str, ...] = ()
    total_units: int = 0
    total_tokens: int = 0
    corpus_checksum: str = ""

    def identity(self) -> tuple[str, ...]:
        return (self.corpus_checksum, str(self.total_units))


@dataclass(frozen=True)
class ContextBudget(CanonicalObject):
    """A *target*, not a fixed size (IIB s.7)."""

    SCHEMA_VERSION = "1.0.0"

    max_context_tokens: int = 0
    evidence_tokens: int = 0
    reserved_output_tokens: int = 0
    reserved_prompt_overhead: int = 0
    max_passages: int = 0
    limiting_factor: str = "model_context"
    factors: dict[str, float] = field(default_factory=dict)
    token_counter: str = "heuristic-v1"

    def identity(self) -> tuple[str, ...]:
        return (str(self.max_context_tokens), self.limiting_factor)


@dataclass(frozen=True)
class OptimizedContext(CanonicalObject):
    """Compacted candidate set plus the compression evidence."""

    SCHEMA_VERSION = "1.0.0"

    units: tuple[SemanticUnit, ...] = ()
    budget: ContextBudget | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    dropped_units: int = 0
    merged_units: int = 0
    duplicate_units: int = 0
    operations: tuple[str, ...] = ()

    def identity(self) -> tuple[str, ...]:
        return (str(self.output_tokens), str(len(self.units)))

    @property
    def compression_ratio(self) -> float:
        """Context Compression Ratio (IIB s.15). 1.0 means nothing was saved."""
        if self.input_tokens <= 0:
            return 0.0
        return round(1.0 - (self.output_tokens / self.input_tokens), 4)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceCandidate(CanonicalObject):
    SCHEMA_VERSION = "1.0.0"

    unit: SemanticUnit | None = None
    score: float = 0.0
    lexical_score: float = 0.0
    structural_boost: float = 0.0
    diversity_penalty: float = 0.0
    matched_terms: tuple[str, ...] = ()

    def identity(self) -> tuple[str, ...]:
        return (self.unit.unit_id if self.unit else "", f"{self.score:.6f}")


@dataclass(frozen=True)
class SelectedEvidence(CanonicalObject):
    """An evidence passage that made it into the prompt, with its label."""

    SCHEMA_VERSION = "1.0.0"

    label: str = ""
    unit_id: str = ""
    document_name: str = ""
    section_path: tuple[str, ...] = ()
    text: str = ""
    score: float = 0.0
    token_estimate: int = 0

    def identity(self) -> tuple[str, ...]:
        return (self.unit_id, self.label)

    @property
    def citation(self) -> str:
        section = " > ".join(self.section_path) if self.section_path else "-"
        return f"{self.document_name} :: {section}"


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptPackage(CanonicalObject):
    SCHEMA_VERSION = "1.0.0"

    system_prompt: str = ""
    user_prompt: str = ""
    rendered: str = ""
    template_id: str = ""
    template_version: str = ""
    prompt_tokens: int = 0
    context_tokens: int = 0
    evidence_labels: tuple[str, ...] = ()

    def identity(self) -> tuple[str, ...]:
        return (self.template_id, self.template_version, self.rendered[:200])


@dataclass(frozen=True)
class ModelResponse(CanonicalObject):
    """Raw model output. Never edited afterwards -- validation only annotates."""

    SCHEMA_VERSION = "1.0.0"

    text: str = ""
    backend: str = ""
    model_id: str = ""
    model_checksum: str = ""
    generated_tokens: int = 0
    prompt_tokens: int = 0
    first_token_latency_ms: float = 0.0
    inference_ms: float = 0.0
    tokens_per_second: float = 0.0
    stop_reason: str = "eos"
    is_simulated: bool = False

    def identity(self) -> tuple[str, ...]:
        return (self.model_id, self.text[:200])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class ConfidenceLevel(str, enum.Enum):
    """Qualitative, observable, never a fabricated probability (IIB s.13)."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class ValidationReport(CanonicalObject):
    SCHEMA_VERSION = "1.0.0"

    passed: bool = True
    confidence: ConfidenceLevel = ConfidenceLevel.MEDIUM
    grounding_rate: float = 0.0
    evidence_coverage: float = 0.0
    question_coverage: float = 0.0
    cited_labels: tuple[str, ...] = ()
    unsupported_sentences: tuple[str, ...] = ()
    uncertainty_declared: bool = False
    checks: dict[str, bool] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def identity(self) -> tuple[str, ...]:
        return (str(self.passed), f"{self.grounding_rate:.4f}")


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricsSnapshot(CanonicalObject):
    """Tier 2 runtime metrics + Tier 3 intelligence metrics (EBPB s.3)."""

    SCHEMA_VERSION = "1.0.0"

    session_id: str = ""
    request_id: str = ""
    profile_id: str = ""
    model_id: str = ""
    # Tier 2 -- runtime
    first_token_latency_ms: float = 0.0
    inference_ms: float = 0.0
    total_ms: float = 0.0
    tokens_per_second: float = 0.0
    generated_tokens: int = 0
    prompt_tokens: int = 0
    context_tokens: int = 0
    peak_rss_mb: float = 0.0
    available_ram_mb: float = 0.0
    cpu_percent: float = 0.0
    temperature_c: float | None = None
    # Tier 3 -- intelligence
    context_compression_ratio: float = 0.0   # corpus tokens -> prompt context
    dedup_compression_ratio: float = 0.0     # compaction step alone
    evidence_coverage: float = 0.0
    retrieval_precision: float = 0.0
    prompt_efficiency: float = 0.0
    response_grounding_rate: float = 0.0
    # traceability
    stage_durations: dict[str, float] = field(default_factory=dict)
    is_simulated: bool = False

    def identity(self) -> tuple[str, ...]:
        return (self.request_id, f"{self.total_ms:.3f}")

    def to_flat(self) -> dict[str, Any]:
        return {k: v for k, v in to_plain(self).items() if not isinstance(v, dict)}
