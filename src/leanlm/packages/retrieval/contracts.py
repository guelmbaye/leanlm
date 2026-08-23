"""CAP-004 public contract."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_INFERENCE, Capability, CapabilitySpec,
)
from ...contracts.iec import InferenceExecutionContext, PipelineStage
from ...shared.errors import contract_error
from ...shared.text import TokenCounter
from .policies import RetrievalPolicy
from .services import EvidenceSelector, retrieval_precision
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-004",
    name="Evidence Retrieval",
    package="packages/retrieval",
    pipeline=PIPELINE_INFERENCE,
    stages=(PipelineStage.EVIDENCE_RETRIEVAL.value,),
    requirement_ids=("FR-04",),
    adtc_criteria=("Accuracy", "Throughput"),
    estimated_memory_mb=16.0,
    expected_duration_ms=20.0,
    produces_metrics=METRICS,
)


class EvidenceRetrievalCapability(Capability):
    """Selects the passages that earn their place in the prompt."""

    def __init__(self, policy: RetrievalPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        super().__init__(SPEC)
        self.selector = EvidenceSelector(policy, counter)

    def check_preconditions(self, iec: InferenceExecutionContext,
                            stage: PipelineStage) -> None:
        if iec.optimized_context is None:
            raise contract_error(
                "RET-001", "retrieval requires an optimized context",
                capability=self.spec.name, stage=self.spec.stages[0],
                recommended_action="run CAP-003 first",
            )

    def run(self, iec: InferenceExecutionContext,
            stage: PipelineStage) -> InferenceExecutionContext:
        selected, candidates = self.selector.select(
            iec.question, iec.optimized_context, iec.intent
        )
        trace = dict(iec.trace)
        trace["retrieval"] = {
            "candidates": len(candidates),
            "selected": len(selected),
            "evidence_tokens": sum(e.token_estimate for e in selected),
            "documents": sorted({e.document_name for e in selected}),
            "precision": retrieval_precision(selected, iec.question),
            "scores": [
                {"label": c.unit.unit_id if c.unit else "", "score": c.score,
                 "lexical": c.lexical_score, "boost": c.structural_boost,
                 "diversity_penalty": c.diversity_penalty}
                for c in candidates
            ],
        }
        result = iec.evolve(evidence=selected, trace=trace)
        if not selected:
            # P3 evidence driven: an empty evidence set is a first class outcome,
            # not a silent fallback to the model's own memory.
            result = result.with_warning(
                "no passage in the corpus matched the question; "
                "the answer will state that explicitly"
            )
        return result

    def check_postconditions(self, iec: InferenceExecutionContext,
                             stage: PipelineStage) -> None:
        budget = iec.optimized_context.budget if iec.optimized_context else None
        if budget is None:
            return
        used = sum(e.token_estimate for e in iec.evidence)
        if len(iec.evidence) > 1 and used > budget.evidence_tokens:
            raise contract_error(
                "RET-002",
                f"evidence exceeds the context budget ({used} > {budget.evidence_tokens})",
                capability=self.spec.name,
                recommended_action="IP-01: never send more context than budgeted",
            )
