"""Package-level micro-benchmark for CAP-004 (MQ-03)."""
from __future__ import annotations

from ...shared.clock import Stopwatch
from ..context.benchmarks import _corpus
from ..context.services import ContextBudgeter, ContextOptimizer, IntentAnalyzer
from .services import EvidenceSelector, retrieval_precision

QUESTION = "Quel est le delai de remboursement des notes de frais ?"


def run(iterations: int = 5) -> dict[str, float]:
    package = _corpus(400)
    intent = IntentAnalyzer().analyze(QUESTION)
    budget = ContextBudgeter().compute(
        model_context_tokens=4096, max_output_tokens=512, question=QUESTION,
        intent=intent, resources=None,
    )
    optimized = ContextOptimizer().optimize(package, budget)
    selector = EvidenceSelector()
    durations, selected = [], ()
    for _ in range(iterations):
        watch = Stopwatch()
        selected, _candidates = selector.select(QUESTION, optimized, intent)
        durations.append(watch.stop())
    return {
        "capability_id": "CAP-004",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "passages": len(selected),
        "evidence_tokens": sum(e.token_estimate for e in selected),
        "precision": retrieval_precision(selected, QUESTION),
    }
