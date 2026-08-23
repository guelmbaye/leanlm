"""Package-level micro-benchmark for CAP-003 (MQ-03)."""
from __future__ import annotations

from ...contracts.dto import ContextPackage, SemanticUnit
from ...shared.clock import Stopwatch
from ...shared.text import DEFAULT_TOKEN_COUNTER
from .services import ContextBudgeter, ContextOptimizer, IntentAnalyzer


def _corpus(size: int = 400) -> ContextPackage:
    units = []
    for index in range(size):
        text = (f"Section {index % 20}: les notes de frais sont remboursees sous "
                f"{20 + index % 10} jours ouvres apres validation hierarchique.")
        if index % 7 == 0:  # deliberate duplicates
            text = "Les notes de frais sont remboursees sous 30 jours ouvres."
        unit = SemanticUnit(
            meta=SemanticUnit.build_meta(identity=("bench", str(index), text[:60])),
            unit_id=f"bench#{index:04d}", document_id="bench", document_name="bench.md",
            text=text, order=index, section_path=(f"S{index % 20}",),
            token_estimate=DEFAULT_TOKEN_COUNTER.count(text),
        ).sealed()
        units.append(unit)
    return ContextPackage(
        meta=ContextPackage.build_meta(identity=("bench", str(size))),
        units=tuple(units), document_ids=("bench",), total_units=size,
        total_tokens=sum(u.token_estimate for u in units), corpus_checksum="bench",
    ).sealed()


def run(iterations: int = 5) -> dict[str, float]:
    package = _corpus()
    intent = IntentAnalyzer().analyze("Quel est le delai de remboursement ?")
    budget = ContextBudgeter().compute(
        model_context_tokens=4096, max_output_tokens=512,
        question="Quel est le delai de remboursement ?", intent=intent, resources=None,
    )
    optimizer = ContextOptimizer()
    durations, ratio, kept = [], 0.0, 0
    for _ in range(iterations):
        watch = Stopwatch()
        optimized = optimizer.optimize(package, budget)
        durations.append(watch.stop())
        ratio, kept = optimized.compression_ratio, len(optimized.units)
    return {
        "capability_id": "CAP-003",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "input_units": package.total_units,
        "kept_units": kept,
        "compression_ratio": ratio,
    }
