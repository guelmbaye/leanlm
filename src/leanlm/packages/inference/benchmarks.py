"""Package-level micro-benchmark for CAP-005 (MQ-03).

Measures prompt assembly only -- model throughput belongs to the benchmark
harness, which controls the profile and the scenario.
"""
from __future__ import annotations

from ...contracts.dto import SelectedEvidence
from ...shared.clock import Stopwatch
from .services import PromptBuilder, prompt_efficiency


def _evidence(count: int = 6) -> tuple[SelectedEvidence, ...]:
    items = []
    for index in range(1, count + 1):
        text = (f"Article {index}. Les notes de frais sont remboursees sous "
                f"{25 + index} jours ouvres apres validation du responsable.")
        items.append(SelectedEvidence(
            meta=SelectedEvidence.build_meta(identity=(f"u{index}", f"S{index}")),
            label=f"S{index}", unit_id=f"u{index}", document_name="rh.md",
            section_path=("Politique", "Delais"), text=text, token_estimate=28,
        ).sealed())
    return tuple(items)


def run(iterations: int = 20) -> dict[str, float]:
    builder = PromptBuilder()
    evidence = _evidence()
    question = "Quel est le delai de remboursement des notes de frais ?"
    durations, package = [], None
    for _ in range(iterations):
        watch = Stopwatch()
        package = builder.build(question, evidence)
        durations.append(watch.stop())
    return {
        "capability_id": "CAP-005",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "prompt_tokens": package.prompt_tokens if package else 0,
        "prompt_efficiency": prompt_efficiency(package) if package else 0.0,
    }
