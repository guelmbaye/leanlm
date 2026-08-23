"""Package-level micro-benchmark for CAP-006 (MQ-03)."""
from __future__ import annotations

from ...contracts.dto import ModelResponse, SelectedEvidence
from ...shared.clock import Stopwatch
from .services import ResponseValidator


def run(iterations: int = 20) -> dict[str, float]:
    evidence = tuple(
        SelectedEvidence(
            meta=SelectedEvidence.build_meta(identity=(f"u{i}", f"S{i}")),
            label=f"S{i}", unit_id=f"u{i}", document_name="rh.md",
            section_path=("Politique",),
            text=("Les notes de frais sont remboursees sous 30 jours ouvres "
                  "apres validation du responsable hierarchique."),
            token_estimate=26,
        ).sealed() for i in range(1, 6)
    )
    response = ModelResponse(
        meta=ModelResponse.build_meta(identity=("m", "a")),
        text=("Les notes de frais sont remboursees sous 30 jours ouvres apres "
              "validation du responsable hierarchique. [S1]"),
    ).sealed()
    validator = ResponseValidator()
    durations, report = [], None
    for _ in range(iterations):
        watch = Stopwatch()
        report = validator.validate(response, evidence, "Quel est le delai de remboursement ?")
        durations.append(watch.stop())
    return {
        "capability_id": "CAP-006",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "grounding_rate": report.grounding_rate if report else 0.0,
        "confidence": report.confidence.value if report else "n/a",
    }
