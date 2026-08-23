"""Package-level micro-benchmark for CAP-002 (MQ-03)."""
from __future__ import annotations

from ...shared.clock import Stopwatch
from ...shared.text import normalize_text
from .services import build_structured_document

SAMPLE = """# Politique de note de frais

## 1. Perimetre
Cette politique s'applique a l'ensemble des collaborateurs.

## 2. Delais
Les notes de frais sont remboursees sous 30 jours ouvres apres validation.

- Repas: plafond 25 EUR
- Transport: tarif reel

## 3. Exceptions
Toute exception doit etre validee par le directeur financier.
""" * 40


def run(iterations: int = 5) -> dict[str, float]:
    text = normalize_text(SAMPLE)
    durations, units = [], 0
    for _ in range(iterations):
        watch = Stopwatch()
        doc = build_structured_document(text, filename="policy.md", descriptor_id="bench")
        durations.append(watch.stop())
        units = len(doc.units)
    return {
        "capability_id": "CAP-002",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "units": units,
    }
