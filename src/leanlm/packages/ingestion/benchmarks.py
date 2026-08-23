"""Package-level micro-benchmark for CAP-001 (MQ-03)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from ...shared.clock import Stopwatch
from .contracts import DocumentIngestionCapability
from ...contracts.ddp import DocumentPipelineState

SAMPLE = ("# Politique interne\n\nLes notes de frais sont remboursees sous 30 jours.\n" * 200)


def run(iterations: int = 5) -> dict[str, float]:
    capability = DocumentIngestionCapability()
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "sample.md"
        path.write_text(SAMPLE, encoding="utf-8")
        durations = []
        for _ in range(iterations):
            watch = Stopwatch()
            capability.process(DocumentPipelineState(source_path=str(path)))
            durations.append(watch.stop())
    return {
        "capability_id": "CAP-001",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "chars": len(SAMPLE),
    }
