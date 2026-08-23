"""Package-level micro-benchmark for CAP-008 (MQ-03)."""
from __future__ import annotations

import tempfile
from pathlib import Path

from ...shared.clock import Stopwatch
from .contracts import SubmissionPackagingCapability
from .models import SubmissionRequest


def run(iterations: int = 3) -> dict[str, float]:
    capability = SubmissionPackagingCapability()
    durations, result = [], None
    with tempfile.TemporaryDirectory() as tmp:
        for index in range(iterations):
            request = SubmissionRequest(
                output_dir=str(Path(tmp) / f"pkg{index}"),
                model_manifest={"id": "bench-model", "format": "gguf",
                                "sha256": "0" * 64, "source_url": "https://example/model.gguf"},
                runtime_profile={"id": "competition"},
                benchmark_summary={"tokens_per_second": 12.0, "is_simulated": False},
                profiler_report={"accuracy": 0.0},
                git_commit="benchmark",
            )
            watch = Stopwatch()
            result = capability.build(request, enforce=False)
            durations.append(watch.stop())
    return {
        "capability_id": "CAP-008",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "files": len(result.files) if result else 0,
        "compliance_passed": result.compliance.passed if result else False,
    }
