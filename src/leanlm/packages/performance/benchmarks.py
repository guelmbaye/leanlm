"""Package-level micro-benchmark for CAP-007 (MQ-03)."""
from __future__ import annotations

from ...shared.clock import Stopwatch
from .services import ResourceSampler


def run(iterations: int = 20) -> dict[str, float]:
    sampler = ResourceSampler()
    durations, observation = [], None
    for _ in range(iterations):
        watch = Stopwatch()
        observation = sampler.sample()
        durations.append(watch.stop())
    return {
        "capability_id": "CAP-007",
        "iterations": iterations,
        "mean_ms": round(sum(durations) / len(durations), 3),
        "min_ms": round(min(durations), 3),
        "max_ms": round(max(durations), 3),
        "available_ram_mb": observation.available_ram_mb if observation else 0.0,
        "thermal_source": observation.thermal_source if observation else "n/a",
    }
