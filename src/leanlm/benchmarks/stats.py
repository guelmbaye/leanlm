"""Statistics for benchmark campaigns (BHB s.13).

A single run is an anecdote. Every reported number carries mean, min, max,
standard deviation, p95 and the sample size, so a reader can tell a real
improvement from noise.

Deliberately built on the standard library: importing numpy to average five
floats would add a hard dependency to a project whose entire claim is that it
runs on a machine where nothing else is installed.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass(frozen=True, slots=True)
class Aggregate:
    metric: str
    samples: int
    mean: float
    minimum: float
    maximum: float
    stdev: float
    median: float
    p95: float = 0.0

    @property
    def coefficient_of_variation(self) -> float:
        """Relative spread. Above ~0.15 the measurement is not stable enough."""
        return round(self.stdev / self.mean, 4) if self.mean else 0.0

    @property
    def stable(self) -> bool:
        return self.samples >= 3 and self.coefficient_of_variation <= 0.15

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric, "samples": self.samples, "mean": self.mean,
            "min": self.minimum, "max": self.maximum, "stdev": self.stdev,
            "median": self.median, "p95": self.p95,
            "cv": self.coefficient_of_variation, "stable": self.stable,
        }


def percentile(values: Sequence[float], fraction: float) -> float:
    """Linear-interpolation percentile; `fraction` in [0, 1]."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return float(ordered[int(position)])
    weight = position - low
    return float(ordered[low] * (1 - weight) + ordered[high] * weight)


def _clean(values: Iterable[Any]) -> list[float]:
    out: list[float] = []
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isnan(float(value)):
            out.append(float(value))
    return out


def aggregate(metric: str, values: Sequence[Any]) -> Aggregate:
    clean = _clean(values)
    if not clean:
        return Aggregate(metric, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return Aggregate(
        metric=metric,
        samples=len(clean),
        mean=round(statistics.fmean(clean), 4),
        minimum=round(min(clean), 4),
        maximum=round(max(clean), 4),
        stdev=round(statistics.pstdev(clean), 4) if len(clean) > 1 else 0.0,
        median=round(statistics.median(clean), 4),
        p95=round(percentile(clean, 0.95), 4),
    )


def aggregate_all(rows: Iterable[dict[str, Any]], metrics: Sequence[str]
                  ) -> dict[str, Aggregate]:
    materialized = list(rows)
    return {
        metric: aggregate(metric, [r.get(metric) for r in materialized
                                   if isinstance(r.get(metric), (int, float))])
        for metric in metrics
    }


def delta_percent(baseline: Any, current: Any) -> float | None:
    """Relative change, positive meaning `current` is the larger value."""
    if baseline in (None, 0) or current is None:
        return None
    return round((float(current) - float(baseline)) / abs(float(baseline)) * 100.0, 2)
