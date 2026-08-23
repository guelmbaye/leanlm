"""Baseline management and regression detection (BHB s.10, s.11).

The baseline is the same model, on the same machine, with the same corpus and
prompts -- *without* the LeanLM optimization layer. Anything else would flatter
the result.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shared.clock import iso_now
from ..shared.serialization import read_json, write_json

# metric -> (higher_is_better, allowed_regression_pct)
TRACKED_METRICS: dict[str, tuple[bool, float]] = {
    "tokens_per_second": (True, 5.0),
    "first_token_latency_ms": (False, 10.0),
    "total_ms": (False, 10.0),
    "peak_rss_mb": (False, 8.0),
    "prompt_tokens": (False, 15.0),
    "response_grounding_rate": (True, 5.0),
    "temperature_c": (False, 5.0),
}


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    metric: str
    baseline: float
    current: float
    delta_pct: float
    improved: bool
    regression: bool

    def to_dict(self) -> dict[str, Any]:
        return {"metric": self.metric, "baseline": self.baseline, "current": self.current,
                "delta_pct": self.delta_pct, "improved": self.improved,
                "regression": self.regression}


def save_baseline(path: str | Path, summary: dict[str, Any]) -> Path:
    payload = dict(summary)
    payload["recorded_at"] = iso_now()
    return write_json(path, payload)


def load_baseline(path: str | Path) -> dict[str, Any] | None:
    target = Path(path)
    if not target.is_file():
        return None
    try:
        return read_json(target)
    except Exception:
        return None


# `configs/benchmark.yaml` speaks in families, not metric names, because a jury
# cares about "throughput regressed", not about which counter carried it.
REGRESSION_FAMILIES: dict[str, tuple[str, ...]] = {
    "throughput_pct": ("tokens_per_second",),
    "latency_pct": ("first_token_latency_ms", "total_ms"),
    "memory_pct": ("peak_rss_mb", "prompt_tokens"),
    "grounding_pct": ("response_grounding_rate",),
    "thermal_pct": ("temperature_c",),
}


def expand_tolerances(config: dict[str, float] | None) -> dict[str, float]:
    """Turn family thresholds into per-metric thresholds."""
    expanded: dict[str, float] = {}
    for key, value in (config or {}).items():
        for metric in REGRESSION_FAMILIES.get(key, (key,)):
            expanded[metric] = float(value)
    return expanded


def is_regression_baseline(payload: dict[str, Any] | None) -> bool:
    """A regression baseline is a previous LeanLM campaign, nothing else.

    The naive baseline (`kind: naive_baseline`) measures the path *without* the
    optimization layer. Treating it as a regression reference would flag every
    intentional trade-off as a defect.
    """
    return bool(payload) and payload.get("kind") != "naive_baseline"


def compare_to_baseline(current: dict[str, Any], baseline: dict[str, Any] | None,
                        overrides: dict[str, float] | None = None
                        ) -> dict[str, BaselineComparison]:
    if not is_regression_baseline(baseline):
        return {}
    assert baseline is not None
    tolerances = expand_tolerances(overrides)
    result: dict[str, BaselineComparison] = {}
    for metric, (higher_is_better, tolerance) in TRACKED_METRICS.items():
        allowed = tolerances.get(metric, tolerance)
        before, after = _value(baseline, metric), _value(current, metric)
        if before is None or after is None or before == 0:
            continue
        delta_pct = round(((after - before) / abs(before)) * 100.0, 2)
        improved = delta_pct > 0 if higher_is_better else delta_pct < 0
        magnitude = abs(delta_pct)
        regression = (not improved) and magnitude > allowed
        result[metric] = BaselineComparison(metric, before, after, delta_pct,
                                            improved, regression)
    return result


def _value(payload: dict[str, Any], metric: str) -> float | None:
    node = payload.get(metric)
    if isinstance(node, dict):
        node = node.get("mean")
    if isinstance(node, (int, float)):
        return float(node)
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        return _value(metrics, metric)
    return None


def regressions(comparison: dict[str, BaselineComparison]) -> list[BaselineComparison]:
    return [c for c in comparison.values() if c.regression]


# metric -> (higher_is_better) for the naive comparison. No tolerances here: a
# difference against the no-layer path is a trade-off to describe, never a defect
# to gate on.
# Metrics for which 0.0 cannot be a real measurement: a zero there means the
# campaign did not record it. Reporting "29.6 -> 0.0, favours LeanLM (-100%)" is
# how a missing metric turns into a boast.
_ZERO_MEANS_MISSING = frozenset({
    "prompt_tokens", "context_tokens", "corpus_coverage", "total_ms", "peak_rss_mb",
})

NAIVE_COMPARISON_METRICS: dict[str, bool] = {
    "prompt_tokens": False,
    "context_tokens": False,
    "corpus_coverage": True,
    "response_grounding_rate": True,
    "tokens_per_second": True,
    "total_ms": False,
    "peak_rss_mb": False,
}


def compare_to_naive(campaign: dict[str, Any], naive: dict[str, Any] | None
                     ) -> dict[str, Any]:
    """The comparison the submission report shows: LeanLM against no LeanLM.

    Returned as descriptive rows, not pass/fail. Where LeanLM is worse the row
    says so; the argument is made by the numbers or not at all.
    """
    if not naive or naive.get("kind") != "naive_baseline":
        return {}
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for metric, higher_is_better in NAIVE_COMPARISON_METRICS.items():
        before, after = _value(naive, metric), _value(campaign, metric)
        if before is None or after is None:
            continue
        if metric in _ZERO_MEANS_MISSING and (before == 0 or after == 0):
            skipped.append(metric)
            continue
        delta = None if before == 0 else round(((after - before) / abs(before)) * 100.0, 2)
        rows.append({
            "metric": metric, "without_leanlm": before, "with_leanlm": after,
            "delta_pct": delta,
            "favours_leanlm": (after > before) if higher_is_better else (after < before),
        })
    return {
        "naive_truncated_corpus": naive.get("corpus_truncated"),
        "naive_corpus_coverage": naive.get("corpus_coverage"),
        "naive_dropped_tokens": naive.get("dropped_tokens"),
        "comparable": (naive.get("is_simulated") == campaign.get("is_simulated")
                       and naive.get("corpus", {}).get("checksum")
                       == campaign.get("corpus", {}).get("checksum")),
        "rows": rows,
        "not_measured": skipped,
    }
