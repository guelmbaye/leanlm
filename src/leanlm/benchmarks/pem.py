"""Performance Evidence Matrix (BHB s.5).

Each benchmark answers four questions: what are we testing, why, what did we
measure, and what decision does the result support. A measurement that supports
no decision is a number, not evidence.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ..shared.clock import iso_now


@dataclass(frozen=True, slots=True)
class PerformanceEvidence:
    benchmark_id: str
    scenario_id: str
    hypothesis: str
    conditions: dict[str, Any] = field(default_factory=dict)
    variables: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    observations: tuple[str, ...] = ()
    adtc_criteria: tuple[str, ...] = ()
    decision: str = "informational"
    verdict: str = "recorded"
    created_at: str = field(default_factory=iso_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "scenario_id": self.scenario_id,
            "hypothesis": self.hypothesis,
            "conditions": dict(self.conditions),
            "variables": dict(self.variables),
            "metrics": dict(self.metrics),
            "observations": list(self.observations),
            "adtc_criteria": list(self.adtc_criteria),
            "decision": self.decision,
            "verdict": self.verdict,
            "created_at": self.created_at,
        }


def render_matrix(evidence: list[PerformanceEvidence]) -> str:
    """Markdown table for REPORT.md and the Engineering Decision Book."""
    header = ("| Scenario | Hypothesis | tok/s | first token | peak RSS | "
              "grounding | verdict |\n|---|---|---|---|---|---|---|")
    rows = []
    for item in evidence:
        metrics = item.metrics
        rows.append(
            f"| {item.scenario_id} | {item.hypothesis} "
            f"| {_get(metrics, 'tokens_per_second')} "
            f"| {_get(metrics, 'first_token_latency_ms')} ms "
            f"| {_get(metrics, 'peak_rss_mb')} MB "
            f"| {_get(metrics, 'response_grounding_rate')} "
            f"| {item.verdict} |"
        )
    return "\n".join([header, *rows])


def _get(metrics: dict[str, Any], key: str) -> str:
    node = metrics.get(key)
    if isinstance(node, dict):
        return str(node.get("mean", "-"))
    return str(node) if node is not None else "-"
