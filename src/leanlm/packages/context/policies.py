"""Context policies (IIB s.7, s.8, s.14).

IP-01: never send more context than necessary.
IP-04: preserve hardware resources.

Every number below is a *policy*, not a constant scattered in the code, so an
experiment can change one variable at a time (EBPB s.7).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ...contracts.dto import IntentType


@dataclass(frozen=True, slots=True)
class BudgetPolicy:
    """How the context budget reacts to hardware and to the question type."""

    min_context_tokens: int = 256
    prompt_overhead_tokens: int = 160
    ram_headroom_mb: float = 512.0
    ram_soft_floor_mb: float = 900.0
    thermal_warn_c: float = 78.0
    thermal_critical_c: float = 92.0
    cpu_warn_percent: float = 85.0
    max_passages_hard_cap: int = 12
    # A factual lookup does not need the same context as a full summary.
    intent_factors: dict[str, float] = field(default_factory=lambda: {
        IntentType.FACTUAL.value: 0.55,
        IntentType.EXTRACTION.value: 0.65,
        IntentType.COMPARISON.value: 0.90,
        IntentType.EXPLANATION.value: 0.80,
        IntentType.SUMMARY.value: 1.00,
        IntentType.DRAFTING.value: 0.85,
    })
    intent_passages: dict[str, int] = field(default_factory=lambda: {
        IntentType.FACTUAL.value: 4,
        IntentType.EXTRACTION.value: 5,
        IntentType.COMPARISON.value: 8,
        IntentType.EXPLANATION.value: 6,
        IntentType.SUMMARY.value: 10,
        IntentType.DRAFTING.value: 6,
    })

    @classmethod
    def from_config(cls, config: dict) -> "BudgetPolicy":
        node = (config or {}).get("budget", {})
        base = cls()
        return cls(
            min_context_tokens=int(node.get("min_context_tokens", base.min_context_tokens)),
            prompt_overhead_tokens=int(node.get("prompt_overhead_tokens",
                                                base.prompt_overhead_tokens)),
            ram_headroom_mb=float(node.get("ram_headroom_mb", base.ram_headroom_mb)),
            ram_soft_floor_mb=float(node.get("ram_soft_floor_mb", base.ram_soft_floor_mb)),
            thermal_warn_c=float(node.get("thermal_warn_c", base.thermal_warn_c)),
            thermal_critical_c=float(node.get("thermal_critical_c", base.thermal_critical_c)),
            cpu_warn_percent=float(node.get("cpu_warn_percent", base.cpu_warn_percent)),
            max_passages_hard_cap=int(node.get("max_passages_hard_cap",
                                               base.max_passages_hard_cap)),
            intent_factors={**base.intent_factors, **(node.get("intent_factors") or {})},
            intent_passages={**base.intent_passages, **(node.get("intent_passages") or {})},
        )


@dataclass(frozen=True, slots=True)
class OptimizationPolicy:
    """What the Context Optimization Engine is allowed to remove or merge."""

    near_duplicate_threshold: float = 0.92
    merge_adjacent_below_tokens: int = 70
    drop_below_content_words: int = 4
    boilerplate_patterns: tuple[str, ...] = (
        r"^page\s+\d+(\s*(/|of|sur)\s*\d+)?$",
        r"^\d+\s*$",
        r"^[-_=*.\s]{3,}$",
        r"^(confidentiel|confidential|internal use only|tous droits reserves)\b.{0,40}$",
    )
    keep_structure: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "OptimizationPolicy":
        node = (config or {}).get("optimization", {})
        base = cls()
        return cls(
            near_duplicate_threshold=float(node.get("near_duplicate_threshold",
                                                    base.near_duplicate_threshold)),
            merge_adjacent_below_tokens=int(node.get("merge_adjacent_below_tokens",
                                                     base.merge_adjacent_below_tokens)),
            drop_below_content_words=int(node.get("drop_below_content_words",
                                                  base.drop_below_content_words)),
            boilerplate_patterns=tuple(node.get("boilerplate_patterns",
                                                base.boilerplate_patterns)),
            keep_structure=bool(node.get("keep_structure", base.keep_structure)),
        )
