"""Performance policies (EBPB s.15, BHB s.11)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PerformancePolicy:
    sample_interval_s: float = 0.5
    thermal_warn_c: float = 78.0
    thermal_critical_c: float = 92.0
    memory_warn_ratio: float = 0.88
    # Regression thresholds used by the harness when comparing to a baseline.
    throughput_regression_pct: float = 5.0
    memory_regression_pct: float = 8.0
    latency_regression_pct: float = 10.0
    accuracy_regression_pct: float = 2.0

    @classmethod
    def from_config(cls, config: dict) -> "PerformancePolicy":
        node = (config or {}).get("performance", {})
        base = cls()
        return cls(**{f: type(getattr(base, f))(node.get(f, getattr(base, f)))
                      for f in base.__slots__})
