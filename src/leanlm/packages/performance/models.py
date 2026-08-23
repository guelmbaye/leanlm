"""Package-local models for CAP-007.

These are the shapes the sampler works with internally. They are deliberately
separate from the canonical ``ResourceObservation`` DTO: the DTO is what other
capabilities consume, these are what this capability measures with.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ResourceReading:
    """One raw sample from the machine, with its provenance.

    ``thermal_source`` is never omitted. "No sensor available" and "41 degrees"
    are different statements, and a report that confuses them is misleading.
    """

    rss_mb: float
    available_ram_mb: float
    total_ram_mb: float
    cpu_percent: float
    temperature_c: float | None
    thermal_source: str
    memory_source: str
    sampled_at_ns: int

    @property
    def memory_pressure(self) -> float:
        if not self.total_ram_mb:
            return 0.0
        return round(1.0 - (self.available_ram_mb / self.total_ram_mb), 4)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rss_mb": self.rss_mb, "available_ram_mb": self.available_ram_mb,
            "total_ram_mb": self.total_ram_mb, "cpu_percent": self.cpu_percent,
            "temperature_c": self.temperature_c, "thermal_source": self.thermal_source,
            "memory_source": self.memory_source, "memory_pressure": self.memory_pressure,
        }


@dataclass
class PeakTracker:
    """Running peaks over the life of a request.

    Peak matters more than the final value: a run that briefly touched 7.6 GB on
    an 8 GB laptop is a run that nearly failed, even if it ended at 2 GB.
    """

    peak_rss_mb: float = 0.0
    peak_cpu_percent: float = 0.0
    peak_temperature_c: float | None = None
    minimum_available_ram_mb: float | None = None
    samples: int = 0
    warnings: list[str] = field(default_factory=list)

    def observe(self, reading: ResourceReading) -> None:
        self.samples += 1
        self.peak_rss_mb = max(self.peak_rss_mb, reading.rss_mb)
        self.peak_cpu_percent = max(self.peak_cpu_percent, reading.cpu_percent)
        if reading.temperature_c is not None:
            self.peak_temperature_c = (reading.temperature_c
                                       if self.peak_temperature_c is None
                                       else max(self.peak_temperature_c,
                                                reading.temperature_c))
        if reading.available_ram_mb:
            self.minimum_available_ram_mb = (
                reading.available_ram_mb if self.minimum_available_ram_mb is None
                else min(self.minimum_available_ram_mb, reading.available_ram_mb))

    def reset(self) -> None:
        self.peak_rss_mb = 0.0
        self.peak_cpu_percent = 0.0
        self.peak_temperature_c = None
        self.minimum_available_ram_mb = None
        self.samples = 0
        self.warnings.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "peak_rss_mb": round(self.peak_rss_mb, 2),
            "peak_cpu_percent": round(self.peak_cpu_percent, 2),
            "peak_temperature_c": self.peak_temperature_c,
            "minimum_available_ram_mb": self.minimum_available_ram_mb,
            "samples": self.samples,
            "warnings": list(self.warnings),
        }
