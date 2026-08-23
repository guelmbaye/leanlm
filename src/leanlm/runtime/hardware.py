"""Comparing this machine against the ADTC standard laptop profile.

A measurement is only transferable if the machine that produced it resembles the
machine that will re-produce it. The audit sandbox re-runs every submission and
compares: peak RSS tolerates +/-15%, throughput +/-25%, and beyond 50% the
comparison *fails*. A number measured on hardware far from the profile is
therefore not merely optimistic or pessimistic -- it is a risk of failing the
comparison outright.

So `leanlm doctor` states the divergence instead of leaving it to be discovered
after submission.
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass, field
from typing import Any

# "4 vCPU, 8 GB RAM, integrated GPU only" -- the standard laptop profile, with
# the scoring limit the profiler actually applies.
PROFILE_VCPU = 4
PROFILE_RAM_GB = 8.0
SCORING_RAM_LIMIT_GB = 7.0

# The comparator's tolerances, quoted so the warning can name the consequence.
THROUGHPUT_TOLERANCE = 0.25
MEMORY_TOLERANCE = 0.15


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    logical_cpus: int
    physical_cpus: int | None
    total_ram_gb: float
    platform_name: str
    machine: str
    processor: str
    findings: tuple[str, ...] = ()
    blocking_for_submission: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_cpus": self.logical_cpus, "physical_cpus": self.physical_cpus,
            "total_ram_gb": self.total_ram_gb, "platform": self.platform_name,
            "machine": self.machine, "processor": self.processor,
            "findings": list(self.findings),
            "blocking_for_submission": self.blocking_for_submission,
            **self.extra,
        }


def _physical_cpus() -> int | None:
    try:
        import psutil  # type: ignore
        return psutil.cpu_count(logical=False)
    except Exception:
        pass
    if sys.platform.startswith("linux"):
        try:
            cores = set()
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                package = core = None
                for line in handle:
                    if line.startswith("physical id"):
                        package = line.split(":")[1].strip()
                    elif line.startswith("core id"):
                        core = line.split(":")[1].strip()
                        cores.add((package, core))
            return len(cores) or None
        except Exception:
            return None
    return None


def inspect(total_ram_mb: float | None = None) -> HardwareProfile:
    from ..packages.performance.services import _read_meminfo

    logical = os.cpu_count() or 1
    physical = _physical_cpus()
    ram_gb = round((total_ram_mb if total_ram_mb is not None
                    else _read_meminfo()[0]) / 1024.0, 2)

    findings: list[str] = []
    blocking = False

    if ram_gb <= 0:
        findings.append("installed memory could not be measured on this platform")
    elif ram_gb > PROFILE_RAM_GB * 1.25:
        findings.append(
            f"{ram_gb:.0f} GB installed against an {PROFILE_RAM_GB:.0f} GB profile. "
            "Peak RSS still transfers, but nothing here will swap, so the failure "
            "mode that disqualifies -- an out-of-memory kill on the target machine "
            "-- cannot be observed. Cap the run instead: "
            "`docker run --memory=7.5g`.")
    elif ram_gb < PROFILE_RAM_GB * 0.9:
        findings.append(
            f"only {ram_gb:.1f} GB installed; the profile assumes "
            f"{PROFILE_RAM_GB:.0f} GB and the scoring limit is "
            f"{SCORING_RAM_LIMIT_GB:.0f} GB")

    if logical < PROFILE_VCPU:
        blocking = True
        findings.append(
            f"{logical} logical CPUs against {PROFILE_VCPU} vCPU in the profile. "
            f"Throughput measured here will read low, and the audit comparison "
            f"fails beyond 50% variance (tolerance is "
            f"+/-{THROUGHPUT_TOLERANCE:.0%}).")
    elif physical is not None and physical < PROFILE_VCPU // 2:
        findings.append(
            f"{logical} logical CPUs but only {physical} physical core"
            f"{'s' if physical != 1 else ''}. Hyper-threaded siblings share "
            "execution units, and llama.cpp is compute- and bandwidth-bound: "
            "expect throughput well below a machine with "
            f"{PROFILE_VCPU} real cores.")

    if sys.platform == "win32":
        findings.append(
            "Windows: the official profiler expects `llama-bench` on PATH and "
            "reads core temperature through Linux sensors. Thermal will be "
            "unmeasured here, and an unmeasured thermal is an unquantified risk "
            "of the 10-point penalty, not an absence of one.")

    return HardwareProfile(
        logical_cpus=logical, physical_cpus=physical, total_ram_gb=ram_gb,
        platform_name=platform.platform(), machine=platform.machine(),
        processor=platform.processor() or "unknown",
        findings=tuple(findings), blocking_for_submission=blocking,
    )


def render(profile: HardwareProfile) -> list[str]:
    lines = [
        f"  cpu              : {profile.logical_cpus} logical"
        + (f", {profile.physical_cpus} physical" if profile.physical_cpus else "")
        + f" (profile: {PROFILE_VCPU} vCPU)",
        f"  installed RAM    : {profile.total_ram_gb:.1f} GB "
        f"(profile: {PROFILE_RAM_GB:.0f} GB, scored against "
        f"{SCORING_RAM_LIMIT_GB:.0f} GB)",
    ]
    if not profile.findings:
        lines.append("  hardware profile : matches the standard laptop profile")
        return lines
    lines.append("  hardware profile : DIVERGES from the standard laptop profile")
    for finding in profile.findings:
        lines.append(f"    - {finding}")
    lines.append("    Use this machine for development and model selection; produce "
                 "the submitted")
    lines.append("    numbers on hardware close to the profile.")
    return lines
