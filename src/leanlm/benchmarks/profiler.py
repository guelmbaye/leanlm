"""Profiling a request (EBPB s.9, submission check SUB-010).

Two things a profiler report has to make explicit, and most do not:

**The numbers here are not the numbers to report.** `cProfile` instruments every
call, which inflates total time substantially. Throughput and latency figures
belong to the benchmark campaign, which runs uninstrumented. This report answers
a different question -- *where does the time go* -- and the two must never be
mixed in the same table.

**A profile of the wrong thing is worse than no profile.** On a simulated
backend, `model_inference` costs almost nothing, so the optimization layer looks
like 100% of the cost. With a real model it is typically a few percent. The
report states which backend produced it so that nobody reads a simulated profile
as a statement about the real distribution.
"""
from __future__ import annotations

import cProfile
import io
import pstats
from dataclasses import dataclass
from typing import Any

from ..shared.clock import iso_now

# Frames belonging to the machinery rather than the work, filtered from the top
# list so that the interesting rows are visible.
_NOISE = ("cProfile", "pstats", "profile.py", "<frozen importlib", "{method 'disable'")


@dataclass(frozen=True, slots=True)
class HotSpot:
    function: str
    calls: int
    total_ms: float
    cumulative_ms: float
    per_call_us: float

    def to_dict(self) -> dict[str, Any]:
        return {"function": self.function, "calls": self.calls,
                "total_ms": self.total_ms, "cumulative_ms": self.cumulative_ms,
                "per_call_us": self.per_call_us}


def profile_request(runtime, question: str, *, repeat: int = 1,
                    top: int = 20) -> dict[str, Any]:
    """Profile one or more requests and return a report ready for the package."""
    runtime.load()          # loading is a one-off cost, profiled separately below
    profiler = cProfile.Profile()
    profiler.enable()
    last = None
    for _ in range(max(1, repeat)):
        last = runtime.ask(question, check_contract=False)
    profiler.disable()

    stats = pstats.Stats(profiler)
    stats.calc_callees()
    hotspots = _hotspots(stats, top)
    stage_share = _stage_share(last)

    return {
        "tool": "cProfile",
        "created_at": iso_now(),
        "question": question,
        "repeat": max(1, repeat),
        "backend": runtime.backend.describe(),
        "profile_id": runtime.profile.id,
        "profile_fingerprint": runtime.profile.fingerprint,
        "instrumented_total_ms": round(stats.total_tt * 1000.0, 3),
        "uninstrumented_total_ms": (last.metrics.total_ms
                                    if last is not None and last.metrics else None),
        "primitive_calls": stats.prim_calls,
        "total_calls": stats.total_calls,
        "top": [spot.to_dict() for spot in hotspots],
        "stage_share": stage_share,
        "caveats": [
            "cProfile instruments every call; the totals here are inflated and are "
            "not the figures reported as performance.",
            "the uninstrumented total from the same request is given alongside so "
            "the instrumentation overhead is visible rather than hidden.",
            ("this profile was produced by the simulated backend, where model "
             "inference is nearly free; the share attributed to the optimization "
             "layer is therefore not representative of a run with a real model."
             if runtime.backend.is_simulated else
             "produced with a real model backend."),
        ],
    }


def _hotspots(stats: pstats.Stats, top: int) -> list[HotSpot]:
    stream = io.StringIO()
    stats.stream = stream  # type: ignore[attr-defined]
    ordered = sorted(
        stats.stats.items(),                      # type: ignore[attr-defined]
        key=lambda item: item[1][3],              # cumulative time
        reverse=True,
    )
    spots: list[HotSpot] = []
    for (filename, line, name), (calls, _, total, cumulative, _) in ordered:
        label = f"{_shorten(filename)}:{line}({name})"
        if any(token in label for token in _NOISE):
            continue
        spots.append(HotSpot(
            function=label,
            calls=calls,
            total_ms=round(total * 1000.0, 3),
            cumulative_ms=round(cumulative * 1000.0, 3),
            per_call_us=round((total / calls) * 1_000_000.0, 2) if calls else 0.0,
        ))
        if len(spots) >= top:
            break
    return spots


def _shorten(filename: str) -> str:
    """Keep the part of the path that identifies the module, drop the rest."""
    if "leanlm" in filename:
        return "leanlm" + filename.split("leanlm", 1)[1]
    parts = filename.replace("\\", "/").split("/")
    return "/".join(parts[-2:]) if len(parts) > 1 else filename


def _stage_share(iec) -> dict[str, Any]:
    """Where the time went, per DIC phase, from the uninstrumented measurement."""
    if iec is None or not iec.stages:
        return {}
    durations = {record.stage: record.duration_ms for record in iec.stages}
    total = sum(durations.values()) or 1.0
    return {
        "total_ms": round(total, 3),
        "phases": [
            {"stage": stage, "ms": round(ms, 3), "share": round(ms / total, 4)}
            for stage, ms in sorted(durations.items(), key=lambda kv: -kv[1])
        ],
    }


def render_profile(report: dict[str, Any]) -> str:
    """Human readable form, used by the CLI and embedded in the report."""
    lines = [
        f"profiler: {report['tool']} on profile {report['profile_id']} "
        f"[{report['profile_fingerprint']}]",
        f"backend : {report['backend'].get('backend')}"
        f"{'  (SIMULATED)' if report['backend'].get('is_simulated') else ''}",
        f"calls   : {report['total_calls']} ({report['primitive_calls']} primitive)",
        f"time    : {report['instrumented_total_ms']} ms instrumented vs "
        f"{report['uninstrumented_total_ms']} ms uninstrumented",
        "",
        "phase share (uninstrumented):",
    ]
    for phase in report.get("stage_share", {}).get("phases", []):
        bar = "#" * max(1, int(phase["share"] * 40))
        lines.append(f"  {phase['stage']:22} {phase['ms']:>8.2f} ms "
                     f"{phase['share']:>6.1%} {bar}")
    lines.extend(["", "hot spots (instrumented, cumulative):"])
    for spot in report.get("top", [])[:12]:
        lines.append(f"  {spot['cumulative_ms']:>9.2f} ms  {spot['calls']:>7} calls  "
                     f"{spot['function']}")
    lines.extend(["", "caveats:"])
    lines.extend(f"  - {caveat}" for caveat in report.get("caveats", []))
    return "\n".join(lines)
