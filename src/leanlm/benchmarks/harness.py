"""Benchmark harness (BHB s.4, s.8).

Runs scenarios against a live runtime, repeats each measurement, aggregates it
with dispersion, and turns the result into Performance Evidence.

Two properties matter more than speed here:
  * warm-up runs are excluded from the statistics but *reported*, because the
    cold-start cost is itself a competition-relevant number;
  * a scenario that fails is recorded as a failed scenario, never dropped. A
    campaign that silently loses its worst run is not a measurement.
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from ..contracts.iec import InferenceExecutionContext
from ..runtime.runtime import LeanLMRuntime
from ..shared.clock import Stopwatch, iso_now
from ..shared.errors import LeanLMError
from ..shared.serialization import write_json
from .baseline import compare_to_baseline, load_baseline, regressions
from .pem import PerformanceEvidence
from .scenarios import Scenario, load_scenarios
from .stats import Aggregate, aggregate_all

REPORTED_METRICS = (
    "tokens_per_second", "first_token_latency_ms", "inference_ms", "total_ms",
    "prompt_tokens", "context_tokens", "generated_tokens", "peak_rss_mb",
    "available_ram_mb", "cpu_percent", "context_compression_ratio",
    "retrieval_precision", "response_grounding_rate", "prompt_efficiency",
)

ProgressHook = Callable[[str, dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class RunRecord:
    scenario_id: str
    question: str
    iteration: int
    warmup: bool
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    confidence: str = ""
    validation_passed: bool = False
    evidence_count: int = 0
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id, "question": self.question,
            "iteration": self.iteration, "warmup": self.warmup, "ok": self.ok,
            "confidence": self.confidence, "validation_passed": self.validation_passed,
            "evidence_count": self.evidence_count, "error": self.error,
            **self.metrics,
        }


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: Scenario
    runs: tuple[RunRecord, ...]
    aggregates: dict[str, Aggregate]
    verdict: str
    observations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.verdict == "pass"

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "runs": [r.to_dict() for r in self.runs],
            "metrics": {k: v.to_dict() for k, v in self.aggregates.items()},
            "verdict": self.verdict,
            "observations": list(self.observations),
        }


class BenchmarkHarness:
    def __init__(self, runtime: LeanLMRuntime, *, repeat: int = 3, warmup: int = 1,
                 progress: ProgressHook | None = None) -> None:
        self.runtime = runtime
        self.repeat = max(1, int(repeat))
        self.warmup = max(0, int(warmup))
        self.progress = progress

    def _emit(self, event: str, payload: dict[str, Any]) -> None:
        if self.progress is not None:
            self.progress(event, payload)

    # -- execution ----------------------------------------------------------
    def run(self, scenarios: Sequence[Scenario] | None = None) -> dict[str, Any]:
        selected = tuple(scenarios) if scenarios else load_scenarios()
        watch = Stopwatch()
        results: list[ScenarioResult] = []
        for scenario in selected:
            self._emit("scenario_started", {"id": scenario.id, "name": scenario.name})
            result = self.run_scenario(scenario)
            results.append(result)
            self._emit("scenario_completed", {"id": scenario.id, "verdict": result.verdict})
        elapsed = watch.stop()
        return self._summarize(results, elapsed)

    def run_scenario(self, scenario: Scenario) -> ScenarioResult:
        self._prepare_corpus(scenario)
        if scenario.cold_start:
            self._force_cold_start()
        runs: list[RunRecord] = []
        questions = scenario.questions or ("",)
        for question in questions:
            for _ in range(max(1, scenario.repeat_question)):
                for index in range(self.warmup + self.repeat):
                    is_warmup = index < self.warmup and not scenario.cold_start
                    runs.append(self._single_run(scenario, question, index, is_warmup))
                    if scenario.cold_start:
                        break  # a cold start cannot be repeated in the same process
        measured = [r for r in runs if not r.warmup and r.ok]
        aggregates = aggregate_all([r.metrics for r in measured], REPORTED_METRICS)
        verdict, observations = self._judge(scenario, runs, measured, aggregates)
        return ScenarioResult(scenario, tuple(runs), aggregates, verdict, observations)

    def _single_run(self, scenario: Scenario, question: str, index: int,
                    warmup: bool) -> RunRecord:
        try:
            iec = self.runtime.ask(question)
        except LeanLMError as error:
            return RunRecord(scenario.id, question, index, warmup, False,
                             error=f"{error.record.code}: {error.record.message}")
        except Exception as error:  # noqa: BLE001 -- an unexpected crash is a result
            return RunRecord(scenario.id, question, index, warmup, False,
                             error=f"UNEXPECTED: {type(error).__name__}: {error}")
        metrics = iec.metrics.to_flat() if iec.metrics else {}
        record = RunRecord(
            scenario_id=scenario.id, question=question, iteration=index, warmup=warmup,
            ok=not iec.errors, metrics=metrics,
            confidence=iec.validation.confidence.value if iec.validation else "",
            validation_passed=bool(iec.validation and iec.validation.passed),
            evidence_count=len(iec.evidence),
            error="; ".join(e.code for e in iec.errors),
        )
        self._emit("run_completed", {"scenario": scenario.id, "warmup": warmup,
                                     "ok": record.ok,
                                     "total_ms": metrics.get("total_ms")})
        return record

    # -- scenario shaping ---------------------------------------------------
    def _prepare_corpus(self, scenario: Scenario) -> None:
        """``documents: n`` restricts the corpus without re-ingesting anything."""
        if scenario.documents and scenario.documents > 0:
            available = self.runtime.corpus.documents()
            ids = [d["document_id"] for d in available[: scenario.documents]]
            self.runtime.use_documents(ids)
        else:
            self.runtime.use_documents(None)

    def _force_cold_start(self) -> None:
        """S5 measures the first request after a load, model unloading included."""
        self.runtime.backend.unload()
        self.runtime._loaded = False  # noqa: SLF001 -- deliberate, documented above

    # -- judgement ----------------------------------------------------------
    def _judge(self, scenario: Scenario, runs: list[RunRecord],
               measured: list[RunRecord], aggregates: dict[str, Aggregate]
               ) -> tuple[str, tuple[str, ...]]:
        observations: list[str] = []
        failures = [r for r in runs if not r.ok]
        if failures:
            observations.append(f"{len(failures)} run(s) failed: "
                                f"{failures[0].error or 'unknown error'}")
            return "fail", tuple(observations)
        if not measured:
            return "inconclusive", ("no measured run",)

        if scenario.expect_insufficient:
            honest = [r for r in measured if r.confidence == "insufficient"
                      or r.evidence_count == 0]
            if len(honest) != len(measured):
                observations.append(
                    "the system answered a question the corpus cannot support: "
                    "this is the failure mode that matters most (P3)")
                return "fail", tuple(observations)
            observations.append("insufficiency correctly declared on every run")

        throughput = aggregates.get("tokens_per_second")
        if throughput and throughput.samples > 1 and not throughput.stable:
            observations.append(
                f"throughput is not stable (cv={throughput.coefficient_of_variation}); "
                "report the dispersion, not just the mean")
        rss = aggregates.get("peak_rss_mb")
        if rss and rss.maximum > 0:
            observations.append(f"peak RSS {rss.maximum} MB")
        grounding = aggregates.get("response_grounding_rate")
        if grounding and grounding.samples and grounding.mean < 0.6:
            observations.append(f"low grounding rate ({grounding.mean})")
        return "pass", tuple(observations)

    # -- reporting ----------------------------------------------------------
    def _summarize(self, results: list[ScenarioResult], elapsed_ms: float) -> dict[str, Any]:
        every_run = [r.metrics for result in results for r in result.runs
                     if not r.warmup and r.ok]
        overall = aggregate_all(every_run, REPORTED_METRICS)
        evidence = [self._evidence(result) for result in results]
        simulated = any(bool(r.metrics.get("is_simulated"))
                        for result in results for r in result.runs)
        summary: dict[str, Any] = {
            "campaign_id": f"bench-{iso_now().replace(':', '').replace('-', '')[:15]}",
            "created_at": iso_now(),
            "elapsed_ms": elapsed_ms,
            "repeat": self.repeat,
            "warmup": self.warmup,
            "environment": environment_fingerprint(),
            "profile": self.runtime.profile.to_dict(),
            "backend": self.runtime.backend.describe(),
            "corpus": {
                "documents": self.runtime.corpus.document_count(),
                "units": self.runtime.corpus.unit_count(),
                "checksum": self.runtime.corpus.corpus_checksum()[:16],
            },
            "is_simulated": simulated,
            "scenarios": [result.to_dict() for result in results],
            "evidence": [item.to_dict() for item in evidence],
            "metrics": {k: v.to_dict() for k, v in overall.items()},
            "verdict": "pass" if all(r.ok for r in results) else "fail",
        }
        for metric, value in overall.items():
            summary[metric] = value.mean
        if simulated:
            summary["warning"] = (
                "SIMULATED BACKEND: these numbers describe the optimization layer only, "
                "not model inference. They must not be reported as model performance."
            )
        return summary

    def _evidence(self, result: ScenarioResult) -> PerformanceEvidence:
        return PerformanceEvidence(
            benchmark_id=f"BM-{result.scenario.id}",
            scenario_id=result.scenario.id,
            hypothesis=result.scenario.hypothesis or result.scenario.name,
            conditions={
                "profile": self.runtime.profile.id,
                "profile_fingerprint": self.runtime.profile.fingerprint,
                "backend": self.runtime.backend.name,
                "repeat": self.repeat, "warmup": self.warmup,
                "documents": result.scenario.documents or "all",
            },
            variables=dict(result.scenario.variables),
            metrics={k: v.to_dict() for k, v in result.aggregates.items()},
            observations=result.observations,
            adtc_criteria=result.scenario.adtc_criteria,
            decision=("evidence for the ADTC criteria: "
                      + ", ".join(result.scenario.adtc_criteria)
                      if result.scenario.adtc_criteria else "informational"),
            verdict=result.verdict,
        )


def environment_fingerprint() -> dict[str, Any]:
    """Everything a reader needs to know before trusting a number."""
    import os
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "measured_at": iso_now(),
    }
    try:
        import psutil  # type: ignore
        info["total_ram_mb"] = round(psutil.virtual_memory().total / 1048576.0, 1)
    except Exception:
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        info["total_ram_mb"] = round(int(line.split()[1]) / 1024.0, 1)
                        break
        except Exception:
            info["total_ram_mb"] = None
    return info


def run_campaign(runtime: LeanLMRuntime, *, repeat: int = 3, warmup: int = 1,
                 scenarios: Sequence[Scenario] | None = None,
                 output: str | Path | None = None,
                 baseline_path: str | Path | None = None,
                 naive_path: str | Path | None = None,
                 progress: ProgressHook | None = None) -> dict[str, Any]:
    """Convenience entry point used by the CLI and the workspace."""
    harness = BenchmarkHarness(runtime, repeat=repeat, warmup=warmup, progress=progress)
    summary = harness.run(scenarios)
    if baseline_path:
        baseline = load_baseline(baseline_path)
        comparison = compare_to_baseline(summary, baseline)
        summary["baseline_comparison"] = {k: v.to_dict() for k, v in comparison.items()}
        summary["regressions"] = [c.to_dict() for c in regressions(comparison)]
        summary["baseline_available"] = baseline is not None
    if naive_path:
        from .baseline import compare_to_naive
        naive = load_baseline(naive_path)
        comparison = compare_to_naive(summary, naive)
        if comparison:
            summary["naive_comparison"] = comparison
        summary["naive_baseline_available"] = naive is not None
    if output:
        write_json(output, summary)
        summary["output_path"] = str(output)
    return summary
