"""Benchmark harness (BHB) -- scenarios, evidence, baselines, regressions."""

from .accuracy import (AccuracyEvaluator, Probe, evaluate_accuracy,
                       load_probes, render_accuracy)
from .baseline import (BaselineComparison, compare_to_baseline, compare_to_naive,
                       is_regression_baseline, load_baseline, regressions,
                       save_baseline)
from .harness import BenchmarkHarness, RunRecord, ScenarioResult, run_campaign
from .naive import NaiveBaseline, run_naive_baseline
from .pem import PerformanceEvidence, render_matrix
from .profiler import profile_request, render_profile
from .scoring import (AdtcScore, ScoringPolicy, render_score, score_campaign)
from .scenarios import (BenchmarkConfig, DEFAULT_SCENARIOS, Scenario,
                        load_benchmark_config, load_scenarios)
from .stats import Aggregate, aggregate, aggregate_all

__all__ = [
    "AdtcScore", "ScoringPolicy", "score_campaign", "render_score",
    "AccuracyEvaluator", "Probe", "evaluate_accuracy", "load_probes",
    "render_accuracy",
    "BenchmarkHarness", "RunRecord", "ScenarioResult", "run_campaign",
    "NaiveBaseline", "run_naive_baseline",
    "Scenario", "load_scenarios", "DEFAULT_SCENARIOS", "BenchmarkConfig",
    "load_benchmark_config",
    "PerformanceEvidence", "render_matrix",
    "profile_request", "render_profile",
    "Aggregate", "aggregate", "aggregate_all",
    "BaselineComparison", "compare_to_baseline", "compare_to_naive",
    "is_regression_baseline", "load_baseline", "save_baseline",
    "regressions",
]
