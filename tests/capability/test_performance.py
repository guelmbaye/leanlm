"""CAP-007 Performance Engineering -- FR-07."""
from __future__ import annotations

from leanlm.contracts.iec import InferenceExecutionContext
from leanlm.packages.performance import SPEC, MetricsAssembler, ResourceSampler
from leanlm.packages.performance.models import PeakTracker, ResourceReading
from leanlm.packages.performance.policies import PerformancePolicy


class TestSpec:
    def test_identity(self):
        assert SPEC.capability_id == "CAP-007"
        assert SPEC.stages == ("metrics_finalization",)


class TestSampler:
    def test_a_sample_reports_memory(self):
        observation = ResourceSampler().sample()
        assert observation.total_ram_mb > 0
        assert observation.available_ram_mb >= 0
        assert observation.process_rss_mb > 0

    def test_the_thermal_source_is_always_stated(self):
        """"No sensor" and "41 degrees" are different statements. A report that
        confuses them is misleading, so the source travels with the value."""
        observation = ResourceSampler().sample()
        assert observation.thermal_source
        if observation.temperature_c is None:
            assert observation.thermal_source in ("unavailable", "none", "unknown")

    def test_sampling_twice_does_not_crash_or_drift_wildly(self):
        sampler = ResourceSampler()
        first, second = sampler.sample(), sampler.sample()
        assert abs(first.total_ram_mb - second.total_ram_mb) < 1.0


class TestPeakTracker:
    def test_it_keeps_the_maximum_not_the_last_value(self):
        """A run that briefly touched 7.6 GB on an 8 GB laptop nearly failed,
        even if it ended at 2 GB."""
        tracker = PeakTracker()
        tracker.observe(_reading(rss=7600.0, available=400.0))
        tracker.observe(_reading(rss=2000.0, available=6000.0))
        assert tracker.peak_rss_mb == 7600.0
        assert tracker.minimum_available_ram_mb == 400.0

    def test_reset_clears_everything(self):
        tracker = PeakTracker()
        tracker.observe(_reading())
        tracker.reset()
        assert tracker.samples == 0 and tracker.peak_rss_mb == 0.0

    def test_memory_pressure_is_derived_not_invented(self):
        reading = _reading(available=2000.0, total=8000.0)
        assert reading.memory_pressure == 0.75


class TestAssembler:
    def test_metrics_exist_even_for_an_empty_context(self):
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="q", profile_id="development")
        snapshot = MetricsAssembler().assemble(iec, total_ms=12.5)
        assert snapshot.total_ms == 12.5
        assert snapshot.request_id == "r"

    def test_stage_durations_are_carried_through(self):
        from leanlm.shared.telemetry import StageRecord
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="q", profile_id="development")
        iec = iec.with_stage(StageRecord(stage="context_optimization",
                                         capability="context",
                                         started_at="2026-06-01T10:00:00Z",
                                         duration_ms=4.2))
        snapshot = MetricsAssembler().assemble(iec, total_ms=10.0)
        assert snapshot.stage_durations.get("context_optimization") == 4.2


class TestPolicy:
    def test_thermal_thresholds_are_ordered(self):
        policy = PerformancePolicy()
        assert policy.thermal_warn_c < policy.thermal_critical_c


def _reading(rss: float = 100.0, available: float = 4000.0, total: float = 8000.0):
    return ResourceReading(rss_mb=rss, available_ram_mb=available, total_ram_mb=total,
                           cpu_percent=10.0, temperature_c=None,
                           thermal_source="unavailable", memory_source="/proc/meminfo",
                           sampled_at_ns=1)


class TestSimulatedTimings:
    def test_a_simulated_run_reports_no_throughput(self):
        """Tens of millions of tokens per second is one copy-paste away from a
        report. A backend that ran no model reports no model timing."""
        from leanlm.contracts.dto import ModelResponse
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="q", profile_id="development")
        response = ModelResponse(
            meta=ModelResponse.build_meta(identity=("x",), stage="model_inference"),
            text="extrait", generated_tokens=12, tokens_per_second=27_750_000.0,
            first_token_latency_ms=0.02, is_simulated=True).sealed()
        snapshot = MetricsAssembler().assemble(iec.evolve(response=response),
                                               total_ms=6.1)
        assert snapshot.tokens_per_second == 0.0
        assert snapshot.first_token_latency_ms == 0.0
        assert snapshot.is_simulated is True
        # The layer's own measurement is real and stays.
        assert snapshot.total_ms == 6.1


class TestPlatformSampling:
    """The target machine is a commodity laptop, and many of them run Windows.

    Reported from a Windows install: every memory field read 0.0 because the
    fallbacks read /proc. The consequence was not a blank field -- the scoring
    module read 0 MB available as an out-of-memory kill.
    """

    def test_a_sample_reports_real_memory_on_this_platform(self):
        observation = ResourceSampler().sample()
        assert observation.total_ram_mb > 0, (
            "no memory sampler works on this platform; scoring will lose the "
            "efficiency component")
        assert observation.process_rss_mb > 0

    def test_a_windows_path_exists(self):
        """Cannot be executed here, but its absence is itself a defect."""
        from leanlm.packages.performance import services
        assert hasattr(services, "_windows_memory")
        assert hasattr(services, "_windows_rss")

    def test_the_fallback_dispatches_on_platform(self):
        import sys
        from leanlm.packages.performance.services import _read_meminfo
        source = _read_meminfo.__doc__ or ""
        assert "Win32" in source and "/proc" in source
        total, available = _read_meminfo()
        if sys.platform != "win32":
            assert total > 0
