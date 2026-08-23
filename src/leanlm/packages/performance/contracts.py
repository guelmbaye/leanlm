"""CAP-007 public contract."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_INFERENCE, Capability, CapabilitySpec,
)
from ...contracts.iec import InferenceExecutionContext, PipelineStage
from ...shared.events import EventBus, EventName
from .policies import PerformancePolicy
from .services import MetricsAssembler, ResourceSampler
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-007",
    name="Performance Engineering",
    package="packages/performance",
    pipeline=PIPELINE_INFERENCE,
    stages=(PipelineStage.METRICS_FINALIZATION.value,),
    requirement_ids=("FR-07",),
    adtc_criteria=("Throughput", "Memory", "Thermal"),
    estimated_memory_mb=6.0,
    expected_duration_ms=8.0,
    produces_metrics=METRICS,
)


class PerformanceEngineeringCapability(Capability):
    """Finalizes the metrics snapshot and raises resource warnings."""

    def __init__(self, sampler: ResourceSampler | None = None,
                 policy: PerformancePolicy | None = None,
                 bus: EventBus | None = None) -> None:
        super().__init__(SPEC)
        self.sampler = sampler or ResourceSampler(policy)
        self.policy = policy or PerformancePolicy()
        self.assembler = MetricsAssembler()
        self.bus = bus

    def run(self, iec: InferenceExecutionContext,
            stage: PipelineStage) -> InferenceExecutionContext:
        total_ms = sum(record.duration_ms for record in iec.stages)
        snapshot = self.assembler.assemble(
            iec,
            peak_rss_mb=self.sampler.peak_rss_mb,
            peak_temperature_c=self.sampler.peak_temperature_c,
            peak_cpu_percent=self.sampler.peak_cpu_percent,
            total_ms=total_ms,
        )
        result = iec.evolve(metrics=snapshot)

        if snapshot.temperature_c is not None:
            if snapshot.temperature_c >= self.policy.thermal_critical_c:
                result = result.with_warning(
                    f"thermal critical: {snapshot.temperature_c} C -- "
                    "the runtime will shrink the context budget on the next request"
                )
                self._warn(iec.session_id, "thermal_critical", snapshot.temperature_c)
            elif snapshot.temperature_c >= self.policy.thermal_warn_c:
                result = result.with_warning(f"thermal warning: {snapshot.temperature_c} C")
                self._warn(iec.session_id, "thermal_warning", snapshot.temperature_c)

        if snapshot.available_ram_mb and iec.resources and iec.resources.total_ram_mb:
            used_ratio = 1.0 - (snapshot.available_ram_mb / iec.resources.total_ram_mb)
            if used_ratio >= self.policy.memory_warn_ratio:
                result = result.with_warning(
                    f"memory pressure: {used_ratio * 100:.0f}% of RAM in use"
                )
                self._warn(iec.session_id, "memory_pressure", round(used_ratio, 3))

        # An out-of-process backend holds the model elsewhere, so this process's
        # RSS excludes the thing that dominates it. Reporting 32 MB for a run
        # backed by a 1.2 GB model is not a small discrepancy: efficiency is 20%
        # of the ADTC score, and a figure that omits the model measures nothing
        # about the model.
        backend = (result.trace.get("runtime", {}) or {}).get("backend", "")
        if backend in ("llama-server", "llama-cli"):
            result = result.with_warning(
                f"peak RSS covers this process only -- the model runs under "
                f"{backend}, so its memory is not counted here. The official "
                "profiler measures the model directly; use it for efficiency")
        return result

    def _warn(self, session_id: str, kind: str, value) -> None:
        if self.bus:
            self.bus.emit(EventName.RESOURCE_WARNING, session_id,
                          capability="performance", kind=kind, value=value)
