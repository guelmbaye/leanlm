"""CAP-003 public contract."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_INFERENCE, Capability, CapabilitySpec,
)
from ...contracts.iec import InferenceExecutionContext, PipelineStage
from ...shared.errors import contract_error
from ...shared.text import TokenCounter
from .policies import BudgetPolicy, OptimizationPolicy
from .services import ContextBudgeter, ContextOptimizer, IntentAnalyzer
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-003",
    name="Context Optimization",
    package="packages/context",
    pipeline=PIPELINE_INFERENCE,
    stages=(PipelineStage.CONTEXT_OPTIMIZATION.value,),
    requirement_ids=("FR-03",),
    adtc_criteria=("Memory", "Throughput", "Accuracy"),
    estimated_memory_mb=24.0,
    expected_duration_ms=30.0,
    produces_metrics=METRICS,
)


class ContextOptimizationCapability(Capability):
    """Intent -> budget -> compaction. Nothing reaches the model yet."""

    def __init__(self, *, budget_policy: BudgetPolicy | None = None,
                 optimization_policy: OptimizationPolicy | None = None,
                 counter: TokenCounter | None = None,
                 model_context_tokens: int = 4096,
                 max_output_tokens: int = 512,
                 model_footprint_mb: float = 0.0) -> None:
        super().__init__(SPEC)
        self.analyzer = IntentAnalyzer()
        self.budgeter = ContextBudgeter(budget_policy, counter)
        self.optimizer = ContextOptimizer(optimization_policy, counter)
        self.model_context_tokens = model_context_tokens
        self.max_output_tokens = max_output_tokens
        self.model_footprint_mb = model_footprint_mb

    def check_preconditions(self, iec: InferenceExecutionContext,
                            stage: PipelineStage) -> None:
        if iec.context_package is None:
            raise contract_error(
                "CTX-001", "context optimization requires a loaded ContextPackage",
                capability=self.spec.name, stage=self.spec.stages[0],
                recommended_action="run document discovery first",
            )

    def run(self, iec: InferenceExecutionContext,
            stage: PipelineStage) -> InferenceExecutionContext:
        intent = self.analyzer.analyze(iec.question)
        budget = self.budgeter.compute(
            model_context_tokens=self.model_context_tokens,
            max_output_tokens=self.max_output_tokens,
            question=iec.question,
            intent=intent,
            resources=iec.resources,
            model_footprint_mb=self.model_footprint_mb,
        )
        optimized = self.optimizer.optimize(iec.context_package, budget)
        trace = dict(iec.trace)
        trace["context"] = {
            "intent": intent.intent.value,
            "budget_tokens": budget.evidence_tokens,
            "limiting_factor": budget.limiting_factor,
            "compression_ratio": optimized.compression_ratio,
            "operations": list(optimized.operations),
        }
        return iec.evolve(intent=intent, optimized_context=optimized, trace=trace)

    def check_postconditions(self, iec: InferenceExecutionContext,
                             stage: PipelineStage) -> None:
        optimized = iec.optimized_context
        if optimized is None or optimized.budget is None:
            raise contract_error("CTX-002", "optimization produced no budget",
                                 capability=self.spec.name)
        if optimized.output_tokens > optimized.input_tokens:
            raise contract_error(
                "CTX-003", "optimization increased the context size",
                capability=self.spec.name,
                recommended_action="review the merge policy; IP-01 forbids growth",
            )
