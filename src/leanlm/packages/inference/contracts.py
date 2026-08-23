"""CAP-005 public contract: prompt assembly + model inference."""
from __future__ import annotations

from ...contracts.capability import (
    PIPELINE_INFERENCE, Capability, CapabilitySpec,
)
from ...contracts.dto import ModelResponse
from ...contracts.iec import InferenceExecutionContext, PipelineStage
from ...shared.errors import contract_error
from ...shared.events import EventBus, EventName
from ...shared.text import TokenCounter
from .backends import InferenceBackend
from .policies import PromptPolicy
from .services import PromptBuilder, prompt_efficiency
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-005",
    name="Inference Execution",
    package="packages/inference",
    pipeline=PIPELINE_INFERENCE,
    stages=(PipelineStage.PROMPT_ASSEMBLY.value, PipelineStage.MODEL_INFERENCE.value),
    requirement_ids=("FR-05",),
    adtc_criteria=("Accuracy", "Throughput", "Memory", "Thermal"),
    estimated_memory_mb=64.0,
    expected_duration_ms=4000.0,
    produces_metrics=METRICS,
)


class InferenceExecutionCapability(Capability):
    """Owns two consecutive stages: build the prompt, then call llama.cpp.

    The model never sees the documents, the pipeline or the session. It sees a
    prompt. That separation is what lets the model be swapped without touching
    a single other package (MSOB s.1).
    """

    def __init__(self, backend: InferenceBackend, *,
                 prompt_policy: PromptPolicy | None = None,
                 counter: TokenCounter | None = None,
                 bus: EventBus | None = None) -> None:
        super().__init__(SPEC)
        self.backend = backend
        self.builder = PromptBuilder(prompt_policy, counter)
        self.bus = bus

    def check_preconditions(self, iec: InferenceExecutionContext,
                            stage: PipelineStage) -> None:
        if stage is PipelineStage.MODEL_INFERENCE and iec.prompt is None:
            raise contract_error(
                "INF-001", "inference requires an assembled prompt",
                capability=self.spec.name, stage=stage.value,
                recommended_action="run the prompt_assembly stage first",
            )

    def run(self, iec: InferenceExecutionContext,
            stage: PipelineStage) -> InferenceExecutionContext:
        if stage is PipelineStage.PROMPT_ASSEMBLY:
            return self._assemble(iec)
        return self._infer(iec)

    # -- stages -------------------------------------------------------------
    def _assemble(self, iec: InferenceExecutionContext) -> InferenceExecutionContext:
        prompt = self.builder.build(iec.question, iec.evidence)
        trace = dict(iec.trace)
        trace["prompt"] = {
            "template": f"{prompt.template_id}@{prompt.template_version}",
            "prompt_tokens": prompt.prompt_tokens,
            "context_tokens": prompt.context_tokens,
            "prompt_efficiency": prompt_efficiency(prompt),
            "labels": list(prompt.evidence_labels),
        }
        if self.bus:
            self.bus.emit(EventName.PROMPT_BUILT, iec.session_id, capability="inference",
                          prompt_tokens=prompt.prompt_tokens)
        result = iec.evolve(prompt=prompt, trace=trace)
        budget = iec.optimized_context.budget if iec.optimized_context else None
        if budget is not None:
            ceiling = budget.max_context_tokens - budget.reserved_output_tokens
            if prompt.prompt_tokens > ceiling:
                result = result.with_warning(
                    f"prompt ({prompt.prompt_tokens} tk) exceeds the usable window "
                    f"({ceiling} tk); llama.cpp will truncate"
                )
        return result

    def _infer(self, iec: InferenceExecutionContext) -> InferenceExecutionContext:
        session = iec.session_id
        if self.bus:
            self.bus.emit(EventName.INFERENCE_STARTED, session, capability="inference",
                          backend=self.backend.name)

        def on_first_token(latency_ms: float) -> None:
            if self.bus:
                self.bus.emit(EventName.FIRST_TOKEN, session, capability="inference",
                              latency_ms=latency_ms)

        # A slow backend that shows nothing is indistinguishable from a hung
        # one, and the user's only recourse is Ctrl-C. When a token hook is
        # installed the output is echoed as it arrives.
        extra = {}
        if getattr(self, "on_token", None) is not None:
            extra["on_token"] = self.on_token
        if getattr(self, "on_progress", None) is not None:
            extra["on_progress"] = self.on_progress
        try:
            generation = self.backend.generate(
                iec.prompt.rendered, on_first_token=on_first_token, **extra)
        except TypeError:
            # A backend that does not stream is not a failure.
            generation = self.backend.generate(iec.prompt.rendered,
                                               on_first_token=on_first_token)
        response = ModelResponse(
            meta=ModelResponse.build_meta(
                identity=(self.backend.binding.model_id, generation.text[:200]),
                parent_id=iec.prompt.artifact_id, stage="model_inference",
            ),
            text=generation.text,
            backend=generation.backend,
            model_id=self.backend.binding.model_id,
            model_checksum=self.backend.binding.expected_checksum,
            generated_tokens=generation.generated_tokens,
            prompt_tokens=generation.prompt_tokens,
            first_token_latency_ms=generation.first_token_latency_ms,
            inference_ms=generation.inference_ms,
            tokens_per_second=generation.tokens_per_second,
            stop_reason=generation.stop_reason,
            is_simulated=generation.is_simulated,
            truncated=generation.truncated,
        ).sealed()
        if self.bus:
            self.bus.emit(EventName.INFERENCE_FINISHED, session, capability="inference",
                          tokens=generation.generated_tokens,
                          tokens_per_second=generation.tokens_per_second)
        trace = dict(iec.trace)
        trace["inference"] = {
            "backend": generation.backend,
            "generated_tokens": generation.generated_tokens,
            "tokens_per_second": generation.tokens_per_second,
            "first_token_latency_ms": generation.first_token_latency_ms,
            "is_simulated": generation.is_simulated,
        }
        result = iec.evolve(response=response, trace=trace)  # type: ignore[arg-type]
        if generation.is_simulated:
            result = result.with_warning(
                "SIMULATED BACKEND: no GGUF model was loaded; this answer is an "
                "extract of the top passage and these timings are not model timings"
            )
        return result

    def check_postconditions(self, iec: InferenceExecutionContext,
                             stage: PipelineStage) -> None:
        if stage is PipelineStage.MODEL_INFERENCE and iec.response is None:
            raise contract_error("INF-002", "inference produced no response",
                                 capability=self.spec.name, stage=stage.value)
