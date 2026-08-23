"""Package-local models for CAP-005."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class ModelBinding:
    """Which model this runtime is allowed to load (RPCB s.10)."""

    model_id: str
    path: str
    quantization: str = "unknown"
    context_tokens: int = 4096
    expected_checksum: str = ""
    parameters: str = ""
    license: str = ""

    def to_dict(self) -> dict:
        return {
            "model_id": self.model_id, "path": self.path,
            "quantization": self.quantization, "context_tokens": self.context_tokens,
            "expected_checksum": self.expected_checksum,
            "parameters": self.parameters, "license": self.license,
        }


@dataclass(frozen=True, slots=True)
class GenerationResult:
    text: str
    backend: str
    generated_tokens: int
    prompt_tokens: int
    first_token_latency_ms: float
    inference_ms: float
    stop_reason: str = "eos"
    is_simulated: bool = False
    extra: dict = field(default_factory=dict)

    @property
    def tokens_per_second(self) -> float:
        seconds = max(1e-6, (self.inference_ms - self.first_token_latency_ms) / 1000.0)
        return round(self.generated_tokens / seconds, 3) if self.generated_tokens else 0.0
