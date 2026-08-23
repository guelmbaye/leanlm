"""CAP-005 Inference Execution -- prompt assembly + llama.cpp call (FR-05)."""

from .contracts import SPEC, InferenceExecutionCapability
from .backends import (
    InferenceBackend, LlamaCppBinaryBackend, LlamaCppPythonBackend,
    LlamaCppServerBackend, SimulatedBackend, build_backend,
)
from .services import PromptBuilder

__all__ = [
    "SPEC", "InferenceExecutionCapability", "InferenceBackend",
    "LlamaCppBinaryBackend", "LlamaCppPythonBackend", "LlamaCppServerBackend",
    "SimulatedBackend", "build_backend", "PromptBuilder",
]
