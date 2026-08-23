"""Prompt and generation policies (IIB s.10, s.12)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptPolicy:
    template_id: str = "leanlm.enterprise.grounded"
    template_version: str = "1.2.0"
    cite_evidence: bool = True
    evidence_separator: str = "\n\n"
    max_evidence_chars: int = 20000
    include_section_path: bool = True

    @classmethod
    def from_config(cls, config: dict) -> "PromptPolicy":
        node = (config or {}).get("prompt", {})
        base = cls()
        return cls(
            template_id=str(node.get("template_id", base.template_id)),
            template_version=str(node.get("template_version", base.template_version)),
            cite_evidence=bool(node.get("cite_evidence", base.cite_evidence)),
            evidence_separator=str(node.get("evidence_separator", base.evidence_separator)),
            max_evidence_chars=int(node.get("max_evidence_chars", base.max_evidence_chars)),
            include_section_path=bool(node.get("include_section_path",
                                               base.include_section_path)),
        )


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    repeat_penalty: float = 1.1
    max_output_tokens: int = 512
    seed: int = 20260601
    stop: tuple[str, ...] = ("</s>", "<|im_end|>", "<|eot_id|>", "\nQuestion:")
    timeout_s: float = 300.0

    @classmethod
    def from_config(cls, config: dict) -> "GenerationPolicy":
        node = (config or {}).get("inference", {})
        base = cls()
        stop = node.get("stop", base.stop)
        return cls(
            temperature=float(node.get("temperature", base.temperature)),
            top_k=int(node.get("top_k", base.top_k)),
            top_p=float(node.get("top_p", base.top_p)),
            repeat_penalty=float(node.get("repeat_penalty", base.repeat_penalty)),
            max_output_tokens=int(node.get("max_output_tokens", base.max_output_tokens)),
            seed=int(node.get("seed", base.seed)),
            stop=tuple(stop) if isinstance(stop, (list, tuple)) else base.stop,
            timeout_s=float(node.get("timeout_s", base.timeout_s)),
        )
