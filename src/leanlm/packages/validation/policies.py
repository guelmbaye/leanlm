"""Validation and hallucination policies (IIB s.11, s.12, s.13)."""
from __future__ import annotations

from dataclasses import dataclass

INSUFFICIENT_MARKERS: tuple[str, ...] = (
    "ne permettent pas de conclure",
    "n'est pas presente dans le corpus",
    "n'est pas présente dans le corpus",
    "verification complementaire est necessaire",
    "vérification complémentaire est nécessaire",
    "do not allow a conclusion",
    "not present in the available corpus",
    "further verification is required",
    "i don't know",
    "je ne sais pas",
)

HEDGE_MARKERS: tuple[str, ...] = (
    "il semble", "probablement", "peut-etre", "peut-être", "il est possible",
    "apparently", "it seems", "possibly", "likely", "i believe",
)


@dataclass(frozen=True, slots=True)
class ValidationPolicy:
    """Thresholds are policy, not opinion: they are benchmarked (EBPB E5)."""

    sentence_grounding_threshold: float = 0.34
    grounding_pass_threshold: float = 0.60
    min_answer_chars: int = 2
    require_citations: bool = True
    high_confidence_grounding: float = 0.85
    medium_confidence_grounding: float = 0.60
    shingle_size: int = 3

    @classmethod
    def from_config(cls, config: dict) -> "ValidationPolicy":
        node = (config or {}).get("validation", {})
        base = cls()
        return cls(
            sentence_grounding_threshold=float(node.get(
                "sentence_grounding_threshold", base.sentence_grounding_threshold)),
            grounding_pass_threshold=float(node.get(
                "grounding_pass_threshold", base.grounding_pass_threshold)),
            min_answer_chars=int(node.get("min_answer_chars", base.min_answer_chars)),
            require_citations=bool(node.get("require_citations", base.require_citations)),
            high_confidence_grounding=float(node.get(
                "high_confidence_grounding", base.high_confidence_grounding)),
            medium_confidence_grounding=float(node.get(
                "medium_confidence_grounding", base.medium_confidence_grounding)),
            shingle_size=int(node.get("shingle_size", base.shingle_size)),
        )
