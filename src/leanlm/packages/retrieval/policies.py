"""Retrieval policies (IIB s.9)."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RetrievalPolicy:
    bm25_k1: float = 1.4
    bm25_b: float = 0.72
    candidate_pool: int = 40
    # Structural signals cost nothing at query time and encode real editorial
    # intent: a heading that repeats the question is a strong hint.
    section_match_boost: float = 0.22
    title_match_boost: float = 0.10
    table_boost: float = 0.06
    exact_phrase_boost: float = 0.18
    numeric_question_boost: float = 0.12
    # Diversity: two passages saying the same thing waste half the budget.
    diversity_lambda: float = 0.35
    max_per_document: int = 4
    min_score: float = 0.02
    # A passage scoring far below the best one buys almost nothing and still
    # costs its tokens. IP-01: never send more context than necessary.
    relative_score_floor: float = 0.15
    # Share of a question's content words that may be absent from the corpus
    # before the question is treated as off-topic. Above this, matching on the
    # words that remain is coincidence, not relevance.
    #
    # Tuned at 0.4 on a French probe set, this refused legitimate English
    # questions until generic framing words (`long`, `last`, `ends`) were
    # excluded from the ratio. Raising the threshold instead was tried and does
    # not work: no single value accepts "how long does the confidentiality
    # obligation last" (0.57 before the fix) while rejecting "reimbursement of
    # children's school fees" (0.40). The ratio alone cannot separate them --
    # what separates them is whether the *subject* words are supported.
    #
    # Still the most corpus-dependent number in the system. Re-measure with
    # `leanlm accuracy` in both languages whenever the corpus changes.
    unsupported_term_ratio: float = 0.4

    @classmethod
    def from_config(cls, config: dict) -> "RetrievalPolicy":
        node = (config or {}).get("retrieval", {})
        base = cls()
        return cls(**{f: type(getattr(base, f))(node.get(f, getattr(base, f)))
                      for f in base.__slots__})
