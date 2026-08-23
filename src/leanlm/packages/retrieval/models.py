"""Package-local models for CAP-004."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredUnit:
    index: int
    lexical: float
    boost: float

    @property
    def total(self) -> float:
        """Structural boosts scale the lexical score rather than being added to it.

        Added as constants they were 5% of a BM25 score of 5 on a small corpus,
        and proportionally less as the corpus grew -- so "this passage sits under
        a heading the question names" faded to nothing exactly when the corpus
        got large enough to need it. Expressed as a fraction, a 0.22 section
        match means +22% at any scale.
        """
        return self.lexical * (1.0 + self.boost)
