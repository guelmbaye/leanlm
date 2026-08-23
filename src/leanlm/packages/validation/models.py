"""Package-local models for CAP-006."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SentenceVerdict:
    text: str
    grounded: bool
    overlap: float
    best_label: str = ""
