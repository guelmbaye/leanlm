"""Package-local models for ingestion."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    text: str
    method: str
    page_count: int = 0
    warnings: tuple[str, ...] = ()
