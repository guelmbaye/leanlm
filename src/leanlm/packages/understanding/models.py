"""Package-local models for CAP-002."""
from __future__ import annotations

from dataclasses import dataclass

from ...contracts.dto import UnitKind


@dataclass(frozen=True, slots=True)
class Block:
    """A structural block detected in the normalized text."""

    kind: UnitKind
    text: str
    char_start: int
    char_end: int
    heading_level: int = 0
