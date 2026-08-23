"""Segmentation policy (DDPB s.10).

The goal is not fixed-size chunks. It is *usable* units: a unit should be small
enough to be cheap in the context budget, and large enough to still answer a
question on its own. The thresholds below are the ones an evidence passage has
to satisfy to be worth spending tokens on.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SegmentationPolicy:
    target_tokens: int = 180
    max_tokens: int = 320
    min_tokens: int = 24
    merge_below_tokens: int = 90
    max_units_per_document: int = 4000
    # A bare heading is a poor evidence passage: it costs a retrieval slot and
    # answers nothing. Section titles are preserved on every unit instead.
    keep_headings_as_units: bool = False

    @classmethod
    def from_config(cls, config: dict) -> "SegmentationPolicy":
        node = (config or {}).get("segmentation", {})
        return cls(
            target_tokens=int(node.get("target_tokens", 180)),
            max_tokens=int(node.get("max_tokens", 320)),
            min_tokens=int(node.get("min_tokens", 24)),
            merge_below_tokens=int(node.get("merge_below_tokens", 90)),
            max_units_per_document=int(node.get("max_units_per_document", 4000)),
            keep_headings_as_units=bool(node.get("keep_headings_as_units", False)),
        )
