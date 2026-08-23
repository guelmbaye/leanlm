"""CAP-002 Document Understanding -- structure, segmentation, metadata (FR-02)."""

from .contracts import SPEC, DocumentUnderstandingCapability
from .services import StructureAnalyzer, Segmenter

__all__ = ["SPEC", "DocumentUnderstandingCapability", "StructureAnalyzer", "Segmenter"]
