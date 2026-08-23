"""CAP-004 Evidence Retrieval -- select the passages worth their tokens (FR-04)."""

from .contracts import SPEC, EvidenceRetrievalCapability
from .services import BM25Index, EvidenceSelector

__all__ = ["SPEC", "EvidenceRetrievalCapability", "BM25Index", "EvidenceSelector"]
