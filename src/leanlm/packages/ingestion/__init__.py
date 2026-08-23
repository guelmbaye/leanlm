"""CAP-001 Document Ingestion -- load and normalize local documents (FR-01)."""

from .contracts import DocumentIngestionCapability, SPEC
from .services import DocumentLoader, extract_text

__all__ = ["DocumentIngestionCapability", "SPEC", "DocumentLoader", "extract_text"]
