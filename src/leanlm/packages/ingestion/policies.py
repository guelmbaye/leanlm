"""Ingestion policies (DDPB s.4, s.7).

Thresholds live here, never inline in the services. Values are overridable
through ``configs/ingestion.yaml``.
"""
from __future__ import annotations

from dataclasses import dataclass

from ...contracts.dto import DocumentFormat

SUPPORTED_EXTENSIONS: dict[str, DocumentFormat] = {
    ".txt": DocumentFormat.TXT,
    ".text": DocumentFormat.TXT,
    ".md": DocumentFormat.MARKDOWN,
    ".markdown": DocumentFormat.MARKDOWN,
    ".pdf": DocumentFormat.PDF,
}


@dataclass(frozen=True, slots=True)
class IngestionPolicy:
    max_file_size_mb: float = 25.0
    min_extractable_chars: int = 40
    max_documents: int = 500
    reject_binary_ratio: float = 0.15

    @classmethod
    def from_config(cls, config: dict) -> "IngestionPolicy":
        node = (config or {}).get("ingestion", {})
        return cls(
            max_file_size_mb=float(node.get("max_file_size_mb", 25.0)),
            min_extractable_chars=int(node.get("min_extractable_chars", 40)),
            max_documents=int(node.get("max_documents", 500)),
            reject_binary_ratio=float(node.get("reject_binary_ratio", 0.15)),
        )
