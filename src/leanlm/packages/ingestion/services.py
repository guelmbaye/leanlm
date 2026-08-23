"""Ingestion services: import, validate, normalize (DDPB stages 1-3).

PDF extraction is pluggable and tried in order of fidelity:
  1. ``pypdf``       -- pure Python, most common on laptops;
  2. ``pdfminer.six``-- better layout handling for column-heavy documents;
  3. ``pdftotext``   -- poppler binary, present on most Linux installs.

If none is available the document is rejected with an actionable error rather
than silently ingesting an empty text -- an empty corpus is worse than a clear
failure, because it produces confident answers with no evidence behind them.
"""
from __future__ import annotations

import datetime as _dt
import shutil
import subprocess
import tempfile
from pathlib import Path

from ...shared.errors import input_error
from ...shared.hashing import sha256_file
from ...shared.text import normalize_text
from ...contracts.dto import DocumentDescriptor, DocumentFormat, RawDocument
from .models import ExtractionResult
from .policies import SUPPORTED_EXTENSIONS, IngestionPolicy


def detect_format(path: Path) -> DocumentFormat:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise input_error(
            "ING-001", f"unsupported document format: {suffix or '(none)'}",
            capability="ingestion", stage="validation",
            recommended_action="convert the file to PDF, TXT or Markdown",
            path=str(path),
        )
    return SUPPORTED_EXTENSIONS[suffix]


def _extract_pdf(path: Path) -> ExtractionResult:
    warnings: list[str] = []
    try:
        from pypdf import PdfReader  # type: ignore

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(pages)
        if text.strip():
            return ExtractionResult(text, "pypdf", len(pages), tuple(warnings))
        warnings.append("pypdf returned no text; trying pdfminer")
    except ImportError:
        warnings.append("pypdf unavailable")
    except Exception as exc:
        warnings.append(f"pypdf failed: {type(exc).__name__}")

    try:
        from pdfminer.high_level import extract_text as _pdfminer_extract  # type: ignore

        text = _pdfminer_extract(str(path)) or ""
        if text.strip():
            return ExtractionResult(text, "pdfminer.six", text.count("\f") + 1,
                                    tuple(warnings))
        warnings.append("pdfminer returned no text; trying pdftotext")
    except ImportError:
        warnings.append("pdfminer.six unavailable")
    except Exception as exc:
        warnings.append(f"pdfminer failed: {type(exc).__name__}")

    if shutil.which("pdftotext"):
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
            out = Path(tmp.name)
        try:
            subprocess.run(["pdftotext", "-layout", str(path), str(out)],
                           check=True, capture_output=True, timeout=120)
            text = out.read_text(encoding="utf-8", errors="replace")
            if text.strip():
                return ExtractionResult(text, "pdftotext", text.count("\f") + 1,
                                        tuple(warnings))
        except Exception as exc:
            warnings.append(f"pdftotext failed: {type(exc).__name__}")
        finally:
            out.unlink(missing_ok=True)

    raise input_error(
        "ING-002", f"no usable text could be extracted from {path.name}",
        capability="ingestion", stage="import",
        recommended_action=(
            "install one of: pypdf, pdfminer.six, poppler-utils; "
            "scanned PDFs need OCR before ingestion"
        ),
        attempts="; ".join(warnings) or "none",
    )


def extract_text(path: Path, doc_format: DocumentFormat) -> ExtractionResult:
    if doc_format is DocumentFormat.PDF:
        return _extract_pdf(path)
    raw = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return ExtractionResult(raw.decode(encoding), f"text:{encoding}", 1)
        except UnicodeDecodeError:
            continue
    raise input_error(
        "ING-003", f"unreadable text encoding in {path.name}",
        capability="ingestion", stage="validation",
        recommended_action="re-save the file as UTF-8",
    )


class DocumentLoader:
    """Stages 1-3 of the DDPB. The source file is opened read-only (DP-02)."""

    def __init__(self, policy: IngestionPolicy | None = None) -> None:
        self.policy = policy or IngestionPolicy()

    def describe(self, path: str | Path) -> DocumentDescriptor:
        target = Path(path).expanduser().resolve()
        if not target.is_file():
            raise input_error("ING-004", f"document not found: {target}",
                              capability="ingestion", stage="import",
                              recommended_action="check the path")
        doc_format = detect_format(target)
        size = target.stat().st_size
        if size == 0:
            raise input_error("ING-005", f"empty document: {target.name}",
                              capability="ingestion", stage="validation",
                              recommended_action="remove the file from the corpus")
        if size > self.policy.max_file_size_mb * 1024 * 1024:
            raise input_error(
                "ING-006",
                f"document exceeds the MVP size limit "
                f"({size / 1048576:.1f} MB > {self.policy.max_file_size_mb} MB)",
                capability="ingestion", stage="validation",
                recommended_action="split the document before ingestion",
            )
        checksum = sha256_file(target)
        modified = _dt.datetime.fromtimestamp(
            target.stat().st_mtime, tz=_dt.timezone.utc
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        descriptor = DocumentDescriptor(
            meta=DocumentDescriptor.build_meta(
                identity=(checksum, target.name), source=str(target), stage="import"
            ),
            path=str(target),
            filename=target.name,
            doc_format=doc_format,
            size_bytes=size,
            file_checksum=checksum,
            modified_at=modified,
        )
        return descriptor.sealed()  # type: ignore[return-value]

    def load(self, descriptor: DocumentDescriptor) -> RawDocument:
        result = extract_text(Path(descriptor.path), descriptor.doc_format)
        if len(result.text.strip()) < self.policy.min_extractable_chars:
            raise input_error(
                "ING-007",
                f"document {descriptor.filename} carries too little text "
                f"({len(result.text.strip())} chars)",
                capability="ingestion", stage="validation",
                recommended_action="check that the PDF is not a scanned image",
            )
        raw = RawDocument(
            meta=RawDocument.build_meta(
                identity=(descriptor.artifact_id, result.method),
                parent_id=descriptor.artifact_id,
                source=descriptor.path,
                stage="import",
            ),
            descriptor_id=descriptor.artifact_id,
            filename=descriptor.filename,
            text=result.text,
            extraction_method=result.method,
            page_count=result.page_count,
        )
        return raw.sealed()  # type: ignore[return-value]

    def normalize(self, raw: RawDocument) -> str:
        """DP-04: uniform line breaks and control characters, meaning untouched."""
        return normalize_text(raw.text)
