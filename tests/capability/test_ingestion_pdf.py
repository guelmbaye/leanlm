"""CAP-001 PDF ingestion -- the format the competition corpus actually uses.

Everything else in the suite reads Markdown, which is a convenience for testing
and a poor proxy for reality: enterprise documents arrive as PDFs, and PDF is
where extraction goes wrong. These tests build real PDFs byte by byte rather
than shipping binary fixtures, so what is being tested is visible in the source.

The important case is the last one: a PDF with no extractable text at all. A
scanned policy document is the most likely real-world input to fail, and the
required behaviour is an explicit refusal, never an empty document indexed as if
it had content.
"""
from __future__ import annotations

import pytest

from leanlm.packages.ingestion import DocumentLoader
from leanlm.packages.ingestion.services import extract_text
from leanlm.shared.errors import LeanLMError

TEXT_LINES = [
    "Politique de remboursement des notes de frais",
    "Les notes de frais validees sont remboursees sous 30 jours ouvres.",
    "Le plafond pour un repas du midi est de 25 EUR par personne.",
    "Les amendes routieres ne sont jamais remboursees.",
]


def _pdf(lines: list[str], *, with_text: bool = True) -> bytes:
    """Build a single-page PDF with a correct cross-reference table.

    Written out longhand because a fixture file would hide what is being parsed,
    and because the failure mode under test (no extractable text) is defined by
    the absence of a content stream operator rather than by a corrupt file.
    """
    if with_text:
        operators = ["BT", "/F1 12 Tf", "72 720 Td", "14 TL"]
        for index, line in enumerate(lines):
            escaped = line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")
            operators.append(f"({escaped}) Tj" if index == 0 else f"T* ({escaped}) Tj")
        operators.append("ET")
        content = "\n".join(operators).encode("latin-1")
    else:
        # A filled rectangle: a valid page that carries no text, which is what a
        # scanned document looks like to a text extractor.
        content = b"0.5 g\n72 600 468 120 re f\n"

    objects: list[bytes] = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content
        + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n".encode() + body + b"\nendobj\n"

    xref_at = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n").encode()
    return bytes(out)


@pytest.fixture
def text_pdf(tmp_path):
    path = tmp_path / "politique.pdf"
    path.write_bytes(_pdf(TEXT_LINES))
    return path


@pytest.fixture
def scanned_pdf(tmp_path):
    path = tmp_path / "scan.pdf"
    path.write_bytes(_pdf([], with_text=False))
    return path


class TestFormatDetection:
    def test_a_pdf_is_recognised_by_content_not_only_by_suffix(self, text_pdf, tmp_path):
        from leanlm.contracts.dto import DocumentFormat
        misnamed = tmp_path / "policy.bin"
        misnamed.write_bytes(text_pdf.read_bytes())
        detected = extract_text(text_pdf, DocumentFormat.PDF)
        assert detected.text
        assert text_pdf.read_bytes().startswith(b"%PDF")


class TestExtraction:
    def test_the_text_comes_back(self, text_pdf):
        loader = DocumentLoader()
        raw = loader.load(loader.describe(text_pdf))
        assert "30 jours ouvres" in raw.text
        assert "25 EUR" in raw.text

    def test_the_extraction_method_is_recorded(self, text_pdf):
        """Which extractor ran changes the text; a measurement attributed to the
        wrong one is not reproducible."""
        loader = DocumentLoader()
        raw = loader.load(loader.describe(text_pdf))
        assert raw.extraction_method
        assert raw.extraction_method != "unknown"

    def test_the_page_count_is_reported(self, text_pdf):
        loader = DocumentLoader()
        raw = loader.load(loader.describe(text_pdf))
        assert raw.page_count >= 1

    def test_extraction_is_deterministic(self, text_pdf):
        loader = DocumentLoader()
        descriptor = loader.describe(text_pdf)
        assert loader.load(descriptor).text == loader.load(descriptor).text


class TestSegmentationOfPdfText:
    def test_the_figures_survive_into_units(self, text_pdf):
        from leanlm.contracts.ddp import DocumentPipelineState
        from leanlm.packages.ingestion import DocumentIngestionCapability
        from leanlm.packages.understanding import DocumentUnderstandingCapability

        state = DocumentPipelineState(source_path=str(text_pdf))
        state = DocumentIngestionCapability().process(state)
        state = DocumentUnderstandingCapability().process(state)
        joined = " ".join(unit.text for unit in state.structured.units)
        assert "30 jours" in joined
        assert "25 EUR" in joined


class TestScannedDocuments:
    def test_a_pdf_without_text_is_refused_not_indexed_empty(self, scanned_pdf):
        """The most likely real-world failure: a scanned policy document. An empty
        document silently added to the corpus would make every later answer about
        it a false negative that nobody can explain."""
        loader = DocumentLoader()
        with pytest.raises(LeanLMError) as excinfo:
            loader.load(loader.describe(scanned_pdf))
        record = excinfo.value.record
        assert record.code.startswith("ING-")
        assert "scan" in record.recommended_action.lower()

    def test_the_runtime_reports_it_as_a_failed_file_and_continues(
            self, tmp_path, scanned_pdf):
        """One unreadable document must not abort a directory ingestion."""
        from leanlm.runtime.corpus import CorpusStore
        from leanlm.runtime.runtime import LeanLMRuntime

        good = tmp_path / "lisible.md"
        good.write_text("# Politique\n\n" + "\n".join(TEXT_LINES) + "\n", encoding="utf-8")

        store = CorpusStore(tmp_path / "mixed.sqlite3")
        engine = LeanLMRuntime("development", corpus=store, backend="simulated",
                               verify_model=False)
        try:
            result = engine.ingest([tmp_path])
            assert len(result["ingested"]) == 1
            assert len(result["failed"]) == 1
            assert result["failed"][0]["file"] == scanned_pdf.name
            assert result["failed"][0]["action"]
        finally:
            engine.close()
