"""CAP-001 Document Ingestion -- FR-01."""
from __future__ import annotations

import pytest

from leanlm.contracts.ddp import DocumentPipelineState, DocumentState
from leanlm.packages.ingestion import SPEC, DocumentIngestionCapability, DocumentLoader
from leanlm.shared.errors import LeanLMError


class TestSpec:
    def test_declares_its_stages_and_requirement(self):
        assert SPEC.capability_id == "CAP-001"
        assert "import" in SPEC.stages
        assert "FR-01" in SPEC.requirement_ids


class TestLoading:
    def test_reads_a_markdown_file(self, sample_file):
        loader = DocumentLoader()
        descriptor = loader.describe(sample_file)
        assert descriptor.file_checksum
        assert descriptor.size_bytes > 0
        raw = loader.load(descriptor)
        assert "note de frais" in raw.text

    def test_normalization_is_applied(self, tmp_path):
        path = tmp_path / "crlf.txt"
        path.write_bytes(
            b"Premiere ligne du document interne.\r\n"
            b"Seconde ligne, avec un retour chariot Windows.\r\n"
        )
        loader = DocumentLoader()
        text = loader.normalize(loader.load(loader.describe(path)))
        assert "\r" not in text

    def test_missing_file_is_rejected_with_an_action(self, tmp_path):
        with pytest.raises(LeanLMError) as excinfo:
            DocumentLoader().describe(tmp_path / "absent.md")
        assert excinfo.value.record.recommended_action

    def test_unsupported_format_is_rejected(self, tmp_path):
        path = tmp_path / "archive.zip"
        path.write_bytes(b"PK\x03\x04")
        with pytest.raises(LeanLMError):
            DocumentLoader().describe(path)

    def test_empty_document_is_rejected_rather_than_silently_indexed(self, tmp_path):
        path = tmp_path / "blank.txt"
        path.write_text("   \n\n  ", encoding="utf-8")
        loader = DocumentLoader()
        with pytest.raises(LeanLMError):
            loader.normalize(loader.load(loader.describe(path)))


class TestCapability:
    def test_advances_the_document_state(self, sample_file):
        state = DocumentPipelineState(source_path=str(sample_file))
        result = DocumentIngestionCapability().process(state)
        assert result.state in (DocumentState.NORMALIZED, DocumentState.VALIDATED)
        assert result.normalized_text
        assert result.descriptor is not None

    def test_checksum_is_stable_across_reads(self, sample_file):
        loader = DocumentLoader()
        assert (loader.describe(sample_file).file_checksum
                == loader.describe(sample_file).file_checksum)
