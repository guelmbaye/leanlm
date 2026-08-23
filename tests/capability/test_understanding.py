"""CAP-002 Document Understanding -- FR-02."""
from __future__ import annotations

from leanlm.packages.understanding import SPEC, Segmenter, StructureAnalyzer
from leanlm.packages.understanding.policies import SegmentationPolicy
from leanlm.packages.understanding.services import build_structured_document, detect_language


class TestSpec:
    def test_identity(self):
        assert SPEC.capability_id == "CAP-002"
        assert "segmentation" in SPEC.stages


class TestStructure:
    def test_detects_atx_headings(self):
        blocks = StructureAnalyzer().analyze("# Titre\n\nUn paragraphe.\n")
        assert any(b.kind == "heading" for b in blocks)

    def test_detects_tables(self):
        text = "| a | b |\n|---|---|\n| 1 | 2 |\n"
        assert any(b.kind == "table" for b in StructureAnalyzer().analyze(text))

    def test_plain_paragraph_is_not_a_heading(self):
        blocks = StructureAnalyzer().analyze("Ceci est une phrase ordinaire.\n")
        assert all(b.kind != "heading" for b in blocks)


class TestSegmentation:
    def test_units_respect_the_maximum(self, structured):
        policy = SegmentationPolicy()
        assert all(u.token_estimate <= policy.max_tokens * 1.15 for u in structured.units)

    def test_section_path_is_recorded(self, structured):
        assert any(u.section_path for u in structured.units)

    def test_a_paragraph_never_absorbs_the_heading_that_follows(self, structured):
        for unit in structured.units:
            assert not unit.text.strip().endswith("##")

    def test_units_are_ordered_and_unique(self, structured):
        orders = [u.order for u in structured.units]
        assert orders == sorted(orders)
        assert len({u.unit_id for u in structured.units}) == len(structured.units)

    def test_segmentation_is_deterministic(self, sample_file):
        """Same bytes in, same unit ids out -- the basis of every reproducibility claim."""
        text = sample_file.read_text(encoding="utf-8")
        first = build_structured_document(text, filename="a.md", descriptor_id="d1")
        second = build_structured_document(text, filename="a.md", descriptor_id="d1")
        assert [u.unit_id for u in first.units] == [u.unit_id for u in second.units]
        assert first.meta.checksum == second.meta.checksum


class TestLanguage:
    def test_detects_french(self):
        assert detect_language("Les notes de frais sont remboursees sous trente jours.") == "fr"

    def test_detects_english(self):
        assert detect_language("The expense report is reimbursed within thirty days.") == "en"
