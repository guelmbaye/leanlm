"""CAP-006 Response Validation -- FR-06.

The tests that matter here are the ones that fail loudly when the model invents
something. Grounding is the product's core claim; if these pass while the model
hallucinates, the claim is worthless.
"""
from __future__ import annotations

from leanlm.contracts.dto import ConfidenceLevel, ModelResponse, SelectedEvidence
from leanlm.packages.validation import SPEC, ResponseValidator
from leanlm.packages.validation.policies import ValidationPolicy


def _evidence(label: str, text: str):
    return SelectedEvidence(
        meta=SelectedEvidence.build_meta(identity=(label, text[:40]),
                                         stage="evidence_retrieval"),
        label=label, unit_id=f"u-{label}", document_name="politique.md",
        text=text, section_path=("2. Delais",), score=1.0, token_estimate=30,
    ).sealed()


def _response(text: str):
    return ModelResponse(
        meta=ModelResponse.build_meta(identity=(text[:40],), stage="model_inference"),
        text=text, generated_tokens=len(text.split()), is_simulated=True,
    ).sealed()


EVIDENCE = (
    _evidence("S1", "Les notes de frais validees sont remboursees sous 30 jours ouvres."),
    _evidence("S2", "Le plafond pour un repas du midi est de 25 EUR par personne."),
)
QUESTION = "Quel est le delai de remboursement des notes de frais ?"


class TestSpec:
    def test_identity(self):
        assert SPEC.capability_id == "CAP-006"
        assert SPEC.stages == ("response_validation",)


class TestGrounding:
    def test_a_faithful_cited_answer_passes(self):
        report = ResponseValidator().validate(
            _response("Les notes de frais sont remboursees sous 30 jours ouvres. [S1]"),
            EVIDENCE, QUESTION)
        assert report.passed
        assert report.grounding_rate > 0.5
        assert report.confidence in (ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM)

    def test_an_invented_answer_is_caught(self):
        """The failure that must never pass silently."""
        report = ResponseValidator().validate(
            _response("Le remboursement est effectue sous 90 jours par virement "
                      "SWIFT depuis la filiale luxembourgeoise. [S1]"),
            EVIDENCE, QUESTION)
        assert report.grounding_rate < 1.0
        assert report.warnings

    def test_a_fabricated_citation_label_is_detected(self):
        report = ResponseValidator().validate(
            _response("Le delai est de 30 jours ouvres. [S7]"), EVIDENCE, QUESTION)
        assert report.checks.get("citations_valid") is False

    def test_an_uncited_answer_is_flagged(self):
        report = ResponseValidator().validate(
            _response("Les notes de frais sont remboursees sous 30 jours ouvres."),
            EVIDENCE, QUESTION)
        assert report.checks.get("has_citations") is False


class TestInsufficiency:
    def test_with_no_evidence_only_a_refusal_passes(self):
        validator = ResponseValidator()
        refusal = validator.validate(
            _response("Les documents fournis ne permettent pas de conclure."),
            (), "Quel est le cours de l'action Tesla ?")
        invented = validator.validate(
            _response("L'action Tesla cote 248 dollars."),
            (), "Quel est le cours de l'action Tesla ?")
        assert refusal.confidence is ConfidenceLevel.INSUFFICIENT
        assert refusal.passed
        assert not invented.passed

    def test_an_empty_answer_never_passes(self):
        report = ResponseValidator().validate(_response(""), EVIDENCE, QUESTION)
        assert not report.passed


class TestConfidence:
    def test_an_answer_unrelated_to_the_question_is_not_high_confidence(self):
        """A verbatim quote of the wrong passage scores a perfect grounding rate
        and answers nothing: grounding alone cannot buy high confidence."""
        report = ResponseValidator().validate(
            _response("Le plafond pour un repas du midi est de 25 EUR par personne. [S2]"),
            EVIDENCE, "Combien de jours de conges exceptionnels pour un mariage ?")
        assert report.confidence is not ConfidenceLevel.HIGH

    def test_validation_is_deterministic(self):
        validator = ResponseValidator(ValidationPolicy())
        response = _response("Remboursement sous 30 jours ouvres. [S1]")
        first = validator.validate(response, EVIDENCE, QUESTION)
        second = validator.validate(response, EVIDENCE, QUESTION)
        assert first.grounding_rate == second.grounding_rate
        assert first.confidence is second.confidence
