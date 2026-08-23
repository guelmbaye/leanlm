"""CAP-004 Evidence Retrieval -- FR-04."""
from __future__ import annotations

import pytest

from leanlm.contracts.dto import IntentType
from leanlm.packages.context import ContextBudgeter, ContextOptimizer, IntentAnalyzer
from leanlm.packages.retrieval import SPEC, BM25Index, EvidenceSelector
from leanlm.packages.retrieval.policies import RetrievalPolicy


@pytest.fixture
def optimized(corpus, structured, sample_file):
    from leanlm.shared.clock import iso_now
    corpus.upsert_document(structured, source_path=str(sample_file),
                           file_checksum="cafe", ingested_at=iso_now())
    package = corpus.context_package()
    question = "Quel est le plafond pour un repas du midi ?"
    budget = ContextBudgeter().compute(
        model_context_tokens=4096, max_output_tokens=512, question=question,
        intent=IntentAnalyzer().analyze(question), resources=None)
    return ContextOptimizer().optimize(package, budget), budget


class TestSpec:
    def test_identity(self):
        assert SPEC.capability_id == "CAP-004"
        assert SPEC.stages == ("evidence_retrieval",)


class TestBM25:
    def _units(self, corpus, structured, sample_file):
        from leanlm.shared.clock import iso_now
        corpus.upsert_document(structured, source_path=str(sample_file),
                               file_checksum="cafe", ingested_at=iso_now())
        return list(corpus.iter_units())

    def test_the_passage_carrying_the_term_ranks_first(self, corpus, structured,
                                                       sample_file):
        units = self._units(corpus, structured, sample_file)
        scores = BM25Index(units).score(["amendes", "routieres"])
        best = units[max(range(len(units)), key=lambda i: scores[i])]
        assert "amendes" in best.text.lower()

    def test_an_unrelated_query_scores_zero_everywhere(self, corpus, structured,
                                                       sample_file):
        units = self._units(corpus, structured, sample_file)
        scores = BM25Index(units).score(["cryptomonnaie", "blockchain", "satoshi"])
        assert max(scores) == 0.0


class TestSelection:
    def test_selection_stays_inside_the_budget(self, optimized):
        context, budget = optimized
        evidence, _ = EvidenceSelector().select(
            "Quel est le plafond pour un repas du midi ?", context)
        assert sum(e.token_estimate for e in evidence) <= budget.evidence_tokens

    def test_evidence_is_labelled_and_traceable(self, optimized):
        context, _budget = optimized
        evidence, _ = EvidenceSelector().select(
            "Quel est le plafond pour un repas du midi ?", context)
        assert evidence
        for item in evidence:
            assert item.label.startswith("S")
            assert item.unit_id and item.document_name

    def test_nothing_relevant_means_nothing_returned(self, optimized):
        """The honest failure mode: no evidence rather than arbitrary evidence."""
        context, _budget = optimized
        evidence, _ = EvidenceSelector().select(
            "Quel est le cours de l'action Tesla aujourd'hui ?", context)
        assert evidence == ()

    def test_selection_is_deterministic(self, optimized):
        context, _budget = optimized
        selector = EvidenceSelector()
        question = "Quel est le plafond pour un repas du midi ?"
        first, _ = selector.select(question, context)
        second, _ = selector.select(question, context)
        assert [e.unit_id for e in first] == [e.unit_id for e in second]

    def test_one_document_cannot_monopolise_the_context(self, optimized):
        context, _budget = optimized
        policy = RetrievalPolicy()
        evidence, _ = EvidenceSelector(policy).select(
            "plafond repas hebergement transport", context)
        counts: dict[str, int] = {}
        for item in evidence:
            counts[item.document_name] = counts.get(item.document_name, 0) + 1
        assert all(count <= policy.max_per_document for count in counts.values())


class TestStructuralBoostScaling:
    """A boost expressed as a fraction must behave like one.

    Added as a constant, a 0.22 section match was 5% of a BM25 score of 5 on the
    reference corpus, and proportionally less as the corpus grew -- so the signal
    faded exactly when the corpus became large enough to need it. The English
    question "what notice period is required to terminate the service contract?"
    retrieved the remote-work termination clause for that reason.
    """

    def test_the_boost_scales_with_the_lexical_score(self):
        from leanlm.packages.retrieval.models import ScoredUnit
        small = ScoredUnit(0, 1.0, 0.5)
        large = ScoredUnit(1, 10.0, 0.5)
        assert small.total == 1.5
        assert large.total == 15.0          # not 10.5

    def test_a_zero_boost_leaves_the_score_untouched(self):
        from leanlm.packages.retrieval.models import ScoredUnit
        assert ScoredUnit(0, 7.25, 0.0).total == 7.25

    def test_the_document_named_by_the_question_wins(self, runtime):
        """Two documents both say "either party may terminate ... notice"."""
        iec = runtime.ask("What notice period is required to terminate the "
                          "service contract?")
        assert iec.evidence
        assert "service_contract" in iec.evidence[0].document_name


class TestGenericQuestionTerms:
    """Framing words must not convict a question of being off-topic.

    "How long does the confidentiality obligation last?" was refused because
    `long`, `last` and `ends` are absent from a corpus that plainly discusses
    confidentiality. Raising the threshold was tried and does not work: no single
    value accepts that question while rejecting one about children's school fees.
    """

    def test_a_duration_question_is_answered(self, runtime):
        iec = runtime.ask("How long does the confidentiality obligation last "
                          "after the contract ends?")
        assert iec.evidence
        assert "five" in iec.response.text.lower() or "5" in iec.response.text

    def test_an_off_topic_question_is_still_refused(self, runtime):
        iec = runtime.ask("What is the reimbursement policy for children's "
                          "school fees?")
        assert iec.evidence == ()

    def test_generic_terms_are_excluded_from_the_gate_only(self):
        """They stay in the scoring vocabulary: "3 days allowed" is worth
        matching on."""
        from leanlm.packages.retrieval.services import _GENERIC_QUESTION_TERMS
        from leanlm.shared.text import content_words
        assert "long" in _GENERIC_QUESTION_TERMS
        assert "long" in content_words("How long is the notice period?")
