"""Accuracy measurement -- the heaviest criterion in the rubric.

Also the place to assert that the ground truth cannot leak into the corpus. An
evaluation set that the retriever can read is an answer key, and a system that
scores 100% by reading its own answer key has measured nothing.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from leanlm.benchmarks.accuracy import (AccuracyEvaluator, Probe, evaluate_accuracy,
                                        load_probes, render_accuracy)
from leanlm.shared.errors import LeanLMError

EVALUATION = Path(__file__).resolve().parents[2] / "datasets/enterprise_en/evaluation.json"
EVALUATION_FR = Path(__file__).resolve().parents[2] / "datasets/enterprise/evaluation.json"


class TestGroundTruthIsolation:
    def test_the_evaluation_file_is_not_an_ingestible_format(self):
        """`.json` is not in the ingestion policy's accepted formats, so the
        answer key sitting next to the corpus can never be retrieved."""
        from leanlm.runtime.runtime import _SUPPORTED_SUFFIXES
        assert EVALUATION.suffix == ".json"
        assert ".json" not in _SUPPORTED_SUFFIXES

    def test_the_runtime_does_not_ingest_it(self, runtime):
        names = {d["filename"] for d in runtime.corpus.documents()}
        assert "evaluation.json" not in names

    def test_every_expected_source_exists_in_the_corpus(self, runtime):
        probes, _ = load_probes(EVALUATION)
        available = {d["filename"] for d in runtime.corpus.documents()}
        for probe in probes:
            if probe.expected_source:
                assert probe.expected_source in available, probe.id


class TestProbeSet:
    def test_it_loads(self):
        probes, meta = load_probes(EVALUATION)
        assert len(probes) >= 10
        assert meta["version"]

    def test_it_contains_unanswerable_questions(self):
        """A set of only answerable questions cannot detect a system that answers
        everything."""
        probes, _ = load_probes(EVALUATION)
        assert any(p.unanswerable for p in probes)

    def test_probe_ids_are_unique(self):
        probes, _ = load_probes(EVALUATION)
        assert len({p.id for p in probes}) == len(probes)

    def test_a_missing_file_is_an_actionable_error(self, tmp_path):
        with pytest.raises(LeanLMError) as excinfo:
            load_probes(tmp_path / "absent.json")
        assert "ground-truth" in excinfo.value.record.recommended_action


class TestScoring:
    def test_the_reference_corpus_is_answered_correctly(self, runtime):
        summary = evaluate_accuracy(runtime, EVALUATION)
        assert summary["accuracy"] >= 0.9, summary["failures"]
        assert summary["refusal_accuracy"] == 1.0, summary["failures"]
        assert summary["hallucination_rate"] == 0.0

    def test_a_wrong_figure_is_scored_wrong_not_partially_right(self):
        from leanlm.benchmarks.accuracy import _check
        probe = Probe(id="X", question="q", must_contain=("25",))
        correct, failure = _check("Le plafond est de 35 EUR.", probe)
        assert correct is False
        assert "25" in failure

    def test_a_forbidden_figure_fails_even_when_the_right_one_is_present(self):
        from leanlm.benchmarks.accuracy import _check
        probe = Probe(id="X", question="q", must_contain=("30",),
                      must_not_contain=("45",))
        correct, failure = _check("Sous 30 jours, ou 45 jours en cas de litige.", probe)
        assert correct is False
        assert "45" in failure

    def test_any_match_accepts_either_spelling(self):
        from leanlm.benchmarks.accuracy import _check
        probe = Probe(id="X", question="q", must_contain=("trois", "3"), match="any")
        assert _check("un preavis de trois mois", probe)[0] is True
        assert _check("un preavis de 3 mois", probe)[0] is True

    def test_answering_an_unanswerable_question_counts_as_hallucination(self, runtime):
        summary = AccuracyEvaluator(runtime).evaluate([
            Probe(id="Z", question="Quel est le cours de l'action Tesla ?",
                  unanswerable=True)])
        assert summary["refusal_accuracy"] == 1.0
        assert summary["hallucination_rate"] == 0.0

    def test_a_correct_answer_from_the_wrong_document_is_reported(self, runtime):
        summary = AccuracyEvaluator(runtime).evaluate([
            Probe(id="Z", question="Quel est le plafond pour un repas du midi ?",
                  must_contain=("25",), expected_source="inexistant.md")])
        assert summary["source_accuracy"] == 0.0
        assert summary["failures"]

    def test_the_rendering_names_the_failures(self, runtime):
        summary = AccuracyEvaluator(runtime).evaluate([
            Probe(id="Z", question="Quel est le plafond pour un repas du midi ?",
                  must_contain=("999",))])
        assert "Z" in render_accuracy(summary)


class TestOffTopicGate:
    def test_a_question_about_an_uncovered_topic_is_declined(self, runtime):
        """Lexical overlap with the wrong policy is the failure this catches:
        "remboursement des frais de scolarite" shares two common words with an
        expense policy and shares nothing that matters."""
        iec = runtime.ask("What is the reimbursement policy for children's "
                          "school fees?")
        assert iec.evidence == ()
        assert iec.validation.confidence.value == "insufficient"

    def test_interrogatives_are_not_treated_as_subject_matter(self):
        from leanlm.shared.text import content_words
        assert "quel" not in content_words("Quel est le delai ?")
        assert "combien" not in content_words("Combien de jours ?")

    def test_elision_is_split(self):
        from leanlm.shared.text import content_words
        assert "indemnite" in content_words("Quelle est l'indemnite mensuelle ?")

    def test_inverted_verbs_lose_their_pronoun(self):
        from leanlm.shared.text import content_words
        assert "faut" in content_words("Quel preavis faut-il ?")

    def test_heading_vocabulary_counts_as_corpus_vocabulary(self, runtime):
        """Headings are not retrievable units, but they are still the document's
        own words. Without this the index disowns its own topics."""
        from leanlm.packages.retrieval import BM25Index
        units = list(runtime.corpus.iter_units())
        assert BM25Index(units).unsupported_terms(["service"]) == []


class TestBothLanguages:
    """The bilingual claim, measured rather than asserted.

    English is the default because that is what the challenge is judged in.
    French stays as evidence that the same pipeline, unmodified, works in a
    second language -- which is only evidence while it is still measured.
    """

    def _score(self, tmp_path, corpus_dir, evaluation, name):
        from leanlm.runtime.corpus import CorpusStore
        from leanlm.runtime.runtime import LeanLMRuntime

        store = CorpusStore(tmp_path / f"{name}.sqlite3")
        engine = LeanLMRuntime("benchmark", corpus=store, backend="simulated",
                               verify_model=False)
        try:
            engine.ingest([corpus_dir])
            return evaluate_accuracy(engine, evaluation)
        finally:
            engine.close()

    def test_english_is_the_default_corpus(self, tmp_path):
        from tests.conftest import DATASET
        summary = self._score(tmp_path, DATASET, EVALUATION, "en")
        assert summary["accuracy"] == 1.0, summary["failures"]
        assert summary["refusal_accuracy"] == 1.0, summary["failures"]
        assert summary["hallucination_rate"] == 0.0

    def test_french_still_works_unmodified(self, tmp_path):
        """No French-specific configuration, no separate profile, no branch."""
        from tests.conftest import DATASET_FR
        summary = self._score(tmp_path, DATASET_FR, EVALUATION_FR, "fr")
        assert summary["accuracy"] == 1.0, summary["failures"]
        assert summary["refusal_accuracy"] == 1.0, summary["failures"]

    def test_the_two_probe_sets_mirror_each_other(self):
        """A comparison between languages is only meaningful if the questions
        are the same questions."""
        english, _ = load_probes(EVALUATION)
        french, _ = load_probes(EVALUATION_FR)
        assert [p.id for p in english] == [p.id for p in french]
        assert ([p.unanswerable for p in english]
                == [p.unanswerable for p in french])


class TestEmptyAnswers:
    """An empty answer is neither wrong nor a refusal.

    A backend that produced nothing was scored as "answered a question the
    corpus cannot support", which sent a reader looking for a hallucination that
    never happened. The distinction matters because the fixes are unrelated.
    """

    def test_an_empty_answer_is_reported_as_empty(self, runtime, monkeypatch):

        def _blank(self, prompt, **kwargs):
            from leanlm.packages.inference.models import GenerationResult
            return GenerationResult(text="", backend="stub", generated_tokens=0,
                                    prompt_tokens=10, first_token_latency_ms=0.0,
                                    inference_ms=0.0, is_simulated=True)

        monkeypatch.setattr(type(runtime.backend), "generate", _blank)
        summary = AccuracyEvaluator(runtime).evaluate([
            Probe(id="Z", question="What is the ceiling for a lunch?",
                  must_contain=("25",))])
        failure = summary["failures"][0]
        assert "empty answer" in failure["failure"]
        assert summary["hallucination_rate"] == 0.0

    def test_an_empty_answer_to_an_unanswerable_probe_is_not_a_hallucination(
            self, runtime, monkeypatch):
        def _blank(self, prompt, **kwargs):
            from leanlm.packages.inference.models import GenerationResult
            return GenerationResult(text="", backend="stub", generated_tokens=0,
                                    prompt_tokens=10, first_token_latency_ms=0.0,
                                    inference_ms=0.0, is_simulated=True)

        monkeypatch.setattr(type(runtime.backend), "generate", _blank)
        summary = AccuracyEvaluator(runtime).evaluate([
            Probe(id="Z", question="What is the share price of Tesla?",
                  unanswerable=True)])
        assert summary["hallucination_rate"] == 0.0
        assert "empty answer" in summary["failures"][0]["failure"]
