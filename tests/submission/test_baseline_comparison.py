"""The comparative claim, asserted rather than described.

Every performance argument in the submission reduces to this: on a corpus larger
than the model's context window, sending everything forces a truncation, and the
answer may lie in the part that was cut. These tests fail the build if that
argument stops being true.

They are deliberately written so that LeanLM can lose. Nothing here is scored
against a handicapped baseline: same model, same backend, same machine, same
session, same grounding metric.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from leanlm.benchmarks.naive import NaiveBaseline
from leanlm.runtime.corpus import CorpusStore
from leanlm.runtime.runtime import LeanLMRuntime

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

PROBE = "Quel est le plafond de delegation de signature pour un chef de projet ?"
PROBE_ANSWER = "7500"


@pytest.fixture(scope="module")
def large_corpus(tmp_path_factory):
    """A generated corpus that exceeds a 4096-token window."""
    from make_corpus import build

    target = tmp_path_factory.mktemp("corpus") / "enterprise_large"
    cwd = Path.cwd()
    import os
    os.chdir(tmp_path_factory.mktemp("notes"))   # keep the answer key out of the repo
    try:
        build(target, documents=40, seed=20260601)
    finally:
        os.chdir(cwd)
    return target


@pytest.fixture(scope="module")
def big_runtime(large_corpus, tmp_path_factory):
    store = CorpusStore(tmp_path_factory.mktemp("db") / "big.sqlite3")
    engine = LeanLMRuntime("benchmark", corpus=store, backend="simulated",
                           verify_model=False)
    engine.ingest([large_corpus])
    yield engine
    engine.close()


class TestCorpusExceedsTheWindow:
    def test_the_corpus_does_not_fit(self, big_runtime):
        """If this fails the rest of the comparison is meaningless."""
        summary = NaiveBaseline(big_runtime, repeat=1, warmup=0).run([PROBE])
        assert summary["corpus"]["tokens"] > summary["context_window"]
        assert summary["corpus_truncated"] is True

    def test_the_dropped_fraction_is_reported(self, big_runtime):
        summary = NaiveBaseline(big_runtime, repeat=1, warmup=0).run([PROBE])
        assert 0.0 < summary["corpus_coverage"] < 1.0
        assert summary["dropped_tokens"] > 0
        assert "never saw" in summary["note"]


class TestTheComparison:
    def test_leanlm_sends_far_fewer_prompt_tokens(self, big_runtime):
        baseline = NaiveBaseline(big_runtime, repeat=1, warmup=0).run([PROBE])
        iec = big_runtime.ask(PROBE)
        baseline_tokens = baseline["metrics"]["prompt_tokens"]["mean"]
        assert iec.metrics.prompt_tokens < baseline_tokens / 3

    def test_leanlm_finds_the_answer_the_baseline_never_saw(self, big_runtime):
        """The demonstration. The clause sits in the last document, which naive
        truncation discards."""
        baseline = NaiveBaseline(big_runtime, repeat=1, warmup=0).run([PROBE])
        answers = [run["answer"] for run in baseline["runs"] if run["ok"]]
        assert answers, "the baseline produced no answer to compare against"
        assert all(PROBE_ANSWER not in answer for answer in answers), answers

        iec = big_runtime.ask(PROBE)
        assert PROBE_ANSWER in iec.response.text
        assert any("delegation_signature" in e.document_name for e in iec.evidence)

    def test_the_corpus_documentation_is_not_part_of_the_corpus(self, big_runtime):
        """A corpus containing its own answer key measures nothing."""
        names = {d["filename"] for d in big_runtime.corpus.documents()}
        assert not any(name.upper().startswith("README") for name in names)

    def test_the_baseline_is_measured_on_the_same_terms(self, big_runtime):
        summary = NaiveBaseline(big_runtime, repeat=1, warmup=0).run([PROBE])
        assert summary["backend"] == big_runtime.backend.describe()
        assert summary["profile"]["fingerprint"] == big_runtime.profile.fingerprint
        assert summary["corpus"]["checksum"] == big_runtime.corpus.corpus_checksum()[:16]

    def test_the_baseline_is_not_scored_on_an_instruction_it_never_got(self):
        """It was never asked to cite, so citations are not held against it."""
        from leanlm.benchmarks.naive import BASELINE_VALIDATION
        assert BASELINE_VALIDATION.require_citations is False
