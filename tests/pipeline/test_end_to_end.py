"""End-to-end pipeline: the ten DIC phases, in order, with evidence.

These are the tests that would catch a regression in the product's central
claim: that an answer is grounded, traceable, budgeted and measured.
"""
from __future__ import annotations

import pytest

from leanlm.contracts.iec import STAGE_ORDER
from leanlm.runtime.dic import verify as verify_dic
from leanlm.shared.errors import LeanLMError

QUESTION = "What is the reimbursement deadline for expense reports?"


class TestPipelineOrder:
    def test_the_ten_phases_run_in_the_declared_order(self, runtime):
        iec = runtime.ask(QUESTION)
        assert list(iec.completed_stages) == [s.value for s in STAGE_ORDER]

    def test_every_stage_is_timed(self, runtime):
        iec = runtime.ask(QUESTION)
        assert all(record.duration_ms >= 0 for record in iec.stages)
        assert iec.metrics.stage_durations

    def test_the_contract_is_satisfied(self, runtime):
        assert verify_dic(runtime.ask(QUESTION)) == ()


class TestGroundedAnswer:
    def test_the_answer_cites_a_real_passage(self, runtime):
        iec = runtime.ask(QUESTION)
        assert iec.evidence
        labels = {e.label for e in iec.evidence}
        assert any(f"[{label}]" in iec.response.text for label in labels)

    def test_evidence_points_back_to_a_document(self, runtime):
        iec = runtime.ask(QUESTION)
        for item in iec.evidence:
            assert item.document_name.endswith((".md", ".txt", ".pdf"))
            assert item.unit_id

    def test_the_answer_carries_the_figure_from_the_corpus(self, runtime):
        iec = runtime.ask(QUESTION)
        assert "30" in iec.response.text

    def test_the_prompt_never_exceeds_the_usable_window(self, runtime):
        iec = runtime.ask(QUESTION)
        budget = iec.optimized_context.budget
        assert iec.prompt.prompt_tokens <= (budget.max_context_tokens
                                            - budget.reserved_output_tokens)


class TestHonesty:
    def test_an_out_of_corpus_question_is_refused(self, runtime):
        """The single most important behaviour: not answering."""
        iec = runtime.ask("What is the current share price of Tesla?")
        assert iec.evidence == ()
        assert iec.validation.confidence.value == "insufficient"

    def test_simulated_runs_are_labelled_everywhere(self, runtime):
        iec = runtime.ask(QUESTION)
        assert iec.response.is_simulated
        assert iec.metrics.is_simulated
        assert any("SIMULATED" in w.upper() for w in iec.warnings)

    def test_an_empty_question_is_rejected_before_any_work(self, runtime):
        with pytest.raises(LeanLMError):
            runtime.ask("   ")


class TestMetrics:
    def test_the_reported_compression_is_end_to_end(self, runtime):
        """The number that demonstrates the layer: how much of the corpus never
        reached the model."""
        iec = runtime.ask(QUESTION)
        corpus_tokens = iec.context_package.total_tokens
        assert iec.prompt.context_tokens < corpus_tokens
        assert 0.0 < iec.metrics.context_compression_ratio < 1.0

    def test_memory_is_measured_not_assumed(self, runtime):
        iec = runtime.ask(QUESTION)
        assert iec.metrics.peak_rss_mb > 0

    def test_metrics_are_persisted_for_the_session(self, runtime):
        iec = runtime.ask(QUESTION)
        rows = runtime.corpus._connection.execute(
            "SELECT request_id FROM metrics WHERE request_id = ?",
            (iec.request_id,)).fetchall()
        assert rows

    def test_two_identical_questions_produce_identical_evidence(self, runtime):
        """Determinism at the layer level: same corpus, same question, same
        passages. Only the model's own sampling may vary, and at temperature 0
        it does not."""
        first = runtime.ask(QUESTION)
        second = runtime.ask(QUESTION)
        assert ([e.unit_id for e in first.evidence]
                == [e.unit_id for e in second.evidence])
        assert first.prompt.rendered == second.prompt.rendered


class TestFailurePath:
    def test_a_failing_stage_still_produces_metrics_and_an_error_record(self, runtime):
        """DIC-07: a failure is an observation, not a silence."""
        def explode(iec, stage):
            from leanlm.shared.errors import inference_error
            raise inference_error("TEST-001", "deliberate failure",
                                  capability="inference", stage=stage.value)

        capability = runtime.registry.get("CAP-005")
        original = capability.execute
        capability.execute = explode  # type: ignore[method-assign]
        try:
            iec = runtime.ask(QUESTION)
        finally:
            capability.execute = original  # type: ignore[method-assign]
        assert iec.errors
        assert iec.errors[0].code == "TEST-001"
        assert iec.metrics is not None
