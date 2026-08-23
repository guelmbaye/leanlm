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


class TestPrepareWithoutGenerating:
    """Stopping after prompt assembly, to hand over the exact backend input.

    Added while diagnosing a run that appeared to hang on a two-core laptop. The
    first version of the diagnostic printed `-f "<prompt file>"` and left the
    reader to reconstruct the prompt -- which is precisely the input that might
    be causing the problem.
    """

    def test_the_pipeline_stops_where_asked(self, runtime):
        from leanlm.contracts.iec import PipelineStage
        iec = runtime.ask(QUESTION, stop_after=PipelineStage.PROMPT_ASSEMBLY,
                          check_contract=False)
        assert iec.prompt is not None
        assert iec.response is None
        assert "model_inference" not in iec.completed_stages

    def test_the_prompt_is_the_one_the_backend_would_receive(self, runtime):
        from leanlm.contracts.iec import PipelineStage
        prepared = runtime.ask(QUESTION, stop_after=PipelineStage.PROMPT_ASSEMBLY,
                               check_contract=False)
        complete = runtime.ask(QUESTION)
        assert prepared.prompt.rendered == complete.prompt.rendered

    def test_the_runtime_is_usable_afterwards(self, runtime):
        """An early exit must not leave the state machine mid-cycle."""
        from leanlm.contracts.iec import PipelineStage
        runtime.ask(QUESTION, stop_after=PipelineStage.PROMPT_ASSEMBLY,
                    check_contract=False)
        iec = runtime.ask(QUESTION)
        assert iec.response is not None
        assert verify_dic(iec) == ()

    def test_evidence_is_already_selected(self, runtime):
        from leanlm.contracts.iec import PipelineStage
        iec = runtime.ask(QUESTION, stop_after=PipelineStage.PROMPT_ASSEMBLY,
                          check_contract=False)
        assert iec.evidence


class TestAnswerIsAlwaysShown:
    """The answer must reach the reader whatever the backend does.

    Observed with the llama-server backend: the sources and the verdict printed
    with nothing between them. The CLI had assumed that installing a token hook
    meant tokens would stream, and only the subprocess backend emits any.
    """

    def test_streaming_state_starts_unstarted(self, runtime):
        from leanlm.apps.cli import _install_stream
        state = _install_stream(runtime, quiet=False)
        assert state["started"] is False

    def test_a_quiet_run_reports_nothing_streamed(self, runtime):
        from leanlm.apps.cli import _install_stream
        assert _install_stream(runtime, quiet=True)["started"] is False

    def test_a_backend_without_a_token_hook_still_answers(self, runtime):
        """The simulated backend emits no tokens; the answer exists regardless."""
        iec = runtime.ask(QUESTION)
        assert iec.response is not None
        assert iec.response.text.strip()


class TestOutOfProcessMemory:
    """Memory measured in this process says nothing about a model in another.

    With llama-server the run reported a 32 MB peak while a 1.2 GB model was
    resident in the server. Efficiency is 20% of the ADTC score; a figure that
    omits the model measures nothing about the model.
    """

    def test_an_out_of_process_backend_is_flagged(self, runtime):
        iec = runtime.ask(QUESTION)
        iec = iec.evolve(trace={**iec.trace,
                                "runtime": {"backend": "llama-server"}})
        capability = runtime.registry.get("CAP-007")
        from leanlm.contracts.iec import PipelineStage
        result = capability.execute(iec, PipelineStage.METRICS_FINALIZATION)
        assert any("this process only" in w for w in result.warnings)

    def test_an_in_process_backend_is_not_flagged(self, runtime):
        iec = runtime.ask(QUESTION)
        iec = iec.evolve(trace={**iec.trace,
                                "runtime": {"backend": "llama-cpp-python"}})
        capability = runtime.registry.get("CAP-007")
        from leanlm.contracts.iec import PipelineStage
        result = capability.execute(iec, PipelineStage.METRICS_FINALIZATION)
        assert not any("this process only" in w for w in result.warnings)
