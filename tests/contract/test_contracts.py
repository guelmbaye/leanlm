"""Contract layer: versioned DTOs, immutable context, unique stage ownership.

These tests protect the boundaries between capabilities. A capability may be
rewritten freely; the contract may not change silently.
"""
from __future__ import annotations

import pytest

from leanlm.contracts.capability import CapabilityRegistry, CapabilitySpec
from leanlm.contracts.ddp import DocumentPipelineState, DocumentState
from leanlm.contracts.iec import STAGE_ORDER, InferenceExecutionContext, PipelineStage
from leanlm.contracts.versions import CONTRACT_VERSIONS
from leanlm.shared.errors import LeanLMError
from leanlm.shared.serialization import canonical_json
from leanlm.shared.telemetry import StageRecord


class TestVersions:
    def test_every_dto_declares_a_version(self):
        assert CONTRACT_VERSIONS
        for name, version in CONTRACT_VERSIONS.items():
            assert version.count(".") == 2, f"{name} is not semantic: {version}"

    def test_dto_classes_match_the_registry(self):
        from leanlm.contracts import dto
        for name, version in CONTRACT_VERSIONS.items():
            klass = getattr(dto, name, None)
            if klass is not None and hasattr(klass, "SCHEMA_VERSION"):
                assert klass.SCHEMA_VERSION == version, name


class TestStageOrder:
    def test_ten_phases_in_a_fixed_order(self):
        assert len(STAGE_ORDER) == 10
        assert STAGE_ORDER[0] is PipelineStage.SESSION_CREATION
        assert STAGE_ORDER[-1] is PipelineStage.SESSION_CLEANUP

    def test_inference_comes_after_prompt_assembly(self):
        order = [s.value for s in STAGE_ORDER]
        assert order.index("prompt_assembly") < order.index("model_inference")
        assert order.index("model_inference") < order.index("response_validation")


class TestIEC:
    def test_evolve_returns_a_new_object(self):
        first = InferenceExecutionContext(session_id="s", request_id="r",
                                          question="q", profile_id="development")
        second = first.evolve(trace={"marker": 1})
        assert first.trace == {} and second.trace == {"marker": 1}
        assert first is not second
        assert second.revision > first.revision

    @pytest.mark.parametrize("field", ["request_id", "session_id", "question"])
    def test_identity_fields_cannot_be_rewritten(self, field):
        """What a run was asked to do cannot be edited after the fact; that is
        what makes the recorded metrics attributable (DIC-04)."""
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="q", profile_id="development")
        with pytest.raises((LeanLMError, ValueError)):
            iec.evolve(**{field: "forged"})

    def test_stages_accumulate_in_order(self):
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="q", profile_id="development")
        for stage in ("session_creation", "resource_assessment"):
            iec = iec.with_stage(StageRecord(stage=stage, capability="runtime",
                                             started_at="2026-06-01T10:00:00Z",
                                             duration_ms=1.0))
        assert iec.completed_stages == ("session_creation", "resource_assessment")

    def test_archival_form_can_exclude_user_text(self):
        """Telemetry must be shippable without shipping the user's documents."""
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="secret commercial",
                                        profile_id="development")
        assert "secret commercial" not in canonical_json(iec.to_dict(include_text=False))
        assert "secret commercial" in canonical_json(iec.to_dict(include_text=True))

    def test_warnings_and_errors_are_appended_not_replaced(self):
        iec = InferenceExecutionContext(session_id="s", request_id="r",
                                        question="q", profile_id="development")
        iec = iec.with_warning("premier").with_warning("second")
        assert len(iec.warnings) == 2


class TestDocumentPipeline:
    def test_states_cannot_be_skipped(self):
        """A document cannot be exported without having been normalized."""
        state = DocumentPipelineState(source_path="/tmp/x.md")
        with pytest.raises((LeanLMError, ValueError)):
            state.advance(DocumentState.EXPORTED)

    def test_normal_progression_is_allowed(self):
        state = DocumentPipelineState(source_path="/tmp/x.md")
        assert state.advance(DocumentState.VALIDATED).state is DocumentState.VALIDATED


class TestRegistry:
    def test_a_stage_has_exactly_one_owner(self):
        from leanlm.apps.ccm import _discover_specs
        owners: dict[str, list[str]] = {}
        for spec in _discover_specs():
            for stage in spec.stages:
                owners.setdefault(stage, []).append(spec.capability_id)
        duplicated = {k: v for k, v in owners.items() if len(v) > 1}
        assert not duplicated, duplicated

    def test_two_capabilities_claiming_one_stage_is_rejected(self):
        registry = CapabilityRegistry()

        class _Fake:
            def __init__(self, cid):
                self.spec = CapabilitySpec(capability_id=cid, name=cid,
                                           package="x", pipeline="inference",
                                           stages=("model_inference",))

        registry.register(_Fake("CAP-900"))
        registry.register(_Fake("CAP-901"))
        with pytest.raises(LeanLMError):
            registry.for_stage(PipelineStage.MODEL_INFERENCE)

    def test_capabilities_never_span_two_pipelines(self):
        from leanlm.apps.ccm import _discover_specs
        for spec in _discover_specs():
            assert spec.pipeline in ("document", "inference", "delivery"), spec.capability_id
