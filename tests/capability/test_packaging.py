"""CAP-008 Submission Packaging -- FR-08.

Capability-level behaviour: the builder writes what it was asked to write, the
gate blocks, and the evidence attached is real. Conformance with the official
template's schema lives in tests/submission/test_template_conformance.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from leanlm.packages.packaging import (SPEC, ComplianceChecker, SubmissionBuilder,
                                       SubmissionRequest)
from leanlm.packages.packaging.models import (CrossDisciplinaryPairing,
                                              ModelDeclaration, Submitter, TestPrompt)
from leanlm.packages.packaging.policies import PackagingPolicy
from leanlm.shared.errors import LeanLMError


def _request(tmp_path: Path, **overrides) -> SubmissionRequest:
    payload = {
        "output_dir": str(tmp_path / "submission"),
        "team_id": "leanlm-ma-01",
        "submitter": Submitter("Firmin", "firmin@example.org", "firmin"),
        "pairing": CrossDisciplinaryPairing("knowledge management", True,
                                            "Offline answers over internal policy."),
        "test_prompts": (TestPrompt("tp_001", "What is the expense deadline?"),
                         TestPrompt("tp_002", "Summarise the remote work policy.")),
        "model": ModelDeclaration(name="model-Q4_K_M", parameters_estimate="4B",
                                  model_path="model/m.gguf",
                                  source_url="https://example.invalid/m.gguf",
                                  sha256="a" * 64, license="Apache-2.0"),
        "benchmark_summary": {"is_simulated": False, "metrics": {}},
        "git_commit": "0123456789abcdef",
        "profiler_report": {"measured_on": "participant_laptop"},
    }
    payload.update(overrides)
    return SubmissionRequest(**payload)


class TestSpec:
    def test_identity(self):
        assert SPEC.capability_id == "CAP-008"
        assert "compliance_check" in SPEC.stages


class TestBuild:
    def test_it_writes_into_the_requested_directory(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        assert Path(result.output_dir) == tmp_path / "submission"
        assert result.files

    def test_building_twice_is_idempotent(self, tmp_path):
        builder = SubmissionBuilder()
        first = builder.build(_request(tmp_path))
        second = builder.build(_request(tmp_path))
        assert first.files == second.files

    def test_the_download_script_verifies_what_it_downloaded(self, tmp_path):
        """A download script that checks nothing is theatre."""
        result = SubmissionBuilder().build(_request(tmp_path))
        script = (Path(result.output_dir) / "download_model.sh").read_text(
            encoding="utf-8")
        assert "GGUF" in script
        assert "sha256" in script.lower()
        assert "mismatch" in script.lower()

    def test_the_download_script_is_idempotent(self, tmp_path):
        """The template requires it: safe to run twice without re-downloading."""
        result = SubmissionBuilder().build(_request(tmp_path))
        script = (Path(result.output_dir) / "download_model.sh").read_text(
            encoding="utf-8")
        assert "already present" in script

    def test_no_weights_are_shipped(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        assert not any(name.endswith(".gguf") for name in result.files)

    def test_the_package_stays_small(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        assert result.size_mb < PackagingPolicy().max_package_mb


class TestComplianceGate:
    def test_a_complete_submission_passes(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        assert result.compliance.passed, [
            (c.check_id, c.detail) for c in result.compliance.failures if c.blocking]

    def test_simulated_results_are_refused(self, tmp_path):
        """The control that stops a green pipeline shipping numbers no model
        produced."""
        result = SubmissionBuilder().build(
            _request(tmp_path, benchmark_summary={"is_simulated": True}))
        failed = {c.check_id for c in result.compliance.failures}
        assert "SUB-020" in failed
        assert not result.compliance.passed

    def test_a_missing_required_file_is_caught(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        (Path(result.output_dir) / "REPORT.md").unlink()
        report = ComplianceChecker().check(Path(result.output_dir), _request(tmp_path))
        assert not report.passed

    def test_a_forbidden_file_is_caught(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        (Path(result.output_dir) / "secrets.key").write_text("x", encoding="utf-8")
        report = ComplianceChecker().check(Path(result.output_dir), _request(tmp_path))
        assert not report.passed

    def test_enforce_raises_on_a_failing_report(self, tmp_path):
        result = SubmissionBuilder().build(
            _request(tmp_path, benchmark_summary={"is_simulated": True}))
        with pytest.raises(LeanLMError) as excinfo:
            ComplianceChecker().enforce(result.compliance)
        assert "template" in excinfo.value.record.recommended_action

    def test_every_check_states_whether_it_blocks(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        assert len(result.compliance.checks) >= 20
        for check in result.compliance.checks:
            assert isinstance(check.blocking, bool)
            assert check.label

    def test_advisory_checks_do_not_block(self, tmp_path):
        """A missing git commit is worth saying and not worth refusing over."""
        result = SubmissionBuilder().build(_request(tmp_path, git_commit=""))
        advisory = {c.check_id for c in result.compliance.failures if not c.blocking}
        assert "SUB-022" in advisory
        assert result.compliance.passed


class TestReport:
    def test_the_report_warns_when_results_are_simulated(self, tmp_path):
        result = SubmissionBuilder().build(
            _request(tmp_path, benchmark_summary={"is_simulated": True}))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        assert "simulated backend" in report.lower()

    def test_the_report_answers_the_required_questions(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        for heading in ("Problem", "Design Decisions", "Constraints", "Benchmarks",
                        "limits"):
            assert heading.lower() in report.lower(), heading

    def test_unmeasured_metrics_say_so(self, tmp_path):
        """An empty benchmark must not render as a zero."""
        result = SubmissionBuilder().build(_request(tmp_path))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        assert "not measured" in report.lower()
