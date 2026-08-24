"""Conformance with the official ADTC 2026 submission template.

Written after reading the template repository rather than inferring it. The
previous version of this suite asserted a format we had invented: nine required
files, a model manifest, a runtime profile, an evidence directory. The template
asks for four files and a directory, and `metadata.json` is a strict schema.

These tests exist so that the correction cannot silently regress.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from leanlm.packages.packaging import ComplianceChecker, SubmissionBuilder
from leanlm.packages.packaging.models import (DOMAINS, PACKAGING_MODES, PLACEHOLDERS,
                                              CrossDisciplinaryPairing,
                                              ModelDeclaration, SubmissionRequest,
                                              Submitter, TestPrompt)
from leanlm.shared.errors import LeanLMError


def _request(tmp_path: Path, **overrides) -> SubmissionRequest:
    payload = {
        "output_dir": str(tmp_path / "submission"),
        "team_id": "leanlm-ma-01",
        "domain": "corporate_enterprise",
        "language_scope": ("en", "fr"),
        "african_alpha_claim": True,
        "submitter": Submitter("Firmin", "firmin@example.org", "firmin"),
        "pairing": CrossDisciplinaryPairing(
            "enterprise knowledge management", True,
            "Answers questions about internal policy documents offline."),
        "test_prompts": (
            TestPrompt("tp_001", "What is the reimbursement deadline for expenses?"),
            TestPrompt("tp_002", "Summarise the remote work policy."),
        ),
        "model": ModelDeclaration(
            name="Qwen3.5-4B-Q4_K_M", parameters_estimate="4B",
            model_path="model/qwen3.5-4b-q4_k_m.gguf",
            source_url="https://example.invalid/model.gguf",
            sha256="a" * 64, license="Apache-2.0"),
        "benchmark_summary": {"is_simulated": False,
                              "metrics": {"tokens_per_second": {"mean": 9.4},
                                          "peak_rss_mb": {"mean": 2900.0}}},
        "git_commit": "0123456789abcdef",
        "profiler_report": {"measured_on": "participant_laptop"},
    }
    payload.update(overrides)
    return SubmissionRequest(**payload)


class TestFileStructure:
    def test_exactly_the_template_files_are_produced(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        assert set(result.files) == {
            "metadata.json", "download_model.sh", "REPORT.md", ".gitignore",
            "model/.gitkeep",
        }

    def test_nothing_we_invented_is_shipped(self, tmp_path):
        """model_manifest.json, runtime_profile.yaml, MANIFEST.sha256 and
        evidence/ were ours. An evaluator reads none of them."""
        result = SubmissionBuilder().build(_request(tmp_path))
        for invented in ("model_manifest.json", "runtime/runtime_profile.yaml",
                         "MANIFEST.sha256", "README.md"):
            assert invented not in result.files

    def test_the_gitignore_excludes_weights(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        ignored = (Path(result.output_dir) / ".gitignore").read_text(encoding="utf-8")
        assert "*.gguf" in ignored and "model/" in ignored

    def test_the_download_script_is_executable(self, tmp_path):
        import os
        result = SubmissionBuilder().build(_request(tmp_path))
        script = Path(result.output_dir) / "download_model.sh"
        assert os.access(script, os.X_OK)


class TestMetadataSchema:
    def _metadata(self, tmp_path, **overrides) -> dict:
        result = SubmissionBuilder().build(_request(tmp_path, **overrides))
        return json.loads(
            (Path(result.output_dir) / "metadata.json").read_text(encoding="utf-8"))

    def test_every_declared_key_is_present(self, tmp_path):
        metadata = self._metadata(tmp_path)
        assert set(metadata) == {
            "team_id", "domain", "language_scope", "african_alpha_claim",
            "budget_laptop_claim", "submitter", "cross_disciplinary_pairing",
            "test_prompts", "model", "_runtime",
        }

    def test_the_domain_is_the_enum_value_not_prose(self, tmp_path):
        """"Corporate / Enterprise" was what we shipped. The enum says
        corporate_enterprise."""
        assert self._metadata(tmp_path)["domain"] in DOMAINS

    def test_the_runtime_path_is_declared(self, tmp_path):
        metadata = self._metadata(tmp_path)
        assert metadata["_runtime"]["model_path"].startswith("model/")

    def test_the_model_block_has_the_declared_fields(self, tmp_path):
        assert set(self._metadata(tmp_path)["model"]) == {
            "name", "runtime", "quantization", "parameters_estimate", "packaging"}

    def test_the_packaging_mode_is_valid(self, tmp_path):
        assert self._metadata(tmp_path)["model"]["packaging"] in PACKAGING_MODES


class TestChecklist:
    def _checks(self, tmp_path, **overrides):
        result = SubmissionBuilder().build(_request(tmp_path, **overrides))
        return result, {c.check_id: c for c in result.compliance.checks}

    def test_a_complete_submission_passes(self, tmp_path):
        result, checks = self._checks(tmp_path)
        assert result.compliance.passed, [
            (c.check_id, c.detail) for c in result.compliance.failures if c.blocking]

    def test_a_placeholder_left_behind_is_caught(self, tmp_path):
        """The checklist's own wording: no placeholder values remain."""
        _, checks = self._checks(tmp_path, team_id="your-team-id")
        assert checks["SUB-005"].passed is False

    @pytest.mark.parametrize("count", [0, 1, 3])
    def test_anything_other_than_two_prompts_is_caught(self, tmp_path, count):
        prompts = tuple(TestPrompt(f"tp_{i:03d}", f"prompt {i}") for i in range(count))
        _, checks = self._checks(tmp_path, test_prompts=prompts)
        assert checks["SUB-007"].passed is False

    def test_an_empty_prompt_is_caught(self, tmp_path):
        prompts = (TestPrompt("tp_001", "a real prompt"), TestPrompt("tp_002", "  "))
        _, checks = self._checks(tmp_path, test_prompts=prompts)
        assert checks["SUB-007"].passed is False

    def test_a_free_text_domain_is_rejected(self, tmp_path):
        _, checks = self._checks(tmp_path, domain="Corporate / Enterprise")
        assert checks["SUB-006"].passed is False

    def test_a_non_llama_cpp_runtime_is_rejected(self, tmp_path):
        model = ModelDeclaration(name="x", runtime="ollama", parameters_estimate="4B",
                                 model_path="model/x.gguf", source_url="https://x/y",
                                 sha256="b" * 64)
        _, checks = self._checks(tmp_path, model=model)
        assert checks["SUB-010"].passed is False

    def test_an_incomplete_submitter_is_caught(self, tmp_path):
        _, checks = self._checks(tmp_path, submitter=Submitter("Firmin", "", "firmin"))
        assert checks["SUB-008"].passed is False

    def test_a_missing_model_url_blocks(self, tmp_path):
        model = ModelDeclaration(name="x", parameters_estimate="4B",
                                 model_path="model/x.gguf", sha256="c" * 64)
        result, checks = self._checks(tmp_path, model=model)
        assert checks["SUB-016"].passed is False
        assert not result.compliance.passed

    def test_a_credential_in_the_download_script_is_caught(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        script = Path(result.output_dir) / "download_model.sh"
        script.write_text(script.read_text(encoding="utf-8")
                          + '\ncurl -H "Authorization: Bearer $TOKEN" "$MODEL_URL"\n',
                          encoding="utf-8")
        report = ComplianceChecker().check(Path(result.output_dir), _request(tmp_path))
        assert {c.check_id for c in report.failures} >= {"SUB-015"}

    def test_committed_weights_are_caught(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        (Path(result.output_dir) / "model" / "weights.gguf").write_bytes(b"GGUF")
        report = ComplianceChecker().check(Path(result.output_dir), _request(tmp_path))
        assert not report.passed

    def test_simulated_results_are_still_refused(self, tmp_path):
        summary = {"is_simulated": True, "metrics": {}}
        result, checks = self._checks(tmp_path, benchmark_summary=summary)
        assert checks["SUB-020"].passed is False
        assert not result.compliance.passed

    def test_our_profiler_is_not_accepted_as_theirs(self, tmp_path):
        """A cProfile report is useful to us and is not the report the organisers
        publish a tool to produce."""
        _, checks = self._checks(tmp_path, profiler_report=None)
        assert checks["SUB-023"].passed is False
        assert checks["SUB-023"].blocking is False
        assert "adtc-profiler" in checks["SUB-023"].detail

    def test_enforce_raises_on_a_blocking_failure(self, tmp_path):
        result = SubmissionBuilder().build(
            _request(tmp_path, benchmark_summary={"is_simulated": True}))
        with pytest.raises(LeanLMError):
            ComplianceChecker().enforce(result.compliance)


class TestReport:
    def test_it_covers_the_four_required_sections(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        for section in ("Problem", "Design Decisions", "Constraints", "Benchmarks"):
            assert section.lower() in report.lower(), section

    def test_it_states_the_limits(self, tmp_path):
        result = SubmissionBuilder().build(_request(tmp_path))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        assert "limits" in report.lower()

    def test_it_reports_the_comparison_including_where_we_lose(self, tmp_path):
        comparison = {"rows": [
            {"metric": "prompt_tokens", "without_leanlm": 3307,
             "with_leanlm": 298, "delta_pct": -91.0},
            {"metric": "total_ms", "without_leanlm": 1.08,
             "with_leanlm": 14.5, "delta_pct": 1240.0},
        ], "naive_truncated_corpus": True, "naive_corpus_coverage": 0.42,
            "naive_dropped_tokens": 4620}
        result = SubmissionBuilder().build(
            _request(tmp_path, naive_comparison=comparison))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        assert "+1240.0%" in report
        assert "-91.0%" in report

    def test_a_simulated_run_is_banner_flagged(self, tmp_path):
        result = SubmissionBuilder().build(
            _request(tmp_path, benchmark_summary={"is_simulated": True}))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        assert "simulated backend" in report.lower()


class TestShippedPrompts:
    """The two prompts we submit, checked for the properties that matter.

    They are judged on the *model's* responses, and the profiler runs the model
    directly -- LeanLM's retrieval is not in the path. A prompt that assumes
    retrieved excerpts hands a bare model a question with no document in front
    of it, and it will invent an answer.
    """

    def _prompts(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / "prompts"
        return {p.name: p.read_text(encoding="utf-8") for p in sorted(root.glob("tp_*.txt"))}

    def test_exactly_two_are_shipped(self):
        assert len(self._prompts()) == 2

    def test_each_carries_its_own_context(self):
        """No prompt may depend on a document the judge does not have."""
        for name, text in self._prompts().items():
            assert len(text) > 400, f"{name} is too short to carry a document"
            # A heading introducing the material, whatever it is called.
            assert any(marker in text for marker in
                       ("### Policy", "### Clauses", "ARTICLE")), name
            # Both prompts share the shape that works, so a failure in one is
            # informative about the other rather than a separate mystery.
            assert "### Questions" in text and "### Answers" in text, name

    def test_none_refers_to_excerpts_it_does_not_supply(self):
        for name, text in self._prompts().items():
            lowered = text.lower()
            assert "the excerpts below" not in lowered, name
            assert "[s1]" not in lowered, name

    def test_one_asks_for_something_the_context_does_not_cover(self):
        """Calibration is the behaviour that matters most for document work, and
        it is visible in a single response.

        The uncovered item must have no near neighbour in the extract. Asking
        about `parking fees` produced "No, traffic fines are never reimbursed" --
        a conflation that reads as an answer while addressing a different
        subject, and that is ambiguous to score. `breakfast` sits parallel to the
        ceilings that are listed and has no such neighbour.
        """
        joined = " ".join(self._prompts().values()).lower()
        assert "breakfast" in joined
        assert "breakfast" not in joined.split("### policy")[1].split(
            "### questions")[0], "the uncovered item must not appear in the data"
        assert "say so" in joined or "does not state" in joined

    def test_the_instructions_come_before_the_data(self):
        """A reasoning model answers when the prompt ends on the shape of the
        answer, and comments when it ends on an instruction. Rules split around
        the data cost two drafts."""
        for name, text in self._prompts().items():
            lowered = text.lower()
            assert lowered.index("rules:") < lowered.index("###"), name
            # End on the *heading*, with nothing after it. A seeded first item
            # ("1." or "-") was completed as an empty list -- the model emitted
            # "2." and "3." and then reasoned. LeanLM's own working prompt ends
            # with "### Answer" and no seed.
            tail = text.rstrip().splitlines()[-1].strip()
            assert tail.startswith("###"), f"{name} must end on the answer heading"
            assert tail not in ("1.", "-"), f"{name} must not seed the first item"

    def test_no_instruction_about_not_reasoning(self):
        """Each one gives a reasoning model more to reason about: the draft that
        said "nothing after the third answer" returned empty answers and a list
        of six constraints."""
        joined = " ".join(self._prompts().values()).lower()
        for phrase in ("do not explain", "no reasoning", "nothing after"):
            assert phrase not in joined, phrase

    def test_they_contain_neighbouring_figures(self):
        """Picking the wrong one of a neighbouring pair is the realistic
        failure: 15 days versus 30, 25 EUR versus 35."""
        first = self._prompts()["tp_001.txt"]
        for figure in ("15", "30", "25", "35"):
            assert figure in first

    def test_the_argument_builder_round_trips_them(self):
        import json
        import subprocess
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, str(root / "scripts/build_metadata_prompts.py"),
             "--json"], capture_output=True, text=True, cwd=root, check=True)
        entries = json.loads(result.stdout)
        assert [e["prompt_id"] for e in entries] == ["tp_001", "tp_002"]
        assert all("\n" in e["prompt"] for e in entries), "newlines were lost"
