"""Delivery pipeline: benchmark campaign, evidence, CCM, submission gate.

This is the Competition-as-Code layer: the rules of the competition expressed as
tests that fail the build rather than as a checklist someone remembers to read.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from leanlm.apps.ccm import verify as verify_ccm
from leanlm.benchmarks import (BenchmarkHarness, compare_to_baseline, load_scenarios,
                               regressions, run_campaign)
from leanlm.packages.packaging import SubmissionBuilder, SubmissionRequest


class TestCCM:
    def test_every_capability_maps_to_code_tests_and_documentation(self):
        report = verify_ccm()
        assert report.ok, report.problems

    def test_eight_capabilities_are_declared(self):
        assert len(verify_ccm().mappings) == 8

    def test_no_stage_is_owned_twice(self):
        owners = verify_ccm().stage_owners
        assert all(len(v) == 1 for v in owners.values()), owners


class TestCampaign:
    def test_a_short_campaign_completes(self, runtime):
        harness = BenchmarkHarness(runtime, repeat=2, warmup=0)
        scenarios = [s for s in load_scenarios() if s.id in ("S1", "S6")]
        summary = harness.run(scenarios)
        assert summary["verdict"] == "pass"
        assert len(summary["scenarios"]) == 2

    def test_every_metric_is_reported_with_its_dispersion(self, runtime):
        """A single run is an anecdote. Mean without spread is not a result."""
        summary = BenchmarkHarness(runtime, repeat=3, warmup=0).run(
            [s for s in load_scenarios() if s.id == "S1"])
        node = summary["metrics"]["total_ms"]
        assert node["samples"] == 3
        assert {"mean", "min", "max", "stdev", "cv", "stable"} <= set(node)

    def test_the_environment_travels_with_the_numbers(self, runtime):
        summary = BenchmarkHarness(runtime, repeat=1, warmup=0).run(
            [s for s in load_scenarios() if s.id == "S1"])
        environment = summary["environment"]
        assert environment["python"] and environment["platform"]
        assert summary["profile"]["fingerprint"]
        assert summary["corpus"]["checksum"]

    def test_a_simulated_campaign_says_so_loudly(self, runtime):
        summary = BenchmarkHarness(runtime, repeat=1, warmup=0).run(
            [s for s in load_scenarios() if s.id == "S1"])
        assert summary["is_simulated"] is True
        assert "SIMULATED" in summary["warning"].upper()

    def test_evidence_is_produced_for_every_scenario(self, runtime):
        summary = BenchmarkHarness(runtime, repeat=1, warmup=0).run(
            [s for s in load_scenarios() if s.id in ("S1", "S6")])
        assert len(summary["evidence"]) == 2
        for item in summary["evidence"]:
            assert item["hypothesis"]
            assert item["conditions"]["profile_fingerprint"]

    def test_results_are_written_where_the_submission_expects_them(self, runtime, tmp_path):
        output = tmp_path / "results.json"
        run_campaign(runtime, repeat=1, warmup=0, output=output,
                     scenarios=[s for s in load_scenarios() if s.id == "S1"])
        assert json.loads(output.read_text(encoding="utf-8"))["verdict"] == "pass"


class TestRegressionDetection:
    def test_a_throughput_drop_is_a_regression(self):
        current = {"tokens_per_second": 8.0, "peak_rss_mb": 2800.0}
        baseline = {"tokens_per_second": 12.0, "peak_rss_mb": 2800.0}
        comparison = compare_to_baseline(current, baseline)
        assert comparison["tokens_per_second"].regression

    def test_an_improvement_is_not_a_regression(self):
        comparison = compare_to_baseline({"tokens_per_second": 14.0},
                                         {"tokens_per_second": 12.0})
        assert comparison["tokens_per_second"].improved
        assert not regressions(comparison)

    def test_noise_below_the_tolerance_is_not_a_regression(self):
        comparison = compare_to_baseline({"tokens_per_second": 11.8},
                                         {"tokens_per_second": 12.0})
        assert not regressions(comparison)

    def test_no_baseline_means_no_false_alarm(self):
        assert compare_to_baseline({"tokens_per_second": 12.0}, None) == {}


class TestSubmissionGate:
    """The end-to-end gate: a campaign becomes a package, or it is refused."""

    def _request(self, tmp_path, runtime, **overrides):
        from leanlm.packages.packaging.models import (CrossDisciplinaryPairing,
                                                      ModelDeclaration, Submitter,
                                                      TestPrompt)
        payload = {
            "output_dir": str(tmp_path / "sub"),
            "team_id": "leanlm-ma-01",
            "submitter": Submitter("Firmin", "firmin@example.org", "firmin"),
            "pairing": CrossDisciplinaryPairing("knowledge management", True,
                                                "Offline policy answers."),
            "test_prompts": (TestPrompt("tp_001", "What is the expense deadline?"),
                             TestPrompt("tp_002", "Summarise the remote work policy.")),
            "model": ModelDeclaration(name="m-Q4_K_M", parameters_estimate="4B",
                                      model_path="model/m.gguf",
                                      source_url="https://example.invalid/m.gguf",
                                      sha256="a" * 64),
            "benchmark_summary": {"verdict": "pass", "is_simulated": False},
            "profiler_report": {"measured_on": "participant_laptop"},
            "git_commit": "abc1234",
        }
        payload.update(overrides)
        return SubmissionRequest(**payload)

    def test_a_simulated_campaign_cannot_be_submitted(self, runtime, tmp_path):
        """A green pipeline must not be able to ship numbers no model produced."""
        summary = BenchmarkHarness(runtime, repeat=1, warmup=0).run(
            [s for s in load_scenarios() if s.id == "S1"])
        result = SubmissionBuilder().build(
            self._request(tmp_path, runtime, benchmark_summary=summary))
        assert not result.compliance.passed
        assert "SUB-020" in {c.check_id for c in result.compliance.failures}

    def test_the_package_never_contains_the_weights(self, runtime, tmp_path):
        result = SubmissionBuilder().build(self._request(tmp_path, runtime))
        for name in result.files:
            assert not name.endswith(".gguf")
        assert result.size_mb < 25

    def test_the_report_can_be_read_without_the_repository(self, runtime, tmp_path):
        result = SubmissionBuilder().build(self._request(tmp_path, runtime))
        report = (Path(result.output_dir) / "REPORT.md").read_text(encoding="utf-8")
        assert len(report) > 1200
        assert "llama.cpp" in report


class TestModelAcquisition:
    """Getting a model must work on both target platforms, with the same checks.

    A downloader that verifies nothing is theatre, and one that exists only for
    Linux is not a downloader for a product whose premise is "the laptop you
    already have".
    """

    def _scripts(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / "scripts"
        return ((root / "download_model.sh").read_text(encoding="utf-8"),
                (root / "download_model.ps1").read_text(encoding="utf-8"))

    def test_both_platforms_have_a_downloader(self):
        bash, powershell = self._scripts()
        assert bash and powershell

    def test_both_verify_the_gguf_magic_bytes(self):
        """An HTML error page saved under a .gguf name is the common failure."""
        for script in self._scripts():
            assert "GGUF" in script

    def test_both_verify_the_checksum(self):
        for script in self._scripts():
            lowered = script.lower()
            assert "sha256" in lowered
            assert "mismatch" in lowered

    def test_both_delete_a_file_that_is_not_a_model(self):
        bash, powershell = self._scripts()
        assert "rm -f" in bash
        assert "Remove-Item" in powershell

    def test_no_model_is_shipped_with_the_repository(self):
        """SUB-002, asserted at the repository level rather than only at packaging."""
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        assert not list(root.rglob("*.gguf"))


class TestMeasurementRoutesAreImplemented:
    """Documentation that describes a command must be backed by the command.

    Two routes were written up in `docs/WHERE-TO-MEASURE.md` with example
    invocations before either existed: the Docker path and the CI workflow. A
    reader would have copied a command that was not there.
    """

    def _root(self):
        from pathlib import Path
        return Path(__file__).resolve().parents[2]

    def test_the_provisioning_script_installs_what_the_profiler_needs(self):
        script = (self._root() / "scripts/provision_ubuntu.sh").read_text(
            encoding="utf-8")
        for needed in ("llama-bench", "adtc-profiler", "build-essential", "3, 11"):
            assert needed in script, needed

    def test_the_docker_script_caps_memory_to_the_profile(self):
        script = (self._root() / "scripts/measure_docker.sh").read_text(
            encoding="utf-8")
        assert "--memory=" in script
        assert "7.5g" in script

    def test_the_docker_script_warns_when_the_host_lacks_cpus(self):
        """Capping reproduces a limit, never a speed."""
        script = (self._root() / "scripts/measure_docker.sh").read_text(
            encoding="utf-8")
        assert "will not invent" in script

    def test_the_ci_workflow_caps_memory_too(self):
        workflow = (self._root() / ".github/workflows/measure.yml").read_text(
            encoding="utf-8")
        assert "--memory=" in workflow
        assert "--cpus=4" in workflow

    def test_the_ci_workflow_is_manual(self):
        """It builds llama.cpp and compiles llama-cpp-python; not a push hook."""
        workflow = (self._root() / ".github/workflows/measure.yml").read_text(
            encoding="utf-8")
        assert "workflow_dispatch" in workflow
        assert "on:\n  push" not in workflow

    def test_both_scripts_are_executable(self):
        import os
        for name in ("scripts/provision_ubuntu.sh", "scripts/measure_docker.sh"):
            assert os.access(self._root() / name, os.X_OK), name


class TestDownloadersDeriveTheFilename:
    """Two candidates must not be able to overwrite each other.

    With a fixed `model.gguf` default, downloading a second model silently
    replaced the first, and every measurement already attributed to "the model"
    became wrong with no error to signal it. Comparing candidates -- which is
    the main reason to download more than one -- was impossible by construction.

    The submission package is the opposite case, and stays fixed: its
    download_model.sh writes to the one path `_runtime.model_path` declares.
    """

    def _scripts(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / "scripts"
        return ((root / "download_model.sh").read_text(encoding="utf-8"),
                (root / "download_model.ps1").read_text(encoding="utf-8"))

    def test_neither_parameter_defaults_to_an_anonymous_name(self):
        """The *parameter* default, not the fallback used when a URL carries no
        filename -- that one is legitimate and must stay."""
        bash, powershell = self._scripts()
        assert "MODEL_FILE:-model.gguf" not in bash
        assert '[string]$Destination = ""' in powershell

    def test_both_strip_query_strings_before_taking_the_leaf(self):
        """`?download=true` is common on model hosts."""
        bash, powershell = self._scripts()
        assert "%%[?#]*" in bash
        assert '-split "[?#]"' in powershell

    def test_both_fall_back_when_the_url_carries_no_filename(self):
        for script in self._scripts():
            assert "model.gguf" in script

    def test_the_packaged_script_still_writes_one_declared_path(self):
        """The template requires _runtime.model_path to match what the script
        writes, so the submission's own downloader is deliberately fixed."""
        from leanlm.packages.packaging.services import DOWNLOAD_TEMPLATE
        assert "{model_path}" in DOWNLOAD_TEMPLATE
        assert "derived" not in DOWNLOAD_TEMPLATE


class TestWindowsDownloaderCompatibility:
    """Windows PowerShell 5.1 is the default on Windows 10 and 11.

    The first version of this script used `Get-Content -AsByteStream`, which
    exists only in PowerShell 6+. Its fallback never ran: a parameter-binding
    error is raised before `-ErrorAction` is consulted, so the script died on
    the very machine it was written for.
    """

    def _script(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[2]
                / "scripts/download_model.ps1").read_text(encoding="utf-8")

    def _code(self) -> str:
        """Executable lines only. A comment explaining why a cmdlet is avoided
        must not read as a use of it."""
        return "\n".join(line for line in self._script().splitlines()
                          if not line.lstrip().startswith("#"))

    def test_no_powershell_6_only_cmdlet_parameters(self):
        code = self._code()
        for modern_only in ("-AsByteStream", "-SkipCertificateCheck",
                            "ConvertFrom-Json -AsHashtable"):
            assert modern_only not in code, modern_only

    def test_the_reason_is_documented_where_someone_would_reintroduce_it(self):
        assert "-AsByteStream" in self._script()

    def test_dotnet_calls_receive_an_absolute_path(self):
        """.NET resolves relative paths against the process working directory --
        often C:\\Windows\\system32 -- while PowerShell cmdlets use $PWD. Mixing
        the two sent the header check looking in the wrong directory for a file
        that had downloaded perfectly."""
        code = self._code()
        assert "GetFullPath" in code
        assert code.index("GetFullPath") < code.index("OpenRead")

    def test_the_header_is_read_through_dotnet(self):
        """Works on both versions, and does not read a 2.5 GB file to see four
        bytes."""
        script = self._script()
        assert "System.IO.File]::OpenRead" in script

    def test_large_files_are_streamed_not_buffered(self):
        """Invoke-WebRequest on 5.1 buffers the whole response in memory, which
        thrashes or fails on a multi-gigabyte model."""
        code = self._code()
        assert "curl.exe" in code or "WebClient" in code
        assert "Invoke-WebRequest" not in code

    def test_a_blob_url_is_corrected(self):
        """/blob/ serves the page, not the file: the failure is silent and the
        result is an HTML document with a .gguf name."""
        script = self._script()
        assert "/blob/" in script and "/resolve/" in script

    def test_the_magic_byte_failure_names_its_causes(self):
        script = self._script()
        assert "gated repository" in script


class TestWindowsProvisioning:
    """Windows needs a route to a real backend that does not involve a compiler.

    Reported from a Windows install: `leanlm doctor` said the model was present
    and the backend was still `simulated`, because llama-cpp-python had failed to
    build and nothing told the user that prebuilt binaries would do instead.
    """

    def _script(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[2]
                / "scripts/provision_windows.ps1").read_text(encoding="utf-8")

    def test_the_script_exists_for_both_platforms(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2] / "scripts"
        assert (root / "provision_windows.ps1").is_file()
        assert (root / "provision_ubuntu.sh").is_file()

    def test_it_fetches_prebuilt_binaries_rather_than_compiling(self):
        script = self._script()
        assert "releases" in script
        assert "cmake" not in script.lower()

    def test_it_matches_assets_by_shape_not_by_a_fixed_name(self):
        """Asset names change between releases; a hardcoded one rots silently."""
        script = self._script()
        assert "-match" in script
        assert "x64" in script

    def test_it_excludes_accelerator_builds(self):
        """Integrated graphics: a CUDA build is a larger download that changes
        nothing."""
        script = self._script()
        for excluded in ("cuda", "vulkan", "arm64"):
            assert excluded in script

    def test_it_verifies_llama_bench_is_present(self):
        """The official profiler shells out to it; without it no throughput can
        be measured at all."""
        assert "llama-bench.exe" in self._script()

    def test_it_does_not_claim_the_machine_becomes_conforming(self):
        assert "conforming" in self._script()

    def test_doctor_reports_the_binaries(self):
        from pathlib import Path
        cli = (Path(__file__).resolve().parents[2]
               / "src/leanlm/apps/cli.py").read_text(encoding="utf-8")
        assert "llama.cpp binaries" in cli
        assert "provision_windows.ps1" in cli

    def test_it_distinguishes_an_api_failure_from_a_missing_asset(self):
        """The first version reported "no Windows x64 CPU asset" when the API had
        returned nothing at all -- a message that sent the reader looking for the
        wrong problem."""
        script = self._script()
        assert "Show-ManualRoute" in script
        assert "rate limit" in script.lower()
        assert "not JSON" in script

    def test_it_offers_a_route_that_needs_no_api_call(self):
        """GitHub rate-limits unauthenticated requests by IP, so the API can be
        unavailable through no fault of the user."""
        script = self._script()
        assert "-AssetUrl" in script
        assert "$Token" in script

    def test_it_forces_tls_12(self):
        """Windows PowerShell 5.1 does not always negotiate it, and GitHub
        requires it."""
        assert "Tls12" in self._script()


class TestUbuntuProvisioning:
    """Ubuntu 24.04 refuses a system-wide pip install.

    Reported from a fresh droplet: llama.cpp built, then the profiler install
    died on `externally-managed-environment`. PEP 668 has been enforced by
    Debian and Ubuntu since 23.04, so a script that calls `pip install`
    system-wide is broken on the distribution most likely to be used for the
    measurement.
    """

    def _script(self) -> str:
        from pathlib import Path
        return (Path(__file__).resolve().parents[2]
                / "scripts/provision_ubuntu.sh").read_text(encoding="utf-8")

    def test_it_does_not_pip_install_system_wide(self):
        script = self._script()
        assert "python3 -m pip install --quiet \"git+" not in script

    def test_it_uses_pipx_with_a_virtualenv_fallback(self):
        """pipx is not present on every derivative; a venv always is."""
        script = self._script()
        assert "pipx install" in script
        assert "python3 -m venv" in script

    def test_the_entry_point_ends_up_on_path(self):
        script = self._script()
        assert "ensurepath" in script or "ln -sf" in script
        assert ".local/bin" in script

    def test_it_fails_loudly_when_the_profiler_is_unreachable(self):
        """Reporting success on a tool the next step cannot call wastes the one
        thing in short supply before a deadline."""
        script = self._script()
        assert "not on PATH in this shell" in script
        assert "exit 1" in script

    def test_presence_is_checked_by_lookup_not_by_a_probe_flag(self):
        """`adtc-profiler` has no --version. Calling one to test for presence
        reported a working installation as MISSING, and sent the reader to
        reinstall a tool that was already there."""
        script = self._script()
        assert "--version 2>/dev/null || echo MISSING" not in script
        assert "command -v" in script

    def test_the_check_reports_the_resolved_path(self):
        """A path answers "which one", which matters when two builds exist."""
        assert 'command -v "$1"' in self._script()
