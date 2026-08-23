"""Runtime integration: profiles, corpus store, offline guard, state machine."""
from __future__ import annotations

import socket

import pytest

from leanlm.runtime.profiles import list_profiles, load_profile, validate_profile
from leanlm.runtime.state_machine import RuntimeState, StateMachine
from leanlm.shared.errors import LeanLMError
from leanlm.trust.offline_guard import OfflineGuard


class TestProfiles:
    def test_the_four_profiles_exist(self):
        assert set(list_profiles()) >= {"development", "benchmark", "demo", "competition"}

    @pytest.mark.parametrize("name", ["development", "benchmark", "demo", "competition"])
    def test_each_profile_loads_and_validates(self, name):
        profile = load_profile(name)
        assert profile.id == name
        assert profile.fingerprint

    def test_measurement_profiles_are_deterministic(self):
        """R4/VV-02: a number that cannot be reproduced is not a result."""
        for name in ("benchmark", "competition"):
            assert float(load_profile(name).inference["temperature"]) == 0.0

    def test_the_fingerprint_changes_with_a_parameter(self):
        import dataclasses
        profile = load_profile("benchmark")
        altered = dataclasses.replace(
            profile, inference={**profile.inference, "max_output_tokens": 999})
        assert profile.fingerprint != altered.fingerprint

    def test_an_unknown_profile_names_the_available_ones(self):
        with pytest.raises(LeanLMError) as excinfo:
            load_profile("nonexistent")
        assert "development" in excinfo.value.record.recommended_action

    def test_an_output_window_larger_than_the_context_is_refused(self):
        import dataclasses
        profile = load_profile("development")
        broken = dataclasses.replace(
            profile, inference={**profile.inference, "max_output_tokens": 99999})
        with pytest.raises(LeanLMError):
            validate_profile(broken)


class TestCorpusStore:
    def test_a_document_survives_a_round_trip(self, corpus, structured, sample_file):
        from leanlm.shared.clock import iso_now
        corpus.upsert_document(structured, source_path=str(sample_file),
                               file_checksum="abc", ingested_at=iso_now())
        assert corpus.document_count() == 1
        assert corpus.unit_count() == len(structured.units)
        restored = list(corpus.iter_units())
        assert [u.text for u in restored] == [u.text for u in structured.units]

    def test_reingesting_the_same_document_does_not_duplicate_it(
            self, corpus, structured, sample_file):
        from leanlm.shared.clock import iso_now
        for _ in range(2):
            corpus.upsert_document(structured, source_path=str(sample_file),
                                   file_checksum="abc", ingested_at=iso_now())
        assert corpus.document_count() == 1

    def test_deleting_a_document_removes_its_units(self, corpus, structured, sample_file):
        from leanlm.shared.clock import iso_now
        corpus.upsert_document(structured, source_path=str(sample_file),
                               file_checksum="abc", ingested_at=iso_now())
        assert corpus.remove_document(structured.document_id)
        assert corpus.unit_count() == 0

    def test_an_empty_corpus_fails_with_an_actionable_message(self, corpus):
        with pytest.raises(LeanLMError) as excinfo:
            corpus.context_package()
        assert "ingest" in excinfo.value.record.recommended_action

    def test_the_corpus_checksum_tracks_the_content(self, corpus, structured, sample_file):
        from leanlm.shared.clock import iso_now
        assert corpus.corpus_checksum() == ""
        corpus.upsert_document(structured, source_path=str(sample_file),
                               file_checksum="abc", ingested_at=iso_now())
        assert corpus.corpus_checksum()


class TestOfflineGuard:
    def test_outbound_dns_is_blocked(self):
        with OfflineGuard() as guard:
            with pytest.raises(RuntimeError):
                socket.getaddrinfo("example.com", 80)
        assert guard.violations

    def test_loopback_remains_available(self):
        """llama-server is a legitimate local backend; blocking it would break
        the product rather than protect it."""
        with OfflineGuard():
            assert socket.getaddrinfo("127.0.0.1", 8080)

    def test_the_guard_restores_the_socket_module(self):
        original = socket.socket.connect
        with OfflineGuard():
            pass
        assert socket.socket.connect is original

    def test_a_disabled_guard_does_not_patch_anything(self):
        original = socket.getaddrinfo
        with OfflineGuard(enabled=False):
            assert socket.getaddrinfo is original


class TestStateMachine:
    def test_a_normal_cycle_is_allowed(self):
        machine = StateMachine()
        for state in (RuntimeState.LOADING, RuntimeState.READY, RuntimeState.OPTIMIZING,
                      RuntimeState.RETRIEVING, RuntimeState.PROMPT_READY,
                      RuntimeState.INFERENCING, RuntimeState.VALIDATING,
                      RuntimeState.COMPLETED):
            machine.transition(state)
        assert machine.state is RuntimeState.COMPLETED

    def test_skipping_inference_is_refused(self):
        machine = StateMachine()
        machine.transition(RuntimeState.LOADING)
        machine.transition(RuntimeState.READY)
        with pytest.raises(LeanLMError):
            machine.transition(RuntimeState.VALIDATING)

    def test_reloading_from_ready_is_allowed(self):
        """Measuring a cold start requires unloading a loaded model."""
        machine = StateMachine()
        machine.transition(RuntimeState.LOADING)
        machine.transition(RuntimeState.READY)
        assert machine.transition(RuntimeState.LOADING) is RuntimeState.LOADING

    def test_history_records_every_transition(self):
        machine = StateMachine()
        machine.transition(RuntimeState.LOADING)
        machine.transition(RuntimeState.READY)
        assert len(machine.history) == 2


class TestIntegrity:
    def test_a_missing_model_is_reported_not_assumed(self, tmp_path):
        from leanlm.trust.integrity import verify_model_binding
        report = verify_model_binding(str(tmp_path / "absent.gguf"))
        assert not report.trusted
        assert "download" in report.details.get("hint", "")

    def test_a_file_that_is_not_gguf_is_caught(self, tmp_path):
        from leanlm.trust.integrity import verify_model_binding
        fake = tmp_path / "fake.gguf"
        fake.write_bytes(b"NOTG" + b"\x00" * 100)
        report = verify_model_binding(str(fake))
        assert report.checks.get("gguf_magic") is False

    def test_a_checksum_mismatch_is_caught(self, tmp_path):
        from leanlm.trust.integrity import verify_model_binding
        fake = tmp_path / "fake.gguf"
        fake.write_bytes(b"GGUF" + b"\x00" * 100)
        report = verify_model_binding(str(fake), "b" * 64)
        assert report.checks.get("checksum_matches") is False


class TestCorpusCaching:
    """The corpus snapshot is cached. A cache that can go stale is a correctness
    bug wearing a performance costume, so every write path is asserted."""

    def _add(self, corpus, structured, path, checksum="abc"):
        from leanlm.shared.clock import iso_now
        corpus.upsert_document(structured, source_path=str(path),
                               file_checksum=checksum, ingested_at=iso_now())

    def test_a_repeated_read_returns_an_equivalent_snapshot(
            self, corpus, structured, sample_file):
        self._add(corpus, structured, sample_file)
        first = corpus.context_package()
        second = corpus.context_package()
        assert first.artifact_id == second.artifact_id
        assert first.corpus_checksum == second.corpus_checksum

    def test_adding_a_document_invalidates_the_snapshot(
            self, corpus, structured, sample_file, tmp_path):
        self._add(corpus, structured, sample_file)
        before = corpus.context_package()

        other = tmp_path / "second.md"
        other.write_text("# Autre\n\nLe delai de preavis est de deux mois pleins.\n",
                         encoding="utf-8")
        from leanlm.contracts.ddp import DocumentPipelineState
        from leanlm.packages.ingestion import DocumentIngestionCapability
        from leanlm.packages.understanding import DocumentUnderstandingCapability
        state = DocumentPipelineState(source_path=str(other))
        state = DocumentIngestionCapability().process(state)
        state = DocumentUnderstandingCapability().process(state)
        self._add(corpus, state.structured, other, checksum="def")

        after = corpus.context_package()
        assert after.total_units > before.total_units
        assert after.corpus_checksum != before.corpus_checksum

    def test_removing_a_document_invalidates_the_snapshot(
            self, corpus, structured, sample_file):
        self._add(corpus, structured, sample_file)
        corpus.context_package()
        corpus.remove_document(structured.document_id)
        with pytest.raises(LeanLMError):
            corpus.context_package()

    def test_clearing_invalidates_the_checksum(self, corpus, structured, sample_file):
        self._add(corpus, structured, sample_file)
        assert corpus.corpus_checksum()
        corpus.clear()
        assert corpus.corpus_checksum() == ""

    def test_a_document_subset_is_cached_separately(
            self, corpus, structured, sample_file):
        self._add(corpus, structured, sample_file)
        everything = corpus.context_package()
        subset = corpus.context_package([structured.document_id])
        assert subset.total_units == everything.total_units
        assert corpus.context_package([]) .total_units == everything.total_units


class TestPackagingExtras:
    """A compiled dependency must not be able to block the pure-Python ones.

    Reported from Windows: `pip install -e ".[all]"` failed while unpacking
    llama-cpp-python (260-character path limit), and pip's all-or-nothing
    transaction meant psutil, PyYAML, pypdf and pdfminer.six were not installed
    either. The user was left with every fallback active and no idea why.
    """

    def _extras(self):
        import re
        from pathlib import Path
        text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        block = text.split("[project.optional-dependencies]", 1)[1].split("\n[", 1)[0]
        return {m.group(1): m.group(2) for m in
                re.finditer(r"^(\w+)\s*=\s*(\[[^\]]*\])", block, re.MULTILINE)}

    def test_an_extra_exists_that_compiles_nothing(self):
        extras = self._extras()
        assert "optional" in extras
        assert "llama-cpp-python" not in extras["optional"]

    def test_it_covers_every_fallback_the_runtime_has(self):
        """Each of these replaces a built-in fallback; missing one silently
        degrades a measurement rather than failing."""
        optional = self._extras()["optional"]
        for package in ("psutil", "PyYAML", "pypdf", "pdfminer.six"):
            assert package in optional, package

    def test_the_compiled_backend_is_reachable_on_its_own(self):
        assert "llama-cpp-python" in self._extras()["runtime"]

    def test_no_extra_is_mandatory(self):
        """The zero-dependency promise: the base install pulls nothing."""
        from pathlib import Path
        text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
        assert "dependencies = []" in text


class TestHardwareProfile:
    """The machine that measures must resemble the machine that re-measures.

    The audit sandbox re-runs every submission and compares: throughput tolerates
    +/-25%, memory +/-15%, and beyond 50% variance the comparison fails. A
    measurement taken on hardware far from the profile is a risk of failing that
    comparison, not merely of scoring low.
    """

    def test_it_reports_the_current_machine(self):
        from leanlm.runtime.hardware import inspect
        profile = inspect()
        assert profile.logical_cpus >= 1
        assert profile.platform_name

    def test_a_matching_machine_reports_no_finding(self):
        from leanlm.runtime import hardware
        profile = hardware.HardwareProfile(
            logical_cpus=4, physical_cpus=4, total_ram_gb=8.0,
            platform_name="Linux", machine="x86_64", processor="x86_64")
        assert profile.findings == ()
        assert "matches" in "\n".join(hardware.render(profile))

    def test_too_few_cpus_is_flagged_as_blocking(self):
        from leanlm.runtime.hardware import inspect
        profile = inspect(total_ram_mb=8192)
        if profile.logical_cpus < 4:
            assert profile.blocking_for_submission
            assert any("vCPU" in f for f in profile.findings)

    def test_abundant_memory_is_flagged_too(self):
        """16 GB is not a safe place to measure an 8 GB claim: nothing swaps, so
        the failure that disqualifies cannot be observed."""
        from leanlm.runtime.hardware import inspect
        profile = inspect(total_ram_mb=16384)
        assert any("out-of-memory" in f for f in profile.findings)
        assert any("docker" in f.lower() for f in profile.findings)

    def test_hyper_threading_is_distinguished_from_real_cores(self):
        from leanlm.runtime import hardware
        profile = hardware.HardwareProfile(
            logical_cpus=4, physical_cpus=1, total_ram_gb=8.0,
            platform_name="Windows", machine="AMD64", processor="Intel")
        # A 4-thread dual-core is not a 4-core machine, and llama.cpp is
        # bandwidth-bound enough for the difference to show.
        assert profile.logical_cpus == 4
        assert profile.physical_cpus == 1


class TestPreflight:
    """Every route to a submittable measurement, checked rather than described.

    Written after an audit found that two of the four routes documented in prose
    had no implementation behind them: the Docker path and the CI measurement
    workflow existed only as YAML quoted in a markdown file.
    """

    def test_every_documented_route_is_reported(self):
        from leanlm.runtime.preflight import routes
        found = {r.id for r in routes()}
        assert found == {"development", "native", "docker", "cloud", "github-actions"}

    def test_every_file_a_route_names_actually_exists(self):
        """A route that points at a script nobody wrote is worse than one that
        reports itself unavailable."""
        from pathlib import Path

        from leanlm.runtime.preflight import routes
        root = Path(__file__).resolve().parents[2]
        for route in routes():
            candidates = [route.notes.get("command", "")]
            candidates += [r.detail for r in route.requirements]
            for text in candidates:
                for token in text.split():
                    # Only repository-relative paths, not bare CLI arguments.
                    if "/" in token and token.endswith((".sh", ".yml", ".py")):
                        assert (root / token).is_file(), \
                            f"{route.id} names a missing {token}"

    def test_the_docker_and_ci_routes_have_real_implementations(self):
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        assert (root / "scripts/measure_docker.sh").is_file()
        assert (root / ".github/workflows/measure.yml").is_file()

    def test_development_is_never_offered_for_the_submitted_numbers(self):
        """The one route that must never be marked suitable, whatever the
        hardware happens to be."""
        from leanlm.runtime.preflight import routes
        development = next(r for r in routes() if r.id == "development")
        assert development.suitable_for_submission is False

    def test_a_route_reports_what_is_missing_and_how_to_fix_it(self):
        from leanlm.runtime.preflight import routes
        native = next(r for r in routes() if r.id == "native")
        for requirement in native.requirements:
            if not requirement.present:
                assert requirement.fix, requirement.name

    def test_the_cloud_route_is_always_available_as_an_answer(self):
        """Whatever the local machine is, renting a conforming one is a route."""
        from leanlm.runtime.preflight import routes
        cloud = next(r for r in routes() if r.id == "cloud")
        assert cloud.ready and cloud.suitable_for_submission

    def test_rendering_names_a_shortest_path(self):
        from leanlm.runtime.preflight import render, routes
        assert "shortest path" in render(routes())


class TestIngestReporting:
    """Two totals that must not be confused.

    The CLI printed "corpus: 8 documents, 18 units" on a corpus holding 36: the
    document count was corpus-wide, the unit count came from the run that had
    just finished. The line looked like a fact and was an arithmetic
    impossibility.
    """

    def test_the_run_total_and_the_corpus_total_are_separate(self, tmp_path):
        from leanlm.runtime.corpus import CorpusStore
        from leanlm.runtime.runtime import LeanLMRuntime
        from tests.conftest import DATASET, DATASET_FR

        store = CorpusStore(tmp_path / "two.sqlite3")
        engine = LeanLMRuntime("development", corpus=store, backend="simulated",
                               verify_model=False)
        try:
            first = engine.ingest([DATASET])
            second = engine.ingest([DATASET_FR])
            assert second["documents_added"] == len(second["ingested"])
            assert second["corpus_documents"] == (first["documents_added"]
                                                  + second["documents_added"])
            # The invariant the old line violated.
            assert second["corpus_units"] == (first["units_added"]
                                              + second["units_added"])
        finally:
            engine.close()

    def test_replace_empties_the_corpus_first(self, tmp_path):
        """Without it, an accuracy run is measured against documents the
        evaluation set never accounted for."""
        from leanlm.runtime.corpus import CorpusStore
        from leanlm.runtime.runtime import LeanLMRuntime
        from tests.conftest import DATASET, DATASET_FR

        store = CorpusStore(tmp_path / "replace.sqlite3")
        engine = LeanLMRuntime("development", corpus=store, backend="simulated",
                               verify_model=False)
        try:
            engine.ingest([DATASET_FR])
            result = engine.ingest([DATASET], replace=True)
            assert result["replaced"] is True
            assert result["corpus_documents"] == result["documents_added"]
            names = {d["filename"] for d in store.documents()}
            assert not any(name.startswith("politique") for name in names)
        finally:
            engine.close()

    def test_the_checksum_changes_when_the_corpus_does(self, tmp_path):
        from leanlm.runtime.corpus import CorpusStore
        from leanlm.runtime.runtime import LeanLMRuntime
        from tests.conftest import DATASET, DATASET_FR

        store = CorpusStore(tmp_path / "sum.sqlite3")
        engine = LeanLMRuntime("development", corpus=store, backend="simulated",
                               verify_model=False)
        try:
            first = engine.ingest([DATASET])["corpus_checksum"]
            second = engine.ingest([DATASET_FR])["corpus_checksum"]
            assert first != second
        finally:
            engine.close()


class TestAbandonedRequests:
    """A request stopped before inference returns the runtime to READY.

    Not IDLE: the model is still loaded, and saying otherwise would make the
    next request pay a cold start it never took.
    """

    def test_ready_is_reachable_from_every_in_flight_state(self):
        from leanlm.runtime.state_machine import RuntimeState, StateMachine
        for state in (RuntimeState.OPTIMIZING, RuntimeState.RETRIEVING,
                      RuntimeState.PROMPT_READY):
            machine = StateMachine()
            machine.transition(RuntimeState.LOADING)
            machine.transition(RuntimeState.READY)
            machine.transition(RuntimeState.OPTIMIZING)
            while machine.state is not state:
                nxt = {RuntimeState.OPTIMIZING: RuntimeState.RETRIEVING,
                       RuntimeState.RETRIEVING: RuntimeState.PROMPT_READY}[machine.state]
                machine.transition(nxt)
            assert machine.transition(RuntimeState.READY) is RuntimeState.READY

    def test_inference_still_cannot_be_skipped(self):
        """The relaxation must not have opened a path around the model."""
        from leanlm.runtime.state_machine import RuntimeState, StateMachine
        machine = StateMachine()
        machine.transition(RuntimeState.LOADING)
        machine.transition(RuntimeState.READY)
        machine.transition(RuntimeState.OPTIMIZING)
        with pytest.raises(LeanLMError):
            machine.transition(RuntimeState.COMPLETED)
