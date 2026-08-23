"""CAP-005 Inference Execution -- FR-05."""
from __future__ import annotations

import pytest

from leanlm.contracts.dto import SelectedEvidence
from leanlm.packages.inference import SPEC, PromptBuilder
from leanlm.packages.inference.backends import SimulatedBackend, build_backend, verify_model
from leanlm.packages.inference.models import ModelBinding
from leanlm.packages.inference.policies import GenerationPolicy, PromptPolicy
from leanlm.packages.inference.services import prompt_efficiency
from leanlm.shared.errors import LeanLMError


def _evidence(label: str, text: str, document: str = "politique.md"):
    return SelectedEvidence(
        meta=SelectedEvidence.build_meta(identity=(label, text[:40]),
                                         stage="evidence_retrieval"),
        label=label, unit_id=f"u-{label}", document_name=document,
        text=text, section_path=("1. Delais",), score=1.0, token_estimate=32,
    ).sealed()


EVIDENCE = (
    _evidence("S1", "Le remboursement intervient sous 30 jours ouvres apres validation."),
    _evidence("S2", "Le plafond pour un repas du midi est de 25 EUR par personne."),
)


class TestSpec:
    def test_owns_two_consecutive_stages(self):
        assert SPEC.capability_id == "CAP-005"
        assert SPEC.stages == ("prompt_assembly", "model_inference")


class TestPrompt:
    def test_prompt_contains_every_passage_and_its_label(self):
        prompt = PromptBuilder().build("Quel est le plafond du midi ?", EVIDENCE)
        for item in EVIDENCE:
            assert item.text in prompt.rendered
            assert f"[{item.label}]" in prompt.rendered

    def test_prompt_instructs_the_model_to_refuse_without_evidence(self):
        """With no excerpt, the prompt must still carry the refusal instruction:
        this is what turns "I don't know" into the expected behaviour rather
        than a failure."""
        from leanlm.packages.inference.services import INSUFFICIENT_ANSWER
        prompt = PromptBuilder().build("Question hors corpus ?", ())
        assert INSUFFICIENT_ANSWER in prompt.rendered
        assert "no excerpt" in prompt.rendered.lower()

    def test_french_question_gets_french_instructions(self):
        """The answer must come back in the language the question was asked in."""
        french = PromptBuilder().build("Quel est le delai de remboursement ?", EVIDENCE)
        english = PromptBuilder().build("What is the reimbursement delay?", EVIDENCE)
        assert french.rendered != english.rendered

    def test_prompt_is_deterministic(self):
        builder = PromptBuilder()
        first = builder.build("Quel est le plafond ?", EVIDENCE)
        second = builder.build("Quel est le plafond ?", EVIDENCE)
        assert first.rendered == second.rendered
        assert first.meta.checksum == second.meta.checksum

    def test_token_accounting_is_reported(self):
        prompt = PromptBuilder().build("Quel est le plafond ?", EVIDENCE)
        assert prompt.prompt_tokens > 0
        assert prompt.context_tokens > 0
        assert 0.0 <= prompt_efficiency(prompt) <= 1.0


class TestSimulatedBackend:
    def test_every_artifact_is_flagged_as_simulated(self):
        backend = SimulatedBackend(ModelBinding(model_id="none", path=""),
                                   GenerationPolicy())
        backend.load()
        prompt = PromptBuilder().build("Quel est le plafond du midi ?", EVIDENCE)
        result = backend.generate(prompt.rendered)
        assert result.is_simulated is True
        assert result.text

    def test_it_never_claims_a_model_identity(self):
        backend = SimulatedBackend(ModelBinding(model_id="none", path=""),
                                   GenerationPolicy())
        assert backend.describe()["is_simulated"] is True


class TestBackendSelection:
    def test_explicit_simulated_selection(self):
        backend = build_backend("simulated", ModelBinding(model_id="x", path=""),
                                GenerationPolicy())
        assert backend.is_simulated

    def test_unknown_backend_is_rejected(self):
        with pytest.raises(LeanLMError):
            build_backend("telepathy", ModelBinding(model_id="x", path=""),
                          GenerationPolicy())

    def test_a_remote_server_url_is_refused(self):
        """TL-01: a backend on another host would make the offline claim false."""
        with pytest.raises(LeanLMError):
            build_backend("llama-server", ModelBinding(model_id="x", path=""),
                          GenerationPolicy(), base_url="http://198.51.100.7:8080")

    def test_verify_model_reports_a_missing_file(self):
        with pytest.raises(LeanLMError):
            verify_model(ModelBinding(model_id="x", path="/nonexistent/model.gguf"))


class TestBinaryBackendInvocation:
    """The llama-cli command line, which is where a working setup went silent.

    Reported from Windows: model verified, backend selected, and every answer
    empty after a five-minute wait. Two causes, both in the flags.

    The arguments are captured by a fake binary that records them, rather than
    by patching subprocess -- so the test exercises the call the backend really
    makes, including how it starts the process.
    """

    def _invoke(self, tmp_path) -> tuple[list[str], object]:
        import stat as stat_module

        from leanlm.packages.inference.backends import LlamaCppBinaryBackend
        record = tmp_path / "argv.txt"
        fake = tmp_path / "llama-cli"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            f'printf "%s\\n" "$@" > "{record}"\n'
            "if [ -t 0 ]; then echo 'stdin is a tty' >&2; fi\n"
            "printf 'an answer\\n'\n"
            "echo 'llama_perf: eval time = 480.00 ms / 12 tokens' >&2\n",
            encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 40)
        backend = LlamaCppBinaryBackend(
            ModelBinding(model_id="m", path=str(model)), GenerationPolicy(),
            binary=str(fake))
        result = backend.generate("prompt")
        return record.read_text(encoding="utf-8").splitlines(), result

    def test_conversation_mode_is_disabled(self, tmp_path):
        """Every instruct model carries a chat template, and llama-cli switches
        to conversation mode when it sees one -- then waits on stdin instead of
        completing the prompt."""
        argv, _ = self._invoke(tmp_path)
        assert "-no-cnv" in argv

    def test_logging_is_not_disabled(self, tmp_path):
        """--log-disable also suppressed the timing lines this backend parses,
        so every run reported 0 tok/s."""
        argv, _ = self._invoke(tmp_path)
        assert "--log-disable" not in argv

    def test_the_model_and_the_limits_are_passed(self, tmp_path):
        argv, _ = self._invoke(tmp_path)
        assert "-m" in argv and "-n" in argv and "-c" in argv

    def test_the_timings_are_parsed_from_stderr(self, tmp_path):
        _, result = self._invoke(tmp_path)
        assert result.generated_tokens == 12

    def test_stdin_is_not_a_terminal(self, tmp_path):
        """An interactive prompt must reach EOF rather than block."""
        import subprocess
        from unittest.mock import patch

        from leanlm.packages.inference.backends import LlamaCppBinaryBackend
        captured: dict = {}
        original = subprocess.Popen

        def _popen(command, **kwargs):
            captured.update(kwargs)
            return original(command, **kwargs)

        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 40)
        fake = tmp_path / "cli"
        fake.write_text("#!/usr/bin/env bash\nprintf 'x\\n'\n", encoding="utf-8")
        import stat as stat_module
        fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
        backend = LlamaCppBinaryBackend(
            ModelBinding(model_id="m", path=str(model)), GenerationPolicy(),
            binary=str(fake))
        with patch("subprocess.Popen", _popen):
            backend.generate("prompt")
        assert captured["stdin"] == subprocess.DEVNULL

    def test_an_empty_generation_raises_rather_than_returning_nothing(self, tmp_path):
        """It reported "(no answer)", scored 0% on every probe, and explained
        nothing."""
        import stat as stat_module

        from leanlm.packages.inference.backends import LlamaCppBinaryBackend
        fake = tmp_path / "silent"
        fake.write_text("#!/usr/bin/env bash\necho 'diagnostic' >&2\nexit 0\n",
                        encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 40)
        backend = LlamaCppBinaryBackend(
            ModelBinding(model_id="m", path=str(model)), GenerationPolicy(),
            binary=str(fake))
        with pytest.raises(LeanLMError) as excinfo:
            backend.generate("prompt")
        record = excinfo.value.record
        assert record.code == "INF-124"
        assert "conversation mode" in record.recommended_action
        assert record.details.get("stderr")


class TestStreaming:
    """A slow backend that shows nothing is indistinguishable from a hung one.

    Reported from a two-core laptop: two runs abandoned with Ctrl-C because the
    terminal stayed silent for minutes. The output is now echoed as it arrives,
    and the first-token latency is measured rather than estimated.
    """

    def _backend(self, script: str, tmp_path):
        import os
        import stat as stat_module

        from leanlm.packages.inference.backends import LlamaCppBinaryBackend
        fake = tmp_path / "llama-cli"
        fake.write_text(script, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 40)
        backend = LlamaCppBinaryBackend(
            ModelBinding(model_id="m", path=str(model)), GenerationPolicy(),
            binary=str(fake))
        os.environ["PATH"] = f"{tmp_path}{os.pathsep}{os.environ['PATH']}"
        return backend

    def test_tokens_arrive_before_the_process_ends(self, tmp_path):
        backend = self._backend(
            "#!/usr/bin/env bash\nprintf 'part one '\nsleep 0.4\nprintf 'part two\\n'\n",
            tmp_path)
        chunks: list[str] = []
        backend.generate("prompt", on_token=chunks.append)
        assert chunks
        assert "".join(chunks).strip().startswith("part one")

    def test_the_first_token_latency_is_measured_not_estimated(self, tmp_path):
        """It used to be 30% of the total whenever llama.cpp's own timings were
        missing -- a fabricated number in a field judges read."""
        backend = self._backend(
            "#!/usr/bin/env bash\nsleep 0.3\nprintf 'answer\\n'\nsleep 0.4\n",
            tmp_path)
        result = backend.generate("prompt")
        assert result.first_token_latency_ms >= 250
        assert result.first_token_latency_ms < result.inference_ms

    def test_the_timing_block_on_stderr_still_reaches_the_parser(self, tmp_path):
        """stderr is drained concurrently: a full pipe buffer would deadlock the
        reader, and the timings live there."""
        backend = self._backend(
            "#!/usr/bin/env bash\nprintf 'answer\\n'\n"
            "echo 'llama_perf: eval time = 480.00 ms / 12 tokens' >&2\n",
            tmp_path)
        assert backend.generate("prompt").generated_tokens == 12

    def test_a_silent_binary_still_raises(self, tmp_path):
        backend = self._backend("#!/usr/bin/env bash\nexit 0\n", tmp_path)
        with pytest.raises(LeanLMError) as excinfo:
            backend.generate("prompt")
        assert excinfo.value.record.code == "INF-124"


class TestProgressHeartbeat:
    """Streaming is not enough when the child buffers its output.

    A process writing to a pipe buffers stdout in blocks, so llama.cpp can
    deliver nothing at all for minutes even though it is working. The heartbeat
    is driven by the parent and reads llama.cpp's stderr for the phase, so it
    keeps ticking regardless of what the child has flushed.

    Reported from a two-core laptop: three runs abandoned with Ctrl-C, the last
    one after the announce line had printed and nothing followed.
    """

    def _slow_binary(self, tmp_path, script: str):
        import stat as stat_module

        from leanlm.packages.inference.backends import LlamaCppBinaryBackend
        fake = tmp_path / "slow-cli"
        fake.write_text(script, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 40)
        return LlamaCppBinaryBackend(
            ModelBinding(model_id="m", path=str(model)), GenerationPolicy(),
            binary=str(fake))

    def test_progress_is_reported_before_any_token(self, tmp_path):
        backend = self._slow_binary(tmp_path, (
            "#!/usr/bin/env bash\n"
            "echo 'llama_model_loader: loading' >&2\n"
            "sleep 2.2\n"
            "printf 'answer\\n'\n"))
        ticks: list[tuple[float, str]] = []
        backend.generate("prompt", on_progress=lambda s, p: ticks.append((s, p)))
        assert ticks, "no heartbeat while the binary was silent"
        assert ticks[0][0] >= 1.0

    def test_the_phase_comes_from_llama_cpp_stderr(self, tmp_path):
        backend = self._slow_binary(tmp_path, (
            "#!/usr/bin/env bash\n"
            "echo 'llama_model_loader: reading' >&2\n"
            "sleep 1.3\n"
            "echo 'prompt eval time = 900 ms' >&2\n"
            "sleep 1.3\n"
            "printf 'answer\\n'\n"))
        phases = []
        backend.generate("prompt", on_progress=lambda s, p: phases.append(p))
        assert "reading the model file" in phases
        assert "processing the prompt" in phases

    def test_no_heartbeat_thread_when_nobody_listens(self, tmp_path):
        """The hook is optional: benchmarks and tests must not pay for it."""
        backend = self._slow_binary(
            tmp_path, "#!/usr/bin/env bash\nprintf 'answer\\n'\n")
        assert backend.generate("prompt").text == "answer"

    def test_output_is_read_one_character_at_a_time(self):
        """`read(16)` blocks until sixteen characters exist, which adds a delay
        of its own on top of the child's buffering."""
        from pathlib import Path
        source = (Path(__file__).resolve().parents[2]
                  / "src/leanlm/packages/inference/backends.py").read_text(
                      encoding="utf-8")
        assert "process.stdout.read(1)" in source


class TestPromptDelivery:
    """The prompt travels in a file, not on the command line.

    On Windows an argument holding newlines and quotes depends on CreateProcess
    quoting rules, and the command line is capped near 32 000 characters. A
    prompt is several kilobytes of user text; neither risk is worth taking.
    """

    def _echo_binary(self, tmp_path):
        import stat as stat_module

        from leanlm.packages.inference.backends import LlamaCppBinaryBackend
        fake = tmp_path / "echo-cli"
        fake.write_text(
            "#!/usr/bin/env bash\n"
            'while [ $# -gt 0 ]; do [ "$1" = "-f" ] && f="$2"; shift; done\n'
            'cat "$f"\n',
            encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat_module.S_IEXEC)
        model = tmp_path / "m.gguf"
        model.write_bytes(b"GGUF" + b"\0" * 40)
        return LlamaCppBinaryBackend(
            ModelBinding(model_id="m", path=str(model)), GenerationPolicy(),
            binary=str(fake))

    def test_a_multiline_prompt_arrives_intact(self, tmp_path):
        backend = self._echo_binary(tmp_path)
        prompt = 'Rules:\n1. Use only the excerpts.\n\n### Question\nWhat "now"?'
        assert backend.generate(prompt).text.strip() == prompt.strip()

    def test_the_prompt_file_is_removed_afterwards(self, tmp_path):
        import os
        backend = self._echo_binary(tmp_path)
        backend.generate("a prompt")
        assert not os.path.exists(backend._last_prompt_file)

    def test_the_command_can_be_shown_without_running_it(self, tmp_path):
        backend = self._echo_binary(tmp_path)
        command = backend.describe_command("anything")
        assert "-no-cnv" in command
        assert "-f" in command
