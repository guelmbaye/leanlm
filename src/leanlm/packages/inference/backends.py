"""llama.cpp backends.

Four bindings, tried in this order of preference:

1. ``llama-cpp-python`` -- in process, streaming, exact tokenizer, best metrics;
2. ``llama-server``     -- a local HTTP server on 127.0.0.1 (still offline);
3. ``llama-cli``        -- the plain binary, when only a build is available;
4. ``simulated``        -- no model at all.

The simulated backend exists so the whole pipeline, the benchmark harness and
CI remain runnable on a machine with no GGUF file. It is not a fallback in
disguise: every artifact it produces is stamped ``is_simulated=true``, the
submission builder refuses to package simulated results, and the workspace
labels it in the UI. A number that did not come from a real model must never be
able to pass for one.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from typing import Any

from ...shared.clock import Stopwatch
from ...shared.errors import inference_error, resource_error
from ...shared.hashing import sha256_file
from ...shared.text import DEFAULT_TOKEN_COUNTER, LlamaTokenCounter, TokenCounter
from .models import GenerationResult, ModelBinding
from .policies import GenerationPolicy


class InferenceBackend(ABC):
    """Uniform surface over every way of reaching llama.cpp."""

    name: str = "abstract"
    is_simulated: bool = False

    def __init__(self, binding: ModelBinding, policy: GenerationPolicy) -> None:
        self.binding = binding
        self.policy = policy
        self._loaded = False

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, *, on_first_token=None) -> GenerationResult: ...

    def token_counter(self) -> TokenCounter:
        return DEFAULT_TOKEN_COUNTER

    def unload(self) -> None:
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def describe(self) -> dict[str, Any]:
        return {"backend": self.name, "model_id": self.binding.model_id,
                "quantization": self.binding.quantization,
                "context_tokens": self.binding.context_tokens,
                "is_simulated": self.is_simulated}


# ---------------------------------------------------------------------------


class LlamaCppPythonBackend(InferenceBackend):
    """In-process llama.cpp through ``llama-cpp-python``."""

    name = "llama-cpp-python"

    def __init__(self, binding: ModelBinding, policy: GenerationPolicy,
                 *, n_threads: int | None = None, n_gpu_layers: int = 0,
                 n_batch: int = 256, use_mmap: bool = True, use_mlock: bool = False) -> None:
        super().__init__(binding, policy)
        self._llama = None
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2) - 1)
        self.n_gpu_layers = n_gpu_layers
        self.n_batch = n_batch
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock

    def load(self) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except ImportError as exc:
            raise inference_error(
                "INF-101", "llama-cpp-python is not installed",
                capability="inference", stage="model_inference",
                recommended_action="pip install llama-cpp-python, or use the "
                                   "llama-cli / llama-server backend",
                detail=str(exc),
            ) from exc
        if not os.path.isfile(self.binding.path):
            raise inference_error(
                "INF-102", f"GGUF model not found: {self.binding.path}",
                capability="inference", stage="model_inference",
                recommended_action="run scripts/download_model.sh",
            )
        self._llama = Llama(
            model_path=self.binding.path,
            n_ctx=self.binding.context_tokens,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
            use_mmap=self.use_mmap,
            use_mlock=self.use_mlock,
            seed=self.policy.seed,
            verbose=False,
        )
        self._loaded = True

    def token_counter(self) -> TokenCounter:
        if self._llama is None:
            return DEFAULT_TOKEN_COUNTER
        return LlamaTokenCounter(self._llama.tokenize)

    def generate(self, prompt: str, *, on_first_token=None) -> GenerationResult:
        if self._llama is None:
            self.load()
        watch = Stopwatch()
        first_token_ms = 0.0
        chunks: list[str] = []
        generated = 0
        stop_reason = "eos"
        stream = self._llama(  # type: ignore[operator]
            prompt,
            max_tokens=self.policy.max_output_tokens,
            temperature=self.policy.temperature,
            top_k=self.policy.top_k,
            top_p=self.policy.top_p,
            repeat_penalty=self.policy.repeat_penalty,
            stop=list(self.policy.stop),
            stream=True,
        )
        for chunk in stream:
            piece = chunk["choices"][0].get("text", "")
            if not piece:
                continue
            if generated == 0:
                first_token_ms = watch.elapsed_ms
                if on_first_token:
                    on_first_token(first_token_ms)
            chunks.append(piece)
            generated += 1
            finish = chunk["choices"][0].get("finish_reason")
            if finish:
                stop_reason = finish
        total = watch.stop()
        return GenerationResult(
            text="".join(chunks).strip(), backend=self.name, generated_tokens=generated,
            prompt_tokens=len(self._llama.tokenize(prompt.encode("utf-8"))),  # type: ignore
            first_token_latency_ms=first_token_ms, inference_ms=total,
            stop_reason=stop_reason,
        )

    def unload(self) -> None:
        self._llama = None
        super().unload()


# ---------------------------------------------------------------------------


class LlamaCppServerBackend(InferenceBackend):
    """Local ``llama-server`` over loopback HTTP.

    Loopback only, and the host is validated: an offline runtime that quietly
    posts a prompt to a remote endpoint would break TL-01 and NFR-01.
    """

    name = "llama-server"
    _ALLOWED_HOSTS = ("127.0.0.1", "localhost", "::1")

    def __init__(self, binding: ModelBinding, policy: GenerationPolicy,
                 *, base_url: str = "http://127.0.0.1:8080") -> None:
        super().__init__(binding, policy)
        host = re.sub(r"^https?://", "", base_url).split(":")[0].split("/")[0]
        if host not in self._ALLOWED_HOSTS:
            raise inference_error(
                "INF-110", f"llama-server backend refuses a non-loopback host: {host}",
                capability="inference", stage="model_inference",
                recommended_action="LeanLM runs offline; point the backend at 127.0.0.1",
            )
        self.base_url = base_url.rstrip("/")

    def load(self) -> None:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as response:
                self._loaded = response.status == 200
        except (urllib.error.URLError, OSError) as exc:
            raise inference_error(
                "INF-111", f"no llama-server reachable at {self.base_url}",
                capability="inference", stage="model_inference",
                recommended_action=("start it: llama-server -m <model.gguf> "
                                    "--host 127.0.0.1 --port 8080"),
                detail=str(exc),
            ) from exc

    def generate(self, prompt: str, *, on_first_token=None) -> GenerationResult:
        payload = json.dumps({
            "prompt": prompt,
            "n_predict": self.policy.max_output_tokens,
            "temperature": self.policy.temperature,
            "top_k": self.policy.top_k,
            "top_p": self.policy.top_p,
            "repeat_penalty": self.policy.repeat_penalty,
            "seed": self.policy.seed,
            "stop": list(self.policy.stop),
            "stream": False,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/completion", data=payload,
            headers={"Content-Type": "application/json"},
        )
        watch = Stopwatch()
        try:
            with urllib.request.urlopen(request, timeout=self.policy.timeout_s) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise inference_error(
                "INF-112", "llama-server call failed",
                capability="inference", stage="model_inference",
                recommended_action="check the server logs", detail=str(exc),
            ) from exc
        total = watch.stop()
        timings = body.get("timings", {}) or {}
        first_ms = float(timings.get("prompt_ms", 0.0)) or total * 0.25
        generated = int(timings.get("predicted_n", 0)) or DEFAULT_TOKEN_COUNTER.count(
            body.get("content", ""))
        if on_first_token:
            on_first_token(first_ms)
        return GenerationResult(
            text=(body.get("content") or "").strip(), backend=self.name,
            generated_tokens=generated,
            prompt_tokens=int(timings.get("prompt_n", 0)) or DEFAULT_TOKEN_COUNTER.count(prompt),
            first_token_latency_ms=round(first_ms, 3), inference_ms=total,
            stop_reason=body.get("stop_type", "eos"), extra={"timings": timings},
        )


# ---------------------------------------------------------------------------


class LlamaCppBinaryBackend(InferenceBackend):
    """The ``llama-cli`` binary, driven as a subprocess."""

    name = "llama-cli"
    _TIMING = re.compile(r"eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*(?:runs|tokens)")

    def __init__(self, binding: ModelBinding, policy: GenerationPolicy,
                 *, binary: str = "llama-cli", n_threads: int | None = None,
                 n_gpu_layers: int = 0) -> None:
        super().__init__(binding, policy)
        self.binary = binary
        self.n_threads = n_threads or max(1, (os.cpu_count() or 2) - 1)
        self.n_gpu_layers = n_gpu_layers

    def describe_command(self, prompt: str) -> str:
        """The exact invocation, for running by hand.

        When a request appears to hang, the fastest diagnosis is to run the same
        command in a terminal: attached to a tty, llama.cpp flushes every token,
        so anything wrong becomes visible immediately.
        """
        parts = [
            self.binary, "-m", self.binding.path, "-f", "<prompt file>",
            "-n", str(self.policy.max_output_tokens),
            "-c", str(self.binding.context_tokens),
            "-t", str(self.n_threads), "--temp", str(self.policy.temperature),
            "-ngl", str(self.n_gpu_layers), "--no-display-prompt", "-no-cnv",
        ]
        return " ".join(f'"{p}"' if " " in p else p for p in parts)

    def load(self) -> None:
        if shutil.which(self.binary) is None:
            raise inference_error(
                "INF-120", f"llama.cpp binary not found: {self.binary}",
                capability="inference", stage="model_inference",
                recommended_action="build llama.cpp and put llama-cli on PATH",
            )
        if not os.path.isfile(self.binding.path):
            raise inference_error(
                "INF-121", f"GGUF model not found: {self.binding.path}",
                capability="inference", stage="model_inference",
                recommended_action="run scripts/download_model.sh",
            )
        self._loaded = True

    # llama.cpp writes these to stderr while it works. They are the only sign of
    # life available before the first token, because a child process writing to
    # a pipe buffers its stdout in blocks -- so "streaming" can deliver nothing
    # at all until several kilobytes have accumulated.
    _PHASES = (
        ("load_tensors", "loading the model"),
        ("llama_model_loader", "reading the model file"),
        ("llama_context", "allocating the context"),
        ("kv cache", "allocating the KV cache"),
        ("prompt eval", "processing the prompt"),
        ("sampling", "generating"),
    )

    def generate(self, prompt: str, *, on_first_token=None,
                 on_token=None, on_progress=None) -> GenerationResult:
        if not self._loaded:
            self.load()
        # The prompt travels in a file rather than on the command line. On
        # Windows an argument containing newlines and quotes is at the mercy of
        # CreateProcess quoting rules, and the whole command line is capped at
        # about 32 000 characters -- neither risk is worth taking for a value
        # that is several kilobytes of user text.
        prompt_file = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False, newline="\n")
        prompt_file.write(prompt)
        prompt_file.close()
        self._last_prompt_file = prompt_file.name

        command = [
            self.binary, "-m", self.binding.path, "-f", prompt_file.name,
            "-n", str(self.policy.max_output_tokens),
            "-c", str(self.binding.context_tokens),
            "-t", str(self.n_threads),
            "--temp", str(self.policy.temperature),
            "--top-k", str(self.policy.top_k),
            "--top-p", str(self.policy.top_p),
            "--repeat-penalty", str(self.policy.repeat_penalty),
            "--seed", str(self.policy.seed),
            "-ngl", str(self.n_gpu_layers),
            "--no-display-prompt", "--simple-io",
            # Without this, llama-cli enters conversation mode whenever the model
            # carries a chat template -- which every instruct model does -- and
            # waits on stdin instead of completing the prompt. The symptom is a
            # run that produces nothing and ends at the timeout.
            "-no-cnv",
        ]
        # `--log-disable` used to be here. It also suppressed the timing lines
        # this backend parses out of stderr, so every run reported 0 tok/s.
        watch = Stopwatch()
        # Streamed rather than captured in one go. On a slow CPU a request takes
        # minutes, and `subprocess.run` gives the caller nothing until it ends:
        # the terminal sits silent and the only way to tell working from hung is
        # to wait or to give up. Streaming also yields a *measured* first-token
        # latency instead of the fraction of the total this backend used to
        # invent when llama.cpp's own timings were unavailable.
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL, text=True, bufsize=1,
        )

        errors: list[str] = []

        def _drain() -> None:
            # stderr must be consumed concurrently: llama.cpp writes its timing
            # block there, and a full pipe buffer would deadlock the reader.
            assert process.stderr is not None
            for line in process.stderr:
                errors.append(line)

        drainer = threading.Thread(target=_drain, daemon=True)
        drainer.start()

        # A heartbeat driven by the parent, so it keeps ticking even when the
        # child has flushed nothing. Without it a slow run is indistinguishable
        # from a hung one, and the only recourse is Ctrl-C.
        heartbeat_stop = threading.Event()

        def _heartbeat() -> None:
            phase = "starting"
            while not heartbeat_stop.wait(1.0):
                recent = "".join(errors[-40:]).lower()
                for needle, label in self._PHASES:
                    if needle in recent:
                        phase = label
                if on_progress:
                    on_progress(watch.elapsed_ms / 1000.0, phase)

        if on_progress:
            ticker = threading.Thread(target=_heartbeat, daemon=True)
            ticker.start()
        else:
            ticker = None

        chunks: list[str] = []
        first_ms: float | None = None
        deadline = time.monotonic() + self.policy.timeout_s
        try:
            assert process.stdout is not None
            # One character at a time: whatever the child flushes appears at
            # once, rather than waiting for a fixed-size block to fill.
            for chunk in iter(lambda: process.stdout.read(1), ""):
                if not chunk:
                    break
                if first_ms is None and chunk.strip():
                    first_ms = watch.elapsed_ms
                    if on_first_token:
                        on_first_token(first_ms)
                chunks.append(chunk)
                if on_token:
                    on_token(chunk)
                if time.monotonic() > deadline:
                    process.kill()
                    raise inference_error(
                        "INF-122",
                        f"llama-cli exceeded {self.policy.timeout_s:.0f}s",
                        capability="inference", stage="model_inference",
                        recommended_action=(
                            "on a slow CPU this can simply be the truth: a 2-core "
                            "laptop generating 384 tokens takes minutes. Lower "
                            "inference.max_output_tokens, or measure on faster "
                            "hardware"),
                        partial_output="".join(chunks)[-200:],
                    )
            returncode = process.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            process.kill()
            raise inference_error(
                "INF-122", f"llama-cli exceeded {self.policy.timeout_s:.0f}s",
                capability="inference", stage="model_inference",
                recommended_action=(
                    "on a slow CPU this can simply be the truth. Lower "
                    "inference.max_output_tokens, or measure on faster hardware"),
            ) from exc
        finally:
            try:
                os.unlink(prompt_file.name)
            except OSError:
                pass
            heartbeat_stop.set()
            if ticker is not None:
                ticker.join(timeout=2.0)
            drainer.join(timeout=2.0)

        total = watch.stop()
        stderr = "".join(errors)

        if returncode != 0:
            if "-no-cnv" in command and (
                    "unknown argument" in stderr.lower()
                    or "invalid argument" in stderr.lower()):
                # Flags come and go between llama.cpp builds. Retry once without
                # the one most likely to be missing rather than failing the run.
                retry = [arg for arg in command if arg != "-no-cnv"]
                completed = subprocess.run(retry, capture_output=True, text=True,
                                           timeout=self.policy.timeout_s,
                                           stdin=subprocess.DEVNULL)
                chunks, stderr, returncode = ([completed.stdout], completed.stderr,
                                              completed.returncode)
            if returncode != 0:
                raise inference_error(
                    "INF-123", "llama-cli returned a non-zero exit code",
                    capability="inference", stage="model_inference",
                    recommended_action="run the command manually to inspect the error",
                    stderr=stderr[-400:],
                )

        text = "".join(chunks).strip()
        if not text:
            # An empty generation is a failure, not an answer. Returning it
            # silently produced a run that reported "(no answer)", scored 0% on
            # every probe, and gave no indication of why.
            raise inference_error(
                "INF-124", "llama-cli produced no output",
                capability="inference", stage="model_inference",
                recommended_action=(
                    "the model loaded but generated nothing. Usual causes: the "
                    "binary entered conversation mode and read EOF, or the chat "
                    "template rejected the prompt. Run the command by hand to see "
                    "what it prints"),
                stderr=stderr[-600:],
                command=" ".join(command[:6]) + " ...",
            )

        eval_ms, generated = 0.0, 0
        match = self._TIMING.search(stderr)
        if match:
            eval_ms, generated = float(match.group(1)), int(match.group(2))
        generated = generated or DEFAULT_TOKEN_COUNTER.count(text)
        if first_ms is None:
            first_ms = total

        return GenerationResult(
            text=text, backend=self.name, generated_tokens=generated,
            prompt_tokens=DEFAULT_TOKEN_COUNTER.count(prompt),
            first_token_latency_ms=round(first_ms, 3), inference_ms=total,
        )


# ---------------------------------------------------------------------------


class SimulatedBackend(InferenceBackend):
    """Extractive stand-in used when no GGUF model is present.

    It answers by quoting the highest ranked excerpt verbatim with its citation
    label. That keeps the pipeline, the validator and the harness exercisable
    end to end, and it keeps the output honest: the text is a quotation, not a
    generation, and every artifact says so.
    """

    name = "simulated"
    is_simulated = True

    def load(self) -> None:
        self._loaded = True

    @staticmethod
    def _refusal(prompt: str) -> str:
        """Decline in the language of the prompt.

        A French refusal to an English question is not merely untidy: the
        validator matches insufficiency on the exact sentence the prompt asked
        for, so the wrong language turns a correct refusal into a failed one.
        """
        from .services import INSUFFICIENT_ANSWER, INSUFFICIENT_ANSWER_FR

        return (INSUFFICIENT_ANSWER_FR
                if "les extraits" in prompt.lower() or "la question" in prompt.lower()
                else INSUFFICIENT_ANSWER)

    @staticmethod
    def _extract_unlabelled(prompt: str) -> str:
        from ...shared.text import content_words

        question_match = re.search(r"^Question:\s*(.+)$", prompt, flags=re.MULTILINE)
        question = question_match.group(1) if question_match else ""
        body = prompt
        for marker in ("Documents:", "### Excerpts"):
            if marker in prompt:
                body = prompt.split(marker, 1)[1]
                break
        if question_match:
            body = body.split("Question:", 1)[0]

        # When the excerpts block is the empty marker there is nothing to quote,
        # and quoting the marker itself produced answers that opened with
        # "(no excerpt matched this question)". A stand-in that echoes the prompt
        # is worse than one that declines.
        if "no excerpt matched" in body or not body.strip():
            return SimulatedBackend._refusal(prompt)

        wanted = set(content_words(question))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n", body)
                     if len(s.strip()) > 25]
        if not sentences:
            return "Les documents fournis ne permettent pas de conclure."
        if not wanted:
            return " ".join(sentences[:2])
        ranked = sorted(
            sentences,
            key=lambda s: (-len(wanted & set(content_words(s))), sentences.index(s)),
        )
        if not (wanted & set(content_words(ranked[0]))):
            return SimulatedBackend._refusal(prompt)
        return " ".join(ranked[:2])

    def generate(self, prompt: str, *, on_first_token=None) -> GenerationResult:
        watch = Stopwatch()
        excerpts = re.findall(r"^\[(S\d+)\][^\n]*\n(.+?)(?=\n\n\[S\d+\]|\n\n### |\Z)",
                              prompt, flags=re.MULTILINE | re.DOTALL)
        if excerpts:
            label, body = excerpts[0]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", body.strip()) if s.strip()]
            answer = " ".join(sentences[:3]) + f" [{label}]"
        else:
            # An unlabelled prompt -- the naive baseline path. The simulator must
            # behave the same way here: quote the most question-relevant lines of
            # whatever context it was handed, without a citation label because
            # none was requested. Returning a refusal instead would hand LeanLM a
            # grounding advantage that came from the simulator rather than from
            # the optimization layer, and would make the baseline comparison a
            # rigged one.
            answer = self._extract_unlabelled(prompt)
        if on_first_token:
            on_first_token(watch.elapsed_ms)
        first_ms = watch.elapsed_ms
        total = watch.stop()
        return GenerationResult(
            text=answer, backend=self.name,
            generated_tokens=DEFAULT_TOKEN_COUNTER.count(answer),
            prompt_tokens=DEFAULT_TOKEN_COUNTER.count(prompt),
            first_token_latency_ms=round(first_ms, 3),
            inference_ms=max(total, 0.001), is_simulated=True,
        )


# ---------------------------------------------------------------------------


BACKENDS = {
    LlamaCppPythonBackend.name: LlamaCppPythonBackend,
    LlamaCppServerBackend.name: LlamaCppServerBackend,
    LlamaCppBinaryBackend.name: LlamaCppBinaryBackend,
    SimulatedBackend.name: SimulatedBackend,
}


def verify_model(binding: ModelBinding) -> str:
    """TL-02 model integrity: the loaded file must be the expected file."""
    if not os.path.isfile(binding.path):
        raise resource_error(
            "TRU-001", f"model file missing: {binding.path}",
            capability="inference", stage="model_inference",
            recommended_action="run scripts/download_model.sh",
        )
    checksum = sha256_file(binding.path)
    if binding.expected_checksum and checksum != binding.expected_checksum:
        from ...shared.errors import trust_error
        raise trust_error(
            "TRU-002", "model checksum mismatch -- refusing to load",
            capability="inference", stage="model_inference",
            recommended_action="re-download the model; do not run with an unverified file",
            expected=binding.expected_checksum, actual=checksum,
        )
    return checksum


def build_backend(name: str, binding: ModelBinding, policy: GenerationPolicy,
                  **options: Any) -> InferenceBackend:
    if name == "auto":
        name = _auto_select(binding)
    if name not in BACKENDS:
        raise inference_error(
            "INF-100", f"unknown inference backend: {name}",
            capability="inference", stage="model_inference",
            recommended_action=f"choose one of: {', '.join(sorted(BACKENDS))}",
        )
    return BACKENDS[name](binding, policy, **options)  # type: ignore[arg-type]


def _auto_select(binding: ModelBinding) -> str:
    has_model = bool(binding.path) and os.path.isfile(binding.path)
    if has_model:
        try:
            import llama_cpp  # type: ignore # noqa: F401
            return LlamaCppPythonBackend.name
        except ImportError:
            pass
        if shutil.which("llama-cli"):
            return LlamaCppBinaryBackend.name
    return SimulatedBackend.name
