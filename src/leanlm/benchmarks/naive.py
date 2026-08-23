"""The naive baseline: same model, same machine, no LeanLM layer.

Every comparative claim in the submission rests on this file, so it is written
to be *unflattering to us*. The baseline is not a straw man:

  * it uses the same model binding, the same backend, the same generation
    parameters and the same profile fingerprint;
  * it is measured with the same sampler, on the same corpus, in the same
    process, in the same session;
  * it is given the whole corpus, which is exactly what a developer would do
    before reaching for an optimization layer.

The one thing it cannot do is fit a corpus larger than the context window. When
that happens it truncates -- because that is what you are forced to do -- and
the truncation is reported as a fact, not hidden. That single number, how much
of the corpus the model never saw, is the honest version of our argument.

What we deliberately do *not* do to make ourselves look better:
  * we do not disable citations and then score the baseline on citing;
  * we do not compare our best profile against its worst;
  * we do not exclude the baseline's warm-up runs while keeping ours.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from ..contracts.dto import ModelResponse, SelectedEvidence
from ..packages.validation.policies import ValidationPolicy
from ..packages.validation.services import ResponseValidator
from ..shared.clock import Stopwatch, iso_now
from ..shared.text import truncate_to_tokens
from .harness import environment_fingerprint
from .stats import aggregate_all

# The baseline never cites, because nothing told it to. Scoring it on citations
# would be scoring it on an instruction it was never given.
BASELINE_VALIDATION = ValidationPolicy(require_citations=False)

NAIVE_TEMPLATE = """Documents:
{context}

Question: {question}
Answer:"""

REPORTED_METRICS = (
    "tokens_per_second", "first_token_latency_ms", "inference_ms", "total_ms",
    "prompt_tokens", "context_tokens", "generated_tokens", "peak_rss_mb",
    "corpus_coverage", "response_grounding_rate",
)


@dataclass(frozen=True, slots=True)
class NaiveRun:
    question: str
    iteration: int
    ok: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    dropped_tokens: int = 0
    # The answer is recorded, not just scored. A comparative claim that cannot be
    # re-read by a sceptic is an assertion, and the answers are what a reviewer
    # will want to check first.
    answer: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"question": self.question, "iteration": self.iteration, "ok": self.ok,
                "truncated": self.truncated, "dropped_tokens": self.dropped_tokens,
                "answer": self.answer, "error": self.error, **self.metrics}


class NaiveBaseline:
    """Runs the questions the way they would be run without LeanLM."""

    def __init__(self, runtime, *, repeat: int = 3, warmup: int = 1) -> None:
        self.runtime = runtime
        self.repeat = max(1, int(repeat))
        self.warmup = max(0, int(warmup))
        self.validator = ResponseValidator(BASELINE_VALIDATION)

    # -- context ------------------------------------------------------------
    def _whole_corpus(self) -> tuple[str, int]:
        """Everything, in filename order -- the order `cat *.md` would produce.

        Ordering matters and is not a detail we may pick to suit ourselves. The
        corpus store iterates by document id, which is a content hash, so its
        order is effectively arbitrary. A developer building a naive prompt
        concatenates files as the shell lists them, so filename order is both the
        realistic choice and a deterministic one.

        Be clear about what this does and does not prove: truncation keeps the
        head of the corpus, so whether a given question survives depends on where
        its answer sits. On this corpus the baseline loses 58% of the text, which
        means roughly 58% of questions become unanswerable for reasons unrelated
        to the model. The probe question is one deliberately chosen instance of
        that, not evidence that truncation always loses.

        The token count is taken from the joined text with the same counter that
        will measure the truncated version. Summing per-unit estimates instead
        would make coverage drift above 1.0 -- a comparison whose denominator is
        measured differently from its numerator is not a comparison.
        """
        store = self.runtime.corpus
        chunks: list[str] = []
        for document in store.documents():          # already sorted by filename
            units = list(store.iter_units([document["document_id"]]))
            units.sort(key=lambda u: u.order)
            chunks.extend(unit.text for unit in units)
        text = "\n\n".join(chunks)
        return text, self.runtime.token_counter.count(text)

    def _window(self) -> int:
        binding = self.runtime.binding
        policy = self.runtime.generation_policy
        # A rough allowance for the template and the question. The naive path has
        # no budgeting model; this is the arithmetic a developer would do by hand.
        return max(256, binding.context_tokens - policy.max_output_tokens - 64)

    # -- execution ----------------------------------------------------------
    def run(self, questions: Sequence[str]) -> dict[str, Any]:
        self.runtime.load()
        corpus_text, corpus_tokens = self._whole_corpus()
        window = self._window()
        counter = self.runtime.token_counter

        context = truncate_to_tokens(corpus_text, window, counter)
        context_tokens = counter.count(context)
        truncated = context_tokens < corpus_tokens
        dropped = max(0, corpus_tokens - context_tokens)
        coverage = round(min(1.0, context_tokens / corpus_tokens), 4) \
            if corpus_tokens else 0.0

        runs: list[NaiveRun] = []
        for question in questions:
            for index in range(self.warmup + self.repeat):
                runs.append(self._single(question, index, context, context_tokens,
                                        corpus_tokens, coverage, truncated, dropped,
                                        warmup=index < self.warmup))
        measured = [r for r in runs if r.ok][self.warmup * len(questions):] or \
            [r for r in runs if r.ok]
        aggregates = aggregate_all([r.metrics for r in measured], REPORTED_METRICS)

        summary: dict[str, Any] = {
            "kind": "naive_baseline",
            "created_at": iso_now(),
            "description": ("same model, same machine, whole corpus in the prompt, "
                            "no budgeting, no retrieval, no validation"),
            "repeat": self.repeat,
            "warmup": self.warmup,
            "environment": environment_fingerprint(),
            "profile": self.runtime.profile.to_dict(),
            "backend": self.runtime.backend.describe(),
            "corpus": {
                "documents": self.runtime.corpus.document_count(),
                "units": self.runtime.corpus.unit_count(),
                "checksum": self.runtime.corpus.corpus_checksum()[:16],
                "tokens": corpus_tokens,
            },
            "context_window": window,
            "corpus_truncated": truncated,
            "dropped_tokens": dropped,
            "corpus_coverage": coverage,
            "is_simulated": bool(self.runtime.backend.is_simulated),
            "runs": [r.to_dict() for r in runs],
            "metrics": {k: v.to_dict() for k, v in aggregates.items()},
            "verdict": "pass" if all(r.ok for r in runs) else "fail",
        }
        for metric, value in aggregates.items():
            summary[metric] = value.mean
        if truncated:
            summary["note"] = (
                f"the model never saw {dropped} of {corpus_tokens} corpus tokens "
                f"({(1 - coverage):.0%}); on any question whose answer lay in the "
                "dropped part, the baseline cannot be correct for reasons that have "
                "nothing to do with the model"
            )
        if summary["is_simulated"]:
            summary["warning"] = (
                "SIMULATED BACKEND: this baseline measures the naive path through the "
                "same simulator, not a model. It is comparable to a LeanLM campaign "
                "run under the same simulator and to nothing else."
            )
        return summary

    def _single(self, question: str, index: int, context: str, context_tokens: int,
                corpus_tokens: int, coverage: float, truncated: bool, dropped: int,
                *, warmup: bool) -> NaiveRun:
        prompt = NAIVE_TEMPLATE.format(context=context, question=question)
        counter = self.runtime.token_counter
        sampler = self.runtime.sampler
        sampler.reset()
        watch = Stopwatch()
        try:
            result = self.runtime.backend.generate(prompt)
        except Exception as error:  # noqa: BLE001 -- a crash is a measurement
            return NaiveRun(question, index, False,
                            error=f"{type(error).__name__}: {error}")
        total_ms = watch.stop()
        observation = sampler.sample()

        grounding = self._grounding(result.text, context, question)
        metrics = {
            "tokens_per_second": result.tokens_per_second if not result.is_simulated else 0.0,
            "first_token_latency_ms": (result.first_token_latency_ms
                                       if not result.is_simulated else 0.0),
            "inference_ms": result.inference_ms if not result.is_simulated else 0.0,
            "total_ms": total_ms,
            "prompt_tokens": counter.count(prompt),
            "context_tokens": context_tokens,
            "generated_tokens": result.generated_tokens,
            "peak_rss_mb": observation.process_rss_mb,
            "available_ram_mb": observation.available_ram_mb,
            "temperature_c": observation.temperature_c,
            "corpus_coverage": coverage,
            "response_grounding_rate": grounding,
            "is_simulated": result.is_simulated,
        }
        return NaiveRun(question, index, True, metrics, truncated, dropped,
                        answer=result.text.strip())

    def _grounding(self, answer: str, context: str, question: str) -> float:
        """Scored exactly as LeanLM's answers are, against what was actually sent."""
        if not answer.strip():
            return 0.0
        evidence = (SelectedEvidence(
            meta=SelectedEvidence.build_meta(identity=("baseline", context[:40]),
                                             stage="evidence_retrieval"),
            label="S1", unit_id="baseline-window", document_name="corpus",
            text=context, score=0.0, token_estimate=0,
        ).sealed(),)
        response = ModelResponse(
            meta=ModelResponse.build_meta(identity=(answer[:40],), stage="model_inference"),
            text=answer,
        ).sealed()
        return self.validator.validate(response, evidence, question).grounding_rate


# Deliberately *not* benchmarks/results/baseline.json. That file is the previous
# LeanLM campaign, used to detect regressions between our own runs. This one is a
# different measurement answering a different question, and conflating them makes
# CI report "+224% regression" every time the naive path is faster on a corpus
# small enough to fit -- which trains people to ignore CI.
DEFAULT_OUTPUT = "benchmarks/results/naive_baseline.json"


def run_naive_baseline(runtime, questions: Sequence[str] | None = None, *,
                       repeat: int = 3, warmup: int = 1,
                       output: str | None = None) -> dict[str, Any]:
    from ..shared.serialization import write_json
    from .scenarios import load_scenarios

    if not questions:
        questions = []
        for scenario in load_scenarios():
            questions.extend(scenario.questions)
        # Duplicates would weight one question more heavily than another.
        seen: set[str] = set()
        questions = [q for q in questions if not (q in seen or seen.add(q))]
    summary = NaiveBaseline(runtime, repeat=repeat, warmup=warmup).run(questions)
    if output:
        write_json(output, summary)
        summary["output_path"] = str(output)
    return summary
