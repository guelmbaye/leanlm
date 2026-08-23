"""Accuracy measurement against ground truth.

Accuracy carries the largest weight in the ADTC rubric, and until this module
existed LeanLM measured everything *except* it: grounding, compression, latency,
memory -- all of them proxies for "the answer is good", none of them a check that
the answer is *correct*.

What is scored here, and why each part is separate:

  **answer accuracy** -- does the answer contain the fact the corpus states? A
  fact-level check, not a similarity score: for enterprise policy questions the
  answer is a figure, and a figure is either right or wrong.

  **source accuracy** -- did the citation point at the document that actually
  says it? An answer that is right while citing the wrong document is right by
  luck, and luck does not survive a corpus change.

  **refusal accuracy** -- on questions the corpus cannot answer, did the system
  decline? Scored in the same table as the rest, because a system that answers
  everything is not more accurate, it is less honest.

  **hallucination rate** -- answered when it should have declined. Tracked
  separately from plain wrongness because the two failures have different costs:
  a wrong figure is caught by whoever knows the policy, an invented one is not.

Deliberately absent: partial credit. A ceiling of 25 EUR reported as 35 EUR is
not 70% correct, it is wrong, and a scorer that says otherwise makes the number
comfortable rather than useful.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from ..shared.clock import iso_now
from ..shared.errors import config_error
from ..shared.text import normalize_text
from .stats import aggregate_all


@dataclass(frozen=True, slots=True)
class Probe:
    """One ground-truth question."""

    id: str
    question: str
    must_contain: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    match: str = "all"                    # "all" or "any"
    expected_source: str = ""
    unanswerable: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Probe":
        return cls(
            id=str(payload.get("id", "?")),
            question=str(payload["question"]),
            must_contain=tuple(payload.get("must_contain") or ()),
            must_not_contain=tuple(payload.get("must_not_contain") or ()),
            match=str(payload.get("match", "all")),
            expected_source=str(payload.get("expected_source", "")),
            unanswerable=bool(payload.get("unanswerable", False)),
            note=str(payload.get("note", "")),
        )


@dataclass(frozen=True, slots=True)
class ProbeResult:
    probe_id: str
    question: str
    answered: bool
    correct: bool
    source_correct: bool | None
    declined: bool
    hallucinated: bool
    answer: str = ""
    cited_sources: tuple[str, ...] = ()
    confidence: str = ""
    failure: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id, "question": self.question,
            "answered": self.answered, "correct": self.correct,
            "source_correct": self.source_correct, "declined": self.declined,
            "hallucinated": self.hallucinated, "answer": self.answer,
            "cited_sources": list(self.cited_sources), "confidence": self.confidence,
            "failure": self.failure,
        }


def load_probes(path: str | Path) -> tuple[tuple[Probe, ...], dict[str, Any]]:
    target = Path(path)
    if not target.is_file():
        raise config_error(
            "EVAL-001", f"no evaluation set at {target}",
            capability="benchmarks",
            recommended_action="accuracy is the heaviest ADTC criterion; it needs a "
                               "ground-truth file next to the corpus",
        )
    payload = json.loads(target.read_text(encoding="utf-8"))
    probes = tuple(Probe.from_dict(entry) for entry in payload.get("questions", []))
    if not probes:
        raise config_error("EVAL-002", f"{target} declares no questions",
                           capability="benchmarks")
    meta = {"version": payload.get("version", "0.0.0"),
            "corpus": payload.get("corpus", ""), "path": str(target)}
    return probes, meta


class AccuracyEvaluator:
    """Runs the probes through the real pipeline and scores the answers."""

    def __init__(self, runtime) -> None:
        self.runtime = runtime

    def evaluate(self, probes: Sequence[Probe]) -> dict[str, Any]:
        results = [self._run(probe) for probe in probes]
        answerable = [r for r in results if not _is_unanswerable(probes, r.probe_id)]
        unanswerable = [r for r in results if _is_unanswerable(probes, r.probe_id)]

        correct = [r for r in answerable if r.correct]
        sourced = [r for r in answerable if r.source_correct]
        declined_well = [r for r in unanswerable if r.declined]
        hallucinated = [r for r in results if r.hallucinated]

        accuracy = _ratio(len(correct), len(answerable))
        source_accuracy = _ratio(len(sourced), len(answerable))
        refusal_accuracy = _ratio(len(declined_well), len(unanswerable))
        # One number for the rubric: every probe counts once, and declining
        # correctly counts exactly as much as answering correctly.
        overall = _ratio(len(correct) + len(declined_well), len(results))

        return {
            "kind": "accuracy_evaluation",
            "created_at": iso_now(),
            "profile": self.runtime.profile.id,
            "profile_fingerprint": self.runtime.profile.fingerprint,
            "backend": self.runtime.backend.describe(),
            "is_simulated": bool(self.runtime.backend.is_simulated),
            "corpus": {
                "documents": self.runtime.corpus.document_count(),
                "units": self.runtime.corpus.unit_count(),
                "checksum": self.runtime.corpus.corpus_checksum()[:16],
            },
            "probes": len(results),
            "answerable": len(answerable),
            "unanswerable": len(unanswerable),
            "accuracy": accuracy,
            "source_accuracy": source_accuracy,
            "refusal_accuracy": refusal_accuracy,
            "overall_accuracy": overall,
            "hallucination_rate": _ratio(len(hallucinated), len(results)),
            "truncated": _ratio(
                len([r for r in results if "cut off" in r.failure]), len(results)),
            "failures": [r.to_dict() for r in results if not _passed(r, probes)],
            "results": [r.to_dict() for r in results],
            "runtime_metrics": {
                k: v.to_dict()
                for k, v in aggregate_all([r.metrics for r in results],
                                          ("total_ms", "tokens_per_second",
                                           "peak_rss_mb", "prompt_tokens")).items()
            },
        }

    def _run(self, probe: Probe) -> ProbeResult:
        try:
            iec = self.runtime.ask(probe.question)
        except Exception as error:  # noqa: BLE001 -- a crash is a wrong answer
            return ProbeResult(probe.id, probe.question, False, False, None, False,
                               False, failure=f"{type(error).__name__}: {error}")

        answer = normalize_text(iec.response.text if iec.response else "")
        declined = bool(iec.validation
                        and iec.validation.confidence.value == "insufficient")
        answered = bool(answer.strip()) and not declined
        sources = tuple(e.document_name for e in iec.evidence)
        metrics = iec.metrics.to_flat() if iec.metrics else {}

        truncated = bool(iec.response and iec.response.truncated)
        if truncated:
            # Observed: a model that reasons before answering spent its whole
            # budget on the reasoning. The figures appeared inside the draft --
            # "the ceiling for lunch is 25 EUR" -- and eight probes were scored
            # correct on a scratchpad. A cut-off generation has not answered.
            return ProbeResult(
                probe.id, probe.question, answered=False, correct=False,
                source_correct=None, declined=False, hallucinated=False,
                answer=answer[:400], cited_sources=sources,
                confidence=iec.validation.confidence.value if iec.validation else "",
                failure=("the generation was cut off by the token budget; any "
                         "figure in it is part of an unfinished draft, not an "
                         "answer"),
                metrics=metrics)

        if not answer.strip():
            # Distinct from both a wrong answer and a refusal: the backend
            # produced nothing. Scoring it as "answered a question the corpus
            # cannot support" sent a reader looking for a hallucination that
            # never happened.
            return ProbeResult(
                probe.id, probe.question, answered=False, correct=False,
                source_correct=None, declined=False, hallucinated=False,
                answer="", cited_sources=sources,
                confidence=iec.validation.confidence.value if iec.validation else "",
                failure="the backend produced an empty answer", metrics=metrics)

        if probe.unanswerable:
            return ProbeResult(
                probe.id, probe.question, answered, correct=declined,
                source_correct=None, declined=declined, hallucinated=answered,
                answer=answer[:400], cited_sources=sources,
                confidence=iec.validation.confidence.value if iec.validation else "",
                failure="" if declined else "answered a question the corpus cannot support",
                metrics=metrics,
            )

        correct, failure = _check(answer, probe)
        source_correct = (probe.expected_source in sources
                          if probe.expected_source else None)
        if correct and source_correct is False:
            failure = f"correct figure but cited {sources or '(nothing)'}"
        return ProbeResult(
            probe.id, probe.question, answered, correct=correct,
            source_correct=source_correct, declined=declined,
            hallucinated=False, answer=answer[:400], cited_sources=sources,
            confidence=iec.validation.confidence.value if iec.validation else "",
            failure=failure, metrics=metrics,
        )


def _check(answer: str, probe: Probe) -> tuple[bool, str]:
    lowered = answer.lower()
    hits = [token for token in probe.must_contain if token.lower() in lowered]
    if probe.match == "any":
        satisfied = bool(hits)
    else:
        satisfied = len(hits) == len(probe.must_contain)
    if not satisfied:
        missing = [t for t in probe.must_contain if t.lower() not in lowered]
        return False, f"missing {missing} from the answer"
    forbidden = [token for token in probe.must_not_contain if token.lower() in lowered]
    if forbidden:
        return False, f"contains {forbidden}, which is a different figure in the corpus"
    return True, ""


def _passed(result: ProbeResult, probes: Sequence[Probe]) -> bool:
    if _is_unanswerable(probes, result.probe_id):
        return result.declined
    return result.correct and result.source_correct is not False


def _is_unanswerable(probes: Sequence[Probe], probe_id: str) -> bool:
    return any(p.id == probe_id and p.unanswerable for p in probes)


def _ratio(part: int, whole: int) -> float:
    return round(part / whole, 4) if whole else 0.0


def evaluate_accuracy(runtime, evaluation_path: str | Path | None = None, *,
                      output: str | Path | None = None) -> dict[str, Any]:
    from ..shared.config import repo_root
    from ..shared.serialization import write_json

    path = Path(evaluation_path) if evaluation_path else \
        repo_root() / "datasets/enterprise_en/evaluation.json"
    probes, meta = load_probes(path)
    summary = AccuracyEvaluator(runtime).evaluate(probes)
    summary["evaluation_set"] = meta
    if output:
        write_json(output, summary)
        summary["output_path"] = str(output)
    return summary


def render_accuracy(summary: dict[str, Any]) -> str:
    backend = summary.get("backend", {}) or {}
    lines = [
        # Printed first, because a number read against the wrong configuration
        # is worse than no number: a profile edited but not selected produced
        # 8% where the intended one produced 92%, with nothing on screen to
        # distinguish the two runs.
        f"profile {summary.get('profile', '?')} "
        f"[{summary.get('profile_fingerprint', '?')[:8]}] | "
        f"backend {backend.get('backend', '?')} | "
        f"model {backend.get('model_id', '?')} {backend.get('quantization', '')}",
        "",
        f"accuracy on {summary['probes']} probes "
        f"({summary['answerable']} answerable, {summary['unanswerable']} not)",
        "",
        f"  answer accuracy   : {summary['accuracy']:.0%}"
        f"   ({summary['answerable']} answerable probes)",
        f"  source accuracy   : {summary['source_accuracy']:.0%}"
        "   (cited the document that states it)",
        f"  refusal accuracy  : {summary['refusal_accuracy']:.0%}"
        f"   ({summary['unanswerable']} probes the corpus cannot answer)",
        f"  hallucination rate: {summary['hallucination_rate']:.0%}",
        "",
        f"  OVERALL           : {summary['overall_accuracy']:.0%}",
    ]
    if summary.get("truncated"):
        lines.extend([
            "",
            f"  !! {summary['truncated']:.0%} of generations were cut off by the "
            "token budget.",
            "     A model that reasons before answering spends the budget on the "
            "reasoning.",
            "     Raise inference.max_output_tokens, or disable thinking with "
            "prompt.thinking_suffix.",
        ])
    if summary.get("failures"):
        lines.extend(["", "failures:"])
        for failure in summary["failures"]:
            lines.append(f"  {failure['probe_id']} {failure['question'][:56]}")
            lines.append(f"     -> {failure['failure'] or 'incorrect'}")
    if summary.get("is_simulated"):
        lines.extend(["", "  !! SIMULATED BACKEND: this measures the retrieval and "
                          "validation layers, not a model's ability to answer."])
    return "\n".join(lines)
