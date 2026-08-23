"""Response validation.

The validator never rewrites the model output. It annotates it. An answer that
fails validation is still returned -- with its failure attached -- because
hiding a weak answer would be a worse failure than showing one (DIC-07).

Grounding is measured by n-gram overlap between each answer sentence and the
evidence actually sent to the model. It is a proxy, and it is deliberately a
cheap one: a second model to judge the first would double the memory budget on
a machine that has none to spare.
"""
from __future__ import annotations

import re

from ...contracts.dto import (
    ConfidenceLevel, ModelResponse, SelectedEvidence, ValidationReport,
)
from ...shared.text import content_words, sentences, shingles
from .models import SentenceVerdict
from .policies import HEDGE_MARKERS, INSUFFICIENT_MARKERS, ValidationPolicy

_CITATION = re.compile(r"\[(S\d+)\]")


class ResponseValidator:
    def __init__(self, policy: ValidationPolicy | None = None) -> None:
        self.policy = policy or ValidationPolicy()

    def validate(self, response: ModelResponse, evidence: tuple[SelectedEvidence, ...],
                 question: str) -> ValidationReport:
        policy = self.policy
        answer = (response.text or "").strip()
        checks: dict[str, bool] = {}
        warnings: list[str] = []

        checks["non_empty"] = len(answer) >= policy.min_answer_chars
        # An observation, not a gate: declaring uncertainty is sometimes the
        # correct answer and sometimes irrelevant. It is reported separately.
        declared_uncertainty = any(marker in answer.lower() for marker in INSUFFICIENT_MARKERS)

        cited = tuple(sorted(set(_CITATION.findall(answer))))
        available = {item.label for item in evidence}
        invalid = tuple(label for label in cited if label not in available)
        checks["citations_valid"] = not invalid
        if invalid:
            warnings.append(
                f"answer cites labels that were never provided: {', '.join(invalid)}"
            )

        verdicts = self._ground(answer, evidence)
        grounded = [v for v in verdicts if v.grounded]
        grounding_rate = round(len(grounded) / len(verdicts), 4) if verdicts else 0.0
        unsupported = tuple(v.text for v in verdicts if not v.grounded)[:5]

        evidence_coverage = round(
            len({label for label in cited if label in available}) / len(available), 4
        ) if available else 0.0

        question_terms = set(content_words(question))
        answer_terms = set(content_words(answer))
        question_coverage = round(
            len(question_terms & answer_terms) / len(question_terms), 4
        ) if question_terms else 0.0

        if not evidence:
            # No evidence at all: the only acceptable answer is an explicit
            # statement of insufficiency (IP-03 / hallucination policy).
            checks["grounded"] = declared_uncertainty
            if not declared_uncertainty:
                warnings.append(
                    "no evidence was available yet the answer asserts content; "
                    "the model went outside the corpus"
                )
            confidence = ConfidenceLevel.INSUFFICIENT
            passed = declared_uncertainty
        else:
            checks["grounded"] = grounding_rate >= policy.grounding_pass_threshold
            if policy.require_citations:
                checks["has_citations"] = bool(cited) or declared_uncertainty
            confidence = self._confidence(grounding_rate, evidence_coverage,
                                          question_coverage, declared_uncertainty, cited)
            if question_coverage == 0.0 and not declared_uncertainty:
                warnings.append(
                    "the answer shares no term with the question; it may be "
                    "grounded in the wrong passage"
                )
            passed = all(checks.values())

        if any(marker in answer.lower() for marker in HEDGE_MARKERS):
            warnings.append("answer contains hedging language; verify against the sources")
        if unsupported:
            warnings.append(f"{len(unsupported)} sentence(s) not supported by the excerpts")

        report = ValidationReport(
            meta=ValidationReport.build_meta(
                identity=(str(passed), f"{grounding_rate:.4f}"),
                parent_id=response.artifact_id, stage="response_validation",
            ),
            passed=passed,
            confidence=confidence,
            grounding_rate=grounding_rate,
            evidence_coverage=evidence_coverage,
            question_coverage=question_coverage,
            cited_labels=cited,
            unsupported_sentences=unsupported,
            uncertainty_declared=declared_uncertainty,
            checks=checks,
            warnings=tuple(warnings),
        )
        return report.sealed()  # type: ignore[return-value]

    # -- internals ----------------------------------------------------------
    def _ground(self, answer: str, evidence: tuple[SelectedEvidence, ...]
                ) -> list[SentenceVerdict]:
        if not answer or not evidence:
            return []
        size = self.policy.shingle_size
        evidence_grams = {item.label: shingles(item.text, size) for item in evidence}
        verdicts: list[SentenceVerdict] = []
        for sentence in sentences(_CITATION.sub("", answer)):
            terms = content_words(sentence)
            if len(terms) < 3:
                continue  # too short to judge; not counted either way
            grams = shingles(sentence, size)
            best_label, best_overlap = "", 0.0
            for label, reference in evidence_grams.items():
                if not grams:
                    continue
                overlap = len(grams & reference) / len(grams)
                if overlap > best_overlap:
                    best_label, best_overlap = label, overlap
            verdicts.append(SentenceVerdict(
                text=sentence.strip(), best_label=best_label,
                overlap=round(best_overlap, 4),
                grounded=best_overlap >= self.policy.sentence_grounding_threshold,
            ))
        return verdicts

    def _confidence(self, grounding: float, coverage: float, question_coverage: float,
                    declared_uncertainty: bool, cited: tuple[str, ...]) -> ConfidenceLevel:
        """Observable criteria only -- never an invented probability (IIB s.13)."""
        if declared_uncertainty:
            return ConfidenceLevel.INSUFFICIENT
        policy = self.policy
        # High confidence requires the answer to be about the question, not
        # merely faithful to *some* passage: a verbatim quote of an unrelated
        # excerpt scores a perfect grounding rate and answers nothing.
        if (grounding >= policy.high_confidence_grounding and cited
                and coverage >= 0.3 and question_coverage > 0.0):
            return ConfidenceLevel.HIGH
        if grounding >= policy.medium_confidence_grounding and (cited or coverage > 0):
            return ConfidenceLevel.MEDIUM
        if grounding > 0.0:
            return ConfidenceLevel.LOW
        return ConfidenceLevel.INSUFFICIENT
