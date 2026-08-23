"""Intent analysis, context budgeting, context optimization.

This package is the "brain" side of the Inference Intelligence Blueprint: it
decides *how much* context the machine can afford right now and *which* of it is
worth keeping, before a single token reaches llama.cpp.

Maximum relevance. Minimum context.
"""
from __future__ import annotations

import re
from dataclasses import replace

from ...contracts.dto import (
    ContextBudget, ContextPackage, IntentClassification, IntentType,
    OptimizedContext, ResourceObservation, SemanticUnit, UnitKind,
)
from ...shared.hashing import sha256_text
from ...shared.text import (
    DEFAULT_TOKEN_COUNTER, TokenCounter, content_words, jaccard, shingles, words,
)

_NUMBER = re.compile(r"\d[\d\s.,%]*")


def numeric_signature(text: str) -> frozenset[str]:
    """Numbers carried by a passage, normalized.

    Two paragraphs that differ only by a figure are *not* duplicates: in an
    enterprise corpus the figure is usually the answer ("remboursement sous 30
    jours" vs "sous 45 jours"). Deduplication must never collapse them.
    """
    return frozenset(
        match.group(0).replace(" ", "").rstrip(".,").replace(",", ".")
        for match in _NUMBER.finditer(text)
        if match.group(0).strip(" .,")
    )
from .models import BudgetFactors
from .policies import BudgetPolicy, OptimizationPolicy

# ---------------------------------------------------------------------------
# Intent analysis (IIB s.5)
# ---------------------------------------------------------------------------

_INTENT_MARKERS: dict[IntentType, tuple[str, ...]] = {
    IntentType.COMPARISON: (
        "compare", "comparer", "comparaison", "difference", "differences", "versus",
        "vs", "plutot que", "par rapport", "mieux que", "ecart", "entre",
    ),
    IntentType.SUMMARY: (
        "resume", "resumer", "synthese", "synthetise", "summary", "summarize",
        "overview", "vue d'ensemble", "en bref", "grandes lignes", "recapitule",
    ),
    IntentType.EXTRACTION: (
        "liste", "lister", "list", "enumere", "extract", "extraire", "combien",
        "quels sont", "quelles sont", "montant", "date", "delai", "taux", "how many",
    ),
    IntentType.EXPLANATION: (
        "pourquoi", "explique", "expliquer", "why", "explain", "comment fonctionne",
        "how does", "raison", "justifie",
    ),
    IntentType.DRAFTING: (
        "redige", "rediger", "ecris", "ecrire", "draft", "write", "propose un",
        "genere", "formule", "mail", "courrier", "note de service",
    ),
}


class IntentAnalyzer:
    """Deterministic, rule based, auditable (P4).

    A model call to classify a question would cost more than the saving it
    enables; on the target hardware, a lexicon is both faster and reproducible.
    """

    def analyze(self, question: str) -> IntentClassification:
        lowered = " " + question.lower().strip() + " "
        scores: dict[IntentType, int] = {}
        signals: dict[IntentType, list[str]] = {}
        for intent, markers in _INTENT_MARKERS.items():
            hits = [m for m in markers if f" {m}" in lowered or f"{m} " in lowered]
            if hits:
                scores[intent] = len(hits)
                signals[intent] = hits
        if not scores:
            intent, confidence, hits = IntentType.FACTUAL, "medium", []
        else:
            best = max(scores.values())
            winners = sorted(i for i, s in scores.items() if s == best)
            intent = winners[0]
            hits = signals[intent]
            confidence = "high" if best >= 2 or len(winners) == 1 else "low"
        classification = IntentClassification(
            meta=IntentClassification.build_meta(
                identity=(intent.value, question[:120]), stage="context_optimization"
            ),
            intent=intent,
            confidence=confidence,
            signals=tuple(sorted(hits)),
            keywords=tuple(content_words(question)[:24]),
        )
        return classification.sealed()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Context budgeting (IIB s.7 -- Resource-Aware Inference)
# ---------------------------------------------------------------------------


class ContextBudgeter:
    """Turns a hardware observation into a *target* context size.

    The result is deliberately a target rather than a fixed number: the same
    question on a cold laptop and on a thermally throttled one must not push the
    same amount of context through llama.cpp.
    """

    def __init__(self, policy: BudgetPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        self.policy = policy or BudgetPolicy()
        self.counter = counter or DEFAULT_TOKEN_COUNTER

    def compute(self, *, model_context_tokens: int, max_output_tokens: int,
                question: str, intent: IntentClassification,
                resources: ResourceObservation | None,
                model_footprint_mb: float = 0.0) -> ContextBudget:
        policy = self.policy
        overhead = policy.prompt_overhead_tokens + self.counter.count(question)
        base = model_context_tokens - max_output_tokens - overhead
        base = max(policy.min_context_tokens, base)

        factors = BudgetFactors(
            ram=self._ram_factor(resources, model_footprint_mb),
            thermal=self._thermal_factor(resources),
            cpu=self._cpu_factor(resources),
            intent=policy.intent_factors.get(intent.intent.value, 0.8),
        )
        target = int(base * factors.multiplier())
        target = max(policy.min_context_tokens, min(base, target))

        max_passages = min(
            policy.max_passages_hard_cap,
            policy.intent_passages.get(intent.intent.value, 5),
        )
        if factors.multiplier() < 0.5:
            max_passages = max(2, max_passages // 2)

        budget = ContextBudget(
            meta=ContextBudget.build_meta(
                identity=(str(target), factors.limiting()), stage="context_optimization"
            ),
            max_context_tokens=model_context_tokens,
            evidence_tokens=target,
            reserved_output_tokens=max_output_tokens,
            reserved_prompt_overhead=overhead,
            max_passages=max_passages,
            limiting_factor=factors.limiting(),
            factors=factors.to_dict(),
            token_counter=self.counter.name,
        )
        return budget.sealed()  # type: ignore[return-value]

    def _ram_factor(self, res: ResourceObservation | None, footprint_mb: float) -> float:
        if res is None or res.available_ram_mb <= 0:
            return 1.0
        usable = res.available_ram_mb - self.policy.ram_headroom_mb - footprint_mb
        if usable <= 0:
            return 0.35
        if usable >= self.policy.ram_soft_floor_mb:
            return 1.0
        return round(max(0.35, 0.35 + 0.65 * (usable / self.policy.ram_soft_floor_mb)), 4)

    def _thermal_factor(self, res: ResourceObservation | None) -> float:
        if res is None or res.temperature_c is None:
            return 1.0
        temp = res.temperature_c
        warn, critical = self.policy.thermal_warn_c, self.policy.thermal_critical_c
        if temp < warn:
            return 1.0
        if temp >= critical:
            return 0.4
        span = max(1e-6, critical - warn)
        return round(max(0.4, 1.0 - 0.6 * ((temp - warn) / span)), 4)

    def _cpu_factor(self, res: ResourceObservation | None) -> float:
        if res is None or res.cpu_percent <= 0:
            return 1.0
        warn = self.policy.cpu_warn_percent
        if res.cpu_percent < warn:
            return 1.0
        over = min(1.0, (res.cpu_percent - warn) / max(1e-6, 100.0 - warn))
        return round(max(0.6, 1.0 - 0.4 * over), 4)


# ---------------------------------------------------------------------------
# Context optimization (IIB s.8)
# ---------------------------------------------------------------------------


class ContextOptimizer:
    """Removes what cannot help, merges what is fragmented, keeps structure.

    Runs *before* retrieval so that scoring never wastes a slot on a duplicate
    or on a page footer. All operations are recorded, because the compression
    ratio is an EBPB Tier-3 metric that has to be defensible.
    """

    def __init__(self, policy: OptimizationPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        self.policy = policy or OptimizationPolicy()
        self.counter = counter or DEFAULT_TOKEN_COUNTER
        self._boilerplate = [re.compile(p, re.IGNORECASE) for p in self.policy.boilerplate_patterns]

    def optimize(self, package: ContextPackage, budget: ContextBudget) -> OptimizedContext:
        units = list(package.units)
        input_tokens = sum(u.token_estimate for u in units)
        operations: list[str] = []

        units, dropped_noise = self._drop_noise(units)
        if dropped_noise:
            operations.append(f"drop_boilerplate:{dropped_noise}")

        units, exact = self._drop_exact_duplicates(units)
        if exact:
            operations.append(f"drop_exact_duplicates:{exact}")

        units, near = self._drop_near_duplicates(units)
        if near:
            operations.append(f"drop_near_duplicates:{near}")

        units, merged = self._merge_fragments(units)
        if merged:
            operations.append(f"merge_adjacent:{merged}")

        output_tokens = sum(u.token_estimate for u in units)
        optimized = OptimizedContext(
            meta=OptimizedContext.build_meta(
                identity=(str(output_tokens), str(len(units))),
                parent_id=package.artifact_id, stage="context_optimization",
            ),
            units=tuple(units),
            budget=budget,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            dropped_units=dropped_noise + exact + near,
            merged_units=merged,
            duplicate_units=exact + near,
            operations=tuple(operations),
        )
        return optimized.sealed()  # type: ignore[return-value]

    # -- operations ---------------------------------------------------------
    def _drop_noise(self, units: list[SemanticUnit]) -> tuple[list[SemanticUnit], int]:
        kept: list[SemanticUnit] = []
        dropped = 0
        for unit in units:
            body = unit.text.strip()
            if any(rx.match(body) for rx in self._boilerplate):
                dropped += 1
                continue
            if len(content_words(body)) < self.policy.drop_below_content_words \
                    and unit.kind is not UnitKind.TABLE:
                dropped += 1
                continue
            kept.append(unit)
        return kept, dropped

    def _drop_exact_duplicates(self, units: list[SemanticUnit]) -> tuple[list[SemanticUnit], int]:
        seen: set[str] = set()
        kept: list[SemanticUnit] = []
        dropped = 0
        for unit in units:
            digest = sha256_text(" ".join(content_words(unit.text)))
            if digest in seen:
                dropped += 1
                continue
            seen.add(digest)
            kept.append(unit)
        return kept, dropped

    def _drop_near_duplicates(self, units: list[SemanticUnit]) -> tuple[list[SemanticUnit], int]:
        """O(n * k): each unit is only compared to units sharing a rare term.

        A naive all-pairs comparison is quadratic and would dominate ingestion
        time on a large corpus. Bucketing by the rarest content word keeps the
        comparison count near linear while catching the duplicates that matter
        (boilerplate paragraphs repeated across documents).
        """
        buckets: dict[str, list[int]] = {}
        signatures = [shingles(u.text) for u in units]
        numbers = [numeric_signature(u.text) for u in units]
        keep = [True] * len(units)
        dropped = 0
        for index, unit in enumerate(units):
            terms = content_words(unit.text)
            if not terms:
                continue
            key = min(terms, key=lambda t: (len(t), t))
            for other in buckets.get(key, []):
                if not keep[other]:
                    continue
                if numbers[index] != numbers[other]:
                    continue  # differing figures => distinct facts
                if jaccard(signatures[index], signatures[other]) >= \
                        self.policy.near_duplicate_threshold:
                    if unit.token_estimate <= units[other].token_estimate:
                        keep[index] = False
                    else:
                        keep[other] = False
                    dropped += 1
                    break
            buckets.setdefault(key, []).append(index)
        return [u for u, k in zip(units, keep) if k], dropped

    def _merge_fragments(self, units: list[SemanticUnit]) -> tuple[list[SemanticUnit], int]:
        merged_count = 0
        result: list[SemanticUnit] = []
        for unit in units:
            if (result
                    and unit.token_estimate < self.policy.merge_adjacent_below_tokens
                    and result[-1].token_estimate < self.policy.merge_adjacent_below_tokens
                    and result[-1].document_id == unit.document_id
                    and result[-1].section_path == unit.section_path):
                previous = result[-1]
                text = f"{previous.text}\n{unit.text}"
                result[-1] = replace(
                    previous, text=text,
                    token_estimate=self.counter.count(text),
                    char_end=unit.char_end,
                ).sealed()
                merged_count += 1
                continue
            result.append(unit)
        return result, merged_count
