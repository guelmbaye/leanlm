"""Lexical retrieval built for a laptop.

Why BM25 and not embeddings: an embedding model is a second model in RAM, a
second load time, and a second thing to ship. On a 4 GB budget shared with the
LLM itself, the marginal accuracy of dense retrieval does not pay for its
memory. BM25 over structured units, boosted by section titles, is dependency
free, instantaneous, and fully deterministic -- which also makes it
reproducible, an ADTC criterion in its own right.

The decision is recorded as EDB-004 with its trade-offs.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ...contracts.dto import (
    EvidenceCandidate, IntentClassification, OptimizedContext, SelectedEvidence,
    SemanticUnit, UnitKind,
)
from ...shared.text import DEFAULT_TOKEN_COUNTER, TokenCounter, content_words, jaccard, shingles
from .models import ScoredUnit
from .policies import RetrievalPolicy

_NUMERIC_QUESTION = re.compile(
    r"\b(combien|montant|delai|taux|pourcentage|nombre|date|plafond|budget|prix|"
    r"how many|how much|amount|rate|deadline|percentage)\b", re.IGNORECASE
)


# Shared prefix length used as a stand-in for stemming (see below).
_PREFIX = 7

# Words that pass the content-word filter but describe the *shape* of a question
# rather than its subject: duration, quantity, occurrence. They are ordinary
# enough that a small corpus may simply never use them, and counting their
# absence as evidence of an off-topic question refuses legitimate ones --
# "how long does the confidentiality obligation last" was refused because
# `long`, `last` and `ends` are missing from a corpus that plainly discusses
# confidentiality.
#
# Applied to the off-topic gate ONLY. They stay in the scoring vocabulary,
# because "3 days allowed" is a phrase worth matching on.
_GENERIC_QUESTION_TERMS = frozenset({
    "long", "longer", "last", "lasts", "end", "ends", "ended", "start", "starts",
    "take", "takes", "taken", "happen", "happens", "occur", "occurs",
    "apply", "applies", "cover", "covers", "mean", "means", "case", "cases",
    "duree", "dure", "durer", "passe", "arrive", "concerne", "sagit",
})


class BM25Index:
    """Okapi BM25 over semantic units. Rebuilt per request; it is cheap.

    Rebuilding costs O(total tokens) and stays in the low milliseconds for a
    laptop-sized corpus, which buys us a guarantee worth more than the saving:
    the index can never drift from the optimized context it scores.
    """

    def __init__(self, units: list[SemanticUnit], policy: RetrievalPolicy | None = None) -> None:
        self.policy = policy or RetrievalPolicy()
        self.units = units
        self._tokens: list[list[str]] = [content_words(u.text) for u in units]
        self._freqs: list[Counter] = [Counter(t) for t in self._tokens]
        self._lengths = [len(t) for t in self._tokens]
        self._avg_length = (sum(self._lengths) / len(self._lengths)) if self._lengths else 0.0
        self._doc_freq: Counter = Counter()
        for tokens in self._tokens:
            self._doc_freq.update(set(tokens))
        self._n = len(units)

        # Vocabulary for topic-support checks only -- never for scoring, where
        # heading terms would double-count. Section titles carry much of a
        # document's subject vocabulary ("Politique de remboursement des notes
        # de frais") and headings are deliberately not kept as retrievable
        # units, so without this the index believes the corpus has never heard
        # of its own topics.
        self._vocabulary: set[str] = set(self._doc_freq)
        for unit in units:
            for part in unit.section_path:
                self._vocabulary.update(content_words(part))
        # French inflects: the corpus says "remboursees", the question asks
        # about "remboursement". A shared seven-character prefix is a cheap and
        # auditable stand-in for a stemmer, and erring here only costs recall on
        # the off-topic gate -- never correctness of what is retrieved.
        self._prefixes = {term[:_PREFIX] for term in self._vocabulary
                          if len(term) >= _PREFIX}

    def _idf(self, term: str) -> float:
        df = self._doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        return math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))

    def unsupported_terms(self, query_terms: list[str]) -> list[str]:
        """Query terms that appear in no unit at all.

        A question made largely of words the corpus has never seen is a question
        about a topic the corpus does not cover, however well its remaining words
        happen to match. "Frais de scolarite des enfants" shares `frais` and
        `remboursement` with an expense policy and shares nothing that matters.
        """
        unknown = []
        for term in dict.fromkeys(query_terms):
            if term in self._vocabulary:
                continue
            if len(term) >= _PREFIX and term[:_PREFIX] in self._prefixes:
                continue
            unknown.append(term)
        return unknown

    def score(self, query_terms: list[str]) -> list[float]:
        k1, b = self.policy.bm25_k1, self.policy.bm25_b
        scores = [0.0] * self._n
        if not query_terms or self._avg_length == 0:
            return scores
        for term in set(query_terms):
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for index in range(self._n):
                freq = self._freqs[index].get(term, 0)
                if not freq:
                    continue
                norm = 1.0 - b + b * (self._lengths[index] / self._avg_length)
                scores[index] += idf * (freq * (k1 + 1.0)) / (freq + k1 * norm)
        return scores


class EvidenceSelector:
    """Scores, diversifies and fits evidence into the context budget."""

    def __init__(self, policy: RetrievalPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        self.policy = policy or RetrievalPolicy()
        self.counter = counter or DEFAULT_TOKEN_COUNTER

    def select(self, question: str, optimized: OptimizedContext,
               intent: IntentClassification | None = None) -> tuple[
                   tuple[SelectedEvidence, ...], tuple[EvidenceCandidate, ...]]:
        units = list(optimized.units)
        if not units:
            return (), ()
        budget = optimized.budget
        max_tokens = budget.evidence_tokens if budget else 1024
        max_passages = budget.max_passages if budget else 5

        query_terms = content_words(question)
        index = BM25Index(units, self.policy)

        # Off-topic questions are rejected before scoring, not after. A question
        # whose distinctive words are absent from the corpus can still match on
        # its ordinary ones -- "remboursement des frais de scolarite des enfants"
        # scores well against an expense policy that says nothing about school
        # fees -- and the resulting answer is fluent, grounded and wrong.
        # The gate judges the question's *subject*, so generic framing words are
        # excluded from both sides of the ratio: they can neither convict a
        # question of being off-topic nor acquit it.
        topical = [t for t in set(query_terms) if t not in _GENERIC_QUESTION_TERMS]
        unsupported = [t for t in index.unsupported_terms(list(query_terms))
                       if t not in _GENERIC_QUESTION_TERMS]
        self.last_unsupported_terms = tuple(unsupported)
        if topical and (len(unsupported) / len(topical)
                        >= self.policy.unsupported_term_ratio):
            return (), ()

        if intent is not None:
            query_terms = list(query_terms) + list(intent.keywords)
        lexical = index.score(query_terms)
        boosts = [self._structural_boost(question, query_terms, unit) for unit in units]

        scored = [ScoredUnit(i, lexical[i], boosts[i]) for i in range(len(units))]
        # Deterministic ordering: score desc, then unit id asc. Never rely on
        # sort stability alone -- an equal-score tie must resolve identically on
        # every machine (P4).
        scored.sort(key=lambda s: (-round(s.total, 9), units[s.index].unit_id))
        best_score = scored[0].total if scored else 0.0
        floor = max(self.policy.min_score, best_score * self.policy.relative_score_floor)
        pool = [s for s in scored if s.total >= floor][:self.policy.candidate_pool]
        if not pool and best_score > self.policy.min_score:
            # There is a signal, just a weak one: keep the top few.
            pool = scored[: min(3, len(scored))]
        # If nothing scored above the absolute floor, return nothing. Sending
        # arbitrary passages would let the model answer confidently from text
        # that has no relation to the question (P3, IP-03).

        selected, candidates = self._greedy_diverse_fit(
            pool, units, max_tokens=max_tokens, max_passages=max_passages
        )
        return selected, candidates

    # -- scoring ------------------------------------------------------------
    def _structural_boost(self, question: str, query_terms: list[str],
                          unit: SemanticUnit) -> float:
        policy = self.policy
        boost = 0.0
        terms = set(query_terms)
        if unit.section_path:
            section_terms = set(content_words(" ".join(unit.section_path)))
            if section_terms & terms:
                overlap = len(section_terms & terms) / max(1, len(terms))
                boost += policy.section_match_boost * min(1.0, overlap * 2.0)
        if unit.document_name:
            if set(content_words(unit.document_name.replace("_", " "))) & terms:
                boost += policy.title_match_boost
        if unit.kind is UnitKind.TABLE:
            boost += policy.table_boost
        lowered_q = question.lower()
        lowered_u = unit.text.lower()
        for phrase in self._phrases(lowered_q):
            if phrase in lowered_u:
                boost += policy.exact_phrase_boost
                break
        if _NUMERIC_QUESTION.search(question) and any(c.isdigit() for c in unit.text):
            boost += policy.numeric_question_boost
        return round(boost, 6)

    @staticmethod
    def _phrases(question: str, size: int = 3) -> list[str]:
        tokens = [t for t in re.split(r"\W+", question) if t]
        if len(tokens) < size:
            return []
        return [" ".join(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

    # -- selection ----------------------------------------------------------
    def _greedy_diverse_fit(self, pool: list[ScoredUnit], units: list[SemanticUnit],
                            *, max_tokens: int, max_passages: int) -> tuple[
                                tuple[SelectedEvidence, ...], tuple[EvidenceCandidate, ...]]:
        """Maximal marginal relevance under a token budget.

        Greedy is the right algorithm here: the budget is small, the pool is
        small, and an optimal knapsack would be both slower and less stable
        across runs for no measurable accuracy gain.
        """
        chosen: list[tuple[ScoredUnit, float]] = []
        chosen_signatures: list[frozenset] = []
        per_document: Counter = Counter()
        used_tokens = 0
        remaining = list(pool)

        while remaining and len(chosen) < max_passages:
            best_item = None
            best_value = float("-inf")
            best_penalty = 0.0
            for item in remaining:
                unit = units[item.index]
                if per_document[unit.document_id] >= self.policy.max_per_document:
                    continue
                cost = unit.token_estimate or self.counter.count(unit.text)
                if used_tokens + cost > max_tokens and chosen:
                    continue
                penalty = 0.0
                if chosen_signatures:
                    signature = shingles(unit.text)
                    penalty = self.policy.diversity_lambda * max(
                        jaccard(signature, other) for other in chosen_signatures
                    )
                value = item.total - penalty
                if value > best_value or (
                    value == best_value and best_item is not None
                    and unit.unit_id < units[best_item.index].unit_id
                ):
                    best_item, best_value, best_penalty = item, value, penalty
            if best_item is None:
                break
            unit = units[best_item.index]
            chosen.append((best_item, best_penalty))
            chosen_signatures.append(shingles(unit.text))
            per_document[unit.document_id] += 1
            used_tokens += unit.token_estimate or self.counter.count(unit.text)
            remaining.remove(best_item)

        selected: list[SelectedEvidence] = []
        candidates: list[EvidenceCandidate] = []
        for position, (item, penalty) in enumerate(chosen, start=1):
            unit = units[item.index]
            label = f"S{position}"
            evidence = SelectedEvidence(
                meta=SelectedEvidence.build_meta(
                    identity=(unit.unit_id, label), parent_id=unit.artifact_id,
                    source=unit.document_name, stage="evidence_retrieval",
                ),
                label=label,
                unit_id=unit.unit_id,
                document_name=unit.document_name,
                section_path=unit.section_path,
                text=unit.text,
                score=round(item.total, 6),
                token_estimate=unit.token_estimate,
            ).sealed()
            selected.append(evidence)  # type: ignore[arg-type]
            candidate = EvidenceCandidate(
                meta=EvidenceCandidate.build_meta(
                    identity=(unit.unit_id, f"{item.total:.6f}"), stage="evidence_retrieval"
                ),
                unit=unit, score=round(item.total, 6), lexical_score=round(item.lexical, 6),
                structural_boost=round(item.boost, 6), diversity_penalty=round(penalty, 6),
            ).sealed()
            candidates.append(candidate)  # type: ignore[arg-type]
        return tuple(selected), tuple(candidates)


def retrieval_precision(selected: tuple[SelectedEvidence, ...], question: str) -> float:
    """Share of selected passages that actually share content with the question.

    A blunt but honest proxy: it cannot be gamed by returning more passages,
    because it is a ratio, and it needs no labelled ground truth -- which a
    hackathon corpus does not have.
    """
    if not selected:
        return 0.0
    terms = set(content_words(question))
    if not terms:
        return 0.0
    hits = sum(1 for e in selected if terms & set(content_words(e.text)))
    return round(hits / len(selected), 4)
