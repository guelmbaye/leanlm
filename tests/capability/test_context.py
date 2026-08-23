"""CAP-003 Context Optimization -- FR-03."""
from __future__ import annotations

from leanlm.contracts.dto import IntentType
from leanlm.packages.context import (SPEC, ContextBudgeter, ContextOptimizer,
                                     IntentAnalyzer)
from leanlm.packages.context.policies import BudgetPolicy, OptimizationPolicy


def _budget(question: str, resources=None):
    """The budgeter needs the real question: its tokens come out of the window."""
    return ContextBudgeter().compute(
        model_context_tokens=4096, max_output_tokens=512, question=question,
        intent=IntentAnalyzer().analyze(question), resources=resources)


def _package(corpus, structured, source):
    from leanlm.shared.clock import iso_now
    corpus.upsert_document(structured, source_path=str(source),
                           file_checksum="deadbeef", ingested_at=iso_now())
    return corpus.context_package()


class TestSpec:
    def test_identity(self):
        assert SPEC.capability_id == "CAP-003"
        assert SPEC.stages == ("context_optimization",)


class TestIntent:
    def test_a_lookup_question_is_narrow(self):
        """"Quel est le delai" is an extraction, not an essay prompt: the budget
        it earns must be one of the narrow intents."""
        intent = IntentAnalyzer().analyze("Quel est le delai de remboursement ?").intent
        assert intent in (IntentType.FACTUAL, IntentType.EXTRACTION)

    def test_summary_question(self):
        assert IntentAnalyzer().analyze("Resume la politique de teletravail.").intent \
            is IntentType.SUMMARY

    def test_classification_is_deterministic(self):
        analyzer = IntentAnalyzer()
        question = "Compare les plafonds de repas et d'hebergement."
        assert analyzer.analyze(question).intent is analyzer.analyze(question).intent


class TestBudget:
    def test_budget_leaves_room_for_the_output(self):
        budget = _budget("Quel est le delai de remboursement ?")
        assert budget.evidence_tokens > 0
        assert budget.evidence_tokens + budget.reserved_output_tokens <= 4096

    def test_summary_gets_more_room_than_a_factual_lookup(self):
        factual = _budget("Quel est le delai de remboursement ?")
        summary = _budget("Resume la politique de teletravail.")
        assert summary.evidence_tokens > factual.evidence_tokens

    def test_the_limiting_factor_is_recorded(self):
        assert _budget("Quel est le plafond des repas ?").limiting_factor


class TestOptimizer:
    def test_exact_duplicates_are_dropped(self, corpus, structured, sample_file):
        package = _package(corpus, structured, sample_file)
        budget = _budget("Quel est le delai de remboursement ?")
        optimized = ContextOptimizer().optimize(package, budget)
        texts = [u.text for u in optimized.units]
        assert len(texts) == len(set(texts))

    def test_numbers_protect_a_passage_from_deduplication(self):
        """Two clauses differing only by a figure are not duplicates: the figure
        is the answer."""
        from leanlm.packages.context.services import numeric_signature
        assert (numeric_signature("remboursement sous 30 jours")
                != numeric_signature("remboursement sous 45 jours"))

    def test_optimization_never_invents_text(self, corpus, structured, sample_file):
        package = _package(corpus, structured, sample_file)
        budget = _budget("Quel est le delai de remboursement ?")
        optimized = ContextOptimizer().optimize(package, budget)
        original = " ".join(u.text for u in package.units)
        for unit in optimized.units:
            for sentence in unit.text.split(". ")[:1]:
                assert sentence.strip(". ")[:40] in original

    def test_operations_are_reported(self, corpus, structured, sample_file):
        package = _package(corpus, structured, sample_file)
        budget = _budget("Quel est le delai de remboursement ?")
        optimized = ContextOptimizer().optimize(package, budget)
        assert isinstance(optimized.operations, tuple)
        assert 0.0 <= optimized.compression_ratio <= 1.0
