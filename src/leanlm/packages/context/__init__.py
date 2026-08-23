"""CAP-003 Context Optimization -- intent, budget, compaction (FR-03)."""

from .contracts import SPEC, ContextOptimizationCapability
from .services import ContextBudgeter, ContextOptimizer, IntentAnalyzer

__all__ = ["SPEC", "ContextOptimizationCapability", "ContextBudgeter",
           "ContextOptimizer", "IntentAnalyzer"]
