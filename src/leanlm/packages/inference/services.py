"""Deterministic prompt assembly (IIB s.10).

The prompt has exactly three parts, always in the same order: system
instructions, optimized context, user question. Nothing unjustified is added --
every extra sentence in the system prompt is tokens spent on every single
request, which on a laptop is measurable.
"""
from __future__ import annotations

from ...contracts.dto import PromptPackage, SelectedEvidence
from ...shared.text import DEFAULT_TOKEN_COUNTER, TokenCounter
from .policies import PromptPolicy

SYSTEM_TEMPLATE = """You are LeanLM, an offline assistant answering strictly from the supplied excerpts.

Rules:
1. Use only the excerpts below. Never rely on outside knowledge.
2. Cite the excerpt label in square brackets after each claim, e.g. [S1].
3. If the excerpts do not contain the answer, reply exactly: {insufficient}
4. Quote figures, dates and amounts exactly as they appear.
5. Answer in the language of the question. Be concise and factual.
"""

INSUFFICIENT_ANSWER = "The supplied documents do not allow a conclusion."
INSUFFICIENT_ANSWER_FR = "Les documents fournis ne permettent pas de conclure."

USER_TEMPLATE = """### Excerpts
{evidence}

### Question
{question}

### Answer
"""

NO_EVIDENCE_BLOCK = "(no excerpt matched this question)"


def _looks_french(question: str) -> bool:
    lowered = f" {question.lower()} "
    return any(marker in lowered for marker in
               (" quel", " quelle", " comment", " pourquoi", " quels", " quelles",
                " combien", " est-ce", " liste", " resume", " qui ", " ou ", " des ",
                " les ", " une ", " dans "))


class PromptBuilder:
    """Builds a PromptPackage. Same inputs, byte-identical prompt (IP-01/P4)."""

    def __init__(self, policy: PromptPolicy | None = None,
                 counter: TokenCounter | None = None) -> None:
        self.policy = policy or PromptPolicy()
        self.counter = counter or DEFAULT_TOKEN_COUNTER

    def render_evidence(self, evidence: tuple[SelectedEvidence, ...]) -> str:
        if not evidence:
            return NO_EVIDENCE_BLOCK
        blocks: list[str] = []
        used = 0
        for item in evidence:
            header = f"[{item.label}] {item.document_name}"
            if self.policy.include_section_path and item.section_path:
                header += " > " + " > ".join(item.section_path)
            block = f"{header}\n{item.text.strip()}"
            if used + len(block) > self.policy.max_evidence_chars:
                break
            blocks.append(block)
            used += len(block)
        return self.policy.evidence_separator.join(blocks)

    def build(self, question: str, evidence: tuple[SelectedEvidence, ...]) -> PromptPackage:
        insufficient = INSUFFICIENT_ANSWER_FR if _looks_french(question) else INSUFFICIENT_ANSWER
        system = SYSTEM_TEMPLATE.format(insufficient=insufficient)
        evidence_block = self.render_evidence(evidence)
        # A hybrid reasoning model answers directly when told to. Without the
        # suffix it may spend the whole token budget thinking and emit no answer.
        asked = question.strip()
        if self.policy.thinking_suffix:
            asked = f"{asked} {self.policy.thinking_suffix.strip()}"
        user = USER_TEMPLATE.format(evidence=evidence_block, question=asked)
        rendered = f"{system}\n{user}"
        context_tokens = self.counter.count(evidence_block)
        package = PromptPackage(
            meta=PromptPackage.build_meta(
                identity=(self.policy.template_id, self.policy.template_version,
                          rendered[:200]),
                stage="prompt_assembly",
            ),
            system_prompt=system,
            user_prompt=user,
            rendered=rendered,
            template_id=self.policy.template_id,
            template_version=self.policy.template_version,
            prompt_tokens=self.counter.count(rendered),
            context_tokens=context_tokens,
            evidence_labels=tuple(item.label for item in evidence),
        )
        return package.sealed()  # type: ignore[return-value]


def prompt_efficiency(prompt: PromptPackage) -> float:
    """Share of the prompt that is actual evidence rather than scaffolding.

    A Tier-3 metric worth watching: when it collapses, the system prompt or the
    formatting is eating the budget the evidence should have.
    """
    if prompt.prompt_tokens <= 0:
        return 0.0
    return round(prompt.context_tokens / prompt.prompt_tokens, 4)
