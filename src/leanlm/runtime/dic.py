"""Deterministic Inference Contract verification (V&V s.9).

The DIC is only worth something if it can be *checked*. These functions run
after every inference in development and benchmark profiles, and in the test
suite, turning seven written guarantees into seven assertions.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..contracts.iec import STAGE_ORDER, InferenceExecutionContext


@dataclass(frozen=True, slots=True)
class DICViolation:
    rule: str
    detail: str


def verify(iec: InferenceExecutionContext, *, allow_simulated: bool = True
           ) -> tuple[DICViolation, ...]:
    violations: list[DICViolation] = []
    executed = [record.stage for record in iec.stages]
    canonical = [stage.value for stage in STAGE_ORDER]

    # DIC-01 fixed lifecycle: executed stages must be a prefix-ordered subset.
    positions = [canonical.index(stage) for stage in executed if stage in canonical]
    if positions != sorted(positions):
        violations.append(DICViolation("DIC-01", f"stages ran out of order: {executed}"))
    unknown = [stage for stage in executed if stage not in canonical]
    if unknown:
        violations.append(DICViolation("DIC-01", f"unknown stages executed: {unknown}"))

    # DIC-02 stable configuration.
    if not iec.profile_id:
        violations.append(DICViolation("DIC-02", "no runtime profile recorded"))

    # DIC-03 metrics collection.
    if iec.metrics is None:
        violations.append(DICViolation("DIC-03", "no metrics snapshot produced"))
    else:
        required = ("total_ms", "prompt_tokens", "peak_rss_mb")
        missing = [f for f in required if getattr(iec.metrics, f, None) is None]
        if missing:
            violations.append(DICViolation("DIC-03", f"missing metrics: {missing}"))

    # DIC-04 traceability.
    if not iec.request_id:
        violations.append(DICViolation("DIC-04", "no request identifier"))
    for evidence in iec.evidence:
        if not evidence.unit_id or not evidence.document_name:
            violations.append(
                DICViolation("DIC-04", f"evidence {evidence.label} is not traceable")
            )

    # DIC-05 resource safety: the prompt must have respected the budget.
    if iec.optimized_context and iec.optimized_context.budget and iec.prompt:
        budget = iec.optimized_context.budget
        ceiling = budget.max_context_tokens - budget.reserved_output_tokens
        if iec.prompt.prompt_tokens > ceiling:
            violations.append(DICViolation(
                "DIC-05",
                f"prompt {iec.prompt.prompt_tokens} tk exceeds the usable window {ceiling} tk",
            ))

    # DIC-06 offline integrity is enforced by the guard; report what it saw.
    breaches = iec.trace.get("offline", {}).get("violations") or []
    if breaches:
        violations.append(DICViolation("DIC-06", f"network access attempted: {breaches}"))

    # DIC-07 failure transparency.
    for record in iec.stages:
        if record.status == "failed" and not record.errors:
            violations.append(
                DICViolation("DIC-07", f"stage {record.stage} failed without an error record")
            )

    if not allow_simulated and iec.response is not None and iec.response.is_simulated:
        violations.append(
            DICViolation("DIC-02", "simulated backend used where a real model was required")
        )
    return tuple(violations)


def assert_contract(iec: InferenceExecutionContext, **kwargs) -> None:
    violations = verify(iec, **kwargs)
    if violations:
        detail = "; ".join(f"{v.rule}: {v.detail}" for v in violations)
        from ..shared.errors import contract_error
        raise contract_error(
            "DIC-000", f"Deterministic Inference Contract violated -- {detail}",
            capability="runtime",
            recommended_action="the pipeline is not conformant; do not report these results",
        )
