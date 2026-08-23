"""CAP-008 public contract (delivery pipeline)."""
from __future__ import annotations

from ...contracts.capability import PIPELINE_DELIVERY, CapabilitySpec
from .models import SubmissionRequest, SubmissionResult
from .policies import PackagingPolicy
from .services import ComplianceChecker, SubmissionBuilder
from .telemetry import METRICS

SPEC = CapabilitySpec(
    capability_id="CAP-008",
    name="Submission Packaging",
    package="packages/packaging",
    pipeline=PIPELINE_DELIVERY,
    stages=("submission_build", "compliance_check"),
    requirement_ids=("FR-08",),
    adtc_criteria=("Submission",),
    estimated_memory_mb=16.0,
    expected_duration_ms=250.0,
    produces_metrics=METRICS,
)


class SubmissionPackagingCapability:
    """Builds the deliverable and refuses to hand over a non-compliant one."""

    spec = SPEC

    def __init__(self, policy: PackagingPolicy | None = None) -> None:
        self.policy = policy or PackagingPolicy()
        self.builder = SubmissionBuilder(self.policy)
        self.checker = ComplianceChecker(self.policy)

    def build(self, request: SubmissionRequest, *, enforce: bool = True) -> SubmissionResult:
        result = self.builder.build(request)
        if enforce:
            self.checker.enforce(result.compliance)
        return result
