"""The capability contract (ICIB s.5) and the capability registry (s.7).

Every capability implements exactly one signature:

    execute(iec) -> iec

Nothing else crosses a boundary. A capability never reaches into another
capability's implementation, never mutates the context it received, and never
knows which capability runs next.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

from ..shared.errors import contract_error
from .ddp import DocumentPipelineState
from .iec import InferenceExecutionContext, PipelineStage
from .versions import CAPABILITY_CONTRACT_VERSION


PIPELINE_DOCUMENT = "document"
PIPELINE_INFERENCE = "inference"
PIPELINE_DELIVERY = "delivery"


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Resource and traceability contract declared by every capability (MIB s.13).

    A capability owns one or more *consecutive* stages of one pipeline. It never
    owns stages in two pipelines: that would make it two capabilities wearing
    one name.
    """

    capability_id: str
    name: str
    package: str
    pipeline: str
    stages: tuple[str, ...]
    requirement_ids: tuple[str, ...] = ()
    adtc_criteria: tuple[str, ...] = ()
    estimated_memory_mb: float = 8.0
    expected_duration_ms: float = 25.0
    produces_metrics: tuple[str, ...] = ()
    contract_version: str = CAPABILITY_CONTRACT_VERSION

    def owns(self, stage: str) -> bool:
        return stage in self.stages

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "package": self.package,
            "pipeline": self.pipeline,
            "stages": list(self.stages),
            "requirement_ids": list(self.requirement_ids),
            "adtc_criteria": list(self.adtc_criteria),
            "estimated_memory_mb": self.estimated_memory_mb,
            "expected_duration_ms": self.expected_duration_ms,
            "produces_metrics": list(self.produces_metrics),
            "contract_version": self.contract_version,
        }


class Capability(ABC):
    """Base class. Subclasses declare ``spec`` and implement ``run``."""

    spec: CapabilitySpec

    def __init__(self, spec: CapabilitySpec) -> None:
        self.spec = spec

    # -- public contract ----------------------------------------------------
    def execute(self, iec: InferenceExecutionContext,
                stage: PipelineStage) -> InferenceExecutionContext:
        if not self.spec.owns(stage.value):
            raise contract_error(
                "ICIB-006",
                f"{self.spec.capability_id} does not own stage {stage.value}",
                capability=self.spec.name, stage=stage.value,
                recommended_action="check the stage->capability wiring in the runtime",
            )
        self.check_preconditions(iec, stage)
        result = self.run(iec, stage)
        if result is None:
            raise contract_error(
                "ICIB-001", f"{self.spec.capability_id} returned no context",
                capability=self.spec.name, stage=stage.value,
                recommended_action="a capability must always return an enriched IEC",
            )
        if result.revision <= iec.revision:
            raise contract_error(
                "ICIB-002", f"{self.spec.capability_id} did not enrich the context",
                capability=self.spec.name, stage=stage.value,
                recommended_action="use iec.evolve(...) instead of mutating fields",
            )
        self.check_postconditions(result, stage)
        return result

    # -- extension points ---------------------------------------------------
    @abstractmethod
    def run(self, iec: InferenceExecutionContext,
            stage: PipelineStage) -> InferenceExecutionContext:
        """Do the work for ``stage``. Must return ``iec.evolve(...)``."""

    def check_preconditions(self, iec: InferenceExecutionContext,
                            stage: PipelineStage) -> None:
        return None

    def check_postconditions(self, iec: InferenceExecutionContext,
                             stage: PipelineStage) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.spec.capability_id} {self.spec.name}>"


class CapabilityRegistry:
    """Central registry (ICIB s.7). Orchestration and tests read from it."""

    def __init__(self) -> None:
        self._by_id: dict[str, Capability] = {}
        self._order: list[str] = []

    def register(self, capability: Capability) -> Capability:
        cid = capability.spec.capability_id
        if cid in self._by_id:
            raise contract_error("ICIB-003", f"duplicate capability id {cid}",
                                 recommended_action="one package = one capability (MQ-01)")
        self._by_id[cid] = capability
        self._order.append(cid)
        return capability

    def get(self, capability_id: str) -> Capability:
        if capability_id not in self._by_id:
            raise contract_error("ICIB-004", f"unknown capability {capability_id}")
        return self._by_id[capability_id]

    def for_stage(self, stage: PipelineStage | str) -> Capability | None:
        """The single capability owning a stage. None when the runtime owns it."""
        value = stage.value if isinstance(stage, PipelineStage) else stage
        owners = [c for c in self._by_id.values() if c.spec.owns(value)]
        if len(owners) > 1:
            raise contract_error(
                "ICIB-007", f"stage {value} is claimed by {len(owners)} capabilities",
                recommended_action="a stage has exactly one owner (MQ-01)",
            )
        return owners[0] if owners else None

    def specs(self) -> list[CapabilitySpec]:
        return [self._by_id[cid].spec for cid in self._order]

    def __iter__(self) -> Iterator[Capability]:
        return (self._by_id[cid] for cid in self._order)

    def __len__(self) -> int:
        return len(self._by_id)

    def __contains__(self, capability_id: object) -> bool:
        return capability_id in self._by_id


class DocumentCapability(ABC):
    """Capability of the document pipeline (DDPB).

    Same discipline as ``Capability`` -- enrichment only, no mutation -- applied
    to ``DocumentPipelineState`` instead of the IEC.
    """

    spec: CapabilitySpec

    def __init__(self, spec: CapabilitySpec) -> None:
        self.spec = spec

    def process(self, state: DocumentPipelineState) -> DocumentPipelineState:
        result = self.run(state)
        if result is None or result.revision <= state.revision:
            raise contract_error(
                "ICIB-005", f"{self.spec.capability_id} did not enrich the document state",
                capability=self.spec.name,
                recommended_action="use state.evolve(...) / state.advance(...)",
            )
        return result

    @abstractmethod
    def run(self, state: DocumentPipelineState) -> DocumentPipelineState:
        ...

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{self.spec.capability_id} {self.spec.name}>"
