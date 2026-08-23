"""Canonical Object Model (DMSB s.14 / RPCB s.14).

Every artifact that circulates in LeanLM inherits the same identity, version,
metadata, lifecycle and serialization rules. That uniformity is what makes the
Knowledge & Inference Artifact Graph (KIAG) possible: any object can state what
it is, where it came from and whether it is intact.

Canonical objects are frozen. A transformation never mutates its input; it
produces a *child* artifact that names its parent. The chain of parents is the
trust chain (TILAB s.10).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar

from .clock import iso_now
from .hashing import sha256_text
from .ids import artifact_id
from .serialization import canonical_json, to_plain


class LifecycleState(str, enum.Enum):
    CREATED = "created"
    VALIDATED = "validated"
    CONSUMED = "consumed"
    ARCHIVED = "archived"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ArtifactMetadata:
    """The common metadata socle carried by every canonical object."""

    artifact_id: str
    artifact_type: str
    schema_version: str
    created_at: str = field(default_factory=iso_now)
    parent_id: str | None = None
    source: str | None = None
    pipeline_stage: str | None = None
    checksum: str = ""
    lifecycle_state: LifecycleState = LifecycleState.CREATED
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_type": self.artifact_type,
            "schema_version": self.schema_version,
            "created_at": self.created_at,
            "parent_id": self.parent_id,
            "source": self.source,
            "pipeline_stage": self.pipeline_stage,
            "checksum": self.checksum,
            "lifecycle_state": self.lifecycle_state.value,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True)
class CanonicalObject:
    """Base class of every LeanLM artifact.

    Subclasses declare ``SCHEMA_VERSION`` and ``identity()``. Identity parts are
    the fields that define *what the object is*; they drive the deterministic
    artifact id and the content checksum. Two runs producing the same content
    therefore produce the same id -- which is exactly what a reproducibility
    claim needs.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0.0"

    meta: ArtifactMetadata

    # -- identity -----------------------------------------------------------
    def identity(self) -> tuple[str, ...]:  # pragma: no cover - overridden
        return (canonical_json(self.content()),)

    def content(self) -> dict[str, Any]:
        """Content used for the checksum. Excludes metadata by construction."""
        payload = to_plain(self)
        if isinstance(payload, dict):
            payload.pop("meta", None)
        return payload

    # -- construction helpers ----------------------------------------------
    @classmethod
    def build_meta(
        cls,
        *,
        identity: tuple[str, ...],
        parent_id: str | None = None,
        source: str | None = None,
        stage: str | None = None,
        **extra: Any,
    ) -> ArtifactMetadata:
        artifact_type = cls.__name__
        return ArtifactMetadata(
            artifact_id=artifact_id(artifact_type, *identity),
            artifact_type=artifact_type,
            schema_version=cls.SCHEMA_VERSION,
            parent_id=parent_id,
            source=source,
            pipeline_stage=stage,
            extra=dict(extra),
        )

    def sealed(self) -> "CanonicalObject":
        """Return the same object with its content checksum computed."""
        checksum = sha256_text(canonical_json(self.content()))
        return replace(self, meta=replace(self.meta, checksum=checksum))

    def with_state(self, state: LifecycleState) -> "CanonicalObject":
        return replace(self, meta=replace(self.meta, lifecycle_state=state))

    # -- verification -------------------------------------------------------
    def verify_checksum(self) -> bool:
        if not self.meta.checksum:
            return False
        return self.meta.checksum == sha256_text(canonical_json(self.content()))

    @property
    def artifact_id(self) -> str:
        return self.meta.artifact_id

    def to_dict(self) -> dict[str, Any]:
        payload = {
            f: to_plain(getattr(self, f))
            for f in self.__dataclass_fields__  # type: ignore[attr-defined]
        }
        payload["meta"] = self.meta.to_dict()
        return payload

    def to_json(self, indent: int | None = 2) -> str:
        return canonical_json(self.to_dict(), indent=indent)
