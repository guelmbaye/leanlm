"""Runtime profiles (RPCB).

Every execution runs under a named, versioned profile. There is no implicit
configuration: if a parameter influenced a measurement, the profile that carried
it is recorded next to that measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..shared.config import config_dir, get_path, load_yaml
from ..shared.errors import config_error
from ..shared.hashing import sha256_text
from ..shared.serialization import canonical_json

KNOWN_PROFILES = ("development", "benchmark", "demo", "competition")


@dataclass(frozen=True)
class RuntimeProfile:
    """Immutable once published (RC-05)."""

    id: str
    version: str
    description: str = ""
    target_hardware: str = "laptop-8gb"
    backend: str = "auto"
    model: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    optimization: dict[str, Any] = field(default_factory=dict)
    retrieval: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] = field(default_factory=dict)
    prompt: dict[str, Any] = field(default_factory=dict)
    segmentation: dict[str, Any] = field(default_factory=dict)
    ingestion: dict[str, Any] = field(default_factory=dict)
    performance: dict[str, Any] = field(default_factory=dict)
    telemetry: dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @property
    def fingerprint(self) -> str:
        """Checksum of everything that can change a measurement."""
        return sha256_text(canonical_json({
            "id": self.id, "version": self.version, "backend": self.backend,
            "model": self.model, "inference": self.inference, "budget": self.budget,
            "optimization": self.optimization, "retrieval": self.retrieval,
            "validation": self.validation, "prompt": self.prompt,
            "segmentation": self.segmentation,
        }))[:16]

    def as_config(self) -> dict[str, Any]:
        """The flat mapping every policy's ``from_config`` expects."""
        return {
            "inference": self.inference, "budget": self.budget,
            "optimization": self.optimization, "retrieval": self.retrieval,
            "validation": self.validation, "prompt": self.prompt,
            "segmentation": self.segmentation, "ingestion": self.ingestion,
            "performance": self.performance, "memory": self.memory,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "version": self.version, "description": self.description,
            "target_hardware": self.target_hardware, "backend": self.backend,
            "fingerprint": self.fingerprint, "model": self.model,
            "inference": self.inference, "memory": self.memory, "budget": self.budget,
            "optimization": self.optimization, "retrieval": self.retrieval,
            "validation": self.validation, "prompt": self.prompt,
            "segmentation": self.segmentation, "ingestion": self.ingestion,
            "performance": self.performance, "telemetry": self.telemetry,
        }


def profile_path(profile_id: str, directory: Path | None = None) -> Path:
    return (directory or (config_dir() / "runtime")) / f"{profile_id}.yaml"


def list_profiles(directory: Path | None = None) -> list[str]:
    folder = directory or (config_dir() / "runtime")
    if not folder.is_dir():
        return []
    return sorted(p.stem for p in folder.glob("*.yaml"))


def load_profile(profile_id: str = "development",
                 directory: Path | None = None) -> RuntimeProfile:
    path = profile_path(profile_id, directory)
    if not path.is_file():
        available = ", ".join(list_profiles(directory)) or "none"
        raise config_error(
            "RPCB-001", f"unknown runtime profile '{profile_id}'",
            capability="runtime", stage="session_creation",
            recommended_action=f"available profiles: {available}",
        )
    raw = load_yaml(path)
    profile = RuntimeProfile(
        id=str(get_path(raw, "id", profile_id)),
        version=str(get_path(raw, "version", "0.0.0")),
        description=str(get_path(raw, "description", "")),
        target_hardware=str(get_path(raw, "target_hardware", "laptop-8gb")),
        backend=str(get_path(raw, "backend", "auto")),
        model=dict(get_path(raw, "model", {}) or {}),
        inference=dict(get_path(raw, "inference", {}) or {}),
        memory=dict(get_path(raw, "memory", {}) or {}),
        budget=dict(get_path(raw, "budget", {}) or {}),
        optimization=dict(get_path(raw, "optimization", {}) or {}),
        retrieval=dict(get_path(raw, "retrieval", {}) or {}),
        validation=dict(get_path(raw, "validation", {}) or {}),
        prompt=dict(get_path(raw, "prompt", {}) or {}),
        segmentation=dict(get_path(raw, "segmentation", {}) or {}),
        ingestion=dict(get_path(raw, "ingestion", {}) or {}),
        performance=dict(get_path(raw, "performance", {}) or {}),
        telemetry=dict(get_path(raw, "telemetry", {}) or {}),
        source_path=str(path),
    )
    validate_profile(profile)
    return profile


def validate_profile(profile: RuntimeProfile) -> None:
    """RC-08 safe defaults, RC-02 deterministic configuration."""
    if not profile.id:
        raise config_error("RPCB-002", "profile has no id", capability="runtime")
    if not profile.version:
        raise config_error("RPCB-003", f"profile {profile.id} has no version",
                           capability="runtime")
    context = int(profile.model.get("context_tokens", 0) or 0)
    if context <= 0:
        raise config_error(
            "RPCB-004", f"profile {profile.id}: model.context_tokens must be positive",
            capability="runtime",
            recommended_action="set it to the value the GGUF model was trained for",
        )
    max_output = int(profile.inference.get("max_output_tokens", 0) or 0)
    if max_output >= context:
        raise config_error(
            "RPCB-005",
            f"profile {profile.id}: max_output_tokens ({max_output}) leaves no room "
            f"for context in a {context}-token window",
            capability="runtime",
            recommended_action="lower max_output_tokens",
        )
    temperature = float(profile.inference.get("temperature", 0.0) or 0.0)
    if profile.id in ("benchmark", "competition") and temperature > 0.0:
        raise config_error(
            "RPCB-006",
            f"profile {profile.id}: temperature must be 0.0 for reproducible runs",
            capability="runtime",
            recommended_action="R4/VV-02: measured results have to be repeatable",
        )
