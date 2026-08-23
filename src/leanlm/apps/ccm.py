"""Capability-to-Code Mapping verifier (COA s.6, CCM).

A capability that exists on a slide and not in the tree is a claim, not an
architecture. This checks, for each of the eight capabilities, that the code,
the contract, the tests, the benchmark and the documentation actually exist and
that the declared owner of each pipeline stage is unique.

It is wired into CI: the mapping cannot silently rot.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..contracts.capability import CapabilitySpec
from ..contracts.iec import STAGE_ORDER

EXPECTED_MODULES = ("__init__.py", "contracts.py", "models.py", "policies.py",
                    "services.py", "telemetry.py", "benchmarks.py", "README.md")

CAPABILITY_PACKAGES: dict[str, str] = {
    "CAP-001": "ingestion",
    "CAP-002": "understanding",
    "CAP-003": "context",
    "CAP-004": "retrieval",
    "CAP-005": "inference",
    "CAP-006": "validation",
    "CAP-007": "performance",
    "CAP-008": "packaging",
}


@dataclass
class CapabilityMapping:
    capability_id: str
    package: str
    present: bool = False
    missing_modules: tuple[str, ...] = ()
    stages: tuple[str, ...] = ()
    test_file: str = ""
    has_tests: bool = False
    has_benchmark: bool = False
    has_readme: bool = False
    requirements: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return (self.present and not self.missing_modules and self.has_tests
                and self.has_benchmark and self.has_readme)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability_id": self.capability_id, "package": self.package,
            "present": self.present, "missing_modules": list(self.missing_modules),
            "stages": list(self.stages), "test_file": self.test_file,
            "has_tests": self.has_tests, "has_benchmark": self.has_benchmark,
            "has_readme": self.has_readme, "requirements": list(self.requirements),
            "complete": self.complete,
        }


@dataclass
class CCMReport:
    mappings: list[CapabilityMapping] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    stage_owners: dict[str, list[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems and all(m.complete for m in self.mappings)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mappings": [m.to_dict() for m in self.mappings],
            "problems": list(self.problems),
            "stage_owners": {k: list(v) for k, v in self.stage_owners.items()},
        }

    def render(self) -> str:
        lines = ["| Capability | Package | Modules | Stages | Tests | Bench | Doc |",
                 "|---|---|---|---|---|---|---|"]
        for m in self.mappings:
            modules = "ok" if m.present and not m.missing_modules else \
                f"missing: {', '.join(m.missing_modules) or 'package'}"
            lines.append(
                f"| {m.capability_id} | `packages/{m.package}/` | {modules} "
                f"| {', '.join(m.stages) or '-'} | {'yes' if m.has_tests else 'NO'} "
                f"| {'yes' if m.has_benchmark else 'NO'} "
                f"| {'yes' if m.has_readme else 'NO'} |")
        if self.problems:
            lines.append("")
            lines.extend(f"- PROBLEM: {p}" for p in self.problems)
        return "\n".join(lines)


def verify(repo_root: Path | None = None,
           specs: list[CapabilitySpec] | None = None) -> CCMReport:
    root = repo_root or Path(__file__).resolve().parents[3].parent
    if not (root / "src" / "leanlm").is_dir():
        root = Path(__file__).resolve().parents[3]
        root = root.parent if not (root / "src").is_dir() else root
    packages_dir = root / "src" / "leanlm" / "packages"
    tests_dir = root / "tests" / "capability"
    report = CCMReport()

    declared = {spec.capability_id: spec for spec in (specs or _discover_specs())}

    for capability_id, package in sorted(CAPABILITY_PACKAGES.items()):
        folder = packages_dir / package
        mapping = CapabilityMapping(capability_id, package, present=folder.is_dir())
        if mapping.present:
            mapping.missing_modules = tuple(
                name for name in EXPECTED_MODULES if not (folder / name).is_file())
            mapping.has_readme = (folder / "README.md").is_file()
            benchmark = folder / "benchmarks.py"
            mapping.has_benchmark = benchmark.is_file() and benchmark.stat().st_size > 200
        else:
            report.problems.append(
                f"{capability_id}: package `packages/{package}/` does not exist")
        test_file = tests_dir / f"test_{package}.py"
        mapping.test_file = str(test_file.relative_to(root)) if test_file.is_file() else ""
        mapping.has_tests = test_file.is_file()
        if not mapping.has_tests:
            report.problems.append(
                f"{capability_id}: no capability test at tests/capability/test_{package}.py")
        spec = declared.get(capability_id)
        if spec is not None:
            mapping.stages = tuple(spec.stages)
            mapping.requirements = tuple(spec.requirement_ids)
            for stage in spec.stages:
                report.stage_owners.setdefault(stage, []).append(capability_id)
        report.mappings.append(mapping)

    # Every non-runtime stage must have exactly one owner (ICIB-007).
    runtime_stages = {"session_creation", "resource_assessment", "document_discovery",
                      "session_cleanup"}
    for stage in (s.value for s in STAGE_ORDER):
        owners = report.stage_owners.get(stage, [])
        if stage in runtime_stages:
            if owners:
                report.problems.append(
                    f"stage {stage} is a runtime stage but is claimed by {owners}")
            continue
        if len(owners) != 1:
            report.problems.append(
                f"stage {stage} has {len(owners)} owners ({owners or 'none'}); "
                "exactly one capability must own it")
    return report


def _discover_specs() -> list[CapabilitySpec]:
    """Read the specs from the code itself, not from a hand-written list.

    Each package declares its own SPEC in ``contracts.py``; the verifier imports
    them so that the mapping can never drift from what the code says.
    """
    import importlib

    specs: list[CapabilitySpec] = []
    for package in CAPABILITY_PACKAGES.values():
        module = importlib.import_module(f"leanlm.packages.{package}.contracts")
        spec = getattr(module, "SPEC", None)
        if isinstance(spec, CapabilitySpec):
            specs.append(spec)
    return specs
