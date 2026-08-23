"""Can this machine produce a submittable measurement, and by which route?

`leanlm doctor` answers "can LeanLM run here". This answers a different and
later question: "can the thing the judges will read be produced here, and if
not, what is the shortest path that works". They are separated because the
answers diverge -- a laptop that runs LeanLM perfectly may be the wrong place to
measure it.

Each route is reported with what it needs, what is missing, and what it cannot
give you even when everything is installed. A route that is available but
unsuitable is worse than one that is plainly unavailable, so suitability is
reported alongside readiness rather than folded into it.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .hardware import PROFILE_RAM_GB, PROFILE_VCPU, inspect


@dataclass(frozen=True, slots=True)
class Requirement:
    name: str
    present: bool
    detail: str = ""
    fix: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "present": self.present,
                "detail": self.detail, "fix": self.fix}


@dataclass(frozen=True, slots=True)
class Route:
    id: str
    label: str
    requirements: tuple[Requirement, ...]
    suitable_for_submission: bool
    caveats: tuple[str, ...] = ()
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return all(r.present for r in self.requirements)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.requirements if not r.present)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "label": self.label, "ready": self.ready,
                "suitable_for_submission": self.suitable_for_submission,
                "missing": list(self.missing),
                "requirements": [r.to_dict() for r in self.requirements],
                "caveats": list(self.caveats)}


def _binary(name: str, fix: str) -> Requirement:
    path = shutil.which(name)
    return Requirement(name, bool(path), path or "not on PATH", fix)


def _module(name: str, fix: str) -> Requirement:
    try:
        __import__(name)
        return Requirement(name, True)
    except ImportError:
        return Requirement(name, False, "not importable", fix)


def _python_version() -> Requirement:
    ok = sys.version_info >= (3, 11)
    return Requirement("python>=3.11", ok, sys.version.split()[0],
                       "the profiler requires 3.11 or newer")


def _docker() -> Requirement:
    path = shutil.which("docker")
    if not path:
        return Requirement("docker", False, "not on PATH",
                           "install Docker Engine or Docker Desktop")
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=10, check=True)
        return Requirement("docker", True, "daemon reachable")
    except Exception:
        return Requirement("docker", False, "installed but the daemon is not running",
                           "start Docker and retry")


def _submission(directory: str | Path) -> Requirement:
    target = Path(directory)
    metadata = target / "metadata.json"
    if not metadata.is_file():
        return Requirement("submission package", False, f"none at {target}",
                           "leanlm submission --output " + str(target))
    try:
        declared = json.loads(metadata.read_text(encoding="utf-8"))
        path = declared.get("_runtime", {}).get("model_path", "")
    except Exception:
        return Requirement("submission package", False, "metadata.json is unreadable",
                           "rebuild it with `leanlm submission`")
    weights = target / path
    if not weights.is_file():
        return Requirement("model weights", False, f"{path} not downloaded",
                           f"cd {target} && bash download_model.sh")
    return Requirement("submission package", True, f"{target} with {path}")


def routes(submission_dir: str | Path = "dist/submission") -> list[Route]:
    hardware = inspect()
    matches_profile = (hardware.logical_cpus >= PROFILE_VCPU
                       and hardware.total_ram_gb >= PROFILE_RAM_GB * 0.9)
    generous_memory = hardware.total_ram_gb > PROFILE_RAM_GB * 1.25

    development = Route(
        "development", "This machine, for development",
        requirements=(
            _module("leanlm", "pip install -e ."),
            Requirement("corpus", True, "datasets/enterprise_en"),
        ),
        suitable_for_submission=False,
        caveats=(
            "functional verification, accuracy on our own probes, model "
            "selection sweeps and the report -- none of which depend on the "
            "hardware",
            "not the place to produce submission.json",
        ),
    )

    native = Route(
        "native", "This machine, running the official profiler directly",
        requirements=(
            _python_version(),
            _binary("llama-bench", "make provision, or build llama.cpp"),
            _binary("adtc-profiler",
                    'pip install "git+https://github.com/Africa-Deep-Tech-Foundation'
                    '/adtc-profiler.git"'),
            _submission(submission_dir),
        ),
        suitable_for_submission=matches_profile and not generous_memory,
        caveats=_native_caveats(hardware, matches_profile, generous_memory),
    )

    container = Route(
        "docker", "This machine, capped to the profile in a container",
        requirements=(_docker(), _submission(submission_dir)),
        suitable_for_submission=hardware.logical_cpus >= PROFILE_VCPU,
        caveats=(
            "--memory=7.5g reproduces the limit that disqualifies; nothing "
            "swaps on a generous machine without it",
            "capping resources reproduces the limit, never the speed: slow "
            "cores stay slow",
        ) + (() if hardware.logical_cpus >= PROFILE_VCPU else (
            f"this host has {hardware.logical_cpus} logical CPUs; Docker will "
            f"not invent the other {PROFILE_VCPU - hardware.logical_cpus}",)),
        notes={"command": "scripts/measure_docker.sh"},
    )

    cloud = Route(
        "cloud", "A 4 vCPU / 8 GB Ubuntu instance",
        requirements=(
            Requirement("provisioning script", Path("scripts/provision_ubuntu.sh").is_file(),
                        "scripts/provision_ubuntu.sh"),
        ),
        suitable_for_submission=True,
        caveats=(
            "the audit itself runs in a cloud VM, so this is closer to the "
            "comparison environment than any laptop",
            "take a dedicated CPU tier: shared vCPUs vary run to run by more "
            "than the 25% you may differ from the audit",
            "no thermal sensors in a VM -- the audit VM has the same limitation",
        ),
        notes={"command": "bash scripts/provision_ubuntu.sh"},
    )

    ci = Route(
        "github-actions", "GitHub Actions, 4 vCPU, logged and re-runnable",
        requirements=(
            Requirement("workflow", Path(".github/workflows/measure.yml").is_file(),
                        ".github/workflows/measure.yml"),
        ),
        suitable_for_submission=True,
        caveats=(
            "free on the public repository the rules already require",
            "the measurement is logged and re-runnable by a judge, which is a "
            "stronger claim than a JSON file you vouch for",
            "the runner has more RAM than the profile, so the workflow caps it",
        ),
        notes={"command": "gh workflow run measure.yml"},
    )

    return [development, native, container, cloud, ci]


def _native_caveats(hardware, matches_profile: bool,
                    generous_memory: bool) -> tuple[str, ...]:
    caveats = list(hardware.findings)
    if matches_profile and not generous_memory:
        caveats.append("this machine matches the profile; measure here directly")
    elif not matches_profile:
        caveats.append(
            "numbers produced here risk exceeding the audit's 50% variance "
            "limit, which fails the comparison rather than lowering the score")
    return tuple(caveats)


def render(all_routes: list[Route]) -> str:
    lines = ["routes to a submittable measurement", ""]
    for route in all_routes:
        state = "ready" if route.ready else f"missing: {', '.join(route.missing)}"
        verdict = ("suitable for submission" if route.suitable_for_submission
                   else "NOT for the submitted numbers")
        lines.append(f"  [{'x' if route.ready else ' '}] {route.label}")
        lines.append(f"      {state}")
        lines.append(f"      {verdict}")
        for requirement in route.requirements:
            if not requirement.present and requirement.fix:
                lines.append(f"      -> {requirement.fix}")
        for caveat in route.caveats:
            lines.append(f"      . {caveat}")
        if route.notes.get("command"):
            lines.append(f"      $ {route.notes['command']}")
        lines.append("")
    usable = [r for r in all_routes if r.suitable_for_submission]
    ready = [r for r in usable if r.ready]
    if ready:
        lines.append(f"  shortest path now: {ready[0].label}")
    elif usable:
        lines.append(f"  shortest path: {usable[0].label} "
                     f"(missing {', '.join(usable[0].missing) or 'nothing'})")
    else:
        lines.append("  no route on this machine produces submittable numbers")
    return "\n".join(lines)
