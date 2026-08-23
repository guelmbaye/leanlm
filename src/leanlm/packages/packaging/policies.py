"""Packaging policy for CAP-008 -- the official template's file list.

Five files, not the nine we invented. `model_manifest.json`,
`runtime_profile.yaml`, `MANIFEST.sha256` and `evidence/` were ours; an evaluator
reads none of them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# From the template's "Required File Structure".
REQUIRED_FILES = ("metadata.json", "download_model.sh", "REPORT.md", ".gitignore")
REQUIRED_DIRS = ("model",)

# Rule 2: "No model weights in git."
FORBIDDEN_PATTERNS = ("*.gguf", "*.bin", "*.safetensors", ".env", "*.key", "*.pem")

# The four sections REPORT.md must cover, per the template.
REPORT_SECTIONS = ("Problem", "Design Decisions", "Constraints", "Benchmarks")


@dataclass(frozen=True, slots=True)
class PackagingPolicy:
    required_files: tuple[str, ...] = REQUIRED_FILES
    required_dirs: tuple[str, ...] = REQUIRED_DIRS
    forbidden_patterns: tuple[str, ...] = FORBIDDEN_PATTERNS
    report_sections: tuple[str, ...] = REPORT_SECTIONS
    required_test_prompts: int = 2
    min_report_chars: int = 1200
    allow_simulated_results: bool = False
    max_package_mb: float = 25.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "PackagingPolicy":
        node = (config or {}).get("submission", {}) or {}
        defaults = cls()
        return cls(
            required_files=tuple(node.get("required_files") or defaults.required_files),
            required_dirs=tuple(node.get("required_dirs") or defaults.required_dirs),
            forbidden_patterns=tuple(node.get("forbidden_patterns")
                                     or defaults.forbidden_patterns),
            report_sections=tuple(node.get("report_sections")
                                  or defaults.report_sections),
            required_test_prompts=int(node.get("required_test_prompts",
                                               defaults.required_test_prompts)),
            min_report_chars=int(node.get("min_report_chars",
                                          defaults.min_report_chars)),
            allow_simulated_results=bool(node.get("allow_simulated_results", False)),
            max_package_mb=float(node.get("max_package_mb", defaults.max_package_mb)),
        )


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    label: str
    passed: bool
    detail: str = ""
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"check_id": self.check_id, "label": self.label, "passed": self.passed,
                "detail": self.detail, "blocking": self.blocking}


@dataclass(frozen=True, slots=True)
class ComplianceReport:
    checks: tuple[CheckResult, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.blocking)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(c for c in self.checks if not c.passed)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks],
                "details": dict(self.details)}
