"""Package-local models for CAP-008, shaped by the official template.

Every field here exists because `adtc-2026-submission-template/metadata.json`
declares it. The previous version of this file was our own invention -- a
`model_manifest`, a `runtime_profile`, an `evidence_files` list -- none of which
the template asks for and none of which an evaluator reads.

Recorded in EDB-027: we designed a submission format from a blueprint and shipped
eleven compliance checks against it. A format an organiser publishes is not
something to infer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# `domain` is an enum in the template's field reference, not free text.
DOMAINS = (
    "math_scientific_reasoning", "healthcare_medical", "agriculture",
    "creative_writing", "coding_assistants", "corporate_enterprise",
    "autonomous_ai_agents",
)

PACKAGING_MODES = ("docker_image", "docker_build_from_repo", "binary_bundle")

# Values shipped in the template. A field still holding one of these has not been
# filled in, whatever else it may look like.
PLACEHOLDERS = frozenset({
    "your-team-id", "your-name", "your-email@domain.com", "your-github",
    "YourModel-Q4_K_M", "model/your-model.gguf",
    "Brief description of how your model serves a real-world domain.",
    "Your first test prompt, written for your chosen domain.",
    "Your second test prompt, written for your chosen domain.",
})


@dataclass(frozen=True, slots=True)
class TestPrompt:
    """One of exactly two prompts judged qualitatively.

    The organisers add two hidden prompts in the same domain to test for
    overfitting, so a prompt tuned to what our pipeline answers well buys
    nothing and signals a great deal.
    """

    __test__ = False   # a submission field, not a pytest collection target

    prompt_id: str
    prompt: str

    def to_dict(self) -> dict[str, str]:
        return {"prompt_id": self.prompt_id, "prompt": self.prompt}


@dataclass(frozen=True, slots=True)
class Submitter:
    name: str = ""
    email: str = ""
    github_handle: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "email": self.email,
                "github_handle": self.github_handle}

    @property
    def complete(self) -> bool:
        return all((self.name, self.email, self.github_handle))


@dataclass(frozen=True, slots=True)
class CrossDisciplinaryPairing:
    discipline: str = ""
    load_bearing: bool = True
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"discipline": self.discipline, "load_bearing": self.load_bearing,
                "description": self.description}


@dataclass(frozen=True, slots=True)
class ModelDeclaration:
    name: str = ""
    runtime: str = "llama.cpp"           # the template accepts nothing else
    quantization: str = "GGUF Q4_K_M"
    parameters_estimate: str = ""
    packaging: str = "binary_bundle"
    model_path: str = "model/model.gguf"
    source_url: str = ""                 # for download_model.sh, not for metadata
    sha256: str = ""
    license: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "runtime": self.runtime,
                "quantization": self.quantization,
                "parameters_estimate": self.parameters_estimate,
                "packaging": self.packaging}


@dataclass(frozen=True, slots=True)
class SubmissionRequest:
    """Everything needed to emit a template-conformant submission."""

    output_dir: str
    team_id: str = ""
    domain: str = "corporate_enterprise"
    language_scope: tuple[str, ...] = ("en",)
    african_alpha_claim: bool = False
    budget_laptop_claim: bool = True     # the template requires true
    submitter: Submitter = field(default_factory=Submitter)
    pairing: CrossDisciplinaryPairing = field(default_factory=CrossDisciplinaryPairing)
    test_prompts: tuple[TestPrompt, ...] = ()
    model: ModelDeclaration = field(default_factory=ModelDeclaration)

    # Ours, for REPORT.md only. Not part of metadata.json.
    benchmark_summary: dict[str, Any] = field(default_factory=dict)
    accuracy_summary: dict[str, Any] = field(default_factory=dict)
    naive_comparison: dict[str, Any] = field(default_factory=dict)
    profiler_report: dict[str, Any] | None = None
    repository_url: str = ""
    git_commit: str = ""

    def metadata(self) -> dict[str, Any]:
        """Exactly the shape `metadata.json` declares, in its own key order."""
        return {
            "team_id": self.team_id,
            "domain": self.domain,
            "language_scope": list(self.language_scope),
            "african_alpha_claim": self.african_alpha_claim,
            "budget_laptop_claim": self.budget_laptop_claim,
            "submitter": self.submitter.to_dict(),
            "cross_disciplinary_pairing": self.pairing.to_dict(),
            "test_prompts": [p.to_dict() for p in self.test_prompts],
            "model": self.model.to_dict(),
            "_runtime": {"model_path": self.model.model_path},
        }


@dataclass(frozen=True, slots=True)
class SubmissionResult:
    output_dir: str
    files: tuple[str, ...]
    compliance: Any
    size_mb: float

    def to_dict(self) -> dict[str, Any]:
        return {"output_dir": self.output_dir, "files": list(self.files),
                "size_mb": self.size_mb,
                "compliance": self.compliance.to_dict() if self.compliance else None}
