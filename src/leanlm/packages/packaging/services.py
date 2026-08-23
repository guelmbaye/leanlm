"""CAP-008 Submission Packaging -- emit and verify a template-conformant package.

The builder writes exactly what `adtc-2026-submission-template` declares, and the
checker verifies the template's own checklist rather than one we imagined. Where
a check cannot be performed here (the evaluator's sandbox, the audit re-run) it
says so instead of passing.
"""
from __future__ import annotations

import fnmatch
import json
import re
import stat
from pathlib import Path
from typing import Any

from ...shared.clock import iso_now
from ...shared.errors import packaging_error
from .models import (DOMAINS, PACKAGING_MODES, PLACEHOLDERS, SubmissionRequest,
                     SubmissionResult)
from .policies import CheckResult, ComplianceReport, PackagingPolicy
from .report import render_report

GITIGNORE = """# Model weights are downloaded by download_model.sh, never committed.
*.gguf
model/
!model/.gitkeep

__pycache__/
*.py[cod]
.venv/
submission.json
audit.json
verdict.json
"""

DOWNLOAD_TEMPLATE = """#!/usr/bin/env bash
# Download the model weights. Idempotent, credential-free, and verified.
#
# The evaluator runs this before profiling begins; once profiling starts no
# outbound request is permitted. Everything below happens in that window.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH="{model_path}"
MODEL_URL="{model_url}"
MODEL_SHA256="{model_sha256}"

mkdir -p "$(dirname "${{MODEL_PATH}}")"

# Idempotent: a completed download is verified, never fetched again.
if [[ -f "${{MODEL_PATH}}" ]]; then
  echo "  ${{MODEL_PATH}} already present; verifying"
else
  if [[ -z "${{MODEL_URL}}" ]]; then
    echo "error: no model URL was declared at packaging time" >&2
    exit 1
  fi
  echo "  downloading ${{MODEL_URL}}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${{MODEL_PATH}}.part" "${{MODEL_URL}}"
  else
    wget -O "${{MODEL_PATH}}.part" "${{MODEL_URL}}"
  fi
  mv "${{MODEL_PATH}}.part" "${{MODEL_PATH}}"
fi

# A GGUF file starts with the four bytes "GGUF". An HTML error page does not.
MAGIC=$(head -c 4 "${{MODEL_PATH}}" || true)
if [[ "${{MAGIC}}" != "GGUF" ]]; then
  echo "error: ${{MODEL_PATH}} is not a GGUF file (magic: '${{MAGIC}}')" >&2
  rm -f "${{MODEL_PATH}}"
  exit 1
fi

if [[ -n "${{MODEL_SHA256}}" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL=$(sha256sum "${{MODEL_PATH}}" | awk '{{print $1}}')
  else
    ACTUAL=$(shasum -a 256 "${{MODEL_PATH}}" | awk '{{print $1}}')
  fi
  if [[ "${{ACTUAL}}" != "${{MODEL_SHA256}}" ]]; then
    echo "error: checksum mismatch" >&2
    echo "  expected ${{MODEL_SHA256}}" >&2
    echo "  actual   ${{ACTUAL}}" >&2
    exit 1
  fi
  echo "  checksum verified"
fi

echo "  ready: ${{MODEL_PATH}}"
"""


class SubmissionBuilder:
    def __init__(self, policy: PackagingPolicy | None = None) -> None:
        self.policy = policy or PackagingPolicy()

    def build(self, request: SubmissionRequest) -> SubmissionResult:
        out = Path(request.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        (out / "metadata.json").write_text(
            json.dumps(request.metadata(), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8")

        script = out / "download_model.sh"
        script.write_text(DOWNLOAD_TEMPLATE.format(
            model_path=request.model.model_path,
            model_url=request.model.source_url,
            model_sha256=request.model.sha256,
        ), encoding="utf-8")
        script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP)

        (out / "REPORT.md").write_text(render_report(request), encoding="utf-8")
        (out / ".gitignore").write_text(GITIGNORE, encoding="utf-8")

        model_dir = out / Path(request.model.model_path).parent
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / ".gitkeep").write_text("", encoding="utf-8")

        files = tuple(sorted(
            str(p.relative_to(out)).replace("\\", "/")
            for p in out.rglob("*") if p.is_file()))
        size_mb = round(sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
                        / 1048576.0, 3)
        compliance = ComplianceChecker(self.policy).check(out, request, size_mb)
        return SubmissionResult(str(out), files, compliance, size_mb)


class ComplianceChecker:
    """The template's checklist, executed.

    Numbered SUB-0xx for traceability; the label is the template's own wording
    wherever there is one.
    """

    def __init__(self, policy: PackagingPolicy | None = None) -> None:
        self.policy = policy or PackagingPolicy()

    def check(self, output: Path, request: SubmissionRequest,
              size_mb: float = 0.0) -> ComplianceReport:
        checks: list[CheckResult] = []
        metadata = request.metadata()
        add = checks.append

        # -- structure --------------------------------------------------------
        missing = [name for name in self.policy.required_files
                   if not (output / name).is_file()]
        add(CheckResult("SUB-001", "required files present", not missing,
                        f"missing: {', '.join(missing)}" if missing else ""))
        missing_dirs = [d for d in self.policy.required_dirs if not (output / d).is_dir()]
        add(CheckResult("SUB-002", "model/ directory present", not missing_dirs,
                        f"missing: {', '.join(missing_dirs)}" if missing_dirs else ""))

        offenders = [f for f in (str(p.relative_to(output)) for p in output.rglob("*")
                                 if p.is_file())
                     if any(fnmatch.fnmatch(f, pattern)
                            for pattern in self.policy.forbidden_patterns)]
        add(CheckResult("SUB-003", "no weights or secrets in the package",
                        not offenders,
                        f"found: {', '.join(offenders)}" if offenders else ""))

        gitignore = (output / ".gitignore")
        ignored = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
        add(CheckResult("SUB-004", ".gitignore excludes *.gguf and model/",
                        "*.gguf" in ignored and "model/" in ignored,
                        "rule 2: the evaluator downloads weights fresh"))

        # -- metadata ---------------------------------------------------------
        flat = json.dumps(metadata, ensure_ascii=False)
        remaining = sorted(p for p in PLACEHOLDERS if p in flat)
        add(CheckResult("SUB-005", "no placeholder values remain", not remaining,
                        f"still present: {', '.join(remaining)}" if remaining else ""))

        add(CheckResult("SUB-006", "domain is a valid enum value",
                        metadata["domain"] in DOMAINS,
                        f"got '{metadata['domain']}'; one of {', '.join(DOMAINS)}"))

        prompts = metadata["test_prompts"]
        exact = len(prompts) == self.policy.required_test_prompts
        add(CheckResult("SUB-007",
                        f"exactly {self.policy.required_test_prompts} test prompts",
                        exact and all(p.get("prompt", "").strip() for p in prompts),
                        f"got {len(prompts)}"))

        add(CheckResult("SUB-008", "submitter fully identified",
                        request.submitter.complete,
                        "name, email and github_handle are all required"))

        add(CheckResult("SUB-009", "budget_laptop_claim is true",
                        metadata["budget_laptop_claim"] is True,
                        "all submissions target the 8 GB laptop profile"))

        add(CheckResult("SUB-010", "runtime is llama.cpp",
                        metadata["model"]["runtime"] == "llama.cpp",
                        "no other runtime is accepted"))
        add(CheckResult("SUB-011", "quantization is a GGUF format",
                        "GGUF" in metadata["model"]["quantization"].upper(),
                        f"got '{metadata['model']['quantization']}'"))
        add(CheckResult("SUB-012", "packaging mode is valid",
                        metadata["model"]["packaging"] in PACKAGING_MODES,
                        f"one of {', '.join(PACKAGING_MODES)}"))
        add(CheckResult("SUB-013", "language_scope is declared",
                        bool(metadata["language_scope"]),
                        "at least one BCP-47 code"))

        # -- the download script ---------------------------------------------
        script = output / "download_model.sh"
        source = script.read_text(encoding="utf-8") if script.is_file() else ""
        declared = metadata["_runtime"]["model_path"]
        add(CheckResult("SUB-014", "download path matches _runtime.model_path",
                        declared in source,
                        f"the script must write to {declared}"))
        add(CheckResult("SUB-015", "download script needs no credentials",
                        not re.search(r"(token|password|api[_-]?key|Authorization)",
                                      source, re.IGNORECASE),
                        "weights must be publicly accessible"))
        add(CheckResult("SUB-016", "model URL is declared",
                        bool(request.model.source_url),
                        "the evaluator fetches the weights from it"))
        add(CheckResult("SUB-017", "model checksum is declared",
                        bool(request.model.sha256),
                        "without it an evaluator cannot verify what it measured",
                        blocking=False))

        # -- the report -------------------------------------------------------
        report = (output / "REPORT.md")
        text = report.read_text(encoding="utf-8") if report.is_file() else ""
        absent = [s for s in self.policy.report_sections if s.lower() not in text.lower()]
        add(CheckResult("SUB-018", "report covers the required sections", not absent,
                        f"missing: {', '.join(absent)}" if absent else ""))
        add(CheckResult("SUB-019", "report is substantive",
                        len(text) >= self.policy.min_report_chars,
                        f"{len(text)} chars; judges and an LLM audit read this"))

        # -- evidence ---------------------------------------------------------
        simulated = bool(request.benchmark_summary.get("is_simulated"))
        add(CheckResult("SUB-020", "results come from a real model",
                        not simulated or self.policy.allow_simulated_results,
                        "a simulated backend produced these numbers"))
        add(CheckResult("SUB-021", "package is small enough to review",
                        size_mb <= self.policy.max_package_mb,
                        f"{size_mb} MB"))
        add(CheckResult("SUB-022", "git commit recorded", bool(request.git_commit),
                        "helps an evaluator reproduce the exact tree", blocking=False))

        # -- what we cannot check here ----------------------------------------
        add(CheckResult(
            "SUB-023", "official profiler report produced",
            bool(request.profiler_report),
            "run `adtc-profiler run --submission . --mode participant`; our own "
            "cProfile report is not a substitute", blocking=False))

        return ComplianceReport(tuple(checks), {
            "generated_at": iso_now(), "size_mb": size_mb,
            "files": len([p for p in output.rglob("*") if p.is_file()]),
        })

    def enforce(self, report: ComplianceReport) -> None:
        if report.passed:
            return
        failed = [c for c in report.failures if c.blocking]
        detail = "; ".join(f"{c.check_id} {c.label}" for c in failed)
        raise packaging_error(
            "PKG-001", f"submission is not conformant: {detail}",
            capability="packaging", stage="compliance_check",
            recommended_action="fix the failing checks; the template's checklist is "
                               "the authority, not this tool",
            details=report.to_dict(),
        )
