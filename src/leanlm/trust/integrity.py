"""Model and runtime integrity (TL-02, TL-04).

Two questions the jury is entitled to ask, answered by measurement:
  * is the file that was loaded the file that was declared?
  * is the runtime that produced these numbers the one in the manifest?
"""
from __future__ import annotations

import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..shared.errors import trust_error
from ..shared.hashing import sha256_file, sha256_text


@dataclass(frozen=True, slots=True)
class TrustReport:
    checks: dict[str, bool] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def trusted(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {"trusted": self.trusted, "checks": dict(self.checks),
                "details": dict(self.details)}


def verify_model_binding(path: str, expected_checksum: str = "",
                         *, require_gguf: bool = True) -> TrustReport:
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}
    target = Path(path)

    checks["model_present"] = target.is_file()
    details["path"] = str(target)
    if not checks["model_present"]:
        details["hint"] = "run scripts/download_model.sh"
        return TrustReport(checks, details)

    details["size_mb"] = round(target.stat().st_size / 1048576.0, 2)

    if require_gguf:
        with open(target, "rb") as handle:
            magic = handle.read(4)
        checks["gguf_magic"] = magic == b"GGUF"
        details["magic"] = magic.decode("ascii", "replace")
    checksum = sha256_file(target)
    details["sha256"] = checksum
    if expected_checksum:
        checks["checksum_matches"] = checksum == expected_checksum
        details["expected_sha256"] = expected_checksum
    else:
        details["note"] = ("no expected checksum declared; record it in the model "
                           "manifest before freezing the model (MSOB s.11)")
    return TrustReport(checks, details)


def enforce_model_binding(report: TrustReport) -> None:
    if report.trusted:
        return
    failed = sorted(k for k, v in report.checks.items() if not v)
    raise trust_error(
        "TRU-010", f"model integrity check failed: {', '.join(failed)}",
        capability="runtime", stage="session_creation",
        recommended_action="re-download the model; never benchmark an unverified file",
        details=report.details,
    )


def verify_runtime_integrity(profile_fingerprint: str, *, package_root: Path | None = None
                             ) -> TrustReport:
    """Fingerprint of the runtime that produced a measurement."""
    root = package_root or Path(__file__).resolve().parents[1]
    sources = sorted(p for p in root.rglob("*.py") if "__pycache__" not in str(p))
    digest = sha256_text("|".join(
        f"{p.relative_to(root)}:{sha256_file(p)}" for p in sources
    ))
    return TrustReport(
        checks={"sources_readable": bool(sources)},
        details={
            "source_fingerprint": digest[:16],
            "source_files": len(sources),
            "profile_fingerprint": profile_fingerprint,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    )
