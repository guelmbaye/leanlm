"""Identifier generation.

Identifiers are *content-derived* whenever possible: two identical inputs must
produce the same artifact id (DM-09 deterministic serialization, DIC-04
traceability). Only session ids are time based, because a session is by
definition a distinct execution.
"""
from __future__ import annotations

import os

from .clock import wall_now
from .hashing import sha256_text

_ARTIFACT_PREFIXES = {
    "RawDocument": "doc",
    "StructuredDocument": "sdoc",
    "SemanticUnit": "unit",
    "ContextPackage": "ctx",
    "OptimizedContext": "octx",
    "PromptPackage": "prompt",
    "InferenceExecutionContext": "iec",
    "ModelResponse": "resp",
    "ValidatedResponse": "vresp",
    "MetricsSnapshot": "metrics",
    "BenchmarkEvidence": "evidence",
    "RuntimeProfile": "profile",
}


def artifact_id(object_type: str, *parts: str) -> str:
    """Deterministic id derived from the object type and its identity parts."""
    prefix = _ARTIFACT_PREFIXES.get(object_type, object_type.lower()[:6])
    digest = sha256_text("\x1f".join((object_type, *parts)))
    return f"{prefix}_{digest[:16]}"


def session_id() -> str:
    """Unique, sortable session identifier."""
    stamp = wall_now().strftime("%Y%m%dT%H%M%S")
    entropy = sha256_text(f"{stamp}:{os.getpid()}:{os.urandom(8).hex()}")[:8]
    return f"sess_{stamp}_{entropy}"


def request_id(session: str, index: int) -> str:
    return f"{session}#{index:04d}"
