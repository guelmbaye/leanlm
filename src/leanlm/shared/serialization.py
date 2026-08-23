"""Deterministic serialization (DM-09).

Every canonical object serializes to the *same* bytes for the same content.
That property is what makes checksums, contract tests and reproducible
benchmark evidence possible.
"""
from __future__ import annotations

import dataclasses
import datetime as _dt
import enum
import json
from pathlib import Path
from typing import Any


def to_plain(value: Any) -> Any:
    """Convert an arbitrary LeanLM value into JSON-compatible primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, _dt.datetime):
        return value.astimezone(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        if hasattr(value, "to_dict"):
            return to_plain(value.to_dict())
        return {f.name: to_plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else list(value)
        return [to_plain(v) for v in items]
    if hasattr(value, "to_dict"):
        return to_plain(value.to_dict())
    return str(value)


def canonical_json(value: Any, *, indent: int | None = None) -> str:
    """Canonical UTF-8 JSON: sorted keys, stable separators, no NaN."""
    return json.dumps(
        to_plain(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":") if indent is None else (",", ": "),
        indent=indent,
        allow_nan=False,
    )


def write_json(path: str | Path, value: Any, *, indent: int = 2) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(value, indent=indent) + "\n", encoding="utf-8")
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
