"""Declarative configuration loading (Technical Blueprint s.11, RPCB s.16).

No critical constant is hard coded: everything lives in ``configs/*.yaml``.

PyYAML is used when available. When it is not, a small deterministic parser
covering the subset LeanLM actually writes (nested maps, lists, scalars,
comments, quoted strings) takes over -- so a fresh laptop with nothing but
CPython can still run the runtime. Zero hard dependency is a feature here, not
an accident: the target machine may have no working package index.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .errors import config_error

try:  # pragma: no cover - environment dependent
    import yaml as _yaml
except Exception:  # pragma: no cover
    _yaml = None

_NUM = re.compile(r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")


def _scalar(token: str) -> Any:
    token = token.strip()
    if not token:
        return None
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    low = token.lower()
    if low in ("true", "yes", "on"):
        return True
    if low in ("false", "no", "off"):
        return False
    if low in ("null", "none", "~"):
        return None
    if _NUM.match(token):
        return float(token) if any(c in token for c in ".eE") else int(token)
    return token


def _significant(lines, start_index):
    """Next non-empty, non-comment line at or after ``start_index``."""
    for index in range(start_index, len(lines)):
        raw = lines[index]
        if raw.strip() and not raw.lstrip().startswith("#"):
            return raw
    return None


def _mini_yaml(text: str) -> dict[str, Any]:
    """Parse the YAML subset LeanLM emits. Raises on anything unexpected."""
    lines = text.splitlines()
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for lineno, raw in enumerate(lines, 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = raw.split(" #", 1)[0].rstrip() if " #" in raw else raw.rstrip()
        indent = len(line) - len(line.lstrip(" "))
        body = line.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if body.startswith("- "):
            if not isinstance(parent, list):
                raise config_error("CFG-102", f"list item outside a list at line {lineno}")
            item = body[2:].strip()
            quoted = item[:1] in ("\"", "'")
            if ":" in item and not quoted:
                key, _, value = item.partition(":")
                child: dict[str, Any] = {}
                parent.append(child)
                stack.append((indent, child))
                if value.strip():
                    child[key.strip()] = _scalar(value)
                else:
                    nested: dict[str, Any] = {}
                    child[key.strip()] = nested
                    stack.append((indent + 1, nested))
            else:
                parent.append(_scalar(item))
            continue

        if ":" not in body:
            raise config_error("CFG-103", f"expected 'key: value' at line {lineno}: {body!r}")
        key, _, value = body.partition(":")
        key, value = key.strip(), value.strip()
        if not isinstance(parent, dict):
            raise config_error("CFG-104", f"mapping inside a list at line {lineno}")

        if value == "":
            following = _significant(lines, lineno)
            if following is not None:
                nxt_indent = len(following) - len(following.lstrip(" "))
                is_list = following.lstrip().startswith("- ") and nxt_indent > indent
            else:
                is_list = False
            container: Any = [] if is_list else {}
            parent[key] = container
            stack.append((indent, container))
        elif value == "[]":
            parent[key] = []
        elif value == "{}":
            parent[key] = {}
        elif value.startswith("[") and value.endswith("]"):
            inner = value[1:-1].strip()
            parent[key] = [_scalar(p) for p in inner.split(",")] if inner else []
        else:
            parent[key] = _scalar(value)
    return root


def parse_yaml(text: str) -> dict[str, Any]:
    if _yaml is not None:
        loaded = _yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise config_error("CFG-105", "configuration root must be a mapping")
        return loaded
    return _mini_yaml(text)


def load_yaml(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if not target.is_file():
        raise config_error("CFG-100", f"configuration file not found: {target}",
                           recommended_action="check configs/ or pass --config")
    return parse_yaml(target.read_text(encoding="utf-8"))


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def get_path(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = config
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def repo_root() -> Path:
    """Locate the repository root from the installed package location."""
    env = os.environ.get("LEANLM_HOME")
    if env:
        return Path(env).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "configs").is_dir() and (parent / "src").is_dir():
            return parent
    return Path.cwd()


def config_dir() -> Path:
    return Path(os.environ.get("LEANLM_CONFIG_DIR", repo_root() / "configs"))


def workdir() -> Path:
    """Local, offline working directory for artifacts and the SQLite index."""
    path = Path(os.environ.get("LEANLM_WORKDIR", repo_root() / ".leanlm"))
    path.mkdir(parents=True, exist_ok=True)
    return path
