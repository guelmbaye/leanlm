"""Structured logging (MIB s.15).

Every record: timestamp, level, capability, session id, message, fields.
User document content is never logged by default (TL-03 / TILAB s.7).
"""
from __future__ import annotations

import logging
import sys
from typing import Any

from .serialization import canonical_json

_CONFIGURED = False
LOGGER_NAME = "leanlm"


class _StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%SZ"),
            "level": record.levelname.lower(),
            "capability": getattr(record, "capability", "runtime"),
            "session_id": getattr(record, "session_id", "-"),
            "message": record.getMessage(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload["fields"] = extra
        if record.exc_info:
            payload["error"] = self.formatException(record.exc_info).splitlines()[-1]
        return canonical_json(payload)


class _HumanFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        cap = getattr(record, "capability", "runtime")
        return f"{record.levelname[:4].lower():>5} [{cap}] {record.getMessage()}"


def configure(level: str = "info", *, structured: bool = True, stream: Any = None) -> None:
    global _CONFIGURED
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(_StructuredFormatter() if structured else _HumanFormatter())
    logger.addHandler(handler)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    _CONFIGURED = True


def get_logger(capability: str = "runtime", session_id: str = "-") -> logging.LoggerAdapter:
    if not _CONFIGURED:
        configure()
    base = logging.getLogger(LOGGER_NAME)
    return logging.LoggerAdapter(base, {"capability": capability, "session_id": session_id})
