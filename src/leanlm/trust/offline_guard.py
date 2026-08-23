"""Offline integrity guard (TL-01, DIC-06, NFR-01).

"No network during inference" is easy to write in a document and easy to break
in code. This guard makes it an enforced property: while it is active, any
attempt to open a non-loopback socket raises and is recorded in the IEC.

Loopback stays allowed because ``llama-server`` is a legitimate local backend.
Everything else is a breach, including DNS.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from typing import Any

_LOOPBACK_PREFIXES = ("127.", "::1", "localhost", "0.0.0.0")


@dataclass(frozen=True, slots=True)
class OfflineViolation:
    host: str
    port: int | str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"host": self.host, "port": self.port, "detail": self.detail}


class OfflineBreach(RuntimeError):
    pass


def _is_loopback(host: str) -> bool:
    text = str(host)
    return any(text.startswith(prefix) or text == prefix for prefix in _LOOPBACK_PREFIXES)


@dataclass
class OfflineGuard:
    """Context manager that blocks outbound network access."""

    enabled: bool = True
    strict: bool = True
    violations: list[OfflineViolation] = field(default_factory=list)
    _original_connect: Any = None
    _original_getaddrinfo: Any = None

    def __enter__(self) -> "OfflineGuard":
        if not self.enabled:
            return self
        guard = self

        self._original_connect = socket.socket.connect
        self._original_getaddrinfo = socket.getaddrinfo

        def guarded_connect(sock, address, *args, **kwargs):
            host = address[0] if isinstance(address, tuple) else address
            if not _is_loopback(host):
                violation = OfflineViolation(str(host),
                                             address[1] if isinstance(address, tuple) else "-",
                                             "socket.connect blocked")
                guard.violations.append(violation)
                if guard.strict:
                    raise OfflineBreach(
                        f"offline runtime refused a connection to {host} "
                        "(TL-01: no network during inference)"
                    )
                return None
            return guard._original_connect(sock, address, *args, **kwargs)

        def guarded_getaddrinfo(host, port, *args, **kwargs):
            if not _is_loopback(host):
                guard.violations.append(
                    OfflineViolation(str(host), port, "DNS resolution blocked"))
                if guard.strict:
                    raise OfflineBreach(
                        f"offline runtime refused to resolve {host} (TL-01)")
            return guard._original_getaddrinfo(host, port, *args, **kwargs)

        socket.socket.connect = guarded_connect  # type: ignore[assignment]
        socket.getaddrinfo = guarded_getaddrinfo  # type: ignore[assignment]
        return self

    def __exit__(self, *exc) -> None:
        if self._original_connect is not None:
            socket.socket.connect = self._original_connect  # type: ignore[assignment]
            self._original_connect = None
        if self._original_getaddrinfo is not None:
            socket.getaddrinfo = self._original_getaddrinfo  # type: ignore[assignment]
            self._original_getaddrinfo = None

    def report(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "strict": self.strict,
            "violations": [v.to_dict() for v in self.violations],
        }
