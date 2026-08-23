"""Session manager (Runtime Blueprint s.8)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..shared.clock import iso_now
from ..shared.ids import request_id, session_id


@dataclass
class Session:
    session_id: str
    profile_id: str
    created_at: str
    request_count: int = 0
    history: list[str] = field(default_factory=list)
    closed_at: str | None = None

    def next_request_id(self) -> str:
        self.request_count += 1
        rid = request_id(self.session_id, self.request_count)
        self.history.append(rid)
        return rid


class SessionManager:
    """Creates sessions, hands out request ids, cleans up on close."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self, profile_id: str) -> Session:
        session = Session(session_id=session_id(), profile_id=profile_id,
                          created_at=iso_now())
        self._sessions[session.session_id] = session
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def close(self, sid: str) -> Session | None:
        session = self._sessions.get(sid)
        if session is not None:
            session.closed_at = iso_now()
        return session

    def active(self) -> list[Session]:
        return [s for s in self._sessions.values() if s.closed_at is None]

    def cleanup(self) -> int:
        """Drop closed sessions. Temporary state is not persisted by default."""
        closed = [sid for sid, s in self._sessions.items() if s.closed_at is not None]
        for sid in closed:
            del self._sessions[sid]
        return len(closed)
