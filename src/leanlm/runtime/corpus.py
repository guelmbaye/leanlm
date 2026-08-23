"""Local corpus store (DMSB s.5, s.6).

SQLite for metadata and the unit index, the filesystem for artifacts. Both are
in the workspace directory, both are offline, and neither requires a server.
The original documents are never copied or modified -- only referenced.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable, Iterator

from ..contracts.dto import ContextPackage, SemanticUnit, StructuredDocument, UnitKind
from ..shared.config import workdir
from ..shared.errors import input_error
from ..shared.hashing import sha256_text
from ..shared.serialization import canonical_json

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   TEXT PRIMARY KEY,
    artifact_id   TEXT NOT NULL,
    filename      TEXT NOT NULL,
    source_path   TEXT NOT NULL,
    title         TEXT,
    file_checksum TEXT NOT NULL,
    language      TEXT,
    token_estimate INTEGER NOT NULL,
    unit_count    INTEGER NOT NULL,
    ingested_at   TEXT NOT NULL,
    payload       TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS units (
    unit_id       TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    document_name TEXT NOT NULL,
    kind          TEXT NOT NULL,
    ordinal       INTEGER NOT NULL,
    section_path  TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    char_start    INTEGER NOT NULL,
    char_end      INTEGER NOT NULL,
    heading_level INTEGER NOT NULL,
    artifact_id   TEXT NOT NULL,
    checksum      TEXT NOT NULL,
    text          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_units_document ON units(document_id, ordinal);
CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    profile_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    closed_at   TEXT,
    requests    INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS metrics (
    request_id  TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL
);
"""


class CorpusStore:
    """The corpus the runtime discovers documents from (DIC phase 3)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else (workdir() / "corpus.sqlite3")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(str(self.path), check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(_SCHEMA)
        self._connection.commit()
        # Caches, invalidated on every write. Profiling showed document_discovery
        # taking 38% of a request's time: the corpus was read from SQLite and every
        # unit re-sealed on each question, even though the corpus had not changed.
        # See docs/ENGINEERING-DECISIONS.md (EDB-019) for the measurement.
        self._package_cache: dict[tuple[str, ...] | None, ContextPackage] = {}
        self._checksum_cache: str | None = None

    def _invalidate(self) -> None:
        """Any write makes both caches wrong. Correctness first, speed second."""
        self._package_cache.clear()
        self._checksum_cache = None

    # -- writes -------------------------------------------------------------
    def upsert_document(self, document: StructuredDocument, *, source_path: str,
                        file_checksum: str, ingested_at: str) -> None:
        self._invalidate()
        with self._connection:
            self._connection.execute("DELETE FROM documents WHERE document_id = ?",
                                     (document.document_id,))
            self._connection.execute(
                "INSERT INTO documents (document_id, artifact_id, filename, source_path,"
                " title, file_checksum, language, token_estimate, unit_count, ingested_at,"
                " payload) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (document.document_id, document.artifact_id, document.filename,
                 source_path, document.title, file_checksum, document.language_hint,
                 document.token_estimate, len(document.units), ingested_at,
                 canonical_json({"section_titles": list(document.section_titles),
                                 "meta": document.meta.to_dict()})),
            )
            self._connection.executemany(
                "INSERT INTO units (unit_id, document_id, document_name, kind, ordinal,"
                " section_path, token_estimate, char_start, char_end, heading_level,"
                " artifact_id, checksum, text) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [(u.unit_id, u.document_id, u.document_name, u.kind.value, u.order,
                  "\x1f".join(u.section_path), u.token_estimate, u.char_start,
                  u.char_end, u.heading_level, u.artifact_id, u.meta.checksum, u.text)
                 for u in document.units],
            )

    def remove_document(self, document_id: str) -> bool:
        self._invalidate()
        with self._connection:
            cursor = self._connection.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,))
        return cursor.rowcount > 0

    def clear(self) -> None:
        self._invalidate()
        with self._connection:
            self._connection.execute("DELETE FROM units")
            self._connection.execute("DELETE FROM documents")

    def record_session(self, session_id: str, profile_id: str, created_at: str) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO sessions (session_id, profile_id, created_at)"
                " VALUES (?,?,?)", (session_id, profile_id, created_at))

    def close_session(self, session_id: str, closed_at: str, requests: int) -> None:
        with self._connection:
            self._connection.execute(
                "UPDATE sessions SET closed_at = ?, requests = ? WHERE session_id = ?",
                (closed_at, requests, session_id))

    def record_metrics(self, request_id: str, session_id: str, created_at: str,
                       payload: dict) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT OR REPLACE INTO metrics (request_id, session_id, created_at,"
                " payload) VALUES (?,?,?,?)",
                (request_id, session_id, created_at, canonical_json(payload)))

    # -- reads --------------------------------------------------------------
    def documents(self) -> list[dict]:
        rows = self._connection.execute(
            "SELECT document_id, filename, title, language, token_estimate, unit_count,"
            " ingested_at, source_path, file_checksum FROM documents ORDER BY filename"
        ).fetchall()
        return [dict(row) for row in rows]

    def document_count(self) -> int:
        return int(self._connection.execute(
            "SELECT COUNT(*) FROM documents").fetchone()[0])

    def unit_count(self) -> int:
        return int(self._connection.execute("SELECT COUNT(*) FROM units").fetchone()[0])

    def iter_units(self, document_ids: Iterable[str] | None = None) -> Iterator[SemanticUnit]:
        if document_ids:
            placeholders = ",".join("?" for _ in document_ids)
            query = (f"SELECT * FROM units WHERE document_id IN ({placeholders})"
                     " ORDER BY document_id, ordinal")
            rows = self._connection.execute(query, tuple(document_ids))
        else:
            rows = self._connection.execute(
                "SELECT * FROM units ORDER BY document_id, ordinal")
        for row in rows:
            yield self._to_unit(row)

    @staticmethod
    def _to_unit(row: sqlite3.Row) -> SemanticUnit:
        section_path = tuple(p for p in (row["section_path"] or "").split("\x1f") if p)
        unit = SemanticUnit(
            meta=SemanticUnit.build_meta(
                identity=(row["document_id"], str(row["ordinal"]), row["text"][:120]),
                parent_id=row["document_id"], source=row["document_name"],
                stage="document_discovery",
            ),
            unit_id=row["unit_id"], document_id=row["document_id"],
            document_name=row["document_name"], kind=UnitKind(row["kind"]),
            text=row["text"], section_path=section_path, order=int(row["ordinal"]),
            char_start=int(row["char_start"]), char_end=int(row["char_end"]),
            token_estimate=int(row["token_estimate"]),
            heading_level=int(row["heading_level"]),
        )
        return unit.sealed()  # type: ignore[return-value]

    def context_package(self, document_ids: Iterable[str] | None = None) -> ContextPackage:
        """DIC phase 3: hand the inference pipeline a ready corpus snapshot.

        Cached per document selection. The artifact is immutable and its id is
        derived from its content, so a cached package and a freshly built one are
        indistinguishable -- which is what makes caching safe here rather than a
        source of stale reads.
        """
        key = tuple(sorted(document_ids)) if document_ids else None
        cached = self._package_cache.get(key)
        if cached is not None:
            return cached
        units = tuple(self.iter_units(document_ids))
        if not units:
            raise input_error(
                "COR-001", "the corpus is empty",
                capability="runtime", stage="document_discovery",
                recommended_action="run `leanlm ingest <path>` first",
            )
        checksum = sha256_text("|".join(u.meta.checksum for u in units))
        package = ContextPackage(
            meta=ContextPackage.build_meta(identity=(checksum, str(len(units))),
                                           stage="document_discovery"),
            units=units,
            document_ids=tuple(sorted({u.document_id for u in units})),
            total_units=len(units),
            total_tokens=sum(u.token_estimate for u in units),
            corpus_checksum=checksum,
        )
        sealed = package.sealed()
        self._package_cache[key] = sealed  # type: ignore[assignment]
        return sealed  # type: ignore[return-value]

    def corpus_checksum(self) -> str:
        if self._checksum_cache is not None:
            return self._checksum_cache
        rows = self._connection.execute(
            "SELECT checksum FROM units ORDER BY document_id, ordinal").fetchall()
        self._checksum_cache = (sha256_text("|".join(row[0] for row in rows))
                                if rows else "")
        return self._checksum_cache

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "CorpusStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
