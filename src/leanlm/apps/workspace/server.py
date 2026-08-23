"""Local workspace server (`leanlm serve`).

`http.server` from the standard library, bound to loopback. Two deliberate
constraints:

* **No web framework.** Adding Flask or FastAPI would put a hard dependency in
  the install path of a tool whose selling point is that it runs on a machine
  with nothing installed. The whole API is five endpoints.
* **Loopback only, and it refuses otherwise.** Binding to `0.0.0.0` would turn
  an offline tool into a network service on someone's café Wi-Fi. TL-01 is not
  a suggestion, so the server declines rather than warns.

The runtime is created once and reused across requests, guarded by a lock: the
model load is the expensive part and a second copy would not fit in 8 GB.
"""
from __future__ import annotations

import json
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from ...runtime import LeanLMRuntime
from ...runtime.dic import verify as verify_dic
from ...shared.errors import LeanLMError
from ...shared.serialization import to_plain
from .assets import PAGE

SUGGESTED_QUESTIONS = (
    "Quel est le delai de remboursement des notes de frais ?",
    "Quel est le plafond journalier pour les repas ?",
    "Qui valide une demande de teletravail ?",
    "Quelles depenses ne sont pas remboursables ?",
)

_LOOPBACK = ("127.0.0.1", "::1", "localhost")


class _State:
    """One runtime, one lock. Requests queue rather than duplicate the model."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.lock = threading.Lock()
        self.runtime: LeanLMRuntime | None = None

    def get(self) -> LeanLMRuntime:
        if self.runtime is None:
            self.runtime = LeanLMRuntime(**self.kwargs)
        return self.runtime

    def close(self) -> None:
        if self.runtime is not None:
            self.runtime.close()
            self.runtime = None


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(to_plain(payload), ensure_ascii=False).encode("utf-8")


def _handler(state: _State) -> type[BaseHTTPRequestHandler]:

    class Handler(BaseHTTPRequestHandler):
        server_version = "LeanLM"
        sys_version = ""

        # -- plumbing -------------------------------------------------------
        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return  # the CLI prints what matters; access logs are noise here

        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            # No external origin can reach a loopback-only server, but stating
            # the policy costs nothing and documents the intent.
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; style-src 'unsafe-inline'; "
                             "script-src 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

        def _body(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if not length:
                return {}
            try:
                return json.loads(self.rfile.read(length).decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                return {}

        # -- routes ---------------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(self._status())
            elif path == "/api/corpus":
                runtime = state.get()
                self._json({"documents": runtime.corpus.documents(),
                            "units": runtime.corpus.unit_count(),
                            "checksum": runtime.corpus.corpus_checksum()})
            elif path == "/api/health":
                self._json({"ok": True})
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            if path == "/api/ask":
                self._json(self._ask(self._body()))
            elif path == "/api/ingest":
                self._json(self._ingest(self._body()))
            else:
                self._json({"error": "not found"}, 404)

        # -- handlers -------------------------------------------------------
        def _status(self) -> dict[str, Any]:
            with state.lock:
                runtime = state.get()
                payload = runtime.status()
                payload["documents"] = runtime.corpus.documents()
            payload["suggested_questions"] = list(SUGGESTED_QUESTIONS)
            return payload

        def _ask(self, body: dict[str, Any]) -> dict[str, Any]:
            question = str(body.get("question") or "").strip()
            if not question:
                return {"error": {"code": "WS-001", "message": "empty question",
                                  "recommended_action": "type a question"}}
            with state.lock:
                runtime = state.get()
                try:
                    iec = runtime.ask(question)
                except LeanLMError as error:
                    return {"error": error.record.to_dict()}
                violations = verify_dic(iec)
            return _view(iec, violations)

        def _ingest(self, body: dict[str, Any]) -> dict[str, Any]:
            paths = body.get("paths") or ([body["path"]] if body.get("path") else [])
            if not paths:
                return {"error": {"code": "WS-002", "message": "no path given"}}
            with state.lock:
                runtime = state.get()
                try:
                    return runtime.ingest(paths)
                except LeanLMError as error:
                    return {"error": error.record.to_dict()}

    return Handler


def _view(iec: Any, violations: Any) -> dict[str, Any]:
    """The shape the page consumes. Nothing computed here -- projection only."""
    response, validation, metrics = iec.response, iec.validation, iec.metrics
    optimized = iec.optimized_context
    budget = optimized.budget if optimized is not None else None
    return {
        "request_id": iec.request_id,
        "answer": response.text if response is not None else "",
        "simulated": bool(response.is_simulated) if response is not None else False,
        "error": iec.errors[-1].to_dict() if iec.errors else None,
        "evidence": [
            {"label": item.label, "document_name": item.document_name,
             "section_path": list(item.section_path), "score": item.score,
             "excerpt": item.text[:320]}
            for item in iec.evidence
        ],
        "validation": validation.to_dict() if validation is not None else None,
        "metrics": metrics.to_flat() if metrics is not None else {},
        "stages": [record.to_dict() for record in iec.stages],
        "budget": _budget_view(budget, optimized, iec),
        "dic_violations": [{"rule": v.rule, "detail": v.detail} for v in violations],
        "iec": iec.to_dict(include_text=False),
    }


def _budget_view(budget: Any, optimized: Any, iec: Any) -> dict[str, Any]:
    if budget is None:
        return {}
    intent = iec.intent
    view = {
        "Intention detectee": (f"{intent.intent.value} (confiance {intent.confidence})"
                               if intent is not None else "-"),
        "Fenetre du modele": f"{budget.max_context_tokens} tokens",
        "Reserve pour la sortie": f"{budget.reserved_output_tokens} tokens",
        "Surcout de prompt": f"{budget.reserved_prompt_overhead} tokens",
        "Budget de preuves": f"{budget.evidence_tokens} tokens",
        "Passages maximum": str(budget.max_passages),
        "Facteur limitant": budget.limiting_factor,
        "Compteur de tokens": budget.token_counter,
    }
    for name, value in sorted((budget.factors or {}).items()):
        view[f"Facteur {name}"] = f"{value}"
    if optimized is not None:
        view["Unites conservees"] = str(len(optimized.units))
        view["Unites ecartees"] = str(optimized.dropped_units)
        view["Doublons supprimes"] = str(optimized.duplicate_units)
        view["Unites fusionnees"] = str(optimized.merged_units)
        view["Tokens avant / apres"] = (f"{optimized.input_tokens} -> "
                                        f"{optimized.output_tokens}")
        view["Taux de compression"] = str(optimized.compression_ratio)
        view["Operations"] = ", ".join(optimized.operations) or "-"
    return view


def serve(*, profile: str = "development", host: str = "127.0.0.1", port: int = 8770,
          backend: str | None = None, model_path: str | None = None,
          open_browser: bool = False) -> int:
    if host not in _LOOPBACK:
        print(f"  [!!] refusing to bind {host}: the workspace is loopback-only "
              "(TL-01, no network exposure)")
        return 2

    state = _State(profile=profile, backend=backend, model_path=model_path)
    server = ThreadingHTTPServer((host, port), _handler(state))
    url = f"http://{host}:{port}/"
    print(f"  LeanLM workspace on {url}")
    print(f"  profile {profile} -- everything stays on this machine")
    print("  Ctrl-C to stop")
    if open_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopping")
    finally:
        server.server_close()
        state.close()
    return 0
