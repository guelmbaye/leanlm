"""LeanLM Runtime facade.

Wires the profile, the backend, the capability registry and the corpus into a
single object with a small surface: ``ingest``, ``ask``, ``status``, ``close``.

Everything the rest of the system needs to know about an execution is reachable
from the IEC it returns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

from ..contracts.capability import CapabilityRegistry
from ..contracts.ddp import DocumentPipelineState
from ..contracts.iec import InferenceExecutionContext, PipelineStage
from ..packages.context import ContextOptimizationCapability
from ..packages.context.policies import BudgetPolicy, OptimizationPolicy
from ..packages.inference import InferenceExecutionCapability, build_backend
from ..packages.inference.models import ModelBinding
from ..packages.inference.policies import GenerationPolicy, PromptPolicy
from ..packages.ingestion import DocumentIngestionCapability
from ..packages.ingestion.policies import IngestionPolicy
from ..packages.performance import PerformanceEngineeringCapability, ResourceSampler
from ..packages.performance.policies import PerformancePolicy
from ..packages.retrieval import EvidenceRetrievalCapability
from ..packages.retrieval.policies import RetrievalPolicy
from ..packages.understanding import DocumentUnderstandingCapability
from ..packages.understanding.policies import SegmentationPolicy
from ..packages.validation import ResponseValidationCapability
from ..packages.validation.policies import ValidationPolicy
from ..shared.clock import iso_now
from ..shared.errors import LeanLMError, input_error
from ..shared.events import EventBus, EventName
from ..shared.logging import get_logger
from ..shared.text import DEFAULT_TOKEN_COUNTER
from ..trust.integrity import enforce_model_binding, verify_model_binding, verify_runtime_integrity
from ..trust.offline_guard import OfflineGuard
from .corpus import CorpusStore
from .dic import verify as verify_dic
from .pipeline import InferencePipeline
from .profiles import RuntimeProfile, load_profile
from .session import Session, SessionManager
from .state_machine import RuntimeState, StateMachine


class LeanLMRuntime:
    """The single orchestrator. No business logic lives here."""

    def __init__(self, profile: RuntimeProfile | str = "development", *,
                 corpus: CorpusStore | None = None,
                 bus: EventBus | None = None,
                 model_path: str | None = None,
                 backend: str | None = None,
                 verify_model: bool = True,
                 enforce_offline: bool | None = None) -> None:
        self.profile = load_profile(profile) if isinstance(profile, str) else profile
        config = self.profile.as_config()
        self.bus = bus or EventBus()
        self.corpus = corpus or CorpusStore()
        self.sessions = SessionManager()
        self.machine = StateMachine()
        self.log = get_logger("runtime")

        self.performance_policy = PerformancePolicy.from_config(config)
        self.sampler = ResourceSampler(self.performance_policy)

        model_config = dict(self.profile.model)
        if model_path:
            model_config["path"] = model_path
        self.binding = ModelBinding(
            model_id=str(model_config.get("id", "unset")),
            path=str(model_config.get("path", "")),
            quantization=str(model_config.get("quantization", "unknown")),
            context_tokens=int(model_config.get("context_tokens", 4096)),
            expected_checksum=str(model_config.get("sha256", "")),
            parameters=str(model_config.get("parameters", "")),
            license=str(model_config.get("license", "")),
        )
        self.generation_policy = GenerationPolicy.from_config(config)
        self.backend = build_backend(backend or self.profile.backend, self.binding,
                                     self.generation_policy,
                                     **self._backend_options(config))
        self.trust_report = verify_model_binding(
            self.binding.path, self.binding.expected_checksum,
        ) if self.binding.path else None
        if verify_model and self.binding.path and not self.backend.is_simulated:
            enforce_model_binding(self.trust_report)

        offline = self.profile.telemetry.get("enforce_offline")
        self.enforce_offline = (enforce_offline if enforce_offline is not None
                                else bool(offline if offline is not None else True))

        self.token_counter = DEFAULT_TOKEN_COUNTER
        self.registry = CapabilityRegistry()
        self._build_registry(config)
        self.pipeline = InferencePipeline(
            self.registry, self.bus,
            runtime_handlers={
                PipelineStage.SESSION_CREATION: self._stage_session,
                PipelineStage.RESOURCE_ASSESSMENT: self._stage_resources,
                PipelineStage.DOCUMENT_DISCOVERY: self._stage_discovery,
                PipelineStage.SESSION_CLEANUP: self._stage_cleanup,
            },
            resource_probe=self.sampler.sample,
        )
        self._document_ids: tuple[str, ...] | None = None
        self._session: Session | None = None
        self._loaded = False

    # -- setup --------------------------------------------------------------
    def _backend_options(self, config: dict) -> dict[str, Any]:
        """Backend-specific kwargs. Only the in-process backend accepts these."""
        name = self.profile.backend
        if name != "llama-cpp-python":
            return {}
        memory = config.get("memory", {}) or {}
        return {
            "n_threads": memory.get("n_threads"),
            "n_gpu_layers": int(memory.get("n_gpu_layers", 0)),
            "n_batch": int(memory.get("n_batch", 256)),
            "use_mmap": bool(memory.get("use_mmap", True)),
            "use_mlock": bool(memory.get("use_mlock", False)),
        }

    def _build_registry(self, config: dict) -> None:
        self.registry.register(ContextOptimizationCapability(
            budget_policy=BudgetPolicy.from_config(config),
            optimization_policy=OptimizationPolicy.from_config(config),
            counter=self.token_counter,
            model_context_tokens=self.binding.context_tokens,
            max_output_tokens=self.generation_policy.max_output_tokens,
            model_footprint_mb=float((config.get("memory") or {}).get(
                "model_footprint_mb", 0.0)),
        ))
        self.registry.register(EvidenceRetrievalCapability(
            RetrievalPolicy.from_config(config), self.token_counter))
        self.registry.register(InferenceExecutionCapability(
            self.backend, prompt_policy=PromptPolicy.from_config(config),
            counter=self.token_counter, bus=self.bus))
        self.registry.register(ResponseValidationCapability(
            ValidationPolicy.from_config(config), bus=self.bus))
        self.registry.register(PerformanceEngineeringCapability(
            self.sampler, self.performance_policy, bus=self.bus))

    def load(self) -> None:
        """Load the model once, ahead of the first request (cold start cost)."""
        if self._loaded:
            return
        self.machine.transition(RuntimeState.LOADING)
        self.backend.load()
        counter = self.backend.token_counter()
        if counter is not DEFAULT_TOKEN_COUNTER:
            self.token_counter = counter
            self._rebuild_counters()
        self._loaded = True
        self.machine.transition(RuntimeState.READY)
        self.bus.emit(EventName.MODEL_LOADED, "-", capability="runtime",
                      backend=self.backend.name, model_id=self.binding.model_id)

    def _rebuild_counters(self) -> None:
        """Swap the heuristic counter for the model's real tokenizer."""
        context = self.registry.get("CAP-003")
        context.budgeter.counter = self.token_counter  # type: ignore[attr-defined]
        context.optimizer.counter = self.token_counter  # type: ignore[attr-defined]
        self.registry.get("CAP-004").selector.counter = self.token_counter  # type: ignore
        self.registry.get("CAP-005").builder.counter = self.token_counter  # type: ignore

    # -- corpus -------------------------------------------------------------
    def ingest(self, paths: Iterable[str | Path], *,
               replace: bool = False,
               on_document: Callable[[str, bool, str], None] | None = None
               ) -> dict[str, Any]:
        """Run the document pipeline (DDPB) over files or directories.

        ``replace`` empties the corpus first. Without it, ingesting a second
        corpus adds to the first, and a later accuracy run is measured against
        documents the evaluation set never accounted for -- a contaminated
        measurement that still reports a clean number.
        """
        config = self.profile.as_config()
        ingestion = DocumentIngestionCapability(IngestionPolicy.from_config(config))
        understanding = DocumentUnderstandingCapability(
            SegmentationPolicy.from_config(config), self.token_counter)
        files = _collect_files(paths)
        if not files:
            raise input_error(
                "COR-002", "no supported document found",
                capability="runtime", stage="import",
                recommended_action="supported formats: .pdf, .txt, .md",
            )
        if replace:
            self.corpus.clear()
        ingested, failed, units = [], [], 0
        for path in files:
            try:
                state = DocumentPipelineState(source_path=str(path))
                state = ingestion.process(state)
                state = understanding.process(state)
                document = state.structured
                self.corpus.upsert_document(
                    document, source_path=str(path),
                    file_checksum=state.descriptor.file_checksum, ingested_at=iso_now())
                units += len(document.units)
                ingested.append({"file": path.name, "units": len(document.units),
                                 "tokens": document.token_estimate,
                                 "document_id": document.document_id})
                self.bus.emit(EventName.DOCUMENT_LOADED, "-", capability="ingestion",
                              filename=path.name, units=len(document.units))
                if on_document:
                    on_document(str(path), True, f"{len(document.units)} units")
            except LeanLMError as error:
                failed.append({"file": path.name, "code": error.record.code,
                               "message": error.record.message,
                               "action": error.record.recommended_action})
                if on_document:
                    on_document(str(path), False, error.record.message)
        # Two different totals, kept apart. Reporting the run's unit count
        # beside the corpus-wide document count read as "8 documents, 18 units"
        # on a corpus that held 36 -- a line that looked like a fact and was an
        # arithmetic impossibility.
        return {"ingested": ingested, "failed": failed,
                "units_added": units,
                "documents_added": len(ingested),
                "corpus_documents": self.corpus.document_count(),
                "corpus_units": self.corpus.unit_count(),
                "replaced": replace,
                "corpus_checksum": self.corpus.corpus_checksum()}

    def use_documents(self, document_ids: Iterable[str] | None) -> None:
        self._document_ids = tuple(document_ids) if document_ids else None

    # -- inference ----------------------------------------------------------
    def ask(self, question: str, *, session: Session | None = None,
            check_contract: bool = True,
            stop_after: PipelineStage | None = None
            ) -> InferenceExecutionContext:
        if not question or not question.strip():
            raise input_error(
                "RT-001", "the question is empty", capability="runtime",
                stage="session_creation", recommended_action="type a question",
            )
        self.load()
        session = session or self._session or self.open_session()
        request_id = session.next_request_id()
        iec = InferenceExecutionContext(
            session_id=session.session_id, request_id=request_id,
            question=question.strip(), profile_id=self.profile.id,
        )
        self.sampler.reset()
        guard = OfflineGuard(enabled=self.enforce_offline, strict=True)
        try:
            with guard:
                iec = self.pipeline.run(iec, self.machine, stop_after=stop_after)
        finally:
            trace = dict(iec.trace)
            trace["offline"] = guard.report()
            trace["runtime"] = {
                "profile": self.profile.id,
                "profile_fingerprint": self.profile.fingerprint,
                "backend": self.backend.name,
                "model_id": self.binding.model_id,
                "token_counter": self.token_counter.name,
            }
            iec = iec.evolve(trace=trace)
        if iec.metrics is not None:
            self.corpus.record_metrics(request_id, session.session_id, iso_now(),
                                       iec.metrics.to_dict())
        if stop_after is not None:
            # Abandoned before inference: back to READY, with the model still
            # loaded. reset() would have claimed otherwise.
            if self.machine.state is not RuntimeState.READY:
                self.machine.transition(RuntimeState.READY)
            return iec
        if check_contract:
            violations = verify_dic(iec)
            if violations:
                iec = iec.evolve(trace={**iec.trace, "dic_violations": [
                    {"rule": v.rule, "detail": v.detail} for v in violations]})
                for violation in violations:
                    self.log.warning("DIC violation %s: %s", violation.rule, violation.detail)
        if self.machine.state in (RuntimeState.COMPLETED, RuntimeState.FAILED):
            self.machine.transition(RuntimeState.READY)
        return iec

    # -- sessions -----------------------------------------------------------
    def open_session(self) -> Session:
        session = self.sessions.create(self.profile.id)
        self.corpus.record_session(session.session_id, self.profile.id, session.created_at)
        self.bus.emit(EventName.SESSION_STARTED, session.session_id, capability="runtime",
                      profile=self.profile.id)
        self._session = session
        return session

    def close_session(self, session: Session | None = None) -> None:
        target = session or self._session
        if target is None:
            return
        self.sessions.close(target.session_id)
        self.corpus.close_session(target.session_id, iso_now(), target.request_count)
        self.bus.emit(EventName.SESSION_CLOSED, target.session_id, capability="runtime",
                      requests=target.request_count)
        if target is self._session:
            self._session = None

    # -- runtime stages -----------------------------------------------------
    def _stage_session(self, iec, stage):
        return iec.evolve(trace={**iec.trace, "session": {
            "profile": self.profile.id, "profile_version": self.profile.version,
            "fingerprint": self.profile.fingerprint,
        }})

    def _stage_resources(self, iec, stage):
        observation = self.sampler.sample()
        self.bus.emit(EventName.RESOURCES_ASSESSED, iec.session_id, capability="runtime",
                      available_ram_mb=observation.available_ram_mb,
                      temperature_c=observation.temperature_c)
        return iec.evolve(resources=observation)

    def _stage_discovery(self, iec, stage):
        package = self.corpus.context_package(self._document_ids)
        self.bus.emit(EventName.CONTEXT_LOADED, iec.session_id, capability="runtime",
                      units=package.total_units, tokens=package.total_tokens)
        return iec.evolve(context_package=package)

    def _stage_cleanup(self, iec, stage):
        """Temporary artifacts are not persisted by default (TL-08 / DM-04)."""
        return iec.evolve(trace={**iec.trace, "cleanup": {
            "temporary_artifacts_released": True,
            "iec_persisted": False,
        }})

    # -- introspection ------------------------------------------------------
    def status(self) -> dict[str, Any]:
        observation = self.sampler.sample()
        return {
            "product": "LeanLM",
            "profile": self.profile.to_dict(),
            "state": self.machine.state.value,
            "backend": self.backend.describe(),
            "model_loaded": self._loaded,
            "offline_enforced": self.enforce_offline,
            "corpus": {
                "documents": self.corpus.document_count(),
                "units": self.corpus.unit_count(),
                "checksum": self.corpus.corpus_checksum()[:16],
            },
            "capabilities": [spec.to_dict() for spec in self.registry.specs()],
            "resources": observation.to_dict(),
            "trust": {
                "model": self.trust_report.to_dict() if self.trust_report else None,
                "runtime": verify_runtime_integrity(self.profile.fingerprint).to_dict(),
            },
        }

    def close(self) -> None:
        self.close_session()
        self.sampler.stop()
        self.backend.unload()
        self.sessions.cleanup()
        self.corpus.close()

    def __enter__(self) -> "LeanLMRuntime":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


_SUPPORTED_SUFFIXES = (".pdf", ".txt", ".text", ".md", ".markdown")


def _collect_files(paths: Iterable[str | Path]) -> list[Path]:
    collected: list[Path] = []
    for entry in paths:
        path = Path(entry).expanduser()
        if path.is_dir():
            collected.extend(sorted(
                p for p in path.rglob("*")
                if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES
            ))
        elif path.is_file():
            collected.append(path)
    seen, unique = set(), []
    for path in collected:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    return unique
