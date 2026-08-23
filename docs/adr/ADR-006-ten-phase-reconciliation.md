# ADR-006 — Reconciling the pipeline into ten phases

**Status:** accepted · **Date:** 2026-06-04 · **Requirement:** DIC-01

## Context
The Runtime Blueprint described nine phases; the Inference Pipeline Blueprint
described a sequence that split session handling from resource assessment. Two
documents, two lifecycles, and a Deterministic Inference Contract that claims a
"fixed order" — the contradiction had to be resolved before the order could be
verified by a machine.

## Decision
Ten phases, in `contracts/iec.py::PipelineStage`:

1. `session_creation` · 2. `resource_assessment` · 3. `document_discovery` ·
4. `context_optimization` · 5. `evidence_retrieval` · 6. `prompt_assembly` ·
7. `model_inference` · 8. `response_validation` · 9. `metrics_finalization` ·
10. `session_cleanup`

Four of them (1, 2, 3, 10) belong to the runtime and to no business capability:
they are infrastructure, and pretending otherwise would have created a
capability with nothing to innovate about.

## Consequences
- `STAGE_ORDER` is the single source of truth. The orchestrator iterates over
  it; `runtime/dic.py` asserts against it; the workspace's trace rail renders it.
  Adding a phase means editing one tuple.
- The trace rail in the workspace displays all ten with their real durations,
  which makes the "the model is one segment among ten" argument visible rather
  than asserted.
