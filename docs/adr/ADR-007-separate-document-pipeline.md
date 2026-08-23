# ADR-007 — The document pipeline is separate from the inference pipeline

**Status:** accepted · **Date:** 2026-06-04 · **Requirement:** FR-01, FR-02, NFR-02

## Context
Ingestion and understanding (CAP-001, CAP-002) could have been stages 0 and 1 of
the inference pipeline. They are not.

## Decision
Two pipelines with different lifecycles:

```
document pipeline   file -> validated -> normalized -> structured -> segmented -> exported
                    (runs on ingestion, once per document)

inference pipeline  ten DIC phases
                    (runs per question, reading the corpus the first pipeline built)
```

`DocumentPipelineState` advances through its own states and refuses to skip one.

## Why
1. **Cost.** Segmenting a 200-page PDF takes tens of milliseconds per document.
   Doing that inside every question would put ingestion cost on the latency path
   for no benefit — the document has not changed since the last question.
2. **Different failure semantics.** An unreadable PDF should be reported as a
   failed *file* while the other documents are ingested. An unreadable corpus
   during inference is a different situation with a different response.
3. **Honest measurement.** If ingestion were a phase of the inference pipeline,
   first-question latency would include it and every published latency figure
   would depend on whether the corpus happened to be warm.

## Consequences
- Two capability pipelines in the registry, `document` and `inference`, plus
  `delivery` for packaging. A capability never spans two, and the contract test
  asserts it.
- The corpus store is the boundary between them, which is why it is also the
  thing worth caching (EDB-019).
