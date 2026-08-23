# ADR-004 — The pipeline order lives in one tuple

**Status:** accepted · **Date:** 2026-06-03 · **Requirement:** DIC-01, NFR-03

## Context
"Deterministic pipeline" is the kind of claim that is true when written and false
six weeks later, because a stage got inserted in the orchestrator, another in a
capability, and the diagram was never updated.

## Decision
`contracts/iec.py::STAGE_ORDER` is the single declaration of the order. Four
consumers read it and none of them may redefine it:

| Consumer | Uses it to |
|---|---|
| `runtime/pipeline.py` | iterate the stages |
| `runtime/dic.py` | assert the executed order is a subsequence |
| `apps/ccm.py` | check each stage has exactly one owner |
| the workspace trace rail | render the ten phases with real durations |

The orchestrator holds no business logic at all: it knows the order, the
telemetry, the state machine and the error contract, and nothing about what a
context budget is.

## Consequences
- Adding a phase means editing one tuple; forgetting to update a consumer is
  impossible because none of them holds its own copy.
- A stage that runs out of order is caught by `verify()` after every request in
  development and benchmark profiles, and by the test suite in CI.
- The determinism claim is checkable by a reader in about a minute, which is the
  actual objective.
