# ADR-003 — Capability-Oriented Architecture

**Status:** accepted · **Date:** 2026-06-02 · **Requirement:** all FR

## Context
A layered architecture (api / services / models) would have put the interesting
decisions — budgeting, retrieval, grounding — in one undifferentiated
"services" folder. The competition judges an *innovation*, and an architecture
that hides where the innovation lives makes it harder to defend and harder to
measure.

## Decision
Eight capabilities, each owning one or more consecutive stages of exactly one
pipeline, each in its own package with the same seven modules:

```
contracts.py   what the capability promises (its CapabilitySpec)
models.py      shapes it works with internally
policies.py    every tunable parameter, in one place
services.py    the algorithms
telemetry.py   what it measures about itself
benchmarks.py  how it proves its own performance
README.md      why it exists and what it refuses to do
```

## Consequences
- Every stage has exactly one owner; two owners is an error the registry raises
  (ICIB-007) and the CCM verifier fails the build on.
- The uniform layout is machine-checkable: `leanlm ccm` walks capability →
  package → contract → tests → benchmark → documentation and fails CI on a gap.
  This ADR is enforced, not merely written down.
- Cost: eight `policies.py` files instead of one settings module. Accepted,
  because a parameter that is easy to find is a parameter that gets measured.

## Reversibility
Capabilities communicate only through canonical DTOs, so merging two of them is
a mechanical refactor.
