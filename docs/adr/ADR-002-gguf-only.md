# ADR-002 — GGUF as the only accepted model format

**Status:** accepted · **Date:** 2026-06-01 · **Requirement:** NFR-01, NFR-03

## Context
Supporting several model formats sounds generous and is, in practice, a way to
ship four half-tested loading paths. The competition needs one that works on the
target hardware and can be verified.

## Decision
GGUF only. The runtime refuses anything else, and `trust/integrity.py` checks the
four magic bytes `GGUF` before a file is ever benchmarked.

## Why the magic-byte check earns its place
The most common failure when fetching a model is not a corrupt download: it is a
redirect to an HTML error page saved under a `.gguf` name. Checking the header
turns a confusing runtime crash deep inside the loader into `download_model.sh`
exiting with a clear message and deleting the file.

## Consequences
- One loading path, tested.
- Quantization becomes a recorded property of the run (`Q4_K_M` and so on) rather
  than an implicit detail, which is required for a measurement to be comparable.
- Converting a model to GGUF is the user's problem, and the tooling for that is
  mature and outside our scope.
