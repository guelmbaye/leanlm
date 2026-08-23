# ADR-008 — No mandatory third-party dependency

**Status:** accepted · **Date:** 2026-06-05 · **Requirement:** NFR-01, NFR-07

## Context
The product's promise is that it works on a laptop with no internet. A judge
may evaluate it on a machine where `pip install` fails, where the corporate
proxy blocks PyPI, or where the Python environment is externally managed.

## Decision
`dependencies = []`. Every third-party library is optional, and each has a
stdlib fallback:

| Library | Used for | Fallback |
|---|---|---|
| PyYAML | configuration | `shared/config.py::_mini_yaml` |
| psutil | resource sampling | `/proc/meminfo`, `/proc/self/status`, `/sys/class/thermal` |
| pypdf / pdfminer.six | PDF extraction | `pdftotext`, then explicit refusal (ING-002) |
| llama-cpp-python | inference | `llama-server` over loopback, `llama-cli`, or the labelled simulator |

## Consequences
- We maintain a small YAML subset parser. It is tested (`tests/unit`) and
  deliberately limited to the shapes our own configs use.
- `leanlm doctor` reports, per library, whether the real implementation or the
  fallback is active — so a measurement can never be attributed to the wrong one.
- The fallbacks are never silent: a run using the PDF fallback records the
  extraction method in the artifact.

## What this does not license
Falling back to the *simulator* is not an acceptable degradation. It is labelled
in the IEC, in the UI, and refused by the submission gate (SUB-006).
