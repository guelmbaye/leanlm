# packages/inference -- CAP-005 Inference Execution

| Field | Value |
|---|---|
| Capability | CAP-005 Inference Execution |
| Requirement | FR-05 |
| Stages | `prompt_assembly`, `model_inference` (DIC phases 6-7) |
| Input | IEC with `evidence` |
| Output | IEC with `prompt` and `response` |
| Tests | `tests/capability/test_inference.py` |
| Benchmark | `packages/inference/benchmarks.py` |

## Backends

| Backend | When | Notes |
|---|---|---|
| `llama-cpp-python` | preferred | in-process, streaming, exact tokenizer, real first-token latency |
| `llama-server` | a local server is already running | loopback only -- a non-loopback host is rejected at construction |
| `llama-cli` | only a llama.cpp build is available | timings parsed from stderr |
| `simulated` | no GGUF present | quotes the top passage; **never** packaged into a submission |

`auto` picks the first that can actually run.

## The simulated backend is not a fallback in disguise

Every artifact it produces carries `is_simulated=true`, the IEC gets an explicit
warning, the workspace labels it, and `submission-builder` refuses to package a
run that used it. A measurement that did not come from a model must never be
able to pass for one -- that is the whole point of TL-07 and of CP-05.

## Prompt contract

Three parts, fixed order: system instructions, excerpts, question. The system
prompt is deliberately short: it is paid on every request. `prompt_efficiency`
(evidence tokens / prompt tokens) is tracked so scaffolding creep is visible.
