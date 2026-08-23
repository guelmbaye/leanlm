# ADR-001 — llama.cpp as the inference runtime

**Status:** accepted · **Date:** 2026-06-01 · **Requirement:** NFR-01, NFR-03

## Context
The target is a laptop with roughly 8 GB of RAM, frequently without a discrete
GPU, and the competition requires fully offline operation. The runtime has to
load a quantized model, run on CPU, and not depend on a Python wheel that only
ships for two platforms.

## Options considered
| Option | Memory on target | Offline | Notes |
|---|---|---|---|
| llama.cpp / GGUF | ~2.6 GB for a 3.8B Q4_K_M | yes | C++ core, mmap, wide platform support |
| Transformers + bitsandbytes | 5–7 GB | yes | heavy dependency tree, CUDA-oriented |
| ONNX Runtime | ~3 GB | yes | conversion step, weaker quantization ecosystem |
| Ollama | ~2.6 GB | yes | wraps llama.cpp, adds a daemon and a model store we do not control |

## Decision
llama.cpp, with GGUF as the only accepted format. LeanLM binds to it through
four interchangeable backends (in-process `llama-cpp-python`, local
`llama-server` over loopback, the `llama-cli` binary, and a clearly labelled
simulator for development).

## Consequences
- Memory behaviour is predictable and mmap-friendly, which is what makes the
  8 GB target realistic.
- We inherit llama.cpp's threading model; `n_threads` becomes a profile
  parameter that must be recorded with every measurement.
- No backend may reach the network except over loopback (TL-01).

## Reversibility
The backend interface is four methods. Replacing the runtime means writing one
new class; nothing above `packages/inference/backends.py` knows llama.cpp
exists.
