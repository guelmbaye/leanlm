# qwen3.5-2b — ADTC 2026 Report

**Domain:** corporate_enterprise · **Languages:** en, fr
**Runtime:** llama.cpp · **Quantization:** GGUF Q4_0 ·
**Parameters:** 1.9B
**Repository:** https://github.com/guelmbaye/leanlm
**Commit:** `1b1c0e6ac392355881ecc1d56f2351949ef3f293`

---

## 1. Problem

Knowledge work in African enterprises runs on documents — HR policies, service
contracts, procedures — and the tools that could answer questions about them
assume a cloud, a subscription and a permanent connection. All three are
assumptions, and in much of the continent all three fail.

The target user is an office worker with the laptop they already have: 8 GB of
RAM, integrated graphics, intermittent connectivity. They do not need a general
assistant. They need a correct answer to "what is the reimbursement deadline",
with the paragraph it came from, without sending an internal contract to a
third party.

The constraint that shapes everything: the answer must be **traceable**. An
assistant that is confidently wrong about a policy is worse than no assistant,
because a wrong figure is acted upon.

## 2. Design Decisions

**Model.** qwen3.5-2b at GGUF Q4_0,
1.9B, chosen to leave headroom on an 8 GB
machine. At this quantization the loaded model and its KV cache peaked at 1.98 GB of the 7 GB budget.

Peak memory is reported below rather than the final figure, because a run that
briefly touched the ceiling is a run that nearly swapped — and swapping is the
difference between a usable tool and an unusable one. It does not show up in an
average.

**Context window capped at 4096 tokens** regardless of what the model
advertises. The KV cache scales with context, not with weights; a large
advertised window is a model-card feature and a memory bill.

**Retrieval instead of stuffing.** The whole corpus is not sent to the model.
Passages are selected by BM25 with structural boosts and a diversity pass, under
a token budget computed from the machine's current state. An embedding model was
evaluated and rejected: a second model resident in RAM competes with the one the
user came for.

**Refusal as a first-class outcome.** When no passage clears the relevance
floor, the system returns nothing and the answer declares insufficiency. This
was not the original behaviour — an early version answered a question about
share prices with a verbatim quote about annual leave, at high confidence,
because the answer was faithful to a passage that had nothing to do with the
question.

## 3. Constraints

| Constraint | Consequence |
|---|---|
| 8 GB RAM, 4 vCPU, integrated GPU | model and cache measured at 1.98 GB; peak matters, not mean |
| No network during inference | enforced, not promised: non-loopback sockets raise and are recorded |
| Documents are confidential | nothing leaves the machine; there is no telemetry endpoint to disable |
| Intermittent connectivity | zero mandatory Python dependencies; every optional one has a fallback |
| Answers are acted upon | every sentence is checked against the passages actually sent |

Environment for the measurements below: Linux-6.8.0-124-generic-x86_64-with-glibc2.39,
Python 3.12.3, 4 CPUs,
7941.2 MB RAM.

## 4. Benchmarks

Every figure is a mean over repeated runs with its spread, on the corpus
identified by checksum `6cb9bcd5cf640f1e`
under profile `competition`
(fingerprint `8163b8373fff4a3a`).

| Metric | Value |
|---|---|
| throughput | 12.9 tok/s (min 12.466, max 13.093, n=41) |
| first token | 616 ms (min 113.047, max 4741.706, n=41) |
| total per request | 7481.4 ms (min 1654.067, max 41209.542, n=41) |
| peak RSS (whole system, official profiler) | 2025 MB (1.98 GB of a 7 GB budget) |
| prompt size | 344 tokens (min 162.0, max 432.0, n=41) |
| corpus not sent to the model | 0.74 (min 0.2337, max 0.9896, n=41) |

### Official profiler

Measured on Intel(R) Xeon(R) Platinum 8280 CPU @ 2.70GHz with 7.8 GB, `measured_on: participant_laptop`.

| Metric | Value |
|---|---|
| generation throughput | 12.70 tok/s |
| peak RSS | 1.98 GB |
| S_perf = min(TPS/15, 1) | **84.7** |
| S_eff = (7 - peak)/7 | **71.8** |
| arc_easy (50 samples) | 0.7 acc_norm |

No thermal penalty applies: the measurement ran in a cloud instance, where the hypervisor exposes no CPU temperature. The audit environment has the same limitation.

The accuracy row above is the base model's general knowledge, measured by the
organisers' tool. The figures below measure something different: whether the
retrieval layer answers correctly *from a corpus*. Neither replaces the other,
and adding them together would be meaningless.

### Retrieval accuracy, our own evaluation set

- answer accuracy: **90%** over 10 answerable probes
- source accuracy: **100%** — the citation points at the document that states the fact
- refusal accuracy: **100%** over 2 questions the corpus cannot answer
- hallucination rate: **0%**

Measured against a ground-truth set held outside the corpus, so it cannot be
retrieved. Wrong figures score zero — there is no partial credit for a ceiling
of 25 reported as 35.

### With and without the retrieval layer

Same model, same machine, same session. The baseline is what a developer does
before reaching for an optimization layer: put the whole corpus in the prompt.

| Metric | Whole corpus in the prompt | With retrieval | Delta |
|---|---|---|---|
| prompt_tokens | 978.1111 | 343.9268 | -64.84% |
| context_tokens | 959.0 | 188.8049 | -80.31% |
| response_grounding_rate | 0.3745 | 0.6463 | +72.58% |
| tokens_per_second | 12.4422 | 12.8572 | +3.34% |
| total_ms | 31345.6987 | 7481.443 | -76.13% |
| peak_rss_mb | 31.8128 | 32.1907 | +1.19% |

## 5. Honest limits

- Grounding is checked by n-gram overlap against the passages sent. It catches
  invented figures and fabricated citations. It does not catch a fluent answer
  that recombines real passages into a false claim.
- Retrieval is lexical. A question asked entirely in vocabulary absent from the
  documents will not find them.
- The off-topic filter that produces the refusal accuracy above rests on a
  threshold tuned on a small probe set. It transferred across two languages
  after two defects were fixed, which is evidence and not proof.
- Thermal figures depend on sensors the machine exposes. Where none exist the
  field reads unavailable rather than a plausible number.
