# LeanLM

**Local Intelligence for the Laptop Africa Already Has.**

> Useful AI. No Cloud. 8 GB.

Most AI assumes the cloud. LeanLM assumes the laptop.

A useful, fully offline language model engineered to run fast and efficiently on
the 8 GB laptops Africa already has. Ask questions about your own documents, on
your own machine, with the network unplugged — and see every sentence traced back
to the document it came from.

Africa Deep Tech Challenge 2026 — Corporate / Enterprise domain.

```
   local LLM  →  no cloud  →  8 GB laptop  →  fast  →  low memory  →  real work
```

The model is the easy part: `llama.cpp` already runs a quantized 3–4B model on
that hardware. What makes it *usable* is the rest of the chain — quantization,
context, retrieval, memory, runtime, throughput. LeanLM is that chain, measured
end to end.

---

## The numbers

Measured on the reference corpus with the shipped evaluation set:

| | |
|---|---|
| answer accuracy | **100%** (10 answerable probes, English and French) |
| refusal accuracy | **100%** (2 probes the corpus cannot answer) |
| hallucination rate | **0%** |
| source accuracy | **100%** — cited the document that states it |
| corpus not sent to the model | **81%** on a typical question |

```bash
leanlm accuracy    # the table above, reproducible
make accuracy-fr   # the same pipeline, unmodified, on a French corpus
leanlm score       # the published rubric applied to our own run
```

The default corpus and probe set are English, because that is what the challenge
is judged in. The French set is kept and still asserted in the test suite: the
same pipeline answering a second language without a single configuration change
is a stronger statement than a claim of multilingual support.

The thesis in one line: the cheapest tokens are the ones you never send — and
the safest answer is the one the system declines to invent.

---

## Quick start

```bash
pip install -e .                        # no third-party dependency required
leanlm doctor                           # can this machine run LeanLM?
leanlm ingest datasets/enterprise_en    # index the sample corpus
leanlm ask "What is the reimbursement deadline for expense reports?"
leanlm serve                            # local workspace on 127.0.0.1:8770
```

To run against a real model — **no weights ship with LeanLM**, by design:

```bash
MODEL_URL=<url-to-a-gguf-file> ./scripts/download_model.sh      # Linux / macOS
```
```powershell
.\scripts\download_model.ps1 -Url "<url-to-a-gguf-file>"         # Windows
```
```bash
leanlm doctor
```

Aim for a 3–4B instruct model quantized to Q4_K_M — roughly 2.2–2.8 GB, which
leaves an 8 GB machine enough headroom not to swap.
[`docs/CHOOSING-A-MODEL.md`](docs/CHOOSING-A-MODEL.md) covers the sizing
arithmetic, what to check on a model card, and how to wire the checksum and
licence into the submission.

Without a model, LeanLM runs a **clearly labelled simulator** so that
development and tests work. Its output is marked `is_simulated` everywhere it
appears, and the submission gate refuses to package results produced by it.

---

## What happens when you ask a question

Ten phases, in a fixed order, every time. The workspace draws them as a
vertical rail with their real durations.

```
 1  session_creation       identity, profile, fingerprint
 2  resource_assessment    RAM, CPU, temperature — measured, not assumed
 3  document_discovery     the corpus snapshot, with its checksum
 4  context_optimization   budget for this machine, this intent; compaction
 5  evidence_retrieval     BM25 + structural boosts + diversity, under budget
 6  prompt_assembly        three fixed parts, labelled passages
 7  model_inference        llama.cpp
 8  response_validation    grounding, citation checking, confidence
 9  metrics_finalization   runtime and intelligence metrics
10  session_cleanup        temporary artifacts released
```

The order lives in exactly one tuple (`contracts/iec.py::STAGE_ORDER`). The
orchestrator iterates it, the contract verifier asserts against it, and the UI
renders it — so "deterministic pipeline" is a property you can check rather
than a sentence in a slide deck.

---

## Architecture

Eight capabilities, each owning consecutive stages of one pipeline, each with
the same seven modules.

| | Capability | Owns | Requirement |
|---|---|---|---|
| CAP-001 | Document Ingestion | import, validation, normalization | FR-01 |
| CAP-002 | Document Understanding | structure, segmentation, metadata | FR-02 |
| CAP-003 | Context Optimization | context_optimization | FR-03 |
| CAP-004 | Evidence Retrieval | evidence_retrieval | FR-04 |
| CAP-005 | Inference Execution | prompt_assembly, model_inference | FR-05 |
| CAP-006 | Response Validation | response_validation | FR-06 |
| CAP-007 | Performance Engineering | metrics_finalization | FR-07 |
| CAP-008 | Submission Packaging | submission_build, compliance_check | FR-08 |

```
src/leanlm/
  shared/        canonical object model, hashing, telemetry, errors, config
  contracts/     versioned DTOs, the immutable execution context, capability ABC
  packages/      the eight capabilities
  runtime/       profiles, corpus store, state machine, orchestrator, DIC verifier
  trust/         offline guard, model and runtime integrity
  benchmarks/    scenarios, statistics, evidence matrix, baselines
  apps/          CLI, local workspace, CCM verifier
```

`leanlm ccm` walks capability → package → contract → tests → benchmark →
documentation and fails if any link is missing. It runs in CI, so the mapping
cannot quietly rot.

---

## Measuring

```bash
leanlm baseline --run                         # the same model with NO LeanLM layer
leanlm --profile benchmark bench              # six scenarios, repeated, with dispersion
leanlm --profile benchmark bench --save-baseline
leanlm profile "your question"                # where the time goes
leanlm evidence                               # the performance evidence matrix
```

Two comparisons, deliberately kept in separate files because they answer
different questions:

| File | Question it answers |
|---|---|
| `naive_baseline.json` | is the layer worth it? — same model, whole corpus in the prompt |
| `baseline.json` | did we get worse? — a previous LeanLM campaign on this machine |

Conflating them produced a phantom "+224% regression" in CI once. Now
`is_regression_baseline()` refuses the substitution, and the report prints them
as two tables under two headings.

Every reported number carries mean, min, max, standard deviation, sample count
and a stability flag. A campaign records the machine, the profile fingerprint
and the corpus checksum alongside the results, because a measurement without
its conditions is not reproducible.

Six scenarios: small document, medium corpus, large corpus, repeated queries,
cold start, and **an unanswerable question** — S6 fails the campaign if the
system answers it.

---

## Four decisions worth defending

**BM25, not embeddings.** A second model in RAM does not pay for itself on an
8 GB laptop. Retrieval runs in 3.7 ms with no model loaded. Reversal conditions
are written down in `docs/ENGINEERING-DECISIONS.md`.

**A figure makes a passage unique.** "Reimbursed within 30 days" and "within 45
days" are 96% lexically identical and mean different things. Deduplication
compares numeric signatures first, so the deduplicator can never delete the
correct answer and keep a plausible wrong one.

**No evidence means no answer.** When nothing clears the relevance floor,
retrieval returns nothing and only a declaration of insufficiency passes
validation. This was not the original behaviour — the first version answered a
question about Tesla's share price with a verbatim quote about annual leave, at
*high* confidence, because the answer was faithful to a passage that had
nothing to do with the question. Two fixes were needed, both documented.

**The simulator is never a fallback.** Labelled in the context, in the UI, and
refused by the submission gate. This is the control that stops a green pipeline
from shipping throughput figures no model ever produced.

---

## The demonstration

On a generated corpus of 41 policy documents (~7 900 tokens against a
4 096-token window), asked for a figure stated in the last document:

| | Without LeanLM | With LeanLM |
|---|---|---|
| prompt tokens | 3 307 | **298** |
| corpus the model saw | 42% | 2.4%, selected |
| answer | *"the documents do not allow a conclusion"* | **7 500 EUR, cited** |

The naive path is not wrong because the model is weak. It is wrong because 58%
of the corpus never reached it. Rebuild the corpus with
`scripts/make_corpus.py`; the assertion lives in
`tests/submission/test_baseline_comparison.py` and fails the build if this stops
being true.

Where LeanLM loses is printed too: under the simulator its wall-clock is higher,
because there is no model cost to amortise the layer against. A comparison that
can only flatter us is not a measurement.

---

## Trust

- **Offline is enforced, not promised.** While a request runs, non-loopback
  sockets and DNS resolution raise, and any attempt is recorded in the execution
  context. Loopback stays open because `llama-server` is a legitimate local
  backend.
- **The model is verified.** GGUF magic bytes, size, and SHA-256 against the
  declared checksum before anything is benchmarked.
- **Telemetry carries no document content** unless you explicitly ask for it.
- **Nothing leaves the machine.** There is no telemetry endpoint to disable,
  because there is none.

---

## Testing

```bash
make test     # the suite
make ccm      # capability-to-code mapping
make ci       # everything the pipeline runs
```

The suite covers the shared foundations, the contracts, each of the eight
capabilities, the runtime integration, the full pipeline, and the delivery
gate. The tests that matter most are the ones asserting the system's honesty:
that an invented answer is caught, that a fabricated citation is detected, that
an out-of-corpus question is refused, and that simulated results cannot be
submitted.

---

## Documentation

| | |
|---|---|
| `docs/adr/` | architecture decision records |
| `docs/ENGINEERING-DECISIONS.md` | decisions, evidence, reversal conditions |
| `docs/TRACEABILITY.md` | requirement → capability → code → test |
| `src/leanlm/packages/*/README.md` | what each capability does and refuses to do |

---

## Where to measure

The audit sandbox re-runs your submission and compares: throughput tolerates
±25%, memory ±15%, and beyond 50% variance the comparison **fails**. The
measuring machine is therefore part of the submission.

`leanlm doctor` compares this machine against the standard profile — 4 vCPU,
8 GB, integrated GPU — and names the divergence. Use a development laptop for
functional verification, model selection and the report; produce the submitted
numbers on hardware close to the profile, or under
`docker run --memory=7.5g --cpus=4`. See
[`docs/WHERE-TO-MEASURE.md`](docs/WHERE-TO-MEASURE.md).

## Platform notes

Tested on Linux and Windows. The zero-dependency fallbacks cover both: `/proc` on
Linux, Win32 through `ctypes` on Windows.

Installing the optional dependencies buys accuracy of *measurement* rather than
accuracy of answers:

```bash
pip install -e ".[optional]"    # psutil, PyYAML, pypdf, pdfminer.six
```

All four are pure Python: no compiler, no toolchain, nothing that trips Windows'
260-character path limit.

**Do not use `.[all]` to get them.** It also pulls `llama-cpp-python`, which
builds from source, and pip aborts the whole transaction when one package fails —
so a single heavyweight dependency denies you the four lightweight ones. The
extras are split for that reason:

| Extra | Contents | Needs a toolchain |
|---|---|---|
| `optional` | psutil, PyYAML, pypdf, pdfminer.six | no |
| `runtime` | llama-cpp-python | yes |
| `dev` | pytest, ruff | no |

- **psutil** — more precise CPU sampling. Without it CPU reads 0 on Windows,
  which costs nothing in scoring but makes the profiler less informative.
- **pypdf / pdfminer.six** — PDF ingestion. Without both, PDFs are refused with
  an explicit error rather than indexed empty.

### Running a real model on Windows

`llama-cpp-python` is one way to reach llama.cpp, and the most demanding one: it
compiles, and on Windows its source archive exceeds the 260-character path limit
unless long paths are enabled. LeanLM supports two alternatives that need no
Python binding and no compiler at all:

```powershell
# 1. Download an official llama.cpp release for Windows and unzip it.
# 2. Either point LeanLM at the server:
.\llama-server.exe -m models\model.gguf --host 127.0.0.1 --port 8080
leanlm --backend llama-server ask "..."

# 3. Or at the binary directly:
leanlm --backend llama-cli --model models\model.gguf ask "..."
```

The server backend refuses any address that is not loopback, so pointing it at a
remote host fails rather than silently breaking the offline guarantee.

If you would rather have the in-process backend, enable long-path support first
(`HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled = 1`, then
reboot), or install a prebuilt wheel if one exists for your Python version —
`llama-cpp-python` publishes them for common platforms.

On Windows PowerShell, if accented characters arrive mangled (`délai` shown as
`dlai`), set the console to UTF-8 before starting:

```powershell
[Console]::OutputEncoding = [Text.Encoding]::UTF8
chcp 65001
```

Retrieval folds accents, so `délai` and `delai` are equivalent — but a character
the terminal has dropped entirely cannot be recovered.

## Honest limits

- Grounding by n-gram overlap catches invented figures and fabricated
  citations. It does not catch a fluent answer that recombines real passages
  into a false claim.
- Retrieval is lexical. A question asked entirely in vocabulary absent from the
  documents will not find them.
- Thermal readings depend on sensors the machine exposes. Where none exist, the
  field reads `unavailable` rather than a plausible number.
- The reference corpus is French enterprise policy documents. Performance on
  other document types is unmeasured, and unmeasured is not the same as good.

## License

Apache-2.0.
