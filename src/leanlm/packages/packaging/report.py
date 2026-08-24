"""REPORT.md -- the four sections the template asks for.

"Judges and the LLM-based audit system will read this to understand your
submission. Keep it factual and specific. One to three pages is ideal."

So: no marketing, no architecture tour, numbers with their conditions attached,
and the limits stated rather than left to be discovered.
"""
from __future__ import annotations

from typing import Any

from .models import SubmissionRequest


def _metric(summary: dict[str, Any], key: str, unit: str = "", digits: int = 1) -> str:
    node = (summary.get("metrics") or {}).get(key)
    if isinstance(node, dict):
        mean, low, high = node.get("mean"), node.get("min"), node.get("max")
        if mean is None:
            return "not measured"
        spread = f" (min {low}, max {high}, n={node.get('samples', '?')})" \
            if low is not None else ""
        return f"{mean:.{digits}f}{unit}{spread}"
    if isinstance(node, (int, float)):
        return f"{node:.{digits}f}{unit}"
    return "not measured"


def _naive_table(naive: dict[str, Any]) -> str:
    """The with/without comparison, including the rows where we lose."""
    lines = ["| Metric | Whole corpus in the prompt | With retrieval | Delta |",
             "|---|---|---|---|"]
    for row in naive["rows"]:
        delta = row.get("delta_pct")
        rendered = "-" if delta is None else f"{delta:+}%"
        lines.append(f"| {row['metric']} | {row['without_leanlm']} | "
                     f"{row['with_leanlm']} | {rendered} |")
    table = "\n".join(lines)
    if naive.get("naive_truncated_corpus"):
        table += (
            f"\n\nWithout retrieval the corpus does not fit the context window: the "
            f"model saw {naive.get('naive_corpus_coverage', 0):.0%} of it and never "
            f"saw {naive.get('naive_dropped_tokens')} tokens. A question whose answer "
            "lies in the discarded part is unanswerable for reasons that have nothing "
            "to do with the model.")
    return table


def _quantization_note(request: SubmissionRequest) -> str:
    """Say what the chosen quantization actually cost, not what a typical one does.

    The first version of this paragraph asserted that Q4_K_M lands near 2.5 GB.
    It said so in a report describing a Q4_0 file of 1.16 GB, which is the kind
    of stale boilerplate a reader checks and then distrusts the rest of.
    """
    profiler = request.profiler_report or {}
    peak = ((profiler.get("memory") or {}).get("peak_rss_mb") or 0) / 1024.0
    if not peak:
        return ""
    return (f" At this quantization the loaded model and its KV cache peaked at "
            f"{peak:.2f} GB of the 7 GB budget.")


def _peak_gb(request: SubmissionRequest) -> str:
    profiler = request.profiler_report or {}
    peak = ((profiler.get("memory") or {}).get("peak_rss_mb") or 0) / 1024.0
    return f"{peak:.2f} GB" if peak else "not yet measured"


def _official_peak(request: SubmissionRequest) -> str:
    """The profiler's figure, never LeanLM's own process RSS.

    LeanLM drives an out-of-process backend, so its own RSS is around 32 MB
    while a 1.16 GB model sits in llama-server. Printing 32 MB in a memory table
    is not a small discrepancy -- efficiency is 20% of the score, and a figure
    that omits the model measures nothing about the model.
    """
    profiler = request.profiler_report or {}
    peak = (profiler.get("memory") or {}).get("peak_rss_mb")
    if not peak:
        return ("not measured here -- LeanLM's own process excludes the model, "
                "which runs under llama-server")
    return f"{peak:.0f} MB ({peak / 1024:.2f} GB of a 7 GB budget)"


def _official_block(request: SubmissionRequest) -> str:
    """The numbers the organisers' own tool produced, kept apart from ours."""
    profiler = request.profiler_report or {}
    if not profiler:
        return ("The official profiler has not been run yet. The figures above "
                "come from our own harness and are not a substitute.")
    environment = profiler.get("environment", {})
    throughput = profiler.get("throughput", {})
    memory = profiler.get("memory", {})
    tps = throughput.get("tokens_per_second_generation") or 0.0
    peak_gb = (memory.get("peak_rss_mb") or 0) / 1024.0
    accuracy = profiler.get("accuracy") or []
    rows = [
        "| Metric | Value |",
        "|---|---|",
        f"| generation throughput | {tps:.2f} tok/s |",
        f"| peak RSS | {peak_gb:.2f} GB |",
        f"| S_perf = min(TPS/15, 1) | **{min(tps / 15.0, 1.0) * 100:.1f}** |",
        f"| S_eff = (7 - peak)/7 | **{max(0.0, (7.0 - peak_gb) / 7.0) * 100:.1f}** |",
    ]
    for entry in accuracy:
        rows.append(f"| {entry.get('benchmark')} ({entry.get('samples')} samples) "
                    f"| {entry.get('score')} {entry.get('metric', '')} |")
    thermal = profiler.get("cpu_thermal", {})
    note = ""
    if thermal.get("core_temp_c_peak") is None:
        note = ("\n\nNo thermal penalty applies: the measurement ran in a cloud "
                "instance, where the hypervisor exposes no CPU temperature. The "
                "audit environment has the same limitation.")
    return (f"Measured on {environment.get('cpu_model', 'unstated')} with "
            f"{environment.get('ram_gb', '?')} GB, `measured_on: "
            f"{environment.get('measured_on', '?')}`.\n\n" + "\n".join(rows) + note)


def render_report(request: SubmissionRequest) -> str:
    summary = request.benchmark_summary or {}
    accuracy = request.accuracy_summary or {}
    naive = request.naive_comparison or {}
    model = request.model
    environment = summary.get("environment") or {}

    simulated_banner = ""
    if summary.get("is_simulated"):
        simulated_banner = (
            "> **These numbers were produced by a simulated backend, not by a model.**\n"
            "> They describe the optimization layer only and must not be read as model\n"
            "> performance.\n\n")

    accuracy_block = "Not yet measured against ground truth."
    if accuracy:
        accuracy_block = (
            f"- answer accuracy: **{accuracy.get('accuracy', 0):.0%}** over "
            f"{accuracy.get('answerable', 0)} answerable probes\n"
            f"- source accuracy: **{accuracy.get('source_accuracy', 0):.0%}** — the "
            "citation points at the document that states the fact\n"
            f"- refusal accuracy: **{accuracy.get('refusal_accuracy', 0):.0%}** over "
            f"{accuracy.get('unanswerable', 0)} questions the corpus cannot answer\n"
            f"- hallucination rate: **{accuracy.get('hallucination_rate', 0):.0%}**"
        )

    naive_block = "Not yet measured."
    if naive.get("rows"):
        naive_block = _naive_table(naive)

    return f"""# {model.name or 'LeanLM'} — ADTC 2026 Report

**Domain:** {request.domain} · **Languages:** {', '.join(request.language_scope)}
**Runtime:** {model.runtime} · **Quantization:** {model.quantization} ·
**Parameters:** {model.parameters_estimate or 'unstated'}
{f"**Repository:** {request.repository_url}" if request.repository_url else ""}
{f"**Commit:** `{request.git_commit}`" if request.git_commit else ""}

{simulated_banner}---

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

**Model.** {model.name or 'A GGUF instruct model'} at {model.quantization},
{model.parameters_estimate or 'sized'}, chosen to leave headroom on an 8 GB
machine.{_quantization_note(request)}

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
| 8 GB RAM, 4 vCPU, integrated GPU | model and cache measured at {_peak_gb(request)}; peak matters, not mean |
| No network during inference | enforced, not promised: non-loopback sockets raise and are recorded |
| Documents are confidential | nothing leaves the machine; there is no telemetry endpoint to disable |
| Intermittent connectivity | zero mandatory Python dependencies; every optional one has a fallback |
| Answers are acted upon | every sentence is checked against the passages actually sent |

Environment for the measurements below: {environment.get('platform', 'unstated')},
Python {environment.get('python', '?')}, {environment.get('cpu_count', '?')} CPUs,
{environment.get('total_ram_mb', '?')} MB RAM.

## 4. Benchmarks

Every figure is a mean over repeated runs with its spread, on the corpus
identified by checksum `{(summary.get('corpus') or {}).get('checksum', 'n/a')}`
under profile `{(summary.get('profile') or {}).get('id', 'n/a')}`
(fingerprint `{(summary.get('profile') or {}).get('fingerprint', 'n/a')}`).

| Metric | Value |
|---|---|
| throughput | {_metric(summary, 'tokens_per_second', ' tok/s')} |
| first token | {_metric(summary, 'first_token_latency_ms', ' ms', 0)} |
| total per request | {_metric(summary, 'total_ms', ' ms', 1)} |
| peak RSS (whole system, official profiler) | {_official_peak(request)} |
| prompt size | {_metric(summary, 'prompt_tokens', ' tokens', 0)} |
| corpus not sent to the model | {_metric(summary, 'context_compression_ratio', '', 2)} |

### Official profiler

{_official_block(request)}

The accuracy row above is the base model's general knowledge, measured by the
organisers' tool. The figures below measure something different: whether the
retrieval layer answers correctly *from a corpus*. Neither replaces the other,
and adding them together would be meaningless.

### Retrieval accuracy, our own evaluation set

{accuracy_block}

Measured against a ground-truth set held outside the corpus, so it cannot be
retrieved. Wrong figures score zero — there is no partial credit for a ceiling
of 25 reported as 35.

### With and without the retrieval layer

Same model, same machine, same session. The baseline is what a developer does
before reaching for an optimization layer: put the whole corpus in the prompt.

{naive_block}

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
"""
