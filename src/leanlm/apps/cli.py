"""LeanLM command line interface.

Design rule: every command prints something a human can act on, and exits with
a status code CI can act on. No command silently succeeds after doing nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .. import __version__
from ..benchmarks import (compare_to_baseline, load_baseline, load_benchmark_config,
                          regressions, render_matrix, run_campaign, save_baseline)
from ..benchmarks.pem import PerformanceEvidence
from ..packages.packaging import ComplianceChecker, SubmissionBuilder
from ..packages.packaging.models import SubmissionRequest
from ..runtime import CorpusStore, LeanLMRuntime
from ..runtime.profiles import list_profiles, load_profile
from ..shared.config import repo_root
from ..shared.errors import LeanLMError
from ..shared.logging import configure
from ..shared.serialization import canonical_json, write_json
from ..trust.integrity import verify_model_binding, verify_runtime_integrity
from .ccm import verify as verify_ccm

EXIT_OK, EXIT_ERROR, EXIT_NONCOMPLIANT = 0, 1, 2


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _runtime(args: argparse.Namespace, **kwargs: Any) -> LeanLMRuntime:
    corpus = CorpusStore(args.corpus) if getattr(args, "corpus", None) else None
    return LeanLMRuntime(args.profile, corpus=corpus,
                         model_path=getattr(args, "model", None) or None,
                         backend=getattr(args, "backend", None) or None,
                         **kwargs)


def _print(payload: Any, as_json: bool) -> None:
    if as_json:
        print(canonical_json(payload))
    else:
        print(payload)


def _fail(error: LeanLMError) -> int:
    record = error.record
    print(f"[{record.code}] {record.message}", file=sys.stderr)
    if record.recommended_action:
        print(f"  -> {record.recommended_action}", file=sys.stderr)
    return EXIT_ERROR


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_doctor(args: argparse.Namespace) -> int:
    """Answer, in one screen: can this machine run LeanLM right now?"""
    lines: list[str] = [f"LeanLM {__version__} -- environment check", ""]
    ok = True
    missing_optional: list[str] = []

    lines.append(f"  python           : {sys.version.split()[0]}")
    for module, purpose in (("llama_cpp", "in-process GGUF inference"),
                            ("pypdf", "PDF text extraction"),
                            ("pdfminer", "PDF fallback extraction"),
                            ("psutil", "precise resource sampling"),
                            ("yaml", "configuration parsing")):
        try:
            __import__(module)
            lines.append(f"  {module:17}: available ({purpose})")
        except ImportError:
            lines.append(f"  {module:17}: MISSING -- {purpose} uses the fallback")
            if module == "llama_cpp":
                lines.append("                     -> optional. LeanLM also drives "
                             "the llama.cpp binaries directly.")
            elif module in ("pypdf", "pdfminer", "psutil", "yaml"):
                missing_optional.append(module)

    # The binaries matter more than the Python binding: they are the route that
    # needs no compiler, and the official profiler shells out to llama-bench.
    import shutil as _shutil
    binaries = {name: _shutil.which(name)
                for name in ("llama-bench", "llama-cli", "llama-server")}
    found = [n for n, p in binaries.items() if p]
    # Whether a server is already up decides which backend `auto` picks, so it
    # belongs on screen next to the binaries rather than being inferred from a
    # port-in-use error.
    from ..packages.inference.backends import _server_is_listening
    if _server_is_listening():
        lines.append("  llama-server     : listening on 127.0.0.1:8080 "
                     "(auto will use it)")
    else:
        lines.append("  llama-server     : not running")

    if found:
        lines.append(f"  llama.cpp binaries: {', '.join(found)}")
        # Which flags this build accepts, asked rather than assumed. `-no-cnv`
        # is rejected outright by some builds, and getting it wrong leaves the
        # process waiting on stdin instead of answering.
        try:
            from ..packages.inference.backends import LlamaCppBinaryBackend
            from ..packages.inference.models import ModelBinding
            from ..packages.inference.policies import GenerationPolicy
            probe = LlamaCppBinaryBackend(ModelBinding(model_id="probe", path=""),
                                          GenerationPolicy())
            flags = probe.supported_flags()
            if not flags["help_available"]:
                lines.append("                     flags: --help unreadable; "
                             "arguments will be sent unverified")
            elif flags["no_conversation"]:
                lines.append(f"                     flags: "
                             f"{flags['no_conversation']} accepted"
                             + (f", {', '.join(flags['optional'])}"
                                if flags["optional"] else ""))
            else:
                lines.append("                     flags: no way to disable "
                             "conversation mode was detected; a request may wait "
                             "on stdin")
        except Exception:
            pass
    else:
        lines.append("  llama.cpp binaries: MISSING -- none of llama-bench, "
                     "llama-cli, llama-server on PATH")
        lines.append("                     -> Windows: .\\scripts\\provision_windows.ps1")
        lines.append("                        Linux:   bash scripts/provision_ubuntu.sh")
        lines.append("                        llama-bench is also what the official "
                     "profiler calls; without it")
        lines.append("                        no throughput can be measured at all.")

    try:
        profile = load_profile(args.profile)
        lines.append(f"  profile          : {profile.id} v{profile.version} "
                     f"[{profile.fingerprint}]")
    except LeanLMError as error:
        lines.append(f"  profile          : ERROR {error.record.message}")
        return _fail(error)

    binding = profile.model
    model_path = args.model or binding.get("path", "")
    if model_path and Path(model_path).is_file():
        report = verify_model_binding(model_path, str(binding.get("sha256", "")))
        state = "verified" if report.trusted else "PRESENT BUT UNVERIFIED"
        from ..runtime.profiles import parameter_mismatch
        mismatch = parameter_mismatch(profile)
        if mismatch:
            lines.append(f"  declaration      : {mismatch}")
            ok = False
        lines.append(f"  model            : {state} "
                     f"({report.details.get('size_mb', '?')} MB)")
        ok = ok and report.trusted
    else:
        lines.append(f"  model            : NOT FOUND at '{model_path or '(unset)'}'")
        lines.append("                     -> run scripts/download_model.sh; without a "
                     "model LeanLM runs in simulated mode and results are not valid")
        ok = False

    try:
        with _runtime(args, verify_model=False) as runtime:
            status = runtime.status()
            resources = status["resources"]
            backend = status["backend"]
            lines.append(f"  backend          : {backend['backend']}"
                         f"{'  (SIMULATED -- results are not model results)'
                            if backend.get('is_simulated') else ''}")
            lines.append(f"  available RAM    : {resources.get('available_ram_mb')} MB")
            lines.append(f"  temperature      : {resources.get('temperature_c')} "
                         f"(source: {resources.get('thermal_source', 'unknown')})")
            lines.append(f"  corpus           : {status['corpus']['documents']} documents, "
                         f"{status['corpus']['units']} units")
            lines.append(f"  offline enforced : {status['offline_enforced']}")
    except LeanLMError as error:
        return _fail(error)

    from ..runtime.hardware import inspect as inspect_hardware
    from ..runtime.hardware import render as render_hardware
    hardware = inspect_hardware()
    lines.extend(render_hardware(hardware))

    ccm = verify_ccm()
    lines.append(f"  capability map   : {'complete' if ccm.ok else 'INCOMPLETE'}")
    if missing_optional:
        lines.append("")
        lines.append(f"  {len(missing_optional)} optional package(s) missing "
                     f"({', '.join(missing_optional)}). All pure Python:")
        lines.append('    pip install -e ".[optional]"')
        lines.append("  Without them LeanLM still runs, but PDFs are refused and "
                     "memory sampling")
        lines.append("  is coarser -- which costs the efficiency component of the "
                     "score, not correctness.")
    lines.append("")
    lines.append("READY" if ok else "NOT READY -- see the lines above")
    _print("\n".join(lines) if not args.json else {"ok": ok, "report": lines}, args.json)
    return EXIT_OK if ok else EXIT_ERROR


def cmd_ingest(args: argparse.Namespace) -> int:
    with _runtime(args, verify_model=False) as runtime:
        try:
            result = runtime.ingest(args.paths, replace=args.replace)
        except LeanLMError as error:
            return _fail(error)
    if args.json:
        _print(result, True)
        return EXIT_OK if not result["failed"] else EXIT_ERROR
    for item in result["ingested"]:
        print(f"  + {item['file']}: {item['units']} units, {item['tokens']} tokens")
    for item in result["failed"]:
        print(f"  ! {item['file']}: [{item['code']}] {item['message']}", file=sys.stderr)
        print(f"    -> {item['action']}", file=sys.stderr)
    print(f"\nadded : {result['documents_added']} documents, "
          f"{result['units_added']} units")
    print(f"corpus: {result['corpus_documents']} documents, "
          f"{result['corpus_units']} units [{result['corpus_checksum'][:16]}]")
    if not result["replaced"] and result["corpus_documents"] > result["documents_added"]:
        print("        note: documents from an earlier ingest are still in the "
              "corpus. An")
        print("        accuracy run measures against all of them, including any "
              "the evaluation")
        print("        set does not account for. Use `--replace` for a clean "
              "measurement.")
    return EXIT_OK if not result["failed"] else EXIT_ERROR


def cmd_ask(args: argparse.Namespace) -> int:
    with _runtime(args, verify_model=not args.allow_simulated) as runtime:
        if args.dry_run:
            return _show_backend_command(runtime, args.question)
        streamed = _install_stream(runtime, quiet=args.json or args.quiet)
        try:
            iec = runtime.ask(args.question)
        except LeanLMError as error:
            return _fail(error)
        if args.json:
            _print(iec.to_dict(include_text=True), True)
            return EXIT_OK
        response = iec.response
        validation = iec.validation
        if response is not None and response.is_simulated:
            print("!! SIMULATED BACKEND -- no model produced this answer\n")
        # Whether anything streamed is observed, never assumed: only the
        # subprocess backend emits tokens, and taking the hook's presence as
        # proof printed the sources and the verdict with no answer between them.
        if not streamed["started"]:
            print(response.text if response else "(no answer)")
        print()
        if iec.evidence:
            print("Sources:")
            for item in iec.evidence:
                section = " > ".join(item.section_path) if item.section_path else "-"
                print(f"  [{item.label}] {item.document_name} :: {section}")
        if validation:
            print(f"\nconfidence: {validation.confidence.value} "
                  f"| grounding {validation.grounding_rate} "
                  f"| {'validated' if validation.passed else 'NOT VALIDATED'}")
            for warning in validation.warnings:
                print(f"  warning: {warning}")
        if iec.metrics:
            metrics = iec.metrics
            print(f"\n{metrics.total_ms} ms total | {metrics.tokens_per_second} tok/s "
                  f"| first token {metrics.first_token_latency_ms} ms "
                  f"| peak RSS {metrics.peak_rss_mb} MB "
                  f"| context compressed {metrics.context_compression_ratio:.0%}")
        for warning in iec.warnings:
            print(f"  ! {warning}")
    return EXIT_OK


def _show_backend_command(runtime, question: str) -> int:
    """Print the command the backend would run, without running it.

    When a request appears to hang, running the same command in a terminal is
    the fastest diagnosis available: attached to a tty llama.cpp flushes every
    token, so a stall becomes visible at the point it happens.
    """
    from ..contracts.iec import PipelineStage
    from ..shared.config import workdir

    runtime.load()
    backend = runtime.backend
    if not hasattr(backend, "describe_command"):
        print(f"the {backend.name} backend runs no external command",
              file=sys.stderr)
        return EXIT_ERROR

    # Build the real prompt: the whole point is to hand over the exact input the
    # backend would receive, not an approximation of it.
    iec = runtime.ask(question, stop_after=PipelineStage.PROMPT_ASSEMBLY,
                      check_contract=False)
    if iec.prompt is None:
        print("the pipeline produced no prompt", file=sys.stderr)
        for error in iec.errors:
            print(f"  [{error.code}] {error.message}", file=sys.stderr)
        return EXIT_ERROR

    path = workdir() / "last_prompt.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(iec.prompt.rendered, encoding="utf-8", newline="\n")

    print(backend.describe_command(iec.prompt.rendered, str(path)))
    print()
    print(f"  prompt      : {path} ({iec.prompt.prompt_tokens} tokens, "
          f"{len(iec.evidence)} excerpts)")
    print(f"  expect      : about {iec.prompt.prompt_tokens / 30:.0f}s of prompt "
          "processing, then generation")
    print("  run it as printed. Attached to a terminal llama.cpp flushes every")
    print("  token, so a stall is visible where it happens.")
    return EXIT_OK


def _install_stream(runtime, *, quiet: bool) -> dict:
    """Echo generated text as it arrives, and say what is happening first.

    On a two-core laptop a request takes minutes. Without this the terminal is
    silent throughout and the only way to tell a working run from a hung one is
    to interrupt it -- which is exactly what happened.
    """
    state = {"started": False}
    if quiet:
        return state
    capability = runtime.registry.get("CAP-005")
    if capability is None:
        return state

    def on_token(chunk: str) -> None:
        if not state["started"]:
            state["started"] = True
            sys.stderr.write("\r" + " " * 72 + "\r")
            sys.stderr.flush()
        sys.stdout.write(chunk)
        sys.stdout.flush()

    def on_progress(elapsed_s: float, phase: str, first: bool = False) -> None:
        # Overwritten in place on stderr, so it never lands in piped output.
        # The first tick ends with a newline: a line rewritten with \r can vanish
        # from a copied transcript, which makes a working run look like silence.
        if state["started"]:
            return
        terminator = "\n" if first else ""
        sys.stderr.write(f"\r  {phase} ... {elapsed_s:.0f}s{terminator}")
        sys.stderr.flush()

    capability.on_token = on_token
    capability.on_progress = on_progress
    backend = runtime.backend.describe().get("backend", "?")
    print(f"  {backend}: a two-core laptop takes minutes before the first token; "
          f"Ctrl-C to stop", file=sys.stderr)
    return state


def cmd_bench(args: argparse.Namespace) -> int:
    config = load_benchmark_config()
    scenarios = config.scenarios
    if args.scenario:
        wanted = {s.lower() for s in args.scenario}
        scenarios = tuple(s for s in scenarios if s.id.lower() in wanted)
        if not scenarios:
            print(f"no scenario matches {args.scenario}", file=sys.stderr)
            return EXIT_ERROR
    output = Path(args.output) if args.output else \
        repo_root() / config.results_dir / "latest.json"
    baseline = args.baseline or (repo_root() / config.baseline_file)
    naive = repo_root() / "benchmarks/results/naive_baseline.json"

    def progress(event: str, payload: dict) -> None:
        if args.quiet:
            return
        if event == "scenario_started":
            print(f"  running {payload['id']} {payload['name']} ...", flush=True)
        elif event == "scenario_completed":
            print(f"    {payload['id']}: {payload['verdict']}", flush=True)

    with _runtime(args, verify_model=not args.allow_simulated) as runtime:
        try:
            runtime.load()
            summary = run_campaign(
                runtime, repeat=args.repeat or config.repeat,
                warmup=args.warmup if args.warmup is not None else config.warmup,
                scenarios=scenarios, output=output, baseline_path=baseline,
                naive_path=naive, progress=progress)
        except LeanLMError as error:
            return _fail(error)

    if args.with_profile:
        # SUB-010 wants a profiler report alongside the results. Running it here
        # keeps the two in the same file, produced by the same corpus and profile.
        from ..benchmarks.profiler import profile_request
        question = next((q for s in scenarios for q in s.questions), None)
        if question:
            with _runtime(args, verify_model=False) as runtime:
                runtime.corpus = runtime.corpus  # explicit: same store, same corpus
                summary["profiler"] = profile_request(runtime, question, repeat=2)
            write_json(output, summary)

    if args.json:
        _print(summary, True)
    else:
        print(f"\ncampaign {summary['campaign_id']}: {summary['verdict']}")
        if summary.get("is_simulated"):
            print("!! " + summary["warning"])
        metrics = summary["metrics"]
        for key in ("tokens_per_second", "first_token_latency_ms", "total_ms",
                    "peak_rss_mb", "context_compression_ratio"):
            node = metrics.get(key)
            if node and node["samples"]:
                print(f"  {key:28}: {node['mean']} "
                      f"(min {node['min']} / max {node['max']} / sd {node['stdev']}, "
                      f"n={node['samples']}{'' if node['stable'] else ', UNSTABLE'})")
        naive_comparison = summary.get("naive_comparison") or {}
        if naive_comparison:
            print("\nagainst the same model with no LeanLM layer:")
            for row in naive_comparison["rows"]:
                if row["delta_pct"] is None:
                    continue
                mark = "+" if row["favours_leanlm"] else "-"
                print(f"  [{mark}] {row['metric']:26}: {row['without_leanlm']} -> "
                      f"{row['with_leanlm']} ({row['delta_pct']:+}%)")
            if not naive_comparison.get("comparable"):
                print("  !! the two runs are not directly comparable")

        found = summary.get("regressions") or []
        if found:
            print("\nREGRESSIONS vs baseline:")
            for item in found:
                print(f"  {item['metric']}: {item['baseline']} -> {item['current']} "
                      f"({item['delta_pct']:+}%)")
        print(f"\nwritten to {output}")
    if args.save_baseline:
        save_baseline(baseline, summary)
        print(f"baseline updated: {baseline}")
    if summary.get("regressions"):
        return EXIT_NONCOMPLIANT
    return EXIT_OK if summary["verdict"] == "pass" else EXIT_ERROR


def cmd_baseline(args: argparse.Namespace) -> int:
    path = Path(args.path) if args.path else repo_root() / load_benchmark_config().baseline_file

    if args.run:
        naive_path = (Path(args.path) if args.path
                      else repo_root() / "benchmarks/results/naive_baseline.json")
        return _run_naive_baseline(args, naive_path)

    baseline = load_baseline(path)
    if baseline is None:
        print(f"no baseline at {path}", file=sys.stderr)
        print("  -> run `leanlm bench --save-baseline` on a reference machine",
              file=sys.stderr)
        return EXIT_ERROR
    if args.compare:
        current = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        comparison = compare_to_baseline(current, baseline)
        for item in comparison.values():
            flag = "REGRESSION" if item.regression else ("better" if item.improved else "ok")
            print(f"  {item.metric:28}: {item.baseline} -> {item.current} "
                  f"({item.delta_pct:+}%) {flag}")
        return EXIT_NONCOMPLIANT if regressions(comparison) else EXIT_OK
    _print(baseline, args.json)
    return EXIT_OK


def _run_naive_baseline(args: argparse.Namespace, path: Path) -> int:
    """Measure the naive path: same model, same machine, no LeanLM layer."""
    from ..benchmarks.naive import run_naive_baseline

    with _runtime(args, verify_model=not args.allow_simulated) as runtime:
        try:
            summary = run_naive_baseline(runtime, repeat=args.repeat or 3,
                                         warmup=1 if args.repeat != 1 else 0,
                                         output=str(path))
        except LeanLMError as error:
            return _fail(error)

    if args.json:
        _print(summary, True)
        return EXIT_OK
    print(f"naive baseline: {summary['verdict']}")
    print(f"  corpus            : {summary['corpus']['tokens']} tokens "
          f"in {summary['corpus']['documents']} documents")
    print(f"  model window      : {summary['context_window']} tokens")
    print(f"  corpus truncated  : {summary['corpus_truncated']}"
          f"{f'  ({summary["dropped_tokens"]} tokens dropped)' if summary['corpus_truncated'] else ''}")
    print(f"  corpus coverage   : {summary['corpus_coverage']:.1%}")
    for key in ("prompt_tokens", "total_ms", "tokens_per_second", "peak_rss_mb",
                "response_grounding_rate"):
        node = summary["metrics"].get(key)
        if node and node["samples"]:
            print(f"  {key:18}: {node['mean']} "
                  f"(min {node['min']} / max {node['max']}, n={node['samples']})")
    if summary.get("note"):
        print(f"\n  {summary['note']}")
    if summary.get("warning"):
        print(f"\n  !! {summary['warning']}")
    print(f"\nwritten to {path}")
    print("  compare with: leanlm bench   (the same corpus through LeanLM)")
    return EXIT_OK


def cmd_profile(args: argparse.Namespace) -> int:
    """Where does the time go? Not: how fast is it."""
    from ..benchmarks.profiler import profile_request, render_profile

    with _runtime(args, verify_model=not args.allow_simulated) as runtime:
        try:
            report = profile_request(runtime, args.question, repeat=args.repeat,
                                     top=args.top)
        except LeanLMError as error:
            return _fail(error)
    if args.output:
        write_json(args.output, report)
    if args.json:
        _print(report, True)
    else:
        print(render_profile(report))
        if args.output:
            print(f"\nwritten to {args.output}")
    return EXIT_OK


def cmd_accuracy(args: argparse.Namespace) -> int:
    """The heaviest criterion, measured against ground truth."""
    from ..benchmarks.accuracy import evaluate_accuracy, render_accuracy

    output = Path(args.output) if args.output else \
        repo_root() / "benchmarks/results/accuracy.json"
    with _runtime(args, verify_model=not args.allow_simulated) as runtime:
        try:
            summary = evaluate_accuracy(runtime, args.evaluation, output=output)
        except LeanLMError as error:
            return _fail(error)
    if args.json:
        _print(summary, True)
    else:
        print(render_accuracy(summary))
        print(f"\nwritten to {output}")
    return EXIT_OK if not summary["failures"] else EXIT_NONCOMPLIANT


def cmd_score(args: argparse.Namespace) -> int:
    """What would this score? Different question from "does it work"."""
    from ..benchmarks.scoring import ScoringPolicy, render_score, score_campaign

    results = Path(args.results) if args.results else \
        repo_root() / "benchmarks/results/latest.json"
    accuracy_path = Path(args.accuracy) if args.accuracy else \
        repo_root() / "benchmarks/results/accuracy.json"
    if not results.is_file():
        print(f"no campaign at {results}", file=sys.stderr)
        print("  -> run `leanlm bench` first", file=sys.stderr)
        return EXIT_ERROR
    campaign = json.loads(results.read_text(encoding="utf-8"))
    accuracy = (json.loads(accuracy_path.read_text(encoding="utf-8"))
                if accuracy_path.is_file() else None)
    score = score_campaign(campaign, accuracy, ScoringPolicy.load())
    if args.json:
        _print(score.to_dict(), True)
    else:
        print(render_score(score))
        if accuracy is None:
            print("\n  (no accuracy file; run `leanlm accuracy`)")
    return EXIT_NONCOMPLIANT if score.disqualified else EXIT_OK


def _load_profiler(explicit: str | None, root: Path) -> dict[str, Any] | None:
    """The profiler's own output, if it exists."""
    candidates = [Path(explicit)] if explicit else [
        root / "dist/submission/submission.json",
        Path("submission.json"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                continue
            # The profiler's file, not ours: it carries an environment block.
            if "environment" in payload and "throughput" in payload:
                return payload
    return None


def cmd_devpost(args: argparse.Namespace) -> int:
    """The two scores the submission form asks for, taken from the file.

    Computing them by hand invites a mismatch between the form and the artifact,
    and the audit compares exactly those two.
    """
    path = Path(args.profiler) if args.profiler else \
        repo_root() / "dist/submission/submission.json"
    if not path.is_file():
        print(f"no profiler output at {path}", file=sys.stderr)
        print("  -> adtc-profiler run --submission . --mode participant "
              "--output submission.json", file=sys.stderr)
        return EXIT_ERROR
    report = json.loads(path.read_text(encoding="utf-8"))

    if "sperf" in report or "seff" in report:
        print("the profiler reports these directly; use its values:")
        print(f"  Sperf : {report.get('sperf')}")
        print(f"  Seff  : {report.get('seff')}")
        return EXIT_OK

    tps = (report.get("throughput") or {}).get("tokens_per_second_generation") or 0.0
    peak_mb = (report.get("memory") or {}).get("peak_rss_mb") or 0.0
    peak_gb = peak_mb / 1024.0
    sperf = min(tps / 15.0, 1.0) * 100.0
    seff = max(0.0, (7.0 - peak_gb) / 7.0) * 100.0
    environment = report.get("environment", {})

    print("Devpost — Additional info")
    print(f"  Sperf : {sperf:.1f}      ({tps:.2f} tok/s / 15)")
    print(f"  Seff  : {seff:.1f}      (7 GB - {peak_gb:.2f} GB) / 7 GB")
    print()
    print(f"  measured_on : {environment.get('measured_on')}")
    print(f"  cpu         : {environment.get('cpu_model')}")
    for entry in report.get("accuracy") or ():
        print(f"  {entry.get('benchmark')} : {entry.get('score')} "
              f"({entry.get('samples')} samples) -- the base model's general "
              "knowledge, not the retrieval layer's accuracy")

    # The form and the report must agree; the audit compares them.
    md = path.parent / "REPORT.md"
    if md.is_file():
        text = md.read_text(encoding="utf-8")
        duplicates = text.count("### Official profiler")
        if duplicates > 1:
            print(f"\n  ! REPORT.md contains {duplicates} 'Official profiler' "
                  "sections. Remove the duplicate.")
        for label, value in (("S_perf", sperf), ("S_eff", seff)):
            if f"{value:.1f}" not in text:
                print(f"  ! REPORT.md does not contain {label} = {value:.1f}. "
                      "The form and the report should state the same number.")
    return EXIT_OK


def cmd_submission(args: argparse.Namespace) -> int:
    """Emit a package shaped by the official ADTC template."""
    from ..packages.packaging.models import (CrossDisciplinaryPairing,
                                             ModelDeclaration, Submitter, TestPrompt)

    root = repo_root()
    results = Path(args.results) if args.results else root / "benchmarks/results/latest.json"
    accuracy_path = Path(args.accuracy) if args.accuracy else \
        root / "benchmarks/results/accuracy.json"
    naive_path = root / "benchmarks/results/naive_baseline.json"

    summary = json.loads(results.read_text(encoding="utf-8")) if results.is_file() else {}
    if not summary and not args.allow_missing_results:
        print(f"no benchmark results at {results}", file=sys.stderr)
        print("  -> run `leanlm bench` first: a submission without measurements "
              "cannot be defended", file=sys.stderr)
        return EXIT_ERROR
    accuracy = (json.loads(accuracy_path.read_text(encoding="utf-8"))
                if accuracy_path.is_file() else {})
    naive = (json.loads(naive_path.read_text(encoding="utf-8"))
             if naive_path.is_file() else None)
    from ..benchmarks.baseline import compare_to_naive
    comparison = compare_to_naive(summary, naive) if naive else {}

    profile = load_profile(args.profile)
    model_config = profile.model
    prompts = tuple(TestPrompt(f"tp_{i + 1:03d}", text)
                    for i, text in enumerate(args.prompt or ()))

    request = SubmissionRequest(
        output_dir=args.output,
        team_id=args.team_id,
        domain=args.domain,
        language_scope=tuple(args.language or ("en",)),
        african_alpha_claim=args.african_use_case,
        submitter=Submitter(args.name, args.email, args.github),
        pairing=CrossDisciplinaryPairing(args.discipline, True, args.pairing),
        test_prompts=prompts,
        model=ModelDeclaration(
            name=str(model_config.get("id", "")),
            quantization=f"GGUF {model_config.get('quantization', 'Q4_K_M')}",
            parameters_estimate=str(model_config.get("parameters", "")),
            # The repository keeps weights in `models/`; the template requires
            # `model/`. Only the basename carries over.
            model_path=args.model_path or
            f"model/{Path(str(model_config.get('path', 'model.gguf'))).name}",
            source_url=str(model_config.get("source_url", "")),
            sha256=str(model_config.get("sha256", "")),
            license=str(model_config.get("license", "")),
        ),
        benchmark_summary=summary,
        accuracy_summary=accuracy,
        naive_comparison=comparison,
        # Read the official measurement directly. It used to be pulled from a
        # key inside the campaign summary, which the profiler never writes, so
        # the report had no figures to cite and the section was pasted in by
        # hand -- twice, in at least one case.
        profiler_report=_load_profiler(args.profiler, root),
        repository_url=args.repository,
        git_commit=args.commit,
    )

    builder = SubmissionBuilder()
    try:
        result = builder.build(request)
    except LeanLMError as error:
        return _fail(error)

    compliance = result.compliance
    print(f"submission: {result.output_dir} ({result.size_mb} MB, "
          f"{len(result.files)} files)")
    for check in compliance.checks:
        mark = "ok  " if check.passed else ("FAIL" if check.blocking else "warn")
        print(f"  [{mark}] {check.check_id} {check.label}")
        if not check.passed and check.detail:
            print(f"         {check.detail}")
    if not compliance.passed:
        print("\nNOT SUBMITTABLE -- fix the failures above", file=sys.stderr)
        return EXIT_NONCOMPLIANT
    print("\nsubmittable")
    print("  next: adtc-profiler run --submission {} --mode participant "
          "--output submission.json".format(result.output_dir))
    return EXIT_OK


def cmd_speed(args: argparse.Namespace) -> int:
    """Raw model speed from llama-bench, without LeanLM in the path."""
    from ..benchmarks.speed import measure, render

    profile = load_profile(args.profile)
    model_path = args.model or str(profile.model.get("path", ""))
    threads = args.threads or int((profile.memory or {}).get("n_threads") or 0) \
        or (os.cpu_count() or 4)
    try:
        result = measure(model_path, threads=threads,
                         prompt_tokens=args.prompt_tokens,
                         generated_tokens=args.generated_tokens)
    except LeanLMError as error:
        return _fail(error)
    if args.json:
        _print(result, True)
    else:
        print(render(result))
    return EXIT_OK


def cmd_candidates(args: argparse.Namespace) -> int:
    """Rank model candidates by what the rubric rewards."""
    from ..benchmarks.scoring import (Candidate, ScoringPolicy, rank_candidates,
                                      render_candidates)

    candidates = []
    for entry in args.candidate or ():
        try:
            name, accuracy, tps, peak = entry.split(":")
            candidates.append(Candidate(name, float(accuracy), float(tps),
                                        float(peak), measured=args.measured))
        except ValueError:
            print(f"could not parse '{entry}'", file=sys.stderr)
            print("  expected name:accuracy:tokens_per_second:peak_gb", file=sys.stderr)
            print("  e.g. qwen3.5-2b:0.75:16.0:1.7", file=sys.stderr)
            return EXIT_ERROR
    if not candidates:
        print("no candidates given", file=sys.stderr)
        print("  leanlm candidates -c qwen3.5-2b:0.75:16.0:1.7 "
              "-c qwen3.5-4b:0.83:8.0:3.0", file=sys.stderr)
        return EXIT_ERROR

    rows = rank_candidates(candidates, ScoringPolicy.load())
    if args.json:
        _print(rows, True)
    else:
        print(render_candidates(rows))
        best = rows[0]
        print(f"\n  highest scoring: {best['name']} at {best['total']}")
        if not best["measured"]:
            print("  these are estimates; measure with `leanlm accuracy` and "
                  "`leanlm bench` before committing")
    return EXIT_OK


def cmd_preflight(args: argparse.Namespace) -> int:
    """Which routes to a submittable measurement work from here."""
    from ..runtime.preflight import render, routes

    found = routes(args.submission)
    if args.json:
        _print([r.to_dict() for r in found], True)
    else:
        print(render(found))
    return EXIT_OK if any(r.ready and r.suitable_for_submission for r in found) \
        else EXIT_NONCOMPLIANT


def cmd_ccm(args: argparse.Namespace) -> int:
    report = verify_ccm()
    if args.json:
        _print(report.to_dict(), True)
    else:
        print(report.render())
        print(f"\n{'CCM complete' if report.ok else 'CCM INCOMPLETE'}")
    return EXIT_OK if report.ok else EXIT_ERROR


def cmd_profiles(args: argparse.Namespace) -> int:
    names = list_profiles()
    if args.show:
        profile = load_profile(args.show)
        _print(profile.to_dict() if args.json else
               "\n".join(f"  {k:18}: {v}" for k, v in profile.to_dict().items()),
               args.json)
        return EXIT_OK
    if args.json:
        _print(names, True)
        return EXIT_OK
    for name in names:
        profile = load_profile(name)
        print(f"  {profile.id:14} v{profile.version:8} [{profile.fingerprint}] "
              f"{profile.description}")
    return EXIT_OK


def cmd_verify(args: argparse.Namespace) -> int:
    """Trust chain: model file, runtime sources, offline posture."""
    profile = load_profile(args.profile)
    model_path = args.model or profile.model.get("path", "")
    payload: dict[str, Any] = {
        "model": verify_model_binding(
            model_path, str(profile.model.get("sha256", ""))).to_dict()
        if model_path else {"trusted": False, "checks": {"model_declared": False},
                            "details": {"hint": "no model path in the profile"}},
        "runtime": verify_runtime_integrity(profile.fingerprint).to_dict(),
        "ccm": {"ok": verify_ccm().ok},
    }
    if args.json:
        _print(payload, True)
    else:
        model = payload["model"]
        print(f"model   : {'TRUSTED' if model['trusted'] else 'NOT TRUSTED'}")
        for key, value in model["checks"].items():
            print(f"  {key:22}: {value}")
        for key, value in model["details"].items():
            print(f"  {key:22}: {value}")
        runtime = payload["runtime"]["details"]
        print(f"runtime : fingerprint {runtime['source_fingerprint']} "
              f"over {runtime['source_files']} files")
        print(f"          profile {runtime['profile_fingerprint']} on "
              f"{runtime['platform']}")
        print(f"ccm     : {'complete' if payload['ccm']['ok'] else 'INCOMPLETE'}")
    return EXIT_OK if payload["model"]["trusted"] else EXIT_ERROR


def cmd_serve(args: argparse.Namespace) -> int:
    from .workspace.server import serve
    try:
        return serve(profile=args.profile, host=args.host, port=args.port,
                     model_path=args.model, backend=args.backend,
                     open_browser=not args.no_browser)
    except LeanLMError as error:
        return _fail(error)


def cmd_evidence(args: argparse.Namespace) -> int:
    """Render the Performance Evidence Matrix from a results file."""
    path = Path(args.results) if args.results else \
        repo_root() / "benchmarks/results/latest.json"
    if not path.is_file():
        print(f"no results at {path}", file=sys.stderr)
        return EXIT_ERROR
    summary = json.loads(path.read_text(encoding="utf-8"))
    evidence = [PerformanceEvidence(**{k: (tuple(v) if isinstance(v, list) else v)
                                       for k, v in item.items()})
                for item in summary.get("evidence", [])]
    if args.json:
        _print([e.to_dict() for e in evidence], True)
    else:
        print(render_matrix(evidence))
    return EXIT_OK


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="leanlm",
        description="LeanLM -- offline LLM optimization layer for commodity laptops",
    )
    parser.add_argument("--version", action="version", version=f"LeanLM {__version__}")
    parser.add_argument("--profile", default="development", help="runtime profile id")
    parser.add_argument("--corpus", default=None, help="path to the corpus database")
    parser.add_argument("--model", default=None, help="override the model path")
    parser.add_argument("--backend", default=None,
                        help="auto | llama-cpp-python | llama-server | llama-cli | simulated")
    parser.add_argument("--json", action="store_true", help="machine readable output")
    parser.add_argument("--log-level", default=None,
                        choices=["debug", "info", "warning", "error"])
    sub = parser.add_subparsers(dest="command", required=True)

    doctor = sub.add_parser("doctor", help="check that this machine can run LeanLM")
    doctor.set_defaults(func=cmd_doctor)

    ingest = sub.add_parser("ingest", help="add documents to the local corpus")
    ingest.add_argument("paths", nargs="+", help="files or directories")
    ingest.add_argument("--replace", action="store_true",
                        help="empty the corpus first (use before measuring)")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="ask a question against the local corpus")
    ask.add_argument("question")
    ask.add_argument("--dry-run", action="store_true",
                     help="print the backend command instead of running it")
    ask.add_argument("--quiet", action="store_true",
                     help="do not stream the answer as it is generated")
    ask.add_argument("--allow-simulated", action="store_true",
                     help="permit running without a real model (results are not valid)")
    ask.set_defaults(func=cmd_ask)

    bench = sub.add_parser("bench", help="run the benchmark campaign")
    bench.add_argument("--scenario", nargs="*", help="scenario ids, e.g. S1 S4")
    bench.add_argument("--repeat", type=int, default=None)
    bench.add_argument("--warmup", type=int, default=None)
    bench.add_argument("--output", default=None)
    bench.add_argument("--baseline", default=None)
    bench.add_argument("--save-baseline", action="store_true")
    bench.add_argument("--allow-simulated", action="store_true")
    bench.add_argument("--with-profile", action="store_true",
                       help="also profile one request and embed the report (SUB-010)")
    bench.add_argument("--quiet", action="store_true")
    bench.set_defaults(func=cmd_bench)

    baseline = sub.add_parser(
        "baseline", help="run, inspect or compare the no-LeanLM baseline")
    baseline.add_argument("--run", action="store_true",
                          help="measure the naive path: whole corpus in the prompt")
    baseline.add_argument("--path", default=None)
    baseline.add_argument("--compare", default=None, help="results file to compare")
    baseline.add_argument("--repeat", type=int, default=None)
    baseline.add_argument("--allow-simulated", action="store_true")
    baseline.set_defaults(func=cmd_baseline)

    profile = sub.add_parser("profile", help="profile a request (where the time goes)")
    profile.add_argument("question")
    profile.add_argument("--repeat", type=int, default=3)
    profile.add_argument("--top", type=int, default=20)
    profile.add_argument("--output", default=None)
    profile.add_argument("--allow-simulated", action="store_true")
    profile.set_defaults(func=cmd_profile)

    submission = sub.add_parser(
        "submission", help="build the ADTC-template submission package")
    submission.add_argument("--output", default="dist/submission")
    submission.add_argument("--team-id", default="", help="as registered on the portal")
    submission.add_argument("--name", default="", help="submitter full name")
    submission.add_argument("--email", default="")
    submission.add_argument("--github", default="", help="submitter github handle")
    submission.add_argument("--domain", default="corporate_enterprise")
    submission.add_argument("--language", nargs="*", help="BCP-47 codes, e.g. en fr")
    submission.add_argument("--african-use-case", action="store_true",
                            help="claim the African Use Case Bonus")
    submission.add_argument("--discipline", default="",
                            help="the discipline the model serves")
    submission.add_argument("--pairing", default="",
                            help="how the model serves that discipline")
    submission.add_argument("--prompt", action="append",
                            help="a test prompt; give this exactly twice")
    submission.add_argument("--model-path", default=None,
                            help="path inside the package, e.g. model/my.gguf")
    submission.add_argument("--results", default=None)
    submission.add_argument("--accuracy", default=None)
    submission.add_argument("--repository", default="")
    submission.add_argument("--commit", default="")
    submission.add_argument("--profiler", default=None,
                            help="path to the official profiler's submission.json")
    submission.add_argument("--allow-missing-results", action="store_true")

    devpost = sub.add_parser(
        "devpost", help="the scores the submission form asks for")
    devpost.add_argument("--profiler", default=None)
    devpost.set_defaults(func=cmd_devpost)
    submission.set_defaults(func=cmd_submission)

    accuracy = sub.add_parser("accuracy", help="measure accuracy against ground truth")
    accuracy.add_argument("--evaluation", default=None)
    accuracy.add_argument("--output", default=None)
    accuracy.add_argument("--allow-simulated", action="store_true")
    accuracy.set_defaults(func=cmd_accuracy)

    score = sub.add_parser("score", help="compute the ADTC score from the results")
    score.add_argument("--results", default=None)
    score.add_argument("--accuracy", default=None)
    score.set_defaults(func=cmd_score)

    speed = sub.add_parser(
        "speed", help="measure raw model speed with llama-bench")
    speed.add_argument("--threads", type=int, default=None)
    speed.add_argument("--prompt-tokens", type=int, default=128)
    speed.add_argument("--generated-tokens", type=int, default=32)
    speed.set_defaults(func=cmd_speed)

    candidates = sub.add_parser(
        "candidates", help="rank model candidates against the rubric")
    candidates.add_argument("-c", "--candidate", action="append",
                            help="name:accuracy:tokens_per_second:peak_gb")
    candidates.add_argument("--measured", action="store_true",
                            help="these are measured, not estimated")
    candidates.set_defaults(func=cmd_candidates)

    preflight = sub.add_parser(
        "preflight", help="which routes can produce a submittable measurement")
    preflight.add_argument("--submission", default="dist/submission")
    preflight.set_defaults(func=cmd_preflight)

    ccm = sub.add_parser("ccm", help="verify the capability-to-code mapping")
    ccm.set_defaults(func=cmd_ccm)

    profiles = sub.add_parser("profiles", help="list or show runtime profiles")
    profiles.add_argument("--show", default=None)
    profiles.set_defaults(func=cmd_profiles)

    verify = sub.add_parser("verify", help="verify the trust chain")
    verify.set_defaults(func=cmd_verify)

    evidence = sub.add_parser("evidence", help="render the performance evidence matrix")
    evidence.add_argument("--results", default=None)
    evidence.set_defaults(func=cmd_evidence)

    serve = sub.add_parser("serve", help="start the local workspace")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8770)
    serve.add_argument("--no-browser", action="store_true")
    serve.set_defaults(func=cmd_serve)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure(level=args.log_level or "warning")
    try:
        return int(args.func(args))
    except LeanLMError as error:
        return _fail(error)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
