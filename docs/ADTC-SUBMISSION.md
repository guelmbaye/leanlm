# What the official template and profiler actually require

Read from the two official repositories, not inferred. This corrects several
things this project had guessed, including one that changes what the work is
*for*.

## The finding that matters most

**The profiler never runs LeanLM.** It reads `_runtime.model_path` from
`metadata.json`, loads the GGUF directly, and measures:

| Stage | How | What it touches |
|---|---|---|
| throughput | `llama-bench -p 512 -n 128` | the model, synthetic prompts |
| memory | peak and steady-state RSS | the model |
| thermal | core temperature, throttling | the machine |
| accuracy | lm-eval `arc_easy`, in-process via llama-cpp-python | the model |

`test_prompts` appears nowhere in the profiler source. The two prompts you
declare, plus two hidden ones the organisers add, are read by **judges**.

So the numeric 70% — throughput and efficiency — and the lm-eval half of
accuracy are properties of the **model and the machine**. A retrieval layer,
however good, does not move them. This has to be said plainly because the
opposite assumption would waste the remaining days.

### Where LeanLM does count

1. **Model selection, evidenced.** The score pulls in two directions: `arc_easy`
   rewards a larger model, while throughput and efficiency reward a smaller one.
   The optimum is empirical, and a benchmark harness that runs candidates under
   identical conditions and reports dispersion is exactly the tool for finding
   it — and for defending the choice afterwards.
2. **`REPORT.md`**, read by judges and by an LLM audit system. Documentation
   quality sits inside the 50% accuracy weight.
3. **The four prompts**, judged qualitatively.
4. **The African Use Case Bonus** and the cross-disciplinary pairing.

## The exact scoring formula

From the profiler's own README, now implemented verbatim in
`configs/scoring.yaml`:

```
S_total = 0.50·S_acc + 0.30·S_perf + 0.20·S_eff − P_thermal

S_perf = min(TPS / 15.0, 1.0) × 100
S_eff  = max(0, (7.0 − peak_rss_gb) / 7.0) × 100
P_thermal = 10 points if the CPU throttles or core temp ≥ 85 °C
```

Two corrections to what we had:

- **`TPS_REFERENCE = 15.0`**, a fixed reference. The Devpost rules say
  "relative to the maximum observed tokens per second"; the profiler that
  computes the score uses 15.0. The implementation wins. Every tok/s below 15
  costs 2 points of the final score.
- **`RAM_LIMIT_GB = 7.0`**, not 8. Scoring against 8 GB overstated every
  efficiency figure we had reported.

## The required submission structure

Far simpler than the one we invented:

```
metadata.json        team, model, exactly 2 test prompts
download_model.sh    idempotent, no credentials, path must match _runtime.model_path
REPORT.md            problem, design decisions, constraints, benchmarks
model/               downloaded, never committed
.gitignore           must exclude *.gguf and model/
```

`model_manifest.json`, `runtime_profile.yaml` and `evidence/` — all of which our
packager produced — are not part of the template. `configs/submission.yaml` and
the compliance checks need to follow this list instead.

### metadata.json is a strict schema

`domain` is an enum: ours is **`corporate_enterprise`** (not "Corporate /
Enterprise"). `budget_laptop_claim` must be `true`. `african_alpha_claim` is
`true` only if claiming the bonus. `model.runtime` must be `llama.cpp`.
`test_prompts` must contain **exactly two**, and the organisers add two hidden
prompts in the same domain to test for overfitting.

## Local verification before submitting

```bash
python3 -m pip install "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"
bash download_model.sh
adtc-profiler run --submission . --mode participant --output submission.json
```

A valid run reports `"measured_on": "participant_laptop"`. Requires
`llama-bench` on `PATH` and Python ≥ 3.11.

**Your numbers will be re-measured in an audit sandbox and compared to yours.**
Peak RSS tolerates ±15%, throughput ±25%; beyond 50% the comparison *fails*, as
do zero or missing values and a mismatched environment. Measure honestly on
hardware close to the 4 vCPU / 8 GB profile.

## Licensing note

Both official repositories are **GPL v3**. LeanLM is Apache-2.0. Apache-2.0 code
may be combined into a GPLv3 work, so a submission repository forked from the
template is effectively GPLv3. Decide deliberately whether LeanLM's own
repository stays separate and Apache-2.0, with the submission repository holding
only the template files and the report.
