# ADTC 2026 — what the published rules say, and where we stand

Read from `adtc-2026.devpost.com/rules` and `/resources` on 18 August 2026.
Everything below is either confirmed against that source or flagged as an open
item. **Gate 1 deadline: 25 August 2026.**

## The rubric, as published

| Category | Weight | What it actually measures |
|---|---|---|
| Model Accuracy & Quality | 50% | multiple-choice benchmarks **and** qualitative evaluation, including prompt accuracy and **documentation quality** |
| Model Throughput | 30% | relative to the **maximum observed** tokens/s across submissions |
| Model Efficiency | 20% | lower RAM utilisation relative to the **maximum memory budget** |
| African Use Case Bonus | +10 pts | applicability to a real African use case |
| Thermal | −10 pts | core/package temperature above **85 °C**, or throttling flagged |
| OOM / sandbox crash | — | **disqualification** |

Our 50/30/20 assumption was right. Three details were not, and
`configs/scoring.yaml` is corrected:

1. **Throughput is relative to the field, not to a target.** We were scoring
   against an invented 12 tok/s. No entrant can know the field maximum before
   judging, so `leanlm score` now reports throughput raw and excludes 30% of the
   weight from the total, saying so. Set `field_max_tokens_per_second` to model
   a scenario; never to flatter a report.
2. **Efficiency is the unused share of the budget** — `1 − peak/8192` — not a
   band between two numbers we chose.
3. **The thermal penalty is a step at 85 °C**, a flat 10 points, and also fires
   on flagged throttling. We had a ramp from 78 °C. Throttling detection is
   still missing (open item below).

## Two things that change the shape of the work

**Accuracy is graded partly on multiple-choice benchmarks.** Our evaluation set
is free-text: the answer must contain the figure the corpus states. That
measures something real and it is not what will be scored. `evaluation.json`
needs an MCQ section, and the accuracy evaluator needs to score option
selection.

**Documentation quality sits inside the 50% accuracy weight**, not beside it.
The eight ADRs, the Engineering Decision Book and the traceability matrix are
not supporting material — they are scored material.

## Official tooling we are not yet using

| Resource | Status |
|---|---|
| [`adtc-2026-submission-template`](https://github.com/Africa-Deep-Tech-Foundation/adtc-2026-submission-template) | **not yet cloned.** `configs/submission.yaml` lists files we guessed. The template is authoritative and SUB-001 should follow it. |
| [`adtc-profiler`](https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler) | **not yet used.** `leanlm profile` is cProfile — useful to us, and not the profiler the organisers published. SUB-008 currently accepts ours. |

Both are one clone away and both are higher priority than any new feature.

## The evaluation machine is Linux

Every resource points at Ubuntu/Debian: the llama.cpp build guide, the 8 GB
tuning discussion, `lm-sensors` for the thermal threshold, and "sandbox
execution crash" in the disqualification rule. Develop on whatever you like;
**measure on Ubuntu**, with `lm-sensors` installed, or the thermal component is
an unmeasured risk rather than a zero.

## Language

Nothing in the rules, the resources or the template is in French. The report,
the repository and the two-minute video are read in English, and the
multiple-choice benchmark is far more likely to be English than not.

The African Use Case Bonus is not a language bonus: the resources point at
[Masakhane](https://github.com/masakhane-io/masakhane-nlp) and at Aya's African
language support, which is about African languages, not about French. French is
an African working language across many eligible countries — including Morocco —
so it belongs in the demonstration as evidence of multilingual capability, not
as the default.

**Decision: English is the default; French stays as a demonstrated capability.**
See `docs/LANGUAGE.md` for what that means file by file.

## Open items, in order

1. Clone the submission template; make `configs/submission.yaml` match it.
2. Clone the official profiler; attach its report, not ours.
3. Add multiple-choice probes to the evaluation set.
4. Translate the corpus, the evaluation set and the report to English.
5. Detect thermal throttling, not only absolute temperature.
6. Run the real measurement on Ubuntu with a real GGUF model.
