# packages/context -- CAP-003 Context Optimization

| Field | Value |
|---|---|
| Capability | CAP-003 Context Optimization |
| Requirement | FR-03 |
| Stage | `context_optimization` (DIC phase 4) |
| Input | IEC with `context_package` + `resources` |
| Output | IEC with `intent`, `optimized_context` (units + `ContextBudget`) |
| Tests | `tests/capability/test_context.py` |
| Benchmark | `packages/context/benchmarks.py` |

## The budget is a target, not a constant

`ContextBudgeter` multiplies the model's usable context by four factors -- RAM,
thermal, CPU, intent -- and records which one was limiting. That record is what
makes a slow run explainable after the fact: `limiting_factor: thermal` in the
metrics is a measurement, not a guess.

Same observation in, same budget out (P4). The factors are stored in the
`ContextBudget` artifact so a benchmark can replay the decision.

## Optimization operations

`drop_boilerplate`, `drop_exact_duplicates`, `drop_near_duplicates`,
`merge_adjacent`. Each one is counted and reported; the resulting Context
Compression Ratio is an EBPB Tier-3 metric and has to survive review.

Near-duplicate detection buckets units by their rarest content word instead of
comparing all pairs: quadratic comparison would dominate ingestion on a real
corpus.
