# packages/validation -- CAP-006 Response Validation

| Field | Value |
|---|---|
| Capability | CAP-006 Response Validation |
| Requirement | FR-06 |
| Stage | `response_validation` (DIC phase 8) |
| Input | IEC with `response` + `evidence` |
| Output | IEC with `validation` (a `ValidationReport`) |
| Tests | `tests/capability/test_validation.py` |
| Benchmark | `packages/validation/benchmarks.py` |

## What it measures

- **grounding rate** -- share of answer sentences whose 3-gram overlap with the
  excerpts clears the threshold;
- **citations** -- labels the answer cites, and any label it invented;
- **evidence coverage** -- share of supplied excerpts actually used;
- **confidence** -- `high / medium / low / insufficient`, derived from those
  observations. Never a probability: an invented number would look precise and
  mean nothing (IIB s.13).

## The empty-evidence case

When retrieval found nothing, the only answer that passes validation is an
explicit statement of insufficiency. An answer with content and no evidence
means the model spoke from its own weights, and that is reported, not smoothed
over.

## Why not an LLM judge

A second model doubles the memory budget on a machine that has none to spare.
The n-gram proxy is cheaper, deterministic and auditable. Its limitation --
heavy paraphrase scores as ungrounded -- is recorded in **EDB-006** together
with the conditions that would justify revisiting it.
