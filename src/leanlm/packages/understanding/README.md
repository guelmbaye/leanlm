# packages/understanding -- CAP-002 Document Understanding

| Field | Value |
|---|---|
| Capability | CAP-002 Document Understanding |
| Requirement | FR-02 |
| DDPB stages | Structure detection, Segmentation, Metadata extraction |
| Input | `DocumentPipelineState` with `normalized_text` |
| Output | `StructuredDocument` (semantic units + section paths) |
| Tests | `tests/capability/test_understanding.py` |
| Benchmark | `packages/understanding/benchmarks.py` |

## Why structure before retrieval

A semantic unit carries its section path. Retrieval can then boost a passage
whose heading matches the question without spending a single token or model
call. This is the cheapest relevance signal available on an 8 GB laptop, and it
is only available because structure is detected before segmentation (DP-03).

## Segmentation contract

- units never exceed `max_tokens`; oversized paragraphs split on sentence ends;
- units below `min_tokens` are folded into the previous unit, so no retrieval
  slot is wasted on a fragment;
- every unit is traceable: `document_id`, `order`, `char_start/char_end`.
