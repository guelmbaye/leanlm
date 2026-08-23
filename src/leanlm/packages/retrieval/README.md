# packages/retrieval -- CAP-004 Evidence Retrieval

| Field | Value |
|---|---|
| Capability | CAP-004 Evidence Retrieval |
| Requirement | FR-04 |
| Stage | `evidence_retrieval` (DIC phase 5) |
| Input | IEC with `optimized_context` |
| Output | IEC with `evidence` (labelled `S1..Sn`, each traceable to its unit) |
| Tests | `tests/capability/test_retrieval.py` |
| Benchmark | `packages/retrieval/benchmarks.py` |

## Why BM25 and not embeddings

An embedding model is a second model in RAM, a second load time and a second
artifact to ship. On a machine where the LLM already owns most of the memory
budget, the marginal accuracy of dense retrieval does not pay for its footprint.
BM25 over *structured* units -- boosted by section titles, exact phrases and
numeric hints -- is dependency free, instant and deterministic.

Recorded as **EDB-004** with its trade-offs and the conditions that would
reverse it (a corpus where paraphrase dominates keyword overlap).

## Selection

Greedy maximal marginal relevance under the token budget: score, subtract a
similarity penalty against what is already chosen, cap passages per document,
stop at the budget. Ties break on `unit_id`, so two runs on two machines select
the same passages in the same order.
