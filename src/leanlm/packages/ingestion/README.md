# packages/ingestion -- CAP-001 Document Ingestion

| Field | Value |
|---|---|
| Capability | CAP-001 Document Ingestion |
| Requirement | FR-01 |
| DDPB stages | Import, Validation, Normalization |
| Input | file path (PDF / TXT / Markdown) |
| Output | `DocumentPipelineState` with `descriptor`, `raw`, `normalized_text` |
| Tests | `tests/capability/test_ingestion.py` |
| Benchmark | `packages/ingestion/benchmarks.py` |

## Guarantees

- The source file is opened read-only and never rewritten (DP-02).
- Normalization fixes encoding artefacts only; no word is altered (DP-04).
- A document that yields no usable text is rejected with `ING-002` rather than
  ingested empty -- an empty corpus produces confident, ungrounded answers.
- Every descriptor carries the SHA-256 of the source file, which anchors the
  trust chain described in TILAB s.10.

## Extension points

Add a format by extending `policies.SUPPORTED_EXTENSIONS` and
`services.extract_text`. No other package changes.
