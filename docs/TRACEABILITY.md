# Traceability matrix

Requirement → capability → code → test → evidence. Every row is checkable by
`leanlm ccm` and by the test suite; nothing here is a claim without a file
behind it.

## Functional requirements

| Req | Description | Capability | Code | Test |
|---|---|---|---|---|
| FR-01 | Import and normalize local documents | CAP-001 | `packages/ingestion/services.py` | `tests/capability/test_ingestion.py` |
| FR-02 | Detect structure and segment into units | CAP-002 | `packages/understanding/services.py` | `tests/capability/test_understanding.py` |
| FR-03 | Budget and compact the context | CAP-003 | `packages/context/services.py` | `tests/capability/test_context.py` |
| FR-04 | Select evidence for a question | CAP-004 | `packages/retrieval/services.py` | `tests/capability/test_retrieval.py` |
| FR-05 | Assemble the prompt and run inference | CAP-005 | `packages/inference/` | `tests/capability/test_inference.py` |
| FR-06 | Validate grounding and citations | CAP-006 | `packages/validation/services.py` | `tests/capability/test_validation.py` |
| FR-07 | Measure runtime and intelligence metrics | CAP-007 | `packages/performance/services.py` | `tests/capability/test_performance.py` |
| FR-08 | Build a compliant submission package | CAP-008 | `packages/packaging/services.py` | `tests/capability/test_packaging.py` |

## Non-functional requirements

| Req | Description | Enforced by | Test |
|---|---|---|---|
| NFR-01 | Fully offline operation | `trust/offline_guard.py` | `tests/integration/test_runtime.py::TestOfflineGuard` |
| NFR-02 | Runs within an 8 GB envelope | `packages/context` budgeting, `ResourceSampler` peaks | benchmark scenarios S1–S3 |
| NFR-03 | Reproducible measurements | frozen profiles, temperature 0, deterministic ids | `TestProfiles::test_measurement_profiles_are_deterministic` |
| NFR-04 | Traceable answers | `SelectedEvidence`, citation checking | `tests/pipeline::TestGroundedAnswer` |
| NFR-05 | Failures are observable | `ErrorRecord`, DIC-07 | `tests/pipeline::TestFailurePath` |
| NFR-06 | No user content in telemetry | `IEC.to_dict(include_text=False)` | `TestIEC::test_archival_form_can_exclude_user_text` |
| NFR-07 | Installable without an index | zero mandatory dependencies | ADR-008, `leanlm doctor` |
| NFR-08 | Submission is verifiable | 11 compliance checks | `tests/submission/test_delivery.py` |

## Comparative evidence

| Claim | Produced by | Asserted in |
|---|---|---|
| The layer sends far fewer tokens | `benchmarks/naive.py` | `test_baseline_comparison.py::test_leanlm_sends_far_fewer_prompt_tokens` |
| A corpus larger than the window forces truncation | `benchmarks/naive.py` | `::test_the_corpus_does_not_fit` |
| LeanLM answers what truncation discarded | `scripts/make_corpus.py` probe | `::test_leanlm_finds_the_answer_the_baseline_never_saw` |
| The corpus contains no answer key | `scripts/make_corpus.py` | `::test_the_corpus_documentation_is_not_part_of_the_corpus` |
| The baseline is not handicapped | `naive.py::BASELINE_VALIDATION` | `::test_the_baseline_is_not_scored_on_an_instruction_it_never_got` |
| Regression and naive baselines never mix | `baseline.py::is_regression_baseline` | `test_delivery.py::TestTwoKindsOfBaseline` |
| Where the time goes | `benchmarks/profiler.py` | EDB-019 measurement |
| PDF extraction and scanned-PDF refusal | `packages/ingestion` | `test_ingestion_pdf.py` |

## Deterministic Inference Contract

| Rule | Guarantee | Asserted in |
|---|---|---|
| DIC-01 | Fixed stage lifecycle | `runtime/dic.py` + `tests/pipeline::test_the_ten_phases_run_in_the_declared_order` |
| DIC-02 | Stable configuration | profile fingerprint recorded on every run |
| DIC-03 | Metrics always collected | `_finalize_on_failure` runs CAP-007 even after a failure |
| DIC-04 | Traceable evidence | every `SelectedEvidence` carries `unit_id` and `document_name` |
| DIC-05 | Resource safety | prompt tokens ≤ usable window, asserted per run |
| DIC-06 | Offline integrity | `OfflineGuard` violations recorded in the IEC |
| DIC-07 | Transparent failure | a failed stage without an error record is a violation |
