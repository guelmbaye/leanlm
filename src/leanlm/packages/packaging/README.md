# packages/packaging -- CAP-008 Submission Packaging

| Field | Value |
|---|---|
| Capability | CAP-008 Submission Packaging |
| Requirement | FR-08 |
| Pipeline | delivery (`submission_build`, `compliance_check`) |
| Input | `SubmissionRequest` (model manifest, profile, evidence, commit) |
| Output | a clean directory + `MANIFEST.sha256` + `compliance_report.json` |
| Tests | `tests/submission/test_packaging.py` |
| Benchmark | `packages/packaging/benchmarks.py` |

## Competition-as-Code

The deliverable is a pure function of the repository state and the evidence the
harness produced. Eleven checks run before anything is handed over, and ten of
them are blocking:

`SUB-001` required files - `SUB-002` no forbidden file (including any `.gguf`)
- `SUB-003` GGUF model declared - `SUB-004` checksum declared - `SUB-005`
download URL declared - `SUB-006` **results came from a real model** -
`SUB-007` benchmark evidence attached - `SUB-008` profiler report attached -
`SUB-009` domain declared - `SUB-010` size - `SUB-011` commit recorded.

`SUB-006` is the one that matters most: it is what stops a green pipeline from
shipping numbers a model never produced.

## The required-file list is configuration

`configs/submission.yaml` owns it. The official template is the authority; hard
coding its contents here would guarantee a mismatch on submission day.
