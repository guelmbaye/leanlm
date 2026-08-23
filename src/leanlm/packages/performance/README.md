# packages/performance -- CAP-007 Performance Engineering

| Field | Value |
|---|---|
| Capability | CAP-007 Performance Engineering |
| Requirement | FR-07 |
| Stage | `metrics_finalization` (DIC phase 9) |
| Input | the whole IEC |
| Output | IEC with `metrics` (a `MetricsSnapshot`) |
| Tests | `tests/capability/test_performance.py` |
| Benchmark | `packages/performance/benchmarks.py` |

## Sampling without dependencies

`psutil` when available, `/proc/meminfo`, `/proc/self/status` and
`/sys/class/thermal` otherwise. A commodity laptop with no working package
index must still produce the memory and thermal numbers the challenge scores.

`thermal_source` is always recorded. "No sensor" and "41 C" are different
facts, and a thermal figure derived from an absent sensor would be a
fabrication -- so the absence is reported instead.

## Tier separation

Tier 2 (runtime) and Tier 3 (intelligence) metrics live in the same snapshot but
are never presented as ADTC results. The official profiler stays the reference
(R4); these numbers exist to explain *why* a profiler result moved.
