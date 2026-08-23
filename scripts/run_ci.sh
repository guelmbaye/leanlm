#!/usr/bin/env bash
# The competition's rules, executed. Same script locally and in CI.
set -euo pipefail
cd "$(dirname "$0")/.."

fail=0
step() { printf '\n=== %s ===\n' "$1"; }

step "capability-to-code mapping"
python3 -m leanlm.apps.cli ccm || fail=1

step "test suite"
python3 -m pytest -q || fail=1

step "runtime profiles"
python3 -m leanlm.apps.cli profiles || fail=1

step "environment"
python3 -m leanlm.apps.cli doctor || true   # a missing model is not a CI failure

step "corpus"
# The benchmark needs a corpus. Relying on one left over from a previous run is
# how a green CI stops meaning anything.
python3 -m leanlm.apps.cli ingest datasets/enterprise_en || fail=1

step "accuracy against ground truth"
# The heaviest criterion in the rubric. Measured before throughput, because a
# fast wrong answer scores nothing.
python3 -m leanlm.apps.cli --profile benchmark accuracy --allow-simulated || fail=1

step "benchmark smoke run"
python3 -m leanlm.apps.cli --profile benchmark bench --scenario S1 S6 \
  --repeat 2 --warmup 0 --allow-simulated --quiet || fail=1

step "provisional score"
python3 -m leanlm.apps.cli score || true

if [[ "${fail}" -ne 0 ]]; then
  printf '\nCI FAILED\n'
  exit 1
fi
printf '\nCI PASSED\n'
