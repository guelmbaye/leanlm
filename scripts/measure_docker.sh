#!/usr/bin/env bash
# Produce a submittable measurement inside a memory-capped container.
#
# For a machine with more RAM than the 8 GB profile. Nothing swaps on a generous
# machine, so the failure that disqualifies -- an out-of-memory kill on the
# target -- cannot be observed unless the run is constrained. `--memory=7.5g`
# constrains it.
#
# What this cannot do is make slow cores fast. Capping resources reproduces the
# *limit*, never the speed.
set -euo pipefail

SUBMISSION="${1:-dist/submission}"
OUTPUT="${2:-submission.json}"
MODE="${MODE:-participant}"
MEMORY="${MEMORY:-7.5g}"
CPUS="${CPUS:-4}"
IMAGE="${IMAGE:-adtc-profiler:latest}"
PROFILER_REF="${PROFILER_REF:-https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git}"

die() { echo "error: $*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "docker is not installed"
[[ -d "${SUBMISSION}" ]] || die "no submission directory at ${SUBMISSION}
  -> run: leanlm submission --output ${SUBMISSION} ..."
[[ -f "${SUBMISSION}/metadata.json" ]] || die "${SUBMISSION} has no metadata.json"

available_cpus="$(nproc 2>/dev/null || echo 1)"
if (( available_cpus < CPUS )); then
  echo "warning: asking for ${CPUS} CPUs on a host with ${available_cpus}." >&2
  echo "         Docker will not invent cores; throughput will read low and the" >&2
  echo "         audit comparison fails beyond 50% variance." >&2
fi

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "  building ${IMAGE} from the official profiler"
  workdir="$(mktemp -d)"
  git clone --depth 1 "${PROFILER_REF}" "${workdir}/adtc-profiler"
  docker build -t "${IMAGE}" "${workdir}/adtc-profiler"
fi

# The weights are fetched before profiling starts; the rules permit that window
# and nothing after it.
if [[ ! -f "${SUBMISSION}/$(python3 -c "import json,sys;print(json.load(open('${SUBMISSION}/metadata.json'))['_runtime']['model_path'])")" ]]; then
  echo "  fetching weights"
  ( cd "${SUBMISSION}" && bash download_model.sh )
fi

mkdir -p "$(dirname "${OUTPUT}")" 2>/dev/null || true
artifacts="$(cd "$(dirname "${OUTPUT}")" && pwd)"

echo "  profiling: memory=${MEMORY} cpus=${CPUS} mode=${MODE}"
docker run --rm \
  --memory="${MEMORY}" \
  --cpus="${CPUS}" \
  -v "$(cd "${SUBMISSION}" && pwd):/submission:ro" \
  -v "${artifacts}:/artifacts" \
  "${IMAGE}" run \
  --submission /submission \
  --mode "${MODE}" \
  --output "/artifacts/$(basename "${OUTPUT}")"

echo "  written: ${OUTPUT}"
python3 - "${OUTPUT}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
env = report.get("environment", {})
print(f"  measured_on : {env.get('measured_on')}")
print(f"  cpu         : {env.get('cpu_model')}")
print(f"  ram         : {env.get('ram_gb')} GB")
thermal = report.get("thermal", {})
if thermal.get("core_temp_c_peak") is None:
    print("  thermal     : unmeasured (no sensor inside the container). The audit "
          "VM has the same limitation.")
PY
