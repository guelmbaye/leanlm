#!/usr/bin/env bash
# Provision a clean Ubuntu machine to produce a submittable measurement.
#
# Written for a 4 vCPU / 8 GB cloud instance, which is the closest available
# match to the ADTC standard laptop profile -- and, since the audit itself runs
# in a cloud VM, closer to the environment your numbers will be compared against
# than a development laptop is.
#
# Everything here happens before profiling begins. Once the profiler starts, no
# outbound request is permitted.
#
#   ./scripts/provision_ubuntu.sh            # install everything
#   ./scripts/provision_ubuntu.sh --check    # report what is present
set -euo pipefail

LLAMA_REF="${LLAMA_REF:-master}"
PREFIX="${PREFIX:-$HOME/.local}"

step() { printf '\n=== %s ===\n' "$1"; }
have() { command -v "$1" >/dev/null 2>&1; }

check() {
  printf 'llama-bench : %s\n' "$(have llama-bench && command -v llama-bench || echo MISSING)"
  printf 'python      : %s\n' "$(python3 --version 2>&1)"
  printf 'profiler    : %s\n' "$(have adtc-profiler && adtc-profiler --version 2>/dev/null || echo MISSING)"
  printf 'sensors     : %s\n' "$(have sensors && echo present || echo 'MISSING (cloud VMs expose no CPU temperature; the audit VM has the same limitation, so thermal is unmeasured on both sides and no penalty applies)')"
  printf 'cpus        : %s\n' "$(nproc)"
  printf 'memory      : %s\n' "$(free -g | awk '/^Mem:/ {print $2" GB"}')"
}

if [[ "${1:-}" == "--check" ]]; then check; exit 0; fi

step "system packages"
sudo apt-get update -qq
# build-essential and cmake build llama.cpp; python3-dev is needed because the
# profiler's accuracy stage compiles llama-cpp-python from source.
sudo apt-get install -y -qq \
  build-essential cmake git curl ccache \
  python3 python3-pip python3-venv python3-dev \
  lm-sensors

step "python version"
python3 -c 'import sys; assert sys.version_info >= (3, 11), "the profiler needs Python 3.11+"; print(sys.version.split()[0])'

step "llama.cpp"
# llama-bench is what the profiler shells out to for throughput. It is not a
# Python package and cannot be pip-installed.
if have llama-bench; then
  echo "  already on PATH: $(command -v llama-bench)"
else
  workdir="$(mktemp -d)"
  git clone --depth 1 --branch "${LLAMA_REF}" https://github.com/ggml-org/llama.cpp "${workdir}/llama.cpp"
  cmake -S "${workdir}/llama.cpp" -B "${workdir}/build" -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
  cmake --build "${workdir}/build" --config Release -j "$(nproc)"
  mkdir -p "${PREFIX}/bin"
  install -m 0755 "${workdir}/build/bin/llama-bench" "${PREFIX}/bin/llama-bench"
  install -m 0755 "${workdir}/build/bin/llama-cli" "${PREFIX}/bin/llama-cli" 2>/dev/null || true
  install -m 0755 "${workdir}/build/bin/llama-server" "${PREFIX}/bin/llama-server" 2>/dev/null || true
  echo "  installed into ${PREFIX}/bin"
  case ":${PATH}:" in
    *":${PREFIX}/bin:"*) ;;
    *) echo "  add to PATH:  export PATH=\"${PREFIX}/bin:\$PATH\"" ;;
  esac
fi

step "adtc-profiler"
# Compiles llama-cpp-python for the accuracy stage; expect several minutes.
python3 -m pip install --quiet --upgrade pip
python3 -m pip install --quiet "git+https://github.com/Africa-Deep-Tech-Foundation/adtc-profiler.git"

step "sensors"
# Harmless on a VM, where no thermal device exists to detect.
sudo sensors-detect --auto >/dev/null 2>&1 || true

step "ready"
check

cat <<'NEXT'

Next, from your submission directory:

  bash download_model.sh
  adtc-profiler run --submission . --mode participant --output submission.json

A valid run reports "measured_on": "participant_laptop". The environment block
records the real cpu_model and ram_gb, so where you measured is visible to the
judges either way -- say so in REPORT.md rather than leaving them to notice.
NEXT
