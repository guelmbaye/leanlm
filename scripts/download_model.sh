#!/usr/bin/env bash
# Fetch the GGUF model LeanLM runs on.
#
# This is the only script in the repository that touches the network, and it is
# never called by the runtime. Once it has run, LeanLM works with the network
# unplugged -- which is the point of the product.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"

# Derived from the URL unless MODEL_FILE is set. A fixed default is anonymous:
# a second candidate silently overwrites the first, and measurements already
# attributed to "the model" become wrong with nothing to signal it.
if [[ -z "${MODEL_FILE:-}" ]]; then
  derived="${MODEL_URL%%[?#]*}"
  derived="${derived##*/}"
  case "${derived,,}" in
    *.gguf) MODEL_FILE="${derived}" ;;
    *) echo "  note: no .gguf filename in the URL; using model.gguf"
       MODEL_FILE="model.gguf" ;;
  esac
fi
MODEL_URL="${MODEL_URL:-}"
MODEL_SHA256="${MODEL_SHA256:-}"

TARGET="${MODEL_DIR}/${MODEL_FILE}"

usage() {
  cat <<'USAGE'
Usage: MODEL_URL=<url> [MODEL_SHA256=<sha>] scripts/download_model.sh

On Windows, use scripts/download_model.ps1 instead -- same checks, same refusals.

Environment:
  MODEL_URL      direct URL to a .gguf file (required)
  MODEL_SHA256   expected SHA-256; the download is rejected if it does not match
  MODEL_DIR      destination directory (default: models)
  MODEL_FILE     destination filename (default: derived from the URL, so that
                 two candidates cannot overwrite each other)

Recommended for an 8 GB laptop: a 3-4B parameter instruct model quantized to
Q4_K_M, which lands around 2.2-2.6 GB on disk. Record the exact URL and
checksum in configs/runtime/competition.yaml once chosen: a model that cannot
be re-fetched byte-for-byte cannot support a reproducible measurement.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then usage; exit 0; fi

if [[ -z "${MODEL_URL}" ]]; then
  echo "error: MODEL_URL is not set" >&2
  echo >&2
  usage >&2
  exit 1
fi

mkdir -p "${MODEL_DIR}"

if [[ -f "${TARGET}" ]]; then
  echo "  ${TARGET} already exists; verifying instead of re-downloading"
else
  echo "  downloading ${MODEL_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${TARGET}.part" "${MODEL_URL}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${TARGET}.part" "${MODEL_URL}"
  else
    echo "error: neither curl nor wget is available" >&2
    exit 1
  fi
  mv "${TARGET}.part" "${TARGET}"
fi

# A GGUF file starts with the four bytes "GGUF". An HTML error page does not.
MAGIC=$(head -c 4 "${TARGET}" || true)
if [[ "${MAGIC}" != "GGUF" ]]; then
  echo "error: ${TARGET} is not a GGUF file (magic: '${MAGIC}')" >&2
  echo "       the URL probably returned an error page rather than the model" >&2
  rm -f "${TARGET}"
  exit 1
fi

if command -v sha256sum >/dev/null 2>&1; then
  ACTUAL=$(sha256sum "${TARGET}" | awk '{print $1}')
elif command -v shasum >/dev/null 2>&1; then
  ACTUAL=$(shasum -a 256 "${TARGET}" | awk '{print $1}')
else
  ACTUAL=""
  echo "  warning: no sha256 tool found; integrity was not verified" >&2
fi

if [[ -n "${MODEL_SHA256}" && -n "${ACTUAL}" ]]; then
  if [[ "${ACTUAL}" != "${MODEL_SHA256}" ]]; then
    echo "error: checksum mismatch" >&2
    echo "  expected ${MODEL_SHA256}" >&2
    echo "  actual   ${ACTUAL}" >&2
    echo "  refusing to keep a file that is not the declared model" >&2
    exit 1
  fi
  echo "  checksum verified"
elif [[ -n "${ACTUAL}" ]]; then
  echo "  sha256: ${ACTUAL}"
  echo "  record this in configs/runtime/competition.yaml under model.sha256"
fi

SIZE_MB=$(( $(wc -c < "${TARGET}") / 1048576 ))
echo "  ready: ${TARGET} (${SIZE_MB} MB)"
echo "  next:  leanlm doctor"
