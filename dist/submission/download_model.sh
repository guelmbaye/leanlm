#!/usr/bin/env bash
# Download the model weights. Idempotent, credential-free, and verified.
#
# The evaluator runs this before profiling begins; once profiling starts no
# outbound request is permitted. Everything below happens in that window.
set -euo pipefail
cd "$(dirname "$0")"

MODEL_PATH="model/Qwen3.5-2B-Q4_0.gguf"
MODEL_URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_0.gguf"
MODEL_SHA256="cd70221bebaee0503e0f6717e174250cd7825aa88438b3aabec9ad55731d9bb1"

mkdir -p "$(dirname "${MODEL_PATH}")"

# Idempotent: a completed download is verified, never fetched again.
if [[ -f "${MODEL_PATH}" ]]; then
  echo "  ${MODEL_PATH} already present; verifying"
else
  if [[ -z "${MODEL_URL}" ]]; then
    echo "error: no model URL was declared at packaging time" >&2
    exit 1
  fi
  echo "  downloading ${MODEL_URL}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --progress-bar -o "${MODEL_PATH}.part" "${MODEL_URL}"
  else
    wget -O "${MODEL_PATH}.part" "${MODEL_URL}"
  fi
  mv "${MODEL_PATH}.part" "${MODEL_PATH}"
fi

# A GGUF file starts with the four bytes "GGUF". An HTML error page does not.
MAGIC=$(head -c 4 "${MODEL_PATH}" || true)
if [[ "${MAGIC}" != "GGUF" ]]; then
  echo "error: ${MODEL_PATH} is not a GGUF file (magic: '${MAGIC}')" >&2
  rm -f "${MODEL_PATH}"
  exit 1
fi

if [[ -n "${MODEL_SHA256}" ]]; then
  if command -v sha256sum >/dev/null 2>&1; then
    ACTUAL=$(sha256sum "${MODEL_PATH}" | awk '{print $1}')
  else
    ACTUAL=$(shasum -a 256 "${MODEL_PATH}" | awk '{print $1}')
  fi
  if [[ "${ACTUAL}" != "${MODEL_SHA256}" ]]; then
    echo "error: checksum mismatch" >&2
    echo "  expected ${MODEL_SHA256}" >&2
    echo "  actual   ${ACTUAL}" >&2
    exit 1
  fi
  echo "  checksum verified"
fi

echo "  ready: ${MODEL_PATH}"
