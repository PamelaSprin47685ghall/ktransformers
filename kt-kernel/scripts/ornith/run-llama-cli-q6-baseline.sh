#!/usr/bin/env bash
# CPU llama-cli baseline on full Q6_K GGUF (readability sanity check; slow).
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
IK_ROOT="${IK_ROOT:-${ORNITH_ROOT}/ktransformers/third_party/ik_llama.cpp}"
CLI="${IK_CLI:-${IK_ROOT}/build-cpu-cli/bin/llama-cli}"
MODEL="${ORNITH_Q6_GGUF:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}"
if [[ ! -x "${CLI}" ]]; then
  echo "missing ${CLI}; cmake -B build-cpu-cli -DGGML_CUDA=OFF && cmake --build build-cpu-cli --target llama-cli" >&2
  exit 1
fi
exec "${CLI}" \
  -m "${MODEL}" \
  -ngl 0 \
  -c 512 \
  -n 32 \
  -t 16 \
  -ot "exps=CPU" \
  -p "The capital of France is" \
  "$@"