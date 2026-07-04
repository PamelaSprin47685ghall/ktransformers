#!/usr/bin/env bash
# Same prompt: ik llama-cli greedy 1tok vs sglang /generate top8 (split_q6 default :30000).
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
PROMPT="${PROMPT:-The capital of France is}"
SGLANG_PORT="${1:-30000}"
IK_ROOT="${IK_ROOT:-${ORNITH_ROOT}/ktransformers/third_party/ik_llama.cpp}"
CLI="${IK_CLI:-${IK_ROOT}/build-cpu-cli/bin/llama-cli}"
MODEL="${ORNITH_Q6_GGUF:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}"
echo "=== ik (full Q6 GGUF, CPU, -n 1) ==="
if [[ ! -x "${CLI}" ]]; then
  echo "missing ${CLI}" >&2
  exit 1
fi
IK_OUT="$("${CLI}" -m "${MODEL}" -ngl 0 -c 256 -n 1 -t 8 -ot exps=CPU --temp 0 --top-k 1 -p "${PROMPT}" 2>/dev/null | tail -1)"
echo "line: ${IK_OUT}"
echo
echo "=== sglang (:${SGLANG_PORT}) ==="
bash "$(dirname "$0")/curl-prompt-next-token-logprobs.sh" "${SGLANG_PORT}"