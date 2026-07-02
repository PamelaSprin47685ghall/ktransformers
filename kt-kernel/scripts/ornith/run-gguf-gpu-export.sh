#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
TOOL="${KT_ROOT}/tools/gguf_gpu_slice_to_hf_awq_prep.py"
exec "$PY" "$TOOL" \
  --gguf "${1:-${ORNITH_ROOT}/ornith-gpu-non-expert.gguf}" \
  --hf-template "${2:-${ORNITH_ROOT}/Ornith-1.0-35B-hf}" \
  --out-dir "${3:-${ORNITH_ROOT}/ornith-gpu-bf16-from-gguf}" \
  --dtype bfloat16 \
  --mtp-source "${4:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}" \
  "${@:5}"