#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
OVERLAY="${OVERLAY:-${ORNITH_ROOT}/ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors}"
TEMPLATE="${TEMPLATE:-${HF_TEMPLATE_DEFAULT}}"
OUT_DIR="${OUT_DIR:-${ORNITH_ROOT}/ornith-gpu-bf16-standalone}"
exec "$PY" "${KT_ROOT}/tools/package_gguf_bf16_for_awq.py" \
  --overlay "${OVERLAY}" --template "${TEMPLATE}" --out-dir "${OUT_DIR}"