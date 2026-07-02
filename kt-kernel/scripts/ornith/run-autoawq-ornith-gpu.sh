#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
MODEL="${1:-${ORNITH_ROOT}/ornith-gpu-bf16-standalone}"
OUT="${2:-${ORNITH_ROOT}/ornith-gpu-awq-from-gguf}"
"$PY" -c "import awq" 2>/dev/null || { echo "pip install autoawq" >&2; exit 1; }
exec "$PY" "${KT_ROOT}/tools/autoawq_ornith_vl.py" \
  --model-path "${MODEL}" --output-dir "${OUT}" \
  --w-bit 4 --q-group-size 128 --dtype float16