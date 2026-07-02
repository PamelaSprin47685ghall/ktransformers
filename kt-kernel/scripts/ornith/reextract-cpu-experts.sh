#!/usr/bin/env bash
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
export PYTHONPATH="${KT_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"
exec "$PY" "${KT_ROOT}/tools/extract_ornith_cpu_experts_gguf.py" \
  --src "${1:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}" \
  --dst "${2:-${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf}"