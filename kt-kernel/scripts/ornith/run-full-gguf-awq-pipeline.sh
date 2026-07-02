#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd -- "$(dirname "$0")" && pwd)"
# shellcheck source=_env.sh
source "${DIR}/_env.sh"
bash "${DIR}/run-gguf-gpu-export.sh" "$@"
bash "${DIR}/package_gguf_bf16_for_awq.sh"
bash "${DIR}/run-autoawq-ornith-gpu.sh" \
  "${ORNITH_ROOT}/ornith-gpu-bf16-standalone" \
  "${ORNITH_ROOT}/ornith-gpu-awq-from-gguf"
echo "MODEL_DIR=${ORNITH_ROOT}/ornith-gpu-awq-from-gguf bash ${DIR}/run_sglang_marlin_ik.sh"