#!/usr/bin/env bash
# GPU awq_marlin + CPU kt LLAMAFILE experts (SGLANG_KT_BYPASS_GPU_MOE=1)
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
MODEL_DIR="${MODEL_DIR:-${ORNITH_ROOT}/ornith-gpu-awq-from-gguf}"
KT_WEIGHT="${KT_WEIGHT:-${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf}"
export SGLANG_KT_BYPASS_GPU_MOE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
exec "${PY}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" --tokenizer-path "${MODEL_DIR}" \
  --quantization awq_marlin --dtype float16 \
  --kt-method LLAMAFILE --kt-weight-path "${KT_WEIGHT}" \
  --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-num-gpu-experts 0 \
  --mem-fraction-static 0.85 --disable-cuda-graph --trust-remote-code \
  --host 127.0.0.1 --port 30000 "$@"