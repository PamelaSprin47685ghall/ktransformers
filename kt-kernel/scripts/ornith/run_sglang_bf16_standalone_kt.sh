#!/usr/bin/env bash
# E2E fork: GPU weights from package_gguf_bf16 standalone (no GGUF on-the-fly) + kt CPU experts
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
MODEL_DIR="${MODEL_DIR:-${ORNITH_ROOT}/ornith-gpu-bf16-standalone}"
CPU_GGUF="${CPU_GGUF:-${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf}"
export SGLANG_KT_BYPASS_GPU_MOE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
exec "${PY}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" \
  --tokenizer-path "${MODEL_DIR}" \
  --dtype bfloat16 \
  --kt-method LLAMAFILE \
  --kt-weight-path "${CPU_GGUF}" \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 0 \
  --mem-fraction-static 0.78 \
  --context-length 2048 \
  --disable-cuda-graph \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 30001 \
  "$@"