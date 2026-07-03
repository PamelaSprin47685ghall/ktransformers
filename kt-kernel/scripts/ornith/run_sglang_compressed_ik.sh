#!/usr/bin/env bash
# GPU compressed w4 + CPU kt LLAMAFILE experts (temporary low-memory path)
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"

MODEL_DIR="${MODEL_DIR:-${ORNITH_ROOT}/ornith-gpu-w4-mlp-only-from-gguf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ORNITH_ROOT}/ornith-gpu-bf16-standalone}"
KT_WEIGHT="${KT_WEIGHT:-${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf}"

export SGLANG_KT_BYPASS_GPU_MOE=1
export SGLANG_DISABLE_CUDNN_CHECK=1

exec "${PY}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" --tokenizer-path "${TOKENIZER_DIR}" \
  --quantization compressed-tensors --dtype bfloat16 \
  --kt-method LLAMAFILE --kt-weight-path "${KT_WEIGHT}" \
  --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-num-gpu-experts 0 \
  --mem-fraction-static 0.90 --context-length 64 \
  --max-mamba-cache-size 2 --max-running-requests 1 --max-total-tokens 512 \
  --mamba-full-memory-ratio 0.08 --disable-radix-cache --disable-cuda-graph \
  --skip-server-warmup \
  --trust-remote-code \
  --host 127.0.0.1 --port 30000 "$@"
