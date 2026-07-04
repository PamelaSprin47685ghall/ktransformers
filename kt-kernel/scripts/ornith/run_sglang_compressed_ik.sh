#!/usr/bin/env bash
# 单轨：仅 compressed-tensors w4（MODEL_DIR）+ CPU kt 专家 GGUF；禁止并行 GGUF on-the-fly / BF16 全量权重。
# 多模态保持默认（VL 壳）；导出无 visual 权重时仅文本路径可用，显存仍按 VLM 剖析预算。
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
pkill -f "sglang.launch_server.*30000" 2>/dev/null || true
sleep 2

MODEL_DIR="${MODEL_DIR:-${ORNITH_ROOT}/ornith-gpu-w4-q6-parity-from-gguf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${MODEL_DIR}}"
KT_WEIGHT="${KT_WEIGHT:-${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf}"
# 权重加载后 avail~0.7GB 时，剖析式 rest=avail-total*(1-f)；f 须够高否则 KV profile≤0
MEM_STATIC="${ORNITH_MEM_FRACTION_STATIC:-0.93}"
CTX="${ORNITH_CONTEXT_LENGTH:-32}"
MAX_TOK="${ORNITH_MAX_TOTAL_TOKENS:-64}"

export SGLANG_KT_BYPASS_GPU_MOE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_KT_HYBRID_TIMING=1

exec "${PY}" -m sglang.launch_server \
  --model-path "${MODEL_DIR}" --tokenizer-path "${TOKENIZER_DIR}" \
  --quantization compressed-tensors --dtype bfloat16 \
  --kt-method LLAMAFILE --kt-weight-path "${KT_WEIGHT}" \
  --kt-cpuinfer "${ORNITH_KT_CPUINFER:-16}" --kt-threadpool-count 1 --kt-num-gpu-experts 0 \
  --mem-fraction-static "${MEM_STATIC}" --context-length "${CTX}" \
  --max-mamba-cache-size 1 --max-running-requests 1 --max-total-tokens "${MAX_TOK}" \
  --mamba-full-memory-ratio 0.01 --disable-radix-cache --disable-cuda-graph \
  --chunked-prefill-size 512 \
  --skip-server-warmup \
  --trust-remote-code \
  --host 127.0.0.1 --port 30000 "$@"
