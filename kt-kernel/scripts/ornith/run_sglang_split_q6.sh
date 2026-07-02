#!/usr/bin/env bash
# Q6 双轨：GPU 非专家 GGUF（on-the-fly）+ CPU 专家 ornith-cpu-experts-q6k.gguf（kt LLAMAFILE）
# 不依赖 AutoAWQ / 70GB safetensors
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
RUNTIME="${RUNTIME:-${ORNITH_ROOT}/ornith-gguf-runtime}"
bash "$(dirname "$0")/setup-gguf-runtime.sh"
GPU_GGUF="${RUNTIME}/ornith-gpu-non-expert.gguf"
CPU_GGUF="${CPU_GGUF:-${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf}"
export SGLANG_APPLY_CONFIG_BACKUP=none
export SGLANG_KT_BYPASS_GPU_MOE=1
export SGLANG_DISABLE_CUDNN_CHECK=1
exec "${PY}" -m sglang.launch_server \
  --model-path "${GPU_GGUF}" \
  --tokenizer-path "${RUNTIME}" \
  --load-format gguf \
  --kt-method LLAMAFILE \
  --kt-weight-path "${CPU_GGUF}" \
  --kt-cpuinfer 16 \
  --kt-threadpool-count 1 \
  --kt-num-gpu-experts 0 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --trust-remote-code \
  --language-only \
  --host 127.0.0.1 \
  --port 30000 \
  "$@"