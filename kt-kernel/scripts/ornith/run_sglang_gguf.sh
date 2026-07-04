#!/usr/bin/env bash
# On-the-fly full GGUF + kt CPU experts
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
MODEL_DIR="${MODEL_DIR:-${ORNITH_ROOT}/ornith-gguf-runtime}"
GGUF="${GGUF:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}"
export KT_SKIP_CPU_EXPERTS=1
export ACCELERATE_KT_SKIP_EXPERT_LOADING=1
export SGLANG_DISABLE_CUDNN_CHECK=1
exec "${PY}" -m sglang.launch_server \
  --model-path "${GGUF}" --tokenizer-path "${MODEL_DIR}" --load-format gguf \
  --kt-method LLAMAFILE --kt-weight-path "${ORNITH_ROOT}/ornith-cpu-experts-q6k.gguf" \
  --kt-cpuinfer 16 --kt-threadpool-count 1 --kt-num-gpu-experts 0 \
  --mem-fraction-static 0.85 --disable-cuda-graph --trust-remote-code \
  --host 127.0.0.1 --port 30000 "$@"