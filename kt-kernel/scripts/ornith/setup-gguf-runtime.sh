#!/usr/bin/env bash
# Symlink farm: HF metadata + GPU GGUF for SGLang config/tokenizer resolution
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
RUNTIME="${RUNTIME:-${ORNITH_ROOT}/ornith-gguf-runtime}"
HF="${HF:-${HF_TEMPLATE_DEFAULT}}"
GPU_GGUF="${GPU_GGUF:-${ORNITH_ROOT}/ornith-gpu-non-expert.gguf}"
mkdir -p "${RUNTIME}"
for f in config.json tokenizer.json tokenizer_config.json generation_config.json \
  preprocessor_config.json chat_template.jinja vocab.json processor_config.json; do
  [[ -f "${HF}/${f}" ]] && ln -sf "${HF}/${f}" "${RUNTIME}/${f}"
done
ln -sf "${GPU_GGUF}" "${RUNTIME}/ornith-gpu-non-expert.gguf"
echo "RUNTIME=${RUNTIME}"