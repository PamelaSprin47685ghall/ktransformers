# HF metadata only (no weights, no tokenizer.json / index)

Use `ORNITH_ROOT/Ornith-1.0-35B-hf` or `huggingface-cli download deepreinforce-ai/Ornith-1.0-35B --include "*.json" --exclude "*.safetensors*"` for tokenizer + index.

Export/package default template: this dir if `config.json` exists, else `ORNITH_ROOT/Ornith-1.0-35B-hf`.
