#!/usr/bin/env bash
# Ornith Q6_K MTP GGUF — resume download (~28GB)
set -euo pipefail
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
TARGET="${1:-${ORNITH_ROOT}/ornith-1.0-35b-Q6_K-MTP-final.gguf}"
URL='https://huggingface.co/skinnyctax/Ornith-1.0-35b-Q6_K-Frankenstein-MTP-GGUF/resolve/main/ornith-1.0-35b-Q6_K-MTP-final.gguf'
TMP="${TARGET}.partial"
log() { printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$*"; }
[[ -f "$TARGET" ]] && { log "exists: $TARGET"; ls -lh "$TARGET"; exit 0; }
if command -v curl &>/dev/null; then
  curl -L -C - --retry 8 --retry-delay 5 -o "$TMP" "$URL" && mv -f "$TMP" "$TARGET"
elif command -v wget &>/dev/null; then
  wget -c -O "$TMP" "$URL" && mv -f "$TMP" "$TARGET"
else
  echo "need curl or wget" >&2; exit 1
fi
log "done: $TARGET"