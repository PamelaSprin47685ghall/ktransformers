#!/usr/bin/env bash
set -euo pipefail
PORT="${1:-30000}"
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || {
  echo "health not ready on :${PORT}" >&2
  exit 1
}
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model":"default","messages":[{"role":"user","content":"The capital of France is"}],"max_tokens":24,"temperature":0.1}'
echo