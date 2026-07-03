#!/usr/bin/env bash
# Last-prompt-position next-token top logprobs via /generate (logprob_start_len=0).
set -euo pipefail
PORT="${1:-30000}"
# shellcheck source=_env.sh
source "$(dirname "$0")/_env.sh"
IDS="$("$PY" -c "
from transformers import AutoTokenizer
t=AutoTokenizer.from_pretrained('${ORNITH_ROOT}/ornith-gguf-runtime', trust_remote_code=True)
print(','.join(str(x) for x in t.encode('The capital of France is', add_special_tokens=False)))
")"
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null
curl -s "http://127.0.0.1:${PORT}/generate" \
  -H 'Content-Type: application/json' \
  -d "{\"input_ids\":[${IDS}],\"sampling_params\":{\"temperature\":0,\"max_new_tokens\":1},\"return_logprob\":true,\"top_logprobs_num\":8,\"logprob_start_len\":0}" \
  | "$PY" -c "
import sys, json
from transformers import AutoTokenizer
d=json.load(sys.stdin)
tok=AutoTokenizer.from_pretrained('${ORNITH_ROOT}/ornith-gguf-runtime', trust_remote_code=True)
mi=d.get('meta_info') or {}
# input_token_logprobs: list of (logprob, token_id, ...) per position
rows=mi.get('input_token_logprobs') or []
if not rows:
    print('no input_token_logprobs', mi.keys())
    sys.exit(1)
last=rows[-1]
print('last_prompt_pos logprob entry:', last)
out=d.get('text','')
print('generated_1tok:', repr(out))
tops=mi.get('output_top_logprobs') or []
if tops and tops[0]:
    rows=tops[0]
    decoded=[(lp, tid, tok.decode([tid]) if tid is not None else '') for lp,tid,_ in rows[:8]]
    print('top8 next token:', decoded)
paris=11751
for name,rows in [('output', mi.get('output_token_logprobs')), ('input', mi.get('input_token_logprobs'))]:
    if rows:
        print(name, 'chosen', rows[0])
print('Paris id', paris, 'decode', tok.decode([paris]))
"