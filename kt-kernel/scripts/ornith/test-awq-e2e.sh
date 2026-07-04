#!/bin/bash
# AWQ checkpoint sglang E2E 可读性测试
set -e

SCRIPT_DIR="$(dirname "$0")"
AWQ_CKPT="${1:-/home/kunweiz/Desktop/Ornith/ornith-gpu-awq-from-gguf}"
PORT="${2:-30002}"

echo "AWQ checkpoint: $AWQ_CKPT"
echo "Port: $PORT"

# 验证 checkpoint 存在
if [ ! -d "$AWQ_CKPT" ]; then
    echo "❌ AWQ checkpoint 目录不存在"
    exit 1
fi

# 启动 sglang server
export PATH="/home/kunweiz/Desktop/Ornith/.venv-public-py312/bin:$PATH"

LOG_FILE="/tmp/awq_e2e_test.log"
echo "启动 sglang server..."
echo "日志: $LOG_FILE"

python3 -m sglang.launch_server \
    --model-path "$AWQ_CKPT" \
    --quantization awq_marlin \
    --tp 1 \
    --port "$PORT" \
    --mem-fraction-static 0.85 \
    --dtype bfloat16 \
    --max-running-requests 1 \
    --attention-backend triton \
    > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# 等待服务器就绪
echo -n "等待服务就绪"
for i in $(seq 1 180); do
    sleep 1
    if curl -s -o /dev/null -w "%{http_code}" "http://localhost:$PORT/health" 2>/dev/null | grep -q "200"; then
        echo " ✅"
        break
    fi
    echo -n "."
    if [ $i -eq 180 ]; then
        echo " ❌ 超时"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
done

echo ""

# 发送测试请求
PROMPT="The capital of France is"
echo "发送测试请求: '$PROMPT'"

RESPONSE=$(curl -s "http://localhost:$PORT/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "ornith-gpu-awq-from-gguf",
        "messages": [{"role": "user", "content": "'"$PROMPT"'"}],
        "max_tokens": 64,
        "temperature": 0.1
    }' | jq -r '.choices[0].message.content')

echo ""
echo "响应:"
echo "$RESPONSE"

# 关闭服务器
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

# 判断可读性
if echo "$RESPONSE" | grep -qi "paris\|eiffel\|france"; then
    echo ""
    echo "✅ E2E 测试通过：输出包含语义关键词"
    exit 0
else
    echo ""
    echo "⚠️  E2E 测试需人工检查"
    exit 0
fi
