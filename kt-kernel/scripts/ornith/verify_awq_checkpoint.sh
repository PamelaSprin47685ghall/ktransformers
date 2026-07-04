#!/bin/bash
# 验证 AWQ checkpoint 格式
set -e

CHECKPOINT_DIR="${1:-/home/kunweiz/Desktop/Ornith/ornith-gpu-awq-from-gguf}"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "❌ AWQ checkpoint 目录不存在: $CHECKPOINT_DIR"
    exit 1
fi

echo "✅ AWQ checkpoint 目录存在"

# 检查 config.json
if [ ! -f "$CHECKPOINT_DIR/config.json" ]; then
    echo "❌ config.json 缺失"
    exit 1
fi
echo "✅ config.json 存在"

# 检查量化配置
QUANT_CONFIG=$(jq -r '.quantization_config.quant_method // empty' "$CHECKPOINT_DIR/config.json")
if [ "$QUANT_CONFIG" != "awq" ]; then
    echo "❌ quant_method 不是 awq: $QUANT_CONFIG"
    exit 1
fi
echo "✅ quant_method = awq"

QUANT_TYPE=$(jq -r '.quantization_config.quant_type // empty' "$CHECKPOINT_DIR/config.json")
if [ "$QUANT_TYPE" != "awq_marlin" ]; then
    echo "❌ quant_type 不是 awq_marlin: $QUANT_TYPE"
    exit 1
fi
echo "✅ quant_type = awq_marlin"

# 检查权重文件
WEIGHT_FILES=$(find "$CHECKPOINT_DIR" -name "*.safetensors" | wc -l)
if [ "$WEIGHT_FILES" -eq 0 ]; then
    echo "❌ 无 .safetensors 权重文件"
    exit 1
fi
echo "✅ .safetensors 权重文件: $WEIGHT_FILES 个"

# 检查模型文件
MODEL_FILES=$(find "$CHECKPOINT_DIR" -type f | wc -l)
echo "✅ 总计文件: $MODEL_FILES 个"

# 显示量化配置详情
echo ""
echo "量化配置详情:"
jq '.quantization_config' "$CHECKPOINT_DIR/config.json"

echo ""
echo "✅ AWQ checkpoint 验证通过"
