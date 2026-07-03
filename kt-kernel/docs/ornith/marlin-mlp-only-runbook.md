# mlp-only Marlin（compressed-tensors w4a16）+ kt CPU 专家

面向 **8GB GPU**（如 RTX 4060）：GPU 非专家走 **Marlin w4**（仅量化各层 `mlp` 的 gate/up/down），attention / `linear_attn` / router / shared_expert / embed / lm_head 保持 **BF16**；MoE 路由专家由 **kt LLAMAFILE** 从 `ornith-cpu-experts-q6k.gguf` mmap。

约束同 `q6-dual-track-runbook.md`：禁止下载 ~70GB 官方 BF16；权重来源为单文件 Q6 全量 GGUF 导出链。

## 产物路径

| 路径 | 说明 |
|------|------|
| `ornith-gpu-bf16-standalone/` | `package_gguf_bf16_for_awq.sh` 产出；量化输入 |
| `ornith-gpu-w4-mlp-only-from-gguf/` | `run_awq_quantization.py` 默认输出（~3.7GB safetensors + `quantization_config`） |
| `ornith-cpu-experts-q6k.gguf` | CPU 专家子集 |
| `ornith-gpu-bf16-standalone/` | tokenizer / `preprocessor_config`（`TOKENIZER_DIR`） |

## 量化（一次性）

venv 建议：`.venv-awq-quantize`（llmcompressor + torch CPU）。

```bash
cd ORNITH_ROOT
export ORNITH_W4_INPUT="${ORNITH_ROOT}/ornith-gpu-bf16-standalone"
export ORNITH_W4_OUTPUT="${ORNITH_ROOT}/ornith-gpu-w4-mlp-only-from-gguf"
.venv-awq-quantize/bin/python ktransformers/kt-kernel/tools/run_awq_quantization.py
```

脚本要点：`num_experts=0` 建骨架防 OOM；`pipeline=datafree`（**无校准**，symmetric w4 仅适合 mlp-only 过渡）；`QuantizationModifier` **ignore** 含 `self_attn`、`linear_attn`、`mlp.gate`、`mlp.shared_expert*`、顶层 `embed_tokens` / `lm_head`；事后 `_patch_router_weights` / `_patch_full_attention_bf16` / `_patch_shared_expert_bf16` / `_patch_lm_head_bf16` 均经 `_align_to_model_weight(src, target_shape)` 对齐（**禁止**对已是 loader 布局的 gate 盲目 `.t()`）；`_validate_quantized_checkpoint_finite()` 拒绝 scale 含 NaN/Inf；`config.architectures` 保持 `Qwen3_5MoeForConditionalGeneration`。

**导出前置**：`gguf_gpu_slice_to_hf_awq_prep` 须与 `gguf_quant_weights_iterator` 一致（`mlp.gate` 为 `(256,2048)`）。旧 `ornith-gpu-bf16-from-gguf` 若 gate 为 `(2048,256)` 需重跑 export；回归 `pytest test/per_commit/test_ornith_export_gate_matches_iterator.py`。

前置：已完成 `run-gguf-gpu-export.sh` + `package_gguf_bf16_for_awq.sh`（见 `export-gpu-awq-marlin.md`）。

## 起服

```bash
cd ORNITH_ROOT/ktransformers/kt-kernel
source ../.venv-public-py312/bin/activate   # 或 _env.sh 默认 PY
bash scripts/ornith/run_sglang_compressed_ik.sh
```

可选覆盖：`MODEL_DIR`、`TOKENIZER_DIR`、`KT_WEIGHT`；更长 context 可追加参数（8GB 上易 OOM，默认 **64**）：

```bash
bash scripts/ornith/run_sglang_compressed_ik.sh --context-length 128
```

### 8GB 默认参数（脚本内）

| 参数 | 值 | 说明 |
|------|-----|------|
| `--quantization` | `compressed-tensors` | Marlin w4a16 |
| `--dtype` | `bfloat16` | 非量化层 / KV |
| `--mem-fraction-static` | `0.90` | 权重占显存比例 |
| `--context-length` | `64` | VL 壳 + visual 占显存 |
| `--max-mamba-cache-size` | `2` | hybrid GDN |
| `--max-running-requests` | `1` | |
| `--max-total-tokens` | `512` | |
| `--disable-radix-cache` | | 省显存 |
| `--skip-server-warmup` | | **必开**：默认 warmup 发 VLM 图文 ~80 token，超过 context=64 会 400 并杀进程 |
| `SGLANG_KT_BYPASS_GPU_MOE` | `1` | 专家仅 kt |

日志期望：`Load weight end` 后 `avail mem≈1.3GB`；`skip_server_warmup=True` 出现在 `server_args`。

## 冒烟

等服务 `Uvicorn running on http://127.0.0.1:30000`：

```bash
bash ktransformers/kt-kernel/scripts/ornith/curl-chat-smoke.sh 30000
```

prompt 已缩短为 `Paris is`（适配 64 context）。也可用 `/generate` 更短路径排查采样。

## sglang-fork 加载契约（本路径依赖）

代码在 `sglang/python/sglang/srt/models/qwen3_5.py`：

- 扁平 checkpoint `model.layers.*` → VL 壳下 `self.model` 的 `model.layers.*`（**不要**再 lift 到 `model.language_model.*`）。
- 层内 norm/attn：`model.layers.N.{input_layernorm,…}` → `model.layers.N.{linear_attn|self_attn}.…`（`mlp.*` **不**嵌进 attn 子模块）。
- `Qwen3_5MoeForConditionalGeneration.hf_to_sglang_mapper` 置空，避免父类 Qwen3-VL 把前缀改成 `language_model.model.*`。
- stacked `q_proj`→`qkv_proj` 使用 **单次** `replace(..., 1)`，避免 `qkqkv_proj`。
- `_qwen35_decoder_layer_type` 读 `config.text_config`，避免 `layers_block_type` 越界。

回归：`pytest sglang/test/srt/test_qwen35_vl_flat_hf_checkpoint_prefix.py test/srt/test_qwen35_vl_self_attn_checkpoint_names.py test/srt/test_qwen35_decoder_layer_type.py -q`

ktransformers：`pytest ktransformers/kt-kernel/test/per_commit/test_ornith_marlin_serve_args.py -q`

## 已知问题（2026-07-03）

1. **采样 NaN / 乱码**：根因常为 **datafree symmetric w4** 前向数值漂移或 **export gate 布局** 与 on-the-fly iterator 不一致（见 export 回归）。排查顺序：`pytest test/per_commit/test_ornith_w4_quant_scales_finite.py`；重导 BF16 standalone；`temperature=0` 或 `/generate` greedy；仍 NaN 则暂勿 `temperature>0` 直至 AWQ/校准路径可用。加载后少量 `Parameter … not found`（`shared_expert.gate/up_proj`、full-attn 别名）需继续对齐 `gate_up_proj` 融合与 BF16 attn patch。
2. **加载告警 ~200 条**：较修复前 ~667 已下降；不影响起服，可能影响生成质量。
3. **context 64**：仅适合极短对话；勿依赖默认 server warmup。
4. **AutoAWQ / awq_marlin**：8GB 校准 OOM；本路径用 llmcompressor **非** AutoAWQ。

## 与终态 Marlin 轨关系

终态目标：`run-full-gguf-awq-pipeline.sh` → `ornith-gpu-awq-from-gguf` → `run_sglang_marlin_ik.sh`（`--quantization awq_marlin`）。当前 **mlp-only compressed** 为 8GB 上的可加载过渡方案，**严禁**用全 BF16 起服代替 Marlin（见 `q6-dual-track-runbook.md` AWQ 节）。