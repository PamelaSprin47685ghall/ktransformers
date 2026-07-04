# mlp-only Marlin（compressed-tensors w4a16）+ kt CPU 专家

面向 **8GB GPU**（如 RTX 4060）：**单轨** GPU 非专家 = `compressed-tensors`；**仅** `shared_expert.{gate,up,down}_proj` 做 Marlin w4（GPU 切片**无** routed `mlp.experts.*`，专家在 CPU Q6_K）。其余 **全部 BF16**：`embed`/`lm_head`、`linear_attn`、`self_attn`、`mlp.gate` router、norm/SSM 变换张量。策略推导见 `w4-quant-policy.md`。MoE 路由专家仅 **kt LLAMAFILE** mmap `ornith-cpu-experts-q6k.gguf`。

约束同 `q6-dual-track-runbook.md`：禁止下载 ~70GB 官方 BF16；权重来源为单文件 Q6 全量 GGUF 导出链。

## 产物路径

| 路径 | 说明 |
|------|------|
| `ornith-gpu-bf16-standalone/` | `package_gguf_bf16_for_awq.sh` 产出；量化输入 |
| `ornith-gpu-w4-q6-parity-from-gguf/` | **Q6 同覆盖** w4（`w4-quant-policy.md`）；重跑 `run_awq_quantization.py` 后盘 ~1–1.5GB。备选 Q6 文件 ~2GB：`run_sglang_split_q6.sh` |
| `ornith-cpu-experts-q6k.gguf` | CPU 专家子集 |
| `ornith-gpu-bf16-standalone/` | tokenizer / `preprocessor_config`（`TOKENIZER_DIR`） |

## 量化（一次性）

venv 建议：`.venv-awq-quantize`（llmcompressor + torch CPU）。

```bash
cd ORNITH_ROOT
export ORNITH_W4_INPUT="${ORNITH_ROOT}/ornith-gpu-bf16-standalone"
export ORNITH_W4_OUTPUT="${ORNITH_ROOT}/ornith-gpu-w4-q6-parity-from-gguf"
.venv-awq-quantize/bin/python ktransformers/kt-kernel/tools/run_awq_quantization.py
```

脚本要点：`num_experts=0` 建骨架防 OOM；`pipeline=datafree` + **CPU 量化**（`CUDA_VISIBLE_DEVICES=""`）；ignore 见 `w4-quant-policy.md`（**勿** ignore 整块 `shared_expert`）；仅 **120** 个 shared proj 进 w4；patch：router + full_attn + lm_head + `shared_expert.gate` 标量 BF16，**不**盖 packed proj；`_align_to_model_weight`；`_validate_quantized_checkpoint_finite()`。

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

### 8GB 显存预算（单轨 compressed，多模态开）

**禁止**：`--language-only`（需 encoder-urls）；禁止与 `run_sglang_split_q6.sh`（GGUF on-the-fly）或 BF16 全量权重同时起第二套服务占显存。

| 参数 | 默认 | 说明 |
|------|------|------|
| `--model-path` / `--tokenizer-path` | 均 `ornith-gpu-w4-mlp-only-from-gguf` | 单目录，避免 standalone 双份元数据路径 |
| `--quantization` | `compressed-tensors` | 仅 Marlin w4 mlp |
| `--mem-fraction-static` | `0.93`（`ORNITH_MEM_FRACTION_STATIC`） | **加载后**剖析：`rest=avail−total×(1−f)`；f 过低会 profiled KV&lt;0。勿盲目降到 0.84 |
| `--context-length` | `48`（`ORNITH_CONTEXT_LENGTH`） | 多模态 VL 剖析仍占预算 |
| `--max-total-tokens` | `128`（`ORNITH_MAX_TOTAL_TOKENS`） | 与 context 同量级，防 profiled KV≤0 |
| `--max-mamba-cache-size` | `1` | GDN 状态 |
| `--kt-cpuinfer` | `12`（`ORNITH_KT_CPUINFER`） | 减 CPU 线程争用与峰值 |
| `--skip-server-warmup` | 必开 | 默认 warmup 图文超短 context |
| `SGLANG_KT_BYPASS_GPU_MOE` | `1` | 路由专家仅 kt mmap |

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