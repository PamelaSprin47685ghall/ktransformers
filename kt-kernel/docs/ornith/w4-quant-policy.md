# Ornith GPU 切片：谁 w4、谁 BF16（Q6 parity）

## 0. 原则

**Q6 双轨在文件里压什么，Marlin 导出轨就对同一批 Linear 做 w4**（源已是 Q6→BF16 一次反量化，再 w4 ≈ 与 Q6 同量级误差预算）。  
**唯一例外（业界共识 + iterator 契约）**：MoE **router** `mlp.gate`、标量 `shared_expert.gate` / `shared_expert_gate`、**MTP** 子树。

Routed `ffn_*_exps` 不在 GPU 切片 → CPU kt，本脚本不碰。

## 1. 决策表（与 `ornith-gpu-non-expert.gguf` 对齐）

| 模块 | w4 Marlin | BF16 保留 |
|------|-----------|-----------|
| `embed_tokens` | **是**（Q6 同覆盖；若 llmcompressor 留 BF16 则盘内仍 BF16） | — |
| `lm_head` | Q6 盘内量化 | Marlin 轨 **暂 BF16 patch**（与乱码调试正交；见 `marlin-garble-debug.md`） |
| `linear_attn.in_proj_a/b` (N=32) | **是 w4** | Marlin **pad N→64**，sglang 侧已修 |
| `linear_attn.*`（含 in_proj_qkv/out_proj 等 Linear） | **是** | norm/conv 等非 Linear 随模块默认 |
| `self_attn.*` `_proj` | **是** | q_norm/k_norm 等小权重 |
| `shared_expert.{gate,up,down}_proj` | **是**（硬约束） | — |
| `mlp.gate` router | — | **是** |
| `shared_expert.gate` 标量、`shared_expert_gate` | — | **是** |
| `mtp` / `nextn` | — | **是** |
| routed experts | — | CPU Q6_K |

旧版「只压 shared、attn/embed BF16」= **比 Q6 更保守**，磁盘 ~4.4GB；**不是** Q6 做不到而我们不敢，是策略没对齐。现默认 **`ornith-gpu-w4-q6-parity-from-gguf`**。

## 2. `run_awq_quantization.py`

```text
ignore:
  mlp.gate, mlp.shared_expert_gate, mlp.shared_expert.gate.weight
  mtp.*

quantize (targets=Linear):
  embed, lm_head, linear_attn, self_attn proj, shared_expert proj

patch BF16 after oneshot:
  router mlp.gate only (+ shared_expert scalar gate)
  不再 _patch_full_attention_bf16 / _patch_lm_head_bf16
```

`pipeline=datafree` + CPU 量化（`CUDA_VISIBLE_DEVICES=""`）。

## 3. 体积预期

| 产物 | 量级 |
|------|------|
| Q6 GGUF 非专家 | ~2 GB 文件 |
| Q6 parity w4 safetensors | **~1–1.5 GB 盘**（视 packed 布局）；GPU 加载仍含 Marlin workspace + BF16 router + VL 壳 |
| 旧 mlp-only w4 | ~4.4 GB（大部分 attn/embed 仍 BF16） |

## 4. 8GB 起服

parity 后 **权重盘更小**，但 VL + kt + KV 仍紧。`run_sglang_compressed_ik.sh` 默认 `MODEL_DIR=ornith-gpu-w4-q6-parity-from-gguf`。  
若 parity 量化后乱码/NaN → `_validate_quantized_checkpoint_finite` + 回退 Q6 `run_sglang_split_q6.sh`（同 Q6 数值路径）。

## 5. 验收

- `test_ornith_w4_ignore_attention.py`：ignore **无** `self_attn`/`linear_attn` 整段。
- `test_ornith_w4_output_has_packed_weights.py`：shared + **linear_attn.in_proj_qkv** + **embed** 均有 `weight_packed`（parity 目录存在时）。