# Marlin 轨乱码 / 起服失败 — 调试索引（非「量化背锅」）

## 1. 起服崩溃（已定位）

```
RuntimeError: gptq_marlin_repack.cuh:309: size_n = 32 is not divisible by tile_n_size = 64
```

- **层**：`linear_attn.in_proj_a` / `in_proj_b`（GDN 输出维 32）
- **修**：sglang `marlin_pad_n_for_repack` + `compressed_tensors_wNa16` 对 N&lt;64 垫零；**Q6 parity 仍 w4 `in_proj_a/b`**（勿 ignore）
- **勿**归因于 lm_head / datafree 泛化

## 2. Chat 乱码 — 根因（2026-07-03）

**Bug**：`qwen3_5.py` `_map_vl_lm_layer_param_to_sglang_attn_submodule` 对 full-attn 层**无条件**剥 `self_attn.`，把  
`model.layers.3.self_attn.q_proj.weight` → `model.layers.3.q_proj.weight`。  
MoE VL 实际参数为 `model.layers.3.self_attn.qkv_proj.weight_packed`，**10 层 full self_attn 整段未加载** → logits 乱码（非量化精度）。

**修**：若 `params_dict` 含 `layers.N.self_attn.*` 则**保留** checkpoint 的 `self_attn.` 前缀；仅 legacy flat `q_proj` 才 strip。

**验**：`pytest test/srt/test_qwen35_compressed_tensors_param_resolve.py` + 重启后 `curl-prompt-next-token-logprobs.sh` 看 `Paris` 是否进 top8。

## 2b. 其它怀疑面（若仍乱码）

`q6-dual-track-runbook.md` §备选 on-the-fly 已记：

- **ik 全量 Q6 CPU**：`The capital of France is` → 连贯英文 → **GGUF 权重可读**
- **sglang 双轨**（GPU 切片 + kt CPU 专家）：`/generate` top1 常乱码 token；单点测试（attn_q、shared、out_proj perm、export down_proj）已过
- **怀疑面**：GPU 切片 **加载/变换/拼接**（embed Q6 行布局、linear_attn 转置链、MoE router+kt），非「没量化」

### 建议调试顺序

1. `pytest test/srt/test_qwen35*.py` + `kt-kernel/test/per_commit/test_ornith_*`
2. 同 prompt 对比：`run_sglang_split_q6.sh` vs `run_sglang_compressed_ik.sh`
3. `curl-prompt-next-token-logprobs.sh`：末位 logits 是否含 `Paris`（11751）
4. 分层 dump：`debug_tensor_dump_*`（sglang）对 layer0/3 hidden

## 3. Q6 parity 产物（2026-07-03 起服成功）

- 盘 **~1.9GB**；`in_proj_a/b` **w4** + sglang **N 维 pad 32→64**
- 加载后 **mem usage ~3.86GB**，avail **~2.75GB**（较 mlp-only ~5.3GB 明显下降）
- pad 须在 **列（N）** 维：`F.pad(w, (0, pad_n, 0, 0))`，勿 pad 行维