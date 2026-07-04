# Q6 轨「压了什么」vs Marlin 轨「压了什么」

## 一句话

| 轨 | 磁盘 | GPU 上实际是什么 | 谁被「压缩」 |
|----|------|------------------|--------------|
| **Q6 双轨** `run_sglang_split_q6.sh` | `ornith-gpu-non-expert.gguf` **~2 GB** | 加载时 **Q6_K 按需 dequant → BF16 算子**；权重主体在 mmap 里仍是 4–6 bit | **几乎全部 GPU 非专家**（attn、linear_attn、shared、router、embed、lm_head…） |
| **Marlin Q6 parity** `run_sglang_compressed_ik.sh` | `ornith-gpu-w4-q6-parity-from-gguf` **~1–1.5 GB**（重跑量化后） | w4 Marlin + **router BF16** | **与 Q6 切片同覆盖**：embed、lm_head、linear_attn、self_attn、shared；**不压** router |
| 旧 mlp-only Marlin | ~4.4 GB | 大部 BF16 | 仅 shared（已废弃为默认） |

「人家能 2GB、你不能」= **不是同一类压缩**：Q6 是 **权重量化格式留在文件里**；Marlin 轨是 **先离线 dequant 成 BF16 safetensors，再只对一小块做 w4**。

## Q6 轨：压缩了哪些

来源：`extract_ornith_gpu_non_expert_gguf.py` 从全量 `Q6_K` GGUF 里**剔除** routed `ffn_*_exps`，**保留**：

- `embed` / `output`（lm_head）
- 每层 `attn_*`（full attention，源多为 Q6_K）
- 每层 `ssm_*` / GDN → 映射为 `linear_attn`（Q6_K）
- `ffn_gate_inp`（router）、`ffn_*_shexp`（shared expert）、norm、conv、MTP 等

**未**在 GPU 文件里保留：256 路 routed 专家 → **26 GB** 在 `ornith-cpu-experts-q6k.gguf`，kt CPU。

**压缩机制**：张量在 GGUF 内为 **Q6_K（~6 bit/weight 量级）**，文件 ~2 GB。SGLang `--load-format gguf` + `gguf_qwen35moe`：**读盘 → dequant → 当前层/张量 BF16 参与计算**，不必把整个 5 GB BF16 矩阵一次性 materialize 到显存（与实现/缓存策略有关，但**磁盘与 mmap 体积**就是 2 GB 轨的核心）。

## Marlin 轨：压缩了哪些

流水线：

1. `gguf_gpu_slice_to_hf_awq_prep`：**Q6_K → BF16** 写入单文件 `model.safetensors`（~4.7 GB standalone）
2. `run_awq_quantization.py`：**datafree w4** 仅命中 `Linear` 且 **ignore** attn / linear_attn / embed / lm_head / router / shared 标量门
3. 产物 ~4.4 GB：**613 个 BF16 张量 + 120 组 shared_expert packed**

**压缩机制**：只有 **shared_expert 三路 Linear** 变成 Marlin w4；**linear_attn ~2 GB、embed+lm_head ~2 GB、self_attn ~0.67 GB** 在导出第 1 步就已是 **全精度 BF16 常驻**，无法再变成 2 GB 文件，除非：

- **改轨**：起服用 `ornith-gpu-non-expert.gguf`（Q6 on-the-fly），或
- **改质量契约**：对 linear_attn/embed 做 w4（业界与 Qwen3.5 配方均 **禁止**，datafree 易乱码/状态发散），或
- **tie / 删 lm_head**（本模型 `tie_word_embeddings=false`，且用户要质量）

## 对照表（同一 GPU 非专家语义）

| 模块 | Q6 文件里 | Marlin 文件里 | Marlin GPU |
|------|-----------|---------------|------------|
| embed + lm_head | Q6_K | BF16 | BF16 ~2 GB |
| linear_attn ×30 | Q6_K | BF16 | BF16 ~2 GB |
| self_attn ×10 | Q6_K | BF16 | BF16 ~0.67 GB |
| router gate | Q6_K/BF16 混合 | BF16 | BF16 |
| shared_expert 3×proj | Q6_K | **w4 packed** | Marlin |
| routed experts | 不在 GPU 文件 | 不在 | CPU kt |

## 为什么「必须 Marlin + shared 必压」仍 ~5 GB

- **Marlin 是算子格式**，不是「把整个 checkpoint 变成 2 GB」。
- 你已要求的 **shared 必 w4** 只动 **~5% 参数量**；**95% 仍在 BF16**，因为 **质量 + GDN/路由契约**。
- **8GB 要接近「人家的 2GB」**：用 **`run_sglang_split_q6.sh`**（Q6 GPU 切片 + kt CPU 专家），**不是**再对 BF16 导出做更多 ignore。

## 两轨怎么选

| 目标 | 选 |
|------|-----|
| 最小 GPU 权重体积 / 8GB 存活 | Q6 `run_sglang_split_q6.sh` |
| shared FFN 走 Marlin、compressed-tensors 单轨 | `run_awq_quantization.py` + `run_sglang_compressed_ik.sh`（接受 ~5 GB GPU 权重级） |
| 两者兼得 | **做不到**在同一份「全 BF16 导出」上；须 **Q6 加载** 或接受 attention w4 掉质量 |

## 脚本

```bash
# ~2GB GPU 非专家（Q6）
bash scripts/ornith/run_sglang_split_q6.sh

# ~4.4GB 盘 / ~5GB GPU 加载（shared Marlin）
bash scripts/ornith/run_sglang_compressed_ik.sh
```