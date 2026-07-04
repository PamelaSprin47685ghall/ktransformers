# Q6 单文件双轨跑通（GPU Marlin + CPU ik）

约束：只用 `ornith-1.0-35b-Q6_K-MTP-final.gguf`，禁止下载官方 70GB BF16 safetensors。

## 数据流

```
Q6 全量 GGUF
  ├─ extract-gpu-non-expert.sh → ornith-gpu-non-expert.gguf (~2GB)
  │     └─ run-gguf-gpu-export.sh → model-gpu-from-gguf.safetensors (~4.6GB BF16)
  │           └─ package_gguf_bf16_for_awq.sh → ornith-gpu-bf16-standalone/
  │                 ├─ run_awq_quantization.py → ornith-gpu-w4-compressed-from-gguf/（当前临时）
  │                 └─ run-autoawq-ornith-gpu.sh → ornith-gpu-awq-from-gguf/（最终）
  └─ reextract-cpu-experts.sh → ornith-cpu-experts-q6k.gguf (~26GB)
```

## 环境

- venv：`ORNITH_ROOT/.venv-public-py312`（`uv venv` + `uv pip install -e sglang/python` + `uv pip install -e ktransformers/kt-kernel`）
- `PYTHONPATH=sglang/python:ktransformers/kt-kernel/python`
- HF 小文件：`Ornith-1.0-35B-hf/`（`hf download` 仅 tokenizer/json，排除 `*.safetensors*`）
- ik 子模块：`ktransformers/third_party/ik_llama.cpp`；构建后 `bash ktransformers/scripts/install-llama-cpp-include-shim.sh`

## 一键（推荐：无 AutoAWQ）

AutoAWQ 当前不支持 `qwen3_5_moe`；8GB 卡用 **2GB GPU 非专家 GGUF + 26GB CPU 专家 GGUF**：

```bash
cd ORNITH_ROOT/ktransformers/kt-kernel/scripts/ornith
bash extract-gpu-non-expert.sh
bash reextract-cpu-experts.sh   # 耗时、大文件
bash run_sglang_split_q6.sh
```

Marlin 轨：仅 **shared_expert** w4；**~5GB GPU** 因导出已是 BF16。要 **~2GB** 非专家 → **Q6** `run_sglang_split_q6.sh`（见 `q6-vs-marlin-compression.md`）。

```bash
# 当前（8GB mlp-only Marlin）：run_awq_quantization.py → run_sglang_compressed_ik.sh
# 详见 marlin-mlp-only-runbook.md
# 最终：run-full-gguf-awq-pipeline.sh → run_sglang_marlin_ik.sh
```

## 起服参数（性能）

| 变量/参数 | 默认 | 说明 |
|-----------|------|------|
| `SGLANG_KT_BYPASS_GPU_MOE` | 1 | GPU 不算 routed expert |
| `--quantization awq_marlin` | | 非专家层 Marlin |
| `--kt-method LLAMAFILE` | | CPU 专家 Q6_K mmap |
| `--kt-num-gpu-experts 0` | | 专家全 CPU |
| `--kt-cpuinfer` | 16 | 对齐物理核数可调 |
| `--mem-fraction-static` | 0.85 | 8GB 卡尽量给权重 |
| `--disable-cuda-graph` | | 8GB 上更稳 |

## 验收

1. `curl http://127.0.0.1:30000/health` ready
2. `POST /v1/chat/completions` 短英文 prompt；`embed_tokens` 须 `quant_config=gguf`；Q6_K `token_embd` 量化缓冲为 **vocab 行**（`index_select` dim=0），勿按逻辑 shape `[hidden,vocab]` 在 dim=1 取列
3. 日志无 `SIGILL`、无 expert 双重加载、`Parameter model.embed_tokens` not found

## AWQ 量化策略演进

**2026-07-03**：AWQModifier 校准需 43GB+ CPU 内存（62GB 系统 OOM），`duo_scaling=both` 双份 scale 缓存 + autograd graph 保存导致峰值内存撞墙。根因：`config.json` 声明 `num_experts=256`，`AutoModelForCausalLM.from_pretrained` 按 config 建出 40 层专家融合张量（BF16 60GB）。

**当前降级方案**：纯 symmetric w4a16 权重量化（无 AWQModifier），显式 `pipeline="datafree"` + `num_calibration_samples=0` + 临时 `num_experts=0` 跳过专家骨架分配。脚本 `run_awq_quantization.py` 手动加载 safetensors 并逐张量转置（GGUF meta 布局 (in,out) → PyTorch (out,in)），峰值内存 ~10GB。

**⚠️ 必须 Marlin，严禁 BF16 缩水**：GPU 非专家权重仅 4.6GB，BF16 直接加载虽可行但性能太差。**必须走 Marlin 量化路径**，不允许用 BF16 逃避量化。当前 compressed-tensors w4a16 遇到 `size_n=32 not divisible by tile_n_size=64`（sglang `compressed_tensors_wNa16` 对 `RowParallelLinear` 的 `output_partition_sizes` 某层切出 32），需修 sglang 侧 Marlin shape 兼容或调整量化 ignore 列表。

**路线图**：先用粗糙方法跑通全流程、确认无乱码、验收 E2E 可读性；待内存/资源允许时换回 AWQModifier 获取更高精度。

## 全量单文件 GGUF（非 split 主路径）

`run_sglang_gguf.sh` 直载 28GB 全量仍可能遇 `UninitializedParameter` / MoE 物化顺序问题（见移交报告）。**主验收用 split_q6**；全量 on-the-fly 为 backlog，与 ik 全量基线（已可读）对照时优先修 loader 而非改权重。

## 备选 on-the-fly

`run_sglang_gguf.sh` / `run_sglang_split_q6.sh`（GPU 切片 GGUF）：`gguf_quant_weights_iterator` 对 Q6_K `linear_attn.in_proj_qkv`：**reader logical reshape → dequant → v-head 重排 → BF16 `(hidden, lead)`**；`load_weights` dim1 分片后 **`.t()`**（`0aff623`）。**`in_proj_z/a/b`**：**`dequant_from_reader_tensor` + `apply_gguf_to_hf_weight`**，qweight 行布局 **`(out, hidden)`**（如 z=`(4096,2048)`）供 `fused_mul_mat_gguf`；勿对 z/a/b 做 on-the-fly v-reorder。旧版误转 `(2048,4096)` 会触发 `mat1 and mat2 shapes cannot be multiplied`。F32 `ssm_conv1d` / `ssm_a` / `ssm_dt.bias` 走 **`apply_gguf_to_hf_weight`**（含 conv1d v 段重排 + unsqueeze，`d8ec94e`），勿仅用 `conv1d_reorder`。`out_proj` Q6_K 仍 **权重列不 perm**，靠 `_enable_gguf_linear_attn_out_proj_perms` 运行时 activation perm。**export safetensors** 存 GGUF meta 布局 `(4096,2048)` 与 dequant meta reshape **逐元素对齐**（`test_out_proj_q6k_dequant_meta_layout_matches_export`）；勿对 Q6 `out_proj` 走 BF16 列 perm 加载（会与 act perm 双置换且与 export 不一致）。存储 dtype 与 `model_config.dtype` 一致（`bfloat16` + `GGMLQuantizationType.BF16`）。**chat 仍乱码（已排除单点）**：layer3 `attn_q` Q6_K fused≈dequant（`test_layer3_attn_q_q6k_fused_matches_dequant`）；shared_expert gate/up Q6_K 数值 OK；`out_proj`+activation perm OK。**export 修复**：`gguf_gpu_slice_to_hf_awq_prep._tensor_to_torch` 在 dequant 已是 `logical[::-1]` 时不再 `reshape(logical)`（修复 `shared_expert.down_proj` 等）。**kt smoke**：`test_ornith_llamafile_moe_layer0_smoke` layer0 forward 有限；**`test_ornith_moe_layer0_router_vs_torch`** 真实 `ffn_gate_inp` softmax→top8→renorm 与 CPU 专家 dequant torch 参考 mean rel err &lt;0.35。**shared_expert** gate/up/down Q6_K fused≈dequant（sglang 测试）。**ik 基线（2026-07-03）**：`build-cpu-cli` + 全量 Q6 GGUF `-ngl 0 -ot exps=CPU`，prompt `The capital of France is` → 连贯英文（Eiffel 地标句），说明 **权重文件本身可读**；sglang 乱码应查 **GPU 切片加载/变换/双轨 kt 拼接**，非 GGUF 坏。验收脚本：`curl-chat-smoke.sh`（chat）、`curl-completion-smoke.sh`（裸 prompt）、`curl-prompt-next-token-logprobs.sh`（`/generate` 末位 top8；期望含 `Paris`≈11751，实测 top1 常为乱码 token 如 `icc`）。`test_token_embd_q6k_fused_embedding_matches_dequant_for_france_prompt_ids`：embed 路径 OK。 与整网堆叠 与 `MOE.load_weights`、**`ffn_gate_inp` router**、40 层堆叠。**`ornith-gpu-bf16-standalone`**：`package_gguf_bf16_for_awq` 须把 embed/lm_head `(hidden,vocab)` 转置为 `(vocab,hidden)` 再加载；`run_sglang_bf16_standalone_kt.sh`（:30001）用于分叉「无 GGUF on-the-fly、仅 kt CPU MoE」。`out_proj` Q6_K 缓冲 `(2048,3360)` 为 4096 输入维的合法 Q6_K 列宽（非截断）；与 `GGUF_on_the_fly.txt` §3.3 一致：**权重列不 perm**，`_gguf_out_proj_act_perm` + `fused_mul_mat_gguf` 与 dequant 数值对齐（见 `test_out_proj_q6k_fused_matches_dense_with_activation_perm`）。`ornith-gpu-bf16-standalone` 仅 ~630 键，**不能**直接 `--model-path` 起服（embed 仅 2048 行 vs 词表 248320）。

## 构建 kt（ik ggml）

```bash
cd ktransformers/kt-kernel
CPUINFER_USE_CUDA=0 CPUINFER_CPU_INSTRUCT=AVX2 CPUINFER_FORCE_REBUILD=1 \
  uv pip install --python $ORNITH_ROOT/.venv-public-py312/bin/python -e .
```

`llamafile` 与 ik ggml API 差异由 `third_party/llamafile/kt_ggml_compute_compat.h` 桥接。
