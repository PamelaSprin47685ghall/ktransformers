# AGENTS — Ornith 成就目录

面向接手 Agent/工程师的**索引**：成就落在哪、怎么跑、禁什么。细节见 `移交报告-2026-07-02.md`、`移交报告.md`、`kt-kernel/docs/ornith/GGUF_on_the_fly.txt`。

## 0. 架构成就（已定稿）

| 决策 | 内容 |
|------|------|
| 双轨推理 | GPU 非专家（AWQ/Marlin 或 GGUF 量化层）+ CPU 路由专家（kt `LLAMAFILE`，`SGLANG_KT_BYPASS_GPU_MOE=1`） |
| 专家权重单一来源 | MoE `ffn_*_exps` 仅 kt `GGUFLoader` mmap；SGLang `gguf_quant_weights_iterator` 跳过，防双重加载 |
| GGUF→GPU 权重 | 禁止下载 ~70GB 官方 BF16；`gguf_gpu_slice_to_hf_awq_prep` → `package_gguf_bf16_for_awq` → AutoAWQ |
| CPU 指令集 | kt avx2 变体禁 AVX512 污染（`setup.py` 钉死子特性 OFF）；llama.cpp **CPU-only** |
| 代码归属 | 三公共 fork，**禁止** patch `site-packages` |

## 1. 仓库地图

| 路径（工作区） | Git 远程 | 本议题主要成就 |
|----------------|----------|----------------|
| `ktransformers/` | `PamelaSprin47685ghall/ktransformers` | 工具链、ornith 脚本、GGUF 提取、AWQ 打包、AVX2 修复、per_commit 测试 |
| `sglang-fork/` | `PamelaSprin47685ghall/sglang` | qwen35moe GGUF 映射/F32 变换、VL checkpoint 名、out_proj 激活置换、MTP 跳过、MoE 融合加载 |
| `ik_llama.cpp/` | `PamelaSprin47685ghall/ik_llama.cpp` | hybrid 量化脚本 `scripts/ornith/`、IQ2 类型源 |

工作区 `Ornith/` 根目录**非 git**；大文件 GGUF/safetensors 不入库。

## 2. ktransformers 成就清单

### 2.1 工具 `kt-kernel/tools/`

| 文件 | 成就 |
|------|------|
| `extract_ornith_gpu_non_expert_gguf.py` | 2GB GPU 切片（630/753 张量类） |
| `extract_ornith_cpu_experts_gguf.py` | CPU 专家子集 GGUF |
| `gguf_gpu_slice_to_hf_awq_prep.py` | Q6_K dequant + `apply_gguf_to_hf_weight` + VL 名 + MTP |
| `package_gguf_bf16_for_awq.py` | **无 70GB**：单分片 + 子集 index standalone HF |
| `merge_gguf_export_into_hf.py` | 可选：已有 16 分片时 overlay 覆写 |
| `autoawq_ornith_vl.py` | AutoAWQ 骨架（8GB 可能 OOM） |

### 2.2 脚本 `kt-kernel/scripts/ornith/`

见 `kt-kernel/scripts/ornith/README.md`。一键主路径：

```bash
bash ktransformers/kt-kernel/scripts/ornith/run-full-gguf-awq-pipeline.sh
bash ktransformers/kt-kernel/scripts/ornith/run_sglang_marlin_ik.sh
```

备选 on-the-fly：`run_sglang_gguf.sh`。

### 2.3 Python `kt-kernel/python/utils/`

| 模块 | 成就 |
|------|------|
| `loader.py` / GGUF 路径 | CPU 专家加载、与 SGLang 分工 |
| `gguf_raw_reader.py` / `gguf_ik_types.py` | ik 量化类型 ID（IQ2_K_R4=337）Python 侧 |

### 2.4 测试 `kt-kernel/test/per_commit/`

| 测试 | 覆盖成就 |
|------|----------|
| `test_package_gguf_bf16_for_awq.py` | 无 HF 分片打包 |
| `test_merge_gguf_export_into_hf.py` | overlay 覆写契约 |
| `test_extract_gpu_non_expert.py` | GPU 切片 |
| `test_setup_avx2_variant_static.py` | AVX2 无 AVX512 泄漏 |
| `test_iqk_iq2_k_r4_switch.py` | 337 缺口文档化（红→待 C++ 回填） |

### 2.5 文档与 fixture

- `kt-kernel/docs/ornith/` — `GGUF_on_the_fly.txt`、`export-gpu-awq-marlin.md`、`ik-kt-iq2_k_r4-backfill.md`
- `kt-kernel/fixtures/Ornith-1.0-35B-hf/` — HF 元数据（无 safetensors）

### 2.6 构建/安装

- `kt-kernel/setup.py` — avx2 变体 AVX512_VNNI/BF16/VBMI=OFF
- `kt-kernel/scripts/install_public_forks_py312.sh` — 公共 fork 引导

## 3. sglang-fork 成就清单

| 区域 | 文件/能力 | 成就 |
|------|-----------|------|
| 映射与变换 | `model_loader/gguf_qwen35moe.py` | 自定义 GGUF→HF、F32 逆变换、V-head、`get_out_proj_activation_perm` |
| Hook | `gguf_qwen35moe_hook.py` | 拦截 `GGUFModelLoader._get_gguf_weights_map` |
| 加载 | `model_loader/loader.py` | GGUF auto 路由、arch/layers 修复 |
| 权重 | `weight_utils.py` | `GGUFUninitializedParameter` 物化顺序、跳过 MoE expert yield |
| 模型 | `models/qwen3_5.py` | VL `_checkpoint_name_to_model_param`、GGUF out_proj perm、MTP 跳过、融合专家 |
| MTP | `models/qwen3_5_mtp.py` | nextn 重映射 |
| 测试 | `test/srt/test_qwen35*.py` | 名字映射、变换、checkpoint 名、export 等 |

安装与测：`cd sglang-fork/python && pip install -e . && pytest test/srt/test_qwen35*.py -q`

## 4. 已关闭/删除的弯路

- ~~下载 70GB `deepreinforce-ai/Ornith-1.0-35B`~~ → 用 `package_gguf_bf16_for_awq`
- ~~工作区根目录重复脚本~~ → 已迁入 `scripts/ornith/`
- ~~用 AVX512 变体规避 SIGILL~~ → 根因修 setup + 重编

## 5. 未完成（接手入口）

1. `run-full-gguf-awq-pipeline.sh` → `run_sglang_marlin_ik.sh` E2E（AWQ OOM 风险）
2. GGUF 全模型：`UninitializedParameter` / `gguf.py` MoE 物化（见 `移交报告.md` 最新段）
3. `model.visual.*` — 导出未含；多模态需 mmproj/官方 visual
4. IQ2_K_R4 C++ — `docs/ornith/ik-kt-iq2_k_r4-backfill.md`

## 6. 环境常量

- venv：`.venv-public-py312`（工作区根）
- `PYTHONPATH=sglang-fork/python:kt-kernel/python`
- 权重示例：`ornith-1.0-35b-Q6_K-MTP-final.gguf`、`ornith-cpu-experts-q6k.gguf`、`ornith-gpu-non-expert.gguf`
- 导出成品：`ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors`（630 keys, ~4.6GB）

## 7. 移交报告

| 文档 | 用途 |
|------|------|
| `移交报告-2026-07-02.md` | 当日快照 |
| `移交报告.md` | 长历史链（GGUF on-the-fly、SIGILL、阻塞演进） |