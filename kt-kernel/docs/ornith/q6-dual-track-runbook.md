# Q6 单文件双轨跑通（GPU Marlin + CPU ik）

约束：只用 `ornith-1.0-35b-Q6_K-MTP-final.gguf`，禁止下载官方 70GB BF16 safetensors。

## 数据流

```
Q6 全量 GGUF
  ├─ extract-gpu-non-expert.sh → ornith-gpu-non-expert.gguf (~2GB)
  │     └─ run-gguf-gpu-export.sh → model-gpu-from-gguf.safetensors (~4.6GB BF16)
  │           └─ package_gguf_bf16_for_awq.sh → ornith-gpu-bf16-standalone/
  │                 └─ run-autoawq-ornith-gpu.sh → ornith-gpu-awq-from-gguf/
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

Marlin 轨（需 AWQ 目录）在 AutoAWQ 支持 MoE 前阻塞；可先完成 `run-gguf-gpu-export.sh` + `package` 备料。

```bash
# 曾规划：run-full-gguf-awq-pipeline.sh → run_sglang_marlin_ik.sh
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
2. `POST /v1/chat/completions` 短英文 prompt；`embed_tokens` 须 `quant_config=gguf` 才有 `qweight`，否则乱码
3. 日志无 `SIGILL`、无 expert 双重加载、`Parameter model.embed_tokens` not found

## 备选 on-the-fly

`run_sglang_gguf.sh`：单 GGUF + kt CPU 专家；8GB 非专家仍可能 OOM，优先 Marlin 轨。

## 构建 kt（ik ggml）

```bash
cd ktransformers/kt-kernel
CPUINFER_USE_CUDA=0 CPUINFER_CPU_INSTRUCT=AVX2 CPUINFER_FORCE_REBUILD=1 \
  uv pip install --python $ORNITH_ROOT/.venv-public-py312/bin/python -e .
```

`llamafile` 与 ik ggml API 差异由 `third_party/llamafile/kt_ggml_compute_compat.h` 桥接。