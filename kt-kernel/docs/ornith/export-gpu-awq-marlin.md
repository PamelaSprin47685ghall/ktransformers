# GPU：GGUF 反推 AWQ → SGLang awq_marlin

权威：同目录 `GGUF_on_the_fly.txt`；HF 模板默认 `kt-kernel/fixtures/Ornith-1.0-35B-hf`。代码：`sglang` 的 `gguf_qwen35moe` + `tools/gguf_gpu_slice_to_hf_awq_prep.py`。

**脚本目录**：`kt-kernel/scripts/ornith/`（`ORNITH_ROOT` 默认 = 含 `sglang-fork` 与 `ktransformers` 的工作区）。

## 双轨

- **GPU**：AWQ/Marlin（非专家 + shared + MTP 等，来自 GGUF 导出）
- **CPU**：`ornith-cpu-experts-q6k.gguf` + `SGLANG_KT_BYPASS_GPU_MOE=1`

## 无 70GB 官方 BF16

1. `bash kt-kernel/scripts/ornith/extract-gpu-non-expert.sh`
2. `bash kt-kernel/scripts/ornith/run-gguf-gpu-export.sh` → `ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors`（~630 keys）
3. `bash kt-kernel/scripts/ornith/package_gguf_bf16_for_awq.sh` → `ornith-gpu-bf16-standalone/`
4. `pip install autoawq` → `bash kt-kernel/scripts/ornith/run-autoawq-ornith-gpu.sh`
5. `bash kt-kernel/scripts/ornith/run_sglang_marlin_ik.sh`

或一步：`bash kt-kernel/scripts/ornith/run-full-gguf-awq-pipeline.sh`

## 备选：on-the-fly GGUF

`bash kt-kernel/scripts/ornith/run_sglang_gguf.sh`（`ornith-gguf-runtime` + `--load-format gguf`）

## 注意

- `merge_gguf_export_into_hf.py` 仅当**已有**官方 16 分片 BF16 时用于覆写；**禁止**再跑 70GB 下载脚本（已移除）。
- 8GB 显存上 35B VL AutoAWQ 可能 OOM。
- `model.visual.*` 不在 GPU 切片导出中；多模态需 mmproj/官方 visual 权重另行规划。