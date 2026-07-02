# Ornith 运维脚本

从 Ornith 工作区根目录执行；自动探测 `ORNITH_ROOT`（`ktransformers` 的上一级）。

| 脚本 | 作用 |
|------|------|
| `ornith_download_gguf.sh` | 下载主 GGUF (~28GB) |
| `extract-gpu-non-expert.sh` | GPU 非专家 GGUF 切片 |
| `reextract-cpu-experts.sh` | CPU 专家 GGUF |
| `run-gguf-gpu-export.sh` | GGUF→BF16 safetensors（630 keys） |
| `package_gguf_bf16_for_awq.sh` | 打包 standalone HF（无 70GB） |
| `run-autoawq-ornith-gpu.sh` | AutoAWQ |
| `run-full-gguf-awq-pipeline.sh` | export→package→awq |
| `run_sglang_marlin_ik.sh` | Marlin GPU + kt CPU MoE |
| `run_sglang_gguf.sh` | on-the-fly GGUF 起服 |

文档：`kt-kernel/docs/ornith/export-gpu-awq-marlin.md`