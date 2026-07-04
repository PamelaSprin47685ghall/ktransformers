# Ornith 运维脚本

从工作区根执行；`HF_TEMPLATE` 默认 `kt-kernel/fixtures/Ornith-1.0-35B-hf`（环境变量 `TEMPLATE` 可覆盖）。

| 脚本 | 作用 |
|------|------|
| `ornith_download_gguf.sh` | 下载主 GGUF (~28GB) |
| `extract-gpu-non-expert.sh` | GPU 非专家 GGUF 切片 |
| `reextract-cpu-experts.sh` | CPU 专家 GGUF |
| `run-gguf-gpu-export.sh` | GGUF→BF16 safetensors（630 keys） |
| `package_gguf_bf16_for_awq.sh` | 打包 standalone HF（无 70GB） |
| `run-autoawq-ornith-gpu.sh` | AutoAWQ |
| `run-full-gguf-awq-pipeline.sh` | export→package→awq |
| `run_sglang_marlin_ik.sh` | Marlin GPU + kt CPU MoE（awq_marlin 终态） |
| `run_sglang_compressed_ik.sh` | compressed-tensors w4 mlp-only + kt（8GB 过渡） |
| `run_awq_quantization.py` | 生成 `ornith-gpu-w4-mlp-only-from-gguf` |
| `curl-chat-smoke.sh` | health + 短 chat 冒烟 |
| `run_sglang_gguf.sh` | on-the-fly GGUF 起服 |

文档：`export-gpu-awq-marlin.md`、`marlin-mlp-only-runbook.md`、`q6-dual-track-runbook.md`