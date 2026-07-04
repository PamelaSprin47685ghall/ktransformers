# Ornith 运维脚本

从工作区根执行。完整技术参数与构建流程见 `ktransformers/最终报告.md`；调试历史见 `ktransformers/更改历史.md`。

## 一键路径

```bash
# split_q6（推荐，~2GB GPU，GGUF on-the-fly）
CPU_GGUF=ornith-cpu-experts-iq2k.gguf \
  bash ktransformers/kt-kernel/scripts/ornith/run_sglang_split_q6.sh

# compressed_ik（W4 Marlin，~3.6GB GPU）
KT_WEIGHT=ornith-cpu-experts-iq2k.gguf \
  bash ktransformers/kt-kernel/scripts/ornith/run_sglang_compressed_ik.sh
```

## 脚本索引

| 脚本 | 作用 |
|------|------|
| `ornith_download_gguf.sh` | 下载主 GGUF (~28GB) |
| `extract-gpu-non-expert.sh` | GPU 非专家 GGUF 切片 (~2GB) |
| `reextract-cpu-experts.sh` | CPU 专家 GGUF |
| `run-gguf-gpu-export.sh` | GGUF→BF16 safetensors（630 keys ~4.7GB） |
| `package_gguf_bf16_for_awq.sh` | 打包 standalone HF（无 70GB 下载） |
| `run_awq_quantization.py` | datafree W4 量化 → `ornith-gpu-w4-q6-parity-from-gguf` |
| `run_sglang_split_q6.sh` | split_q6 起服（Q6 on-the-fly + kt CPU） |
| `run_sglang_compressed_ik.sh` | compressed_ik 起服（W4 Marlin + kt CPU） |
| `run_sglang_gguf.sh` | 全量 GGUF on-the-fly 起服（backlog） |
| `run_sglang_marlin_ik.sh` | Marlin GPU + kt CPU（awq_marlin 终态，backlog） |
| `curl-completion-smoke.sh` | 裸 prompt 冒烟 |
| `curl-chat-smoke.sh` | chat 冒烟 |
| `curl-prompt-next-token-logprobs.sh` | next-token top8 logprobs（验收 Paris） |
| `compare-next-token-ik-vs-sglang.sh` | ik vs sglang 对比 |
| `verify_awq_checkpoint.sh` | 量化产物校验 |
| `test-awq-e2e.sh` | AWQ E2E |

## 冒烟验收

```bash
bash curl-completion-smoke.sh 30000
bash curl-prompt-next-token-logprobs.sh 30000  # 期望含 Paris(11751)
```
