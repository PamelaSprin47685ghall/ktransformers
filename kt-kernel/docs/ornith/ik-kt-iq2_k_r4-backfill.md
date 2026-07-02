# kt CPU 轨：IQ2_K_R4 (337) 回填

Python 侧已就绪：`gguf_ik_types` / `gguf_raw_reader` / `GGUFLoader` fallback。

## 阻塞点

`third_party/llamafile/iqk_mul_mat.inc` 的 `MulMat::set_mul_mat` **无** `GGML_TYPE_IQ2_K_R4`。  
`gate_type=337` → `default: return false`。

## ik 参考

| 项 | ik 路径 |
|----|---------|
| `block_iq2_k_r4` | `ik_llama.cpp/ggml/src/ggml-common.h` |
| `GGML_TYPE_IQ2_K_R4 = 337` | `ik_llama.cpp/ggml/include/ggml.h` |
| MoE matmul | `iqk_gemm_iqk_quants.cpp` |

## 验收

- `test/per_commit/test_iqk_iq2_k_r4_switch.py` 回填后断言 337 存在。
- 当前 Q6_K CPU 专家轨不依赖此项。