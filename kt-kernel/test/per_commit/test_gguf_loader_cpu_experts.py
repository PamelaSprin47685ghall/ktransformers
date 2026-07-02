"""GGUFLoader must mmap expert-only GGUF (relative TI offsets)."""

import pytest

from kt_kernel.utils.loader import GGUFLoader

CPU_EXPERTS = "/home/kunweiz/Desktop/Ornith/ornith-cpu-experts-q6k.gguf"


@pytest.mark.skipif(
    not __import__("os").path.isfile(CPU_EXPERTS),
    reason="run extract_ornith_cpu_experts_gguf.py first",
)
def test_load_gate_exps_q6_k():
    loader = GGUFLoader(CPU_EXPERTS)
    data, ggml_type = loader.get_undequanted_tensor_and_ggml_type(
        "blk.0.ffn_gate_exps.weight"
    )
    assert int(ggml_type) == 14  # Q6_K
    assert data.numel() > 0