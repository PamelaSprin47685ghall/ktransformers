"""Layer0 Llamafile MOE forward smoke (CPU experts GGUF)."""

import os

import pytest
import torch

CPU_GGUF = "/home/kunweiz/Desktop/Ornith/ornith-cpu-experts-q6k.gguf"


@pytest.mark.skipif(not os.path.isfile(CPU_GGUF), reason="need ornith-cpu-experts-q6k.gguf")
def test_layer0_moe_forward_finite():
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper

    mask = torch.zeros(256, dtype=torch.bool)
    w = LlamafileMoEWrapper(
        layer_idx=0,
        num_experts=256,
        num_experts_per_tok=8,
        hidden_size=2048,
        moe_intermediate_size=512,
        gpu_experts_mask=mask,
        cpuinfer_threads=4,
        threadpool_count=1,
        weight_path=CPU_GGUF,
        chunked_prefill_size=512,
        method="LLAMAFILE",
    )
    w.load_weights(torch.arange(256, dtype=torch.int32))
    x = torch.randn(2, 2048, dtype=torch.bfloat16, device="cuda")
    ids = torch.zeros(2, 8, dtype=torch.long, device="cuda")
    weights = torch.full((2, 8), 1.0 / 8, device="cuda", dtype=torch.float32)
    y = w.forward(x, ids, weights, torch.cuda.current_stream())
    assert y.shape == x.shape
    assert torch.isfinite(y.float()).all()