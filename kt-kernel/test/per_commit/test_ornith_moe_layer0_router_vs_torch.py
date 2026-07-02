"""Layer0 routed MoE: real gate + top8 vs dequant torch reference."""

from __future__ import annotations

import os

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from gguf import GGUFReader

CPU_GGUF = "/home/kunweiz/Desktop/Ornith/ornith-cpu-experts-q6k.gguf"
GPU_SLICE = "/home/kunweiz/Desktop/Ornith/ornith-gguf-runtime/ornith-gpu-non-expert.gguf"


def _moe_torch_ref(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    weights: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
) -> torch.Tensor:
    expert_num = gate.shape[0]
    cnts = expert_ids.new_zeros((expert_ids.shape[0], expert_num))
    cnts.scatter_(1, expert_ids, 1)
    idxs = expert_ids.reshape(-1).argsort()
    sorted_tokens = x[idxs // expert_ids.shape[1]]
    outputs = []
    start = 0
    for e, n in enumerate(cnts.sum(dim=0).tolist()):
        if n == 0:
            continue
        end = start + n
        h = sorted_tokens[start:end]
        g = F.silu(h @ gate[e].T) * (h @ up[e].T)
        outputs.append(g @ down[e].T)
        start = end
    outs = torch.cat(outputs, dim=0)
    new_x = torch.empty_like(outs)
    new_x[idxs] = outs
    return (
        new_x.view(*expert_ids.shape, -1)
        .float()
        .mul_(weights.unsqueeze(-1))
        .sum(dim=1)
    )


def _dequant_expert_stack(path: str, layer: int, proj: str) -> torch.Tensor:
    gt = f"blk.{layer}.ffn_{proj}_exps.weight"
    t = next(x for x in GGUFReader(path).tensors if x.name == gt)
    raw = np.array(t.data, dtype=np.uint8)
    w = torch.from_numpy(np.array(gguf.dequantize(raw, t.tensor_type))).float()
    return w


@pytest.mark.skipif(
    not os.path.isfile(CPU_GGUF) or not os.path.isfile(GPU_SLICE),
    reason="need cpu experts + gpu slice gguf",
)
def test_layer0_router_top8_kt_near_torch_dequant():
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
    from sglang.srt.model_loader.gguf_qwen35moe import qwen35moe_gguf_to_hf
    from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator
    from sglang.srt.model_loader.gguf_qwen35moe import qwen35moe_linear_attn_vcfg

    cfg = qwen35moe_linear_attn_vcfg(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    gt_gate = "blk.0.ffn_gate_inp.weight"
    hf_gate = qwen35moe_gguf_to_hf(gt_gate).replace("model.language_model.", "model.")
    gate_w = dict(
        gguf_quant_weights_iterator(GPU_SLICE, {gt_gate: hf_gate}, qwen35_linear_attn_vcfg=cfg)
    )[hf_gate].float()

    torch.manual_seed(42)
    x = torch.randn(4, 2048, dtype=torch.bfloat16, device="cuda")
    logits = x.float() @ gate_w.T.cuda()
    scores = F.softmax(logits, dim=-1)
    topw, topi = torch.topk(scores, 8, dim=-1)
    topw = topw / topw.sum(dim=-1, keepdim=True)

    gate = _dequant_expert_stack(CPU_GGUF, 0, "gate")
    up = _dequant_expert_stack(CPU_GGUF, 0, "up")
    down = _dequant_expert_stack(CPU_GGUF, 0, "down")
    ref = _moe_torch_ref(x.float().cpu(), topi.cpu(), topw.cpu(), gate, up, down)

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
    y = w.forward(x, topi, topw.float(), torch.cuda.current_stream()).float().cpu()

    rel = (y - ref).abs().mean() / ref.abs().mean().clamp(min=1e-6)
    assert rel < 0.35, f"mean rel err {rel.item():.4f}"