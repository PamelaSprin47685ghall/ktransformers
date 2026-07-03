"""exported mlp.gate must match gguf_quant_weights_iterator numerically.

Background: ``blk.0.ffn_gate_inp.weight`` is F32, logical shape ``(256, 2048)``.
The iterator yields the tensor in logical layout.  The export path
``gguf_gpu_slice_to_hf_awq_prep._tensor_to_torch`` must preserve that layout
for F32 — old artifacts transposed to ``(2048, 256)`` via ``reshape(logical)``
on a ``(2048, 256)`` raw view, producing values that differ rel~1.43.

Acceptance:
- If ``ornith-gguf-runtime/ornith-gpu-non-expert.gguf`` is missing → skip.
- Iterator gate shape must be ``(256, 2048)``.
- If ``ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors`` exists:
  exported key ``model.language_model.layers.0.mlp.gate.weight`` must have
  shape ``(256, 2048)`` and be allclose to iterator (rtol 0.02, atol 0.05).
- If only ``ornith-gpu-bf16-standalone/model.safetensors`` exists with the
  wrong shape ``(2048, 256)``: fail with a message documenting the need to
  re-export with the fixed ``_tensor_to_torch``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[4]
GGUF_PATH = REPO / "ornith-gguf-runtime" / "ornith-gpu-non-expert.gguf"
EXPORT_SAFETENSORS = REPO / "ornith-gpu-bf16-from-gguf" / "model-gpu-from-gguf.safetensors"
STANDALONE_SAFETENSORS = REPO / "ornith-gpu-bf16-standalone" / "model.safetensors"

GGUF_TENSOR = "blk.0.ffn_gate_inp.weight"
HF_NAME = "model.layers.0.mlp.gate.weight"
EXPORT_KEY = "model.language_model.layers.0.mlp.gate.weight"
EXPECTED_SHAPE = (256, 2048)


def _skip_if_no_gguf():
    if not GGUF_PATH.is_file():
        import pytest

        pytest.skip(f"GGUF slice missing: {GGUF_PATH}")


def _iterator_gate():
    from sglang.srt.model_loader.gguf_qwen35moe import qwen35moe_linear_attn_vcfg
    from sglang.srt.model_loader.weight_utils import gguf_quant_weights_iterator

    cfg = qwen35moe_linear_attn_vcfg(
        linear_num_key_heads=16,
        linear_num_value_heads=32,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
    )
    items = dict(
        gguf_quant_weights_iterator(
            str(GGUF_PATH),
            {GGUF_TENSOR: HF_NAME},
            qwen35_linear_attn_vcfg=cfg,
        )
    )
    return items[HF_NAME]


def test_iterator_gate_shape():
    _skip_if_no_gguf()
    w = _iterator_gate()
    assert w.shape == EXPECTED_SHAPE, f"iterator gate shape {w.shape} != {EXPECTED_SHAPE}"


def test_exported_gate_matches_iterator():
    _skip_if_no_gguf()
    if not EXPORT_SAFETENSORS.is_file() and not STANDALONE_SAFETENSORS.is_file():
        import pytest

        pytest.skip(
            f"Neither {EXPORT_SAFETENSORS} nor {STANDALONE_SAFETENSORS} present"
        )

    from safetensors import safe_open

    w_iter = _iterator_gate().float()

    if EXPORT_SAFETENSORS.is_file():
        with safe_open(str(EXPORT_SAFETENSORS), framework="pt") as f:
            assert EXPORT_KEY in f.keys(), (
                f"{EXPORT_KEY} missing in export; available gate-like: "
                + str([k for k in f.keys() if "gate" in k][:10])
            )
            w_exp = f.get_tensor(EXPORT_KEY).float()
        assert w_exp.shape == EXPECTED_SHAPE, (
            f"exported gate shape {w_exp.shape} != {EXPECTED_SHAPE}; "
            "old artifact transposed — re-export with fixed _tensor_to_torch"
        )
        assert torch.allclose(w_exp, w_iter, rtol=0.02, atol=0.05), (
            f"exported gate differs from iterator; "
            f"max|diff|={(w_exp - w_iter).abs().max().item():.4f}"
        )
        return

    # Only standalone exists — check if it carries the wrong shape
    if STANDALONE_SAFETENSORS.is_file():
        with safe_open(str(STANDALONE_SAFETENSORS), framework="pt") as f:
            gate_keys = [k for k in f.keys() if k.endswith(".mlp.gate.weight")]
            assert gate_keys, "no .mlp.gate.weight key in standalone"
            w_sa = f.get_tensor(gate_keys[0]).float()
        if w_sa.shape != EXPECTED_SHAPE:
            raise AssertionError(
                f"standalone gate shape {w_sa.shape} != {EXPECTED_SHAPE}; "
                "old artifact transposed — re-export with fixed _tensor_to_torch "
                f"(iterator shape is {EXPECTED_SHAPE})"
            )
        assert torch.allclose(w_sa, w_iter, rtol=0.02, atol=0.05)
