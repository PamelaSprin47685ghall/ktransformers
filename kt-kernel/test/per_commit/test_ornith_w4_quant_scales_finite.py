"""w4 mlp-only output must not store NaN/Inf in quant scale tensors."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import torch

W4 = Path("/home/kunweiz/Desktop/Ornith/ornith-gpu-w4-mlp-only-from-gguf/model.safetensors")


def test_w4_checkpoint_quant_scales_are_finite_when_present():
    if not W4.is_file():
        return

    from safetensors import safe_open

    bad = []
    with safe_open(str(W4), framework="pt") as f:
        for key in f.keys():
            if "scale" not in key and "zero_point" not in key:
                continue
            t = f.get_tensor(key)
            if t.is_floating_point() and not t.isfinite().all():
                bad.append(key)
    assert not bad, f"non-finite quant metadata: {bad[:10]}"


def test_w4_mlp_gate_weight_finite_when_present():
    if not W4.is_file():
        return
    from safetensors import safe_open

    with safe_open(str(W4), framework="pt") as f:
        k = "model.layers.0.mlp.gate.weight"
        if k not in f.keys():
            return
        g = f.get_tensor(k)
    assert g.isfinite().all()