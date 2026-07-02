"""Export must not transpose dequant when logical shape is reversed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from gguf import GGMLQuantizationType


def _load_prep():
    path = Path(__file__).resolve().parents[2] / "tools" / "gguf_gpu_slice_to_hf_awq_prep.py"
    spec = importlib.util.spec_from_file_location("prep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_tensor_to_torch_keeps_dequant_when_logical_is_transpose():
    prep = _load_prep()

    class T:
        name = "blk.0.ffn_down_shexp.weight"
        tensor_type = GGMLQuantizationType.F32
        data = np.arange(2048 * 512, dtype=np.float32).reshape(2048, 512)
        shape = [512, 2048]

    w = prep._tensor_to_torch(T(), torch.bfloat16)
    assert w.shape == (2048, 512)