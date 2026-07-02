"""GPU non-expert GGUF slice name filter."""

import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "extract_ornith_gpu_non_expert_gguf.py"
    )
    spec = importlib.util.spec_from_file_location("extract_gpu", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_is_gpu_tensor():
    m = _load_module()
    assert m._is_gpu_tensor("blk.0.attn_q.weight")
    assert m._is_gpu_tensor("blk.0.ffn_gate_inp.weight")
    assert not m._is_gpu_tensor("blk.0.ffn_gate_exps.weight")
    assert not m._is_gpu_tensor("blk.0.ffn_up_exps.weight")
    assert not m._is_gpu_tensor("blk.0.ffn_down_exps.weight")