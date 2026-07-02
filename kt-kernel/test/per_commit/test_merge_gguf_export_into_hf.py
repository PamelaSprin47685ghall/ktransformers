"""merge_gguf_export_into_hf shard replace logic."""

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def test_merge_replaces_key_in_shard(tmp_path):
    hf = tmp_path / "hf"
    hf.mkdir()
    shard = "model-00001.safetensors"
    save_file(
        {"model.language_model.layers.0.input_layernorm.weight": torch.zeros(4)},
        str(hf / shard),
    )
    (hf / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.language_model.layers.0.input_layernorm.weight": shard}})
    )
    overlay = tmp_path / "ov.safetensors"
    save_file(
        {"model.language_model.layers.0.input_layernorm.weight": torch.ones(4)},
        str(overlay),
    )
    out = tmp_path / "out"
    import importlib.util

    mod_path = Path(__file__).resolve().parents[2] / "tools" / "merge_gguf_export_into_hf.py"
    spec = importlib.util.spec_from_file_location("merge_mod", mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    merge = mod.merge

    merge(hf, overlay, out, allow_new=False)
    from safetensors import safe_open

    with safe_open(str(out / shard), framework="pt") as f:
        t = f.get_tensor("model.language_model.layers.0.input_layernorm.weight")
    assert torch.all(t == 1)