"""package_gguf_bf16_for_awq: subset index + single shard from overlay only.

The function takes a single safetensors overlay (GPU non-expert BF16) + template HF
metadata dir and produces a minimal self-contained HF directory loadable by AutoAWQ.
It must NOT require the full 70GB HF shard download.

This test verifies the index-subsetting behaviour: given a template ``weight_map``
with 3 keys but an overlay that only contains 2 of them, the output must contain
exactly 1 shard whose ``model.safetensors.index.json`` only maps the overlay keys.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "tools"
        / "package_gguf_bf16_for_awq.py"
    )
    import importlib.util

    spec = importlib.util.spec_from_file_location("package_mod", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_subset_index_and_single_shard_from_overlay(tmp_path):
    # ── template HF metadata dir ──────────────────────────────────
    template = tmp_path / "template"
    template.mkdir()
    (template / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "num_hidden_layers": 40,
                    "linear_num_key_heads": 16,
                    "linear_num_value_heads": 32,
                    "linear_key_head_dim": 128,
                    "linear_value_head_dim": 128,
                },
            }
        )
    )
    # template index with 3 keys pointing to hypothetical shards
    (template / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 999},
                "weight_map": {
                    "model.layers.0.input_layernorm.weight": "model-00001.safetensors",
                    "model.layers.0.self_attn.qkv_proj.weight": "model-00001.safetensors",
                    "model.layers.0.mlp.gate_proj.weight": "model-00002.safetensors",
                },
            }
        )
    )

    # ── overlay safetensors (2 of the 3 keys above) ──────────────
    overlay = tmp_path / "overlay.safetensors"
    save_file(
        {
            "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.bfloat16),
            "model.layers.0.self_attn.qkv_proj.weight": torch.full(
                (4, 4), 2.0, dtype=torch.bfloat16
            ),
        },
        str(overlay),
    )

    mod = _load_module()  # will fail: module does not exist yet
    out = tmp_path / "out"
    mod.package_gguf_bf16_for_awq(overlay, template, out)

    # ── assertions ────────────────────────────────────────────────
    # exactly one shard (single file, not multiple)
    shard_path = out / "model.safetensors"
    assert shard_path.is_file(), f"missing {shard_path}"
    shards = list(out.glob("model-*.safetensors"))
    assert len(shards) == 0, (
        f"got multi-shard files: {[s.name for s in shards]}; "
        "expected single model.safetensors only"
    )

    # index only maps overlay keys, not the 3rd template-only key
    idx = json.loads((out / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    assert set(wm.keys()) == {
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.qkv_proj.weight",
    }, f"weight_map keys mismatch: {set(wm.keys())}"
    for v in wm.values():
        assert v == "model.safetensors", f"key mapped to unexpected shard: {v}"

    # tensor values match overlay
    from safetensors import safe_open

    with safe_open(str(shard_path), framework="pt") as f:
        keys_in_shard = set(f.keys())
    assert keys_in_shard == set(wm.keys()), f"shard keys mismatch: {keys_in_shard}"

    with safe_open(str(shard_path), framework="pt") as f:
        t1 = f.get_tensor("model.layers.0.input_layernorm.weight")
    assert torch.all(t1 == 1.0), f"layernorm: {t1}"

    with safe_open(str(shard_path), framework="pt") as f:
        t2 = f.get_tensor("model.layers.0.self_attn.qkv_proj.weight")
    assert torch.all(t2 == 2.0), f"qkv_proj: {t2}"

    # config.json and other metadata should be copied
    assert (out / "config.json").is_file(), "config.json not copied"
