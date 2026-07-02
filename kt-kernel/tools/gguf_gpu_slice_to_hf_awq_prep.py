#!/usr/bin/env python3
"""GGUF (GPU non-expert slice) → HF safetensors for AutoAWQ / awq_marlin.

Follows GGUF_on_the_fly.txt: dequant Q6_K, map names, apply_gguf_to_hf_weight.
Skips routed experts (already omitted in slice). Optionally copies MTP from
``--mtp-source`` full GGUF nextn.* tensors.

Usage::

    python gguf_gpu_slice_to_hf_awq_prep.py \\
      --gguf ornith-gpu-non-expert.gguf \\
      --hf-template Ornith-1.0-35B-hf \\
      --out-dir ornith-gpu-bf16-stub \\
      --dtype bfloat16 \\
      --mtp-source ornith-1.0-35b-Q6_K-MTP-final.gguf
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import gguf
import numpy as np
import torch
from gguf import GGMLQuantizationType
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def _load_text_config(template_dir: Path) -> dict:
    cfg = json.loads((template_dir / "config.json").read_text())
    tc = cfg.get("text_config", cfg)
    k_heads = int(tc["linear_num_key_heads"])
    v_heads = int(tc["linear_num_value_heads"])
    return {
        "k_heads": k_heads,
        "num_v_per_k": v_heads // k_heads,
        "num_value_heads": v_heads,
        "head_k_dim": int(tc["linear_key_head_dim"]),
        "head_v_dim": int(tc["linear_value_head_dim"]),
        "num_hidden_layers": int(tc["num_hidden_layers"]),
    }


def _flat_to_vl(hf_flat: str) -> str:
    if hf_flat.startswith("model.layers."):
        return hf_flat.replace("model.layers.", "model.language_model.layers.", 1)
    if hf_flat.startswith("model.embed_tokens"):
        return hf_flat.replace("model.embed_tokens", "model.language_model.embed_tokens", 1)
    if hf_flat.startswith("model.norm"):
        return hf_flat.replace("model.norm", "model.language_model.norm", 1)
    if hf_flat.startswith("lm_head."):
        return hf_flat
    if hf_flat.startswith("model.lm_head."):
        return hf_flat.replace("model.lm_head.", "lm_head.", 1)
    return hf_flat


def _tensor_to_torch(tensor, dtype: torch.dtype) -> torch.Tensor:
    name = tensor.name
    ttype = tensor.tensor_type
    if ttype == GGMLQuantizationType.F32:
        w = torch.from_numpy(np.array(tensor.data, copy=True)).to(dtype)
    elif ttype == GGMLQuantizationType.F16:
        w = torch.from_numpy(np.array(tensor.data, copy=True)).to(torch.float16).to(dtype)
    elif ttype.name == "Q6_K" or ttype == GGMLQuantizationType.Q6_K:
        raw = np.array(tensor.data, dtype=np.uint8)
        w = torch.from_numpy(gguf.quants.dequantize(raw, ttype)).to(dtype)
    else:
        try:
            raw = np.array(tensor.data, dtype=np.uint8)
            w = torch.from_numpy(gguf.quants.dequantize(raw, ttype)).to(dtype)
        except Exception as exc:
            raise RuntimeError(f"dequant failed for {name} type {ttype}") from exc
    logical = tuple(int(x) for x in tensor.shape)
    if w.ndim == 2 and w.shape != logical and w.shape == logical[::-1]:
        return w.contiguous()
    return w.reshape(logical)


def _import_transforms():
    from sglang.srt.model_loader.gguf_qwen35moe import (
        apply_gguf_to_hf_weight,
        qwen35moe_gguf_to_hf,
    )

    return apply_gguf_to_hf_weight, qwen35moe_gguf_to_hf


def _iter_mtp_tensors(path: str):
    reader = gguf.GGUFReader(path)
    for t in reader.tensors:
        if t.name.startswith("nextn.") or ".nextn." in t.name:
            yield t


def export_slice(
    gguf_path: Path,
    template_dir: Path,
    out_dir: Path,
    dtype: torch.dtype,
    mtp_source: Optional[Path],
    max_tensors: int = 0,
) -> None:
    apply_gguf_to_hf_weight, qwen35moe_gguf_to_hf = _import_transforms()
    vcfg = _load_text_config(template_dir)
    reader = gguf.GGUFReader(str(gguf_path))
    out_dir.mkdir(parents=True, exist_ok=True)

    for fname in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "vocab.json",
    ):
        src = template_dir / fname
        if src.is_file():
            shutil.copy2(src, out_dir / fname)

    state: Dict[str, torch.Tensor] = {}
    n = 0
    for tensor in reader.tensors:
        if max_tensors and n >= max_tensors:
            break
        gguf_name = tensor.name
        if (
            gguf_name.endswith(".ffn_gate_exps.weight")
            or gguf_name.endswith(".ffn_up_exps.weight")
            or gguf_name.endswith(".ffn_down_exps.weight")
        ):
            continue
        hf_flat = qwen35moe_gguf_to_hf(gguf_name)
        if hf_flat is None:
            log.warning("unmap %s", gguf_name)
            continue
        hf_name = _flat_to_vl(hf_flat)
        w = _tensor_to_torch(tensor, dtype)
        w = apply_gguf_to_hf_weight(w, hf_name, vcfg)
        state[hf_name] = w.contiguous().cpu()
        n += 1
        if n % 50 == 0:
            log.info("exported %d tensors", n)

    if mtp_source and mtp_source.is_file():
        log.info("MTP/nextn from %s (not in GPU slice)", mtp_source)
        for t in _iter_mtp_tensors(str(mtp_source)):
            hf_flat = qwen35moe_gguf_to_hf(t.name)
            if hf_flat is None:
                continue
            hf_name = _flat_to_vl(hf_flat)
            if hf_name in state:
                continue
            w = _tensor_to_torch(t, dtype)
            w = apply_gguf_to_hf_weight(w, hf_name, vcfg)
            state[hf_name] = w.contiguous().cpu()

    out_file = out_dir / "model-gpu-from-gguf.safetensors"
    save_file(state, str(out_file))
    log.info("Wrote %s (%d tensors)", out_file, len(state))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--gguf", required=True)
    p.add_argument("--hf-template", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    p.add_argument("--mtp-source", default=None)
    p.add_argument("--max-tensors", type=int, default=0)
    args = p.parse_args()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    export_slice(
        Path(args.gguf),
        Path(args.hf_template),
        Path(args.out_dir),
        dtype,
        Path(args.mtp_source) if args.mtp_source else None,
        args.max_tensors,
    )


if __name__ == "__main__":
    main()