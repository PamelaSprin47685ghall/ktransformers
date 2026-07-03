#!/usr/bin/env python3
"""Package GGUF-derived BF16 overlay into self-contained HF dir for AutoAWQ.

Reads a single safetensors overlay (GPU non-expert BF16 from
gguf_gpu_slice_to_hf_awq_prep) + template HF metadata dir, produces a minimal
HF directory with subset index and single shard that AutoAWQ can load.

Does NOT require the full 70 GB HF shard download — only the metadata files
(config.json, tokenizer.json, etc.) and the overlay safetensors.

Usage::

    python package_gguf_bf16_for_awq.py \\
        --overlay ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors \\
        --template Ornith-1.0-35B-hf \\
        --out-dir ornith-gpu-bf16-standalone
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_META_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "vocab.json",
)


def _embed_shape(template: Path) -> tuple[int, int] | None:
    """Return (hidden_size, vocab_size) from template config.json, or None if unknown."""
    cfg_path = template / "config.json"
    if not cfg_path.is_file():
        return None
    cfg = json.loads(cfg_path.read_text())
    tc = cfg.get("text_config", cfg)
    hs = tc.get("hidden_size") or cfg.get("hidden_size")
    vs = tc.get("vocab_size") or cfg.get("vocab_size")
    if hs is None or vs is None:
        return None
    return int(hs), int(vs)


def _maybe_transpose_vocab_embed(
    tensors: dict[str, torch.Tensor],
    hidden_size: int,
    vocab_size: int,
) -> None:
    """Transpose embed_tokens/lm_head from (hidden, vocab) to (vocab, hidden) in-place.

    SGLang VocabParallelEmbedding.weight_loader expects
    ``loaded_weight.shape[output_dim] == org_vocab_size`` (output_dim=0).
    GGUF stores these weights as (n_embd, n_vocab); HF/SGLang expects
    (n_vocab, n_embd).
    """
    for k in list(tensors.keys()):
        if not (k.endswith("embed_tokens.weight") or k.endswith("lm_head.weight")):
            continue
        t = tensors[k]
        if t.ndim == 2 and t.shape[0] == hidden_size and t.shape[1] == vocab_size:
            tensors[k] = t.t().contiguous()
            log.info("Transposed %s (%s → %s)", k, t.shape, tensors[k].shape)


def package_gguf_bf16_for_awq(overlay: Path, template: Path, out: Path) -> None:
    """Package overlay safetensors + template metadata → single-shard HF dir.

    Args:
        overlay: Path to a single ``.safetensors`` file containing BF16 tensors
            (from ``gguf_gpu_slice_to_hf_awq_prep``).
        template: Path to the HF metadata directory (``config.json``,
            ``model.safetensors.index.json``, tokenizer files, etc.).
        out: Output directory — created if not exists; receives
            ``model.safetensors``, ``model.safetensors.index.json``, and
            copied metadata files.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load overlay tensors ──────────────────────────────────────
    overlay_tensors: dict[str, "torch.Tensor"] = {}
    overlay_size = 0
    overlay_items = []
    with safe_open(str(overlay), framework="pt") as f:
        for k in f.keys():
            tensor = f.get_tensor(k)
            overlay_tensors[k] = tensor
            overlay_size += tensor.numel() * tensor.element_size()
            overlay_items.append(k)

    log.info("Loaded %d tensors (%.1f MB) from %s", len(overlay_tensors),
             overlay_size / (1024 * 1024), overlay)

    # ── 1b. Transpose embed/lm_head from (hidden, vocab) → (vocab, hidden) ──
    embed_dims = _embed_shape(template)
    if embed_dims is not None:
        _maybe_transpose_vocab_embed(overlay_tensors, *embed_dims)

    # ── 2. Copy metadata files from template ────────────────────────
    for fname in _META_FILES:
        src = template / fname
        if src.is_file():
            shutil.copy2(src, out / fname)

    # ── 3. Write single shard ────────────────────────────────────────
    shard_path = out / "model.safetensors"
    # Convert back to dict for save_file
    save_file(overlay_tensors, str(shard_path))
    log.info("Wrote %s (%d tensors)", shard_path, len(overlay_tensors))

    # ── 4. Write subset index ────────────────────────────────────────
    # Only keys present in overlay, all pointing to the single shard
    weight_map = {k: "model.safetensors" for k in overlay_items}

    idx = {
        "metadata": {"total_size": overlay_size},
        "weight_map": weight_map,
    }
    idx_path = out / "model.safetensors.index.json"
    idx_path.write_text(json.dumps(idx, indent=2))
    log.info("Wrote %s (%d keys, total_size=%d)", idx_path,
             len(weight_map), overlay_size)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Package GGUF-derived BF16 overlay into self-contained HF dir"
    )
    parser.add_argument("--overlay", required=True,
                        help="Single safetensors from gguf_gpu_slice_to_hf_awq_prep")
    parser.add_argument("--template", required=True,
                        help="HF metadata directory (config.json, tokenizer files, etc.)")
    parser.add_argument("--out-dir", required=True,
                        help="Output directory for standalone HF tree")
    args = parser.parse_args()

    package_gguf_bf16_for_awq(
        Path(args.overlay),
        Path(args.template),
        Path(args.out_dir),
    )


if __name__ == "__main__":
    main()
