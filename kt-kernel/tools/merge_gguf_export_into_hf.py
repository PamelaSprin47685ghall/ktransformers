#!/usr/bin/env python3
"""Overwrite HF safetensor shards with keys from GGUF export (GGUF→AWQ prep).

Does not re-shard: each key in ``--overlay`` replaces the same key in the shard
listed in ``model.safetensors.index.json``. Unlisted overlay keys are skipped
with a warning unless ``--allow-new-keys``.

Usage::

    python merge_gguf_export_into_hf.py \\
      --hf-dir /path/to/Ornith-1.0-35B \\
      --overlay ornith-gpu-bf16-from-gguf/model-gpu-from-gguf.safetensors \\
      --out-dir ornith-gpu-bf16-merged
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def _load_index(hf_dir: Path) -> dict[str, str]:
    idx_path = hf_dir / "model.safetensors.index.json"
    if not idx_path.is_file():
        raise FileNotFoundError(idx_path)
    data = json.loads(idx_path.read_text())
    return data["weight_map"]


def _shard_tensors(shard_path: Path) -> dict:
    out = {}
    with safe_open(str(shard_path), framework="pt") as f:
        for k in f.keys():
            out[k] = f.get_tensor(k)
    return out


def merge(hf_dir: Path, overlay_path: Path, out_dir: Path, allow_new: bool) -> None:
    weight_map = _load_index(hf_dir)
    shard_names = sorted(set(weight_map.values()))
    out_dir.mkdir(parents=True, exist_ok=True)

    for meta in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "chat_template.jinja",
        "vocab.json",
        "model.safetensors.index.json",
    ):
        src = hf_dir / meta
        if src.is_file():
            shutil.copy2(src, out_dir / meta)

    overlay: dict = {}
    with safe_open(str(overlay_path), framework="pt") as f:
        for k in f.keys():
            overlay[k] = f.get_tensor(k)
    log.info("Overlay %d keys from %s", len(overlay), overlay_path)

    replaced = 0
    skipped_new = 0
    by_shard: dict[str, dict] = {}
    for key, shard in weight_map.items():
        by_shard.setdefault(shard, {})

    for shard in shard_names:
        src_shard = hf_dir / shard
        if not src_shard.is_file():
            raise FileNotFoundError(src_shard)
        tensors = _shard_tensors(src_shard)
        for key in list(tensors):
            if key in overlay:
                tensors[key] = overlay[key].contiguous().cpu()
                replaced += 1
        out_shard = out_dir / shard
        save_file(tensors, str(out_shard))
        log.info("Wrote %s (%d tensors)", out_shard.name, len(tensors))

    for key in overlay:
        if key not in weight_map:
            if allow_new:
                log.warning("New key not in index (ignored unless you extend index): %s", key)
            else:
                skipped_new += 1

    log.info("Replaced %d keys; %d overlay keys not in HF index", replaced, skipped_new)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-dir", required=True, help="Official BF16 tree with shards + index")
    p.add_argument("--overlay", required=True, help="Single safetensors from gguf_gpu_slice_to_hf_awq_prep")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--allow-new-keys", action="store_true")
    args = p.parse_args()
    merge(Path(args.hf_dir), Path(args.overlay), Path(args.out_dir), args.allow_new_keys)


if __name__ == "__main__":
    main()