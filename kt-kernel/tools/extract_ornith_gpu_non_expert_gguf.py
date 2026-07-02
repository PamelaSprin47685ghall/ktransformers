#!/usr/bin/env python3
"""Extract non-expert tensors from Qwen3.5 MoE GGUF (GPU / AWQ-prep slice).

Skips routed MoE: ``ffn_gate_exps``, ``ffn_up_exps``, ``ffn_down_exps``.
Keeps attn, SSM, shared, embed, output, norms, router biases, nextn, etc.

Use for offline dequant → BF16 safetensors → AutoAWQ (see scripts/export-gpu-awq-marlin.md).
Not loaded by SGLang awq_marlin directly (still GGUF bytes).

Usage::

    python extract_ornith_gpu_non_expert_gguf.py \\
        --src /path/to/model.gguf \\
        --dst ornith-gpu-non-expert.gguf
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from kt_kernel.utils.gguf_raw_reader import _DEFAULT_ALIGNMENT, _read_gguf_full

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

_EXPERT_SUFFIXES = (
    ".ffn_gate_exps.weight",
    ".ffn_up_exps.weight",
    ".ffn_down_exps.weight",
)


def _is_gpu_tensor(name: str) -> bool:
    return not any(name.endswith(s) for s in _EXPERT_SUFFIXES)


def _align(offset: int, alignment: int = _DEFAULT_ALIGNMENT) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _build_tensor_index_entry(name: str, logical_shape: Tuple[int, ...], dtype_id: int) -> bytes:
    name_b = name.encode("utf-8")
    n_dims = len(logical_shape)
    buf = struct.pack("<Q", len(name_b)) + name_b
    buf += struct.pack("<I", n_dims)
    for d in reversed(logical_shape):
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", dtype_id)
    buf += struct.pack("<Q", 0)
    return buf


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract non-expert GGUF tensors")
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    args = parser.parse_args()
    src_path = Path(args.src)
    dst_path = Path(args.dst)
    if not src_path.is_file():
        log.error("missing %s", src_path)
        sys.exit(1)

    log.info("Reading %s ...", src_path)
    src_idx, raw_kv_bytes, alignment = _read_gguf_full(str(src_path))
    selected = sorted(n for n in src_idx if _is_gpu_tensor(n))
    if not selected:
        log.error("no tensors selected")
        sys.exit(1)
    log.info("Selected %d / %d tensors", len(selected), len(src_idx))

    import numpy as np

    src_mmap = np.memmap(src_path, mode="r")
    tensor_meta = [(n, src_idx[n]) for n in selected]
    total_payload = sum(ti.n_bytes for _, ti in tensor_meta)

    kv_count = 0
    offs = 0
    while offs < len(raw_kv_bytes):
        key_len = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]
        offs += 8 + key_len
        val_type = struct.unpack_from("<I", raw_kv_bytes, offs)[0]
        offs += 4
        if val_type == 0:
            offs += 1
        elif val_type == 1:
            offs += 1
        elif val_type in (2, 3):
            offs += 2
        elif val_type in (4, 5, 6):
            offs += 4
        elif val_type == 7:
            offs += 1
        elif val_type == 8:
            slen = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]
            offs += 8 + slen
        elif val_type == 9:
            arr_type = struct.unpack_from("<I", raw_kv_bytes, offs)[0]
            offs += 4
            arr_len = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]
            offs += 8
            for _ in range(arr_len):
                if arr_type == 8:
                    slen = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]
                    offs += 8 + slen
                elif arr_type in (0, 1, 7):
                    offs += 1
                elif arr_type in (2, 3):
                    offs += 2
                elif arr_type in (4, 5, 6):
                    offs += 4
                elif arr_type in (10, 11, 12):
                    offs += 8
                else:
                    raise ValueError(arr_type)
        elif val_type in (10, 11, 12):
            offs += 8
        else:
            raise ValueError(val_type)
        kv_count += 1

    header = struct.pack("<I", 0x46554747)
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", len(tensor_meta))
    header += struct.pack("<Q", kv_count)

    ti_templates = [_build_tensor_index_entry(n, ti.shape, ti.dtype_id) for n, ti in tensor_meta]
    ti_section = b"".join(ti_templates)
    data_start = _align(len(header) + len(raw_kv_bytes) + len(ti_section), alignment)

    rel_offs = 0
    ti_fixed: List[bytes] = []
    for (_, ti), tmpl in zip(tensor_meta, ti_templates):
        ti_fixed.append(tmpl[:-8] + struct.pack("<Q", rel_offs))
        rel_offs = _align(rel_offs + ti.n_bytes, alignment)

    with dst_path.open("wb") as f:
        f.write(header)
        f.write(raw_kv_bytes)
        for entry in ti_fixed:
            f.write(entry)
        pad = data_start - f.tell()
        if pad > 0:
            f.write(b"\x00" * pad)
        for _, ti in tensor_meta:
            chunk = src_mmap[ti.data_offset : ti.data_offset + ti.n_bytes]
            f.write(bytes(chunk))
            pad = _align(ti.n_bytes, alignment) - ti.n_bytes
            if pad > 0:
                f.write(b"\x00" * pad)

    log.info("Done %.1f MB payload → %s", total_payload / (1024**2), dst_path)


if __name__ == "__main__":
    main()