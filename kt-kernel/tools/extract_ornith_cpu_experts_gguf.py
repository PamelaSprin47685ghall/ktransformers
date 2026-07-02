#!/usr/bin/env python3
"""Extract CPU-only expert tensors from a Qwen3.5 MoE GGUF file.

The output GGUF contains only the tensors that kt-kernel's ``LlamafileMoEWrapper``
needs on the CPU side:

  - ``ffn_gate_exps.weight``  (routed expert gate proj)
  - ``ffn_up_exps.weight``    (routed expert up proj)
  - ``ffn_down_exps.weight``  (routed expert down proj)
  - ``ffn_gate_inp.weight``   (router bias / gating input weight, if present)

KV metadata from the source file is preserved **verbatim** (raw bytes copied)
so that downstream code can read architecture, expert counts, layer counts,
alignment, etc. from the extracted file without any re-encoding risk.

Usage::

    # Extract from Q6_K GGUF
    python extract_ornith_cpu_experts_gguf.py \\
        --src ornith-1.0-35b-Q6_K-MTP-final.gguf \\
        --dst ornith-cpu-experts.gguf

    # Extract from hybrid (ik-quant) GGUF
    python extract_ornith_cpu_experts_gguf.py \\
        --src ornith-1.0-35b-IQ4KS-IQ2K-R4-hybrid.gguf \\
        --dst ornith-cpu-experts.gguf
"""

from __future__ import annotations

import argparse
import logging
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Local raw reader (bypasses gguf library enum, returns raw KV bytes)
from kt_kernel.utils.gguf_raw_reader import (
    _read_gguf_full,
    _DEFAULT_ALIGNMENT,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tensor name patterns
# ---------------------------------------------------------------------------
EXPERT_GATE_PATTERN = ".ffn_gate_exps.weight"
EXPERT_UP_PATTERN   = ".ffn_up_exps.weight"
EXPERT_DOWN_PATTERN = ".ffn_down_exps.weight"
ROUTER_PATTERN      = ".ffn_gate_inp.weight"


def _is_expert_tensor(name: str) -> bool:
    """Return True if *name* should be included in the CPU-experts output."""
    return (
        name.endswith(EXPERT_GATE_PATTERN)
        or name.endswith(EXPERT_UP_PATTERN)
        or name.endswith(EXPERT_DOWN_PATTERN)
        or name.endswith(ROUTER_PATTERN)
    )


# ---------------------------------------------------------------------------
# GGUF writer (low-level struct — arbitrary dtype IDs are preserved)
# ---------------------------------------------------------------------------

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3


def _align(offset: int, alignment: int = _DEFAULT_ALIGNMENT) -> int:
    return (offset + alignment - 1) // alignment * alignment


def _build_tensor_index_entry(
    name: str,
    logical_shape: Tuple[int, ...],
    dtype_id: int,
) -> bytes:
    """Encode a single tensor index entry with a **placeholder** offset (0).

    The caller (``_build_gguf``) patches in the correct absolute data offset
    before writing.
    """
    name_bytes = name.encode("utf-8")
    n_dims = len(logical_shape)
    buf = struct.pack("<Q", len(name_bytes)) + name_bytes
    buf += struct.pack("<I", n_dims)
    # GGUF stores shape in reverse order (innermost dim first).
    for d in reversed(logical_shape):
        buf += struct.pack("<Q", d)
    buf += struct.pack("<I", dtype_id)
    buf += struct.pack("<Q", 0)  # placeholder offset
    return buf


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract CPU-expert tensors from a Qwen3.5 MoE GGUF file.",
    )
    parser.add_argument("--src", required=True, help="Source GGUF file path")
    parser.add_argument("--dst", required=True, help="Destination GGUF file path")
    args = parser.parse_args()

    src_path = Path(args.src)
    dst_path = Path(args.dst)

    if not src_path.is_file():
        log.error("Source file not found: %s", src_path)
        sys.exit(1)

    # ---- 1. Read source tensor index + raw KV bytes -----------------------
    log.info("Reading %s ...", src_path)
    src_idx, raw_kv_bytes, alignment = _read_gguf_full(str(src_path))

    selected_names = sorted(n for n in src_idx if _is_expert_tensor(n))
    if not selected_names:
        log.error(
            "No expert tensors found.  Expected names ending with %s, %s, "
            "%s, or %s",
            EXPERT_GATE_PATTERN,
            EXPERT_UP_PATTERN,
            EXPERT_DOWN_PATTERN,
            ROUTER_PATTERN,
        )
        sys.exit(1)

    log.info("Selected %d tensors:", len(selected_names))
    for n in selected_names:
        ti = src_idx[n]
        log.info(
            "  %s  shape=%s  dtype_id=%d  n_bytes=%d  offset=%d",
            n, ti.shape, ti.dtype_id, ti.n_bytes, ti.data_offset,
        )

    # ---- 2. Compute kv_count from raw KV bytes (needed for correct header)
    # Each KV entry: key_len(Q) + key + val_type(I) + val.
    # Count by scanning.  This is redundant with the raw reader but ensures
    # the header matches exactly.
    kv_count = 0
    offs = 0
    while offs < len(raw_kv_bytes):
        key_len = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]
        offs += 8 + key_len
        val_type = struct.unpack_from("<I", raw_kv_bytes, offs)[0]
        offs += 4
        if val_type == 0:          offs += 1         # UINT8
        elif val_type == 1:        offs += 1         # INT8
        elif val_type == 2:        offs += 2         # UINT16
        elif val_type == 3:        offs += 2         # INT16
        elif val_type == 4:        offs += 4         # UINT32
        elif val_type == 5:        offs += 4         # INT32
        elif val_type == 6:        offs += 4         # FLOAT32
        elif val_type == 7:        offs += 1         # BOOL
        elif val_type == 8:        slen = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]; offs += 8 + slen  # STRING
        elif val_type == 9:        # ARRAY
            arr_type = struct.unpack_from("<I", raw_kv_bytes, offs)[0]; offs += 4
            arr_len = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]; offs += 8
            for _ in range(arr_len):
                if arr_type == 8:  slen = struct.unpack_from("<Q", raw_kv_bytes, offs)[0]; offs += 8 + slen
                elif arr_type in (0, 1, 7):  offs += 1
                elif arr_type in (2, 3):      offs += 2
                elif arr_type in (4, 5, 6):   offs += 4
                elif arr_type in (10, 11, 12): offs += 8
                else: raise ValueError(f"Unsupported array element type {arr_type}")
        elif val_type == 10:       offs += 8         # UINT64
        elif val_type == 11:       offs += 8         # INT64
        elif val_type == 12:       offs += 8         # FLOAT64
        else: raise ValueError(f"Unknown KV value type {val_type}")
        kv_count += 1

    # ---- 3. Copy payload slices via mmap (avoid loading full 28GB file) ----
    import numpy as np

    src_mmap = np.memmap(src_path, mode="r")

    tensor_meta: List[Tuple[str, object]] = [
        (name, src_idx[name]) for name in selected_names
    ]
    total_payload_bytes = sum(ti.n_bytes for _, ti in tensor_meta)

    # ---- 4. Build header with correct kv_count ----------------------------
    header = struct.pack("<I", GGUF_MAGIC)
    header += struct.pack("<I", GGUF_VERSION)
    header += struct.pack("<Q", len(tensor_meta))
    header += struct.pack("<Q", kv_count)

    ti_templates = [
        _build_tensor_index_entry(name, ti.shape, ti.dtype_id)
        for name, ti in tensor_meta
    ]
    ti_section = b"".join(ti_templates)
    pre_data_size = len(header) + len(raw_kv_bytes) + len(ti_section)
    data_start = _align(pre_data_size, alignment)

    # ---- 5. Stream-write output GGUF (no full-RAM payload buffer) -------
    # TI stores offsets relative to aligned data section (GGUF spec / GGUFReader).
    rel_offs = 0
    ti_entries_fixed: List[bytes] = []
    for (_, ti), ti_entry_template in zip(tensor_meta, ti_templates):
        ti_entry = ti_entry_template[:-8] + struct.pack("<Q", rel_offs)
        ti_entries_fixed.append(ti_entry)
        rel_offs = _align(rel_offs + ti.n_bytes, alignment)

    with dst_path.open("wb") as f:
        f.write(header)
        f.write(raw_kv_bytes)
        for entry in ti_entries_fixed:
            f.write(entry)
        pad = data_start - f.tell()
        if pad > 0:
            f.write(b"\x00" * pad)
        for name, ti in tensor_meta:
            chunk = src_mmap[ti.data_offset : ti.data_offset + ti.n_bytes]
            f.write(chunk.tobytes() if hasattr(chunk, "tobytes") else bytes(chunk))
            pad = _align(ti.n_bytes, alignment) - ti.n_bytes
            if pad > 0:
                f.write(b"\x00" * pad)

    total_mb = total_payload_bytes / (1024 * 1024)
    log.info(
        "Done — %d tensors (%.1f MB) written to %s (alignment=%d)",
        len(selected_names),
        total_mb,
        dst_path,
        alignment,
    )


if __name__ == "__main__":
    main()
