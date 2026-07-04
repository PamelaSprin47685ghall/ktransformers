"""Low-level GGUF tensor-index reader that works with *any* dtype ID.

The official ``gguf.GGUFReader`` raises ``ValueError`` when the file contains
quantisation types unknown to the PyPI ``GGMLQuantizationType`` enum (e.g. ik
extended types like IQ2_K_R4 = 337).  This module bypasses the enum entirely
by reading raw uint32 dtype IDs from the binary tensor index.

Usage::

    from kt_kernel.utils.gguf_raw_reader import read_gguf_tensor_index

    idx = read_gguf_tensor_index("/path/to/model.gguf")
    t = idx["blk.0.ffn_gate_exps.weight"]
    assert t.dtype_id == 337   # raw integer, no enum
    assert t.n_bytes == ...
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

try:
    from .gguf_ik_types import IK_GGML_QUANT_SIZES
except ImportError:
    from gguf_ik_types import IK_GGML_QUANT_SIZES

# ---------------------------------------------------------------------------
# Standard GGML quant sizes (subset needed for n_bytes computation)
# ---------------------------------------------------------------------------
# Keyed by integer dtype ID.  Each value is (block_size, type_size).
_STANDARD_QUANT_SIZES: dict[int, tuple[int, int]] = {
    0:   (1, 4),         # F32
    1:   (1, 2),         # F16
    2:   (32, 18),       # Q4_0    (2 + 16)
    3:   (32, 20),       # Q4_1    (2 + 2 + 16)
    6:   (32, 22),       # Q5_0    (2 + 4 + 16)
    7:   (32, 24),       # Q5_1    (2 + 2 + 4 + 16)
    8:   (32, 34),       # Q8_0    (2 + 32)
    9:   (32, 40),       # Q8_1    (4 + 4 + 32)
    10:  (256, 84),      # Q2_K    (2 + 2 + 16 + 64)
    11:  (256, 110),     # Q3_K    (2 + 64 + 32 + 12)
    12:  (256, 144),     # Q4_K    (2 + 2 + 128 + 12)
    13:  (256, 176),     # Q5_K    (2 + 2 + 128 + 32 + 12)
    14:  (256, 210),     # Q6_K    (2 + 128 + 64 + 16)
    15:  (256, 292),     # Q8_K    (4 + 256 + 32)
    16:  (256, 66),      # IQ2_XXS (2 + 64)
    17:  (256, 74),      # IQ2_XS  (2 + 64 + 8)
    18:  (256, 98),      # IQ3_XXS (2 + 64 + 32)
    19:  (256, 34),      # IQ1_S   (2 + 32)
    20:  (32, 18),       # IQ4_NL  (2 + 16)
    21:  (256, 110),     # IQ3_S   (2 + 64 + 32 + 8 + 4)
    22:  (256, 82),      # IQ2_S   (2 + 64 + 16)
    23:  (256, 136),     # IQ4_XS  (2 + 2 + 128 + 4)
    24:  (1, 1),         # I8
    25:  (1, 2),         # I16
    26:  (1, 4),         # I32
    27:  (1, 8),         # I64
    28:  (1, 8),         # F64
    29:  (256, 56),      # IQ1_M   (32 + 16 + 8)
    30:  (1, 2),         # BF16
}

# Merge ik-extended sizes on top.
_QUANT_SIZES: dict[int, tuple[int, int]] = {**_STANDARD_QUANT_SIZES, **IK_GGML_QUANT_SIZES}

GGUF_MAGIC = 0x46554747
GGUF_VERSION = 3
_DEFAULT_ALIGNMENT = 32


@dataclass
class TensorIndex:
    """Raw tensor index entry for one GGUF tensor."""

    name: str
    dtype_id: int       # raw uint32 from the file (may be > 41 for ik types)
    shape: tuple        # logical shape (PyTorch order, innermost dim last)
    n_bytes: int        # byte count of the packed quantised payload
    data_offset: int    # byte offset of payload data from start of file
    n_elements: int     # product of logical shape


def _align(offset: int, alignment: int = _DEFAULT_ALIGNMENT) -> int:
    """Return *offset* rounded up to the next *alignment* boundary."""
    return (offset + alignment - 1) // alignment * alignment


def read_gguf_tensor_index(path: str | Path) -> Dict[str, TensorIndex]:
    """Parse the GGUF file at *path* and return every tensor's index entry.

    Only the header + KV metadata + tensor index are read (no payload data).
    This is safe for files with ik-extended quantisation types.

    Returns:
        ``{tensor_name: TensorIndex}``
    """
    return _read_gguf_full(path)[0]


def _read_gguf_full(
    path: str | Path,
) -> tuple[Dict[str, TensorIndex], bytes, int]:
    """Like ``read_gguf_tensor_index`` but also returns raw KV bytes + alignment.

    Returns:
        ``(tensor_index, raw_kv_bytes, data_alignment)`` where:
        - ``tensor_index`` maps each tensor name to its ``TensorIndex``.
        - ``raw_kv_bytes`` is the verbatim byte slice from the file containing
          all KV metadata entries (suitable for direct re-use when building a
          new GGUF).
        - ``data_alignment`` is the file's declared alignment (default 32).
    """
    path = Path(path)
    raw = np.memmap(path, mode="r")
    offs = 0

    # ---- header -----------------------------------------------------------
    magic = struct.unpack_from("<I", raw, offs)[0]
    offs += 4
    if magic != GGUF_MAGIC:
        raise ValueError(f"Not a GGUF file (magic 0x{magic:08X})")
    version = struct.unpack_from("<I", raw, offs)[0]
    offs += 4
    if version != GGUF_VERSION:
        raise ValueError(f"Unsupported GGUF version {version} (expected {GGUF_VERSION})")
    tensor_count = struct.unpack_from("<Q", raw, offs)[0]
    offs += 8
    kv_count = struct.unpack_from("<Q", raw, offs)[0]
    offs += 8

    kv_start = offs
    data_alignment = _DEFAULT_ALIGNMENT

    # ---- KV metadata (skip, capture raw bytes, extract alignment) ---------
    for _ in range(kv_count):
        key_len = struct.unpack_from("<Q", raw, offs)[0]
        offs += 8
        key = bytes(raw[offs : offs + key_len]).decode("utf-8")
        offs += key_len
        val_type = struct.unpack_from("<I", raw, offs)[0]
        offs += 4
        # Track alignment metadata
        if key == "general.alignment":
            data_alignment = struct.unpack_from("<I", raw, offs)[0]
        # Skip value according to type
        if val_type == 0:          # UINT8
            offs += 1
        elif val_type == 1:        # INT8
            offs += 1
        elif val_type == 2:        # UINT16
            offs += 2
        elif val_type == 3:        # INT16
            offs += 2
        elif val_type == 4:        # UINT32
            offs += 4
        elif val_type == 5:        # INT32
            offs += 4
        elif val_type == 6:        # FLOAT32
            offs += 4
        elif val_type == 7:        # BOOL
            offs += 1
        elif val_type == 8:        # STRING
            slen = struct.unpack_from("<Q", raw, offs)[0]
            offs += 8 + slen
        elif val_type == 9:        # ARRAY
            arr_type = struct.unpack_from("<I", raw, offs)[0]
            offs += 4
            arr_len = struct.unpack_from("<Q", raw, offs)[0]
            offs += 8
            for _ in range(arr_len):
                if arr_type == 8:  # string array
                    slen = struct.unpack_from("<Q", raw, offs)[0]
                    offs += 8 + slen
                elif arr_type in (0, 1, 7):  # uint8, int8, bool
                    offs += 1
                elif arr_type in (2, 3):      # uint16, int16
                    offs += 2
                elif arr_type in (4, 5, 6):   # uint32, int32, float32
                    offs += 4
                elif arr_type in (10, 11, 12): # uint64, int64, float64
                    offs += 8
                else:
                    raise ValueError(f"Unsupported array element type {arr_type}")
        elif val_type == 10:       # UINT64
            offs += 8
        elif val_type == 11:       # INT64
            offs += 8
        elif val_type == 12:       # FLOAT64
            offs += 8
        else:
            raise ValueError(f"Unknown KV value type {val_type}")
    kv_end = offs
    raw_kv_bytes = bytes(raw[kv_start:kv_end])

    ti_start = offs
    skip_offs = offs
    for _ in range(tensor_count):
        name_len = struct.unpack_from("<Q", raw, skip_offs)[0]
        skip_offs += 8 + name_len
        n_dims = struct.unpack_from("<I", raw, skip_offs)[0]
        skip_offs += 4 + 8 * n_dims + 4 + 8
    ti_end = skip_offs
    data_start = _align(ti_end, data_alignment)

    # ---- tensor index -----------------------------------------------------
    result: Dict[str, TensorIndex] = {}
    offs = ti_start

    for _ in range(tensor_count):
        name_len = struct.unpack_from("<Q", raw, offs)[0]
        offs += 8
        name = bytes(raw[offs : offs + name_len]).decode("utf-8")
        offs += name_len
        n_dims = struct.unpack_from("<I", raw, offs)[0]
        offs += 4
        dims = struct.unpack_from(f"<{n_dims}Q", raw, offs)
        offs += 8 * n_dims
        dtype_id = struct.unpack_from("<I", raw, offs)[0]
        offs += 4
        rel_offs = struct.unpack_from("<Q", raw, offs)[0]
        offs += 8
        tensor_data_offs = data_start + int(rel_offs)

        # GGUF stores shape in reverse order (innermost dim first).
        logical_shape = tuple(reversed(dims))

        n_elems = int(np.prod(logical_shape))
        if dtype_id in _QUANT_SIZES:
            block_size, type_size = _QUANT_SIZES[dtype_id]
            n_bytes = n_elems * type_size // block_size
        else:
            # Unknown dtype — assume per-element (F32 fallback).  Caller will
            # need to handle this; warn so the user knows.
            import warnings
            warnings.warn(
                f"Unknown quantisation dtype_id={dtype_id} for tensor {name!r}; "
                f"assuming F32 byte size ({n_elems * 4} bytes)."
            )
            n_bytes = n_elems * 4

        result[name] = TensorIndex(
            name=name,
            dtype_id=dtype_id,
            shape=logical_shape,
            n_bytes=n_bytes,
            data_offset=tensor_data_offs,
            n_elements=n_elems,
        )

    return result, raw_kv_bytes, data_alignment
