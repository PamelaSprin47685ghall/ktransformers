"""Raw GGUF reader must load ik quant type ids without PyPI gguf enum."""

import struct
import tempfile
from pathlib import Path

import numpy as np

from kt_kernel.utils.gguf_raw_reader import read_gguf_tensor_index


def _write_minimal_gguf(path: Path, tensor_name: str, raw_dtype: int, payload: bytes, shape):
    alignment = 32
    n_dims = len(shape)
    ti = b""
    name_b = tensor_name.encode()
    ti += struct.pack("<Q", len(name_b)) + name_b
    ti += struct.pack("<I", n_dims)
    for d in reversed(shape):
        ti += struct.pack("<Q", d)
    ti += struct.pack("<I", raw_dtype)
    ti += struct.pack("<Q", 0)
    kv = b""
    arch_key = b"general.architecture"
    arch_val = b"qwen35moe"
    kv += struct.pack("<Q", len(arch_key)) + arch_key
    kv += struct.pack("<I", 8)
    kv += struct.pack("<Q", len(arch_val)) + arch_val
    header = struct.pack("<I", 0x46554747)
    header += struct.pack("<I", 3)
    header += struct.pack("<Q", 1)
    header += struct.pack("<Q", 1)
    off = alignment * ((len(header) + len(kv) + len(ti) + alignment - 1) // alignment)
    pad_kv = b"\x00" * (off - len(header) - len(kv) - len(ti))
    pad_payload = b"\x00" * (alignment - (len(payload) % alignment or alignment))
    with path.open("wb") as f:
        f.write(header)
        f.write(kv)
        f.write(ti)
        f.write(pad_kv)
        f.write(payload)
        f.write(pad_payload)


def test_read_iq2_k_r4_tensor_metadata():
    payload = np.zeros(304, dtype=np.uint8).tobytes()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.gguf"
        _write_minimal_gguf(p, "blk.0.ffn_gate_exps.weight", 337, payload, [1, 1024])
        idx = read_gguf_tensor_index(str(p))
        t = idx["blk.0.ffn_gate_exps.weight"]
        assert t.dtype_id == 337
        assert t.n_bytes == 304