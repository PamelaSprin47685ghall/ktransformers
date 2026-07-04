"""w4 产物须含 compressed-tensors 打包键（Q6 parity：attn+shared+embed 等）。"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

ORNITH = Path("/home/kunweiz/Desktop/Ornith")
W4 = Path(
    os.environ.get(
        "ORNITH_W4_OUTPUT",
        str(ORNITH / "ornith-gpu-w4-q6-parity-from-gguf"),
    )
) / "model.safetensors"
LEGACY = ORNITH / "ornith-gpu-w4-mlp-only-from-gguf/model.safetensors"


def _header_keys(path: Path) -> list[str]:
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]


def test_w4_checkpoint_has_weight_packed_when_present():
    path = W4 if W4.is_file() else LEGACY
    if not path.is_file():
        return
    keys = _header_keys(path)
    packed = [k for k in keys if "weight_packed" in k]
    if not packed:
        raise AssertionError(
            f"{path}: no weight_packed ({len(keys)} keys); re-run run_awq_quantization.py"
        )
    shared = [
        k
        for k in packed
        if "shared_expert" in k
        and any(x in k for x in ("gate_proj", "up_proj", "down_proj"))
    ]
    linear = [k for k in packed if "linear_attn" in k and "in_proj_qkv" in k]
    lm_bf16 = "lm_head.weight" in keys and "lm_head.weight_packed" not in keys
    embed_bf16 = "model.embed_tokens.weight" in keys
    assert len(shared) >= 90, f"shared_expert w4 missing: {len(shared)}"
    if path == W4:
        assert linear, "Q6 parity: linear_attn must be w4"
        assert lm_bf16, "lm_head BF16 serving contract"
        assert embed_bf16 or any("embed" in k for k in packed), "embed missing"