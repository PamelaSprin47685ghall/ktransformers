"""quantization_config.ignore：Q6 parity 不豁免 attn/shared proj。"""

from __future__ import annotations

import json
import os
from pathlib import Path

ORNITH = Path("/home/kunweiz/Desktop/Ornith")
CFG = Path(
    os.environ.get(
        "ORNITH_W4_OUTPUT",
        str(ORNITH / "ornith-gpu-w4-q6-parity-from-gguf"),
    )
) / "config.json"
TOOL = Path("tools/run_awq_quantization.py")


def test_w4_tool_ignore_q6_parity():
    source = TOOL.read_text()
    assert "self_attn(\\.|$)" not in source
    assert "linear_attn(\\.|$)" not in source
    assert "_ORNITH_W4_IGNORE_TOP_LEVEL" not in source


def test_w4_output_ignore_does_not_block_shared_expert_proj():
    if not CFG.is_file():
        return
    cfg = json.loads(CFG.read_text())
    ignore = cfg.get("quantization_config", {}).get("ignore") or []
    joined = "\n".join(ignore)
    assert "shared_expert(\\.|$)" not in joined
    assert "self_attn(\\.|$)" not in joined
    assert any("mlp" in x and "gate" in x for x in ignore)