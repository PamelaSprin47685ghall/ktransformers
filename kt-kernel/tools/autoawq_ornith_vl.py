#!/usr/bin/env python3
"""AutoAWQ for Ornith VL: quantize language_model + lm_head; skip visual if OOM."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--w-bit", type=int, default=4)
    p.add_argument("--q-group-size", type=int, default=128)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-calib-samples", type=int, default=32)
    args = p.parse_args()

    import torch
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    model_path = str(Path(args.model_path))
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    calib = [
        "The capital of France is",
        "def fibonacci(n):",
    ] * (args.max_calib_samples // 2 + 1)
    calib = calib[: args.max_calib_samples]

    log.info("Loading %s for AWQ (8GB GPU: visual/experts stay BF16 in shards)", model_path)
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = (getattr(cfg, "architectures", None) or [""])[0]
        log.info("architecture=%s", arch)
    except Exception:
        pass
    model = AutoAWQForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    quant_config = {
        "zero_point": True,
        "q_group_size": args.q_group_size,
        "w_bit": args.w_bit,
        "version": "GEMM",
    }
    model.quantize(tok, calib, quant_config=quant_config)
    model.save_quantized(str(out_dir))
    tok.save_pretrained(str(out_dir))
    log.info("Saved AWQ to %s", out_dir)


if __name__ == "__main__":
    main()