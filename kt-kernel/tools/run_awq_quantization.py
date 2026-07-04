#!/usr/bin/env python3
"""Low-memory w4 compressed quantization for the Ornith GPU slice.

Quant policy: same coverage as ornith-gpu-non-expert.gguf (Q6 slice) — w4 all Linear
except MoE router gates; routed experts not in slice. See w4-quant-policy.md + q6-vs-marlin-compression.md.

三重障碍及对策：
1. config.json 声明 num_experts=256 → from_pretrained 建 60GB 专家骨架 → OOM
   → 临时 patch num_experts=0
2. safetensors 权重存 GGUF meta 布局 (in,out)，模型期望 (out,in) → MISMATCH
   → 手动加载+逐张量转置
3. oneshot 默认 pipeline=independent 会跑校准 → OOM
   → 显式 pipeline=datafree
"""
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
from _compressed_tensors_distributed_shim import install as _install_ct_dist_shim

_install_ct_dist_shim()

import torch
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.recipe import Recipe
from llmcompressor.entrypoints import oneshot

import os

INPUT_MODEL = os.environ.get(
    "ORNITH_W4_INPUT", "/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-standalone"
)
OUTPUT_DIR = os.environ.get(
    "ORNITH_W4_OUTPUT",
    "/home/kunweiz/Desktop/Ornith/ornith-gpu-w4-q6-parity-from-gguf",
)
CONFIG_PATH = Path(INPUT_MODEL) / "config.json"
SAFETENSORS_PATH = Path(INPUT_MODEL) / "model.safetensors"

# Q6 切片 parity：仅 router / 标量门 / MTP 保持 BF16；勿 ignore 整段 shared_expert / attn
_ORNITH_W4_IGNORE_TOP = ("lm_head",)

_ORNITH_W4_IGNORE_PATTERNS = (
    r"re:^model\.(language_model\.)?layers\.\d+\.mlp\.gate$",
    r"re:^model\.(language_model\.)?layers\.\d+\.mlp\.shared_expert_gate$",
    r"re:^model\.(language_model\.)?layers\.\d+\.mlp\.shared_expert\.gate\.weight$",
    r"re:^model\.(language_model\.)?mtp\.",
    r"re:^mtp\.",
)


def _ornith_w4_ignore_entries(num_layers: int) -> list[str]:
    top = [f"model.{s}" for s in _ORNITH_W4_IGNORE_TOP]
    return top + list(_ORNITH_W4_IGNORE_PATTERNS)


def _set_num_experts(cfg: dict, n: int):
    if "text_config" in cfg:
        cfg["text_config"]["num_experts"] = n
    else:
        cfg["num_experts"] = n


def _load_model_with_corrected_weights():
    """从 config 建骨架（num_experts=0），手动加载 safetensors 并转置不匹配的权重。"""
    cfg = AutoConfig.from_pretrained(INPUT_MODEL)
    model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)

    ckpt = load_file(str(SAFETENSORS_PATH))
    model_sd = model.state_dict()

    # 构建 ckpt key → tensor 映射，归一化前缀差异
    # safetensors: model.language_model.layers... / lm_head.weight
    # model params: model.layers... / lm_head.weight
    ckpt_normalized = {}
    for k, v in ckpt.items():
        nk = k.replace("model.language_model.", "model.")
        ckpt_normalized[nk] = v
        ckpt_normalized[k] = v

    loaded, skipped, transposed = 0, 0, 0
    for name, param in model.named_parameters():
        ckpt_w = ckpt_normalized.get(name)
        if ckpt_w is None:
            # 尝试去掉 model. 前缀
            alt = name.removeprefix("model.")
            ckpt_w = ckpt_normalized.get(alt)
        if ckpt_w is None:
            skipped += 1
            continue

        if ckpt_w.shape == param.shape:
            param.data.copy_(ckpt_w)
            loaded += 1
        elif ckpt_w.dim() == 2 and tuple(reversed(ckpt_w.shape)) == tuple(param.shape):
            param.data.copy_(ckpt_w.t().contiguous())
            transposed += 1
        elif ckpt_w.dim() == 3 and tuple(reversed(ckpt_w.shape)) == tuple(param.shape):
            param.data.copy_(ckpt_w.permute(2, 1, 0).contiguous())
            transposed += 1
        elif param.numel() == 0:
            # num_experts=0 导致 gate 维度为 0，跳过
            skipped += 1
        else:
            print(f"  SKIP {name}: ckpt {ckpt_w.shape} vs model {param.shape}", file=sys.stderr)
            skipped += 1

    print(f"Loaded={loaded} transposed={transposed} skipped={skipped}", file=sys.stderr)
    return model


def main():
    original = json.loads(CONFIG_PATH.read_text())
    tc = original.get("text_config", original)
    original_experts = tc["num_experts"]

    print(f"Patching num_experts {original_experts}→0", file=sys.stderr)
    _set_num_experts(original, 0)
    CONFIG_PATH.write_text(json.dumps(original, indent=2, ensure_ascii=False))

    try:
        model = _load_model_with_corrected_weights()

        num_bits = int(os.environ.get("ORNITH_W4_BITS", "4"))
        quant_modifier = QuantizationModifier(
            config_groups={
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": num_bits,
                        "group_size": 128,
                        "symmetric": True,
                        "strategy": "group",
                    },
                }
            },
            ignore=_ornith_w4_ignore_entries(40),
        )
        recipe = Recipe.from_modifiers([quant_modifier])

        print(f"w{num_bits}a16 quantization (datafree, experts=0, weights corrected)", file=sys.stderr)
        print(f"Output: {OUTPUT_DIR}", file=sys.stderr)

        model.cpu()
        # llmcompressor calibration 会把 offload 权重 onload 到 CUDA；8GB 上须 CPU-only 量化
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        oneshot(
            model=model,
            dataset=None,
            recipe=recipe,
            output_dir=Path(OUTPUT_DIR),
            pipeline="datafree",
            num_calibration_samples=0,
            moe_calibrate_all_experts=False,
            batch_size=1,
            max_seq_length=1,
            clear_sparse_session=True,
            save_compressed=True,
        )
    finally:
        print(f"Restoring input num_experts→{original_experts}", file=sys.stderr)
        cfg_in = json.loads(CONFIG_PATH.read_text())
        _set_num_experts(cfg_in, original_experts)
        CONFIG_PATH.write_text(json.dumps(cfg_in, indent=2, ensure_ascii=False))

    out_cfg = Path(OUTPUT_DIR) / "config.json"
    if out_cfg.exists():
        cfg_out = json.loads(out_cfg.read_text())
        qcfg = cfg_out.get("quantization_config", {})
        if qcfg:
            qcfg["ignore"] = _ornith_w4_ignore_entries(40)
        cfg_out = dict(cfg_in)
        cfg_out["quantization_config"] = qcfg
        out_cfg.write_text(json.dumps(cfg_out, indent=2, ensure_ascii=False))

    _patch_router_weights()
    _patch_shared_expert_gate_bf16()
    _patch_lm_head_bf16()
    _patch_serving_config_causal_lm()
    _patch_quantization_ignore_config()
    _sync_serving_assets()
    _validate_quantized_checkpoint_finite()

    print(f"✓ Quantization complete. Output: {OUTPUT_DIR}", file=sys.stderr)


def _sync_serving_assets() -> None:
    import shutil

    out = Path(OUTPUT_DIR)
    inp = Path(INPUT_MODEL)
    names = (
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "chat_template.jinja",
        "preprocessor_config.json",
    )
    for name in names:
        src = inp / name
        if src.is_file():
            shutil.copy2(src, out / name)
    hf_tpl = Path(
        os.environ.get(
            "ORNITH_HF_TEMPLATE",
            "/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-standalone",
        )
    )
    if not (out / "preprocessor_config.json").is_file():
        alt = hf_tpl / "preprocessor_config.json"
        if alt.is_file():
            shutil.copy2(alt, out / "preprocessor_config.json")


def _patch_router_weights():
    """num_experts=0 量化后 mlp.gate.weight 变成 (0,hidden) 空张量。
    从输入 BF16 safetensors 拷贝 gate 权重到输出，按目标形状对齐不盲目转置。
    """
    from safetensors.torch import load_file, save_file

    in_ckpt = load_file(str(SAFETENSORS_PATH))
    out_path = Path(OUTPUT_DIR) / "model.safetensors"
    out_ckpt = load_file(str(out_path))

    patched = 0
    for k, v in in_ckpt.items():
        out_k = k.replace("model.language_model.", "model.")
        if not ("mlp.gate.weight" in out_k and out_k in out_ckpt):
            continue
        target = out_ckpt[out_k]
        out_ckpt[out_k] = _align_to_model_weight(v, target.shape)
        patched += 1

    save_file(out_ckpt, str(out_path))
    print(f"Patched {patched} router gate weights from input", file=sys.stderr)


def _align_to_model_weight(src: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    if src.shape == target_shape:
        return src.contiguous().clone()
    if src.dim() == 2 and tuple(reversed(src.shape)) == tuple(target_shape):
        return src.t().contiguous().clone()
    if src.dim() == 3 and tuple(reversed(src.shape)) == tuple(target_shape):
        return src.permute(2, 1, 0).contiguous().clone()
    return src.contiguous().clone()


def _delete_quantized_variants(out_ckpt: dict, weight_name: str) -> None:
    stem = weight_name[: -len(".weight")]
    for suffix in (".weight_packed", ".weight_scale", ".weight_shape"):
        out_ckpt.pop(stem + suffix, None)


def _replace_bf16_weights(predicate, label: str) -> None:
    from safetensors.torch import load_file, save_file

    in_ckpt = load_file(str(SAFETENSORS_PATH))
    out_path = Path(OUTPUT_DIR) / "model.safetensors"
    out_ckpt = load_file(str(out_path))
    patched = 0
    for key, value in in_ckpt.items():
        out_key = key.replace("model.language_model.", "model.")
        if not predicate(out_key) or not out_key.endswith(".weight"):
            continue
        if out_key not in out_ckpt:
            continue
        _delete_quantized_variants(out_ckpt, out_key)
        target_shape = out_ckpt[out_key].shape
        out_ckpt[out_key] = _align_to_model_weight(value, target_shape)
        patched += 1
    save_file(out_ckpt, str(out_path))
    print(f"Patched {patched} {label} BF16 weights from input", file=sys.stderr)


def _patch_shared_expert_gate_bf16() -> None:
    """仅 scalar shared_expert.gate；gate/up/down_proj 留给 w4 Marlin。"""
    _replace_bf16_weights(
        lambda name: name.endswith(".mlp.shared_expert.gate.weight"),
        "shared_expert_gate",
    )


def _patch_lm_head_bf16() -> None:
    """datafree w4 lm_head → 乱码；Q6 文件里虽量化，Marlin 轨出口保持 BF16。"""
    from safetensors.torch import load_file, save_file

    in_ckpt = load_file(str(SAFETENSORS_PATH))
    out_path = Path(OUTPUT_DIR) / "model.safetensors"
    out_ckpt = load_file(str(out_path))
    src = in_ckpt.get("lm_head.weight")
    if src is None:
        return
    for k in list(out_ckpt.keys()):
        if k.startswith("lm_head."):
            del out_ckpt[k]
    out_ckpt["lm_head.weight"] = src.contiguous().clone()
    save_file(out_ckpt, str(out_path))
    print("Patched lm_head BF16 from input", file=sys.stderr)


def _patch_serving_config_causal_lm() -> None:
    """Keep VL ``ConditionalGeneration`` + ``quantization_config`` for SGLang config coercion."""
    out_cfg = Path(OUTPUT_DIR) / "config.json"
    cfg = json.loads(out_cfg.read_text())
    qcfg = cfg.get("quantization_config")
    if "text_config" not in cfg:
        standalone = json.loads((Path(INPUT_MODEL) / "config.json").read_text())
        cfg = dict(standalone)
        if qcfg:
            cfg["quantization_config"] = qcfg
    else:
        if qcfg:
            cfg["quantization_config"] = qcfg
    cfg["architectures"] = ["Qwen3_5MoeForConditionalGeneration"]
    out_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def _patch_quantization_ignore_config() -> None:
    out_cfg = Path(OUTPUT_DIR) / "config.json"
    cfg = json.loads(out_cfg.read_text())
    qcfg = cfg.get("quantization_config")
    if qcfg is None:
        return
    qcfg["ignore"] = _ornith_w4_ignore_entries(40)
    out_cfg.write_text(json.dumps(cfg, indent=2, ensure_ascii=False))


def _validate_quantized_checkpoint_finite() -> None:
    """Reject checkpoints whose float scales/bias contain NaN/Inf (common datafree pitfall)."""
    from safetensors import safe_open

    out_path = Path(OUTPUT_DIR) / "model.safetensors"
    if not out_path.is_file():
        return
    bad: list[str] = []
    with safe_open(str(out_path), framework="pt") as f:
        for key in f.keys():
            if not any(
                s in key
                for s in (".weight_scale", ".weight_zero_point", ".scales", ".scale")
            ):
                continue
            t = f.get_tensor(key)
            if t.is_floating_point() and not t.isfinite().all():
                bad.append(key)
    if bad:
        raise RuntimeError(
            "non-finite quant scales in output (datafree symmetric w4): "
            + ", ".join(bad[:8])
            + (" …" if len(bad) > 8 else "")
        )
    print("Quant scale finite check OK", file=sys.stderr)


if __name__ == "__main__":
    main()
