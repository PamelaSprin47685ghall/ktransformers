from pathlib import Path


def test_compressed_ik_script_caps_kv_for_8gb_mlp_only():
    text = Path("scripts/ornith/run_sglang_compressed_ik.sh").read_text()
    assert "ornith-gpu-w4-q6-parity-from-gguf" in text
    assert "compressed-tensors" in text
    assert "max-mamba-cache-size" in text
    assert "max-total-tokens" in text
    assert "ORNITH_MEM_FRACTION_STATIC" in text
    assert "TOKENIZER_DIR:-${MODEL_DIR}" in text
    assert "compressed-tensors" in text
    assert "language-only" not in text
    assert "disable-radix-cache" in text


def test_w4_quant_patches_causal_lm_arch():
    source = Path("tools/run_awq_quantization.py").read_text()
    assert "_patch_serving_config_causal_lm" in source
    assert "Qwen3_5MoeForConditionalGeneration" in source
    assert "text_config" in source


def test_compressed_ik_script_skips_vlm_warmup_for_short_context():
    # 8GB MLP-only serve uses context-length 64; default server warmup sends
    # VLM chat with image+text (~80 tokens) which OOMs after weight load.
    # The launch script must disable warmup outright.
    text = Path("scripts/ornith/run_sglang_compressed_ik.sh").read_text()
    assert "skip-server-warmup" in text