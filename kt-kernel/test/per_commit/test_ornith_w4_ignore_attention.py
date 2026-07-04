from pathlib import Path


def test_w4_ignore_q6_parity_quantizes_attn_not_router():
    source = Path("tools/run_awq_quantization.py").read_text()
    assert "_ornith_w4_ignore_entries" in source
    assert "mlp.gate" in source
    assert "self_attn(\\.|$)" not in source
    assert "linear_attn(\\.|$)" not in source
    assert "shared_expert(\\.|$)" not in source
    assert "_patch_shared_expert_gate_bf16" in source
    assert "_patch_full_attention_bf16" not in source
    assert "_patch_lm_head_bf16" in source
    assert '"lm_head"' in source


def test_w4_q6_parity_output_dir_default():
    source = Path("tools/run_awq_quantization.py").read_text()
    assert "ornith-gpu-w4-q6-parity-from-gguf" in source
    assert "_sync_serving_assets" in source