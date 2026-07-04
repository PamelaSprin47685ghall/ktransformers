from pathlib import Path
import json
import struct


def test_low_mem_marlin_quantization_uses_datafree_path():
    source = Path("tools/run_awq_quantization.py").read_text()

    assert "AWQModifier" not in source
    assert "CALIBRATION_DATASET" not in source
    assert 'pipeline="datafree"' in source
    assert "num_calibration_samples=0" in source
    assert "moe_calibrate_all_experts=False" in source
    assert "_set_num_experts" in source
    # 必须手动加载+转置权重
    assert "_load_model_with_corrected_weights" in source
    assert "_align_to_model_weight" in source
    assert "_validate_quantized_checkpoint_finite" in source
    assert "_ornith_w4_ignore_entries" in source
    assert "attn" in source





def test_existing_marlin_output_keeps_full_attention_bf16_when_present():
    path = Path("/home/kunweiz/Desktop/Ornith/ornith-gpu-w4-mlp-only-from-gguf/model.safetensors")
    if not path.is_file():
        return

    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))

    keys = set(header) - {"__metadata__"}
    assert "model.layers.3.self_attn.q_proj.weight" in keys
    assert "model.layers.3.self_attn.q_proj.weight_packed" not in keys


def test_existing_marlin_output_config_ignores_attention_recursively_when_present():
    path = Path("/home/kunweiz/Desktop/Ornith/ornith-gpu-w4-mlp-only-from-gguf/config.json")
    if not path.is_file():
        return

    cfg = json.loads(path.read_text())
    ignore = cfg["quantization_config"]["ignore"]
    assert r"re:^model\.(language_model\.)?layers\.\d+\.self_attn(\.|$)" in ignore
    assert "model.layers.3.self_attn" not in ignore
