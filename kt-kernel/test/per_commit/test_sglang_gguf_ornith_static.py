import json
from types import SimpleNamespace


def test_auto_load_format_uses_gguf_loader_for_gguf_file(tmp_path):
    from sglang.srt.configs.load_config import LoadConfig, LoadFormat
    from sglang.srt.model_loader.loader import GGUFModelLoader, get_model_loader

    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    load_config = LoadConfig(load_format="auto")

    loader = get_model_loader(
        load_config,
        model_config=SimpleNamespace(model_path=str(gguf_path), quantization=None),
    )

    assert isinstance(loader, GGUFModelLoader)
    assert load_config.load_format == LoadFormat.GGUF


def test_qwen3_5_moe_causal_lm_is_registered():
    from sglang.srt.models.registry import ModelRegistry

    assert "Qwen3_5MoeForCausalLM" in ModelRegistry.get_supported_archs()


def test_gguf_num_hidden_layers_supports_nested_text_config():
    from sglang.srt.model_loader.loader import _get_gguf_num_hidden_layers

    nested_config = SimpleNamespace(num_hidden_layers=40)
    config = SimpleNamespace(text_config=nested_config)

    assert _get_gguf_num_hidden_layers(config) == 40


def test_gguf_dummy_config_uses_nested_text_config_for_moe(tmp_path):
    from sglang.srt.model_loader.loader import _get_gguf_dummy_config

    text_config = SimpleNamespace(vocab_size=151936)
    gguf_path = tmp_path / "model.gguf"
    gguf_path.write_bytes(b"GGUF")
    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"layer_types": ["linear_attention"]}})
    )
    config = SimpleNamespace(text_config=text_config)

    assert _get_gguf_dummy_config(config, str(gguf_path)) is text_config
    assert text_config.layer_types == ["linear_attention"]
