"""RED: llmcompressor 能否识别 qwen3_5_moe 并构建 AWQ recipe。

Acceptance criteria:
1. AutoModelForCausalLM._model_mapping[type(cfg)] 返回
   Qwen3_5MoeForCausalLM（transformers 5.12 内置，无需 trust_remote_code 额外文件）。
2. llmcompressor.modifiers.quantization 中存在 AWQModifier（或 GPTQModifier），
   可实例化并描述 4-bit/group-128/AWQ 配置。
3. 不实例化模型权重、不实际运行量化（仅配置层验证）。

如果 1) 失败 → transformers 版本需 ≥ 5.12。
如果 2) 失败 → llmcompressor API 变更或 AWQ recipe 路径不存在。
"""
from __future__ import annotations

import pytest

STANDALONE = "/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-standalone"


def test_transformers_native_qwen3_5_moe_causal_lm():
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(STANDALONE, trust_remote_code=True)
    assert cfg.model_type == "qwen3_5_moe"
    cls = AutoModelForCausalLM._model_mapping[type(cfg)]
    assert cls.__name__ == "Qwen3_5MoeForCausalLM"
    assert "qwen3_5_moe" in cls.__module__


def test_llmcompressor_awq_modifier_exists_and_instantiates():
    # 仅验证 API 与 recipe 配置合法，不实例化模型
    from llmcompressor.modifiers.quantization import AWQModifier
    recipe = {"modifiers": [AWQModifier(config_groups={"group_0": {"weights": {"num_bits": 4, "group_size": 128}}})]}
    assert "modifiers" in recipe
    assert isinstance(recipe["modifiers"][0], AWQModifier)


def test_transformers_models_qwen3_5_moe_has_expected_modules():
    # AWQ 量化需要遍历 layers 与 linear；先验证模型类能暴露出这些结构签名
    import inspect
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer
    src = inspect.getsource(Qwen3_5MoeDecoderLayer.__init__)
    for expected in ("self_attn", "mlp"):
        assert expected in src, f"Qwen3_5MoeDecoderLayer missing {expected}"


def test_llmcompressor_recipe_apply_dry_run_can_resolve_model_schema():
    # 只解析 model config / config group — 不加载权重、不做校准
    import os
    import tempfile
    from transformers import AutoConfig
    from llmcompressor.recipe import Recipe
    awq_recipe_str = """
quant_stage:
    quant_modifiers:
        AWQModifier:
            config_groups:
                group_0:
                    weights:
                        num_bits: 4
                        group_size: 128
                        symmetric: true
"""
    recipe = Recipe.create_instance(awq_recipe_str)
    # recipe 本身应能解析；与模型无关
    assert recipe is not None
    cfg = AutoConfig.from_pretrained(STANDALONE, trust_remote_code=True)
    assert cfg.vocab_size == 248320
    assert cfg.hidden_size == 2048


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
