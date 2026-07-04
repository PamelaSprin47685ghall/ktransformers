"""RED: AutoAWQ 能否通过加 map 条目识别 qwen3_5_moe。

Acceptance criteria (RED means tests FAIL until patched):
1. check_and_get_model_type("/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-standalone")
   返回 "qwen3_5_moe"（而不是抛 TypeError）。
2. AWQ_CAUSAL_LM_MODEL_MAP 含 "qwen3_5_moe" key.
3. TRANSFORMERS_AUTO_MAPPING_DICT 含 "qwen3_5_moe" key, value == "AutoModelForCausalLM".

注意：本测试不启动真正量化（需要完整模型实例化），只验证注册路径。
"""


def test_autoawq_qwen3_5_moe_registered_in_causal_lm_map():
    from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
    assert "qwen3_5_moe" in AWQ_CAUSAL_LM_MODEL_MAP


def test_autoawq_qwen3_5_moe_registered_in_transformers_mapping():
    from awq.models.base import TRANSFORMERS_AUTO_MAPPING_DICT
    assert TRANSFORMERS_AUTO_MAPPING_DICT["qwen3_5_moe"] == "AutoModelForCausalLM"


def test_autoawq_check_and_get_model_type_returns_qwen3_5():
    from awq.models.auto import check_and_get_model_type
    model_type = check_and_get_model_type(
        "/home/kunweiz/Desktop/Ornith/ornith-gpu-bf16-standalone",
        trust_remote_code=True,
    )
    assert model_type == "qwen3_5_moe"


def test_autoawq_qwen3_5_moe_class_inherits_base():
    import importlib
    from awq.models.base import BaseAWQForCausalLM
    try:
        mod = importlib.import_module("awq.models.qwen3_5_moe")
    except ModuleNotFoundError:
        # 模块缺失 → test fail (RED)
        raise
    cls = getattr(mod, "Qwen3_5MoeAWQForCausalLM")
    assert issubclass(cls, BaseAWQForCausalLM)
    assert hasattr(cls, "get_model_layers")
    assert hasattr(cls, "get_layers_for_scaling")
