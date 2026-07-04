"""IK extended GGML quant block sizes (must match ik_llama.cpp ggml-common)."""

from kt_kernel.utils.gguf_ik_types import IK_GGML_QUANT_SIZES, ik_ggml_type_id


def test_iq2_k_r4_type_id_and_block():
    assert ik_ggml_type_id("iq2_k_r4") == 337
    assert IK_GGML_QUANT_SIZES[337] == (1024, 304)


def test_iq4_ks_r4_type_id():
    assert ik_ggml_type_id("iq4_ks_r4") == 344
    assert 344 in IK_GGML_QUANT_SIZES