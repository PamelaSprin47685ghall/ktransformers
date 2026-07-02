"""CPU i-quant iq2_k_r4 (337) must be wired in llamafile iqk_mul_mat MoE switch."""

from pathlib import Path


def test_iqk_mul_mat_moe_switch_documents_iq2_k_r4_gap():
    inc = (
        Path(__file__).resolve().parents[3]
        / "third_party"
        / "llamafile"
        / "iqk_mul_mat.inc"
    )
    text = inc.read_text()
    assert "GGML_TYPE_Q6_K" in text
    # ik GGML_TYPE_IQ2_K_R4 = 337; pinned ggml.h ends at BF16=30
    assert "337" not in text and "IQ2_K_R4" not in text