"""IK extended GGML quantization type IDs and block sizes.

Must match ik_llama.cpp ggml-common.h exactly.  Each entry:
    (block_size, type_size)
where block_size is the number of elements in a quant block and type_size is
the byte count of that block's packed representation.
"""

# ---------------------------------------------------------------------------
# IK-extended type IDs (mainline ggml/gguf library does not know these)
# ---------------------------------------------------------------------------
# The integer values are defined in ik_llama.cpp ggml-common.h as
#   GGML_TYPE_IQ2_K_R4 = 337
#   GGML_TYPE_IQ4_KS_R4 = 344
# These IDs must be kept in sync with the ik fork — an ID mismatch causes
# silent weight corruption.

# ── Type ID constants (from ik_llama.cpp ggml/include/ggml.h) ─────────────────

GGML_TYPE_IQ2_K_R4: int = 337   # block_iq2_k_r4
GGML_TYPE_IQ4_KS_R4: int = 344  # block_iq4_ks_r4
GGML_TYPE_IQ5_KS_R4: int = 352  # block_iq5_ks_r4

QK_K: int = 256

# ── (block_size, type_size) from ggml-common.h static_assert ────────────────
#
# block_iq2_k:   ggml_half d(2) + uint16_t extra(2) + scales[8] + qs[64] = 76
# block_iq2_k_r4: ggml_half d[4](8) + uint8_t extra[8](8) + scales[32] + qs[256] = 304
#   static_assert(sizeof(block_iq2_k_r4) == 4*sizeof(block_iq2_k)) = 304 ✓
#
# block_iq4_ks:   scales[8] + qs[128] = 136
# block_iq4_ks_r4: scales[32] + qs[512] = 544
#   static_assert(sizeof(block_iq4_ks_r4) == 4*sizeof(block_iq4_ks)) = 544 ✓
#
# block_iq5_ks:   scales[8] + qs[128] + qh[32] = 168
# block_iq5_ks_r4: scales[32] + qs[512] + qh[128] = 672
#   static_assert(sizeof(block_iq5_ks_r4) == 4*sizeof(block_iq5_ks)) = 672 ✓

IK_GGML_QUANT_SIZES: dict[int, tuple[int, int]] = {
    GGML_TYPE_IQ2_K_R4:  (QK_K, 304),
    GGML_TYPE_IQ4_KS_R4: (QK_K, 544),
    GGML_TYPE_IQ5_KS_R4: (QK_K, 672),
}


def ik_ggml_type_id(name: str) -> int:
    """Map lowercase IK type name → integer type ID.

    Args:
        name: e.g. ``"iq2_k_r4"``, ``"iq4_ks_r4"``, ``"iq5_ks_r4"``.

    Returns:
        Integer type ID matching ik_llama.cpp ggml/include/ggml.h.

    Raises:
        KeyError: If *name* is unknown.
    """
    _MAPPING: dict[str, int] = {
        "iq2_k_r4":  GGML_TYPE_IQ2_K_R4,
        "iq4_ks_r4": GGML_TYPE_IQ4_KS_R4,
        "iq5_ks_r4": GGML_TYPE_IQ5_KS_R4,
    }
    try:
        return _MAPPING[name.lower()]
    except KeyError:
        known = ", ".join(sorted(_MAPPING))
        raise KeyError(f"Unknown IK GGML type name: {name!r}. Known: {known}") from None
