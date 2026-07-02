from pathlib import Path


def test_avx2_variant_pins_all_avx512_subfeatures_off():
    setup_py = Path(__file__).resolve().parents[2] / "setup.py"
    text = setup_py.read_text()
    avx2_start = text.index('"avx2"')
    avx2_end = text.index('"avx512_base"')
    avx2_block = text[avx2_start:avx2_end]

    assert '"CPUINFER_CPU_INSTRUCT": "AVX2"' in avx2_block
    assert '"CPUINFER_ENABLE_AVX512": "OFF"' in avx2_block
    assert '"CPUINFER_ENABLE_AVX512_VNNI": "OFF"' in avx2_block
    assert '"CPUINFER_ENABLE_AVX512_BF16": "OFF"' in avx2_block
    assert '"CPUINFER_ENABLE_AVX512_VBMI": "OFF"' in avx2_block
    assert '"CPUINFER_ENABLE_AMX": "OFF"' in avx2_block
