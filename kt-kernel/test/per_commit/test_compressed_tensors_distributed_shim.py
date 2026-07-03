import importlib.util
from pathlib import Path


def test_shim_enables_llmcompressor_dist_import():
    shim_path = Path("tools/_compressed_tensors_distributed_shim.py")
    spec = importlib.util.spec_from_file_location("ct_shim", shim_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.install()
    from compressed_tensors.distributed import greedy_bin_packing, wait_for_comms

    items, bins, mapping = greedy_bin_packing(["a", "b"], 2)
    assert len(bins) == 2
    wait_for_comms([])