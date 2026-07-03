"""Single-process stubs for llmcompressor when compressed-tensors lacks distributed."""
from __future__ import annotations

import sys
import types
from typing import Callable, Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


def greedy_bin_packing(
    items: list[T],
    num_bins: int,
    item_weight_fn: Callable[[T], int | float] = lambda x: 1,
) -> tuple[list[T], list[list[T]], dict[T, int]]:
    items.sort(key=item_weight_fn, reverse=True)
    bin_to_items: list[list[T]] = [[] for _ in range(num_bins)]
    item_to_bin: dict[T, int] = {}
    bin_weights: list[float] = [0.0] * num_bins
    for item in items:
        target_bin = bin_weights.index(min(bin_weights))
        bin_to_items[target_bin].append(item)
        item_to_bin[item] = target_bin
        bin_weights[target_bin] += float(item_weight_fn(item))
    return items, bin_to_items, item_to_bin


def wait_for_comms(pending_comms: list | None = None) -> None:
    if not pending_comms:
        return
    for work in pending_comms:
        if hasattr(work, "wait"):
            work.wait()
    pending_comms.clear()


def is_distributed() -> bool:
    return False


def is_source_process() -> bool:
    return True


def _patch_offload_dist_utils() -> None:
    import compressed_tensors.offload.dist_utils as du

    if not hasattr(du, "is_source_process"):
        du.is_source_process = du.is_rank0


def _patch_dispatch_device_memory() -> None:
    import torch

    import compressed_tensors.offload.dispatch as dispatch

    def get_device_memory():
        import os

        total_ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        return {torch.device("cpu"): total_ram}

    dispatch.get_device_memory = get_device_memory


def install() -> None:
    _patch_offload_dist_utils()
    _patch_dispatch_device_memory()
    if "compressed_tensors.distributed" in sys.modules:
        return
    pkg = types.ModuleType("compressed_tensors.distributed")
    pkg.greedy_bin_packing = greedy_bin_packing
    pkg.wait_for_comms = wait_for_comms
    pkg.is_distributed = is_distributed
    pkg.is_source_process = is_source_process
    pkg.as_broadcastable = lambda t, *a, **k: t
    pkg.get_source_rank = lambda: 0
    pkg.init_dist = lambda *a, **k: None
    pkg.set_source_process = lambda *a, **k: None
    pkg.replace_module_parallel = lambda *a, **k: None
    sys.modules["compressed_tensors.distributed"] = pkg