"""Which attention kernels a forward reached, whatever label its config carries. Two kernels agree closely enough that
the numbers alone do not tell them apart, and a label below its kernel's limit keeps a different one in silence."""

import contextlib
import functools
import importlib
from collections import Counter
from collections.abc import Iterator

from transformers import modeling_flash_attention_utils as flash_utils

_FLASH_ENTRY_POINTS = ("flash_attn_func", "flash_attn_varlen_func")
# Per label, where its kernel's entry points live.
_ENTRY_POINTS = {
    "flash_attention_4": ("flash_attn.cute", _FLASH_ENTRY_POINTS),
    "flash_attention_3": ("flash_attn_interface", _FLASH_ENTRY_POINTS),
    "flash_attention_2": ("flash_attn", _FLASH_ENTRY_POINTS),
    "sdpa": ("torch.nn.functional", ("scaled_dot_product_attention",)),
    "sdpa_flex_sliding": ("halo.models.attention.sdpa_flex_sliding", ("_block_sparse_attention",)),
    "qsa_block_sparse": ("halo.models.families.qwen4_exp.sparse_attention", ("block_sparse_attention",)),
}


def _counted(fn, label: str, calls: Counter):
    @functools.wraps(fn)
    def counted(*args, **kwargs):
        calls[label] += 1
        return fn(*args, **kwargs)

    return counted


@contextlib.contextmanager
def kernel_calls() -> Iterator[Counter]:
    """Calls per label over the body. transformers re-imports the flash entry points on its next call, either side."""
    calls: Counter = Counter()
    originals = []
    for label, (module_name, names) in _ENTRY_POINTS.items():
        try:
            module = importlib.import_module(module_name)
        except ImportError:  # An image carries the flash builds of its GPU generation.
            continue
        for name in names:
            original = getattr(module, name)
            originals.append((module, name, original))
            setattr(module, name, _counted(original, label, calls))
    flash_utils._loaded_implementation = None
    try:
        yield calls
    finally:
        for module, name, original in originals:
            setattr(module, name, original)
        flash_utils._loaded_implementation = None
