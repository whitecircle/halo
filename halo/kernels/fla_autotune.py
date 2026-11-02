"""Blackwell autotune restriction for FLA's GatedDeltaNet state kernel.

On sm_10x under Triton 3.6, ``chunk_gated_delta_rule_fwd_kernel_h_blockdim64``'s ``num_warps=4`` configs give
run-to-run different, at some positions wrong, outputs and gradients on packed (``cu_seqlens``) rows, and on B300
also on larger unpacked batches; the autotuner picks them at Qwen3.5/3.6 shapes. Upstream keeps the kernel on
``num_warps=2`` there (fla #953, first released in 0.5.2); this applies the same restriction to the locked FLA, and
is a no-op on a release that ships it. Triton keys its autotune disk cache on the config list, so a pick cached from
the full list is not reused. Every caller (gated delta rule, KDA, delta rule, the backward's recompute) launches the
same kernel object, so restricting it once covers them all.
"""

import importlib
import sys

import torch
from triton.runtime.autotuner import Autotuner

_FLA_PACKAGE = "fla"
_FWD_H_MODULE = "fla.ops.common.chunk_delta_h"
_FWD_H_KERNEL = "chunk_gated_delta_rule_fwd_kernel_h_blockdim64"
_BLACKWELL_MAJOR = 10
_BLACKWELL_NUM_WARPS = 2


def triton_autotuner(kernel) -> Autotuner | None:
    """The ``triton.Autotuner`` under ``kernel``'s ``triton.heuristics`` wrappers, or ``None`` if it is not autotuned."""
    while kernel is not None and not isinstance(kernel, Autotuner):
        kernel = getattr(kernel, "fn", None)
    return kernel


def restrict_fla_blackwell_autotune() -> None:
    """Keep the GatedDeltaNet state kernel on ``num_warps=2`` once FLA is loaded on a Blackwell (sm_10x) process.

    Keyed on FLA being imported, which every model running its kernels has done by the end of its load, whether
    through transformers' hub-kernel factory or a remote-code module's own import. FLA is not imported here before
    that: its import probes the CUDA device.
    """
    if _FLA_PACKAGE not in sys.modules:
        return
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != _BLACKWELL_MAJOR:
        return
    try:
        kernel = getattr(importlib.import_module(_FWD_H_MODULE), _FWD_H_KERNEL)
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            f"{_FWD_H_MODULE}.{_FWD_H_KERNEL} is unavailable ({error}); re-check the Blackwell gate"
        ) from error
    autotuner = triton_autotuner(kernel)
    if autotuner is None:
        raise RuntimeError(f"{_FWD_H_MODULE}.{_FWD_H_KERNEL} is no longer autotuned; re-check the Blackwell gate")
    kept = [config for config in autotuner.configs if config.num_warps == _BLACKWELL_NUM_WARPS]
    if not kept:
        raise RuntimeError(f"{_FWD_H_KERNEL} has no num_warps={_BLACKWELL_NUM_WARPS} config left to autotune")
    autotuner.configs = kept
    # A launch before this call may already have picked a dropped config for its shape.
    autotuner.cache = {
        key: config for key, config in autotuner.cache.items() if config.num_warps == _BLACKWELL_NUM_WARPS
    }
