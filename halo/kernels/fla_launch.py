"""Launch options for FLA's autotuned Triton kernels, set before a loaded model's first launch: the delta rule's state
kernel on its ``num_warps=2`` configs on Blackwell."""

import torch
from fla.ops.common import chunk_delta_h
from triton.runtime.autotuner import Autotuner

_BLACKWELL_MAJOR = 10
_BLACKWELL_NUM_WARPS = 2
# Every caller (gated delta rule, KDA, delta rule, the backward's recompute) launches this one kernel object.
_STATE_KERNEL = (chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")


def triton_autotuner(kernel) -> Autotuner | None:
    """The ``triton.Autotuner`` under ``kernel``'s ``triton.heuristics`` wrappers, or ``None``."""
    while kernel is not None and not isinstance(kernel, Autotuner):
        kernel = getattr(kernel, "fn", None)
    return kernel


def apply_fla_launch_options() -> None:
    """Idempotent; run before a loaded model's first FLA launch. Under Triton 3.6 the state kernel's ``num_warps=4``
    configs give run-to-run different, at some positions wrong, outputs and gradients on packed rows, and on B300 also
    on larger unpacked batches; the autotuner picks them at Qwen3.5/3.6 shapes. fla #953 (first released in 0.5.2)
    makes this a no-op. Triton's disk cache keys on the config list."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != _BLACKWELL_MAJOR:
        return
    module, kernel_name = _STATE_KERNEL
    autotuner = triton_autotuner(getattr(module, kernel_name, None))
    if autotuner is None:
        raise RuntimeError(
            f"{module.__name__}.{kernel_name} is not an autotuned FLA kernel here; re-check the Blackwell restriction"
        )
    kept = [config for config in autotuner.configs if config.num_warps == _BLACKWELL_NUM_WARPS]
    if not kept:
        raise RuntimeError(f"{kernel_name} has no num_warps={_BLACKWELL_NUM_WARPS} config left to autotune")
    autotuner.configs = kept
    # A launch before this call may already have picked a dropped config for its shape.
    autotuner.cache = {
        key: config for key, config in autotuner.cache.items() if config.num_warps == _BLACKWELL_NUM_WARPS
    }
