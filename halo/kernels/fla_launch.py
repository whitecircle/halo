"""Launch options for FLA's autotuned Triton kernels, set before a loaded model's first launch: the delta rule's state
kernel on its ``num_warps=2`` configs on Blackwell, and, under ``HALO_FLA_L2NORM_FIXED_NB``, the q/k L2 norm on one
compiled kernel and one tuned config per feature width and dtype."""

import ast

import torch
import torch.nn as nn
from fla.modules import l2norm as fla_l2norm
from fla.ops.common import chunk_delta_h
from triton.runtime.autotuner import Autotuner

from halo.env import env_flag

_BLACKWELL_MAJOR = 10
_BLACKWELL_NUM_WARPS = 2
# Every caller (gated delta rule, KDA, delta rule, the backward's recompute) launches this one kernel object.
_STATE_KERNEL = (chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
_L2NORM_KERNELS = ("l2norm_fwd_kernel", "l2norm_bwd_kernel")
_L2NORM_LAUNCH_NB = 1
# One FLA NB bucket of rows (``cdiv(rows, 2048 * 32)``): the single tuned config is picked at a size a packed run reaches.
_L2NORM_WARM_ROWS = 2048 * 32
_L2NORM_WARM_DTYPES = (torch.bfloat16, torch.float16)


def triton_autotuner(kernel) -> Autotuner | None:
    """The ``triton.Autotuner`` under ``kernel``'s ``triton.heuristics`` wrappers, or ``None``."""
    while kernel is not None and not isinstance(kernel, Autotuner):
        kernel = getattr(kernel, "fn", None)
    return kernel


class _ConstantNB:
    """The launcher FLA's autotuner calls in place of the JIT function, with ``NB`` held constant; every other attribute
    reads through to the function, so the autotuner's disk-cache key is unchanged."""

    def __init__(self, fn):
        self.fn = fn

    def run(self, *args, **kwargs):
        if "NB" in kwargs:
            kwargs["NB"] = _L2NORM_LAUNCH_NB
        return self.fn.run(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.fn, name)


def uses_fla_l2norm_fixed_nb() -> bool:
    return env_flag("HALO_FLA_L2NORM_FIXED_NB", False)


def _reads_nb(jit_function) -> bool:
    function = next(
        (node for node in ast.walk(ast.parse(jit_function.src)) if isinstance(node, ast.FunctionDef)), None
    )
    if function is None:
        raise RuntimeError(
            f"{jit_function.__name__} has no function definition in its source here, so whether it reads NB cannot be "
            "checked; unset HALO_FLA_L2NORM_FIXED_NB or update patch_fla_l2norm_nb for this fla release"
        )
    return any(
        isinstance(node, ast.Name) and node.id == "NB" for statement in function.body for node in ast.walk(statement)
    )


def patch_fla_l2norm_nb() -> None:
    """Idempotent, on every architecture. NB leaves the tuning key of FLA's ``l2norm_fwd_kernel`` and
    ``l2norm_bwd_kernel`` and their launches get a constant one: no kernel line reads NB, yet each new value compiles and
    tunes again mid-run. A pick can change the reduction order, not the value beyond fp32 rounding. Raises where a
    kernel is not an autotuned function keyed on NB, or reads NB."""
    for name in _L2NORM_KERNELS:
        tuner = triton_autotuner(getattr(fla_l2norm, name, None))
        if tuner is not None and isinstance(tuner.fn, _ConstantNB):
            continue
        if tuner is None or "NB" not in tuner.keys:
            raise RuntimeError(
                f"{fla_l2norm.__name__}.{name} is not an autotuned FLA kernel keyed on NB here; "
                "HALO_FLA_L2NORM_FIXED_NB needs updating for this fla release, or unset it"
            )
        if _reads_nb(tuner.fn):
            raise RuntimeError(
                f"{fla_l2norm.__name__}.{name} reads NB in its body in this fla release, so a constant NB would change "
                "it: unset HALO_FLA_L2NORM_FIXED_NB"
            )
        tuner.keys = [key for key in tuner.keys if key != "NB"]
        tuner.cache = {}  # picks already made were keyed on NB
        tuner.fn = _ConstantNB(tuner.fn)


def _warm_l2norm_width(dim: int, device: torch.device) -> None:
    for dtype in _L2NORM_WARM_DTYPES:
        x = torch.randn(_L2NORM_WARM_ROWS, dim, device=device, dtype=dtype)
        y, rstd = fla_l2norm.l2norm_fwd(x)
        fla_l2norm.l2norm_bwd(y, rstd, x)


def warm_fla_l2norm(model: nn.Module) -> None:
    """Compile and tune the q/k L2 norm, forward and backward, at each key width and device of ``model``'s
    gated-delta-rule layers (the modules holding ``head_k_dim``), so no step does. Rank-local; a no-op without
    ``HALO_FLA_L2NORM_FIXED_NB``."""
    if not uses_fla_l2norm_fixed_nb():
        return
    layouts = set()
    for module in model.modules():
        width = getattr(module, "head_k_dim", None)
        parameter = next(module.parameters(), None)
        if width is not None and parameter is not None and parameter.device.type == "cuda":
            layouts.add((width, parameter.device))
    for width, device in sorted(layouts, key=str):
        _warm_l2norm_width(width, device)


def _restrict_state_kernel() -> None:
    """Under Triton 3.6 the state kernel's ``num_warps=4`` configs give run-to-run different, at some positions wrong,
    outputs and gradients on packed rows, and on B300 also on larger unpacked batches; the autotuner picks them at
    Qwen3.5/3.6 shapes. fla #953 (first released in 0.5.2) makes this a no-op. Triton's disk cache keys on the config
    list."""
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


def apply_fla_launch_options() -> None:
    """Idempotent; run before a loaded model's first FLA launch: the Blackwell state-kernel restriction, and the
    constant-NB q/k norm under ``HALO_FLA_L2NORM_FIXED_NB``, whose kernels the restriction does not touch."""
    _restrict_state_kernel()
    if uses_fla_l2norm_fixed_nb():
        patch_fla_l2norm_nb()
