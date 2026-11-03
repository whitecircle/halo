"""Launch options for FLA's autotuned Triton kernels, set before a loaded model's first launch: the delta rule's state
kernel on its ``num_warps=2`` configs on Blackwell, the norm kernels at a fixed ``NB`` under ``HALO_FLA_FIXED_NB``."""

import ast

import torch
from fla.modules import fused_norm_gate, l2norm
from fla.ops.common import chunk_delta_h
from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import JITFunction

from halo.env import env_flag

_BLACKWELL_MAJOR = 10
_BLACKWELL_NUM_WARPS = 2
# Every caller (gated delta rule, KDA, delta rule, the backward's recompute) launches this one kernel object.
_STATE_KERNEL = (chunk_delta_h, "chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
FIXED_NB_FLAG = "HALO_FLA_FIXED_NB"
# FLA module -> its autotuned kernels whose ``NB`` constexpr is unread.
FIXED_NB_KERNELS = {
    l2norm: ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
    fused_norm_gate: ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"),
}
_NB = "NB"
_LAUNCH_NB = 1


def triton_autotuner(kernel) -> Autotuner | None:
    """The ``triton.Autotuner`` under ``kernel``'s ``triton.heuristics`` wrappers, or ``None``."""
    while kernel is not None and not isinstance(kernel, Autotuner):
        kernel = getattr(kernel, "fn", None)
    return kernel


def _autotuner(module, kernel_name: str, remedy: str) -> Autotuner:
    autotuner = triton_autotuner(getattr(module, kernel_name, None))
    if autotuner is None:
        raise RuntimeError(f"{module.__name__}.{kernel_name} is not an autotuned FLA kernel here; {remedy}")
    return autotuner


def _restrict_blackwell_state_kernel() -> None:
    """Under Triton 3.6 the ``num_warps=4`` configs give run-to-run different, at some positions wrong, outputs and
    gradients on packed rows, and on B300 also on larger unpacked batches; the autotuner picks them at Qwen3.5/3.6
    shapes. fla #953 (first released in 0.5.2) makes this a no-op. Triton's disk cache keys on the config list."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != _BLACKWELL_MAJOR:
        return
    autotuner = _autotuner(*_STATE_KERNEL, "re-check the Blackwell restriction")
    kept = [config for config in autotuner.configs if config.num_warps == _BLACKWELL_NUM_WARPS]
    if not kept:
        raise RuntimeError(f"{_STATE_KERNEL[1]} has no num_warps={_BLACKWELL_NUM_WARPS} config left to autotune")
    autotuner.configs = kept
    # A launch before this call may already have picked a dropped config for its shape.
    autotuner.cache = {
        key: config for key, config in autotuner.cache.items() if config.num_warps == _BLACKWELL_NUM_WARPS
    }


class _FixedNB:
    """``fn`` stays the JIT function, as the autotuner's disk-cache key walks ``.fn`` down to it."""

    def __init__(self, fn: JITFunction):
        self.fn = fn
        self._position = fn.arg_names.index(_NB)

    def _fixed(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        fixed = dict(kwargs)
        if len(args) > self._position:
            nb, args = args[self._position], (*args[: self._position], _LAUNCH_NB, *args[self._position + 1 :])
        else:
            nb, fixed[_NB] = fixed[_NB], _LAUNCH_NB
        if callable(grid := fixed["grid"]):
            fixed["grid"] = lambda meta: grid({**meta, _NB: nb})
        return args, fixed

    def run(self, *args, **kwargs):
        args, kwargs = self._fixed(args, kwargs)
        return self.fn.run(*args, **kwargs)

    def warmup(self, *args, **kwargs):
        args, kwargs = self._fixed(args, kwargs)
        return self.fn.warmup(*args, **kwargs)

    def __getattr__(self, name: str):
        return getattr(self.fn, name)


def _body_reads(fn: JITFunction, name: str) -> bool:
    """Whether ``fn``'s body names ``name``; its signature declares it either way."""
    (function,) = ast.parse(fn.src).body
    return any(isinstance(node, ast.Name) and node.id == name for stmt in function.body for node in ast.walk(stmt))


def _fix_norm_nb() -> None:
    """FLA takes ``NB = cdiv(rows, 65536)`` as a constexpr that only widens the autotune key. Rows are tokens times
    heads, so at 32 heads a new ``NB`` arrives every 2,048 tokens and Triton recompiles every config. An unread
    constexpr compiles to the same binary, so results are bit-identical; a kernel that reads ``NB`` raises."""
    if not env_flag(FIXED_NB_FLAG):
        return
    remedy = f"unset {FIXED_NB_FLAG} or update FIXED_NB_KERNELS"
    for module, kernel_names in FIXED_NB_KERNELS.items():
        for kernel_name in kernel_names:
            autotuner = _autotuner(module, kernel_name, remedy)
            fn = autotuner.fn
            if isinstance(fn, _FixedNB):
                continue
            if not isinstance(fn, JITFunction) or _NB not in fn.arg_names or _body_reads(fn, _NB):
                raise RuntimeError(
                    f"{module.__name__}.{kernel_name} no longer takes an unread {_NB} constexpr, so a fixed {_NB} "
                    f"would change the kernel; {remedy}"
                )
            autotuner.fn = _FixedNB(fn)


def apply_fla_launch_options() -> None:
    """Idempotent; run before a loaded model's first FLA launch."""
    _restrict_blackwell_state_kernel()
    _fix_norm_nb()
