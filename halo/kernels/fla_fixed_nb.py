"""Launch FLA's row-blocked norm kernels without per-``NB`` recompiles (``HALO_FLA_FIXED_NB``, default off).

FLA's l2norm kernels (the q/k norm of the gated delta rule and KDA) and its gated-norm kernels take
``NB = cdiv(rows, 65536)`` as a constexpr that only widens their autotune key; the bodies never read it. Rows are
tokens times heads, so at 32 heads a new ``NB`` arrives every 2,048 tokens, and Triton compiles every autotune config
again for each value its cache has not seen. With the flag the autotuner still tunes on the real ``NB`` (same key,
same disk cache) and launches its pick with ``NB`` fixed, and a callable launch grid still sees the real one: an
unread constexpr compiles to the same binary, so the results are bit-identical.
"""

import ast
import sys

from triton.runtime.jit import JITFunction

from halo.env import env_flag
from halo.kernels.fla_autotune import triton_autotuner

FIXED_NB_FLAG = "HALO_FLA_FIXED_NB"
# FLA module -> its autotuned kernels whose ``NB`` constexpr is unread.
FIXED_NB_KERNELS = {
    "fla.modules.l2norm": ("l2norm_fwd_kernel", "l2norm_bwd_kernel"),
    "fla.modules.fused_norm_gate": ("layer_norm_gated_fwd_kernel", "layer_norm_gated_bwd_kernel"),
}
_NB = "NB"
_LAUNCH_NB = 1


class _FixedNB:
    """The JIT function an autotuner launches and warms up, with ``NB`` fixed. ``fn`` stays that JIT function: the
    autotuner's disk-cache key walks ``.fn`` down to it."""

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


def fix_fla_norm_nb() -> None:
    """With :data:`FIXED_NB_FLAG` set, launch the :data:`FIXED_NB_KERNELS` of every FLA module already imported at
    a fixed ``NB``. Idempotent.

    Keyed on the module being imported, which a model running these kernels has done by the end of its load; FLA is
    not imported here (its import probes the CUDA device). A kernel that no longer takes an unread ``NB`` raises:
    fixing it would then change what the kernel computes.
    """
    if not env_flag(FIXED_NB_FLAG):
        return
    for module_name, kernel_names in FIXED_NB_KERNELS.items():
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for kernel_name in kernel_names:
            qualname = f"{module_name}.{kernel_name}"
            autotuner = triton_autotuner(getattr(module, kernel_name, None))
            if autotuner is None:
                raise RuntimeError(
                    f"{FIXED_NB_FLAG}: {qualname} is no longer autotuned; unset the flag or update the table"
                )
            fn = autotuner.fn
            if isinstance(fn, _FixedNB):
                continue
            if not isinstance(fn, JITFunction) or _NB not in fn.arg_names or _body_reads(fn, _NB):
                raise RuntimeError(
                    f"{FIXED_NB_FLAG}: {qualname} no longer takes an unread {_NB} constexpr, so a fixed {_NB} would "
                    f"change the kernel; unset the flag or update the table"
                )
            autotuner.fn = _FixedNB(fn)
