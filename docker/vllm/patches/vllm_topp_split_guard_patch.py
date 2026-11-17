"""Keep the split top-p mask from wiping a row whose max logit is huge (a thinking-budget cut).

vLLM's Triton top-k/top-p (``vllm.v1.sample.ops.topk_topp_triton``) sends the p-only rows (top-p < 1,
top-k off) of a CUDA batch of at most 64 rows through a split-row pipeline. Its search runs in
probability space; ``_topp_sb_mask_kernel`` then converts the chosen pivot to a logit,
``log(pivot) + log(Z) + M``, and keeps ``x > pivot_logit``. When the budget runs out, Model Runner V2
writes 1e9 on the forced token, so ``M`` is 1e9, where the fp32 spacing is 64: the few-unit
``log(pivot) + log(Z)`` (about -7.7 for any p < 1) rounds away, ``pivot_logit`` lands on ``M`` and the
strict ``>`` masks every token, the forced one included. Gumbel sampling then takes token 0 from an all
``-inf`` row and the processed log-prob is ``-inf - (-inf)``, NaN. The pipeline's own no-mask guard
(``pivot * Z >= 1``) is tested in probability space, before the lossy conversion; the monolithic kernel's
guard (``final_pivot < max_logit``) is tested in logit space and catches it, which is why a row with an
active top-k, or any row of a batch over 64, is unaffected.

Fix: the monolithic kernel's logit-space guard in the split mask kernel. A pivot that does not sit below
the row max leaves the row unmasked, as the monolithic kernel and the pipeline's no-mask branch do, so
only rows whose pivot conversion collapses onto the max change. The kernel is rebuilt from vLLM's own
source with the guard inserted, so a vLLM that edits the anchor line fails ``apply()`` instead of serving
a stale kernel.
"""

from __future__ import annotations

import inspect
import linecache
import logging

import torch
from vllm_thinking_budget_completion_patch import FORCED_LOGIT

logger = logging.getLogger(__name__)

# Stamped on the replacement kernel so ``Dockerfile.vllm`` can assert the patch took.
PATCH_MARKER = "_halo_topp_split_guard"

_ANCHOR = "    pivot_logit = tl.log(pivot) + logZ + M\n"
_GUARD = "    if not (pivot_logit < M):\n        return\n"
_SOURCE_NAME = "<halo vllm_topp_split_guard_patch: _topp_sb_mask_kernel>"

_APPLIED = False
_ORIGINAL_KERNEL = None


def _guarded_kernel(module):
    """Compile ``_topp_sb_mask_kernel`` from vLLM's source with the guard after the pivot conversion."""
    source = inspect.getsource(module._topp_sb_mask_kernel.fn)
    if source.count(_ANCHOR) != 1:
        raise RuntimeError("topk_topp_triton._topp_sb_mask_kernel changed: pivot conversion anchor not found once")
    source = source.replace(_ANCHOR, _ANCHOR + _GUARD)
    # Triton reads a kernel's source through ``inspect``; register it so the rebuilt one has some.
    linecache.cache[_SOURCE_NAME] = (len(source), None, source.splitlines(keepends=True), _SOURCE_NAME)
    namespace = dict(vars(module))
    exec(compile(source, _SOURCE_NAME, "exec"), namespace)
    return namespace["_topp_sb_mask_kernel"]


def apply() -> None:
    """Idempotently swap the split top-p mask kernel for the guarded one."""
    global _APPLIED, _ORIGINAL_KERNEL
    if _APPLIED:
        return

    from vllm.v1.sample.ops import topk_topp_triton  # noqa: PLC0415 — vLLM-only lazy import

    dispatcher = topk_topp_triton._topp_split_mask
    if dispatcher.kernel is not topk_topp_triton._topp_sb_mask_kernel:
        raise RuntimeError("topk_topp_triton._topp_split_mask no longer launches _topp_sb_mask_kernel")
    kernel = _guarded_kernel(topk_topp_triton)
    setattr(kernel, PATCH_MARKER, True)
    _ORIGINAL_KERNEL = dispatcher.kernel
    dispatcher.kernel = kernel
    topk_topp_triton._topp_sb_mask_kernel = kernel
    _APPLIED = True
    logger.info("vllm_topp_split_guard_patch applied to topk_topp_triton._topp_split_mask")


def _split_top_p(module, row: torch.Tensor, p: float, num_sm: int) -> torch.Tensor:
    logits = row.clone().unsqueeze(0)
    module._apply_topp_split(logits, None, torch.full((1,), p), float("-inf"), num_sm)
    return logits[0]


def self_check() -> None:
    """Drive the split pipeline on a budget-cut row and an ordinary one; raise if either misbehaves.

    The guarded kernel must leave the cut row untouched and mask the ordinary row, whose integer logits
    tie at the cut, exactly as upstream does. Upstream must still wipe the cut row; once it does not,
    the bug is fixed there and this patch should be dropped. Runs on CPU under ``TRITON_INTERPRET=1``.
    """
    from vllm.v1.sample.ops import topk_topp_triton  # noqa: PLC0415 — vLLM-only lazy import

    if not _APPLIED:
        raise RuntimeError("vllm_topp_split_guard_patch not applied")
    vocab_size, top_p = 16384, 0.6
    num_sm = 148  # B200: 32 slices per row
    ordinary = torch.round(torch.randn(vocab_size, generator=torch.Generator().manual_seed(0)) * 1.5)
    cut = ordinary.clone()
    cut[-1] = FORCED_LOGIT

    if not torch.equal(_split_top_p(topk_topp_triton, cut, top_p, num_sm), cut):
        raise RuntimeError("split top-p still masks a budget-cut row")

    masked = _split_top_p(topk_topp_triton, ordinary, top_p, num_sm)
    survivors = int(torch.isfinite(masked).sum())
    if not 0 < survivors < vocab_size or not torch.isfinite(masked[int(ordinary.argmax())]):
        raise RuntimeError(f"split top-p no longer masks an ordinary row ({survivors}/{vocab_size} kept)")

    dispatcher = topk_topp_triton._topp_split_mask
    patched, dispatcher.kernel = dispatcher.kernel, _ORIGINAL_KERNEL
    try:
        upstream_cut = _split_top_p(topk_topp_triton, cut, top_p, num_sm)
        upstream_ordinary = _split_top_p(topk_topp_triton, ordinary, top_p, num_sm)
    finally:
        dispatcher.kernel = patched
    if int(torch.isfinite(upstream_cut).sum()) != 0:
        raise RuntimeError("upstream split top-p no longer wipes a budget-cut row; drop this patch")
    if not torch.equal(upstream_ordinary, masked):
        raise RuntimeError("guard changed the mask of an ordinary row")
