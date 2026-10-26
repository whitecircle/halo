"""fp32 sequence log-probs on TRL's DPO and KTO loss and reference paths.

Outside its chunked log-prob path (``use_liger_kernel``), TRL sums bf16 per-token log-probs from
``selective_log_softmax`` (``selective_log_softmax_and_entropy`` on the policy) into a bf16 sequence log-prob
(64-nat grid at |logp| ≥ 8192). :class:`FP32LogprobsMixin` rebinds both names for the duration of TRL's call to
functions taking their log-probs from the fp32 chunked :func:`~src.kernels.logprobs.selective_logprobs`; what TRL
computes from the logits directly (the entropy included) keeps its precision.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import torch
from trl.trainer.utils import entropy_from_logits, selective_log_softmax, selective_log_softmax_and_entropy

from src.kernels.logprobs import selective_logprobs


def _fp32_logprobs_and_entropy(logits, index, entropy_requires_grad=True, temperature=1.0, row_mask=None):
    """TRL's ``selective_log_softmax_and_entropy`` with its log-probs from :func:`selective_logprobs`;
    the entropy is TRL's own."""
    if temperature != 1.0:
        logits = logits / temperature
    logps = selective_logprobs(logits, index, row_mask=row_mask)
    with torch.set_grad_enabled(entropy_requires_grad and torch.is_grad_enabled()):
        entropy = entropy_from_logits(logits)
    if row_mask is not None:
        entropy = entropy.masked_fill(row_mask == 0, 0.0)
    return logps, entropy


# The name TRL's trainer modules import each log-prob function under: (TRL's function, its fp32 stand-in).
_FP32_REBINDS = {
    selective_log_softmax.__name__: (selective_log_softmax, selective_logprobs),
    selective_log_softmax_and_entropy.__name__: (selective_log_softmax_and_entropy, _fp32_logprobs_and_entropy),
}


@contextmanager
def _fp32_logprobs(trl_method: Callable) -> Iterator[None]:
    """Rebind TRL's log-prob functions to their fp32 stand-ins in the globals ``trl_method`` resolves
    them from, while the block runs."""
    namespace = trl_method.__func__.__globals__
    if all(namespace.get(name) is fp32 for name, (_, fp32) in _FP32_REBINDS.items()):
        # An enclosing call already rebound them and restores them on exit.
        yield
        return
    stale = [name for name, (trl_fn, _) in _FP32_REBINDS.items() if namespace.get(name) is not trl_fn]
    if stale:
        raise RuntimeError(
            f"{trl_method.__qualname__} no longer reads TRL's {', '.join(stale)} from "
            f"{namespace.get('__name__')}, so its log-probs cannot be routed through the fp32 path. "
            f"Re-audit FP32LogprobsMixin against the installed TRL."
        )
    for name, (_, fp32) in _FP32_REBINDS.items():
        namespace[name] = fp32
    try:
        yield
    finally:
        for name, (trl_fn, _) in _FP32_REBINDS.items():
            namespace[name] = trl_fn


class FP32LogprobsMixin:
    """Runs TRL's non-Liger loss and its reference log-prob pass with fp32 per-token log-probs.

    Listed ahead of the TRL base so both overrides wrap it; Liger and the pipeline-parallel loss
    never reach either method.
    """

    # Recorded in every reference split the precompute saves, so one summed at another precision is
    # not reused on resume.
    logprob_precision = "float32"

    def _compute_loss(self, model, inputs, return_outputs):
        trl_compute_loss = super()._compute_loss
        with _fp32_logprobs(trl_compute_loss):
            return trl_compute_loss(model, inputs, return_outputs)

    def compute_ref_log_probs(self, model, inputs):
        trl_compute_ref_log_probs = super().compute_ref_log_probs
        with _fp32_logprobs(trl_compute_ref_log_probs):
            return trl_compute_ref_log_probs(model, inputs)
