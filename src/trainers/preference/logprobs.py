"""fp32 sequence log-probs on TRL's DPO and KTO loss and reference paths.

Outside Liger, TRL scores every sequence through ``selective_log_softmax``, whose bf16 branch returns
bf16 per-token log-probs that TRL then sums into a bf16 sequence log-prob: at ``|logp|`` in [8192,
16384) the bf16 grid is 64 nats, so a DPO margin carries tens of nats of rounding. That path runs
wherever TRL's fused Liger loss does not: every KTO run, and every DPO run except an unsharded one
whose model was loaded with Liger's fused linear cross-entropy. :class:`FP32LogprobsMixin` rebinds
that name, for the duration of TRL's call, to the fp32 chunked
:func:`~src.kernels.logprobs.selective_logprobs`, so TRL's own code downstream — the
``[chosen ⧺ rejected]`` split, the completion masks, ``ld_alpha``'s shared/tail sums, the WPO
weights' log-prob term, KTO's KL term and the ``logps/*`` / ``rewards/*`` metrics — reads fp32
values. What TRL computes from the logits directly (the ``sft`` loss type's cross-entropy, WPO's
normalizer, the entropy and logit metrics) keeps TRL's precision.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from trl.trainer.utils import selective_log_softmax

from src.kernels.logprobs import selective_logprobs

_TRL_LOGPROB_FN = selective_log_softmax.__name__


@contextmanager
def _fp32_logprobs(trl_method: Callable) -> Iterator[None]:
    """Rebind ``selective_log_softmax`` to :func:`selective_logprobs` in the globals ``trl_method``
    resolves it from, while the block runs."""
    namespace = trl_method.__func__.__globals__
    bound = namespace.get(_TRL_LOGPROB_FN)
    if bound is selective_logprobs:
        # An enclosing call already rebound it and restores it on exit.
        yield
        return
    if bound is not selective_log_softmax:
        raise RuntimeError(
            f"{trl_method.__qualname__} no longer reads TRL's {_TRL_LOGPROB_FN} from "
            f"{namespace.get('__name__')}, so its log-probs cannot be routed through the fp32 path. "
            f"Re-audit FP32LogprobsMixin against the installed TRL."
        )
    namespace[_TRL_LOGPROB_FN] = selective_logprobs
    try:
        yield
    finally:
        namespace[_TRL_LOGPROB_FN] = selective_log_softmax


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

    def compute_ref_log_probs(self, inputs):
        trl_compute_ref_log_probs = super().compute_ref_log_probs
        with _fp32_logprobs(trl_compute_ref_log_probs):
            return trl_compute_ref_log_probs(inputs)
