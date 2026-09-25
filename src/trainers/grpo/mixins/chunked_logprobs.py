"""Chunked per-token log-probs for GRPO, without materializing the full ``[B, T, vocab]`` logits.

TRL computes log-probs from full logits, which is the binding memory peak for large-vocab models.
This computes the same log-probs from the backbone's ``last_hidden_state`` via a dual-chunked
(sequence × vocab) matmul and online softmax with a recompute backward, bounding peak memory by the
tile size, applying the family's verified head transform (:mod:`src.models.head_transform`) on the
way. TRL's ``_compute_loss`` runs unchanged on the resulting ``(B, T)`` log-probs. Works under
FSDP2 (``lm_head`` gathered differentiably with ``full_tensor``), ep1/EP (``lm_head`` dense), and TP
(a replicated head; a gathered-output TP plan takes the full-logits path with the head's output
cloned — ``_writable_logits``).
"""

import inspect
import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.models.utils import _ForwardRedirection

from src.distributed.expert_parallel.dispatcher import bump_forward_generation
from src.distributed.runtime import materialize_dtensor, rank_consensus
from src.distributed.tensor_parallel.state_dict import tp_plan_shards_params
from src.models.head_transform import IDENTITY_HEAD_TRANSFORM, HeadTransform, resolve_head_transform
from src.models.loading.config_levels import text_config
from src.models.modality import config_declares_multimodality
from src.models.structure import base_transformers_model

logger = logging.getLogger(__name__)

# Tiles of the chunked matmul / online softmax. The working set is a few [seq, vocab] fp32 tiles
# (4096 × 16384 × 4 B = 256 MiB each); the loop runs (T / seq) × (V / vocab) iterations, so tiles
# sized well below this leave the sweep launch-bound on large-vocabulary models.
_SEQ_CHUNK = 4096
_VOCAB_CHUNK = 16384

# Bytes per logit the full-logits loss forward holds at its peak: an fp32 plane, or the bf16 logits
# plus the bf16 log-softmax TRL's selective_log_softmax saves for backward.
_FULL_LOGITS_BYTES_PER_LOGIT = 4
# Share of the device memory free at trainer init above which the full-logits plane warns. At init the
# optimizer state, gradients and activations the same step needs are not allocated yet.
_FULL_LOGITS_WARN_FRACTION = 0.5


@dataclass(frozen=True)
class LogitsWidth:
    """The completion logits row a loss forward carries, in tokens, and the setting that bounds it."""

    tokens: int
    set_by: str


def full_logits_verdict(
    rows: int, rows_set_by: str, width: LogitsWidth, vocab: int, free_bytes: int
) -> tuple[bool, str] | None:
    """``(fatal, message)`` for a full-logits plane of ``rows × width × vocab`` against ``free_bytes``:
    fatal when it cannot fit at all, a warning when it takes over :data:`_FULL_LOGITS_WARN_FRACTION` of
    the free memory, ``None`` when it fits comfortably. ``rows_set_by`` names the batch-size setting
    behind ``rows``. Pure, so the thresholds are CPU-testable."""
    plane = rows * width.tokens * vocab * _FULL_LOGITS_BYTES_PER_LOGIT
    if plane <= _FULL_LOGITS_WARN_FRACTION * free_bytes:
        return None
    gib = 1024**3
    fatal = plane > free_bytes
    share = (
        f"more than the {free_bytes / gib:.1f} GiB this device has free"
        if fatal
        else f"{plane / free_bytes:.0%} of the {free_bytes / gib:.1f} GiB free before the optimizer state and "
        f"activations are allocated"
    )
    return fatal, (
        f"The GRPO loss forward materializes full logits: {rows} rows × {width.tokens} tokens × {vocab} vocab × "
        f"{_FULL_LOGITS_BYTES_PER_LOGIT} B = {plane / gib:.1f} GiB, {share}. Set use_chunked_grpo_logprobs: true "
        f"to compute the log-probs in vocab chunks instead, or lower {rows_set_by} ({rows}) or "
        f"{width.set_by}, which sets the {width.tokens}-token width."
    )


def dense_row_spans(attention_mask: torch.Tensor) -> list[tuple[int, int]]:
    """Real-token span ``[lo, hi)`` per row of a padded batch, as forwarded by the dense per-row path.

    Used by both ``_dense_last_hidden_state`` and routing-replay capture/arm trimming: the trimmed
    widths decide how many routing tokens each per-row forward produces and consumes, so both sides
    must derive them from the same rule, including the <2-token widening (a 1-token span is widened
    by a neighbour so the next-token shift leaves at least one position).
    """
    # One D2H for the whole mask: a per-row read serializes CPU and GPU on every row, which defeats
    # CPU launch-ahead on a path that runs repeatedly per gradient microbatch.
    mask = attention_mask.detach().to("cpu", torch.bool)
    spans = []
    for i in range(mask.size(0)):
        real = mask[i].nonzero().view(-1)
        lo, hi = int(real[0]), int(real[-1]) + 1
        if len(real) != hi - lo:
            # A hole in the mask shifts every logprob after it (the dense path tail-aligns).
            raise ValueError(
                f"dense_row_spans: row {i} has a non-contiguous attention mask "
                f"({len(real)} real tokens across span [{lo}, {hi})); the per-row dense "
                f"logprob path requires one contiguous real-token span per row."
            )
        if hi - lo < 2:
            lo, hi = (lo - 1, hi) if lo > 0 else (lo, hi + 1)
        spans.append((lo, hi))
    return spans


def uses_fa4(model) -> bool:
    """Whether ``model`` runs FlashAttention-4. The chunked forward then trims to a dense per-row call
    for exact RoPE positions and rank-uniform collective counts (see ``_dense_last_hidden_state``)."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    impl = getattr(text_config(config), "_attn_implementation", None) or getattr(config, "_attn_implementation", None)
    return impl == "flash_attention_4"


def rows_forward_densely(model, batch_size: int) -> bool:
    """Whether the chunked forward trims each row to its real span, one backbone call per row.

    Mandatory under FA4 (exact RoPE positions and rank-uniform collective counts — see
    ``_dense_last_hidden_state``). Taken on every attention path when rows are forwarded singly anyway
    (``batch_size == 1``): a padded row otherwise costs its rank's widest prompt+completion in every
    layer — quadratic in the full-attention layers — and the head sweep runs over the padded
    completion width, while the trimmed row also takes SDPA's mask-free causal kernel.
    """
    return uses_fa4(model) or batch_size == 1


def _pad_completion_window(values: torch.Tensor, logits_to_keep: int) -> torch.Tensor:
    """Right-pad a ``(1, n)`` per-token tensor to the ``(1, logits_to_keep)`` completion window with
    zeros: the padded positions are masked downstream and must only stay finite."""
    return torch.nn.functional.pad(values, (0, logits_to_keep - values.size(1)))


def _sweep(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    completion_ids: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    head_transform: HeadTransform,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(logps, entropy)``, each ``(B, T)``: the head transform's hidden scale and vocabulary cut are
    applied to the inputs, its logit scale and softcap inside every tile."""
    b, t, h = hidden.shape
    weight, bias = head_transform.kept_rows(weight, bias)
    logps, entropy = _ChunkedSelectiveLogProbEntropyFunction.apply(
        head_transform.scale_hidden(hidden).reshape(b * t, h),
        weight,
        completion_ids.reshape(b * t),
        bias,
        temperature,
        head_transform.logit_scale,
        head_transform.softcap,
        compute_entropy,
    )
    return logps.reshape(b, t), entropy.reshape(b, t)


def chunked_selective_log_softmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    completion_ids: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    head_transform: HeadTransform = IDENTITY_HEAD_TRANSFORM,
) -> torch.Tensor:
    """Log-probs of ``completion_ids`` from ``hidden`` without materializing ``[B, T, vocab]`` logits.

    ``hidden`` ``(B, T, H)``, ``weight`` ``(V, H)``, ``completion_ids`` ``(B, T)`` -> ``(B, T)``
    log-probs. Differentiable in ``hidden``/``weight``; ``temperature`` matches TRL's pre-softmax
    division. ``head_transform`` is the family's head path (:func:`resolve_head_transform`), applied
    before the temperature as the model's own forward applies it.
    """
    return _sweep(hidden, weight, completion_ids, bias, temperature, head_transform, False)[0]


def _matmul_fp32(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``a @ b`` as fp32, or ``out += a @ b`` into an fp32 ``out``.

    On CUDA with a bf16 or fp16 ``b``, both operands are in ``b``'s dtype (an fp32 ``a``, such as the
    backward's gradient tile, is cast to it first) and the tensor-core GEMM accumulates in fp32 and
    returns fp32, so neither the product nor the running sum ``out`` is rounded to half precision. Any
    other ``b`` (CPU, fp32) takes an fp32 matmul. Half operands are not upcast on CUDA: under the pinned
    ``highest`` fp32 matmul precision that product would run on the CUDA cores.
    """
    if b.is_cuda and b.dtype in (torch.bfloat16, torch.float16):
        a = a.to(b.dtype)
        if out is None:
            return torch.mm(a, b, out_dtype=torch.float32)
        return torch.addmm(out, a, b, out_dtype=torch.float32, out=out)
    product = a.float() @ b.float()
    return product if out is None else out.add_(product)


def _logits_tile(
    hidden_chunk, weight_chunk, bias, vocab_start, vocab_end, logit_scale, softcap, inv_t
) -> torch.Tensor:
    """One ``[seq, vocab-tile]`` plane of temperature-scaled logits: the family's logit scale, then its
    softcap (``cap · tanh(logits / cap)``), then the temperature, as the model's forward orders them."""
    logits_chunk = _matmul_fp32(hidden_chunk, weight_chunk.to(hidden_chunk.dtype).t())
    if bias is not None:
        logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
    if logit_scale is not None:
        logits_chunk.mul_(logit_scale)
    if softcap is not None:
        logits_chunk.div_(softcap).tanh_().mul_(softcap)
    return logits_chunk.mul_(inv_t)


@torch.no_grad()
def _selective_logprob_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    logit_scale: float | None,
    softcap: float | None,
    compute_entropy: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-chunked (sequence × vocab) selective log-softmax with an entropy accumulator in the same sweep.

    The entropy needs only ``u = Σ exp(z−m)·z`` on top of the online-softmax accumulators the logprob
    pass already tracks, so fusing them avoids a second full-vocab matmul per gradient microbatch.
    Returns ``(logprobs, log_z, entropy)``, each ``(n_rows,)`` fp32; ``entropy = log_z − u/s``. With
    ``compute_entropy=False`` the ``u`` accumulator (one extra ``[seq, vocab]`` product and reduction
    per tile) is skipped and the entropy comes back as zeros.
    """
    device = hidden.device
    n_rows, _ = hidden.shape
    vocab_size, _ = weight.shape
    inv_t = 1.0 / temperature
    seq_chunk_size = _SEQ_CHUNK

    logprobs = torch.empty((n_rows,), device=device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=device, dtype=torch.float32)
    entropy = torch.zeros((n_rows,), device=device, dtype=torch.float32)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        n_chunk = seq_end - seq_start
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]

        max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        sum_exp_z = torch.zeros((n_chunk,), device=device, dtype=torch.float32) if compute_entropy else None
        target_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        row_idx = torch.arange(n_chunk, device=device)

        for vocab_start in range(0, vocab_size, _VOCAB_CHUNK):
            vocab_end = min(vocab_start + _VOCAB_CHUNK, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end]
            logits_chunk = _logits_tile(
                hidden_chunk, weight_chunk, bias, vocab_start, vocab_end, logit_scale, softcap, inv_t
            )

            chunk_max = logits_chunk.amax(dim=-1)
            max_new = torch.maximum(max_old, chunk_max)
            rescale = torch.exp(max_old - max_new)
            chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

            sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
            if compute_entropy:
                sum_exp_z = sum_exp_z * rescale + (chunk_exp * logits_chunk).sum(dim=-1)
            max_old = max_new

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            target_logit += logits_chunk[row_idx, local_idx] * in_chunk

        log_z_chunk = max_old + torch.log(sum_exp)
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = target_logit - log_z_chunk
        if compute_entropy:
            entropy[seq_start:seq_end] = log_z_chunk - sum_exp_z / sum_exp

    return logprobs, log_z, entropy


def _selective_logprob_backward(
    hidden, weight, targets, bias, log_z, grad_logprobs, temperature, logit_scale, softcap
):
    """Dual-chunked backward: each logits tile is recomputed from the saved ``log_z`` instead of a
    stored ``[T, V]`` plane. Both gradient GEMMs take the fp32 gradient tile cast to the activations'
    dtype (bf16 in training) and accumulate in fp32 (:func:`_matmul_fp32`). Under a softcap the tile's
    gradient carries the ``tanh`` derivative, ``1 − (logits / cap)²`` on the capped logits, and under a
    logit scale that scale."""
    inv_t = 1.0 / temperature
    n_rows, _ = hidden.shape
    vocab_size = weight.shape[0]
    has_bias = bias is not None

    grad_hidden = torch.zeros(hidden.shape, device=hidden.device, dtype=torch.float32)
    grad_weight = torch.zeros(weight.shape, device=weight.device, dtype=torch.float32)
    grad_bias = torch.zeros((vocab_size,), device=weight.device, dtype=torch.float32) if has_bias else None
    grad_logprobs = grad_logprobs.to(torch.float32)

    for seq_start in range(0, n_rows, _SEQ_CHUNK):
        seq_end = min(seq_start + _SEQ_CHUNK, n_rows)
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]
        grad_chunk = grad_logprobs[seq_start:seq_end]
        logz_chunk = log_z[seq_start:seq_end]
        row_idx = torch.arange(seq_end - seq_start, device=hidden.device)

        for vocab_start in range(0, vocab_size, _VOCAB_CHUNK):
            vocab_end = min(vocab_start + _VOCAB_CHUNK, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end]
            logits_chunk = _logits_tile(
                hidden_chunk, weight_chunk, bias, vocab_start, vocab_end, logit_scale, softcap, inv_t
            )

            probs = torch.exp(logits_chunk - logz_chunk.unsqueeze(-1))
            grad_logits = (-grad_chunk).unsqueeze(-1) * probs

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            grad_logits[row_idx, local_idx] += grad_chunk * in_chunk
            grad_logits.mul_(inv_t)
            if softcap is not None:
                grad_logits.mul_(1.0 - (logits_chunk * (temperature / softcap)).square())
            if logit_scale is not None:
                grad_logits.mul_(logit_scale)

            _matmul_fp32(grad_logits, weight_chunk.to(hidden_chunk.dtype), out=grad_hidden[seq_start:seq_end])
            _matmul_fp32(grad_logits.t(), hidden_chunk, out=grad_weight[vocab_start:vocab_end])
            if has_bias:
                grad_bias[vocab_start:vocab_end].add_(grad_logits.sum(dim=0))

    return grad_hidden, grad_weight, grad_bias


def _tp_gathered_output_head(model) -> torch.nn.Module | None:
    """The output embedding when a TP plan shards it, else ``None``.

    Read off the plan transformers applied rather than off ``is_tp_mode``: the MoE path applies TP to
    attention alone and leaves the head whole, and FSDP2 makes the head's weight a DTensor in every
    mode, so neither the axis flag nor the parameter type separates a gathered head from a plain one.
    """
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    plan = getattr(model, "_tp_plan", None)
    if head is None or not plan:
        return None
    name = next((n for n, module in model.named_modules() if module is head), None)
    return head if name is not None and tp_plan_shards_params(f"{name}.weight", plan) else None


class _ChunkedSelectiveLogProbEntropyFunction(torch.autograd.Function):
    """Selective logprob and entropy in one vocab sweep; backward is the dual-chunked recompute above.

    The entropy output is a detached diagnostic (``mark_non_differentiable``); only the logprobs
    carry gradient. The plain (entropy-free) entry shares this Function with the entropy accumulator
    off, so both paths run one backward.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy):
        logprobs, log_z, entropy = _selective_logprob_entropy_forward(
            hidden, weight, targets, bias, temperature, logit_scale, softcap, compute_entropy
        )
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.logit_scale = logit_scale
        ctx.softcap = softcap
        ctx.mark_non_differentiable(entropy)
        return logprobs, entropy

    @staticmethod
    def backward(ctx, grad_logprobs, _grad_entropy):
        hidden, weight, targets, bias, log_z = ctx.saved_tensors
        grad_hidden, grad_weight, grad_bias = _selective_logprob_backward(
            hidden=hidden,
            weight=weight,
            targets=targets,
            bias=bias if ctx.has_bias else None,
            log_z=log_z,
            grad_logprobs=grad_logprobs,
            temperature=ctx.temperature,
            logit_scale=ctx.logit_scale,
            softcap=ctx.softcap,
        )
        return (
            grad_hidden.to(hidden.dtype),
            grad_weight.to(weight.dtype),
            None,
            grad_bias.to(bias.dtype) if ctx.has_bias else None,
            None,
            None,
            None,
            None,
        )


def chunked_selective_log_softmax_with_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    completion_ids: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    head_transform: HeadTransform = IDENTITY_HEAD_TRANSFORM,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probs of ``completion_ids`` and token entropy from one chunked vocab sweep.

    Same contract as :func:`chunked_selective_log_softmax` plus a detached ``(B, T)`` entropy
    (``-Σ p·log p`` over the family's vocabulary at the same temperature). Used by the gradient
    forward, which needs both; a separate entropy pass would repeat the full-vocab matmul.
    """
    return _sweep(hidden, weight, completion_ids, bias, temperature, head_transform, True)


class ChunkedLogprobsCore:
    """Trainer-agnostic chunked-logprob machinery: the batched implementation, the FA4 per-row dense
    forward, and the FSDP2-safe redirection. Subclasses provide ``_get_last_hidden_state`` (the
    backbone forward), ``self.temperature`` and ``self.accelerator``; the construction-time head-path
    check also reads ``self.model``, ``self._use_chunked_grpo_logprobs`` and ``self._pp_runtime``.
    """

    def _check_full_logits_fit(self, width: LogitsWidth | None) -> None:
        """Refuse a full-logits loss forward that cannot fit on this device; warn when it takes a large
        share of it. ``width`` is the logits row the loss forward carries (``None``: nothing bounds it,
        so nothing is checked).

        The rows are the chunk one loss forward materializes at once: ``per_device_train_batch_size``,
        or ``per_device_eval_batch_size`` when evaluation runs and it is larger, since the eval loss
        forward chunks by it. The estimate is a floor on the step's peak. Collective when it runs: a
        plane that fits on one rank and not another raises on every rank, never on one alone.
        """
        if self._use_chunked_grpo_logprobs or width is None or not torch.cuda.is_available():
            return
        head = self.accelerator.unwrap_model(self.model).get_output_embeddings()
        if head is None:
            raise ValueError(
                "GRPO scores completions through the model's output embedding, and this model reports none "
                "(get_output_embeddings() is None)."
            )
        vocab = head.weight.shape[0]
        args = self.args
        rows, rows_set_by = args.per_device_train_batch_size, "per_device_train_batch_size"
        evaluates = args.eval_strategy not in ("no", None) or args.eval_on_start
        if evaluates and args.per_device_eval_batch_size > rows:
            rows, rows_set_by = args.per_device_eval_batch_size, "per_device_eval_batch_size"
        free_bytes, _ = torch.cuda.mem_get_info(self.accelerator.device)
        verdict = full_logits_verdict(rows, rows_set_by, width, vocab, free_bytes)
        fatal = verdict is not None and verdict[0]
        if not rank_consensus(not fatal)[0]:
            raise RuntimeError(
                verdict[1]
                if fatal
                else "Another rank's full-logits plane cannot fit in its free device memory "
                "(its error states the size); set use_chunked_grpo_logprobs: true."
            )
        if verdict is not None and self.accelerator.is_main_process:
            logger.warning(verdict[1])

    @property
    def _chunked_forward_redirection(self) -> _ForwardRedirection:
        """Reuse TRL's ``_forward_redirection`` if built (only when ``use_liger_kernel``), else make one."""
        fr = getattr(self, "_forward_redirection", None)
        if fr is None:
            fr = _ForwardRedirection()
            self._forward_redirection = fr
        return fr

    def _chunked_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False):
        """Chunked ``(logps, entropies)`` for a possibly-wrapped model; the entry point for callers.

        Redirects through the wrapped model so its pre-forward hooks fire (FSDP2 unshards root
        params, ``lm_head`` included) before the implementation runs on the unwrapped module.
        """
        unwrapped = self.accelerator.unwrap_model(model)
        return self._chunked_forward_redirection(
            model,
            unwrapped,
            self._chunked_logps_impl,
            unwrapped,
            input_ids,
            attention_mask,
            logits_to_keep,
            batch_size,
            compute_entropy,
        )

    def _assert_output_embeddings_unadapted(self, unwrapped_model, lm_head) -> None:
        """Reject a PEFT tuner on the output embedding rather than ignore its delta.

        With ``lm_head`` in ``target_modules``, ``get_output_embeddings()`` resolves either to the
        tuner wrapper, whose ``.weight`` property is the base weight, or to the base linear with the
        wrapper elsewhere; both hand the chunked matmul a delta-free weight. Checked once per run.
        """
        if getattr(self, "_lm_head_adapter_checked", False):
            return
        adapted = getattr(lm_head, "base_layer", None) is not None or any(
            module is not lm_head and getattr(module, "base_layer", None) is lm_head
            for module in unwrapped_model.modules()
        )
        if adapted:
            raise ValueError(
                "use_chunked_grpo_logprobs computes log-probs from the base lm_head weight, but a "
                "PEFT adapter wraps the output embedding — its delta would be silently ignored. "
                "Remove lm_head from the adapter's target_modules or disable use_chunked_grpo_logprobs."
            )
        self._lm_head_adapter_checked = True

    def _head_transform(self, unwrapped_model) -> HeadTransform:
        """``unwrapped_model``'s verified head transform, resolved once per model: the verification
        builds a meta-device shell of the model's class."""
        base = base_transformers_model(unwrapped_model)
        transforms = getattr(self, "_head_transforms", None)
        if transforms is None:
            transforms = self._head_transforms = {}
        if id(base) not in transforms:
            transforms[id(base)] = resolve_head_transform(base)
        return transforms[id(base)]

    def _resolve_chunked_head_transform(self) -> None:
        """Verify the policy's head path at construction when the chunked sweep will score it, so a
        family it cannot reproduce is refused before the first rollout rather than at the first loss
        forward. Pure local computation on the class and config: every rank reaches the same verdict.
        Under PP the sweep never runs; the last stage verifies the same contract when it is built."""
        if self._use_chunked_grpo_logprobs and self._pp_runtime is None:
            self._head_transform(self.accelerator.unwrap_model(self.model))

    def _chunked_logps_impl(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy
    ):
        batch_size = batch_size or input_ids.size(0)
        lm_head = unwrapped_model.get_output_embeddings()
        self._assert_output_embeddings_unadapted(unwrapped_model, lm_head)
        weight = materialize_dtensor(lm_head.weight)
        bias = materialize_dtensor(getattr(lm_head, "bias", None))
        head_transform = self._head_transform(unwrapped_model)
        dense = rows_forward_densely(unwrapped_model, batch_size)

        def sweep(hidden, completion_ids):
            if compute_entropy:
                return chunked_selective_log_softmax_with_entropy(
                    hidden, weight, completion_ids, bias, self.temperature, head_transform
                )
            logps = chunked_selective_log_softmax(
                hidden, weight, completion_ids, bias, self.temperature, head_transform
            )
            return logps, None

        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            completion_ids = ids[:, -logits_to_keep:]
            if not dense:
                hidden = self._backbone_hidden_state(unwrapped_model, ids, mask, logits_to_keep)  # (b, ltk, H)
                logps, entropies = sweep(hidden, completion_ids)
                all_logps.append(logps)
                all_entropies.append(entropies)
                continue
            # Dense rows: the sweep covers each row's real completion tokens only (the window's zero
            # tail is masked downstream), so the head pays the row's length, not the padded width.
            hidden, n_comps = self._dense_last_hidden_state(unwrapped_model, ids, mask, logits_to_keep)
            for row, n_comp in enumerate(n_comps):
                width = max(n_comp, 1)
                logps, entropies = sweep(hidden[row : row + 1, :width], completion_ids[row : row + 1, :width])
                all_logps.append(_pad_completion_window(logps, logits_to_keep))
                all_entropies.append(_pad_completion_window(entropies, logits_to_keep) if compute_entropy else None)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _backbone_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """One sub-forward, opening its own EP capacity scope first.

        These enter the backbone directly, so the per-forward pre-hook that scopes the shared DeepEP
        arena never runs and every sub-forward would be judged against the first one's capacity. That
        breaks the dense arm below, whose rows each carry their own length. Rank-uniform: the
        sub-forward count follows the per-rank batch shape, identical on every rank.
        """
        bump_forward_generation()
        return self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

    def _dense_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Per-row, unpadded (dense) backbone forward -> completion hidden ``(rows, logits_to_keep, H)``
        plus each row's real completion-token count.

        Forwarding each row trimmed to its real span with ``attention_mask=None`` keeps FA4 on its
        dense kernel and restores RoPE positions ``[0, len)``. Every supported family's attention is
        relative, so the shift moves no score and the log-probs match the padded forward to
        floating-point noise. One forward per row keeps the FSDP/EP collective count in step. The
        detour is motivated by those position and collective invariants, not by compile cost: FA4's
        varlen kernel JIT-compiles once per head-config and is then length-agnostic. Rows forwarded
        singly on the other attention paths take it for the cost alone: see ``rows_forward_densely``.
        """
        spans = dense_row_spans(attention_mask)
        # Every row's completion count in one D2H: read inside the loop this is another sync per row.
        if logits_to_keep > 0:
            n_comps = attention_mask[:, -logits_to_keep:].bool().sum(dim=1).tolist()
        else:
            n_comps = [0] * attention_mask.size(0)
        outputs = []
        for i, (lo, hi) in enumerate(spans):
            n_comp = int(n_comps[i])
            dense_ids = input_ids[i : i + 1, lo:hi]
            hidden = self._backbone_hidden_state(unwrapped_model, dense_ids, None, max(n_comp, 1))  # (1, >=1, H)
            # Every row stays autograd-connected: a pruned row drops its FSDP/EP backward collectives.
            n_keep = max(n_comp, 1)
            out = hidden.new_zeros(1, logits_to_keep, hidden.size(-1))
            out[:, :n_keep] = hidden[:, -n_keep:]
            outputs.append(out)
        return torch.cat(outputs, dim=0), [int(n) for n in n_comps]


class ChunkedGRPOLogprobsMixin(ChunkedLogprobsCore):
    """Overrides ``_get_per_token_logps_and_entropies`` with a chunked path that does not materialize
    full logits.

    Gated by ``self._use_chunked_grpo_logprobs``; falls back to TRL's full-logits path when off or for
    multimodal inputs (the chunked path is text-only). Place before the trainer's other bases so it
    wins the MRO.
    """

    # Derived from TRL's signature rather than hand-listed, so a release adding a modality kwarg does
    # not slip past this gate into the text-only chunked path.
    _CHUNK_TEXT_KEYS = frozenset(
        ("self", "model", "input_ids", "attention_mask", "logits_to_keep", "batch_size", "compute_entropy")
    )

    @classmethod
    def _multimodal_keys(cls) -> frozenset[str]:
        params = inspect.signature(GRPOTrainer._get_per_token_logps_and_entropies).parameters
        return frozenset(name for name in params if name not in cls._CHUNK_TEXT_KEYS)

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False, **kwargs
    ):
        # TRL's loss forward passes batch_size=None; rows are independent, so bounding it is exact. The
        # bound follows the TRAINER's mode, not the passed model's: a frozen reference model is always
        # in eval, and a different bound there can route the KL's two sides through different paths.
        if batch_size is None:
            batch_size = (
                self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size
            )
        chunked = getattr(self, "_use_chunked_grpo_logprobs", False)
        is_multimodal = any(kwargs.get(k) is not None for k in self._multimodal_keys())
        # Mixed full (image) / chunked (text) paths across ranks diverge FSDP2's collectives and deadlock
        # backward, so any multimodal batch routes all ranks to the full forward; text-only models skip this.
        if chunked and self._model_accepts_multimodal:
            is_multimodal = rank_consensus(is_multimodal)[1]
        if not chunked or is_multimodal:
            with self._writable_logits(model):
                return super()._get_per_token_logps_and_entropies(
                    model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy, **kwargs
                )
        return self._chunked_logps(model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy)

    @contextmanager
    def _writable_logits(self, model):
        """Let TRL's full-logits path divide the logits in place under TP.

        A gathered ``lm_head`` plan (HF's ``colwise_gather_output``, which ``tp_plan="auto"`` gives
        every dense ``*ForCausalLM``) ends its output prep in ``DTensor.to_local()``, returning the
        local tensor straight out of an autograd Function. Autograd forbids an in-place write to such
        a tensor and to every view of it, and TRL divides the sliced logits by the sampling
        temperature in place, so without this a ``tp_size > 1`` run raises ``Output 0 of
        SliceBackward0 is a view and is being modified inplace``. Cloning the head's output restores
        the write, bounded by ``logits_to_keep``. Only TP prepares the output this way, so the clone
        is scoped to TP.
        """
        head = _tp_gathered_output_head(self.accelerator.unwrap_model(model))
        if head is None:
            yield
            return
        handle = head.register_forward_hook(lambda _module, _inputs, output: output.clone())
        try:
            yield
        finally:
            handle.remove()

    @property
    def _model_accepts_multimodal(self) -> bool:
        """Whether the model can receive vision inputs at all (cached: config-static per run)."""
        cached = getattr(self, "_model_accepts_multimodal_cache", None)
        if cached is None:
            config = getattr(self.accelerator.unwrap_model(self.model), "config", None)
            cached = config is not None and config_declares_multimodality(config)
            self._model_accepts_multimodal_cache = cached
        return cached
