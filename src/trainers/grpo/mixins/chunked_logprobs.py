"""Chunked per-token log-probs for GRPO — never materializes the full ``[B, T, vocab]`` logits.

TRL computes log-probs from full logits, which for large-vocab models is the binding memory peak and
OOMs long trajectories. This computes the SAME log-probs from the backbone's ``last_hidden_state`` via
a vocab-chunked matmul + online softmax (Liger's ``_ChunkedSelectiveLogProbFunction``) with a recompute
backward, bounding peak memory by chunk size. TRL's ``_compute_loss`` runs unchanged on the resulting
``(B, T)`` log-probs. Works under FSDP2 (``lm_head`` gathered differentiably with ``full_tensor``),
ep1/EP (``lm_head`` dense), and TP (a replicated head; a gathered-output TP plan takes the
full-logits path with the head's output cloned — ``_writable_logits``).
"""

import inspect
from contextlib import contextmanager

import torch
from liger_kernel.chunked_loss.fused_linear_ppo import (
    _SELECTIVE_LOGPROB_SEQ_CHUNK_SIZE,
    _ChunkedSelectiveLogProbFunction,
    _selective_logprob_backward,
)
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.models.utils import _ForwardRedirection

from src.distributed.expert_parallel.dispatcher import bump_forward_generation
from src.distributed.runtime import materialize_dtensor, rank_consensus
from src.distributed.tensor_parallel.state_dict import tp_plan_shards_params
from src.models.loading.config_levels import text_config
from src.models.modality import config_declares_multimodality

# Vocab tile for the chunked matmul / online softmax, independent of full vocab size.
_VOCAB_CHUNK = 4096


def dense_row_spans(attention_mask: torch.Tensor) -> list[tuple[int, int]]:
    """Real-token span ``[lo, hi)`` per row of a padded batch, as forwarded by the FA4 dense path.

    Single source for ``_dense_last_hidden_state`` and routing-replay capture/arm trimming: the
    trimmed widths decide how many routing tokens each per-row forward produces/consumes, so both
    sides must derive them from the same rule — including the <2-token widening (a 1-token span is
    widened by a neighbour so the next-token shift leaves >=1 position).
    """
    # One D2H for the whole mask: per-row on device costs ~4 CPU-GPU serializations a row, on a path
    # that runs 2-3x per gradient microbatch, which defeats CPU launch-ahead.
    mask = attention_mask.detach().to("cpu", torch.bool)
    spans = []
    for i in range(mask.size(0)):
        real = mask[i].nonzero().view(-1)
        lo, hi = int(real[0]), int(real[-1]) + 1
        if len(real) != hi - lo:
            # A hole in the mask would silently shift every logprob after it (the dense path tail-aligns).
            raise ValueError(
                f"dense_row_spans: row {i} has a non-contiguous attention mask "
                f"({len(real)} real tokens across span [{lo}, {hi})); the FA4 per-row dense "
                f"logprob path requires one contiguous real-token span per row."
            )
        if hi - lo < 2:
            lo, hi = (lo - 1, hi) if lo > 0 else (lo, hi + 1)
        spans.append((lo, hi))
    return spans


def uses_fa4(model) -> bool:
    """Whether ``model`` runs FlashAttention-4 — the chunked forward then trims to a dense per-row
    call for exact RoPE positions and rank-uniform collective counts (see ``_dense_last_hidden_state``)."""
    config = getattr(model, "config", None)
    if config is None:
        return False
    impl = getattr(text_config(config), "_attn_implementation", None) or getattr(config, "_attn_implementation", None)
    return impl == "flash_attention_4"


def chunked_selective_log_softmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    completion_ids: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
) -> torch.Tensor:
    """Log-probs of ``completion_ids`` from ``hidden`` without materializing ``[B, T, vocab]`` logits.

    ``hidden`` ``(B, T, H)``, ``weight`` ``(V, H)``, ``completion_ids`` ``(B, T)`` → ``(B, T)`` log-probs.
    Differentiable in ``hidden``/``weight``; ``temperature`` matches TRL's pre-softmax division.
    """
    b, t, h = hidden.shape
    logps = _ChunkedSelectiveLogProbFunction.apply(
        hidden.reshape(b * t, h), weight, completion_ids.reshape(b * t), bias, temperature, _VOCAB_CHUNK
    )
    return logps.reshape(b, t)


@torch.no_grad()
def _selective_logprob_entropy_forward(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
    vocab_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Liger's ``_selective_logprob_forward`` extended with an entropy accumulator in the SAME sweep.

    The entropy needs only ``u = Σ exp(z−m)·z`` on top of the online-softmax accumulators the logprob
    pass already tracks, so fusing them halves the head-side work — a second full-vocab matmul over a
    ~200k vocab otherwise runs per gradient microbatch purely for the entropy diagnostic. Returns
    ``(logprobs, log_z, entropy)``, each ``(n_rows,)`` fp32; ``entropy = log_z − u/s``.
    """
    device = hidden.device
    n_rows, _ = hidden.shape
    vocab_size, _ = weight.shape
    inv_t = 1.0 / temperature
    seq_chunk_size = _SELECTIVE_LOGPROB_SEQ_CHUNK_SIZE

    logprobs = torch.empty((n_rows,), device=device, dtype=torch.float32)
    log_z = torch.empty((n_rows,), device=device, dtype=torch.float32)
    entropy = torch.empty((n_rows,), device=device, dtype=torch.float32)

    for seq_start in range(0, n_rows, seq_chunk_size):
        seq_end = min(seq_start + seq_chunk_size, n_rows)
        n_chunk = seq_end - seq_start
        hidden_chunk = hidden[seq_start:seq_end]
        targets_chunk = targets[seq_start:seq_end]

        max_old = torch.full((n_chunk,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        sum_exp_z = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((n_chunk,), device=device, dtype=torch.float32)
        row_idx = torch.arange(n_chunk, device=device)

        for vocab_start in range(0, vocab_size, vocab_chunk_size):
            vocab_end = min(vocab_start + vocab_chunk_size, vocab_size)
            weight_chunk = weight[vocab_start:vocab_end]
            logits_chunk = (hidden_chunk @ weight_chunk.to(hidden.dtype).t()).float()
            if bias is not None:
                logits_chunk.add_(bias[vocab_start:vocab_end].to(torch.float32))
            logits_chunk.mul_(inv_t)

            chunk_max = logits_chunk.amax(dim=-1)
            max_new = torch.maximum(max_old, chunk_max)
            rescale = torch.exp(max_old - max_new)
            chunk_exp = torch.exp(logits_chunk - max_new.unsqueeze(-1))

            sum_exp = sum_exp * rescale + chunk_exp.sum(dim=-1)
            sum_exp_z = sum_exp_z * rescale + (chunk_exp * logits_chunk).sum(dim=-1)
            max_old = max_new

            in_chunk = (targets_chunk >= vocab_start) & (targets_chunk < vocab_end)
            local_idx = torch.clamp(targets_chunk - vocab_start, 0, vocab_end - vocab_start - 1)
            target_logit += logits_chunk[row_idx, local_idx] * in_chunk

        log_z_chunk = max_old + torch.log(sum_exp)
        log_z[seq_start:seq_end] = log_z_chunk
        logprobs[seq_start:seq_end] = target_logit - log_z_chunk
        entropy[seq_start:seq_end] = log_z_chunk - sum_exp_z / sum_exp

    return logprobs, log_z, entropy


def _tp_gathered_output_head(model) -> torch.nn.Module | None:
    """The output embedding when a TP plan shards it, else ``None``.

    Read off the plan transformers actually applied, not off ``is_tp_mode``: the MoE path applies TP
    to attention alone and leaves the head whole, and FSDP2 makes the head's weight a DTensor in
    every mode — so neither the axis flag nor the parameter type separates a gathered head from a
    plain one.
    """
    head = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    plan = getattr(model, "_tp_plan", None)
    if head is None or not plan:
        return None
    name = next((n for n, module in model.named_modules() if module is head), None)
    return head if name is not None and tp_plan_shards_params(f"{name}.weight", plan) else None


class _ChunkedSelectiveLogProbEntropyFunction(torch.autograd.Function):
    """Selective logprob + entropy in one vocab sweep; backward is Liger's recompute backward.

    The entropy output is a detached diagnostic (``mark_non_differentiable``); only the logprobs
    carry gradient, through exactly the same saved tensors and backward as Liger's
    ``_ChunkedSelectiveLogProbFunction``.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, bias, temperature, vocab_chunk_size):
        logprobs, log_z, entropy = _selective_logprob_entropy_forward(
            hidden, weight, targets, bias, temperature, vocab_chunk_size
        )
        if bias is None:
            bias = hidden.new_empty((0,))
        ctx.save_for_backward(hidden, weight, targets, bias, log_z)
        ctx.has_bias = bias.numel() > 0
        ctx.temperature = temperature
        ctx.vocab_chunk_size = vocab_chunk_size
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
            vocab_chunk_size=ctx.vocab_chunk_size,
        )
        return (
            grad_hidden.to(hidden.dtype),
            grad_weight.to(weight.dtype),
            None,
            grad_bias.to(bias.dtype) if ctx.has_bias else None,
            None,
            None,
        )


def chunked_selective_log_softmax_with_entropy(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    completion_ids: torch.Tensor,
    bias: torch.Tensor | None,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-probs of ``completion_ids`` AND token entropy from one chunked vocab sweep.

    Same contract as :func:`chunked_selective_log_softmax` plus a detached ``(B, T)`` entropy
    (``-Σ p·log p`` over the full vocab at the same temperature) — for the gradient forward, where
    TRL always wants both and a separate entropy pass would repeat the full-vocab matmul.
    """
    b, t, h = hidden.shape
    logps, entropy = _ChunkedSelectiveLogProbEntropyFunction.apply(
        hidden.reshape(b * t, h), weight, completion_ids.reshape(b * t), bias, temperature, _VOCAB_CHUNK
    )
    return logps.reshape(b, t), entropy.reshape(b, t)


class ChunkedLogprobsCore:
    """Trainer-agnostic chunked-logprob machinery: the batched impl, the FA4 per-row dense forward,
    and the FSDP2-safe redirection. Subclasses provide ``_get_last_hidden_state`` (the backbone
    forward), ``self.temperature``, and ``self.accelerator``.
    """

    @property
    def _chunked_forward_redirection(self) -> _ForwardRedirection:
        """Reuse TRL's ``_forward_redirection`` if built (only when ``use_liger_kernel``), else own one."""
        fr = getattr(self, "_forward_redirection", None)
        if fr is None:
            fr = _ForwardRedirection()
            self._forward_redirection = fr
        return fr

    def _chunked_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False):
        """Chunked ``(logps, entropies)`` for a possibly-wrapped model — the single entry for every trainer.

        Redirects through the wrapped model so its pre-forward hooks fire (FSDP2 unshards root
        params, incl. ``lm_head``) before the impl runs on the unwrapped module.
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
        """Refuse a PEFT tuner on the output embedding rather than silently ignore it.

        With ``lm_head`` in ``target_modules``, ``get_output_embeddings()`` resolves either to the
        tuner wrapper itself — whose ``.weight`` property is the BASE weight — or to the base linear
        with the wrapper elsewhere; both hand the chunked matmul a delta-free weight, measured ~0.25
        nats of silent log-prob error. Checked once per run.
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

    def _chunked_logps_impl(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy
    ):
        batch_size = batch_size or input_ids.size(0)
        lm_head = unwrapped_model.get_output_embeddings()
        self._assert_output_embeddings_unadapted(unwrapped_model, lm_head)
        weight = materialize_dtensor(lm_head.weight)
        bias = materialize_dtensor(getattr(lm_head, "bias", None))
        dense = uses_fa4(unwrapped_model)

        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            hidden = (
                self._dense_last_hidden_state(unwrapped_model, ids, mask, logits_to_keep)
                if dense
                else self._backbone_hidden_state(unwrapped_model, ids, mask, logits_to_keep)
            )  # (b, ltk, H)
            completion_ids = ids[:, -logits_to_keep:]
            if compute_entropy:
                logps, entropies = chunked_selective_log_softmax_with_entropy(
                    hidden, weight, completion_ids, bias, self.temperature
                )
                all_entropies.append(entropies)
            else:
                logps = chunked_selective_log_softmax(hidden, weight, completion_ids, bias, self.temperature)
            all_logps.append(logps)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _backbone_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """One sub-forward, opening its own EP capacity scope first.

        These enter the BACKBONE directly, so the per-forward pre-hook that scopes the shared DeepEP
        arena never runs and every sub-forward would be judged against the first one's capacity —
        fatal on the dense arm below, whose rows each carry their own length. Rank-uniform: the
        sub-forward count is the per-rank batch shape, identical on every rank.
        """
        bump_forward_generation()
        return self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

    def _dense_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Per-row, unpadded (dense) backbone forward → completion hidden ``(rows, logits_to_keep, H)``.

        Forwarding each row trimmed to its real span with ``attention_mask=None`` keeps FA4 on its
        dense kernel and restores RoPE positions ``[0, len)``; result is bit-identical. One forward
        per row keeps the FSDP/EP collective count in step. (FA4's varlen kernel JIT-compiles once
        per head-config and is then length-agnostic — measured on flash-attn-4 4.0.0b16/B300 — so
        the dense detour is motivated by the position/collective invariants above, not compile cost;
        the dense kernel itself is a separate one-time JIT.)
        """
        spans = dense_row_spans(attention_mask)
        # Every row's completion count in ONE D2H: read inside the loop this is another sync per row.
        if logits_to_keep > 0:
            n_comps = attention_mask[:, -logits_to_keep:].bool().sum(dim=1).tolist()
        else:
            n_comps = [0] * attention_mask.size(0)
        outputs = []
        for i, (lo, hi) in enumerate(spans):
            n_comp = int(n_comps[i])
            dense_ids = input_ids[i : i + 1, lo:hi]
            hidden = self._backbone_hidden_state(unwrapped_model, dense_ids, None, max(n_comp, 1))  # (1, >=1, H)
            # Every row must stay autograd-connected: a pruned row drops its FSDP/EP backward collectives.
            n_keep = max(n_comp, 1)
            out = hidden.new_zeros(1, logits_to_keep, hidden.size(-1))
            out[:, :n_keep] = hidden[:, -n_keep:]
            outputs.append(out)
        return torch.cat(outputs, dim=0)


class ChunkedGRPOLogprobsMixin(ChunkedLogprobsCore):
    """Overrides ``_get_per_token_logps_and_entropies`` with a chunked path that never materializes
    full logits.

    Gated by ``self._use_chunked_grpo_logprobs``; falls back to TRL's full-logits path when off or for
    multimodal inputs (chunked path is text-only). Place BEFORE the trainer's other bases so it wins
    the MRO.
    """

    # From TRL's own signature, not hand-listed: a release adding a modality kwarg must not slip
    # past this gate into the text-only chunked path.
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
        # TRL's loss forward passes batch_size=None (OOMs on env-GRPO); rows are independent, so bounding is exact.
        # Per mode, like TRL's own default: eval runs at the eval batch size, so lowering that for
        # memory reaches the chunked path too instead of silently keeping the train bound.
        if batch_size is None:
            batch_size = (
                self.args.per_device_train_batch_size
                if getattr(model, "training", True)
                else self.args.per_device_eval_batch_size
            )
        chunked = getattr(self, "_use_chunked_grpo_logprobs", False)
        is_multimodal = any(kwargs.get(k) is not None for k in self._multimodal_keys())
        # Mixed full (image) / chunked (text) paths across ranks diverge FSDP2's collectives and deadlock
        # backward, so any multimodal batch routes ALL ranks to the full forward; text-only models skip this.
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
        every dense ``*ForCausalLM``) ends its output prep in ``DTensor.to_local()`` — the local
        tensor handed straight back out of an autograd Function. Autograd forbids an in-place write
        to such a tensor and to every view of it, and TRL divides the sliced logits by the sampling
        temperature in place, so without this the first ``compute_loss`` of a ``tp_size > 1`` run
        raises ``Output 0 of SliceBackward0 is a view and is being modified inplace``. Cloning the
        head's output restores the write; it is bounded by ``logits_to_keep`` (the completion
        window), not the full sequence. Only TP prepares the head's output this way — the MoE path
        applies TP to attention alone and leaves the head replicated, and every other mode returns a
        plain tensor — so the clone is scoped to TP rather than paid on every run.
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
