"""Last-stage loss adapters and the batch shaping they require.

``torch.distributed.pipelining`` calls ``loss_fn(output, target)`` on the last stage once per
microbatch, and sums nothing itself. Two properties decide whether an objective survives that:

1. **Microbatch-invariance.** The per-microbatch losses must SUM to the full-batch loss. Any
   objective normalized by a whole-batch denominator (token count, group weight) must be given that
   denominator explicitly — see ``num_items_in_batch`` on :meth:`PipelineRuntime.step`. Normalizing
   per microbatch silently changes the objective.
2. **Split-compatibility.** The microbatch split is ``tensor_split(dim=0)``. An objective that pairs
   examples (preference losses over a chosen/rejected concatenation) is only safe if paired examples
   land in the SAME microbatch — hence the interleaved layout below.

The collator-space shaping an adapter declares to satisfy those two properties lives here as well —
the interleaved pair layout, the completion-only labels, the fixed-shape padding and the
normalizers — so a loss and the batch form it demands cannot drift apart across modules.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.data.spans import LABEL_IGNORE_INDEX

# fp32 ELEMENTS per CE chunk: the fp32 upcast of a [tokens, V] plane is the last stage's memory peak,
# and chunking under a non-reentrant checkpoint bounds the held fp32 state to one chunk. Budgeted in
# elements, not token rows, because the plane is tokens×V — a fixed row count scales the held state
# with the vocabulary (at V=201k a 4096-row chunk is 3.3 GB). 128M elements = 512 MB fp32, i.e. 4096
# rows at V=32k. The fused path (:func:`fused_causal_lm_token_loss`) reads the same budget, where it
# bounds the head projection too, so no full plane is built.
_CE_CHUNK_ELEMENTS = 128 * 1024 * 1024


def _ce_chunk_rows(vocab_size: int) -> int:
    """Token rows whose fp32 [rows, V] plane fits the chunk budget; at least one row."""
    return max(1, _CE_CHUNK_ELEMENTS // max(vocab_size, 1))


def _shift_labels_left(labels: torch.Tensor) -> torch.Tensor:
    """Next-token targets: ``labels`` rolled one position left, the last position ignored.

    Shifting the LABELS rather than slicing the logits is what keeps the caller's ``reshape`` a view
    instead of a ``[mb, S, V]`` copy. The one home for the convention both causal-LM losses use, and
    the one :func:`loss_token_counts_per_row` must agree with.
    """
    shifted = torch.full_like(labels, LABEL_IGNORE_INDEX)
    shifted[:, :-1] = labels[:, 1:]
    return shifted


def _ce_sum_chunk(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(chunk_logits.float(), chunk_labels, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")


def _head_ce_sum_chunk(head: nn.Module, chunk_hidden: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    """Project one token chunk through the head and sum its fp32 CE — the unit the checkpoint recomputes."""
    return _ce_sum_chunk(head(chunk_hidden), chunk_labels)


def _chunked_token_results(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
    flat_labels: torch.Tensor,
    chunk_rows: int,
) -> Iterator[torch.Tensor]:
    """``fn(rows[chunk], flat_labels[chunk])`` per token chunk, each under a non-reentrant checkpoint.

    ``rows`` is the flattened logits plane (unfused) or the flattened hidden states (fused), sliced
    in lockstep with its labels. Checkpointing bounds the held fp32 state to one chunk; a single
    chunk needs none, so the short-sequence case keeps the plain call. The one home for the chunk
    budget and the checkpoint — callers add only their reduction over the results.
    """
    n_tokens = flat_labels.numel()
    if n_tokens <= chunk_rows:
        yield fn(rows, flat_labels)
        return
    for start in range(0, n_tokens, chunk_rows):
        end = start + chunk_rows
        yield checkpoint(fn, rows[start:end], flat_labels[start:end], use_reentrant=False)


def _chunked_token_sum(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    rows: torch.Tensor,
    flat_labels: torch.Tensor,
    chunk_rows: int,
) -> torch.Tensor:
    """``sum(fn(rows[chunk], flat_labels[chunk]))`` over token chunks, accumulated in fp32."""
    total = rows.new_zeros((), dtype=torch.float32)
    for value in _chunked_token_results(fn, rows, flat_labels, chunk_rows):
        total = total + value
    return total


def _chunked_ce_sum(flat_logits: torch.Tensor, flat_labels: torch.Tensor) -> torch.Tensor:
    """Summed fp32 cross-entropy over an already-flattened ``[tokens, V]`` plane, chunked."""
    return _chunked_token_sum(_ce_sum_chunk, flat_logits, flat_labels, _ce_chunk_rows(flat_logits.size(-1)))


def _logprob_chunk(chunk_logits: torch.Tensor, chunk_labels: torch.Tensor) -> torch.Tensor:
    """fp32 log-probs of ``chunk_labels`` under one ``[rows, V]`` chunk — a ``[rows]`` vector."""
    logps = torch.log_softmax(chunk_logits.float(), dim=-1)
    return logps.gather(-1, chunk_labels.unsqueeze(-1)).squeeze(-1)


def _chunked_token_logprobs(flat_logits: torch.Tensor, flat_labels: torch.Tensor) -> torch.Tensor:
    """Per-token fp32 log-probs over a flattened ``[tokens, V]`` plane, chunked like the CE path.

    The vector counterpart of :func:`_chunked_token_sum`: same chunking, but each chunk yields a
    ``[rows]`` slice rather than a scalar, so the results are concatenated instead of summed.
    Chunking is what keeps the two full fp32 planes (the ``.float()`` upcast and the ``log_softmax``
    output saved for backward) off the stage that carries the head — ~26 GB per microbatch at
    ``V=201088``, ``S=8192``, on top of the bf16 logits.
    """
    results = _chunked_token_results(_logprob_chunk, flat_logits, flat_labels, _ce_chunk_rows(flat_logits.size(-1)))
    return torch.cat(list(results))


@dataclass(frozen=True)
class PPLossAdapter:
    """A trainer's pipeline-loss contract, declared per class like ``_supports_pp``.

    The mixin reads this instead of branching on trainer types: the adapter says how the trainer's
    objective becomes a last-stage pipeline loss.

    Attributes:
        token_loss_fn: ``(logits, target) -> summed loss`` — MUST sum (the runtime divides every
            microbatch loss by the step normalizer). ``target`` is the labels tensor, or a
            ``{"labels": ..., **extras}`` dict when ``extra_target_keys`` is non-empty.
        paired_examples: interleaved ``[c0, r0, c1, r1, ...]`` batches; the runtime then validates
            that no pair is split across microbatches.
        batch_transform: collator-space transform applied to the prepared inputs before the step
            (interleaving pairs, deriving pool positions); ``None`` = identity.
        normalizer: ``(inputs) -> step normalizer`` for the runtime's division; ``None`` =
            the mixin's default DP-global loss-token count / dp (the SFT semantics).
        extra_target_keys: batch keys shipped to the last stage's loss as per-example side tensors
            (advantages, ref log-probs, pool positions), leading dim == batch rows.
        pad_spec: pad values for seq-shaped extra batch keys the fixed-shape collator must also pad.
        rows_per_example: rows the ``batch_transform`` emits per example (2 when interleaving
            pairs); the pipeline's frozen row count is ``per_device_train_batch_size`` × this.
        eval_normalizer: ``(inputs) -> denominator`` for the loss-only PP eval, called on
            the transformed batch BEFORE the inert row padding so pair/row-count denominators see
            the true batch; ``None`` = the causal-LM token count over this rank's replica batch
            (pad-invariant, since inert rows are all-ignore).
        pin_runtime_columns: ``True`` pins the runtime's batch contract (plus
            ``extra_target_keys``/``extra_signature_columns``) as HF's signature columns; ``False``
            restores the trainer's own ``_set_signature_columns_if_needed`` set — for trainers whose
            dataset columns feed a collator-space transform rather than the runtime contract.
        extra_signature_columns: dataset columns appended to the pinned set so HF's column pruning
            keeps them for the collator (only meaningful with ``pin_runtime_columns=True``).
        predictions_fn: ``(last_stage_output, transformed_inputs) -> predictions`` — reduces the
            last stage's raw output to what ``compute_metrics`` consumes, on the stage that holds it
            and before the chain broadcast. ``None`` = the raw output (the causal-LM contract, where
            the logits themselves are the prediction). A head whose useful output is far smaller than
            the raw plane (pooled sequence classification) declares the reduction here so the
            broadcast carries ``[B, num_labels]`` rather than ``[B, S, num_labels]``. This is the
            only pre-broadcast seam: HF applies ``preprocess_logits_for_metrics`` after the trainer's
            prediction step returns, so it cannot shrink the hop.
        eval_labels_fn: ``(transformed_inputs) -> labels`` paired with those predictions. ``None`` =
            ``inputs["labels"]``. Trainers whose ``batch_transform`` rewrites ``labels`` into a
            runtime-shaped plane recover the real targets here.
        metrics_fn: ``() -> {name: 0-dim tensor}`` — this rank's per-step metrics, drained once per
            pipeline step. Only the last stage runs ``token_loss_fn``, so only it holds real values;
            the mixin broadcasts them down the chain (one small collective per step) so every rank
            logs the same numbers. The KEY SET must be rank-uniform — derived from config, never
            from the batch — because the broadcast carries values only; the mixin pins the keys at
            setup and validates them across the chain there. ``None`` = the trainer logs nothing
            per step (the causal-LM contract, whose only per-step scalar is the loss itself).
        row_aligned_eval_outputs: ``False`` when ``predictions_fn``/``eval_labels_fn`` reduce the
            whole batch to a fixed-size summary rather than one entry per example: the fixed-shape
            padding the pipeline needs is then already folded away, so trimming the filler rows off
            it would cut into the summary itself. It governs the prediction and label legs
            TOGETHER — they are a pair that ``compute_metrics`` zips, and trimming one without the
            other yields mismatched lengths on every partial final eval batch.
    """

    token_loss_fn: Callable[[torch.Tensor, torch.Tensor | dict], torch.Tensor]
    paired_examples: bool = False
    batch_transform: Callable[[dict], dict] | None = None
    normalizer: Callable[[dict], torch.Tensor | float] | None = None
    extra_target_keys: tuple[str, ...] = ()
    pad_spec: Mapping[str, int] | None = None
    rows_per_example: int = 1
    eval_normalizer: Callable[[dict], torch.Tensor | float] | None = None
    pin_runtime_columns: bool = True
    extra_signature_columns: tuple[str, ...] = ()
    predictions_fn: Callable[[torch.Tensor, dict], torch.Tensor] | None = None
    eval_labels_fn: Callable[[dict], torch.Tensor] | None = None
    metrics_fn: Callable[[], dict[str, torch.Tensor]] | None = None
    row_aligned_eval_outputs: bool = True

    @property
    def supports_fused_head_loss(self) -> bool:
        """Whether the last stage may compute this objective INSIDE its forward, in token chunks.

        Only the causal-LM contract qualifies. Its loss is a per-token cross-entropy over the head
        output, which decomposes exactly over disjoint token chunks, so head and CE can run
        chunk-by-chunk and the ``[mb, S, V]`` logits plane never exists whole
        (:func:`fused_causal_lm_token_loss`). Every other adapter's loss reads the ASSEMBLED output —
        per-sequence log-probs, a pooled row, a comparison across paired rows — and must keep the
        logits. Derived from the declared contract rather than a trainer list, so a new adapter opts
        in only by declaring the same loss.
        """
        return self.token_loss_fn is causal_lm_token_loss and self.predictions_fn is None


def causal_lm_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal-LM cross-entropy SUMMED over tokens (the runtime applies the normalizer).

    Returning a sum rather than a mean is what lets the runtime normalize by the whole step's token
    count, so per-microbatch losses add up to the full-batch loss exactly.
    """
    return _chunked_ce_sum(logits.reshape(-1, logits.size(-1)), _shift_labels_left(labels).reshape(-1))


def fused_causal_lm_token_loss(head: nn.Module, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """:func:`causal_lm_token_loss` with the HEAD folded in, so no ``[mb, S, V]`` plane ever exists.

    The last pipeline stage calls this from inside its own forward
    (:attr:`~src.distributed.pipeline_parallel.stage.PipelineStageModule.fused_loss_fn`) instead of
    returning logits for the schedule's ``loss_fn`` to consume. Head projection and cross-entropy run
    together over one token chunk at a time under a non-reentrant checkpoint, so the live logits are
    ``[chunk, V]`` rather than ``[mb, S, V]`` — and the backward never materializes a full-plane
    gradient either, because each chunk's logits are recomputed and discarded in turn. At
    ``V=201088``, ``S=8192`` and one microbatch row that replaces 3.29 GB of bf16 logits plus 3.29 GB
    of logit gradient with one chunk's worth.

    The loss is the same value the unfused path computes: the summed CE decomposes exactly over
    disjoint token chunks, and slicing the head's input along the token axis leaves each row's
    projection unchanged. Activation gradients likewise.

    The HEAD WEIGHT's gradient is reassociated, though, and that is the one thing chunking changes.
    Unfused, it is a single GEMM reducing over every token with fp32 accumulation; fused, it is one
    partial GEMM per chunk summed in the parameter's grad dtype. In fp32 the difference is rounding
    noise (~7e-7 relative); in bf16 — the production dtype — it is ~1e-2 relative, roughly 3x the
    error bf16 already carries against fp64 on the unchunked reduction. Everything else is unchanged,
    so the trade is that head-weight precision against the logits plane. ``HALO_PP_FUSED_HEAD_LOSS=0``
    takes the unfused path. Only the last stage may call this — every other stage has no head.
    """
    vocab_size = getattr(head, "out_features", None)
    if vocab_size is None:
        raise TypeError(
            f"The fused last-stage loss sizes its token chunks from the head's output width, and "
            f"{type(head).__name__} exposes no `out_features`. Only a linear task head can be fused; "
            f"set HALO_PP_FUSED_HEAD_LOSS=0 to keep the unfused logits path."
        )
    return _chunked_token_sum(
        functools.partial(_head_ce_sum_chunk, head),
        hidden_states.reshape(-1, hidden_states.size(-1)),
        _shift_labels_left(labels).reshape(-1),
        _ce_chunk_rows(vocab_size),
    )


def completion_labels(input_ids: torch.Tensor, completion_mask: torch.Tensor) -> torch.Tensor:
    """Labels that score the completion only: token ids where ``completion_mask`` is set, ignore elsewhere.

    The label form every completion-masked batch transform (DPO, KTO, offline GRPO) feeds the
    pipeline, so the prompt carries no loss and the counts below see exactly the scored tokens.
    """
    return torch.where(completion_mask.bool(), input_ids, LABEL_IGNORE_INDEX)


def loss_token_counts_per_row(labels: torch.Tensor) -> torch.Tensor:
    """Per-row ``[rows]`` count of non-ignored next-token targets in ``labels``.

    The one home for the shift convention the normalizers must agree with: raw labels lose
    position 0 to the last stage's shift, so counting them unshifted mis-scales the loss by a
    token per row.
    """
    return (labels[:, 1:] != LABEL_IGNORE_INDEX).sum(dim=-1)


def loss_token_count(labels: torch.Tensor) -> torch.Tensor:
    """Number of non-ignored next-token targets in ``labels`` — the per-row counts summed."""
    return loss_token_counts_per_row(labels).sum()


def interleave_pairs(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Lay a preference batch out as ``[c0, r0, c1, r1, ...]`` along the batch dimension.

    Preference trainers concatenate as ``[chosen ⧺ rejected]`` and recover the halves with
    ``chunk(2, dim=0)``. Under a pipeline that layout is wrong: ``tensor_split`` would put the chosen
    examples in early microbatches and their rejected partners in later ones, so each microbatch's
    "halves" would compare unrelated examples. Interleaving keeps every pair inside one microbatch
    for any even microbatch size.
    """
    if chosen.shape != rejected.shape:
        raise ValueError(f"chosen {tuple(chosen.shape)} and rejected {tuple(rejected.shape)} must match")
    stacked = torch.stack((chosen, rejected), dim=1)
    return stacked.reshape(2 * chosen.size(0), *chosen.shape[1:])


def split_pairs(interleaved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_pairs` — ``(chosen, rejected)`` from the interleaved layout.

    This is the microbatch-safe replacement for ``chunk(2, dim=0)``: strided indexing recovers the
    pairs correctly whether it runs on the whole batch or on one microbatch of it.
    """
    if interleaved.size(0) % 2 != 0:
        raise ValueError(
            f"A paired batch must have an even leading dimension, got {interleaved.size(0)}. "
            f"Under a pipeline this means per_device_train_batch_size / pipeline_microbatches must "
            f"be even, so no pair is split across microbatches."
        )
    return interleaved[0::2], interleaved[1::2]


def token_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Shifted per-token log-probs of ``labels`` under ``logits`` plus the non-ignored mask.

    Returns ``([B, S-1]`` fp32 log-probs, ``[B, S-1]`` bool mask``)``. The log-probs are NOT
    pre-masked: ignored positions carry the (finite) log-prob of label 0, so callers that clamp or
    reweight per token do so before applying the mask — exactly the order the trainer losses need.
    """
    # Shift the LABELS, not the logits: slicing dim 1 leaves a non-contiguous view whose reshape
    # would copy the whole bf16 plane. Same convention as :func:`causal_lm_token_loss`, so the
    # flatten below stays a view and the chunked pass never materializes a full fp32 plane. The
    # last shifted position is ignore-only, so dropping it recovers the [B, S-1] contract.
    shifted = _shift_labels_left(labels)
    mask = shifted != LABEL_IGNORE_INDEX
    safe_labels = shifted.masked_fill(~mask, 0)
    flat = _chunked_token_logprobs(logits.reshape(-1, logits.size(-1)), safe_labels.reshape(-1))
    return flat.view(shifted.shape)[:, :-1], mask[:, :-1]


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence summed log-probability of ``labels`` under ``logits`` (ignore_index aware).

    The building block for preference objectives: they compare per-sequence log-probs, which the
    last stage can compute because PP keeps the whole sequence on it (unlike CP).
    """
    logps, mask = token_logprobs(logits, labels)
    return (logps * mask).sum(dim=-1)


def rows_with_labels(labels: torch.Tensor) -> torch.Tensor:
    """Bool ``[rows]`` row-validity mask — ``False`` for all-ignore rows.

    The loss-only PP eval row-pads a partial final batch with inert rows whose labels are entirely
    ``LABEL_IGNORE_INDEX``. Token-summed losses ignore those rows for free, but pair/row losses
    evaluate to a nonzero constant at zero log-probs (``-logsigmoid(0)``, ipo's ``(0 − 1/(2β))²``,
    ``σ(0)``), so they must multiply their per-pair/per-row terms by this mask before summing.
    """
    return (labels != LABEL_IGNORE_INDEX).any(dim=-1)


def row_count_normalizer(inputs: Mapping[str, torch.Tensor]) -> float:
    """Chain-local row count for unpaired per-row objectives (KTO).

    TRL's per-row losses are a per-replica MEAN over rows (DP then averages gradients — a mean of
    per-replica means), so no cross-rank reduction, and PP's frozen batch shapes make it a constant.
    """
    return float(inputs["input_ids"].size(0))


def pair_count_normalizer(inputs: Mapping[str, torch.Tensor]) -> float:
    """Chain-local pair count for interleaved preference batches (DPO, Bradley-Terry reward).

    :func:`row_count_normalizer` halved: the interleaved layout carries two rows per pair.
    """
    return row_count_normalizer(inputs) / 2


def pad_to_pipeline_length(
    tensors: dict[str, tuple[torch.Tensor, int | float]], max_length: int, hint: str
) -> dict[str, torch.Tensor]:
    """Right-pad each ``{key: (tensor, fill)}`` along the sequence axis to the pipeline's fixed shape.

    An over-length tensor is a hard error — truncating here would silently drop loss tokens after
    the P2P buffer shapes froze; ``hint`` names the trainer's own remedy.
    """
    out = {}
    for key, (tensor, fill) in tensors.items():
        if tensor.shape[-1] > max_length:
            raise ValueError(
                f"Collated '{key}' is {tensor.shape[-1]} tokens, over the pipeline's fixed shape "
                f"max_length={max_length}; truncating here would silently drop loss tokens. {hint}"
            )
        out[key] = F.pad(tensor, (0, max_length - tensor.shape[-1]), value=fill)
    return out
