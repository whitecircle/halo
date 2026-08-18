"""Last-stage loss adapters and the batch shaping they require.

``torch.distributed.pipelining`` calls ``loss_fn(output, target)`` on the last stage once per
microbatch and sums nothing itself, so an objective must satisfy two properties:

1. Microbatch-invariance: the per-microbatch losses must sum to the full-batch loss. An objective
   normalized by a whole-batch denominator (token count, group weight) must be given that
   denominator explicitly — see ``num_items_in_batch`` on :meth:`PipelineRuntime.step`; normalizing
   per microbatch silently changes the objective.
2. Split-compatibility: the microbatch split is ``tensor_split(dim=0)``, so an objective that pairs
   examples (preference losses over a chosen/rejected concatenation) is safe only when paired
   examples land in the same microbatch — hence the interleaved layout below.

The collator-space shaping an adapter needs for those properties lives here too: the interleaved
pair layout, the completion-only labels, the fixed-shape padding and the normalizers.
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

# fp32 elements per CE chunk. The fp32 upcast of a [tokens, V] plane is the last stage's memory peak;
# chunking under a non-reentrant checkpoint bounds the held fp32 state to one chunk. Budgeted in
# elements rather than token rows because the plane is tokens×V, so a fixed row count would scale the
# held state with the vocabulary. 128M elements = 512 MB fp32, i.e. 4096 rows at V=32k. The fused
# path (:func:`fused_causal_lm_token_loss`) uses the same budget to bound its head projection.
_CE_CHUNK_ELEMENTS = 128 * 1024 * 1024


def _ce_chunk_rows(vocab_size: int) -> int:
    """Token rows whose fp32 [rows, V] plane fits the chunk budget; at least one row."""
    return max(1, _CE_CHUNK_ELEMENTS // max(vocab_size, 1))


def _shift_labels_left(labels: torch.Tensor) -> torch.Tensor:
    """Next-token targets: ``labels`` rolled one position left, the last position ignored.

    Shifting the labels rather than slicing the logits keeps the caller's ``reshape`` a view instead
    of a ``[mb, S, V]`` copy. Both causal-LM losses and :func:`loss_token_counts_per_row` follow this
    convention.
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
    chunk needs none, so the short-sequence case keeps the plain call. Callers add their own
    reduction over the results.
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

    The vector counterpart of :func:`_chunked_token_sum`: each chunk yields a ``[rows]`` slice rather
    than a scalar, so the results are concatenated instead of summed. Chunking keeps the two full
    fp32 planes (the ``.float()`` upcast and the ``log_softmax`` output saved for backward) off the
    stage carrying the head — ~26 GB per microbatch at ``V=201088``, ``S=8192``.
    """
    results = _chunked_token_results(_logprob_chunk, flat_logits, flat_labels, _ce_chunk_rows(flat_logits.size(-1)))
    return torch.cat(list(results))


@dataclass(frozen=True)
class PPLossAdapter:
    """A trainer's pipeline-loss contract, declared per class like ``_supports_pp``.

    The mixin reads this instead of branching on trainer types: the adapter states how the trainer's
    objective becomes a last-stage pipeline loss.

    Attributes:
        token_loss_fn: ``(logits, target) -> summed loss``; it must sum, since the runtime divides
            every microbatch loss by the step normalizer. ``target`` is the labels tensor, or a
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
        eval_normalizer: ``(inputs) -> denominator`` for the loss-only PP eval, called on the
            transformed batch before the inert row padding so pair/row-count denominators see the
            true batch; ``None`` = the causal-LM token count over this rank's replica batch
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
            the logits are the prediction). A head whose useful output is far smaller than the raw
            plane (pooled sequence classification) declares the reduction here so the broadcast
            carries ``[B, num_labels]`` rather than ``[B, S, num_labels]``. This is the only
            pre-broadcast seam: HF applies ``preprocess_logits_for_metrics`` after the trainer's
            prediction step returns, so it cannot shrink the hop.
        eval_labels_fn: ``(transformed_inputs) -> labels`` paired with those predictions. ``None`` =
            ``inputs["labels"]``. Trainers whose ``batch_transform`` rewrites ``labels`` into a
            runtime-shaped plane recover the real targets here.
        metrics_fn: ``() -> {name: 0-dim tensor}`` — this rank's per-step metrics, drained once per
            pipeline step. Only the last stage runs ``token_loss_fn``, so only it holds real values;
            the mixin broadcasts them down the chain (one small collective per step) so every rank
            logs the same numbers. The key set must be rank-uniform — derived from config, never
            from the batch — because the broadcast carries values only; the mixin pins the keys at
            setup and validates them across the chain there. ``None`` = the trainer logs nothing
            per step (the causal-LM contract, whose only per-step scalar is the loss).
        row_aligned_eval_outputs: ``False`` when ``predictions_fn``/``eval_labels_fn`` reduce the
            whole batch to a fixed-size summary rather than one entry per example: the fixed-shape
            padding is then already folded away, so trimming the filler rows would cut into the
            summary. It governs the prediction and label legs together — ``compute_metrics`` zips
            them, and trimming one without the other yields mismatched lengths on a partial final
            eval batch.
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
        """Whether the last stage may compute this objective inside its forward, in token chunks.

        Only the causal-LM contract qualifies: its per-token cross-entropy over the head output
        decomposes exactly over disjoint token chunks, so head and CE can run chunk-by-chunk and the
        ``[mb, S, V]`` logits plane is never built whole (:func:`fused_causal_lm_token_loss`). Every
        other adapter's loss reads the assembled output — per-sequence log-probs, a pooled row, a
        comparison across paired rows — and needs the logits. Derived from the declared contract
        rather than a trainer list, so a new adapter opts in by declaring the same loss.
        """
        return self.token_loss_fn is causal_lm_token_loss and self.predictions_fn is None


def causal_lm_token_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal-LM cross-entropy summed over tokens (the runtime applies the normalizer).

    Returning a sum rather than a mean lets the runtime normalize by the whole step's token count,
    so per-microbatch losses add up to the full-batch loss exactly.
    """
    return _chunked_ce_sum(logits.reshape(-1, logits.size(-1)), _shift_labels_left(labels).reshape(-1))


def fused_causal_lm_token_loss(head: nn.Module, hidden_states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """:func:`causal_lm_token_loss` with the head folded in, so no ``[mb, S, V]`` plane is built.

    The last pipeline stage calls this from inside its own forward
    (:attr:`~src.distributed.pipeline_parallel.stage.PipelineStageModule.fused_loss_fn`) instead of
    returning logits for the schedule's ``loss_fn``. Head projection and cross-entropy run together
    over one token chunk at a time under a non-reentrant checkpoint, so the live logits are
    ``[chunk, V]``; the backward materializes no full-plane gradient either, since each chunk's
    logits are recomputed and discarded in turn. At ``V=201088``, ``S=8192`` and one microbatch row
    that replaces 3.29 GB of bf16 logits plus 3.29 GB of logit gradient with one chunk's worth.

    The loss value matches the unfused path: the summed CE decomposes exactly over disjoint token
    chunks, and slicing the head's input along the token axis leaves each row's projection unchanged.
    Activation gradients likewise. The head weight's gradient is reassociated, which is the one thing
    chunking changes: unfused it is a single GEMM reducing over every token with fp32 accumulation,
    fused it is one partial GEMM per chunk summed in the parameter's grad dtype. In fp32 the
    difference is rounding noise (~7e-7 relative); in bf16, the production dtype, it is ~1e-2
    relative, roughly 3x the error bf16 already carries against fp64 on the unchunked reduction.
    ``HALO_PP_FUSED_HEAD_LOSS=0`` takes the unfused path. Only the last stage has a head, so only it
    may call this.
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

    Matches the last stage's label shift, which the normalizers depend on: raw labels lose position 0
    to that shift, so counting them unshifted mis-scales the loss by a token per row.
    """
    return (labels[:, 1:] != LABEL_IGNORE_INDEX).sum(dim=-1)


def loss_token_count(labels: torch.Tensor) -> torch.Tensor:
    """Number of non-ignored next-token targets in ``labels`` — the per-row counts summed."""
    return loss_token_counts_per_row(labels).sum()


def interleave_pairs(chosen: torch.Tensor, rejected: torch.Tensor) -> torch.Tensor:
    """Lay a preference batch out as ``[c0, r0, c1, r1, ...]`` along the batch dimension.

    Preference trainers concatenate as ``[chosen ⧺ rejected]`` and recover the halves with
    ``chunk(2, dim=0)``. Under a pipeline ``tensor_split`` would put the chosen examples in early
    microbatches and their rejected partners in later ones, so each microbatch's "halves" would
    compare unrelated examples. Interleaving keeps every pair inside one microbatch for any even
    microbatch size.
    """
    if chosen.shape != rejected.shape:
        raise ValueError(f"chosen {tuple(chosen.shape)} and rejected {tuple(rejected.shape)} must match")
    stacked = torch.stack((chosen, rejected), dim=1)
    return stacked.reshape(2 * chosen.size(0), *chosen.shape[1:])


def split_pairs(interleaved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`interleave_pairs` — ``(chosen, rejected)`` from the interleaved layout.

    The microbatch-safe replacement for ``chunk(2, dim=0)``: strided indexing recovers the pairs
    whether it runs on the whole batch or on one microbatch of it.
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

    Returns ``([B, S-1]`` fp32 log-probs, ``[B, S-1]`` bool mask``)``. The log-probs are not
    pre-masked: ignored positions carry the (finite) log-prob of label 0, so callers that clamp or
    reweight per token do so before applying the mask, which is the order the trainer losses need.
    """
    # Shift the labels, not the logits: slicing dim 1 leaves a non-contiguous view whose reshape
    # would copy the whole bf16 plane, so the flatten below stays a view. The last shifted position
    # is ignore-only, so dropping it recovers the [B, S-1] contract.
    shifted = _shift_labels_left(labels)
    mask = shifted != LABEL_IGNORE_INDEX
    safe_labels = shifted.masked_fill(~mask, 0)
    flat = _chunked_token_logprobs(logits.reshape(-1, logits.size(-1)), safe_labels.reshape(-1))
    return flat.view(shifted.shape)[:, :-1], mask[:, :-1]


def sequence_logprobs(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Per-sequence summed log-probability of ``labels`` under ``logits`` (ignore_index aware).

    The building block for preference objectives, which compare per-sequence log-probs; the last
    stage can compute them because PP keeps the whole sequence on it (unlike CP).
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

    TRL's per-row losses are a per-replica mean over rows (DP then averages gradients), so no
    cross-rank reduction is needed and PP's frozen batch shapes make this a constant.
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

    An over-length tensor raises: truncating here would drop loss tokens after the P2P buffer shapes
    froze. ``hint`` names the trainer's own remedy.
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
