"""GRPO's trainer-side log-prob scoring: the chunked hidden-state sweep, the full-logits memory guard, the KL reference
policy and the protected-token entropy mask."""

import inspect
import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.models.utils import _ForwardRedirection

from halo.distributed.context_parallel.sequence_chunks import cp_shift_against_full_labels
from halo.distributed.context_parallel.ulysses_model import find_ulysses_model
from halo.distributed.expert_parallel.deepep.capacity import bump_forward_generation
from halo.distributed.tensor_parallel.hf_plan import plan_shards_param
from halo.distributed.tensor_parallel.local_linears import runs_local_forward
from halo.kernels.head_logprobs import chunked_head_logprobs
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.attention.selection import effective_attn_implementation
from halo.models.forward_capture import capturing_forward_outputs
from halo.models.head_transform import HeadTransform, resolve_head_transform
from halo.models.modality import config_declares_multimodality
from halo.models.module_tree import embedding_module, find_module_path, hidden_state_backbone
from halo.models.moe.family_specs import attention_reads_row_positions
from halo.models.segment_markers import segment_markers_for
from halo.models.unwrapping import base_transformers_model
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import materialize_dtensor
from halo.runtime.ranks import is_global_main_process
from halo.trainers.grpo.packed_forward import build_packed_rows
from halo.trainers.training_args_overrides import uses_evaluation

logger = logging.getLogger(__name__)

# Bytes per logit the full-logits loss forward holds at its peak: an fp32 plane, or the bf16 logits
# plus the bf16 log-softmax TRL's selective_log_softmax saves for backward.
_FULL_LOGITS_BYTES_PER_LOGIT = 4
# Share of the device memory free at trainer init above which the full-logits plane warns. At init the
# optimizer state, gradients and activations the same step needs are not allocated yet.
_FULL_LOGITS_WARN_FRACTION = 0.5


@dataclass(frozen=True)
class LogitsWidth:
    tokens: int
    set_by: str


def full_logits_verdict(
    rows: int, rows_set_by: str, width: LogitsWidth, vocab: int, free_bytes: int
) -> tuple[bool, str] | None:
    """``(fatal, message)``: fatal when the plane cannot fit, a warning past :data:`_FULL_LOGITS_WARN_FRACTION`,
    ``None`` when it fits comfortably."""
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


class RowSpan(IntEnum):
    """The columns of a :func:`row_spans_on_device` table. :attr:`MALFORMED` closes the columns a stashed table keeps."""

    LO = 0
    HI = 1
    COMPLETION = 2
    MALFORMED = 3


STASHED_COLUMNS = slice(RowSpan.MALFORMED)


def row_spans_on_device(attention_mask: torch.Tensor, completion_width: int = 0) -> torch.Tensor:
    """``[rows, 4]`` int64 on the mask's device, no host sync, with the columns of :class:`RowSpan`: each row's
    real-token span (a 1-token span is widened by a neighbour so the next-token shift leaves a position), its count in
    the mask's last ``completion_width`` columns, and whether it is malformed (no real token, or a hole in them)."""
    real = attention_mask.detach().bool()
    width = real.size(1)
    columns = torch.arange(width, device=real.device)
    lo = torch.where(real, columns, width).amin(dim=1)
    hi = torch.where(real, columns + 1, 0).amax(dim=1)
    malformed = real.sum(dim=1) != hi - lo
    completion = real[:, -completion_width:].sum(dim=1) if completion_width > 0 else torch.zeros_like(lo)
    thin = hi - lo < 2
    widen_left = thin & (lo > 0)
    return torch.stack(
        [
            torch.where(widen_left, lo - 1, lo),
            torch.where(thin & ~widen_left, hi + 1, hi),
            completion,
            malformed.long(),
        ],
        dim=1,
    )


def validate_row_spans(spans: torch.Tensor) -> None:
    """Raise on the first malformed row of a host-side :func:`row_spans_on_device` table. A hole in the mask shifts
    every log-prob after it, as the dense path tail-aligns."""
    malformed = spans[:, RowSpan.MALFORMED].nonzero()
    if malformed.numel():
        raise ValueError(
            f"Row {int(malformed[0])} has an empty or non-contiguous attention mask; the per-row dense log-prob path "
            f"requires one contiguous real-token span per row."
        )


def host_row_spans(table: torch.Tensor) -> tuple[list[tuple[int, int]], list[int]]:
    """A host-side table's ``[lo, hi)`` span and completion count per row."""
    return (
        [(lo, hi) for lo, hi in table[:, RowSpan.LO : RowSpan.COMPLETION].tolist()],
        table[:, RowSpan.COMPLETION].tolist(),
    )


def dense_row_spans(
    attention_mask: torch.Tensor, completion_width: int = 0
) -> tuple[list[tuple[int, int]], list[int]]:
    """Real-token span ``[lo, hi)`` per row and each row's count in the mask's last ``completion_width`` columns."""
    # One D2H for the whole mask: a per-row read serializes CPU and GPU on every row, which defeats
    # CPU launch-ahead on a path that runs repeatedly per gradient microbatch.
    table = row_spans_on_device(attention_mask, completion_width).cpu()
    validate_row_spans(table)
    return host_row_spans(table)


def _completion_window(parts: Sequence[torch.Tensor], width: int) -> torch.Tensor:
    return torch.stack([F.pad(part, (0, 0) * (part.dim() - 1) + (0, width - part.size(0))) for part in parts])


def rows_forward_densely(model, batch_size: int) -> bool:
    """Mandatory under FA4 (exact RoPE positions and rank-uniform collective counts) and where attention reads row
    indices (:func:`attention_reads_row_positions`); at ``batch_size == 1`` a padded row otherwise costs its rank's
    widest prompt+completion in every layer."""
    config = getattr(model, "config", None)
    return (
        effective_attn_implementation(config) == "flash_attention_4"
        or batch_size == 1
        or attention_reads_row_positions(config)
    )


class LogprobScoringMixin:
    """Hosts provide ``temperature``, ``accelerator``, ``args``, ``model``, ``ref_model`` and
    ``_use_chunked_grpo_logprobs``."""

    # Set by a trainer whose train-mode rows reach the backbone as flattened micro-batches (a packed varlen call per
    # call of the scorer) instead of one call per row.
    _packs_microbatches = False
    # Armed by :meth:`_capturing_completion_logits`: where the scoring forward also leaves its logits.
    _completion_logits_sink: list[torch.Tensor] | None = None
    # Armed for one loss forward: the ``[rows, 3]`` host table of ``lo, hi, completion`` its masks were built with.
    _armed_row_spans: torch.Tensor | None = None
    # Rows whose armed span disagreed with the mask the forward read, summed on the device.
    _row_span_mismatches: torch.Tensor | None = None
    # Set by :meth:`_scoring_with_row_spans`: the host spans of the one mask the next scoring call reads.
    _given_row_spans: tuple[list[tuple[int, int]], list[int]] | None = None

    @contextmanager
    def _capturing_completion_logits(self) -> Iterator[list[torch.Tensor]]:
        """Yield a list the block's scoring forwards append ``(rows, logits_to_keep, V)`` logits to, pre-temperature
        and carrying the graph, so a host whose loss needs them need not forward the policy twice."""
        sink: list[torch.Tensor] = []
        self._completion_logits_sink = sink
        try:
            yield sink
        finally:
            self._completion_logits_sink = None

    @property
    def _trains_rows_densely(self) -> bool:
        """Whether the train-mode scoring forwards, the loss pass and its no-grad recompute, take the dense per-row
        path."""
        return self._use_chunked_grpo_logprobs and rows_forward_densely(
            self.accelerator.unwrap_model(self.model), self.args.per_device_train_batch_size
        )

    def _reject_unreproducible_chunked_head(self) -> None:
        """Refuse a family the chunked sweep cannot reproduce before the first rollout. Pure local computation, so
        every rank reaches the same verdict."""
        if self._use_chunked_grpo_logprobs:
            self._head_transform(self.accelerator.unwrap_model(self.model))

    def _verify_full_logits_fit(self, width: LogitsWidth | None) -> None:
        """COLLECTIVE when it runs. Refuse a full-logits loss forward that cannot fit, on every rank when one cannot;
        warn on a large share. A ``width`` of ``None`` means nothing bounds it."""
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
        if uses_evaluation(args) and args.per_device_eval_batch_size > rows:
            rows, rows_set_by = args.per_device_eval_batch_size, "per_device_eval_batch_size"
        free_bytes, _ = torch.cuda.mem_get_info(self.accelerator.device)
        verdict = full_logits_verdict(rows, rows_set_by, width, vocab, free_bytes)
        fatal = verdict is not None and verdict[0]
        if not rank_consensus(not fatal).all_ok:
            raise RuntimeError(
                verdict[1]
                if fatal
                else "Another rank's full-logits plane cannot fit in its free device memory "
                "(its error states the size); set use_chunked_grpo_logprobs: true."
            )
        if verdict is not None and self.accelerator.is_main_process:
            logger.warning(verdict[1])

    @contextmanager
    def _reference_policy(self) -> Iterator[nn.Module]:
        """The frozen ``ref_model`` when the trainer holds one, else the PEFT policy with its adapters disabled."""
        if self.ref_model is not None:
            yield self.ref_model
            return
        with self.accelerator.unwrap_model(self.model).disable_adapter():
            yield self.model

    @contextmanager
    def _scoring_with_row_spans(self, spans: tuple[list[tuple[int, int]], list[int]]) -> Iterator[None]:
        """The block's scoring forwards take ``spans`` (:func:`dense_row_spans` of the very mask they are handed) in
        place of reading it again, for a caller that already holds them."""
        self._given_row_spans = spans
        try:
            yield
        finally:
            self._given_row_spans = None

    def _chunked_completion_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False
    ):
        """Redirects through the wrapper so its pre-forward hooks fire (FSDP2 unshards root params, ``lm_head``
        included) before the sweep runs on the unwrapped module."""
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

    def _cp_chunked_logps(self, model, input_ids, attention_mask, labels):
        """Score CP-owned next-token targets without vocabulary-wide logits. ``labels`` describes the full right-padded
        row on every peer; log-probs at ignored positions have no training meaning."""
        unwrapped = self.accelerator.unwrap_model(model)
        wrapper = find_ulysses_model(unwrapped)
        if wrapper is None:
            raise TypeError("CP GRPO scoring requires a UlyssesModel")
        return self._chunked_forward_redirection(
            model, unwrapped, self._cp_chunked_logps_impl, wrapper, input_ids, attention_mask, labels
        )

    @property
    def _chunked_forward_redirection(self) -> _ForwardRedirection:
        """Reuse TRL's ``_forward_redirection`` if built (only when ``use_liger_kernel``), else make one."""
        fr = getattr(self, "_forward_redirection", None)
        if fr is None:
            fr = _ForwardRedirection()
            self._forward_redirection = fr
        return fr

    def _chunked_logps_impl(
        self, unwrapped_model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy
    ):
        batch_size = batch_size or input_ids.size(0)
        sweep = self._head_sweep(unwrapped_model)
        if self._packs_microbatches and self.model.training:
            return self._packed_logps(
                unwrapped_model, input_ids, attention_mask, logits_to_keep, compute_entropy, sweep
            )
        dense = rows_forward_densely(unwrapped_model, batch_size)
        spans, completions = self._row_spans_for(attention_mask, logits_to_keep) if dense else ([], [])
        all_logps, all_entropies = [], []
        for start in range(0, input_ids.size(0), batch_size):
            ids = input_ids[start : start + batch_size]
            mask = attention_mask[start : start + batch_size]
            completion_ids = ids[:, -logits_to_keep:]
            if not dense:
                hidden = self._backbone_hidden_state(unwrapped_model, ids, mask, logits_to_keep)  # (b, ltk, H)
                logps, entropies = sweep(hidden, completion_ids, compute_entropy)
                all_logps.append(logps)
                all_entropies.append(entropies)
                if self._completion_logits_sink is not None:
                    self._completion_logits_sink.append(self._head_logits(unwrapped_model, hidden))
                continue
            # Dense rows: one sweep over every row's real completion tokens, the window's zero tail masked
            # downstream, so the head pays the rows' lengths, not the padded width, and runs once.
            rows = slice(start, start + batch_size)
            hidden, widths = self._dense_last_hidden_state(
                unwrapped_model, ids, mask, logits_to_keep, spans=(spans[rows], completions[rows])
            )
            targets = torch.cat([row_ids[:width] for row_ids, width in zip(completion_ids, widths, strict=True)])
            logps, entropies = sweep(hidden, targets, compute_entropy)
            all_logps.append(_completion_window(logps.split(widths), logits_to_keep))
            all_entropies.append(
                _completion_window(entropies.split(widths), logits_to_keep) if compute_entropy else None
            )
            if self._completion_logits_sink is not None:
                window = _completion_window(hidden.split(widths), logits_to_keep)
                self._completion_logits_sink.append(self._head_logits(unwrapped_model, window))

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _packed_logps(self, unwrapped_model, input_ids, attention_mask, logits_to_keep, compute_entropy, sweep):
        """All rows handed in as one flattened varlen backbone call (their ``batch_size`` chunking ignored: the caller
        picked the micro-batch). Each row's positions restart at 0, so it scores as it does alone."""
        if self._completion_logits_sink is not None:
            raise RuntimeError("Packed micro-batches score no completion logits for a loss that needs them.")
        spans, n_comps = self._row_spans_for(attention_mask, logits_to_keep)
        widths = [max(n_comp, 1) for n_comp in n_comps]
        rows = build_packed_rows(input_ids, spans, widths, segment_markers_for(unwrapped_model.config))
        hidden = self._packed_hidden_state(unwrapped_model, rows)
        completion_ids = input_ids[:, -logits_to_keep:]
        targets = torch.cat([row_ids[:width] for row_ids, width in zip(completion_ids, widths, strict=True)])
        logps, entropies = sweep(hidden, targets, compute_entropy)
        return (
            _completion_window(logps.split(widths), logits_to_keep),
            _completion_window(entropies.split(widths), logits_to_keep) if compute_entropy else None,
        )

    def _packed_hidden_state(self, unwrapped_model, rows):
        """Hidden states at the rows' scored positions, from one backbone call under one EP capacity scope."""
        bump_forward_generation()
        backbone = hidden_state_backbone(unwrapped_model)
        hidden = backbone(
            input_ids=rows.input_ids,
            attention_mask=None,
            position_ids=rows.position_ids,
            use_cache=False,
            **rows.kwargs,
        ).last_hidden_state
        return hidden[0, rows.hidden_index]

    def _head_logits(self, unwrapped_model, hidden: torch.Tensor) -> torch.Tensor:
        """Called inside the redirected forward, where FSDP2 has gathered the head."""
        return self._head_transform(unwrapped_model).project(unwrapped_model.get_output_embeddings(), hidden)

    def _row_spans_for(self, attention_mask, logits_to_keep) -> tuple[list[tuple[int, int]], list[int]]:
        """:func:`dense_row_spans` of the whole batch, the spans a caller gave (:meth:`_scoring_with_row_spans`), or the
        armed table of a loss forward, whose agreement with the mask read here is tallied on the device and refused at
        the window's end (``RowSpanStashMixin``)."""
        given = self._given_row_spans
        if given is not None:
            if len(given[0]) != attention_mask.size(0):
                raise RuntimeError(
                    f"Row spans for {len(given[0])} rows were given to a scoring call over {attention_mask.size(0)} rows: "
                    f"they describe another mask."
                )
            return given
        armed = self._armed_row_spans
        if armed is None:
            return dense_row_spans(attention_mask, logits_to_keep)
        rows, width = attention_mask.shape
        if armed.device.type != "cpu":
            raise RuntimeError(
                f"The armed row spans sit on {armed.device}, not the host: a stage moved the batch to the device, "
                f"which would make the spans' read a sync. Keep ROW_SPANS_KEY on the CPU."
            )
        if armed.shape != (rows, RowSpan.MALFORMED) or (rows and int(armed[:, RowSpan.HI].max()) > width):
            raise RuntimeError(
                f"The armed row spans {tuple(armed.shape)} do not fit the {rows}-row, {width}-column batch the loss "
                f"forward scores: a stage reshaped the batch after its spans were built."
            )
        fresh = row_spans_on_device(attention_mask, logits_to_keep)[:, STASHED_COLUMNS]
        # Pinned: a non-blocking copy from pageable memory is staged by the host.
        expected = (armed.pin_memory() if fresh.is_cuda else armed).to(fresh.device, non_blocking=True)
        mismatched = (fresh != expected).any(dim=1).sum()
        self._row_span_mismatches = (
            mismatched if self._row_span_mismatches is None else self._row_span_mismatches + mismatched
        )
        return host_row_spans(armed)

    def _dense_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep, spans=None):
        """Each row's width is at least 1 so it stays autograd-connected (a pruned row drops its FSDP/EP backward
        collectives). Trimming with ``attention_mask=None`` keeps FA4 on its dense kernel and restores positions
        ``[0, len)``, so a row scores as it does alone. ``spans`` are the rows' :meth:`_row_spans_for` when the caller
        holds them."""
        spans, n_comps = spans if spans is not None else dense_row_spans(attention_mask, logits_to_keep)
        widths = [max(n_comp, 1) for n_comp in n_comps]
        rows = [
            self._backbone_hidden_state(unwrapped_model, input_ids[i : i + 1, lo:hi], None, width)[0, -width:]
            for i, ((lo, hi), width) in enumerate(zip(spans, widths, strict=True))
        ]
        return torch.cat(rows), widths

    def _backbone_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Open its own EP capacity scope: it enters the backbone directly, past the pre-hook that scopes the shared
        DeepEP arena, and each dense row carries its own length."""
        bump_forward_generation()
        return self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Forward the backbone, drop the final position (the causal next-token shift), keep the last
        ``logits_to_keep``."""
        backbone = hidden_state_backbone(unwrapped_model)
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return hidden[:, :-1, :][:, -logits_to_keep:, :]  # (B, logits_to_keep, H)

    def _cp_chunked_logps_impl(self, unwrapped_model, input_ids, attention_mask, labels):
        if labels.shape != input_ids.shape:
            raise ValueError("CP GRPO scoring needs labels aligned with the full input row")

        sweep = self._head_sweep(unwrapped_model)
        hidden = unwrapped_model.forward_hidden_states(input_ids=input_ids, attention_mask=attention_mask)
        shifted_hidden, shifted_labels = cp_shift_against_full_labels(
            hidden, labels, unwrapped_model.cp_config.cp_rank, unwrapped_model.cp_config.cp_size
        )
        logps, _ = sweep(shifted_hidden, shifted_labels)
        return logps, shifted_labels

    def _head_sweep(self, unwrapped_model) -> Callable[..., tuple[torch.Tensor, torch.Tensor | None]]:
        """``sweep(hidden, targets, compute_entropy=False) -> (logps, entropy)`` under the verified head transform."""
        lm_head = unwrapped_model.get_output_embeddings()
        self._assert_output_embeddings_unadapted(unwrapped_model, lm_head)
        weight = materialize_dtensor(lm_head.weight)
        bias = materialize_dtensor(getattr(lm_head, "bias", None))
        transform = self._head_transform(unwrapped_model)
        weight, bias = transform.kept_rows(weight, bias)

        def sweep(hidden, targets, compute_entropy=False):
            return chunked_head_logprobs(
                transform.scale_hidden(hidden),
                weight,
                targets,
                bias,
                self.temperature,
                logit_scale=transform.logit_scale,
                softcap=transform.softcap,
                compute_entropy=compute_entropy,
            )

        return sweep

    def _assert_output_embeddings_unadapted(self, unwrapped_model, lm_head) -> None:
        """``get_output_embeddings()`` resolves to the tuner wrapper or the base linear, and both hand the chunked
        matmul a delta-free weight. Checked once per run."""
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
        """Resolved once per model, since the verification builds a meta-device shell of the model's class."""
        base = base_transformers_model(unwrapped_model)
        transforms = getattr(self, "_head_transforms", None)
        if transforms is None:
            transforms = self._head_transforms = {}
        if id(base) not in transforms:
            transforms[id(base)] = resolve_head_transform(base)
        return transforms[id(base)]


def _tp_gathered_output_head(model) -> torch.nn.Module | None:
    """The output embedding when a TP plan shards it and transformers' DTensor forward computes it. Read off the plan:
    the MoE path applies TP to attention alone, and FSDP2 makes the head's weight a DTensor in every mode."""
    head = embedding_module(model, "output")
    plan = getattr(model, "_tp_plan", None)
    if head is None or not plan or runs_local_forward(head):
        return None
    name = find_module_path(model, head)
    return head if name is not None and plan_shards_param(f"{name}.weight", plan) else None


class ChunkedGRPOLogprobsMixin(LogprobScoringMixin):
    """TRL's full-logits scorer serves the gate off and multimodal inputs. Mixed in ahead of ``GRPOTrainer`` and behind
    ``ProtectedTokenEntropyMixin``, whose override reaches it through ``super()``."""

    # Whether this trainer's scored batches can carry images. A trainer that builds its batches text-only sets False
    # and its composite scoring forwards skip the cross-rank image consensus (one all-reduce and host read each).
    _SCORED_BATCHES_CARRY_IMAGES: ClassVar[bool] = True

    # Derived from TRL's full-logits scorer's signature, so a release adding a modality input does not slip past this
    # gate into the text-only chunked path.
    _MODALITY_KEYS = frozenset(inspect.signature(GRPOTrainer._full_logits_logps).parameters) - {
        "self",
        "model",
        "input_ids",
        "attention_mask",
        "logits_to_keep",
        "batch_size",
        "compute_entropy",
        "compute_aux_loss",
    }

    @profiling_decorator
    def _get_per_token_logps_and_entropies(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        batch_size=None,
        compute_entropy=False,
        compute_aux_loss=False,
        **kwargs,
    ):
        """``aux_loss`` is always ``None``: the trainer mixin hands TRL's GRPO a zero coefficient, the term being inert
        under a policy-gradient loss."""
        if compute_aux_loss:
            raise RuntimeError(
                "TRL asked the GRPO log-prob forward for the router aux loss, but the toolkit hands GRPO a zero "
                "router_aux_loss_coef and its scorers return none."
            )
        # TRL's loss forward passes batch_size=None; rows are independent, so bounding it is exact. The
        # bound follows the TRAINER's mode: a frozen reference model is always in eval, and a different bound there can
        # route the KL's two sides through different paths.
        if batch_size is None:
            batch_size = (
                self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size
            )
        chunked = self._use_chunked_grpo_logprobs
        is_multimodal = any(kwargs.get(k) is not None for k in self._MODALITY_KEYS)
        # Mixed full (image) / chunked (text) paths across ranks diverge FSDP2's collectives and deadlock
        # backward, so any multimodal batch routes all ranks to the full forward; text-only models skip this.
        if chunked and self._model_accepts_multimodal:
            if self._SCORED_BATCHES_CARRY_IMAGES:
                is_multimodal = rank_consensus(is_multimodal).any_ok
            elif is_multimodal:
                # A bug trap: this rank alone raises, so a peer waits until the collective timeout.
                raise RuntimeError(
                    f"{type(self).__name__} scores a batch with modality inputs, but its batches are built text-only "
                    f"and its scoring skips the cross-rank image check. Set _SCORED_BATCHES_CARRY_IMAGES where "
                    f"batches carry images."
                )
        if not chunked or is_multimodal:
            # TRL's own dispatch would send a use_liger_kernel run to its chunked head, which skips the
            # family's head transform and the EP capacity scope.
            with self._writable_logits(model), self._capturing_forward_logits(model, logits_to_keep):
                return self._full_logits_logps(
                    model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=batch_size,
                    compute_entropy=compute_entropy,
                    **kwargs,
                )
        logps, entropies = self._chunked_completion_logps(
            model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy
        )
        return logps, entropies, None

    @property
    def _model_accepts_multimodal(self) -> bool:
        cached = getattr(self, "_model_accepts_multimodal_cache", None)
        if cached is None:
            config = getattr(self.accelerator.unwrap_model(self.model), "config", None)
            cached = config is not None and config_declares_multimodality(config)
            self._model_accepts_multimodal_cache = cached
        return cached

    @contextmanager
    def _capturing_forward_logits(self, model, logits_to_keep: int) -> Iterator[None]:
        """While a sink is armed, append the completion logits of every forward of ``model`` in the block."""
        sink = self._completion_logits_sink
        if sink is None:
            yield
            return
        with capturing_forward_outputs(model, lambda output: output.logits[:, :-1][:, -logits_to_keep:]) as logits:
            yield
        sink.extend(logits)

    @contextmanager
    def _writable_logits(self, model):
        """HF's ``colwise_gather_output`` plan ends in ``DTensor.to_local()``, whose output autograd forbids writing in
        place, and TRL's full-logits path divides the logits in place under TP; clone it."""
        head = _tp_gathered_output_head(self.accelerator.unwrap_model(model))
        if head is None:
            yield
            return
        handle = head.register_forward_hook(lambda _module, _inputs, output: output.clone())
        try:
            yield
        finally:
            handle.remove()


class ProtectedTokenEntropyMixin:
    """Union special and added token positions into ``top_entropy_quantile``'s mask, which alone never reinforces the
    lowest-entropy tokens (role markers, tool delimiters, BOS/EOS). Mixed in ahead of ``ChunkedGRPOLogprobsMixin``,
    whose chunked path never calls ``super()``. Inert at ``top_entropy_quantile == 1.0``."""

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        # Stash completion ids for get_high_entropy_mask, called on the same micro-batch by _compute_loss.
        self._entropy_completion_ids = input_ids[:, input_ids.size(1) - logits_to_keep :]
        return super()._get_per_token_logps_and_entropies(
            model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
        )

    def get_high_entropy_mask(self, entropies: torch.Tensor, mask: torch.Tensor, threshold: float) -> torch.Tensor:
        high_entropy = super().get_high_entropy_mask(entropies, mask, threshold)

        protected_ids = self._protected_token_ids()
        if protected_ids is None:
            return high_entropy
        completion_ids = getattr(self, "_entropy_completion_ids", None)
        if completion_ids is None or completion_ids.shape != high_entropy.shape:
            raise RuntimeError(
                f"top_entropy_quantile protection lost its completion ids: the entropy mask is "
                f"{tuple(high_entropy.shape)} but the ids stashed by the last log-prob forward are "
                f"{None if completion_ids is None else tuple(completion_ids.shape)}, so the two come from "
                f"different micro-batches and the structural tokens would silently leave the trained set."
            )

        if protected_ids.device != completion_ids.device:
            protected_ids = protected_ids.to(completion_ids.device)
            self._protected_ids_cache = protected_ids
        is_protected = torch.isin(completion_ids, protected_ids)
        return (high_entropy | is_protected) & mask.bool()

    def _protected_token_ids(self) -> torch.Tensor | None:
        cached = getattr(self, "_protected_ids_cache", None)
        if cached is not None:
            # An empty cache means resolved with nothing to protect, which keeps the warning below to once per run.
            return cached if cached.numel() else None

        tokenizer = resolve_tokenizer(self.processing_class)
        ids: set[int] = set(tokenizer.all_special_ids or ())
        # Added tokens carry template scaffolding `all_special_ids` misses (ChatML/harmony, tool delimiters).
        added = getattr(tokenizer, "get_added_vocab", None)
        if callable(added):
            ids.update(added().values())
        self._protected_ids_cache = torch.tensor(sorted(ids), dtype=torch.long)
        if not ids:
            if is_global_main_process():
                logger.warning("No special/added tokens found on the tokenizer; entropy mask left unprotected.")
            return None

        if is_global_main_process():
            logger.info(f"Entropy mask protects {len(ids)} special/added tokens (always policy-trained).")
        return self._protected_ids_cache
