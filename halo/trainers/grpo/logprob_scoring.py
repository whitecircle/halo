"""GRPO's trainer-side log-prob scoring: the chunked hidden-state sweep, the full-logits memory guard, the
KL reference policy and the protected-token entropy mask.

The chunked sweep scores the backbone's hidden states through :func:`halo.kernels.head_logprobs.chunked_head_logprobs`
under the family's verified head transform; TRL's ``_compute_loss`` runs unchanged on the ``(B, T)`` result.
"""

import inspect
import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass

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
from halo.kernels.head_logprobs import chunked_head_logprobs
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.attention.selection import effective_attn_implementation
from halo.models.head_transform import HeadTransform, resolve_head_transform
from halo.models.modality import config_declares_multimodality
from halo.models.module_tree import embedding_module, find_module_path
from halo.models.unwrapping import base_transformers_model
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import materialize_dtensor
from halo.runtime.ranks import is_global_main_process
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
    """The completion logits row a loss forward carries, in tokens, and the setting that bounds it."""

    tokens: int
    set_by: str


def full_logits_verdict(
    rows: int, rows_set_by: str, width: LogitsWidth, vocab: int, free_bytes: int
) -> tuple[bool, str] | None:
    """``(fatal, message)`` for a full-logits plane of ``rows × width × vocab`` against ``free_bytes``: fatal
    when it cannot fit, a warning past :data:`_FULL_LOGITS_WARN_FRACTION` of it, ``None`` when it fits
    comfortably. ``rows_set_by`` names the batch-size setting behind ``rows``."""
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
    """Real-token span ``[lo, hi)`` per row of a padded batch, as the dense per-row path forwards it.

    Every reader of the per-row widths derives them here, the <2-token widening included (a 1-token
    span is widened by a neighbour so the next-token shift leaves at least one position).
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


def rows_forward_densely(model, batch_size: int) -> bool:
    """Whether the chunked forward trims each row to its real span, one backbone call per row.

    Mandatory under FA4 (exact RoPE positions and rank-uniform collective counts). Taken on every
    attention path when rows are forwarded singly anyway (``batch_size == 1``): a padded row otherwise
    costs its rank's widest prompt+completion in every layer, and the head sweep its padded width.
    """
    return effective_attn_implementation(getattr(model, "config", None)) == "flash_attention_4" or batch_size == 1


class LogprobScoringMixin:
    """Trainer-agnostic GRPO log-prob scoring: the chunked sweep (batched, per-row dense, CP-sharded), the
    full-logits memory guard and the KL reference policy. Hosts provide ``temperature``, ``accelerator``,
    ``args``, ``model``, ``ref_model`` and ``_use_chunked_grpo_logprobs``."""

    def _resolve_chunked_head_transform(self) -> None:
        """Verify the policy's head path at construction when the chunked sweep will score it, so a
        family it cannot reproduce is refused before the first rollout. Pure local computation on the
        class and config: every rank reaches the same verdict."""
        if self._use_chunked_grpo_logprobs:
            self._head_transform(self.accelerator.unwrap_model(self.model))

    def _verify_full_logits_fit(self, width: LogitsWidth | None) -> None:
        """COLLECTIVE when it runs: refuse a full-logits loss forward that cannot fit on this device, on
        every rank when one cannot; warn when it takes a large share. ``width`` is the logits row the
        loss forward carries (``None``: nothing bounds it, nothing is checked).

        The rows are the chunk one loss forward materializes: ``per_device_train_batch_size``, or
        ``per_device_eval_batch_size`` when evaluation runs and it is larger. A floor on the step's peak.
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
        """Yield the model that scores the KL reference: the frozen ``ref_model`` when the trainer holds
        one, else the PEFT policy itself, with its adapters disabled for the duration."""
        if self.ref_model is not None:
            yield self.ref_model
            return
        with self.accelerator.unwrap_model(self.model).disable_adapter():
            yield self.model

    def _chunked_completion_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None, compute_entropy=False
    ):
        """Chunked ``(logps, entropies)`` for a possibly-wrapped model. Redirects through the wrapper so
        its pre-forward hooks fire (FSDP2 unshards root params, ``lm_head`` included) before the sweep
        runs on the unwrapped module."""
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
        """Score CP-owned next-token targets without constructing vocabulary-wide logits.

        ``labels`` describes the full right-padded row on every CP peer. The returned local labels
        mark supervised positions; log-probs at ignored positions have no training meaning.
        """
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
        dense = rows_forward_densely(unwrapped_model, batch_size)
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
                continue
            # Dense rows: the sweep covers each row's real completion tokens only and the window's zero
            # tail is masked downstream, so the head pays the row's length, not the padded width.
            hidden, n_comps = self._dense_last_hidden_state(unwrapped_model, ids, mask, logits_to_keep)
            for row, n_comp in enumerate(n_comps):
                width = max(n_comp, 1)
                logps, entropies = sweep(
                    hidden[row : row + 1, :width], completion_ids[row : row + 1, :width], compute_entropy
                )
                window = (0, logits_to_keep - logps.size(1))
                all_logps.append(F.pad(logps, window))
                all_entropies.append(F.pad(entropies, window) if compute_entropy else None)

        logps = torch.cat(all_logps, dim=0)
        entropies = torch.cat(all_entropies, dim=0) if compute_entropy else None
        return logps, entropies

    def _dense_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Per-row, unpadded backbone forward -> completion hidden ``(rows, logits_to_keep, H)`` plus each
        row's real completion-token count.

        Each row trimmed to its real span with ``attention_mask=None`` keeps FA4 on its dense kernel and
        restores RoPE positions ``[0, len)``; every supported family's attention is relative, so the
        log-probs match the padded forward, and one forward per row keeps the FSDP/EP collective count in step.
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

    def _backbone_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """One sub-forward, opening its own EP capacity scope first: it enters the backbone directly, past
        the pre-hook that scopes the shared DeepEP arena, and each dense row carries its own length.
        Rank-uniform: the sub-forward count follows the per-rank batch shape."""
        bump_forward_generation()
        return self._get_last_hidden_state(unwrapped_model, input_ids, attention_mask, logits_to_keep)

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Backbone hidden states for the last ``logits_to_keep`` completion positions (text-only):
        forward the backbone (no ``lm_head``), drop the final position (the causal next-token shift),
        keep the last ``logits_to_keep``."""
        backbone = base_transformers_model(unwrapped_model).base_model
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
        """``sweep(hidden, targets, compute_entropy=False) -> (logps, entropy)`` through the output
        embedding, materialized off any DTensor, under the verified head transform."""
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


def _tp_gathered_output_head(model) -> torch.nn.Module | None:
    """The output embedding when a TP plan shards it, else ``None``.

    Read off the plan transformers applied rather than off ``is_tp_mode``: the MoE path applies TP to
    attention alone and leaves the head whole, and FSDP2 makes the head's weight a DTensor in every
    mode, so neither the axis flag nor the parameter type separates a gathered head from a plain one.
    """
    head = embedding_module(model, "output")
    plan = getattr(model, "_tp_plan", None)
    if head is None or not plan:
        return None
    name = find_module_path(model, head)
    return head if name is not None and plan_shards_param(f"{name}.weight", plan) else None


class ChunkedGRPOLogprobsMixin(LogprobScoringMixin):
    """Overrides TRL's ``_get_per_token_logps_and_entropies`` with the chunked sweep, gated by
    ``self._use_chunked_grpo_logprobs``; TRL's full-logits scorer serves it off and for multimodal inputs
    (the sweep is text-only). Mixed in ahead of ``GRPOTrainer`` and behind ``ProtectedTokenEntropyMixin``,
    whose override reaches it through ``super()``.
    """

    # The modality inputs are the rest of TRL's full-logits scorer's parameters: derived from its
    # signature rather than hand-listed, so a release adding one does not slip past this gate into the
    # text-only chunked path.
    _CHUNK_TEXT_KEYS = frozenset(
        (
            "self",
            "model",
            "input_ids",
            "attention_mask",
            "logits_to_keep",
            "batch_size",
            "compute_entropy",
            "compute_aux_loss",
        )
    )

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
        """``(logps, entropies, aux_loss)``, with ``aux_loss`` always ``None``: the trainer mixin hands TRL's
        GRPO a zero router aux-loss coefficient, the term being inert under a policy-gradient loss."""
        if compute_aux_loss:
            raise RuntimeError(
                "TRL asked the GRPO log-prob forward for the router aux loss, but the toolkit hands GRPO a zero "
                "router_aux_loss_coef and its scorers return none."
            )
        # TRL's loss forward passes batch_size=None; rows are independent, so bounding it is exact. The
        # bound follows the TRAINER's mode, not the passed model's: a frozen reference model is always
        # in eval, and a different bound there can route the KL's two sides through different paths.
        if batch_size is None:
            batch_size = (
                self.args.per_device_train_batch_size if self.model.training else self.args.per_device_eval_batch_size
            )
        chunked = self._use_chunked_grpo_logprobs
        modality_keys = inspect.signature(GRPOTrainer._full_logits_logps).parameters.keys() - self._CHUNK_TEXT_KEYS
        is_multimodal = any(kwargs.get(k) is not None for k in modality_keys)
        # Mixed full (image) / chunked (text) paths across ranks diverge FSDP2's collectives and deadlock
        # backward, so any multimodal batch routes all ranks to the full forward; text-only models skip this.
        if chunked and self._model_accepts_multimodal:
            is_multimodal = rank_consensus(is_multimodal).any_ok
        if not chunked or is_multimodal:
            # TRL's own dispatch would send a use_liger_kernel run to its chunked head, which skips the
            # family's head transform and the EP capacity scope.
            with self._writable_logits(model):
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
        """Whether the model can receive vision inputs at all (cached: config-static per run)."""
        cached = getattr(self, "_model_accepts_multimodal_cache", None)
        if cached is None:
            config = getattr(self.accelerator.unwrap_model(self.model), "config", None)
            cached = config is not None and config_declares_multimodality(config)
            self._model_accepts_multimodal_cache = cached
        return cached

    @contextmanager
    def _writable_logits(self, model):
        """Let TRL's full-logits path divide the logits in place under TP by cloning a gathered head's output.

        A gathered ``lm_head`` plan (HF's ``colwise_gather_output``) ends in ``DTensor.to_local()``, whose
        output autograd forbids writing in place, and TRL divides the sliced logits by the temperature in
        place (``Output 0 of SliceBackward0 is a view and is being modified inplace``).
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


class ProtectedTokenEntropyMixin:
    """Union the tokenizer's special and added token positions into ``top_entropy_quantile``'s mask, which
    alone never reinforces the lowest-entropy tokens (role markers, tool delimiters, BOS/EOS). Mixed in ahead
    of ``ChunkedGRPOLogprobsMixin``, whose chunked path returns without calling ``super()``: behind it, the
    completion-id stash would never run. Inert at ``top_entropy_quantile == 1.0``."""

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
        """Special/added token ids for this tokenizer, cached. ``None`` when the tokenizer exposes none."""
        cached = getattr(self, "_protected_ids_cache", None)
        if cached is not None:
            # An empty cache means "resolved, nothing to protect"; caching it keeps the warning below
            # to once per run rather than once per micro-batch.
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
