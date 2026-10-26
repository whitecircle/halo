"""Distributed SFT trainer (EP/CP/TP/ETP) — DistributedTrainerMixin + TRL SFTTrainer."""

import inspect
from functools import cached_property

import torch
from accelerate.logging import get_logger
from transformers import Trainer
from trl import SFTTrainer
from trl.trainer.utils import entropy_from_logits

from src.distributed.context_parallel.sequence_chunks import cp_chunk_bounds, cp_shift_against_full_labels
from src.kernels.logprobs import LABEL_IGNORE_INDEX
from src.model_loading.tokenizer_setup import resolve_tokenizer
from src.models.attention.tree_plan import TREE_ATTENTION_KEY
from src.models.moe.balancing import expert_load_token_weights
from src.models.moe.config_fields import require_router_aux_loss_coef
from src.models.unwrapping import base_transformers_model
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.validation import ctor_config, ctor_positions, ctor_value

logger = get_logger(__name__)

# TRL SFTTrainer positional slots, for ctor params arriving via *args — derived from the installed signature.
_CTOR_POSITIONS = ctor_positions(SFTTrainer, "model", "args", "data_collator")

# CP metric row deferred to log time: every column is a sum, so one reduce per log gives the same
# totals as one reduce per micro-batch. Fixed width, so the collective's shape does not depend on
# what a rank accumulated.
_CP_METRIC_COLUMNS = (
    "microbatches",
    "correct_tokens",
    "loss_tokens",
    "entropy_sum",
    "entropy_tokens",
    "attended_tokens",
    "aux_loss_sum",
    "aux_loss_count",
)
# fp64: the token counts stay exact through the reduce (integers well under 2**53) while the
# entropy/aux sums keep more precision than the bf16 logits they were computed from.
_CP_METRIC_DTYPE = torch.float64


class DistributedSFTTrainer(DistributedTrainerMixin, SFTTrainer):
    """Distributed SFT trainer supporting EP, CP, and/or TP."""

    SUPPORTS_CP = True
    # Canonical last-stage loss; the PP-incompatible knobs are rejected in _maybe_prepare_pipeline_model.
    PP_REFUSAL = None
    SUPPORTS_LOWP_PRECISION = True
    # Forwards with ``labels``, so ``*ForCausalLM.forward`` folds the router aux loss into outputs.loss.
    _consumes_router_aux_loss = True
    # False lets loss-only eval take Liger's fused-CE ``skip_logits`` path; a subclass whose loss reads
    # ``outputs.logits`` must flip it (lce_forward returns None there, and raises without labels).
    _loss_reads_logits = False

    @property
    def _loss_is_own_mean(self) -> bool:
        """CP only. CP chunks the same sequence and FSDP averaging is already correct, so HF's
        num_items_in_batch scaling would inflate loss/grad_norm cp_size×; on every other axis TRL's
        token-scaled loss applies."""
        return self.is_cp_mode

    def __init__(self, *args, **kwargs):
        self._score_through_the_model_forward(ctor_config(args, kwargs, _CTOR_POSITIONS))
        kwargs = self._init_distributed_config(kwargs, ctor_args=args, ctor_positions=_CTOR_POSITIONS)
        collator = ctor_value(args, kwargs, "data_collator", _CTOR_POSITIONS)
        self._reject_cp_incompatible_collator(collator)
        model = ctor_value(args, kwargs, "model", _CTOR_POSITIONS)
        self._reject_compressed_kv_collator(collator, getattr(model, "config", None))
        super().__init__(*args, **kwargs)
        self._setup_distributed_modes()

        # Per-mode CP metric sums awaiting their once-per-log reduce (:meth:`_drain_cp_metrics`).
        self._cp_metric_accum: dict[str, torch.Tensor | None] = {}

    @staticmethod
    def _score_through_the_model_forward(training_args) -> None:
        """Replace TRL's ``chunked_nll`` loss type with ``nll``.

        TRL defaults an unset ``loss_type`` to ``chunked_nll`` whenever ``use_liger_kernel`` is off, and that
        type replaces the model's forward with one that projects ``lm_head`` itself: it skips the family's head
        transform, the router aux loss the forward folds in, and the logits a subclass's loss reads.
        """
        if training_args is None or getattr(training_args, "loss_type", None) != "chunked_nll":
            return
        logger.warning(
            "Setting loss_type to 'nll': TRL's 'chunked_nll' replaces the model's forward with its own lm_head "
            "projection, which skips the family's head transform and the router aux loss."
        )
        training_args.loss_type = "nll"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Validate that inputs carry training signal; under CP reconcile local-chunk logits with
        full-sequence labels (metrics are computed on the local chunk)."""
        self._validate_inputs(inputs)

        if self.is_cp_mode:
            full_labels = inputs.get("labels")
            full_attention_mask = inputs.get("attention_mask")

            # CP requires seq_len divisible by cp_size.
            if full_labels is not None:
                seq_len = full_labels.shape[1]
                if seq_len % self.cp_size != 0:
                    pad_len = self.cp_size - (seq_len % self.cp_size)
                    pad_token_id = resolve_tokenizer(self.processing_class).pad_token_id
                    if pad_token_id is None:
                        raise ValueError(
                            "Context parallelism pads every sequence to a multiple of "
                            "context_parallel_size and needs the tokenizer's pad_token_id; the "
                            "processing_class passed to the trainer has none. Set one (commonly the "
                            "EOS token) rather than padding with vocabulary token 0."
                        )
                    fills = {
                        "input_ids": pad_token_id,
                        "labels": LABEL_IGNORE_INDEX,
                        "attention_mask": 0,
                        "position_ids": 0,
                    }
                    for key, fill in fills.items():
                        if key in inputs:
                            inputs[key] = torch.nn.functional.pad(inputs[key], (0, pad_len), value=fill)
                    full_labels = inputs.get("labels")
                    full_attention_mask = inputs.get("attention_mask")

            inputs["use_cache"] = False
            # num_items_in_batch intentionally not passed (per-token loss scaling is wrong under CP).
            (loss, outputs) = Trainer.compute_loss(
                self,
                model,
                inputs,
                return_outputs=True,
            )

            self._compute_cp_metrics(outputs, full_labels, full_attention_mask)

            return (loss, outputs) if return_outputs else loss

        # TRL's gather().sum() double-counts attended tokens under TP/ETP (identical data per rank).
        prev_tokens = self._total_train_tokens

        tree_plan = inputs.get(TREE_ATTENTION_KEY)
        with expert_load_token_weights(None if tree_plan is None else tree_plan["leaf_count"]):
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )
        loss = self._weigh_router_aux_loss(model, loss, outputs, num_items_in_batch)
        result = (loss, outputs) if return_outputs else loss

        non_dp_factor = self.parallelism_config.non_dp_replication_factor
        if non_dp_factor > 1 and self.model.training:
            added = self._total_train_tokens - prev_tokens
            if added > 0:
                self._total_train_tokens = prev_tokens + added // non_dp_factor
                self._metrics["train"]["num_tokens"] = [self._total_train_tokens]
        self._relabel_attended_tokens()
        return result

    def _weigh_router_aux_loss(self, model, loss, outputs, num_items_in_batch):
        """Weigh the router aux loss at its coefficient: ``coef x mean(aux)`` per optimizer step, and
        ``coef x aux`` in an eval batch's loss.

        On HF's ``num_items_in_batch`` path ``Trainer.compute_loss`` multiplies a micro-batch loss by the process
        count and ``Trainer.training_step`` sums the accumulation window instead of averaging it. That makes the
        token-summed language-model term the window's mean, but the aux term the forward folds in is a
        per-micro-batch mean and would weigh ``processes x accumulation`` times its coefficient. The forward adds
        it only when it receives the labels, which HF withholds under a label smoother or a ``compute_loss_func``;
        the term is then added here alone.
        """
        aux = outputs.get("aux_loss") if isinstance(outputs, dict) else None
        if aux is None or num_items_in_batch is None:
            return loss
        weighted = require_router_aux_loss_coef(base_transformers_model(model).config) * aux.to(loss.dtype)
        forward_added = weighted if self.label_smoother is None and self.compute_loss_func is None else 0.0
        target = weighted / self.current_gradient_accumulation_steps if model.training else weighted
        return loss - forward_added * self._hf_loss_scale() + target

    def _hf_loss_scale(self) -> int:
        """The factor ``Trainer.compute_loss`` multiplies a loss by on the ``num_items_in_batch`` path, which HF
        computes only for a model accepting loss kwargs or a ``compute_loss_func``."""
        if not self.args.average_tokens_across_devices:
            return 1
        if self.args.n_gpu > 1:
            return self.args.n_gpu
        scale = self.accelerator.num_processes
        if (pc := getattr(self.accelerator, "parallelism_config", None)) is not None:
            scale //= pc.tp_size
        return scale

    def _relabel_attended_tokens(self):
        """Rename TRL's cumulative ``num_tokens`` (non-padding attended tokens) to
        ``num_attended_tokens_seen`` — disambiguates it from ``num_input_tokens_seen`` (incl. padding)
        and ``num_unmasked_output_tokens_seen`` (loss-contributing only). Value unchanged.
        """
        for mode_metrics in self._metrics.values():
            if "num_tokens" in mode_metrics:
                mode_metrics["num_attended_tokens_seen"] = mode_metrics.pop("num_tokens")

    def _compute_cp_metrics(self, outputs, full_labels, full_attention_mask=None):
        """Accumulate mean_token_accuracy/entropy/aux_loss/num_attended_tokens_seen for CP on the local chunk.

        Logits cover the local chunk only; non-final ranks' logits[-1] predicts the next chunk's first
        token (matching loss compute).

        Accumulation is rank-local: no collective and no ``.item()``. Each quantity is a plain sum,
        so folding the micro-batches into one on-device row and reducing it once per log
        (:meth:`_drain_cp_metrics`) gives the same totals as reducing each micro-batch separately.
        """
        mode = "train" if self.model.training else "eval"

        if full_labels is None or outputs.logits is None:
            return

        cp_rank = self.cp_config.cp_rank

        with torch.no_grad():
            shift_logits, shift_labels = cp_shift_against_full_labels(
                outputs.logits, full_labels, cp_rank, self.cp_size
            )

            predictions = shift_logits.argmax(dim=-1)
            mask = shift_labels != LABEL_IGNORE_INDEX
            correct_predictions = (predictions == shift_labels) & mask

            per_token_entropy = entropy_from_logits(outputs.logits)
            if full_attention_mask is not None:
                start, end = cp_chunk_bounds(full_attention_mask.size(1), cp_rank, self.cp_size)
                local_attention_mask = full_attention_mask[:, start:end]
                entropy_sum = torch.sum(per_token_entropy * local_attention_mask)
                entropy_tokens = local_attention_mask.sum()
                attended_tokens = entropy_tokens
            else:
                entropy_sum = per_token_entropy.sum()
                entropy_tokens = torch.tensor(per_token_entropy.numel(), device=per_token_entropy.device)
                attended_tokens = torch.tensor(
                    outputs.logits.shape[0] * outputs.logits.shape[1], device=outputs.logits.device
                )

            aux_loss = getattr(outputs, "aux_loss", None)
            aux_present = float(aux_loss is not None)
            columns = {
                # Whether this window saw a CP forward at all, world-wide; the drain emits nothing
                # rather than a 0.0 accuracy when it did not.
                "microbatches": entropy_sum.new_ones(()),
                "correct_tokens": correct_predictions.sum(),
                "loss_tokens": mask.sum(),
                "entropy_sum": entropy_sum,
                "entropy_tokens": entropy_tokens,
                # Only train accumulates the run's token counter; eval contributes a structural zero.
                "attended_tokens": attended_tokens * int(mode == "train"),
                # mean(), not the raw tensor: torch.stack needs a 0-dim column, and a family whose
                # aux loss arrives per-layer would otherwise crash the metric path.
                "aux_loss_sum": entropy_sum.new_zeros(()) if aux_loss is None else aux_loss.detach().mean(),
                "aux_loss_count": entropy_sum.new_full((), aux_present),
            }
            # Stacked by name, so a renamed or dropped column raises here instead of swapping two
            # metrics' slots; the width check catches a column added to the dict alone. Cast per
            # column: the counts are int64 and the sums float, and torch.stack refuses a mixed list.
            if len(columns) != len(_CP_METRIC_COLUMNS):
                raise ValueError(f"CP metric columns {sorted(columns)} do not match {_CP_METRIC_COLUMNS}")
            row = torch.stack([columns[name].to(_CP_METRIC_DTYPE) for name in _CP_METRIC_COLUMNS])

        previous = self._cp_metric_accum.get(mode)
        self._cp_metric_accum[mode] = row if previous is None else previous + row

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """Fold the deferred CP metric sums into ``_metrics`` before the base trainer averages them."""
        self._drain_cp_metrics()
        super().log(logs, start_time)

    def _drain_cp_metrics(self):
        """Reduce this log window's CP metric sums across the world, once, and emit the ratios.

        Rank-uniform by construction: the gate is ``is_cp_mode`` (a config constant every rank agrees
        on), the buffer is a fixed-width row whatever a rank accumulated, and a rank with an empty
        window enters the reduce with zeros rather than skipping it. ``reduce``, not ``gather``: only
        the sum is used, and an all-gather would put an O(world) tensor on every rank.

        The emitted ratios are token-weighted over the window, not an unweighted mean of the
        per-micro-batch ratios; ``num_attended_tokens_seen`` is a plain sum.
        """
        if not self.is_cp_mode:
            return
        mode = "train" if self.model.training else "eval"
        row = self._cp_metric_accum.get(mode)
        if row is None:
            row = torch.zeros(len(_CP_METRIC_COLUMNS), dtype=_CP_METRIC_DTYPE, device=self.accelerator.device)
        # One tolist() is the window's only device→host sync, against three .item()s per micro-batch.
        totals = dict(zip(_CP_METRIC_COLUMNS, self.accelerator.reduce(row, reduction="sum").tolist(), strict=True))
        self._cp_metric_accum[mode] = None
        # Read off the reduced count, after the reduce, so the decision is world-wide and identical
        # on every rank. A window with no CP forward anywhere (the final log of a run, or a mode
        # that never ran) emits nothing rather than a 0.0 accuracy.
        if totals["microbatches"] == 0:
            return

        loss_tokens = totals["loss_tokens"]
        entropy_tokens = totals["entropy_tokens"]
        accuracy = totals["correct_tokens"] / loss_tokens if loss_tokens > 0 else 0.0
        self._metrics[mode]["mean_token_accuracy"].append(accuracy)
        self._metrics[mode]["entropy"].append(totals["entropy_sum"] / entropy_tokens if entropy_tokens > 0 else 0.0)

        # One token per CP rank, distinct sequences per DP group and no TP+CP, so this cannot double-count.
        self._total_train_tokens += int(totals["attended_tokens"])
        self._metrics[mode]["num_attended_tokens_seen"] = [self._total_train_tokens]

        if totals["aux_loss_count"] > 0:
            self._metrics[mode]["aux_loss"].append(totals["aux_loss_sum"] / totals["aux_loss_count"])

    @cached_property
    def _forward_accepts_skip_logits(self) -> bool:
        """Whether forward was patched with Liger ``lce_forward`` (exposes ``skip_logits``).

        Evaluated lazily (first eval step) so both the class-level patch and TRL's instance-level
        re-application are visible; cached.
        """
        model = self._top_level_model()
        try:
            return "skip_logits" in inspect.signature(model.forward).parameters
        except (TypeError, ValueError):
            return False

    def _should_skip_eval_logits(self, prediction_loss_only: bool, inputs) -> bool:
        """Whether loss-only eval may take Liger's fused linear-cross-entropy ``skip_logits`` path.

        Liger's ``lce_forward`` defaults ``skip_logits`` to ``self.training and labels is not None``,
        so eval otherwise materializes full ``[B, S, vocab]`` fp32 logits. Each clause names a
        consumer that still needs real logits: ``prediction_loss_only=False`` (the eval loop returns
        them), ``_loss_reads_logits`` (a subclass loss reads ``outputs.logits`` and forwards
        ``inputs`` without labels, where ``skip_logits=True`` makes lce_forward raise), CP mode, and
        no Liger / no ``skip_logits`` parameter.
        """
        return bool(
            prediction_loss_only
            and not self._loss_reads_logits
            and not self.is_cp_mode
            and self.args.use_liger_kernel
            and "skip_logits" not in inputs
            and inputs.get("labels") is not None
            and self._forward_accepts_skip_logits
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Loss-only eval routes through Liger's fused CE (see :meth:`_should_skip_eval_logits`)."""
        if self._should_skip_eval_logits(prediction_loss_only, inputs):
            inputs = {**inputs, "skip_logits": True}
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
