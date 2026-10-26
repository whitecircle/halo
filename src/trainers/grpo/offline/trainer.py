# Copyright 2024 White Circle
#
# Licensed under the Halo License (the "License"): the Apache License, Version 2.0
# (a copy is provided in APACHE-2.0.txt), as modified by, and subject to, the
# Supplemental Terms in the LICENSE file at the root of this repository.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline GRPO trainer: group-relative policy optimization over stored completions and rewards, under
EP / TP / CP / PP. Full fine-tuning scores raw run-start reference log-probs once and checkpoints them; CP
owns token shards of the same rows, and PP's microbatch numerators share the whole-batch denominators.
"""

import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any, Union

import datasets
import torch
import torch.nn.functional as F
from peft import PeftConfig
from torch import nn
from torch.utils.data import Dataset, IterableDataset
from transformers import (
    BaseImageProcessor,
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
)
from trl.trainer.utils import disable_dropout_in_model, selective_log_softmax

from src.callbacks.variable_scheduler import VariableSchedulerCallback
from src.configs.grpo_offline import OfflineGRPOConfig
from src.data.collators.offline_grpo import (
    REF_PER_TOKEN_LOGPS_COLUMN,
    OfflineGRPOCPDataCollatorWithPadding,
    OfflineGRPODataCollatorWithPadding,
)
from src.distributed.context_parallel.sequence_chunks import cp_chunk_bounds, cp_sum_rows
from src.distributed.parallelism_config import ParallelismConfig
from src.kernels.logprobs import LABEL_IGNORE_INDEX, loss_token_counts_per_row
from src.model_loading.parallel.frozen_models import place_and_freeze
from src.model_loading.parallel.peft_setup import peft_bf16_autocast, prepare_policy_model
from src.model_loading.parallel.whole_model import resolve_trainer_model
from src.models.context_window import is_bounded_length
from src.runtime.collectives import all_gather_cat
from src.runtime.ranks import is_global_main_process
from src.trainers.distributed_trainer import DistributedTrainerMixin
from src.trainers.grpo.logprob_scoring import LogitsWidth, LogprobScoringMixin
from src.trainers.grpo.objective.advantages import GROUP_ADVANTAGE_METHODS
from src.trainers.grpo.objective.offline_loss import (
    LOSS_TYPES,
    PG_FORMULATIONS,
    clamp_negative_advantage_logps,
    offline_loss,
    offline_token_objective,
)
from src.trainers.grpo.offline.group_rows import tokenize_offline_splits
from src.trainers.grpo.offline.group_sampler import GroupedLoaderMixin
from src.trainers.grpo.offline.offline_rules import (
    reject_encoder_decoder,
    reject_inert_max_length,
    reject_offline_cp_options,
    reject_offline_pp_options,
    reject_unsupported_reference_input,
    require_processing_class,
    resolve_pp_max_length,
)
from src.trainers.grpo.offline.pipeline_loss import OfflineGRPOPipelineMixin
from src.trainers.grpo.offline.reference import OfflineGRPOReferenceMixin
from src.trainers.training_args_overrides import keep_all_dataset_columns
from src.trainers.world_metrics import WorldMetrics

logger = logging.getLogger(__name__)


class OfflineGRPOTrainer(
    LogprobScoringMixin,
    OfflineGRPOPipelineMixin,
    GroupedLoaderMixin,
    OfflineGRPOReferenceMixin,
    DistributedTrainerMixin,
    Trainer,
):
    """Offline GRPO over ``prompt`` / ``completions`` / ``rewards`` rows. Full fine-tuning freezes run-start
    per-token reference scores in each checkpoint; PEFT scores its base with adapters disabled; native
    expert LoRA uses an explicit frozen base model."""

    _tag_names = ["trl", "offline-grpo"]

    # Single-forward objective whose denominators come from batch metadata alone; the KL reference
    # is scored by the shared reference sweep before training.
    PP_REFUSAL = None
    SUPPORTS_CP = True
    POLICY_GRADIENT_LOSS = True
    # The objective divides by its own loss-type denominator, so the parent's grad-accum scaling stands.
    LOSS_IS_OWN_MEAN = True

    # Buffered per-sample key → logged sign prefixes (unclamped: negative side only, where it binds).
    _SIGN_METRIC_KEYS = {
        "logps": ("positive", "negative"),
        "pg_objective": ("positive", "negative"),
        "logps_unclamped": ("negative",),
        "kl": ("positive", "negative"),
        "ref_logps": ("positive", "negative"),
        "ref_logps_unclamped": ("negative",),
    }

    def __init__(
        self,
        model: PreTrainedModel | nn.Module = None,
        args: OfflineGRPOConfig = None,
        train_dataset: Union[Dataset, IterableDataset, "datasets.Dataset"] | None = None,
        eval_dataset: Union[Dataset, dict[str, Dataset], "datasets.Dataset"] | None = None,
        processing_class: PreTrainedTokenizerBase
        | BaseImageProcessor
        | FeatureExtractionMixin
        | ProcessorMixin
        | None = None,
        model_init: Callable[[], PreTrainedModel] | None = None,
        compute_loss_func: Callable | None = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        optimizer_cls_and_kwargs: tuple[type[torch.optim.Optimizer], dict[str, Any]] | None = None,
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: PeftConfig | None = None,
        ref_model: PreTrainedModel | nn.Module | None = None,
        parallelism_config: "ParallelismConfig" = None,
        save_sharded_ep: bool = False,
        **kwargs,
    ):
        resume_context_given = "resume_checkpoint" in kwargs
        self._init_reference_logps(
            resume_checkpoint=kwargs.pop("resume_checkpoint", None), resume_context_given=resume_context_given
        )
        reject_offline_cp_options(
            parallelism_config, ref_model, self._reference_mode(model, parallelism_config, peft_config)
        )
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._sign_metrics = {"train": WorldMetrics(), "eval": WorldMetrics()}

        self._configure_objective(args, require_processing_class(processing_class))
        self._precompute_reference = self.sweeps_run_start_reference(
            model, self.beta, parallelism_config, peft=peft_config is not None
        )
        reject_unsupported_reference_input(
            train_dataset,
            eval_dataset,
            active=self._precompute_reference,
            presharded=kwargs.get("dataset_presharded", False),
        )
        # Read by LogprobScoringMixin: CP always scores through the chunked sweep, PP refuses it.
        self._use_chunked_grpo_logprobs = args.use_chunked_grpo_logprobs or (
            parallelism_config is not None and parallelism_config.is_cp_mode
        )
        # Stored completions carry no sampling distribution to match, so log-probs are plain
        # (the full-logits path's implicit temperature); the chunked kernel divides by this.
        self.temperature = 1.0

        reject_inert_max_length(args, parallelism_config)
        reject_offline_pp_options(args, parallelism_config, peft_config, compute_metrics)
        args.max_length = resolve_pp_max_length(args, parallelism_config)
        model = resolve_trainer_model(model, args, parallelism_config=parallelism_config)
        reject_encoder_decoder(model)
        model, self._peft_has_been_casted_to_bf16 = prepare_policy_model(
            model, peft_config, args, disable_dropout=args.disable_dropout
        )
        # The model rides through the distributed seam so PP can split it into this rank's stage.
        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            model=model,
        )
        model = dist_kwargs.pop("model")
        data_collator = (
            OfflineGRPOCPDataCollatorWithPadding(pad_token_id=self.padding_value, cp_size=parallelism_config.cp_size)
            if parallelism_config.is_cp_mode
            else OfflineGRPODataCollatorWithPadding(pad_token_id=self.padding_value)
        )
        keep_all_dataset_columns(
            args,
            "OfflineGRPODataCollatorWithPadding requires remove_unused_columns=False (it reads "
            "prompt/completion/reward columns the Trainer would otherwise strip); forcing it off. "
            "Set it explicitly in your OfflineGRPOConfig.",
        )
        train_dataset, eval_dataset = tokenize_offline_splits(
            train_dataset, eval_dataset, args, processing_class, getattr(model, "config", None)
        )

        self.ref_model = self._kl_reference(model, ref_model, self.beta, self.parallelism_config, peft_config)
        if self.ref_model is not None and args.disable_dropout:
            disable_dropout_in_model(self.ref_model)
        callbacks = self._with_min_log_prob_scheduler(args, model, callbacks)

        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            processing_class,
            model_init,
            compute_loss_func,
            compute_metrics,
            callbacks,
            optimizers,
            optimizer_cls_and_kwargs,
            preprocess_logits_for_metrics,
            **dist_kwargs,
        )
        self._finish_offline_init(train_dataset, eval_dataset)

    def _configure_objective(self, args: OfflineGRPOConfig, tokenizer) -> None:
        """Store the objective's settings. The loss-type and PG dispatches branch on these strings, so
        they are refused here, before model and dataset construction, not at the first microbatch."""
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_completion_length = args.max_completion_length
        self.min_log_prob = args.min_log_prob
        self.loss_type = args.loss_type
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}. Supported types: {list(LOSS_TYPES)}")
        if args.advantage_method not in GROUP_ADVANTAGE_METHODS:
            raise ValueError(
                f"Unknown advantage_method: {args.advantage_method!r}. Supported: {list(GROUP_ADVANTAGE_METHODS)}"
            )
        if args.policy_gradient_formulation not in PG_FORMULATIONS:
            raise ValueError(
                f"Unknown policy_gradient_formulation: {args.policy_gradient_formulation!r}. "
                f"Supported: {list(PG_FORMULATIONS)}"
            )
        if self.loss_type == "dr_grpo" and not is_bounded_length(self.max_completion_length):
            raise ValueError(
                f"loss_type='dr_grpo' normalizes by max_completion_length, so it must be set to a "
                f"positive value (got {self.max_completion_length!r}); it is the loss constant, not a "
                f"generation budget. A zero or negative constant divides the loss by zero or flips its sign."
            )
        self.beta = args.kl_beta
        if is_global_main_process():
            logger.info(f"Using loss type: {self.loss_type}")
            logger.info(f"Using PG formulation: {args.policy_gradient_formulation}")

    @staticmethod
    def _with_min_log_prob_scheduler(args: OfflineGRPOConfig, model, callbacks):
        """``callbacks`` plus the ``min_log_prob`` ramp when ``initial_min_log_prob`` schedules one, which
        seeds the floor on ``model``."""
        if args.initial_min_log_prob is not None and args.min_log_prob is not None:
            model.min_log_prob = args.initial_min_log_prob
            if is_global_main_process():
                logger.info(f"Initialized model.min_log_prob = {args.initial_min_log_prob} for scheduling")

            min_log_prob_scheduler = VariableSchedulerCallback(
                attribute_name="min_log_prob", initial_value=args.initial_min_log_prob, final_value=args.min_log_prob
            )

            if callbacks is None:
                callbacks = []
            callbacks.append(min_log_prob_scheduler)
            if is_global_main_process():
                logger.info(f"Added min_log_prob scheduler: {args.initial_min_log_prob} → {args.min_log_prob}")
        elif args.initial_min_log_prob is not None and args.min_log_prob is None:
            if is_global_main_process():
                logger.warning("initial_min_log_prob is set but min_log_prob is None. Scheduler will not be added.")
        elif args.initial_min_log_prob is None and args.min_log_prob is not None:
            if is_global_main_process():
                logger.info(f"Using constant min_log_prob = {args.min_log_prob} (no scheduling)")
        elif is_global_main_process():
            logger.info("No min_log_prob clipping enabled")
        return callbacks

    def _finish_offline_init(self, train_dataset, eval_dataset) -> None:
        """Post-``Trainer`` setup: tags, reference placement, the loaders' group ids, the parallel modes,
        the head-path and memory checks, then the run-start reference sweep."""
        self.model.add_model_tags(self._tag_names)
        if self.ref_model is not None:
            place_and_freeze(self.ref_model, self.model)
            if not self._precompute_reference:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        self._cache_group_ids(train_dataset, eval_dataset)
        self._configure_distributed_modes()
        self._resolve_chunked_head_transform()
        self._verify_full_logits_fit(self._loss_logits_width())
        if self._precompute_reference:
            self.train_dataset = self._precompute_reference_logps(self.train_dataset, "training")
            if self.eval_dataset is not None:
                self.eval_dataset = self._precompute_reference_logps(self.eval_dataset, "evaluation")
            if self.ref_model is not None:
                self.ref_model.to("cpu")
                self.ref_model = None

    def _loss_logits_width(self) -> LogitsWidth | None:
        """The completion logits row the loss forward carries, one logit past the completion kept for
        the next-token shift. ``None`` under PP, where the last stage's plane is the pipeline's and
        use_chunked_grpo_logprobs cannot remove it, and with max_completion_length unset, which leaves
        the width to the stored completions."""
        if self._pp_runner is not None or not is_bounded_length(self.max_completion_length):
            return None
        return LogitsWidth(self.max_completion_length + 1, "max_completion_length")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("OfflineGRPOTrainer does not support returning outputs")

        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            return self._compute_loss_inner(model, inputs)

    def _compute_loss_inner(self, model, inputs):
        if self.parallelism_config.is_cp_mode:
            return self._compute_cp_loss_inner(model, inputs)
        input_ids, attention_mask, logits_to_keep = self._joined_rows(inputs)
        completion_mask = inputs["completion_attention_mask"]
        advantages = inputs["advantage"]  # [B], pre-computed at tokenization (no gather needed)
        logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        ref_logps = None
        if self.beta != 0.0:
            if self._precompute_reference:
                ref_logps = inputs.get(REF_PER_TOKEN_LOGPS_COLUMN)
                if ref_logps is None:
                    raise RuntimeError("Offline GRPO needs the checkpointed run-start reference on every row")
            else:
                with torch.no_grad(), self._reference_policy() as reference:
                    ref_logps = self._get_per_token_logps(reference, input_ids, attention_mask, logits_to_keep)
        per_token_loss, sample_values = self._token_objective(
            logps, advantages, ref_logps, self._live_min_log_prob
        )  # [B, T]

        loss = offline_loss(
            per_token_loss,
            completion_mask,
            inputs["group_size"],
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
        )

        self._record_sign_metrics(sample_values, advantages, completion_mask)
        return loss

    def _compute_cp_loss_inner(self, model, inputs):
        """Score local CP targets and reduce only each logical row's token sums.

        The collator right-pads full rows. The scorer shifts the target at every shard boundary;
        reference values are stored at the target-token positions in that same full-row grid.
        """
        ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        local_logps, local_labels = self._cp_chunked_logps(model, ids, attention_mask, labels)
        advantages = inputs["advantage"]

        ref_logps = None
        if self.beta != 0.0:
            if REF_PER_TOKEN_LOGPS_COLUMN not in inputs:
                raise RuntimeError(
                    "CP offline GRPO with kl_beta > 0 needs the run-start raw reference log-probs "
                    "on every row; a fresh run must sweep them before training."
                )
            full_reference = inputs[REF_PER_TOKEN_LOGPS_COLUMN]
            if full_reference.shape != labels.shape:
                raise ValueError("CP reference log-probs must align with the full-row target grid")
            start, _ = cp_chunk_bounds(ids.size(1), self.cp_config.cp_rank, self.cp_config.cp_size)
            ref_logps = full_reference[:, start + 1 : start + 1 + local_logps.size(1)]

        per_token_loss, sample_values = self._token_objective(
            local_logps, advantages, ref_logps, self._live_min_log_prob
        )
        supervised = local_labels != LABEL_IGNORE_INDEX
        loss = offline_loss(
            per_token_loss,
            supervised,
            inputs["group_size"],
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
            row_token_counts=loss_token_counts_per_row(labels),
            cp_config=self.cp_config,
        )
        self._record_sign_metrics(sample_values, advantages, supervised)
        return loss

    @staticmethod
    def _joined_rows(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, int]:
        """``(input_ids, attention_mask, logits_to_keep)`` of a collated batch: each left-padded prompt joined
        to its right-padded completion in one causal row, and the completion width the scorer keeps."""
        completion_ids = batch["completion_input_ids"]
        input_ids = torch.cat([batch["prompt_input_ids"], completion_ids], dim=1)
        attention_mask = torch.cat([batch["prompt_attention_mask"], batch["completion_attention_mask"]], dim=1)
        return input_ids, attention_mask, completion_ids.size(1)

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        """Raw per-token completion log-probs; full logits restricted with ``logits_to_keep``, or never
        materialized at all under ``use_chunked_grpo_logprobs``."""
        if self._scores_full_logits:
            # +1 because the last logit (next-token pred) is excluded below.
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V)

            input_ids = input_ids[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]  # restrict to completion tokens
            selected_logps = selective_log_softmax(logits, input_ids)
        else:
            selected_logps, _ = self._chunked_completion_logps(model, input_ids, attention_mask, logits_to_keep)
        return selected_logps

    @property
    def _scores_full_logits(self) -> bool:
        """Whether log-probs come from TRL's ``selective_log_softmax`` over full logits, which keeps the
        logits' dtype, the run's compute dtype. The chunked kernel (always on under CP) and the
        pipeline's last-stage loss score in fp32."""
        return not self._use_chunked_grpo_logprobs and self._pp_runner is None

    @property
    def _live_min_log_prob(self) -> float | None:
        """The negative-advantage log-prob floor in force right now (``None`` = no clipping configured).

        ``VariableSchedulerCallback`` writes the scheduled value onto the unwrapped ``self.model``; the
        DDP/FSDP-wrapped module handed to ``compute_loss`` never carries it, and reading the floor off
        that wrapper would pin step 0's value for the whole run.
        """
        return getattr(self.model, "min_log_prob", self.min_log_prob)

    def _token_objective(self, logps, advantages, ref_logps, min_log_prob):
        """:func:`offline_token_objective` over raw policy and reference log-probs, each floored at
        ``min_log_prob`` on the negative-advantage rows (:func:`clamp_negative_advantage_logps`)."""
        return offline_token_objective(
            clamp_negative_advantage_logps(logps, advantages, min_log_prob),
            logps,
            advantages,
            policy_gradient_formulation=self.args.policy_gradient_formulation,
            beta=self.beta,
            ref_logps=None
            if ref_logps is None
            else clamp_negative_advantage_logps(ref_logps, advantages, min_log_prob),
            ref_logps_unclamped=ref_logps,
        )

    def _record_sign_metrics(
        self,
        sample_values: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        mask: torch.Tensor,
        rows: torch.Tensor | None = None,
    ) -> None:
        """Record the moments of one batch's per-sample means of the objective's diagnostics, split by advantage
        sign, for ``log``'s flush (on device: a per-microbatch ``.item()`` would be dozens of host syncs a step).
        ``rows`` selects the real rows of a row-padded pipeline batch, whose inert rows would count as zero-valued
        positive samples."""
        # Count and every detached diagnostic travel in one CP sum. Each CP shard owns
        # tokens of the same logical rows; off CP this is the same local row reduction.
        keys = tuple(sample_values)
        totals = torch.stack(
            [mask.sum(dim=1, dtype=torch.float32)]
            + [(sample_values[key].detach() * mask).sum(dim=1, dtype=torch.float32) for key in keys]
        )
        totals = cp_sum_rows(totals, self.cp_config if self.parallelism_config.is_cp_mode else None)
        counts = totals[0].clamp(min=1)
        metrics = self._sign_metrics["train" if self.model.training else "eval"]
        positive = advantages.detach() >= 0
        real = torch.ones_like(positive) if rows is None else rows
        signs = {"positive": real & positive, "negative": real & ~positive}
        for index, key in enumerate(keys, start=1):
            for sign in self._SIGN_METRIC_KEYS[key]:
                metrics.moments(f"{sign}/{key}", totals[index] / counts, signs[sign])

    def _score_reference_batch(self, batch) -> list[torch.Tensor]:
        """Mode-specific scoring inside the shared reference-sweep lifecycle."""
        if self.parallelism_config.is_cp_mode:
            return self._cp_score_reference_batch(batch)
        if self._pp_runner is not None:
            return self._pp_score_reference_batch(batch)
        logps = self._get_per_token_logps(
            self.ref_model if self.ref_model is not None else self.model, *self._joined_rows(batch)
        )
        return [
            values[valid.bool()].float().cpu()
            for values, valid in zip(logps, batch["completion_attention_mask"], strict=True)
        ]

    def _cp_score_reference_batch(self, batch) -> list[torch.Tensor]:
        local_logps, _ = self._cp_chunked_logps(
            self.model, batch["input_ids"], batch["attention_mask"], batch["labels"]
        )
        start, end = cp_chunk_bounds(batch["input_ids"].size(1), self.cp_config.cp_rank, self.cp_config.cp_size)
        local_logps = F.pad(local_logps.float(), (0, end - start - local_logps.size(1)))
        full_logps = all_gather_cat(local_logps, 1, self.cp_config.process_group, self.cp_config.cp_size)
        # Every rank holds the full labels; the shards' shifted targets tile this grid.
        full_valid = F.pad(batch["labels"][:, 1:] != LABEL_IGNORE_INDEX, (0, 1))
        return [logps[valid].cpu() for logps, valid in zip(full_logps, full_valid, strict=True)]

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: list[str] | None = None):
        if self._pp_runner is not None:
            # compute_loss (and its sign-metric recording) needs the full model, so the mixin drives
            # the schedule instead.
            return self._pp_prediction_step(inputs, prediction_loss_only, ignore_keys)
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        # model.training, not control.should_evaluate: HF sets should_evaluate alongside should_log in
        # on_step_end and only clears it after evaluate() runs, so on a step that both logs and evals
        # the training log call would read "eval", dropping the train sign metrics (their records are
        # never flushed) and prefixing eval_* onto a train line.
        mode = "train" if self.model.training else "eval"
        # COLLECTIVE when anything was recorded. Under PP only the last stage's ranks record anything, as DP
        # replicas of the whole sample set, so they fold over the stage group (a world flush would block the other
        # stages in a collective they never enter) and the chain gets the scalars from the broadcast below.
        sign_metrics = self._sign_metrics[mode]
        if sign_metrics:
            sign_metrics.flush(self._metrics[mode], group=self._pp_stage_group)
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        current_min_log_prob = self._live_min_log_prob
        if current_min_log_prob is not None:  # None = no clipping configured
            metrics["min_log_prob_restriction"] = current_min_log_prob

        if self._pp_runner is not None:
            metrics = self._pp_broadcast_log_metrics(metrics)

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
