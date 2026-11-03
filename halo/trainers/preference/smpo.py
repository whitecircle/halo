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
"""SMPO: reference-free margin loss plus an optional SFT term. EP/CP/TP and PP (:class:`SMPOPipelineLoss`). ``is_vlm``
takes raw message rows plus ``images``, processed in the collator, and excludes ``padding_free``, CP and PP."""

import logging
from collections.abc import Callable
from typing import Any, Literal

import torch
import torch.nn as nn
from datasets import Dataset, IterableDataset
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from halo.callbacks.variable_scheduler import VariableSchedulerCallback
from halo.configs.preference import SMPOConfig
from halo.data.collators.collator_rules import reject_multi_document_rows
from halo.data.collators.smpo import DataCollatorForSMPO, DataCollatorForVLMSMPO, concatenate_pairs
from halo.data.pipeline.coordinated_map import coordinated_map, resolve_map_num_proc
from halo.data.pipeline.preferences import VLM_SMPO_FEATURES, tokenize_preference_row, tokenize_vlm_preference_row
from halo.data.turn_markers import resolve_eos_token_ids
from halo.distributed.context_parallel.sequence_chunks import cp_chunk_bounds
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, next_token_logprobs, token_logprobs
from halo.model_loading.parallel.peft_setup import peft_bf16_autocast, prepare_policy_model
from halo.model_loading.parallel.whole_model import resolve_trainer_model
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.attention.selection import effective_attn_implementation, is_flash_attn_implementation
from halo.models.segment_markers import (
    SegmentMarkers,
    document_ids,
    flattened_document_positions,
    segment_marker_kwargs,
    segment_markers_for,
)
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.evaluation import split_rows_head, split_rows_mean
from halo.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter
from halo.trainers.preference.preference_rules import (
    reject_smpo_options,
    reject_smpo_pp_options,
    reject_smpo_vlm_options,
)
from halo.trainers.preference.smpo_objective import (
    SMPOObjective,
    SMPOStepConstants,
    cp_global_mean,
    masked_logit_mean,
    metric_sums,
    metrics_from_sums,
)
from halo.trainers.preference.smpo_pipeline_loss import SMPOPipelineLoss
from halo.trainers.stored_metrics import StoredMetricsMixin
from halo.trainers.trainer_rules import reject_cp_packed_collator

logger = logging.getLogger(__name__)

# The columns each data path's prepared rows carry; present, the dataset is already prepared.
_TEXT_COLUMNS = ("prompt_input_ids", "chosen_input_ids", "rejected_input_ids")
_VLM_COLUMNS = tuple(VLM_SMPO_FEATURES)
# The raw columns both tokenize maps read; a dataset without them cannot be tokenized again.
_RAW_PAIR_COLUMNS = ("chosen", "rejected")


def resolve_vlm_mode(processing_class, is_vlm: bool | None) -> bool:
    """The checkpoint cannot decide the data path: a natively multimodal checkpoint on text pairs is a text run, and
    only the text path keeps CP, ``padding_free`` and PP. The script decides it; omitted, it follows
    ``processing_class``."""
    if is_vlm is None:
        return resolve_tokenizer(processing_class) is not processing_class
    if is_vlm and not isinstance(processing_class, ProcessorMixin):
        raise ValueError(
            f"is_vlm=True requires a ProcessorMixin as processing_class, got "
            f"{type(processing_class).__name__}: the VLM path chat-templates rows through the "
            f"processor and DataCollatorForVLMSMPO expands image placeholders with it. Pass the "
            f"processor, or run in text mode."
        )
    return is_vlm


def _margin_schedule_callback(args: SMPOConfig, model) -> VariableSchedulerCallback:
    """Seed ``model.target_margin`` at ``initial_margin`` and schedule it up to ``target_margin``."""
    model.target_margin = args.initial_margin
    if is_global_main_process():
        logger.info(f"Initialized model.target_margin = {args.initial_margin} for scheduling")
        logger.info(f"Added target_margin scheduler: {args.initial_margin} → {args.target_margin}")
    return VariableSchedulerCallback(
        attribute_name="target_margin", initial_value=args.initial_margin, final_value=args.target_margin
    )


def _side_statistics(
    seq_logps: torch.Tensor,
    num_chosen: int,
    cp_config,
    *,
    chosen: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    rejected: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """A forward's per-side outputs from each side's ``(logits, loss_mask, per-token NLL)``."""
    chosen_logits, chosen_mask, chosen_nll = chosen
    rejected_logits, rejected_mask, rejected_nll = rejected
    return {
        "chosen_logps": seq_logps[:num_chosen],
        "rejected_logps": seq_logps[num_chosen:],
        # Masked fp32 reductions: boolean-indexing [tokens, V] logits copies them, `.any()` syncs.
        "mean_chosen_logits": masked_logit_mean(chosen_logits, chosen_mask, cp_config),
        "mean_rejected_logits": masked_logit_mean(rejected_logits, rejected_mask, cp_config),
        # Counted off the mask, never the NLL's support: a saturated token's NLL is legitimately 0 and counts.
        "chosen_sft_loss": cp_global_mean(chosen_nll.sum(), chosen_mask.sum(), cp_config),
        "rejected_sft_loss": cp_global_mean(rejected_nll.sum(), rejected_mask.sum(), cp_config),
    }


class SMPOTrainer(StoredMetricsMixin, DistributedTrainerMixin, Trainer):
    _tag_names = ["trl", "smpo", "smooth-margin-po"]

    SUPPORTS_CP = True
    # Reference-free single forward; correct once the concat is interleaved so no pair splits.
    PP_REFUSAL = None
    LOSS_IS_OWN_MEAN = True
    # chosen ⧺ rejected ride one forward, so every MoE layer sees twice the collator's rows.
    FORWARD_ROWS_PER_EXAMPLE = 2

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SMPOConfig | None = None,
        data_collator: DataCollator | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: Dataset | IterableDataset | dict[str, Dataset] | None = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        compute_metrics: Callable[[EvalLoopOutput], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Any | None = None,
        parallelism_config: ParallelismConfig = None,
        save_sharded_ep: bool = False,
        dataset_presharded: bool = False,
        is_vlm: bool | None = None,
        **kwargs,
    ):
        if args is None:
            args = SMPOConfig("smpo-output")
        if processing_class is None:
            raise ValueError("processing_class (tokenizer) must be provided")

        tokenizer = resolve_tokenizer(processing_class)
        self.is_vlm = resolve_vlm_mode(processing_class, is_vlm)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.objective = SMPOObjective.from_args(args)
        _, self.max_prompt_length, self.max_completion_length = args.resolve_length_budget()
        reject_smpo_vlm_options(args, parallelism_config, is_vlm=self.is_vlm)
        reject_smpo_options(args, parallelism_config)
        reject_smpo_pp_options(args, parallelism_config, peft_config, is_vlm=self.is_vlm)

        model = self._load_policy(model, args, peft_config, parallelism_config, tokenizer)

        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
            model=model,
        )
        model = dist_kwargs.pop("model")
        reject_cp_packed_collator(data_collator, is_cp_mode=self.parallelism_config.is_cp_mode)

        if data_collator is None and self.is_vlm:
            data_collator = DataCollatorForVLMSMPO(
                pad_token_id=self.pad_token_id, processor=processing_class, max_prompt_length=self.max_prompt_length
            )
        elif data_collator is None:
            data_collator = DataCollatorForSMPO(pad_token_id=self.pad_token_id)

        train_dataset = self._prepare_dataset(train_dataset, processing_class, args, "train")
        eval_dataset = self._prepare_dataset(eval_dataset, processing_class, args, "eval")

        if callbacks is None:
            callbacks = []
        if args.use_margin_schedule:
            callbacks.append(_margin_schedule_callback(args, model))

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **dist_kwargs,
        )
        self.model.add_model_tags(self._tag_names)
        self._configure_distributed_modes()

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            loss, metrics, _ = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")
        return (loss, metrics) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """The per-pair ``[pairs, 2]`` (chosen, rejected) rewards, one row per pair so the evaluation gather
        cuts the final round's padding pairs."""
        if self._pp_runner is not None:
            # No pipeline rank holds the full model the path below needs; the mixin drives the
            # schedule instead (loss-only, or loss plus adapter-reduced predictions for metrics).
            return self._pp_prediction_step(inputs, prediction_loss_only, ignore_keys)
        with torch.no_grad(), peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            loss, metrics, rewards = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval", rows=self.eval_split_rows(rewards.size(0)))

        if prediction_loss_only:
            return loss.detach(), None, None

        # On the accelerator device to match labels (a CPU-vs-GPU mismatch breaks the eval gather).
        rewards = rewards.to(self.accelerator.device)
        return loss.detach(), rewards, torch.zeros(rewards.size(0), device=rewards.device)

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float], torch.Tensor]:
        """Metrics are detached rank-local 0-dim tensors, drained at log time. In eval the loss and metrics read the
        split's own pairs; the rewards keep every pair, the shape every rank's evaluation gather expects."""
        prefix = "eval_" if train_eval == "eval" else ""
        real_pairs = self.eval_split_rows(batch["prompt_input_ids"].size(0)) if train_eval == "eval" else None

        model_output = self.concatenated_forward(model, batch, real_pairs)
        current_margin = self._get_current_margin()

        margin_losses, all_chosen_rewards, all_rejected_rewards = self.objective.margin_loss(
            model_output["chosen_logps"], model_output["rejected_logps"], current_margin
        )
        kept_pairs = margin_losses.numel() if real_pairs is None else real_pairs
        chosen_rewards = split_rows_head(all_chosen_rewards, kept_pairs)
        rejected_rewards = split_rows_head(all_rejected_rewards, kept_pairs)

        chosen_sft_loss = model_output["chosen_sft_loss"]
        rejected_sft_loss = model_output["rejected_sft_loss"]
        ratio = self.objective.chosen_sft_ratio
        # No cp_size factor under CP: both terms reach the loss through autograd all-reduces, whose
        # backward sums the cp_size rank-identical copies and so cancels FSDP2's 1/cp_size average.
        total_loss = split_rows_mean(margin_losses, kept_pairs) + (
            ratio * chosen_sft_loss + (1 - ratio) * rejected_sft_loss
        )

        # The logit means and SFT losses already arrive as means over the kept pairs (under CP, CP-global
        # ones): unit divisors report them as they are.
        unit = torch.ones_like(model_output["mean_chosen_logits"])
        sums = metric_sums(
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            chosen_logps=split_rows_head(model_output["chosen_logps"], kept_pairs),
            rejected_logps=split_rows_head(model_output["rejected_logps"], kept_pairs),
            pair_valid=torch.ones_like(chosen_rewards),
            chosen_logits_sum=model_output["mean_chosen_logits"],
            rejected_logits_sum=model_output["mean_rejected_logits"],
            chosen_logit_tokens=unit,
            rejected_logit_tokens=unit,
            chosen_sft_sum=chosen_sft_loss,
            rejected_sft_sum=rejected_sft_loss,
        )
        metrics = metrics_from_sums(
            sums,
            SMPOStepConstants(pair_count=chosen_rewards.numel(), target_margin=current_margin),
            report_margin=self.args.use_margin_schedule,
        )

        rewards = torch.stack([all_chosen_rewards, all_rejected_rewards], dim=1)
        return total_loss, {f"{prefix}{key}": value for key, value in metrics.items()}, rewards

    def concatenated_forward(
        self, model: nn.Module, batch: dict[str, torch.Tensor], real_pairs: int | None = None
    ) -> dict[str, torch.Tensor]:
        """One forward over the chosen ⧺ rejected concat. ``real_pairs`` supervises only the batch's first pairs, since
        the rest pad the split's final round; ``None`` supervises every pair."""
        concat_batch = concatenate_pairs(batch, self.pad_token_id, self.parallelism_config.cp_size)
        num_chosen = concat_batch["num_chosen"]
        device = model.device if hasattr(model, "device") else next(model.parameters()).device

        input_ids = concat_batch["input_ids"].to(device)
        attention_mask = concat_batch["attention_mask"].to(device)
        labels = concat_batch["labels"].to(device)
        if real_pairs is not None and real_pairs < num_chosen:
            pair = torch.arange(labels.size(0), device=device) % num_chosen
            labels = labels.masked_fill((pair >= real_pairs).unsqueeze(1), LABEL_IGNORE_INDEX)

        vision_inputs = {
            key: value.to(device)
            for key, value in concat_batch.items()
            if key not in ("input_ids", "attention_mask", "labels", "num_chosen")
        }

        if self.args.padding_free:
            if vision_inputs:
                raise ValueError(
                    "padding_free received a batch carrying vision tensors "
                    f"({sorted(vision_inputs)}) — VLM SMPO does not support padding_free."
                )
            return self._forward_padding_free(model, input_ids, attention_mask, labels, num_chosen)

        cp_config = self.cp_config
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **vision_inputs).logits

        # Under CP the logits are this rank's chunk of the sequence the full labels span.
        chunk_start = (
            cp_chunk_bounds(labels.size(1), cp_config.cp_rank, cp_config.cp_size)[0] if cp_config is not None else 0
        )
        per_token_logps, loss_mask = next_token_logprobs(logits, labels, chunk_start)
        per_token_logps = per_token_logps.masked_fill(~loss_mask, 0)
        # Pre-clip NLL for the SFT term: it anchors on the true likelihood, not the clipped one.
        per_token_nll = -per_token_logps

        row_is_chosen = torch.arange(per_token_logps.size(0), device=per_token_logps.device) < num_chosen
        per_token_logps = self.objective.clip_log_probs(
            per_token_logps, loss_mask, row_is_chosen.unsqueeze(1), cp_config
        )
        seq_logps = cp_global_mean(per_token_logps.sum(dim=-1), loss_mask.sum(dim=-1), cp_config)
        return _side_statistics(
            seq_logps,
            num_chosen,
            cp_config,
            chosen=(logits[:num_chosen], loss_mask[:num_chosen], per_token_nll[:num_chosen]),
            rejected=(logits[num_chosen:], loss_mask[num_chosen:], per_token_nll[num_chosen:]),
        )

    def _load_policy(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SMPOConfig,
        peft_config: Any | None,
        parallelism_config: ParallelismConfig | None,
        tokenizer: PreTrainedTokenizerBase,
    ) -> nn.Module:
        """Refuses a padding-free run whose kernels cannot keep the flattened rows apart."""
        model = resolve_trainer_model(model, args, parallelism_config=parallelism_config)
        # After the load: the ctor accepts a model id, and the config is where a family that closes
        # turns with a role marker (GLM-4, Gemma) declares it. Resolved once, because the tokenize map
        # runs in worker processes and fingerprints its fn_kwargs.
        self._eos_token_ids = resolve_eos_token_ids(tokenizer, getattr(model, "config", None))
        model, self._peft_has_been_casted_to_bf16 = prepare_policy_model(
            model, peft_config, args, disable_dropout=args.disable_dropout
        )

        if args.padding_free:
            reject_multi_document_rows(
                model.config,
                "padding_free",
                attn_impl=effective_attn_implementation(model.config),
                allow_cross_document_linear_attention=args.allow_cross_document_linear_attention,
                keeps_packed_rows=False,
                varlen_rows=True,
            )
        # position_ids alone keep the padding-free row's documents apart in attention; these are the
        # segment markers this family's conv / linear-attention mixers read on top. A flash backend takes the
        # varlen set too, which every attention layer would otherwise derive from position_ids, each a host read.
        markers = segment_markers_for(model.config)
        self._segment_markers = SegmentMarkers(
            seq_idx=markers.seq_idx,
            cu_seq_lens=markers.cu_seq_lens
            or is_flash_attn_implementation(effective_attn_implementation(model.config)),
        )
        return model

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset | dict[str, Dataset] | None,
        processing_class: PreTrainedTokenizerBase,
        args: SMPOConfig,
        dataset_name: str,
    ) -> Dataset | IterableDataset | dict[str, Dataset] | None:
        """Text mode tokenizes templated strings; VLM mode keeps the prompt as text plus images, so the collator
        processes them per batch instead of persisting pixel tensors into the Arrow cache."""
        if dataset is None:
            return None
        if isinstance(dataset, dict):
            return {key: self._prepare_dataset(split, processing_class, args, key) for key, split in dataset.items()}
        if self.is_vlm:
            prepared_columns, tokenize_fn, row_processing_class = (
                _VLM_COLUMNS,
                tokenize_vlm_preference_row,
                processing_class,
            )
            # The pinned schema also keeps the Image feature, so rows hand back PIL images.
            map_kwargs = {"features": VLM_SMPO_FEATURES}
        else:
            prepared_columns, tokenize_fn, map_kwargs = _TEXT_COLUMNS, tokenize_preference_row, {}
            # The text row fn calls ``processing_class(text, add_special_tokens=False)``, whose second slot a processor
            # reads as images, and reads bos/eos ids a ProcessorMixin does not carry.
            row_processing_class = resolve_tokenizer(processing_class)
        if isinstance(dataset, Dataset):
            missing = [column for column in prepared_columns if column not in dataset.column_names]
            if not missing:
                if is_global_main_process():
                    logger.info(f"Dataset '{dataset_name}' already prepared, skipping _prepare_dataset")
                return dataset
            missing_raw = [column for column in _RAW_PAIR_COLUMNS if column not in dataset.column_names]
            if missing_raw:
                raise ValueError(
                    f"Dataset '{dataset_name}' can be neither used nor tokenized: it lacks the prepared columns "
                    f"{missing} and the raw columns {missing_raw} the tokenize map reads. Pass the raw "
                    "(prompt, chosen, rejected[, images]) rows."
                )

        # Coordinated, not local_main_process_first + raw map: one rank tokenizes to a deterministic
        # cache file and the rest load it. Wrapping this in a main-first block would desync it.
        return coordinated_map(
            dataset,
            tokenize_fn,
            num_proc=resolve_map_num_proc(args.dataset_num_proc),
            remove_columns=dataset.column_names,
            desc=f"Tokenizing {dataset_name} dataset",
            # In fn_kwargs: the map fn must stay picklable, and the cache key fingerprints fn_kwargs.
            fn_kwargs={
                "processing_class": row_processing_class,
                "max_prompt_length": self.max_prompt_length,
                "max_completion_length": self.max_completion_length,
                "truncation_mode": args.truncation_mode,
                "eos_token_ids": self._eos_token_ids,
            },
            **map_kwargs,
        )

    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            self._signature_columns = list(_VLM_COLUMNS if self.is_vlm else _TEXT_COLUMNS)
        # The dataloader mixin's union of required_dataset_columns sits below this override in the MRO; unchained, a
        # collator input would be pruned here alone.
        super()._set_signature_columns_if_needed()

    def _forward_padding_free(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        num_chosen: int,
    ) -> dict[str, torch.Tensor]:
        """The concat flattened to ``[1, total_non_pad_tokens]``: each row becomes one document whose
        ``position_ids`` restart at 0, with the family's segment markers for mixers that read them."""
        flat_input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
        flat_labels = labels[attention_mask.bool()].unsqueeze(0)
        position_ids = flattened_document_positions(attention_mask)

        flat_logits = model(
            input_ids=flat_input_ids,
            position_ids=position_ids,
            use_cache=False,
            **segment_marker_kwargs(position_ids, self._segment_markers),
        ).logits

        token_logps, loss_mask = token_logprobs(flat_logits, flat_labels)
        # A document's first token is not predicted from inside its document.
        flat_mask = loss_mask[0] & (position_ids[0, 1:] != 0)
        flat_logps = token_logps[0].masked_fill(~flat_mask, 0)
        flat_nll = -flat_logps

        shift_seq_idx = document_ids(position_ids[0])[1:]
        is_chosen = shift_seq_idx < num_chosen
        flat_logps = self.objective.clip_log_probs(flat_logps, flat_mask, is_chosen)

        # Scatter-adds over the masked values: nonzero() would sync.
        num_seqs = 2 * num_chosen
        logp_sums = flat_logps.new_zeros(num_seqs).scatter_add_(0, shift_seq_idx, flat_logps)
        token_counts = flat_logps.new_zeros(num_seqs).scatter_add_(0, shift_seq_idx, flat_mask.to(flat_logps.dtype))
        # padding_free is refused under CP, so these means are rank-local.
        seq_logps = cp_global_mean(logp_sums, token_counts, None)

        shift_logits_flat = flat_logits[0, :-1]
        chosen_valid, rejected_valid = flat_mask & is_chosen, flat_mask & ~is_chosen
        return _side_statistics(
            seq_logps,
            num_chosen,
            None,
            chosen=(shift_logits_flat, chosen_valid, flat_nll * chosen_valid),
            rejected=(shift_logits_flat, rejected_valid, flat_nll * rejected_valid),
        )

    def _get_current_margin(self) -> float:
        """Written by the scheduler callback each step. No default: ``__init__`` seeds ``target_margin`` on the model
        it hands the base Trainer, so a miss means that seeding was skipped."""
        if not self.args.use_margin_schedule:
            return self.args.target_margin
        margin = getattr(self.model, "target_margin", None)
        return margin if margin is not None else self._top_level_model().target_margin

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        self._pp_loss = SMPOPipelineLoss(
            objective=self.objective,
            pad_token_id=self.pad_token_id,
            max_length=self.args.max_length,
            current_margin=self._get_current_margin,
            report_margin=self.args.use_margin_schedule,
            constants=SMPOStepConstants(target_margin=self.args.target_margin),
        )
        return self._pp_loss.build()
