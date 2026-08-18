# Copyright 2024 White Circle
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline GRPO trainer (pre-computed rewards) under EP / TP / PP.

CP unsupported: relies on ``logits_to_keep``, which CP's sequence splitting breaks.

PP runs every loss type (grpo / bnpo / dr_grpo), whose whole-batch denominators are computable from
batch metadata alone, and the KL term at any ``kl_beta``: no pipeline rank holds a full model to
forward a reference, so the reference log-probs are scored once before the first optimizer step by a
forward-only sweep through the pipeline and carried into every step as a per-example side tensor.
See ``_pp_loss_adapter``.
"""

import warnings
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import Any, Union, get_args

import datasets
import numpy as np
import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from accelerate.utils import is_peft_model
from peft import PeftConfig
from scipy import stats
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    IterableDataset,
)
from transformers import (
    AutoTokenizer,
    BaseImageProcessor,
    EvalPrediction,
    FeatureExtractionMixin,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from trl import create_reference_model
from trl.trainer.utils import (
    disable_dropout_in_model,
    selective_log_softmax,
)

from src.callbacks.variable_scheduler import VariableSchedulerCallback
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN, OfflineGRPODataCollatorWithPadding
from src.data.pipeline.processing import coordinated_map
from src.data.pipeline.rendered import probe_tokenizer_specials
from src.data.spans import LABEL_IGNORE_INDEX, ends_with_terminator, resolve_eos_token_ids
from src.distributed.loading.model_loading import load_model_from_pretrained
from src.distributed.loading.peft_setup import peft_bf16_autocast, prepare_peft_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.losses import (
    PPLossAdapter,
    completion_labels,
    loss_token_counts_per_row,
    pad_to_pipeline_length,
    rows_with_labels,
    token_logprobs,
)
from src.distributed.runtime import current_device, get_global_world_size
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.model_preparation import resolve_auto_model_class
from src.models.loading.tokenizer_setup import is_bounded_length
from src.models.modality import config_declares_multimodality
from src.models.structure import resolve_tokenizer
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedLogprobsCore
from src.trainers.grpo.mixins.dataloader import MultiGroupSampler
from src.trainers.grpo.objective.advantages import STD_EPS
from src.trainers.grpo.objective.offline import offline_token_objective
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.ep_introspection import named_ep_layers
from src.trainers.mixins.pp_gates import reject_pp_compute_metrics, reject_pp_peft

logger = get_logger(__name__, log_level="INFO")

# Derived from the config's own Literal annotations, so the trainer's dispatch ladders and the
# parse-time gate can never drift apart.
LOSS_TYPES: tuple[str, ...] = get_args(OfflineGRPOConfig.__annotations__["loss_type"])
PG_FORMULATIONS: tuple[str, ...] = get_args(OfflineGRPOConfig.__annotations__["policy_gradient_formulation"])


def compute_group_advantages(
    rewards_list: list[float],
    method: str,
    best_completion_emphasis: float | str,
) -> list[float]:
    """Advantages from a group's rewards via ``method`` (z_norm, minmax, quantile_norm,
    quantile_uniform, robust), with optional best-completion emphasis, clipped to [-10, 10].
    """
    # float64 explicitly: integral rewards (0/1 verifiable) would keep an int64 dtype through the
    # `np.zeros_like` degenerate-group branches, and the in-place emphasis multiply below then raises
    # UFuncTypeError inside datasets.map.
    rewards_array = np.asarray(rewards_list, dtype=np.float64)
    if not np.all(np.isfinite(rewards_array)):
        # Every method divides by a spread derived from these rewards, so one NaN/Inf reaches the
        # whole group's advantages and from there the micro-batch gradient.
        raise ValueError(
            f"Non-finite reward in a completion group: {rewards_list}. Fix the reward column — "
            f"training through it silently either zeroes the row's advantage or NaNs the batch, "
            f"depending only on the group size."
        )

    if method == "z_norm":
        reward_mean = np.mean(rewards_array)
        # ddof=1 explicitly: numpy defaults to 0 while torch's .std() is correction=1, and the online
        # and environmental z-norms take the torch path; an implicit default would split the two by
        # sqrt((n-1)/n).
        reward_std = np.std(rewards_array, ddof=1) if len(rewards_array) > 1 else 1.0
        advantages = (rewards_array - reward_mean) / (reward_std + STD_EPS)

    elif method == "minmax":
        reward_min = np.min(rewards_array)
        reward_max = np.max(rewards_array)
        if reward_max == reward_min:
            advantages = np.zeros_like(rewards_array)
        else:
            advantages = 2 * (rewards_array - reward_min) / (reward_max - reward_min) - 1

    elif method == "quantile_norm":
        ranks = stats.rankdata(rewards_array)
        # (ranks - 0.5)/n → uniform [0,1] avoiding exact boundaries, then to normal
        uniform_scores = (ranks - 0.5) / len(ranks)
        advantages = stats.norm.ppf(uniform_scores)

    elif method == "quantile_uniform":
        # A single or all-equal group has no spread to rank, and n-1 == 0 would divide by zero.
        if len(rewards_array) == 1 or np.all(rewards_array == rewards_array[0]):
            advantages = np.zeros(len(rewards_array))
        else:
            ranks = stats.rankdata(rewards_array)
            uniform_scores = (ranks - 1) / (len(ranks) - 1)
            advantages = 2 * uniform_scores - 1

    elif method == "robust":
        q75, q25 = np.percentile(rewards_array, [75, 25])
        iqr = q75 - q25
        median = np.median(rewards_array)
        advantages = np.zeros_like(rewards_array) if iqr == 0 else (rewards_array - median) / iqr

    else:
        raise ValueError(f"Unknown advantage method: {method}")

    if len(rewards_array) > 1:
        if best_completion_emphasis == "auto":
            # Scale emphasis with std: 3.0 at std=0 → 5.0 at std→∞. Population std (numpy's default),
            # unlike the z_norm divisor above: this heuristic has no torch counterpart to match, and
            # it multiplies the best row under every method, so aligning it here would also move the
            # advantages of the rank/minmax methods.
            reward_std = np.std(rewards_array)
            emphasis_factor = 3.0 + 2.0 * reward_std / (1.0 + reward_std)
        else:
            emphasis_factor = float(best_completion_emphasis)

        if emphasis_factor > 1.0:
            max_reward = np.max(rewards_array)
            best_mask = rewards_array == max_reward
            advantages[best_mask] *= emphasis_factor

    advantages = np.clip(advantages, -10.0, 10.0)

    return advantages.tolist()


def tokenize_prompt_completion(
    prompt: str,
    completion: str,
    processing_class: Any,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    is_encoder_decoder: bool,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, list[int]]:
    """Tokenize one stored ``(prompt, completion)`` rollout under the two length budgets."""
    batch = {}

    tokenizer = resolve_tokenizer(processing_class)
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    # Left-truncate the prompt (keep tokens nearest the completion — TRL convention).
    prompt_input_ids = processing_class(prompt, add_special_tokens=False)["input_ids"]
    if is_bounded_length(max_prompt_length):
        prompt_input_ids = prompt_input_ids[-max_prompt_length:]
    # truncation must follow the cap: HF resolves `truncation=True, max_length=None` against
    # `tokenizer.model_max_length`, so an unset cap would truncate at whatever the run pinned there
    # instead of meaning "no cap".
    completion_input_ids = processing_class(
        completion,
        truncation=is_bounded_length(max_completion_length),
        max_length=max_completion_length,
        add_special_tokens=False,
    )["input_ids"]

    # BOS only when the tokenizer's own post-processor emits one (gpt-oss/Bailing define one it
    # never emits), and only within the prompt budget — the caps bound PP's fixed P2P shape.
    if (
        bos_token_id is not None
        and probe_tokenizer_specials(tokenizer).adds_leading_bos
        and (not prompt_input_ids or prompt_input_ids[0] != bos_token_id)
        and (not is_bounded_length(max_prompt_length) or len(prompt_input_ids) < max_prompt_length)
    ):
        prompt_input_ids = [bos_token_id] + prompt_input_ids

    if is_encoder_decoder and bos_token_id is not None:
        completion_input_ids = [bos_token_id] + completion_input_ids if completion_input_ids else [bos_token_id]

    # EOS only within the budget: supervising it at a truncation cut teaches premature stopping. Any
    # declared terminator counts as already-ended, not just tokenizer.eos_token_id — GLM-4 and Gemma
    # close turns with a role marker the config lists instead, and testing the single id there appends
    # a second ender the policy never emits.
    terminators = eos_token_ids or ({eos_token_id} if eos_token_id is not None else set())
    if eos_token_id is not None and not completion_input_ids:
        completion_input_ids = [eos_token_id]
    elif (
        eos_token_id is not None
        and not ends_with_terminator(completion_input_ids, tokenizer, terminators)
        and (not is_bounded_length(max_completion_length) or len(completion_input_ids) < max_completion_length)
    ):
        completion_input_ids = completion_input_ids + [eos_token_id]

    batch["prompt_input_ids"] = prompt_input_ids
    batch["completion_input_ids"] = completion_input_ids

    return batch


def tokenize_offline_grpo_rows(
    batch: dict[str, list],
    indices: list[int],
    *,
    processing_class: Any,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    advantage_method: str,
    best_completion_emphasis: float | str,
    is_encoder_decoder: bool,
    eos_token_ids: frozenset[int] = frozenset(),
    drop_degenerate_groups: bool = False,
) -> dict[str, list]:
    """Tokenize a batch, expanding each ``(prompt, completions, rewards)`` group into one
    training example per completion (with its computed advantage and group metadata).
    """
    all_prompt_input_ids = []
    all_completion_input_ids = []
    all_group_ids = []
    all_group_sizes = []
    all_advantages = []

    for prompt, completions_list, rewards_list, idx in zip(
        batch["prompt"], batch["completions"], batch["rewards"], indices, strict=True
    ):
        if len(completions_list) != len(rewards_list):
            raise ValueError(
                f"Offline GRPO row {idx}: {len(completions_list)} completions but "
                f"{len(rewards_list)} rewards — each completion needs exactly one reward "
                f"(silent zip truncation would train on mispaired advantages and record a "
                f"wrong group_size)."
            )
        # Exact ties only: every advantage method maps an exactly-tied (or singleton) group to
        # all-zero advantages, so its rows carry no gradient yet dilute the loss normalizer. A
        # near-tie still trains with full-scale advantages under the rank methods, so a spread
        # threshold would change the objective. Dropping at tokenization rather than in the loss also
        # skips those rows' forwards and holds across every mode, PP's metadata denominators included.
        if drop_degenerate_groups and (len(rewards_list) < 2 or max(rewards_list) == min(rewards_list)):
            continue
        advantages_list = compute_group_advantages(rewards_list, advantage_method, best_completion_emphasis)

        for completion, advantage in zip(completions_list, advantages_list, strict=True):
            tokenized = tokenize_prompt_completion(
                prompt,
                completion,
                processing_class,
                max_prompt_length=max_prompt_length,
                max_completion_length=max_completion_length,
                is_encoder_decoder=is_encoder_decoder,
                eos_token_ids=eos_token_ids,
            )

            all_prompt_input_ids.append(tokenized["prompt_input_ids"])
            all_completion_input_ids.append(tokenized["completion_input_ids"])
            all_group_ids.append(idx)
            all_group_sizes.append(len(completions_list))
            all_advantages.append(advantage)

    return {
        "prompt_input_ids": all_prompt_input_ids,
        "completion_input_ids": all_completion_input_ids,
        "group_id": all_group_ids,
        "group_size": all_group_sizes,
        "advantage": all_advantages,
    }


def live_min_log_prob(model: nn.Module, configured: float | None) -> float | None:
    """The negative-advantage log-prob floor in force right now (``None`` = no clipping configured).

    ``VariableSchedulerCallback`` writes the scheduled value onto the unwrapped module (the trainer's
    ``self.model``), so the DDP/FSDP-wrapped module handed to ``compute_loss`` never carries it and
    reading the floor off that wrapper would pin step 0's value for the whole run.
    """
    return getattr(model, "min_log_prob", configured)


class OfflineGRPOTrainer(ChunkedLogprobsCore, DistributedTrainerMixin, Trainer):
    """Offline GRPO trainer for pre-computed-reward data (``prompt``/``completions``/``rewards``)
    under EP / TP / PP (see ``_reject_pp_explicit_options`` for the PP-specific rejections)."""

    _tag_names = ["trl", "offline-grpo"]

    # Single-forward objective whose denominators come from batch metadata alone; the KL reference
    # is scored by a pipeline sweep before training (``_pp_precompute_reference_logps``).
    _supports_pp = True
    # The objective never passes labels into the forward, so Liger CE/FLCE cannot fire.
    _loss_outside_model_forward = True
    # The objective divides by its own loss-type denominator, so the parent's grad-accum scaling stands.
    _loss_is_own_mean = True
    # Non-None only while the PP reference sweep drives the schedule: the last stage's loss stashes
    # its per-token log-probs here instead of computing the objective.
    _pp_ref_sweep: list[torch.Tensor] | None = None

    # Buffered per-sample key → logged sign prefixes (unclamped: negative side only, where it binds).
    _SIGN_METRIC_KEYS = {
        "logps": ("positive", "negative"),
        "rewards": ("positive", "negative"),
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
        parallelism_config: "ParallelismConfig" = None,
        save_sharded_ep: bool = False,
        **kwargs,
    ):
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._sign_metric_buffer = {"train": defaultdict(list), "eval": defaultdict(list)}

        # A VLM ProcessorMixin keeps token ids on its inner tokenizer, not on itself.
        tokenizer = resolve_tokenizer(processing_class)
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.advantage_method = args.advantage_method
        self.best_completion_emphasis = args.best_completion_emphasis
        self.min_log_prob = args.min_log_prob
        self.loss_type = args.loss_type
        # Every loss/PG dispatch in this file (compute_loss, _pp_normalizer, _pp_token_loss) branches
        # on these strings; validate once at construction, not per microbatch or PP-only. Each
        # three-way ``loss_type`` ladder still ends in a raise, since a fallthrough would substitute a
        # different denominator instead of failing; the two-way PG choice ends in a labelled else.
        if self.loss_type not in LOSS_TYPES:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}. Supported types: {list(LOSS_TYPES)}")
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
        self.policy_gradient_formulation = args.policy_gradient_formulation
        logger.info(f"Using loss type: {self.loss_type}")
        logger.info(f"Using PG formulation: {self.policy_gradient_formulation}")

        # Read by ChunkedLogprobsCore (avoids full [B,T,vocab] logits). Inert under PP: the pipeline
        # drives stages through the PP loss adapter and never calls _get_per_token_logps (the last
        # stage still materializes its own logits plane — see _pp_loss_adapter).
        self._use_chunked_grpo_logprobs = args.use_chunked_grpo_logprobs
        # Stored completions carry no sampling distribution to match, so log-probs are plain
        # (the full-logits path's implicit temperature); the chunked kernel divides by this.
        self.temperature = 1.0

        self._reject_inert_max_length(args, parallelism_config)
        self._reject_pp_explicit_options(args, parallelism_config, peft_config, compute_metrics)

        model, model_id = load_model_from_pretrained(model, args)

        # QLoRA's bf16 adapter cast leaves activations fp32, so every adapted linear's forward needs
        # the autocast region compute_loss opens.
        self._peft_has_been_casted_to_bf16 = False
        if peft_config is not None:
            model, self._peft_has_been_casted_to_bf16 = prepare_peft_model(model, peft_config, args)

        self.is_encoder_decoder = model.config.is_encoder_decoder
        # The canonical predicate, not a raw ITT-mapping membership test: mistral4's ITT entry is a
        # text-only quirk, and a remote-code VLM declaring vision_config is in no mapping at all.
        self.is_vision_model = config_declares_multimodality(model.config)

        if self.is_vision_model:
            warnings.warn(
                "Vision models are not fully supported in OfflineGRPOTrainer (no pixel_values)",
                UserWarning,
                stacklevel=2,
            )

        self.beta = args.kl_beta
        if args.disable_dropout:
            disable_dropout_in_model(model)

        # The model rides through the distributed seam so PP can split it into this rank's stage.
        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            model=model,
        )
        model = dist_kwargs.pop("model")

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(model_id, padding_side="right")

        data_collator = OfflineGRPODataCollatorWithPadding(
            pad_token_id=self.padding_value,
        )

        if args.remove_unused_columns:
            args.remove_unused_columns = False
            warnings.warn(
                "OfflineGRPODataCollatorWithPadding requires remove_unused_columns=False (it reads "
                "prompt/completion/reward columns the Trainer would otherwise strip); forcing it off. "
                "Set it explicitly in your OfflineGRPOConfig.",
                UserWarning,
                stacklevel=2,
            )

        original_columns = train_dataset.column_names

        tokenize_num_proc = self._dataset_map_num_proc(args.dataset_num_proc)
        # In fn_kwargs, not read off self: the map fn must stay picklable, and the cache key
        # fingerprints fn_kwargs.
        tokenize_kwargs = {
            "processing_class": processing_class,
            "max_prompt_length": self.max_prompt_length,
            "max_completion_length": self.max_completion_length,
            "advantage_method": self.advantage_method,
            "best_completion_emphasis": self.best_completion_emphasis,
            "is_encoder_decoder": self.is_encoder_decoder,
            "eos_token_ids": resolve_eos_token_ids(
                resolve_tokenizer(processing_class), getattr(model, "config", None)
            ),
            "drop_degenerate_groups": args.drop_degenerate_groups,
        }

        num_train_groups_in = len(train_dataset) if args.drop_degenerate_groups else 0
        train_dataset = coordinated_map(
            train_dataset,
            tokenize_offline_grpo_rows,
            num_proc=tokenize_num_proc,
            remove_columns=original_columns,
            desc="Tokenizing and expanding training dataset",
            batched=True,
            batch_size=32,
            with_indices=True,
            fn_kwargs=tokenize_kwargs,
        )
        self._check_degenerate_drop(args, train_dataset, num_train_groups_in, "training")
        if eval_dataset is not None:
            eval_original_columns = eval_dataset.column_names
            num_eval_groups_in = len(eval_dataset) if args.drop_degenerate_groups else 0
            eval_dataset = coordinated_map(
                eval_dataset,
                tokenize_offline_grpo_rows,
                num_proc=tokenize_num_proc,
                remove_columns=eval_original_columns,
                desc="Tokenizing and expanding evaluation dataset",
                batched=True,
                batch_size=32,
                with_indices=True,
                fn_kwargs=tokenize_kwargs,
            )
            self._check_degenerate_drop(args, eval_dataset, num_eval_groups_in, "evaluation")

        if self.beta == 0.0 or is_peft_model(model) or self.parallelism_config.is_pp_mode:
            # PEFT: no separate ref — disable_adapter() reverts to base (EP-aware via
            # _setup_distributed_modes). PP: the reference is scored through the pipeline once the
            # runtime exists (_pp_precompute_reference_logps), so no full model sits beside a stage.
            self.ref_model = None
        else:
            self.ref_model = self._create_full_ft_reference(model, model_id)
            if args.disable_dropout:
                disable_dropout_in_model(self.ref_model)

        if args.initial_min_log_prob is not None and args.min_log_prob is not None:
            model.min_log_prob = args.initial_min_log_prob
            logger.info(f"Initialized model.min_log_prob = {args.initial_min_log_prob} for scheduling")

            min_log_prob_scheduler = VariableSchedulerCallback(
                attribute_name="min_log_prob",
                initial_value=args.initial_min_log_prob,
                final_value=args.min_log_prob,
                schedule_type="linear",
            )

            if callbacks is None:
                callbacks = []
            callbacks.append(min_log_prob_scheduler)
            logger.info(f"Added min_log_prob scheduler: {args.initial_min_log_prob} → {args.min_log_prob}")
        elif args.initial_min_log_prob is not None and args.min_log_prob is None:
            logger.warning("initial_min_log_prob is set but min_log_prob is None. Scheduler will not be added.")
        elif args.initial_min_log_prob is None and args.min_log_prob is not None:
            logger.info(f"Using constant min_log_prob = {args.min_log_prob} (no scheduling)")
        else:
            logger.info("No min_log_prob clipping enabled")

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

        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if self.accelerator.is_main_process:
            logger.info("Extracting group_ids from training dataset...")
        self._cached_train_group_ids = list(train_dataset["group_id"])
        if self.accelerator.is_main_process:
            logger.info(
                f"Training dataset info: {len(train_dataset)} examples, {len(set(self._cached_train_group_ids))} unique groups"
            )

        if eval_dataset is not None:
            if self.accelerator.is_main_process:
                logger.info("Extracting group_ids from evaluation dataset...")
            self._cached_eval_group_ids = list(eval_dataset["group_id"])
            if self.accelerator.is_main_process:
                logger.info(
                    f"Eval dataset info: {len(eval_dataset)} examples, {len(set(self._cached_eval_group_ids))} unique groups"
                )
        else:
            self._cached_eval_group_ids = None

        self._setup_distributed_modes()

        if self._pp_runtime is not None and self.beta != 0.0:
            self.train_dataset = self._pp_precompute_reference_logps(self.train_dataset, "training")
            if self.eval_dataset is not None:
                self.eval_dataset = self._pp_precompute_reference_logps(self.eval_dataset, "evaluation")

    @staticmethod
    def _reject_inert_max_length(args: OfflineGRPOConfig, parallelism_config: "ParallelismConfig | None") -> None:
        """Refuse ``max_length`` off pipeline parallelism, the only mode that reads it.

        Off PP the run's window is ``max_prompt_length + max_completion_length``, the two caps the
        tokenization applies; ``max_length`` reaches nothing there, so setting it would read as a
        sequence cap while changing nothing.
        """
        if parallelism_config is not None and parallelism_config.is_pp_mode:
            return
        if args.max_length is None:
            return
        raise ValueError(
            f"max_length={args.max_length!r} is a pipeline-parallel-only knob (the fixed shape every "
            f"batch is padded to), and this run has no pipeline parallelism — nothing would read it. "
            f"The offline-GRPO budget is two caps: set max_prompt_length (prompt left-truncation) "
            f"and/or max_completion_length (completion truncation) instead, and remove max_length."
        )

    def _reject_pp_explicit_options(
        self, args: OfflineGRPOConfig, parallelism_config: "ParallelismConfig | None", peft_config, compute_metrics
    ) -> None:
        """Reject offline-GRPO options pipeline parallelism cannot honor, at construction.

        Its own gate rather than the mixin's ``_validate_pp_mode`` hook: ``peft_config`` and
        ``compute_metrics`` are explicit ctor parameters the mixin never sees, and the rejections
        must precede the in-ctor model load rather than follow it.
        """
        if parallelism_config is None or not parallelism_config.is_pp_mode:
            return
        if args.use_chunked_grpo_logprobs:
            logger.warning(
                "use_chunked_grpo_logprobs has NO EFFECT under pipeline parallelism: the pipeline drives "
                "stages through the PP loss adapter and never calls _get_per_token_logps, so the last "
                "stage still materializes its own [rows, max_length, vocab] logits plane. Lower "
                "per_device_train_batch_size or raise pipeline_parallel_size to fit it instead."
            )
        reject_pp_compute_metrics(
            compute_metrics,
            (
                "this trainer's own "
                "prediction_step returns the loss and NO predictions (loss, None, None) in every "
                "mode, so compute_metrics is never called off-PP either — there is no prediction "
                "convention for the pipeline to reproduce. Serving it under PP would broadcast the "
                "last stage's raw [rows, max_length, vocab] logits to every rank in the chain and "
                "feed metrics a tensor no non-PP run ever produces. Drop compute_metrics; the "
                "objective's own statistics are logged as metrics already."
            ),
        )
        reject_pp_peft(peft_config, explicit_param_trainer="OfflineGRPOTrainer")
        if not is_bounded_length(self.max_prompt_length) or not is_bounded_length(self.max_completion_length):
            raise ValueError(
                f"Pipeline parallelism requires bounded sequences (the P2P buffer shapes freeze on "
                f"the first step): set both max_prompt_length (got {self.max_prompt_length!r}) and "
                f"max_completion_length (got {self.max_completion_length!r}) to positive values."
            )
        fixed_length = self.max_prompt_length + self.max_completion_length
        # Unset max_length falls back to the caps' sum: the fixed shape is what bounds P2P buffers.
        if args.max_length is None or args.max_length <= 0:
            args.max_length = fixed_length
        elif args.max_length < fixed_length:
            raise ValueError(
                f"max_length={args.max_length} is below max_prompt_length + max_completion_length "
                f"({fixed_length}): a batch at the caps would not fit the pipeline's fixed shape, "
                f"and truncating there would silently drop loss tokens."
            )

    @staticmethod
    def _check_degenerate_drop(args: OfflineGRPOConfig, dataset, num_groups_in: int, what: str) -> None:
        """Report ``drop_degenerate_groups``' effect and refuse an all-degenerate dataset.

        An empty dataset would otherwise surface steps later as an opaque dataloader error, and a
        100% drop is indistinguishable from a mis-scored dataset.
        """
        if not args.drop_degenerate_groups:
            return
        num_groups_kept = len(set(dataset["group_id"])) if len(dataset) else 0
        logger.info(
            f"drop_degenerate_groups: kept {num_groups_kept}/{num_groups_in} {what} groups "
            f"(dropped groups scored all-alike or held < 2 completions)"
        )
        if num_groups_kept == 0:
            raise ValueError(
                f"drop_degenerate_groups removed every {what} group: all groups scored all-alike "
                f"rewards or held fewer than two completions. Such data carries no group-relative "
                f"signal — fix the rewards (or disable drop_degenerate_groups to train on "
                f"zero-advantage rows anyway)."
            )

    @staticmethod
    def _create_full_ft_reference(model, model_id: str):
        """KL reference for full-finetune runs.

        Plain dense models deepcopy the live policy (TRL's ``create_reference_model`` — on resume
        this re-anchors the KL to the resumed weights, TRL semantics). EP / grouped-GEMM wrapped MoE
        models hold live NCCL process groups that ``deepcopy`` cannot pickle, so the reference
        reloads dense from ``model_id`` instead — anchoring to the checkpoint weights (also on
        resume), with the policy's attention implementation so logprobs share sink/backend semantics.
        """
        if not named_ep_layers(model):
            return create_reference_model(model)
        if not model_id:
            raise ValueError(
                "kl_beta > 0 with a wrapped MoE model needs the checkpoint path to load a dense "
                "KL reference (deepcopy cannot pickle the EP process groups). Pass the model by "
                "path, use PEFT, or set kl_beta: 0."
            )
        logger.info(f"Loading dense KL reference from {model_id} (wrapped MoE cannot be deep-copied)")
        # Same Auto class the policy resolved: a multimodal MoE is registered under
        # AutoModelForImageTextToText, and AutoModelForCausalLM would drop its vision tower, giving a
        # reference that reads different inputs than the policy. Revision pinned as the policy pins
        # it (hub main can drift from the trained commit's format). Coverage-gated: a reference
        # loaded with missing keys is part random weights, and a wrong KL reference biases every step
        # with no other symptom. Remote code has already run in-process.
        return from_pretrained_verified(
            resolve_auto_model_class(model.config),
            model_id,
            dtype=model.dtype,
            attn_implementation=model.config._attn_implementation,
            revision=getattr(model.config, "_commit_hash", None) or "main",
            trust_remote_code=True,
        )

    def _build_grouped_dataloader(self, dataset, group_ids, *, batch_size: int, shuffle: bool) -> DataLoader:
        """Build a DataLoader over a ``MultiGroupSampler`` sharded by DP rank/size.

        Shards by DP (orthogonal to EP) so TP siblings see the same batch. Per-rank batch counts are
        equalized to the global minimum, else faster ranks exit and block the next all-reduce. The
        sampler already shards, so accelerate does device placement only.

        A pre-sharded dataset is already this rank's disjoint slice, so the sampler must not cut it
        again; doing so would drop (dp-1)/dp of the data.
        """
        dp_size, dp_rank = self.dp_shard_geometry()

        dataloader_params = {
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(dataset, torch.utils.data.IterableDataset):
            sampler = MultiGroupSampler(
                group_ids=group_ids,
                rank=dp_rank,
                world_size=dp_size,
                shuffle=shuffle,
                # HF's convention: data_seed decouples data order from the global seed when set.
                seed=(self.args.data_seed if self.args.data_seed is not None else self.args.seed) if shuffle else 0,
            )
            if get_global_world_size() > 1:
                # compute_loss gathers every batch: a rank with an extra batch blocks its peers.
                t = torch.tensor([len(sampler) // batch_size], device=current_device())
                dist.all_reduce(t, op=dist.ReduceOp.MIN)
                min_batches = int(t.item())
                if min_batches == 0:
                    raise ValueError(
                        f"Batch-count equalization left 0 full batches on the smallest rank "
                        f"(batch_size={batch_size}, world min over ranks) — the dataset split is too "
                        f"small for this world size; eval/training would silently run on nothing."
                    )
                if min_batches * batch_size < len(sampler.indices_sequence):
                    sampler.indices_sequence = sampler.indices_sequence[: min_batches * batch_size]
            dataloader_params["batch_sampler"] = BatchSampler(
                sampler=sampler,
                batch_size=batch_size,
                drop_last=True,
            )
            if self.args.dataloader_num_workers > 0:
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker, num_workers=self.args.dataloader_num_workers, rank=dp_rank
                )
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor
        else:
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(dataset, **dataloader_params)

    def get_train_dataloader(self) -> DataLoader:
        """Training DataLoader (see ``_build_grouped_dataloader``)."""
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        dataloader = self._build_grouped_dataloader(
            self.train_dataset,
            self._cached_train_group_ids,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=True,
        )
        return self._prepare_dataloader(dataloader, num_processes=1, process_index=0)

    def get_eval_dataloader(self, eval_dataset: str | Dataset | None = None) -> DataLoader:
        """Evaluation DataLoader (see ``_build_grouped_dataloader``; no shuffle, deterministic)."""

        def build(dataset):
            group_ids = self._cached_eval_group_ids
            if group_ids is None and dataset is not None and "group_id" in dataset.column_names:
                group_ids = list(dataset["group_id"])
            eval_dataloader = self._build_grouped_dataloader(
                dataset,
                group_ids,
                batch_size=self.args.per_device_eval_batch_size,
                shuffle=False,
            )
            # Already sharded by the sampler; num_processes=1 avoids accelerator re-sharding.
            return self._prepare_dataloader(eval_dataloader, num_processes=1, process_index=0)

        return self._cached_eval_dataloader(eval_dataset, build)

    def _get_last_hidden_state(self, unwrapped_model, input_ids, attention_mask, logits_to_keep):
        """Backbone hidden states for the last ``logits_to_keep`` completion positions (text-only).

        The offline counterpart of TRL's ``GRPOTrainer._get_last_hidden_state``, consumed by
        ``ChunkedLogprobsCore``: forward the backbone (no ``lm_head``), drop the final position (the
        causal next-token shift), keep the last ``logits_to_keep``.
        """
        if is_peft_model(unwrapped_model):
            unwrapped_model = unwrapped_model.base_model.model
        backbone = unwrapped_model.base_model
        hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return hidden[:, :-1, :][:, -logits_to_keep:, :]  # (B, logits_to_keep, H)

    def _get_per_token_logps(
        self,
        model,
        input_ids,
        attention_mask,
        logits_to_keep,
        advantages=None,
        min_log_prob=None,
    ):
        """Per-token completion log-probs; full logits restricted with ``logits_to_keep``, or never
        materialized at all under ``use_chunked_grpo_logprobs``.

        Returns ``(clamped, unclamped)``. ``min_log_prob`` clamping is applied only to examples
        with negative ``advantages`` (or all examples when ``advantages`` is None).
        """
        if self._use_chunked_grpo_logprobs:
            selected_logps, _ = self._chunked_logps(model, input_ids, attention_mask, logits_to_keep)
        else:
            # +1 because the last logit (next-token pred) is excluded below.
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[:, :-1, :]  # (B, L-1, V)

            input_ids = input_ids[:, -logits_to_keep:]
            logits = logits[:, -logits_to_keep:]  # restrict to completion tokens
            selected_logps = selective_log_softmax(logits, input_ids)

        if min_log_prob is not None and advantages is not None:
            negative_advantage_mask = (advantages < 0).unsqueeze(1)
            clamped_logps = torch.where(
                negative_advantage_mask,
                torch.clamp(selected_logps, min=min_log_prob),
                selected_logps,
            )
            return clamped_logps, selected_logps
        elif min_log_prob is not None:
            return torch.clamp(selected_logps, min=min_log_prob), selected_logps
        else:
            return selected_logps, selected_logps

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("OfflineGRPOTrainer does not support returning outputs")

        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            return self._compute_loss_inner(model, inputs)

    def _compute_loss_inner(self, model, inputs):
        prompt_ids, prompt_mask = (
            inputs["prompt_input_ids"],
            inputs["prompt_attention_mask"],
        )
        completion_ids, completion_mask = (
            inputs["completion_input_ids"],
            inputs["completion_attention_mask"],
        )
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        advantages = inputs["advantage"]  # [B], pre-computed at tokenization (no gather needed)

        current_min_log_prob = live_min_log_prob(self.model, self.min_log_prob)
        per_token_logps, per_token_logps_unclamped = self._get_per_token_logps(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            advantages=advantages,
            min_log_prob=current_min_log_prob,
        )

        ref_per_token_logps = ref_per_token_logps_unclamped = None
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps, ref_per_token_logps_unclamped = self._get_per_token_logps(
                        self.ref_model,
                        input_ids,
                        attention_mask,
                        logits_to_keep,
                        advantages=advantages,
                        min_log_prob=current_min_log_prob,
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps, ref_per_token_logps_unclamped = self._get_per_token_logps(
                            self.model,
                            input_ids,
                            attention_mask,
                            logits_to_keep,
                            advantages=advantages,
                            min_log_prob=current_min_log_prob,
                        )
        per_token_loss, sample_values = offline_token_objective(
            per_token_logps,
            per_token_logps_unclamped,
            advantages,
            policy_gradient_formulation=self.policy_gradient_formulation,
            beta=self.beta,
            ref_logps=ref_per_token_logps,
            ref_logps_unclamped=ref_per_token_logps_unclamped,
        )  # [B, T]

        # 1/group_size so every original group contributes equally.
        group_weights = 1.0 / inputs["group_size"].float()  # [B]

        weighted_per_token_loss = per_token_loss * group_weights.unsqueeze(1)  # [B, T]
        weighted_completion_mask = completion_mask.float() * group_weights.unsqueeze(1)  # [B, T]

        if self.loss_type == "grpo":
            per_sequence_loss = (weighted_per_token_loss * completion_mask).sum(1) / completion_mask.sum(dim=1).clamp(
                min=1.0
            )  # [B]
            loss = per_sequence_loss.sum() / group_weights.sum()

        elif self.loss_type == "bnpo":
            loss = (weighted_per_token_loss * completion_mask).sum() / weighted_completion_mask.sum().clamp(min=1.0)

        elif self.loss_type == "dr_grpo":
            effective_batch_size = group_weights.sum()
            loss = (weighted_per_token_loss * completion_mask).sum() / (
                effective_batch_size * self.max_completion_length
            )

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}. Supported types: {list(LOSS_TYPES)}")

        self._buffer_sign_metrics(sample_values, advantages, completion_mask)
        return loss

    def _buffer_sign_metrics(
        self,
        sample_values: dict[str, torch.Tensor],
        advantages: torch.Tensor,
        mask: torch.Tensor,
        rows: torch.Tensor | None = None,
    ) -> None:
        """Buffer one batch's per-sample diagnostics on device, for the once-per-log-step drain.

        A per-microbatch ``.item()`` would be dozens of host syncs a step, so the objective's
        per-token quantities are reduced to per-sample means over the completion tokens and kept on
        device. ``rows`` selects the real rows of a row-padded pipeline batch; the inert ones carry
        all-ignore labels and a zero advantage, and would otherwise count as zero-valued
        positive-advantage samples. The mode keys the buffer ``log`` drains.
        """
        counts = mask.sum(dim=1).clamp(min=1)
        buffer = self._sign_metric_buffer["train" if self.model.training else "eval"]
        positive = advantages.detach() >= 0
        buffer["positive_mask"].append(positive if rows is None else positive[rows])
        for key, tensor in sample_values.items():
            values = ((tensor.detach() * mask).sum(dim=1) / counts).float()
            buffer[key].append(values if rows is None else values[rows])

    def _drain_sign_metrics(self, mode: str) -> None:
        """Gather the buffered per-sample values once per log step and emit summary statistics.

        Runs on every rank (the gather is a collective; log() is invoked rank-uniformly); statistics
        are computed over the pooled global sample set of the whole logging window.
        """
        buffer = self._sign_metric_buffer[mode]
        if not buffer:
            return
        # Under PP only the last stage's ranks buffer anything, and they are the DP replicas of the
        # whole sample set; a world gather would block the other stages in a collective they never
        # enter. The chain gets the drained scalars from ``log``.
        gather = self._pp_stage_gather if self._pp_runtime is not None else self.accelerator.gather
        positive = gather(torch.cat(buffer["positive_mask"]))
        for key, chunks in buffer.items():
            if key == "positive_mask":
                continue
            values = gather(torch.cat(chunks))
            for sign in self._SIGN_METRIC_KEYS[key]:
                subset = values[positive if sign == "positive" else ~positive]
                if subset.numel() == 0:
                    continue
                prefix = f"{sign}/{key}"
                self._metrics[mode][f"{prefix}_mean"].append(subset.mean().item())
                self._metrics[mode][f"{prefix}_std"].append(
                    subset.std(unbiased=False).item() if subset.numel() > 1 else 0.0
                )
                low, high = subset.min().item(), subset.max().item()
                self._metrics[mode][f"{prefix}_min"].append(low)
                self._metrics[mode][f"{prefix}_max"].append(high)
                self._metrics[mode][f"{prefix}_range"].append(high - low)
        buffer.clear()

    def _setup_pipeline_parallel(self):
        """Seed the PP-only live clamp floor, then run the mixin's setup.

        Kept out of ``_pp_loss_adapter``, which the mixin may call as a declarative accessor;
        ``_pp_batch_transform`` refreshes it from the scheduler every step.
        """
        self._pp_min_log_prob: float | None = self.min_log_prob
        super()._setup_pipeline_parallel()

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """Offline GRPO's pipeline-loss contract.

        The per-token objective is microbatch-local and every whole-batch denominator is a pure
        function of batch metadata, so it rides the runtime's single per-step normalizer and the
        microbatch losses sum exactly to the full-batch loss. That normalizer is rank-local on
        purpose: the non-PP loss is a rank-local quotient whose gradients FSDP averages over the DP
        group, and the PP stage wrap averages over the same group. At ``kl_beta != 0`` the
        reference per-token log-probs ship as a third per-example side tensor, scored before
        training by :meth:`_pp_precompute_reference_logps`.

        Memory: the non-PP path trims the forward with ``logits_to_keep``; a pipeline's last stage
        has no such lever, so its loss receives full ``[rows, max_length, vocab]`` logits.
        """
        extra_target_keys = ("advantage", "group_size")
        if self.beta != 0.0:
            extra_target_keys += (REF_PER_TOKEN_LOGPS_COLUMN,)
        return PPLossAdapter(
            token_loss_fn=self._pp_token_loss,
            batch_transform=self._pp_batch_transform,
            normalizer=self._pp_normalizer,
            extra_target_keys=extra_target_keys,
            # Row filler for partial eval batches: with 0, an inert row's 1/group_size is inf → NaN.
            pad_spec={"group_size": 1},
            # The same whole-batch denominator as training, over the pre-pad batch's real rows.
            eval_normalizer=self._pp_normalizer,
        )

    def _pp_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Concat the split prompt/completion keys into the pipeline's fixed-shape batch contract.

        Mirrors ``compute_loss``'s ``torch.cat`` layout ([left-pad | prompt | completion]), derives
        ``labels`` as the completion token ids at their positions (ignore elsewhere), and right-pads
        to ``args.max_length`` because the P2P buffer shapes freeze on the first step. Also
        refreshes the live scheduled ``min_log_prob`` for the microbatch loss.

        With a KL term, the collated reference log-probs (one per completion token) are laid onto
        the last stage's shifted ``[rows, max_length - 1]`` grid: completion token ``j`` sits at
        input position ``P + j`` (``P`` = the batch's left-padded prompt width) and is scored by the
        logits one position earlier, so its reference value lands at ``P + j - 1``. Positions the
        loss mask excludes stay 0. The reference sweep itself feeds batches through here before the
        column exists, and is the only caller permitted to do so.
        """
        prompt_ids, prompt_mask = batch["prompt_input_ids"], batch["prompt_attention_mask"]
        completion_ids, completion_mask = batch["completion_input_ids"], batch["completion_attention_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        labels = torch.cat(
            [torch.full_like(prompt_ids, LABEL_IGNORE_INDEX), completion_labels(completion_ids, completion_mask)],
            dim=1,
        )
        padded = pad_to_pipeline_length(
            {
                "input_ids": (input_ids, self.padding_value),
                "attention_mask": (attention_mask, 0),
                "labels": (labels, LABEL_IGNORE_INDEX),
            },
            self.args.max_length,
            "Lower max_prompt_length/max_completion_length so their sum fits, or raise max_length.",
        )
        self._pp_min_log_prob = live_min_log_prob(self.model, self.min_log_prob)
        out = {
            **padded,
            "advantage": batch["advantage"],
            "group_size": batch["group_size"],
        }
        if self.beta != 0.0 and self._pp_ref_sweep is None:
            ref_logps = batch.get(REF_PER_TOKEN_LOGPS_COLUMN)
            if ref_logps is None:
                raise RuntimeError(
                    f"kl_beta={self.beta} under pipeline parallelism needs the '{REF_PER_TOKEN_LOGPS_COLUMN}' "
                    f"column the construction-time reference sweep writes, but this batch carries none — "
                    f"the dataset was replaced after construction without it."
                )
            prompt_width = prompt_ids.size(1)
            shifted = ref_logps.new_zeros((input_ids.size(0), self.args.max_length - 1), dtype=torch.float32)
            shifted[:, prompt_width - 1 : prompt_width - 1 + ref_logps.size(1)] = ref_logps.float()
            out[REF_PER_TOKEN_LOGPS_COLUMN] = shifted
        return out

    def _pp_normalizer(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """This loss type's whole-batch denominator, from batch metadata (see ``compute_loss``).

        grpo: sum of group weights; bnpo: group-weighted completion-token count; dr_grpo: effective
        group count × max_completion_length. Shifted and unshifted completion-token counts are equal
        (the collator guarantees a non-empty prompt, so position 0 is never a completion token).
        """
        group_weights = 1.0 / inputs["group_size"].float()
        if self.loss_type == "grpo":
            return group_weights.sum()
        if self.loss_type == "bnpo":
            token_counts = loss_token_counts_per_row(inputs["labels"]).float()
            return (group_weights * token_counts).sum().clamp(min=1.0)
        if self.loss_type == "dr_grpo":
            return group_weights.sum() * self.max_completion_length
        raise ValueError(f"Unknown loss type: {self.loss_type!r}. Supported types: {list(LOSS_TYPES)}")

    def _pp_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """The offline-GRPO numerator over one microbatch; contributions sum to the full-batch loss.

        Per-token log-probs of the completion labels, the negative-advantage ``min_log_prob`` clamp
        (on the policy and, as off PP, on the reference), then :func:`offline_token_objective` (the
        objective ``compute_loss`` runs), followed by group weighting and, for loss_type 'grpo', the
        per-sequence token mean (row-local, so microbatch-safe). The reference arrives precomputed as
        a target column, and the runtime divides by ``_pp_normalizer``'s full-batch denominator.

        While :meth:`_pp_precompute_reference_logps` drives the schedule, the same call scores the
        reference instead: the raw per-token log-probs are stashed and no objective is formed.
        """
        token_logps, mask = token_logprobs(logits, target["labels"])
        if self._pp_ref_sweep is not None:
            self._pp_ref_sweep.append(token_logps.detach())
            return token_logps.new_zeros(())

        advantages = target["advantage"].float()
        group_weights = 1.0 / target["group_size"].float()
        negative_advantage = (advantages < 0).unsqueeze(1)

        min_log_prob = self._pp_min_log_prob
        token_logps_unclamped = token_logps
        if min_log_prob is not None:
            token_logps = torch.where(negative_advantage, token_logps.clamp(min=min_log_prob), token_logps)

        ref_logps = ref_logps_unclamped = None
        if self.beta != 0.0:
            ref_logps = ref_logps_unclamped = target[REF_PER_TOKEN_LOGPS_COLUMN]
            if min_log_prob is not None:
                ref_logps = torch.where(negative_advantage, ref_logps.clamp(min=min_log_prob), ref_logps)
        per_token_loss, sample_values = offline_token_objective(
            token_logps,
            token_logps_unclamped,
            advantages,
            policy_gradient_formulation=self.policy_gradient_formulation,
            beta=self.beta,
            ref_logps=ref_logps,
            ref_logps_unclamped=ref_logps_unclamped,
        )
        self._buffer_sign_metrics(sample_values, advantages, mask, rows=rows_with_labels(target["labels"]))
        weighted = per_token_loss * mask * group_weights.unsqueeze(1)
        if self.loss_type == "grpo":
            return (weighted.sum(dim=1) / mask.sum(dim=1).clamp(min=1)).sum()
        return weighted.sum()  # bnpo / dr_grpo share the numerator; only the normalizer differs

    def _pp_precompute_reference_logps(self, dataset: "datasets.Dataset", what: str) -> "datasets.Dataset":
        """Score the KL reference through the pipeline and return ``dataset`` with the values as a column.

        At construction the split model is the reference (the checkpoint weights, or on resume the
        resumed ones, which is what ``create_reference_model`` anchors to off PP), so one
        forward-only sweep through the schedule before the first optimizer step scores every token,
        and the values ride into every step as the ``REF_PER_TOKEN_LOGPS_COLUMN`` extra target.
        Each data-parallel replica sweeps a contiguous shard and the shards are all-gathered over the
        stage group, so every rank holds the whole column (a pre-sharded dataset is already this
        rank's own rows and is swept whole). The last stage's loss stashes per-microbatch log-probs
        rather than the schedule merging the batch's logits, so the sweep's memory is the training
        path's own per-microbatch footprint; a partial final batch is row-padded to the pipeline's
        frozen shape with inert rows and trimmed back.
        """
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                f"kl_beta={self.beta} under pipeline parallelism scores the reference over the whole "
                f"{what} dataset before training, which needs a sized, indexable dataset; the "
                f"{what} dataset is an IterableDataset."
            )
        dp_size, dp_rank = self.dp_shard_geometry()
        rows = len(dataset)
        shard = dataset.select(range(dp_rank * rows // dp_size, (dp_rank + 1) * rows // dp_size))
        loader = DataLoader(
            shard, batch_size=self.args.per_device_train_batch_size, shuffle=False, collate_fn=self.data_collator
        )
        frozen_rows = self.args.per_device_train_batch_size
        pads = self._pp_frozen_row_pads()
        # Every stage forward is a collective across the stage's replicas (FSDP2's all-gathers, a
        # DeepEP dispatch under PP+EP), so the replicas must run the same number of batches: a shard
        # one row longer would block its peers on its extra batch. The shorter shards replay their
        # last (already frozen-shape) batch for the difference and discard the result.
        if len(loader) == 0:
            raise ValueError(
                f"The {what} dataset leaves data-parallel rank {dp_rank} no rows to score the KL reference "
                f"over ({rows} rows across {dp_size} replicas); the dataset is too small for this world size."
            )
        sweep_batches = torch.tensor([len(loader)], device=current_device())
        if dist.get_world_size(self._pp_stage_group) > 1:
            dist.all_reduce(sweep_batches, op=dist.ReduceOp.MAX, group=self._pp_stage_group)
        replays = int(sweep_batches.item()) - len(loader)
        logger.info(f"Scoring the KL reference over the {what} dataset through the pipeline ({rows} rows)...")

        was_training = self.model.training
        self.model.eval()
        per_row: list[torch.Tensor] = []
        inputs = None
        try:
            for batch in loader:
                batch = self._prepare_inputs(batch)
                real_rows = batch["prompt_input_ids"].size(0)
                prompt_width = batch["prompt_input_ids"].size(1)
                completion_width = batch["completion_input_ids"].size(1)
                counts = batch["completion_attention_mask"].sum(dim=1).tolist()
                # Set before the transform: it signals to the transform (no column yet) and to the
                # last stage's loss (score, do not train) that this pass is the sweep.
                self._pp_ref_sweep = []
                inputs = self._pp_pad_rows_to_frozen(self._pp_batch_transform(batch), frozen_rows, pads)
                self._pp_runtime.eval_loss(
                    inputs["input_ids"],
                    inputs["labels"],
                    attention_mask=inputs["attention_mask"],
                    num_items_in_batch=1.0,
                    extra_targets={key: inputs[key] for key in ("advantage", "group_size")},
                )
                stashed = torch.cat(self._pp_ref_sweep) if self.parallelism_config.is_last_pp_stage else None
                self._pp_ref_sweep = None
                logps = self._pp_broadcast_output_from_last_stage(stashed)
                completion_logps = logps[:real_rows, prompt_width - 1 : prompt_width - 1 + completion_width].cpu()
                per_row.extend(completion_logps[r, : counts[r]] for r in range(real_rows))
            for _ in range(replays):
                self._pp_ref_sweep = []
                self._pp_runtime.eval_loss(
                    inputs["input_ids"],
                    inputs["labels"],
                    attention_mask=inputs["attention_mask"],
                    num_items_in_batch=1.0,
                    extra_targets={key: inputs[key] for key in ("advantage", "group_size")},
                )
                stashed = torch.cat(self._pp_ref_sweep) if self.parallelism_config.is_last_pp_stage else None
                self._pp_ref_sweep = None
                self._pp_broadcast_output_from_last_stage(stashed)
        finally:
            self._pp_ref_sweep = None
            self.model.train(was_training)

        shards = [(dp_rank, per_row)]
        if dp_size > 1:
            gathered: list = [None] * dist.get_world_size(self._pp_stage_group)
            dist.all_gather_object(gathered, shards[0], group=self._pp_stage_group)
            # ETP partners and chain peers land the same dp_rank with identical values; keyed dedup.
            by_rank = dict(gathered)
            shards = [(rank, by_rank[rank]) for rank in range(dp_size)]
        column = [values.tolist() for _, shard_rows in shards for values in shard_rows]
        if len(column) != rows:
            raise RuntimeError(
                f"The KL reference sweep scored {len(column)} rows of the {what} dataset but it has {rows}; "
                f"the data-parallel shards did not reassemble into the dataset."
            )
        return dataset.add_column(REF_PER_TOKEN_LOGPS_COLUMN, column)

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys: list[str] | None = None,
    ):
        if self._pp_runtime is not None:
            # compute_loss (and its sign-metric buffering) needs the full model, so the mixin drives
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
        # the training log call would read "eval", dropping the train sign metrics (their buffer is
        # never drained) and prefixing eval_* onto a train line.
        mode = "train" if self.model.training else "eval"
        self._drain_sign_metrics(mode)
        metrics = {key: sum(val) / len(val) for key, val in self._metrics[mode].items()}

        current_min_log_prob = live_min_log_prob(self.model, self.min_log_prob)
        if current_min_log_prob is not None:  # None = no clipping configured
            metrics["min_log_prob_restriction"] = current_min_log_prob

        if self._pp_runtime is not None:
            metrics = self._pp_broadcast_log_metrics(metrics)

        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
