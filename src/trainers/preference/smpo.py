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
"""Smooth Margin Preference Optimization (SMPO): reference-model-free preference optimization via a
margin-based loss plus an optional SFT term. Supports EP/CP/TP (log-prob sums/counts aggregate across
CP ranks) and PP (the chosen|rejected concat re-laid out as interleaved pairs; see
``_pp_loss_adapter``).

``is_vlm`` selects the data path: VLM mode takes raw message-list rows plus an ``images`` column,
image processing in the collator, vision tensors riding the chosen|rejected concat row-major. Image
tokens live only in the prompt region (labels ``-100``), so the loss math is untouched. VLM mode
excludes ``padding_free``, CP and PP; a text run on a multimodal checkpoint keeps all three.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from datasets import Dataset, IterableDataset
from datasets import Image as ImageFeature
from datasets import Sequence as SequenceFeature
from transformers import (
    DataCollator,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
    Trainer,
)
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from trl.trainer.utils import (
    disable_dropout_in_model,
    selective_log_softmax,
)

from src.callbacks.variable_scheduler import VariableSchedulerCallback
from src.configs.smpo_config import SmoothMarginPOConfig
from src.data.collators.smpo import (
    PREFERENCE_BATCH_KEYS,
    SEQUENCE_ALIGNED_VISION_KEYS,
    DataCollatorForSMPO,
    DataCollatorForVLMSMPO,
)
from src.data.pipeline.preferences import split_rendered_completion, split_vlm_preference_row
from src.data.pipeline.processing import coordinated_map
from src.data.pipeline.rendered import probe_tokenizer_specials
from src.data.spans import LABEL_IGNORE_INDEX, ends_with_terminator, resolve_eos_token_ids
from src.data.vlm import render_vlm_text
from src.distributed.context_parallel.config import cp_boundary_shift, split_sequence_for_cp
from src.distributed.loading.model_loading import load_model_from_pretrained
from src.distributed.loading.peft_setup import peft_bf16_autocast, prepare_peft_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.losses import (
    PPLossAdapter,
    interleave_pairs,
    loss_token_counts_per_row,
    pad_to_pipeline_length,
    rows_with_labels,
    split_pairs,
    token_logprobs,
)
from src.models.patches.attention import VARLEN_ATTN_IMPLEMENTATIONS
from src.models.structure import resolve_tokenizer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.pp_gates import reject_pp_peft
from src.trainers.mixins.stored_metrics import StoredMetricsMixin

logger = get_logger(__name__, log_level="info")


def tokenize_preference_row(
    features: dict[str, str],
    processing_class: PreTrainedTokenizerBase,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, list[int]]:
    """Tokenize one (prompt, chosen, rejected) example.

    Tokenizes full prompt+completion sequences to handle boundary token-merging correctly.
    """
    prompt = features["prompt"]
    chosen = features["chosen"]
    rejected = features["rejected"]

    prompt_tokens = processing_class(prompt, add_special_tokens=False)["input_ids"]

    full_chosen = processing_class(prompt + chosen, add_special_tokens=False)
    full_rejected = processing_class(prompt + rejected, add_special_tokens=False)

    # Token merging at the prompt/completion boundary shifts the split point by one. The prompt must
    # move with it: the trainer concatenates prompt_input_ids ⧺ completion_input_ids verbatim, so
    # keeping the full prompt while the completion starts one token earlier duplicates the boundary
    # token. The split has to be the SAME for both completions — they share one prompt field, and a
    # per-side split would condition chosen and rejected on different contexts, making the margin a
    # comparison between two different prompts.
    split = len(prompt_tokens)
    if full_chosen["input_ids"][:split] != prompt_tokens or full_rejected["input_ids"][:split] != prompt_tokens:
        # full_*[: split - 1] == prompt_tokens[: split - 1] holds on the non-merging side too, so
        # rolling both back stays exact there.
        split -= 1
        prompt_tokens = prompt_tokens[:split]

    chosen_input_ids = full_chosen["input_ids"][split:]
    rejected_input_ids = full_rejected["input_ids"][split:]

    # BOS only when the tokenizer's own post-processor emits one: gpt-oss/Bailing define a
    # nominal bos_token it never emits, so forcing it trains on a token the policy never sees.
    bos_id = processing_class.bos_token_id
    if (
        bos_id is not None
        and probe_tokenizer_specials(processing_class).adds_leading_bos
        and (len(prompt_tokens) == 0 or prompt_tokens[0] != bos_id)
    ):
        prompt_tokens = [bos_id] + prompt_tokens

    # "Already terminated?" has two traps, and both append an ender the policy never emits — inside
    # the mean log-prob that IS the SMPO margin. The render closes with ``<terminator>\n``, so the
    # last token is a newline; and GLM-4/Gemma close turns with a role marker carried on the config,
    # not on tokenizer.eos_token_id. Hence the whitespace-tolerant walk over the full terminator set.
    eos_id = processing_class.eos_token_id
    terminators = eos_token_ids or ({eos_id} if eos_id is not None else set())
    if eos_id is not None:
        if not ends_with_terminator(chosen_input_ids, processing_class, terminators):
            chosen_input_ids = chosen_input_ids + [eos_id]
        if not ends_with_terminator(rejected_input_ids, processing_class, terminators):
            rejected_input_ids = rejected_input_ids + [eos_id]

    if max_prompt_length and len(prompt_tokens) > max_prompt_length:
        if truncation_mode == "keep_start":
            prompt_tokens = prompt_tokens[:max_prompt_length]
        else:  # keep_end
            prompt_tokens = prompt_tokens[-max_prompt_length:]

    if max_completion_length:
        # A plain tail slice would cut the EOS appended above, so the model never learns to stop there.
        def _truncate_keep_eos(ids: list[int]) -> list[int]:
            if len(ids) <= max_completion_length:
                return ids
            if eos_id is None:
                return ids[:max_completion_length]
            return ids[: max_completion_length - 1] + [eos_id]

        chosen_input_ids = _truncate_keep_eos(chosen_input_ids)
        rejected_input_ids = _truncate_keep_eos(rejected_input_ids)

    return {
        "prompt_input_ids": prompt_tokens,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }


def tokenize_vlm_preference_row(
    features: dict[str, Any],
    processing_class: ProcessorMixin,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Prepare one raw VLM (prompt, chosen, rejected [, images]) example.

    The pre-render half is :func:`~src.data.pipeline.preferences.split_vlm_preference_row`,
    shared with the VLM reward map. Completions are then rendered by the same prefix-strip invariant
    as the text pipeline (``template(prompt + completion)`` minus ``template(prompt)``) and tokenized
    through :func:`tokenize_preference_row`, so boundary merges, EOS appending and truncation stay
    byte-identical to it. The prompt stays text + PIL images; the collator expands placeholders per
    batch.
    """
    tokenizer = resolve_tokenizer(processing_class)
    prompt_history, pil_images, completions = split_vlm_preference_row(features, "VLM SMPO row")

    prompt_text = render_vlm_text(processing_class, prompt_history)
    completion_texts = {
        side: split_rendered_completion(prompt_text, render_vlm_text(processing_class, prompt_history + history), side)
        for side, history in completions.items()
    }

    tokenized = tokenize_preference_row(
        {"prompt": prompt_text, "chosen": completion_texts["chosen"], "rejected": completion_texts["rejected"]},
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        truncation_mode=truncation_mode,
        eos_token_ids=eos_token_ids,
    )
    return {
        "prompt_text": prompt_text,
        "images": pil_images,
        "chosen_input_ids": tokenized["chosen_input_ids"],
        "rejected_input_ids": tokenized["rejected_input_ids"],
    }


@dataclass
class _PPStepState:
    """Per-step constants the PP microbatch loss reads; refreshed by ``_pp_batch_transform``.

    A microbatch loss cannot derive SMPO's normalizers from its own rows — the pair count and the
    chosen/rejected token counts are full-batch quantities — nor read the live scheduled margin, so
    the batch transform (which sees the whole prepared batch every step) stashes them here.
    """

    pair_count: int = 1
    chosen_token_count: torch.Tensor | float = 1.0
    rejected_token_count: torch.Tensor | float = 1.0
    target_margin: float = 0.0
    # Per-microbatch metric SUMS (and the token counts the logit means divide by), folded into the
    # batch's means once per step. Off PP the same numbers come from one whole-batch forward.
    metric_sums: dict[str, torch.Tensor] = field(default_factory=dict)


class SmoothMarginPOTrainer(StoredMetricsMixin, DistributedTrainerMixin, Trainer):
    """Reference-model-free preference optimization. Supports EP/CP/TP/PP.

    Loss is L_margin + L_sft, where L_margin encourages
    log_prob(chosen) - log_prob(rejected) >= target_margin and L_sft is a weighted
    cross-entropy on the completions. Under CP, log-prob sums and token counts are
    all-reduced across the CP group before averaging. Under PP the chosen|rejected concat is
    re-laid out as interleaved pairs (see ``_pp_loss_adapter``).

    ``is_vlm`` enables VLM mode: raw message-list prompt/chosen/rejected rows plus an ``images``
    column, with image processing at collation and vision tensors threaded through the
    chosen|rejected concat. VLM mode excludes ``padding_free``, CP and PP. Omitted, it falls back to
    the modality of ``processing_class`` (a ProcessorMixin means VLM).
    """

    _tag_names = ["trl", "smpo", "smooth-margin-po"]

    _supports_cp = True
    # Reference-free single forward; correct once the concat is interleaved so no pair splits.
    _supports_pp = True
    # The objective is already its own mean, so HF's num_items_in_batch rescaling must stay off.
    _loss_is_own_mean = True
    # chosen ⧺ rejected ride ONE forward, so every MoE layer sees twice the collator's rows.
    _forward_rows_per_example = 2

    def __init__(
        self,
        model: str | PreTrainedModel | nn.Module,
        args: SmoothMarginPOConfig | None = None,
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
            args = SmoothMarginPOConfig("smpo-output")

        if processing_class is None:
            raise ValueError("processing_class (tokenizer) must be provided")

        # A VLM ProcessorMixin keeps token ids on its inner tokenizer, not on itself.
        tokenizer = resolve_tokenizer(processing_class)
        self.is_vlm = self._resolve_vlm_mode(processing_class, is_vlm)
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.beta = args.beta
        self.target_margin = args.target_margin
        self.chosen_sft_ratio = args.chosen_sft_ratio
        self.loss_type = args.loss_type
        self.lower_clip_percentile = args.lower_clip_percentile
        self.upper_clip_percentile = args.upper_clip_percentile
        self.min_log_prob = args.min_log_prob
        self.label_pad_token_id = args.label_pad_token_id
        _, self.max_prompt_length, self.max_completion_length = args.resolve_length_budget()
        self.truncation_mode = args.truncation_mode
        self.dataset_num_proc = args.dataset_num_proc

        self.padding_free = args.padding_free
        self._validate_vlm_mode(parallelism_config)
        if self.padding_free and parallelism_config and parallelism_config.cp_size > 1:
            raise ValueError(
                "padding_free mode is incompatible with Context Parallelism (CP). "
                "CP requires fixed-length sequences for collective synchronization. "
                "Please disable padding_free when using CP."
            )
        self._reject_pp_explicit_options(parallelism_config, peft_config)

        model, _ = load_model_from_pretrained(model, args)

        # AFTER the load: the ctor accepts a model id, and the config is where a family that closes
        # turns with a role marker (GLM-4, Gemma) declares it — resolving earlier would silently
        # degrade to tokenizer-eos-only for exactly those models. Resolved once, because the tokenize
        # map runs in worker processes and fingerprints its fn_kwargs.
        self._eos_token_ids = resolve_eos_token_ids(tokenizer, getattr(model, "config", None))

        self._peft_has_been_casted_to_bf16 = False
        if peft_config is not None:
            # Same three flags prepare_peft_model tests: a torchao/quanto model sets only is_quantized,
            # and missing it here re-enables the input-requires-grad hook the k-bit prep owns.
            quantized = (
                getattr(model, "is_loaded_in_8bit", False)
                or getattr(model, "is_loaded_in_4bit", False)
                or getattr(model, "is_quantized", False)
            )
            # The input-requires-grad hook must precede the PEFT wrap (which covers the k-bit path).
            if not quantized and args.gradient_checkpointing:
                self._enable_input_require_grads(model)
            model, self._peft_has_been_casted_to_bf16 = prepare_peft_model(model, peft_config, args)
        elif args.gradient_checkpointing:
            self._enable_input_require_grads(model)

        if args.disable_dropout:
            disable_dropout_in_model(model)

        if self.padding_free:
            attn_impl = getattr(model.config, "_attn_implementation", None)
            if attn_impl not in VARLEN_ATTN_IMPLEMENTATIONS:
                raise ValueError(
                    f"padding_free is not supported with attn_implementation='{attn_impl}': it has no "
                    f"varlen kernel. Padding-free flattens the batch into one [1, total_tokens] "
                    f"sequence carrying no attention mask, so every example would attend across the "
                    f"whole batch and the preference logps would be computed on contaminated "
                    f"contexts — silently. Use one of {list(VARLEN_ATTN_IMPLEMENTATIONS)}, or set "
                    f"padding_free: false."
                )

        # The model rides through the distributed seam so PP can split it into this rank's stage.
        dist_kwargs = self._init_distributed_config(
            kwargs,
            training_args=args,
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
            model=model,
        )
        model = dist_kwargs.pop("model")
        self._reject_cp_incompatible_collator(data_collator)

        if data_collator is None:
            if self.is_vlm:
                data_collator = DataCollatorForVLMSMPO(
                    pad_token_id=self.pad_token_id,
                    processor=processing_class,
                    max_prompt_length=self.max_prompt_length,
                )
            else:
                data_collator = DataCollatorForSMPO(pad_token_id=self.pad_token_id)

        # Coordinated, not local_main_process_first + raw map: one rank tokenizes to a deterministic
        # cache file and the rest load it. Wrapping this in a main-first block would desync it.
        if train_dataset is not None:
            train_dataset = self._prepare_dataset(train_dataset, processing_class, "train")
        if eval_dataset is not None:
            if isinstance(eval_dataset, dict):
                eval_dataset = {k: self._prepare_dataset(v, processing_class, k) for k, v in eval_dataset.items()}
            else:
                eval_dataset = self._prepare_dataset(eval_dataset, processing_class, "eval")

        self.use_margin_schedule = args.use_margin_schedule
        self.initial_margin = args.initial_margin
        self.final_margin = self.target_margin

        if callbacks is None:
            callbacks = []

        if args.use_margin_schedule:
            model.target_margin = self.initial_margin
            logger.info(f"Initialized model.target_margin = {self.initial_margin} for scheduling")

            margin_scheduler = VariableSchedulerCallback(
                attribute_name="target_margin",
                initial_value=self.initial_margin,
                final_value=self.final_margin,
                schedule_type="linear",
            )
            callbacks.append(margin_scheduler)
            logger.info(f"Added target_margin scheduler: {self.initial_margin} → {self.final_margin}")

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

        self._setup_distributed_modes()

    @staticmethod
    def _resolve_vlm_mode(processing_class, is_vlm: bool | None) -> bool:
        """The RUN's data path — VLM rows + vision collator, or the text tokenize path.

        The checkpoint cannot decide it: a natively multimodal checkpoint (Gemma 4, Qwen3.5/3.6)
        trained on text-only preference pairs is a TEXT run, and the text path is the one that keeps
        CP, ``padding_free`` and PP. The script decides it with ``is_vlm_run`` (multimodal checkpoint
        AND image data) and passes it here; the model class and ``processing_class`` still follow the
        checkpoint, so a text run legitimately arrives with a processor.

        Omitted (``None``), it falls back to the modality of ``processing_class`` — the contract for
        direct trainer users, who have no run to consult.
        """
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

    def _validate_vlm_mode(self, parallelism_config: ParallelismConfig | None) -> None:
        """Reject option combinations VLM mode cannot honor (fail loud at construction).

        ``padding_free`` flattens the batch into one sequence — the flat forward drops the
        prompt-tensor alignment the vision tower needs to map patches onto expanded image
        placeholder spans. CP splits the sequence across ranks, scattering a prompt's placeholder
        span while every rank would still receive the full pixel tensors.
        """
        if not self.is_vlm:
            return
        if self.padding_free:
            raise ValueError(
                "SMPO VLM mode is incompatible with padding_free: flattening the batch breaks the "
                "image-placeholder ↔ pixel-patch alignment. Disable padding_free for VLM training."
            )
        if parallelism_config is not None and parallelism_config.cp_size > 1:
            raise ValueError(
                "SMPO VLM mode is incompatible with Context Parallelism (CP): sequence splitting "
                "scatters expanded image placeholder spans across CP ranks. Drop "
                "context_parallel_size for VLM training."
            )

    def _reject_pp_explicit_options(
        self, parallelism_config: ParallelismConfig | None, peft_config: Any | None
    ) -> None:
        """Reject SMPO options pipeline parallelism cannot honor (fail loud at construction).

        Covers the ones that arrive as explicit constructor parameters, which the mixin's
        kwargs-inspecting hook cannot see.
        """
        if parallelism_config is None or not parallelism_config.is_pp_mode:
            return
        if self.is_vlm:
            raise ValueError(
                "SMPO VLM mode is not supported under pipeline parallelism: vision tensors ride the "
                "chosen|rejected concat into the vision tower of the full multimodal forward, but a "
                "pipeline stage's forward takes hidden_states and the fixed-shape P2P contract "
                "carries no patch-space tensors. Train VLM SMPO without PP."
            )
        if self.padding_free:
            raise ValueError(
                "padding_free is not supported under pipeline parallelism: it flattens each batch "
                "into one variable-length row, and the pipeline's P2P buffer shapes are frozen "
                "after the first step. Disable padding_free."
            )
        if self.lower_clip_percentile is not None or self.upper_clip_percentile is not None:
            raise ValueError(
                "lower_clip_percentile/upper_clip_percentile are not supported under pipeline "
                "parallelism: the clip bound is a token quantile over the WHOLE batch half, and a "
                "microbatch loss sees only its own tokens — the bound (and thus the objective) "
                "would change with pipeline_microbatches. Set both to null under PP "
                "(min_log_prob, a per-token constant, stays supported)."
            )
        if self.label_pad_token_id != LABEL_IGNORE_INDEX:
            raise ValueError(
                f"label_pad_token_id={self.label_pad_token_id} is not supported under pipeline "
                f"parallelism: the pipeline's fixed-shape label padding and token counting are "
                f"keyed to {LABEL_IGNORE_INDEX}."
            )
        reject_pp_peft(peft_config, explicit_param_trainer="SMPO")

    def _prepare_dataset(
        self,
        dataset: Dataset | IterableDataset,
        processing_class: PreTrainedTokenizerBase,
        dataset_name: str,
    ) -> Dataset | IterableDataset:
        """Prepare dataset for SMPO training (skips if already prepared).

        Text mode tokenizes templated (prompt, chosen, rejected) strings; VLM mode normalizes raw
        message-list rows, keeping the prompt as text + images so the collator can run the heavy
        image processing per batch instead of persisting pixel tensors into the Arrow cache.
        """
        if self.is_vlm:
            required_cols = {"prompt_text", "chosen_input_ids", "rejected_input_ids"}
            tokenize_fn = tokenize_vlm_preference_row
            row_processing_class = processing_class
        else:
            required_cols = {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
            tokenize_fn = tokenize_preference_row
            # The TEXT row fn calls ``processing_class(text, add_special_tokens=False)`` — a
            # processor reads that second slot as images — and reads bos/eos ids a ProcessorMixin
            # does not carry. A text run on a multimodal checkpoint still keeps the processor as
            # processing_class (every saved checkpoint owes a processor_config.json), so the row
            # fn gets its inner tokenizer instead.
            row_processing_class = resolve_tokenizer(processing_class)
        if isinstance(dataset, Dataset) and required_cols.issubset(dataset.column_names):
            logger.info(f"Dataset '{dataset_name}' already prepared, skipping _prepare_dataset")
            return dataset

        original_columns = dataset.column_names

        dataset = coordinated_map(
            dataset,
            tokenize_fn,
            num_proc=self._dataset_map_num_proc(self.dataset_num_proc),
            remove_columns=original_columns,
            desc=f"Tokenizing {dataset_name} dataset",
            # In fn_kwargs, not read off self: the map fn must stay picklable, and the cache key
            # fingerprints fn_kwargs.
            fn_kwargs={
                "processing_class": row_processing_class,
                "max_prompt_length": self.max_prompt_length,
                "max_completion_length": self.max_completion_length,
                "truncation_mode": self.truncation_mode,
                "eos_token_ids": self._eos_token_ids,
            },
        )

        if self.is_vlm:
            # dataset.map drops the Image decoding feature — recast so rows hand back PIL images.
            dataset = dataset.cast_column("images", SequenceFeature(ImageFeature()))

        return dataset

    def _set_signature_columns_if_needed(self):
        """Set signature columns for data collation."""
        if self._signature_columns is None:
            if self.is_vlm:
                self._signature_columns = [
                    "prompt_text",
                    "images",
                    "chosen_input_ids",
                    "rejected_input_ids",
                ]
            else:
                self._signature_columns = [
                    "prompt_input_ids",
                    "chosen_input_ids",
                    "rejected_input_ids",
                ]
        # The dataloader mixin's union of the collator's required_dataset_columns sits BELOW this
        # override in the MRO — chain to it, or a collator input would be pruned here alone.
        super()._set_signature_columns_if_needed()

    @staticmethod
    def concatenated_inputs(
        batch: dict[str, torch.Tensor],
        pad_token_id: int,
        label_pad_token_id: int = LABEL_IGNORE_INDEX,
        cp_size: int = 1,
    ) -> dict[str, torch.Tensor]:
        """Concatenate chosen+rejected into one batch (input_ids/attention_mask/labels) for a single forward."""
        max_prompt_len = batch["prompt_input_ids"].shape[1]
        max_completion_len = max(
            batch["chosen_input_ids"].shape[1],
            batch["rejected_input_ids"].shape[1],
        )

        total_len = max_prompt_len + max_completion_len
        if cp_size > 1:
            remainder = total_len % cp_size
            if remainder != 0:
                max_completion_len += cp_size - remainder

        def _pad(t: torch.Tensor, fill: int) -> torch.Tensor:
            return F.pad(t, (0, max_completion_len - t.shape[1]), value=fill)

        chosen_padded = _pad(batch["chosen_input_ids"], pad_token_id)
        rejected_padded = _pad(batch["rejected_input_ids"], pad_token_id)
        chosen_mask = _pad(batch["chosen_attention_mask"], 0)
        rejected_mask = _pad(batch["rejected_attention_mask"], 0)

        batch_size = batch["prompt_input_ids"].shape[0]

        chosen_full_ids = torch.cat([batch["prompt_input_ids"], chosen_padded], dim=1)
        rejected_full_ids = torch.cat([batch["prompt_input_ids"], rejected_padded], dim=1)
        chosen_full_mask = torch.cat([batch["prompt_attention_mask"], chosen_mask], dim=1)
        rejected_full_mask = torch.cat([batch["prompt_attention_mask"], rejected_mask], dim=1)

        chosen_labels = torch.cat(
            [
                torch.full_like(batch["prompt_input_ids"], label_pad_token_id),
                torch.where(chosen_mask == 1, chosen_padded, label_pad_token_id),
            ],
            dim=1,
        )
        rejected_labels = torch.cat(
            [
                torch.full_like(batch["prompt_input_ids"], label_pad_token_id),
                torch.where(rejected_mask == 1, rejected_padded, label_pad_token_id),
            ],
            dim=1,
        )

        output = {
            "input_ids": torch.cat([chosen_full_ids, rejected_full_ids], dim=0),
            "attention_mask": torch.cat([chosen_full_mask, rejected_full_mask], dim=0),
            "labels": torch.cat([chosen_labels, rejected_labels], dim=0),
            "num_chosen": batch_size,
        }

        # The rejected rows repeat the SAME prompts, so vision tensors need an identical second copy;
        # sequence-aligned token types must first extend over the completion region to stay 1:1.
        for key, value in batch.items():
            if key in PREFERENCE_BATCH_KEYS:
                continue
            if key in SEQUENCE_ALIGNED_VISION_KEYS:
                value = F.pad(value, (0, max_completion_len), value=0)
            output[key] = torch.cat([value, value], dim=0)

        return output

    def smpo_loss(
        self,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        target_margin: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Margin-based loss encouraging log_prob(chosen) - log_prob(rejected) >= target_margin.

        Returns (per-example losses, chosen_rewards, rejected_rewards); rewards are beta-scaled log
        probs kept only for metrics. target_margin defaults to self.target_margin.
        """
        margin = target_margin if target_margin is not None else self.target_margin

        pi_logratios = chosen_logps - rejected_logps
        logits = pi_logratios - margin  # positive when margin satisfied

        if self.loss_type == "sigmoid":
            losses = -F.logsigmoid(self.beta * logits)
        elif self.loss_type == "hinge":
            losses = torch.relu(-self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (self.beta * logits).pow(2)  # IPO-style squared loss
        elif self.loss_type == "smooth_lower_bound":
            losses = torch.relu(-self.beta * logits).pow(2)  # smooth hinge (squared ReLU)
        else:
            raise ValueError(
                f"Unknown loss type: {self.loss_type}. Choose from ['sigmoid', 'hinge', 'ipo', 'smooth_lower_bound']"
            )

        chosen_rewards = (self.beta * chosen_logps).detach()
        rejected_rewards = (self.beta * rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def _aggregate_logps_with_cp(
        self,
        logp_sums: torch.Tensor,
        token_counts: torch.Tensor,
        cp_config,
    ) -> torch.Tensor:
        """Average per-sequence log probs, all-reducing partial sums/counts across the CP group.

        Each CP rank holds only its chunk's partial sums and counts.
        """
        if cp_config is None or cp_config.cp_size <= 1:
            return logp_sums / token_counts.clamp(min=1)

        # fp32 collectives: a bf16 all-reduce(SUM) of per-sequence logp sums is lossy.
        global_logp_sums = logp_sums.clone().float()
        global_token_counts = token_counts.clone().float()

        dist.all_reduce(global_logp_sums, op=dist.ReduceOp.SUM, group=cp_config.process_group)
        dist.all_reduce(global_token_counts, op=dist.ReduceOp.SUM, group=cp_config.process_group)

        return global_logp_sums / global_token_counts.clamp(min=1)

    def _get_boundary_labels(
        self,
        labels: torch.Tensor,
        cp_config,
    ) -> torch.Tensor | None:
        """First token of the next rank's chunk, which this rank's last logit predicts.

        Returns [batch, 1], or None on the last rank (and without CP). ``labels`` is the full
        pre-split sequence [batch, seq_len].
        """
        if cp_config is None or cp_config.cp_size <= 1:
            return None

        if cp_config.cp_rank == cp_config.cp_size - 1:  # last rank needs no boundary labels
            return None

        chunk_size = labels.size(1) // cp_config.cp_size
        next_chunk_start = (cp_config.cp_rank + 1) * chunk_size
        boundary_label = labels[:, next_chunk_start : next_chunk_start + 1]

        return boundary_label

    def concatenated_forward(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Forward on the concatenated chosen+rejected batch → log probs and logits for both.

        Handles CP (per-rank sequence chunks) and padding_free (flatten to drop padding waste).
        """
        concat_batch = self.concatenated_inputs(
            batch,
            pad_token_id=self.pad_token_id,
            label_pad_token_id=self.label_pad_token_id,
            cp_size=self.cp_size,
        )

        num_chosen = concat_batch["num_chosen"]

        device = model.device if hasattr(model, "device") else next(model.parameters()).device

        input_ids = concat_batch["input_ids"].to(device)
        attention_mask = concat_batch["attention_mask"].to(device)
        labels = concat_batch["labels"].to(device)

        vision_inputs = {
            key: value.to(device)
            for key, value in concat_batch.items()
            if key not in ("input_ids", "attention_mask", "labels", "num_chosen")
        }

        if self.padding_free:
            if vision_inputs:
                raise ValueError(
                    "padding_free received a batch carrying vision tensors "
                    f"({sorted(vision_inputs)}) — VLM SMPO does not support padding_free."
                )
            return self._forward_padding_free(
                model,
                input_ids,
                attention_mask,
                labels,
                num_chosen,
                device,
            )

        cp_config = self.cp_config
        boundary_labels = self._get_boundary_labels(labels, cp_config) if cp_config else None

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            **vision_inputs,
        )
        logits = outputs.logits

        if cp_config is not None and cp_config.cp_size > 1:
            local_labels = split_sequence_for_cp(labels, cp_config, seq_dim=1)
            is_last_rank = cp_config.cp_rank == cp_config.cp_size - 1
            shift_logits, shift_labels = cp_boundary_shift(logits, local_labels, boundary_labels, is_last_rank)
        else:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

        loss_mask = shift_labels != self.label_pad_token_id

        shift_labels_masked = shift_labels.clone()
        shift_labels_masked[~loss_mask] = 0

        per_token_logps = selective_log_softmax(shift_logits, shift_labels_masked)
        per_token_logps[~loss_mask] = 0
        # Pre-clip NLL for the SFT term: it anchors on the true likelihood, not the clipped one.
        per_token_nll = -per_token_logps

        if (
            self.lower_clip_percentile is not None
            or self.upper_clip_percentile is not None
            or self.min_log_prob is not None
        ):
            # Rows [0, num_chosen) are the chosen side; broadcast that split over the time axis.
            row_is_chosen = torch.arange(per_token_logps.size(0), device=per_token_logps.device) < num_chosen
            per_token_logps = self._clip_log_probs(per_token_logps, loss_mask, row_is_chosen.unsqueeze(1), cp_config)

        # fp32 sum: a bf16 accumulation of hundreds of per-token logps loses ~4 nats, flipping near-ties.
        logp_sums = per_token_logps.sum(dim=-1, dtype=torch.float32)
        token_counts = loss_mask.sum(dim=-1)

        seq_logps = self._aggregate_logps_with_cp(logp_sums, token_counts, cp_config)

        chosen_logps = seq_logps[:num_chosen]
        rejected_logps = seq_logps[num_chosen:]

        chosen_loss_mask = loss_mask[:num_chosen]
        rejected_loss_mask = loss_mask[num_chosen:]

        # Masked fp32 reductions: boolean-indexing [tokens, V] logits copies them, `.any()` syncs.
        mean_chosen_logits = self._masked_logit_mean(shift_logits[:num_chosen], chosen_loss_mask)
        mean_rejected_logits = self._masked_logit_mean(shift_logits[num_chosen:], rejected_loss_mask)

        # Shift here: under CP a deferred shift pairs local-chunk logits with full-length labels.
        chosen_sft_loss = self._compute_cp_aggregated_sft_loss(per_token_nll[:num_chosen], chosen_loss_mask, cp_config)
        rejected_sft_loss = self._compute_cp_aggregated_sft_loss(
            per_token_nll[num_chosen:], rejected_loss_mask, cp_config
        )

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_sft_loss": chosen_sft_loss,
            "rejected_sft_loss": rejected_sft_loss,
            "mean_chosen_logits": mean_chosen_logits,
            "mean_rejected_logits": mean_rejected_logits,
        }

    @staticmethod
    def _masked_logit_mean(logits: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        """Mean over the logits of unmasked positions without materializing a masked copy.

        Equals ``logits[loss_mask].mean()`` (fp32-accumulated); an empty mask yields 0.
        """
        total = (logits.sum(dim=-1, dtype=torch.float32) * loss_mask).sum()
        count = loss_mask.sum() * logits.size(-1)
        return total / count.clamp(min=1)

    def _compute_cp_aggregated_sft_loss(
        self, per_token_nll: torch.Tensor, loss_mask: torch.Tensor, cp_config
    ) -> torch.Tensor:
        """Mean NLL over unmasked shifted tokens, aggregated across the CP group.

        ``per_token_nll`` is ``-per_token_logps`` pre-clip, already zeroed at masked positions — the
        per-token cross-entropy without a second full-vocab pass over the logits. The count comes from
        ``loss_mask`` (never from the NLL's nonzero support: a saturated token's NLL is legitimately
        exactly 0 and must still count). Summing loss and token count and all-reducing both reproduces
        the non-CP ``reduction="mean"`` value regardless of how the sequence was split.
        """
        # fp32 sum + fp32 all-reduce — same bf16-accumulation hazard as the logp sums.
        total_loss = per_token_nll.sum(dtype=torch.float32)
        total_count = loss_mask.sum().float()
        if cp_config is not None and cp_config.cp_size > 1:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM, group=cp_config.process_group)
            dist.all_reduce(total_count, op=dist.ReduceOp.SUM, group=cp_config.process_group)
        return total_loss / total_count.clamp(min=1)

    def _forward_padding_free(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        num_chosen: int,
        device: torch.device,
    ) -> dict[str, torch.Tensor]:
        """Padding-free forward: flatten [2*B, seq_len] → [1, total_non_pad_tokens] to drop padding compute."""
        flat_input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
        flat_labels = labels[attention_mask.bool()].unsqueeze(0)
        position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1

        outputs = model(
            input_ids=flat_input_ids,
            position_ids=position_ids,
            use_cache=False,
        )
        flat_logits = outputs.logits

        shift_logits = flat_logits[:, :-1, :]
        shift_labels = flat_labels[:, 1:]
        shift_position_ids = position_ids[:, 1:]

        loss_mask = (shift_labels != self.label_pad_token_id) & (shift_position_ids != 0)

        shift_labels_masked = shift_labels.clone()
        shift_labels_masked[~loss_mask] = 0

        per_token_logps = selective_log_softmax(shift_logits, shift_labels_masked)
        per_token_logps[~loss_mask] = 0
        flat_nll = -per_token_logps[0]

        seq_idx = (position_ids[0] == 0).cumsum(0) - 1
        shift_seq_idx = seq_idx[1:]
        is_chosen = shift_seq_idx < num_chosen
        flat_logps = per_token_logps[0]
        flat_mask = loss_mask[0]

        if (
            self.lower_clip_percentile is not None
            or self.upper_clip_percentile is not None
            or self.min_log_prob is not None
        ):
            flat_logps = self._clip_log_probs(flat_logps, flat_mask, is_chosen)

        # fp32 scatter-add (bf16 saturates near 256), mask-multiplied — nonzero() would sync.
        num_seqs = 2 * num_chosen
        logp_sums = torch.zeros(num_seqs, device=device, dtype=torch.float32)
        token_counts = torch.zeros(num_seqs, device=device, dtype=torch.float32)
        logp_sums.scatter_add_(0, shift_seq_idx, (flat_logps * flat_mask).float())
        token_counts.scatter_add_(0, shift_seq_idx, flat_mask.float())
        seq_logps = logp_sums / token_counts.clamp(min=1)

        chosen_logps = seq_logps[:num_chosen]
        rejected_logps = seq_logps[num_chosen:]

        chosen_valid = flat_mask & is_chosen
        rejected_valid = flat_mask & ~is_chosen
        shift_logits_flat = shift_logits[0]

        mean_chosen_logits = self._masked_logit_mean(shift_logits_flat, chosen_valid)
        mean_rejected_logits = self._masked_logit_mean(shift_logits_flat, rejected_valid)

        chosen_sft_loss = (flat_nll * chosen_valid).sum(dtype=torch.float32) / chosen_valid.sum().clamp(min=1)
        rejected_sft_loss = (flat_nll * rejected_valid).sum(dtype=torch.float32) / rejected_valid.sum().clamp(min=1)

        return {
            "chosen_logps": chosen_logps,
            "rejected_logps": rejected_logps,
            "chosen_sft_loss": chosen_sft_loss,
            "rejected_sft_loss": rejected_sft_loss,
            "mean_chosen_logits": mean_chosen_logits,
            "mean_rejected_logits": mean_rejected_logits,
        }

    def _cp_quantile(self, values: torch.Tensor, q: float, cp_config) -> torch.Tensor | None:
        """Quantile of ``values`` over the WHOLE sequence, not this CP rank's slice.

        A local ``torch.quantile`` would give each rank a different clip bound, so the CP-aggregated
        sum would drift with ``cp_size``. Gather the (ragged) masked log-probs across the CP group and
        take one global quantile. Returns ``None`` when the group holds no tokens.

        Detached: the bound is a FLOOR, not a term of the objective. Left differentiable, every
        clamped token routes its gradient back through the quantile into the single element that IS
        the quantile, handing that one token ``clamped_count`` times the gradient — the opposite of
        the outlier suppression the clip exists for. The multi-rank branch is detached anyway (the
        all-gather is not autograd-aware), so this also keeps the objective independent of ``cp_size``.
        """
        group = cp_config.process_group if cp_config else None
        if group is None or dist.get_world_size(group) == 1:
            return torch.quantile(values.float(), q).detach() if values.numel() > 0 else None

        world = dist.get_world_size(group)
        counts = torch.zeros(world, dtype=torch.long, device=values.device)
        dist.all_gather_into_tensor(
            counts, torch.tensor([values.numel()], dtype=torch.long, device=values.device), group=group
        )
        if int(counts.sum()) == 0:
            return None

        width = int(counts.max())
        padded = values.float().new_zeros(width)
        padded[: values.numel()] = values.float()
        gathered = padded.new_zeros(world * width)
        dist.all_gather_into_tensor(gathered, padded, group=group)
        gathered = gathered.view(world, width)
        real = torch.cat([gathered[r, : int(counts[r])] for r in range(world)])
        return torch.quantile(real, q)

    def _clip_log_probs(
        self,
        per_token_logps: torch.Tensor,
        loss_mask: torch.Tensor,
        is_chosen: torch.Tensor,
        cp_config=None,
    ) -> torch.Tensor:
        """Percentile-clip the rejected low tail / chosen high tail, then floor the rejected side.

        One rule for both batch shapes: ``is_chosen`` marks the chosen side of ``per_token_logps``
        (a ``[rows, 1]`` column for the padded ``[2*num_chosen, T]`` batch, a flat per-token mask for
        the padding-free row), so only the selector differs between them, never the clip. Bounds come
        from :meth:`_cp_quantile`; the padding-free caller has no ``cp_config`` — ``padding_free`` and
        CP are mutually exclusive, rejected in ``__init__``.

        Order matters: the lower percentile runs before ``min_log_prob``, so the floor can only raise
        the tail further. Masked positions are never touched, and the input is left unmodified.
        """
        clipped = per_token_logps.clone()  # avoid in-place mutation
        chosen_valid = loss_mask & is_chosen
        rejected_valid = loss_mask & ~is_chosen

        if self.lower_clip_percentile is not None:
            lower_bound = self._cp_quantile(clipped[rejected_valid], self.lower_clip_percentile, cp_config)
            if lower_bound is not None:
                clipped = torch.where(rejected_valid & (clipped < lower_bound), lower_bound.to(clipped.dtype), clipped)

        if self.upper_clip_percentile is not None:
            upper_bound = self._cp_quantile(clipped[chosen_valid], self.upper_clip_percentile, cp_config)
            if upper_bound is not None:
                clipped = torch.where(chosen_valid & (clipped > upper_bound), upper_bound.to(clipped.dtype), clipped)

        if self.min_log_prob is not None:
            clipped = torch.where(rejected_valid & (clipped < self.min_log_prob), self.min_log_prob, clipped)

        return clipped

    def _get_current_margin(self) -> float:
        """Current target margin — the scheduler callback writes it onto the model each step.

        Read straight off ``self.model``, which is where the callback writes it (and where the
        toolkit wrappers that keep their own identity — CP, PEFT — are read from anyway). This runs
        once per microbatch, and :meth:`_top_level_model` unwraps recursively, re-``setattr``-ing
        every child module of the whole tree; that walk is paid only on the miss a framework wrapper
        which does not delegate attribute access (DDP) produces.

        No default: ``__init__`` (and the PP re-seed) puts ``target_margin`` on the model whenever
        the schedule is on, so a miss means the seeding path was skipped and the run would silently
        train against the static config margin for the whole schedule.
        """
        if self.use_margin_schedule:
            margin = getattr(self.model, "target_margin", None)
            return margin if margin is not None else self._top_level_model().target_margin
        return self.target_margin

    def _setup_pipeline_parallel(self):
        """Seed the PP-only loss state, then run the mixin's setup.

        Kept out of ``_pp_loss_adapter``, which the mixin may call as a declarative accessor.
        """
        self._pp_step_state = _PPStepState(target_margin=self.target_margin)
        if self.use_margin_schedule:
            # __init__ seeded target_margin on the PRE-SPLIT model, which the stage wrapper does not
            # expose — without re-seeding, a pre-train evaluate() would read the FINAL margin.
            self.model.target_margin = self.initial_margin
        super()._setup_pipeline_parallel()

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """SMPO's pipeline-loss contract: interleaved pairs, closure loss over per-step constants.

        ``pin_runtime_columns=False``: SMPO's dataset columns feed the pair transform, so HF's
        column pruning must keep its own signature set (prompt/chosen/rejected token ids), not the
        runtime contract. Eval runs loss-only through the same closure — the batch transform
        refreshes ``_PPStepState`` from every batch, eval included, before the inert row padding, so
        eval_loss equals the non-PP ``get_batch_loss_metrics`` value.
        """
        return PPLossAdapter(
            token_loss_fn=self._pp_token_loss,
            paired_examples=True,
            batch_transform=self._pp_batch_transform,
            normalizer=self._pp_normalizer,
            rows_per_example=2,  # the pair transform doubles rows: B pairs -> 2B interleaved sequences
            eval_normalizer=self._pp_normalizer,
            pin_runtime_columns=False,
            predictions_fn=self._pp_smpo_predictions,
            eval_labels_fn=self._pp_smpo_eval_labels,
            metrics_fn=self._pp_step_metrics,
            # Prediction and label are both a fixed [2] batch summary, not one entry per pair.
            row_aligned_eval_outputs=False,
        )

    def _pp_smpo_predictions(self, logits: torch.Tensor, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Per-token logits → the ``(mean chosen logits, mean rejected logits)`` pair SMPO predicts.

        :meth:`prediction_step` reports exactly these two batch-level masked logit means, so the
        pipeline must reduce to them on the last stage; without this the raw ``[rows, S, vocab]``
        plane would cross the chain instead, which is not what a metric written against this
        trainer reads. Chosen rows are the even indices of the interleaved batch, rejected the odd;
        inert eval-padding rows are all-ignore, so the loss mask drops them with no separate guard.
        """
        loss_mask = inputs["labels"][:, 1:] != self.label_pad_token_id
        shift_logits = logits[:, :-1, :]
        return torch.stack(
            (
                self._masked_logit_mean(shift_logits[0::2], loss_mask[0::2]),
                self._masked_logit_mean(shift_logits[1::2], loss_mask[1::2]),
            )
        )

    def _pp_smpo_eval_labels(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """The placeholder targets ``prediction_step`` pairs those two means with (no per-token
        target exists for a batch-level statistic)."""
        return torch.zeros(2, device=inputs["input_ids"].device)

    def _pp_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Concat + pair-interleave the preference batch into the pipeline's fixed-shape contract.

        ``concatenated_inputs`` lays the batch out chunk-style ``[chosen ⧺ rejected]``; the pipeline
        tensor-splits along the batch dimension, so that layout would compare unrelated examples —
        re-lay it out as ``[c0, r0, c1, r1, ...]`` (every pair lands in one microbatch). Sequences
        are then right-padded to ``args.max_length`` because the P2P buffer shapes freeze on the
        first step. Also refreshes the per-step loss state: the full-batch normalizers and the live
        scheduled margin, which the per-microbatch loss cannot derive on its own.
        """
        concat = self.concatenated_inputs(
            batch, pad_token_id=self.pad_token_id, label_pad_token_id=self.label_pad_token_id
        )
        num_chosen = concat["num_chosen"]
        inputs = pad_to_pipeline_length(
            {
                key: (interleave_pairs(concat[key][:num_chosen], concat[key][num_chosen:]), fill)
                for key, fill in (
                    ("input_ids", self.pad_token_id),
                    ("attention_mask", 0),
                    ("labels", self.label_pad_token_id),
                )
            },
            self.args.max_length,
            "Lower max_prompt_length/max_completion_length so their sum fits, or raise max_length.",
        )

        # The PP gate pins label_pad_token_id to LABEL_IGNORE_INDEX, so the shared count applies.
        counts = loss_token_counts_per_row(inputs["labels"])
        state = self._pp_step_state
        state.pair_count = num_chosen
        state.chosen_token_count = counts[0::2].sum().float().clamp(min=1)
        state.rejected_token_count = counts[1::2].sum().float().clamp(min=1)
        state.target_margin = self._get_current_margin()
        return inputs

    def _pp_normalizer(self, inputs: dict) -> float:
        """Pin the runtime's per-step divisor to 1 — SMPO normalizes inside the loss closure.

        The objective carries THREE normalizers (pair count for the margin mean, chosen and
        rejected token counts for the SFT term), which cannot ride the runtime's single divisor;
        ``_pp_token_loss`` divides each term by the full-batch constants instead. They are
        rank-local on purpose: the non-PP loss is a rank-local mean whose gradients FSDP averages
        over the DP group, and the PP stage wrap averages over the same group. Doubles as the eval
        divisor — the state constants come from the pre-pad eval batch.
        """
        del inputs
        return 1.0

    @staticmethod
    def _smpo_metric_sums(
        *,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
        chosen_logps: torch.Tensor,
        rejected_logps: torch.Tensor,
        pair_valid: torch.Tensor,
        chosen_logits_sum: torch.Tensor,
        rejected_logits_sum: torch.Tensor,
        chosen_logit_tokens: torch.Tensor,
        rejected_logit_tokens: torch.Tensor,
        chosen_sft_sum: torch.Tensor,
        rejected_sft_sum: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """One batch's metric SUM contributions — the single definition both loss paths build from.

        ``pair_valid`` is a per-pair 0/1 mask, so inert PP eval-padding pairs contribute nothing
        (off PP every pair is valid). Every argument is used as the VALUE it already is and is never
        re-reduced: under CP the log-probs, logit means and SFT losses arriving here are already
        CP-global means, and dividing again would double-count the group. Detached at this seam so
        no metric pins the autograd graph.
        """
        return {
            key: value.detach()
            for key, value in (
                ("rewards/chosen", (chosen_rewards * pair_valid).sum()),
                ("rewards/rejected", (rejected_rewards * pair_valid).sum()),
                ("rewards/accuracies", ((chosen_rewards > rejected_rewards).float() * pair_valid).sum()),
                ("rewards/margins", ((chosen_rewards - rejected_rewards) * pair_valid).sum()),
                ("logps/chosen", (chosen_logps * pair_valid).sum()),
                ("logps/rejected", (rejected_logps * pair_valid).sum()),
                ("logits/chosen", chosen_logits_sum),
                ("logits/rejected", rejected_logits_sum),
                ("logit_tokens/chosen", chosen_logit_tokens),
                ("logit_tokens/rejected", rejected_logit_tokens),
                ("sft_loss/chosen", chosen_sft_sum),
                ("sft_loss/rejected", rejected_sft_sum),
            )
        }

    def _smpo_metrics_from_sums(
        self,
        sums: dict[str, torch.Tensor],
        *,
        pair_count: int | float,
        chosen_token_count: torch.Tensor | float,
        rejected_token_count: torch.Tensor | float,
        target_margin: float,
    ) -> dict[str, torch.Tensor]:
        """Fold metric sums back into the batch's means (unprefixed keys — the caller prefixes).

        The key set is fixed here, never by what the caller accumulated: a missing key reads as
        zero, so a pipeline stage that ran no microbatch reports the same names as one that did.
        Divisors are the caller's full-batch constants; a quantity that already arrives as a mean is
        handed a unit divisor, which is what lets both loss paths share one expression.
        """
        # On the device the step's own constants live on: a caller that accumulated nothing (every
        # pipeline stage but the last) would otherwise divide a CPU zero by a device token count.
        device = next(
            (
                value.device
                for value in (*sums.values(), chosen_token_count, rejected_token_count)
                if isinstance(value, torch.Tensor)
            ),
            None,
        )
        zero = torch.zeros((), dtype=torch.float32, device=device)

        def total(key: str) -> torch.Tensor:
            return sums.get(key, zero)

        pairs = max(float(pair_count), 1.0)
        metrics = {
            key: total(key) / pairs
            for key in (
                "rewards/chosen",
                "rewards/rejected",
                "rewards/accuracies",
                "rewards/margins",
                "logps/chosen",
                "logps/rejected",
            )
        }
        for side in ("chosen", "rejected"):
            metrics[f"logits/{side}"] = total(f"logits/{side}") / total(f"logit_tokens/{side}").clamp(min=1)
        metrics["sft_loss/chosen"] = total("sft_loss/chosen") / chosen_token_count
        metrics["sft_loss/rejected"] = total("sft_loss/rejected") / rejected_token_count
        if self.use_margin_schedule:
            metrics["target_margin"] = zero + target_margin
        return metrics

    def _pp_token_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """SMPO objective over one INTERLEAVED microbatch; contributions SUM to the full-batch loss.

        Each term is divided by the FULL-batch constant from ``_pp_step_state`` (never by this
        microbatch's own counts), so summing the per-microbatch values reproduces exactly the
        non-PP ``get_batch_loss_metrics`` total: margin_losses.mean() over all pairs plus the
        SFT term over all chosen/rejected tokens.
        """
        state = self._pp_step_state
        # token_logprobs masks on LABEL_IGNORE_INDEX; the PP gate pins label_pad_token_id to it.
        token_logps, loss_mask = token_logprobs(logits, labels)
        token_logps = token_logps * loss_mask

        chosen_logps, rejected_logps = split_pairs(token_logps)
        chosen_mask, rejected_mask = split_pairs(loss_mask)

        # SFT anchor on the PRE-clip likelihood (mirrors per_token_nll in concatenated_forward).
        chosen_nll_sum = -chosen_logps.sum()
        rejected_nll_sum = -rejected_logps.sum()

        if self.min_log_prob is not None:
            rejected_logps = torch.where(
                rejected_mask & (rejected_logps < self.min_log_prob), self.min_log_prob, rejected_logps
            )

        chosen_mean_logps = chosen_logps.sum(dim=-1) / chosen_mask.sum(dim=-1).clamp(min=1)
        rejected_mean_logps = rejected_logps.sum(dim=-1) / rejected_mask.sum(dim=-1).clamp(min=1)

        margin_losses, chosen_rewards, rejected_rewards = self.smpo_loss(
            chosen_mean_logps, rejected_mean_logps, target_margin=state.target_margin
        )
        # Inert eval-padding pairs must contribute 0: at zero log-probs the margin loss is nonzero.
        pair_valid = rows_with_labels(labels)[0::2].float()
        sft_term = (
            self.chosen_sft_ratio * chosen_nll_sum / state.chosen_token_count
            + (1 - self.chosen_sft_ratio) * rejected_nll_sum / state.rejected_token_count
        )
        # The logit means carry their own token counts, since a microbatch holds no fixed share of them.
        shift_logits = logits[:, :-1, :].detach()
        chosen_tokens, rejected_tokens = loss_mask[0::2], loss_mask[1::2]
        self._pp_accumulate_metrics(
            self._smpo_metric_sums(
                chosen_rewards=chosen_rewards,
                rejected_rewards=rejected_rewards,
                chosen_logps=chosen_mean_logps,
                rejected_logps=rejected_mean_logps,
                pair_valid=pair_valid,
                chosen_logits_sum=(shift_logits[0::2].float().sum(-1) * chosen_tokens).sum(),
                rejected_logits_sum=(shift_logits[1::2].float().sum(-1) * rejected_tokens).sum(),
                chosen_logit_tokens=chosen_tokens.sum() * shift_logits.size(-1),
                rejected_logit_tokens=rejected_tokens.sum() * shift_logits.size(-1),
                chosen_sft_sum=chosen_nll_sum,
                rejected_sft_sum=rejected_nll_sum,
            )
        )
        return (margin_losses * pair_valid).sum() / state.pair_count + sft_term

    def _pp_accumulate_metrics(self, contributions: dict[str, torch.Tensor]) -> None:
        """Fold one microbatch's metric contributions into the step's running sums.

        Sums, not means: the pipeline hands the loss one microbatch at a time, and only the totals
        over all of them reproduce the whole-batch means ``get_batch_loss_metrics`` reports off PP.
        """
        sums = self._pp_step_state.metric_sums
        for key, value in contributions.items():
            previous = sums.get(key)
            sums[key] = value.detach().float() if previous is None else previous + value.detach().float()

    def _pp_step_metrics(self) -> dict[str, torch.Tensor]:
        """This step's SMPO metrics — the accumulated sums turned back into the batch's means.

        Runs on every rank (the mixin broadcasts the last stage's values down the chain), so the
        key set must not depend on what accumulated: an empty accumulator yields the same names
        with zeros. Divisors are the full-batch constants the batch transform stashed, which is
        what makes the result equal to the non-PP ``get_batch_loss_metrics`` value.
        """
        state = self._pp_step_state
        metrics = self._smpo_metrics_from_sums(
            state.metric_sums,
            pair_count=state.pair_count,
            chosen_token_count=state.chosen_token_count,
            rejected_token_count=state.rejected_token_count,
            target_margin=state.target_margin,
        )
        state.metric_sums.clear()
        return metrics

    def get_batch_loss_metrics(
        self,
        model: nn.Module,
        batch: dict[str, torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | float]]:
        """Compute SMPO loss and metrics for a batch.

        Metrics are detached 0-dim tensors (drained by ``StoredMetricsMixin`` at log time — no
        per-microbatch host sync) and are rank-local estimates: TRL's DPO gathers them across DP
        ranks per compute_loss call, which would reintroduce a per-microbatch collective —
        deliberately not done here.
        """
        prefix = "eval_" if train_eval == "eval" else ""

        model_output = self.concatenated_forward(model, batch)
        current_margin = self._get_current_margin()

        margin_losses, chosen_rewards, rejected_rewards = self.smpo_loss(
            model_output["chosen_logps"],
            model_output["rejected_logps"],
            target_margin=current_margin,
        )

        chosen_sft_loss = model_output["chosen_sft_loss"]
        rejected_sft_loss = model_output["rejected_sft_loss"]

        combined_sft_loss = self.chosen_sft_ratio * chosen_sft_loss + (1 - self.chosen_sft_ratio) * rejected_sft_loss

        total_loss = margin_losses.mean() + combined_sft_loss

        # Both terms are already a CP-global mean, but FSDP2 averages grads over cp_size * dp_size —
        # cancel the 1/cp_size. Any future aux term must be added AFTER this scaling.
        if self.cp_size > 1:
            total_loss = total_loss * self.cp_size

        # Every pair is valid off PP, and the logit means / SFT losses already arrive as means (under
        # CP, CP-global ones) — unit divisors report them as they are instead of re-reducing them.
        unit = torch.ones_like(model_output["mean_chosen_logits"])
        sums = self._smpo_metric_sums(
            chosen_rewards=chosen_rewards,
            rejected_rewards=rejected_rewards,
            chosen_logps=model_output["chosen_logps"],
            rejected_logps=model_output["rejected_logps"],
            pair_valid=torch.ones_like(chosen_rewards),
            chosen_logits_sum=model_output["mean_chosen_logits"],
            rejected_logits_sum=model_output["mean_rejected_logits"],
            chosen_logit_tokens=unit,
            rejected_logit_tokens=unit,
            chosen_sft_sum=chosen_sft_loss,
            rejected_sft_sum=rejected_sft_loss,
        )
        metrics = self._smpo_metrics_from_sums(
            sums,
            pair_count=chosen_rewards.numel(),
            chosen_token_count=1.0,
            rejected_token_count=1.0,
            target_margin=current_margin,
        )

        return total_loss, {f"{prefix}{key}": value for key, value in metrics.items()}

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Compute training loss."""
        with peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Evaluation prediction step."""
        if self._pp_runtime is not None:
            # No pipeline rank holds the full model the path below needs; the mixin drives the
            # schedule instead (loss-only, or loss plus adapter-reduced predictions for metrics).
            return self._pp_prediction_step(inputs, prediction_loss_only, ignore_keys)
        with torch.no_grad(), peft_bf16_autocast(self._peft_has_been_casted_to_bf16, self.accelerator.device):
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return loss.detach(), None, None

        # On the accelerator device to match labels (a CPU-vs-GPU mismatch breaks the eval gather).
        logits = torch.stack([metrics["eval_logits/chosen"], metrics["eval_logits/rejected"]]).to(
            self.accelerator.device
        )
        labels = torch.zeros(2, device=self.accelerator.device)

        return loss.detach(), logits, labels
