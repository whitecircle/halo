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
"""Distributed Bradley-Terry reward trainer: EP, TP and PP; CP unsupported (pooling needs the whole sequence).

Under PP the whole sequence lives on the last stage, so the pooled loss is a last-stage function once the
pairs are interleaved. A VLM run swaps TRL's tokenize map and collator for the vision pair
(:mod:`src.data.pipeline.preferences`, :mod:`src.data.collators.vlm_preference`).
"""

import logging
from functools import partial

import torch
import torch.nn.functional as F
from datasets import IterableDataset
from transformers import ProcessorMixin
from trl import RewardTrainer

from src.data.collators.vlm_preference import DataCollatorForVLMPreference
from src.data.pipeline.coordinated_map import coordinated_filter, coordinated_map, resolve_map_num_proc
from src.data.pipeline.preferences import (
    MARGIN_COLUMN,
    VLM_PREFERENCE_COLUMNS,
    render_vlm_preference_row,
    vlm_preference_features,
)
from src.data.pipeline.rejected_rows import report_rejected_rows
from src.model_loading.tokenizer_setup import resolve_tokenizer
from src.runtime.consensus import agree_probe_across_ranks
from src.runtime.ranks import is_global_main_process
from src.trainers.ctor_slots import get_ctor_config, resolve_ctor_positions
from src.trainers.distributed_trainer import DistributedTrainerMixin
from src.trainers.parallelism.pipeline_loss_adapter import (
    PipelineLossAdapter,
    decode_pooling_plane,
    encode_pooling_plane,
    interleave_halves,
    pair_count_normalizer,
    pair_placeholder_labels,
    pooled_logits,
    split_pairs,
)
from src.trainers.parallelism.pipeline_rules import require_pooling_pad_id
from src.trainers.trl_dataset_prep import TRLDatasetPrepMixin

logger = logging.getLogger(__name__)

# The two rendered sides; their presence marks a dataset the VLM map has already prepared.
_RENDERED_SIDE_COLUMNS = frozenset(VLM_PREFERENCE_COLUMNS[:2])


def _rendered_sides_fit(row: dict, tokenizer, max_length: int) -> bool:
    """Whether both rendered sides fit ``max_length`` counting text tokens only.

    Vision tokens are not counted: their number is processor-geometry-specific and only known once
    the batch is expanded. This drops the worst text rows cheaply; the collator's over-length raise
    remains the backstop for image-heavy rows near the budget.
    """
    return all(
        len(tokenizer(row[column], add_special_tokens=False)["input_ids"]) <= max_length
        for column in _RENDERED_SIDE_COLUMNS
    )


class DistributedRewardTrainer(DistributedTrainerMixin, TRLDatasetPrepMixin, RewardTrainer):
    """TRL RewardTrainer plus EP/TP/PP via DistributedTrainerMixin (CP excluded: the pooled score needs the
    whole sequence)."""

    _tag_names = ["trl", "reward"]

    PP_REFUSAL = None
    # TRL RewardTrainer's positional slots, derived from its installed signature.
    CTOR_POSITIONS = resolve_ctor_positions(RewardTrainer, "model", "args")
    # The BT pair scores in one forward that is chunked in two, so a MoE layer sees twice the rows.
    FORWARD_ROWS_PER_EXAMPLE = 2

    _pp_pool_pad_id: int | None = None
    _pp_use_margins: bool = False

    # The VLM run's processor. Held here rather than read off ``processing_class`` because the base
    # ctor is handed the inner tokenizer (see ``__init__``).
    _processor: ProcessorMixin | None = None

    def __init__(self, *args, **kwargs):
        # The run's data path. rewards.py loads a processor only for a run that declares image data
        # (``is_vlm_run``), so the modality of the processing class determines it: a text recipe on
        # a multimodal checkpoint arrives with a plain tokenizer and keeps TRL's own tokenize map
        # and collator unchanged.
        processing_class = kwargs.get("processing_class")
        self._is_vlm = processing_class is not None and resolve_tokenizer(processing_class) is not processing_class
        if self._is_vlm:
            self._processor = processing_class
            if kwargs.get("data_collator") is None:
                # Built here, not in the script: the collator and ``_prepare_dataset`` are one
                # contract (rendered-text columns in, [chosen ⧺ rejected] out) and must not be
                # paired by hand.
                kwargs["data_collator"] = DataCollatorForVLMPreference(
                    processor=processing_class,
                    max_length=getattr(get_ctor_config(args, kwargs, self.CTOR_POSITIONS), "max_length", None),
                )
            # TRL's reward ctor settles the pad token through the tokenizer api (``pad_token``,
            # ``get_vocab``), none of which a ProcessorMixin carries — handed a processor it raises
            # AttributeError before the first batch. The base therefore sees the inner tokenizer,
            # and the processor is reinstated below because ``save_pretrained`` writes
            # ``processing_class``: without it the export carries no processor_config.json.
            kwargs["processing_class"] = resolve_tokenizer(processing_class)
        kwargs = self._init_distributed_config(kwargs, ctor_args=args, ctor_positions=self.CTOR_POSITIONS)
        super().__init__(*args, **kwargs)
        if self._is_vlm:
            self.processing_class = self._processor
        self._configure_distributed_modes()

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name: str):
        """Render VLM preference rows; text runs take TRL's own tokenize + filter path unchanged.

        The VLM map renders both sides whole, keeping the prompt's images as PIL objects: tokenizing before
        placeholder expansion would not match the batch the model sees. It renders through ``self._processor``,
        since the base ctor passes the inner tokenizer, which cannot expand image placeholders.
        """
        if not self._is_vlm:
            return super()._prepare_dataset(dataset, processing_class, args, dataset_name)
        processor = self._processor
        if isinstance(dataset, IterableDataset):
            raise ValueError(
                "VLM reward modeling needs a map-style dataset: the image render is a coordinated "
                "dataset map with a pinned Arrow schema, which an IterableDataset cannot carry. "
                "Load the preference data as a regular dataset."
            )
        if _RENDERED_SIDE_COLUMNS.issubset(dataset.column_names):
            if is_global_main_process():
                logger.info(f"Dataset '{dataset_name}' already rendered, skipping the VLM preference map")
            return dataset

        num_proc = resolve_map_num_proc(args.dataset_num_proc)
        dataset = coordinated_map(
            dataset,
            render_vlm_preference_row,
            num_proc=num_proc,
            # Everything but the margin, which the Bradley-Terry loss reads straight off the batch.
            remove_columns=[column for column in dataset.column_names if column != MARGIN_COLUMN],
            desc=f"Rendering {dataset_name} VLM preference dataset",
            features=vlm_preference_features(dataset),
            fn_kwargs={"processing_class": processor},
        )
        if args.max_length is None:
            return dataset
        before = len(dataset)
        dataset = coordinated_filter(
            dataset,
            _rendered_sides_fit,
            desc=f"Dropping over-length VLM {dataset_name} rows",
            num_proc=num_proc,
            fn_kwargs={"tokenizer": resolve_tokenizer(processor), "max_length": args.max_length},
        )
        # A per-rank fact on a presharded corpus; the trainer construction that follows is collective,
        # so the check is agreed across the world and the raise lands on every rank at once.
        if agree_probe_across_ranks(
            len(dataset) == 0, f"the {dataset_name} split", "split emptied by the length filter"
        ):
            raise ValueError(
                f"All {before} rows of the {dataset_name} split exceed max_length={args.max_length} on at "
                f"least one data-parallel rank, by rendered text alone (image tokens are on top of that) — "
                f"raise the budget."
            )
        report_rejected_rows(
            before,
            len(dataset),
            f"the VLM {dataset_name} split's max_length={args.max_length} (text tokens alone)",
        )
        return dataset

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        """The Bradley-Terry objective as a last-stage pipeline loss, over a scalar head.

        The collator's ``[chosen ⧺ rejected]`` is interleaved so no microbatch split separates a pair, and
        each row's pooled position rides the synthesized labels plane, from which the loss pools and reads
        row validity. ``extra_signature_columns`` keeps the preference columns past HF's pruning.
        """
        tokenizer = resolve_tokenizer(self.processing_class)
        self._pp_pool_pad_id = require_pooling_pad_id(self.model.config, getattr(tokenizer, "pad_token_id", None))
        if self.model.config.num_labels != 1:
            raise ValueError(
                f"Bradley-Terry reward modeling under pipeline parallelism needs a scalar head "
                f"(num_labels=1), got num_labels={self.model.config.num_labels}."
            )
        self._pp_use_margins = MARGIN_COLUMN in (getattr(self.train_dataset, "column_names", None) or [])
        return PipelineLossAdapter(
            token_loss_fn=self._pp_bt_token_loss,
            batch_transform=self._pp_reward_batch_transform,
            normalizer=pair_count_normalizer,
            extra_target_keys=(MARGIN_COLUMN,) if self._pp_use_margins else (),
            # Every row is marked and the eval padding appends unmarked rows: half the marked rows are pairs.
            eval_normalizer=lambda inputs: (decode_pooling_plane(inputs["labels"]).valid.sum() / 2).clamp(min=1.0),
            extra_signature_columns=("chosen_ids", "rejected_ids", MARGIN_COLUMN),
            # The [pairs, 2] (chosen, rejected) scores preference metrics read (argmax == 0 is the accuracy),
            # not the [rows, S, 1] plane, where the score sits at one pooled position per row.
            predictions_fn=lambda logits, inputs: torch.stack(
                split_pairs(pooled_logits(logits, decode_pooling_plane(inputs["labels"])).squeeze(-1).float()), dim=-1
            ),
            eval_labels_fn=partial(pair_placeholder_labels, dtype=torch.long),
        )

    def _pp_reward_batch_transform(self, batch: dict) -> dict:
        """``[chosen ⧺ rejected]`` → interleaved ``[c0, r0, ...]``, plus the synthesized labels plane."""
        input_ids = interleave_halves(batch["input_ids"])
        attention_mask = interleave_halves(batch["attention_mask"])
        # Both rows of the pair are marked: the loss needs a chosen and a rejected score per pair.
        labels = encode_pooling_plane(input_ids, self._pp_pool_pad_id, 0)
        out = {**batch, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if self._pp_use_margins:
            margin = batch.get(MARGIN_COLUMN)
            if margin is None:
                raise ValueError(
                    "The PP loss adapter was resolved WITH margins (the train dataset has a 'margin' "
                    "column) but this batch carries none. A zero fill would silently train the "
                    "margin-free objective on exactly these rows. Use a margin column in both "
                    "datasets or in neither."
                )
            out[MARGIN_COLUMN] = margin.repeat_interleave(2)
        elif MARGIN_COLUMN in batch:
            raise ValueError(
                "This batch carries a 'margin' column but the train dataset had none, so the PP "
                "loss adapter was resolved without margins and would silently ignore them. Use a "
                "margin column in both datasets or in neither."
            )
        return out

    def _pp_bt_token_loss(self, logits: torch.Tensor, target: torch.Tensor | dict) -> torch.Tensor:
        """Summed BT loss over the microbatch's pairs (the runner divides by the pair count)."""
        if isinstance(target, dict):
            labels, margin = target["labels"], target.get(MARGIN_COLUMN)
        else:
            labels, margin = target, None
        plane = decode_pooling_plane(labels)
        chosen, rejected = split_pairs(pooled_logits(logits, plane).squeeze(-1).float())
        chosen_valid, rejected_valid = split_pairs(plane.valid)
        pair_valid = (chosen_valid & rejected_valid).float()
        difference = chosen - rejected
        if margin is not None:
            difference = difference - margin.float()[0::2]
        loss = -(F.logsigmoid(difference) * pair_valid).sum()
        coefficient = self.args.center_rewards_coefficient
        if coefficient is not None:
            loss = loss + coefficient * (((chosen + rejected) ** 2) * pair_valid).sum()
        return loss
