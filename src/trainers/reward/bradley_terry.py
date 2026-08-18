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
"""
Distributed Reward Trainer with EP/TP/PP support (CP excluded).

CP is incompatible: reward models pool over the complete sequence, which CP's
per-rank sequence split cannot provide without a full gather. PP is compatible:
the whole sequence lives on the last stage, so the pooled Bradley-Terry loss is
a valid last-stage function once the chosen/rejected pairs are interleaved so no
microbatch split can separate them.

A VLM run (the processor is the ``processing_class``) swaps TRL's text tokenize map and preference
collator for the vision pair — the map in :mod:`src.data.pipeline.preferences`, the collator in
:mod:`src.data.collators.vlm_preference`; everything downstream of the batch — the Bradley-Terry
loss, the metrics, the pooled head — is modality-blind, because image placeholder ids are ordinary
content tokens to the pad-keyed pooling rule.
"""

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from datasets import IterableDataset
from transformers import ProcessorMixin
from trl import RewardTrainer

from src.data.collators.vlm_preference import DataCollatorForVLMPreference
from src.data.pipeline.preferences import (
    MARGIN_COLUMN,
    VLM_PREFERENCE_COLUMNS,
    render_vlm_preference_row,
    vlm_preference_features,
)
from src.data.pipeline.processing import (
    coordinated_filter,
    coordinated_map,
    report_rejected_rows,
)
from src.data.probe_consensus import agree_probe_across_ranks
from src.distributed.pipeline_parallel.losses import (
    PPLossAdapter,
    interleave_pairs,
    pair_count_normalizer,
    split_pairs,
)
from src.models.structure import resolve_tokenizer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.validation import ctor_config
from src.trainers.reward.pooling import (
    decode_pooling_plane,
    encode_pooling_plane,
    pooled_outputs,
    pooling_pad_id_for,
)

logger = get_logger(__name__, log_level="info")

# The two rendered sides; their presence marks a dataset the VLM map has already prepared.
_RENDERED_SIDE_COLUMNS = frozenset(VLM_PREFERENCE_COLUMNS[:2])


def _rendered_sides_fit(row: dict, tokenizer, max_length: int) -> bool:
    """Whether both rendered sides fit ``max_length`` counting TEXT tokens only.

    Vision tokens are not counted: their number is processor-geometry-specific and only known once
    the batch is expanded. This removes the pathological text rows cheaply — the collator's
    over-length raise stays the exact backstop for image-heavy rows near the budget.
    """
    return all(
        len(tokenizer(row[column], add_special_tokens=False)["input_ids"]) <= max_length
        for column in ("chosen_text", "rejected_text")
    )


class DistributedRewardTrainer(DistributedTrainerMixin, RewardTrainer):
    """TRL RewardTrainer plus EP/TP/PP via DistributedTrainerMixin (CP excluded;
    pooled representations are incompatible with sequence splitting)."""

    _tag_names = ["trl", "reward"]

    _supports_pp = True
    # The BT pair scores in ONE forward that is chunked in two, so a MoE layer sees twice the rows.
    _forward_rows_per_example = 2

    _pp_pool_pad_id: int | None = None
    _pp_use_margins: bool = False

    # The VLM run's processor. Held here rather than read off ``processing_class`` because the base
    # ctor is deliberately handed the inner tokenizer (see ``__init__``).
    _processor: ProcessorMixin | None = None

    def __init__(self, *args, **kwargs):
        # The RUN's data path. rewards.py loads a processor only for a run that declares image data
        # (``is_vlm_run``), so the modality of the processing class IS the run's verdict: a text
        # recipe on a multimodal checkpoint arrives with a plain tokenizer and keeps TRL's own
        # tokenize map and collator, byte for byte.
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
                    max_length=getattr(ctor_config(args, kwargs, position=1), "max_length", None),
                )
            # TRL's reward ctor settles the pad token through the TOKENIZER api (``pad_token``,
            # ``get_vocab``), none of which a ProcessorMixin carries — handed a processor it raises
            # AttributeError before the first batch. So the base sees the inner tokenizer and the
            # processor is reinstated below, because ``processing_class`` is what
            # ``save_pretrained`` writes: without it the export carries no processor_config.json
            # and no engine can serve the reward model on images.
            kwargs["processing_class"] = resolve_tokenizer(processing_class)
        kwargs = self._init_distributed_config(kwargs)
        super().__init__(*args, **kwargs)
        if self._is_vlm:
            self.processing_class = self._processor
        self._setup_distributed_modes()

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name: str):
        """Render VLM preference rows; text runs take TRL's own tokenize + filter path unchanged.

        The VLM map renders both sides whole and keeps the prompt's images as PIL objects — TRL's
        text ``tokenize_fn`` is bypassed entirely, since tokenizing before placeholder expansion
        would produce ids that no longer match the batch the model sees. ``max_length`` stays a
        FILTER as it is on the text path, applied to the rendered text (see
        :func:`_rendered_sides_fit`).

        The VLM branch renders through ``self._processor``, not the ``processing_class`` argument:
        the base ctor calls this with the inner tokenizer, which chat-templates without expanding
        image placeholders into the parts-form content the processor understands.
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
            logger.info(f"Dataset '{dataset_name}' already rendered, skipping the VLM preference map")
            return dataset

        num_proc = self._dataset_map_num_proc(args.dataset_num_proc)
        dataset = coordinated_map(
            dataset,
            render_vlm_preference_row,
            num_proc=num_proc,
            # Everything but the margin, which the Bradley-Terry loss reads straight off the batch.
            remove_columns=[column for column in dataset.column_names if column != MARGIN_COLUMN],
            desc=f"Rendering {dataset_name} VLM preference dataset",
            features=vlm_preference_features(dataset),
            # In fn_kwargs, not read off self: the map fn must stay picklable, and the cache key
            # fingerprints fn_kwargs.
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
        # so the verdict is agreed on the world and the raise lands on every rank at once.
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

    def _setup_pipeline_parallel(self):
        """Seed the PP-only pooling state and refuse a head this objective cannot score, then run
        the mixin's setup.

        Kept out of ``_pp_loss_adapter``, which the mixin may call as a declarative accessor.
        """
        self._pp_pool_pad_id = pooling_pad_id_for(self)
        if self.model.config.num_labels != 1:
            raise ValueError(
                f"Bradley-Terry reward modeling under pipeline parallelism needs a scalar head "
                f"(num_labels=1), got num_labels={self.model.config.num_labels}."
            )
        columns = getattr(self.train_dataset, "column_names", None) or []
        self._pp_use_margins = "margin" in columns
        super()._setup_pipeline_parallel()

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """The Bradley-Terry objective as a last-stage pipeline loss.

        The collator's ``[chosen ⧺ rejected]`` is re-laid out interleaved so no microbatch split can
        separate a pair, and each row's pooled position is recorded as a marker in the synthesized
        labels plane — the one per-row channel the runtime already ships. The loss therefore pools,
        and reads row validity, straight off that plane: no ``input_ids`` side tensor and no second
        pooling pass. ``eval_normalizer`` counts the marked rows, so eval_loss means over pairs —
        the unit the BT loss means over. ``extra_signature_columns`` keeps the preference dataset's
        own columns past HF's pruning.
        """
        return PPLossAdapter(
            token_loss_fn=self._pp_bt_token_loss,
            paired_examples=True,
            batch_transform=self._pp_reward_batch_transform,
            normalizer=pair_count_normalizer,
            extra_target_keys=("margin",) if self._pp_use_margins else (),
            rows_per_example=2,
            eval_normalizer=self._pp_bt_eval_pair_count,
            extra_signature_columns=("chosen_ids", "rejected_ids", "margin"),
            predictions_fn=self._pp_bt_predictions,
            eval_labels_fn=self._pp_bt_eval_labels,
        )

    def _pp_bt_predictions(self, logits: torch.Tensor, inputs: dict) -> torch.Tensor:
        """Per-token scores → the ``[pairs, 2]`` (chosen, rejected) a preference metric consumes.

        Without this the raw ``[rows, S, 1]`` plane would reach ``compute_metrics``, which for a
        reward model is not a prediction at all — the score lives at one pooled position per row.
        The pairwise layout matches the convention preference metrics expect, so
        ``argmax == 0`` is the accuracy.
        """
        _valid, positions, _values = decode_pooling_plane(inputs["labels"])
        pooled = pooled_outputs(logits, positions).squeeze(-1).float()
        chosen, rejected = split_pairs(pooled)
        return torch.stack((chosen, rejected), dim=-1)

    def _pp_bt_eval_labels(self, inputs: dict) -> torch.Tensor:
        """Chosen is column 0 of every pair, so the target index is 0 — one label per pair."""
        pairs = inputs["input_ids"].size(0) // 2
        return torch.zeros(pairs, dtype=torch.long, device=inputs["input_ids"].device)

    def _pp_reward_batch_transform(self, batch: dict) -> dict:
        """``[chosen ⧺ rejected]`` → interleaved ``[c0, r0, ...]``, plus the synthesized labels plane."""
        chosen_ids, rejected_ids = batch["input_ids"].chunk(2)
        chosen_mask, rejected_mask = batch["attention_mask"].chunk(2)
        input_ids = interleave_pairs(chosen_ids, rejected_ids)
        attention_mask = interleave_pairs(chosen_mask, rejected_mask)
        # Both rows of the pair are marked: the loss needs a chosen AND a rejected score per pair.
        labels = encode_pooling_plane(input_ids, self._pp_pool_pad_id, 0)
        out = {**batch, "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
        if self._pp_use_margins:
            margin = batch.get("margin")
            if margin is None:
                raise ValueError(
                    "The PP loss adapter was resolved WITH margins (the train dataset has a 'margin' "
                    "column) but this batch carries none. A zero fill would silently train the "
                    "margin-free objective on exactly these rows. Use a margin column in both "
                    "datasets or in neither."
                )
            out["margin"] = margin.repeat_interleave(2)
        elif "margin" in batch:
            raise ValueError(
                "This batch carries a 'margin' column but the train dataset had none, so the PP "
                "loss adapter was resolved without margins and would silently ignore them. Use a "
                "margin column in both datasets or in neither."
            )
        return out

    def _pp_bt_token_loss(self, logits: torch.Tensor, target: torch.Tensor | dict) -> torch.Tensor:
        """Summed BT loss over the microbatch's pairs (the runtime divides by the pair count)."""
        if isinstance(target, dict):
            plane, margin = target["labels"], target.get("margin")
        else:
            plane, margin = target, None
        valid, positions, _ = decode_pooling_plane(plane)
        pooled = pooled_outputs(logits, positions).squeeze(-1).float()
        chosen, rejected = split_pairs(pooled)
        # Eval row-padding appends inert rows whose pooled score is garbage — mask those pairs.
        chosen_valid, rejected_valid = split_pairs(valid)
        pair_valid = (chosen_valid & rejected_valid).float()
        difference = chosen - rejected
        if margin is not None:
            difference = difference - margin.float()[0::2]
        loss = -(F.logsigmoid(difference) * pair_valid).sum()
        coefficient = self.args.center_rewards_coefficient
        if coefficient is not None:
            loss = loss + coefficient * (((chosen + rejected) ** 2) * pair_valid).sum()
        return loss

    def _pp_bt_eval_pair_count(self, inputs: dict) -> torch.Tensor:
        """Real-pair count for the loss-only PP eval, read off the synthesized labels plane.

        The batch transform marks every row and the eval row-padding appends only all-ignore rows,
        so half the marked-row count IS the batch's real pair count — the unit the BT loss means over.
        """
        valid, _, _ = decode_pooling_plane(inputs["labels"])
        return (valid.sum() / 2).clamp(min=1.0)
