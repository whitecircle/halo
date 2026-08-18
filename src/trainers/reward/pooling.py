"""Pooled sequence-classification seams shared by the reward and classification trainers.

Transformers scores a sequence-classification head at each row's rightmost non-pad token. Under
pipeline parallelism the last stage emits per-token head outputs, so that pooling moves out of the
model and into the trainer: the batch transform applies the rule and encodes the result into the
labels plane the runtime already ships, and the last stage's loss decodes it back.
"""

import torch

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.pipeline_parallel.losses import rows_with_labels
from src.models.structure import resolve_tokenizer


def sequence_classification_pad_id(config, tokenizer_pad_id: int | None) -> int:
    """The pad id transformers' sequence-classification pooling keys on, checked against the padder.

    Raises when unset: pooling picks the rightmost token != ``config.pad_token_id``, and
    transformers itself rejects batches > 1 without it — a PP batch always has several rows.

    Also raises when the two ids disagree. PP pads every batch to ``max_length`` with the
    tokenizer's pad id while pooling keys on the config's; if they differ, no filler token counts as
    padding and every row pools at ``max_length - 1``, scoring a pad token at a finite loss. The
    training scripts arrive here already synced (``setup_model_and_tokenizer`` records the
    tokenizer's pad id on every config level that declares it), so both raises gate a hand-built
    trainer whose config and tokenizer disagree, or whose tokenizer carries no pad id.
    """
    pad_id = config.get_text_config().pad_token_id
    if pad_id is None:
        raise ValueError(
            "Sequence-classification pooling under pipeline parallelism picks each row's rightmost "
            "token != config.pad_token_id, but the model config has pad_token_id=None (transformers "
            "itself cannot pool batches > 1 in this state). The setup seam records the tokenizer's "
            "pad id for you, so reaching here means the tokenizer settled on none either: give it "
            "one with --pad_token (a token already in the vocabulary) or --added_special_tokens (to "
            "add the token first)."
        )
    if tokenizer_pad_id is not None and tokenizer_pad_id != pad_id:
        raise ValueError(
            f"config.pad_token_id ({pad_id}) disagrees with the tokenizer's pad_token_id "
            f"({tokenizer_pad_id}). Pipeline parallelism pads every batch to max_length with the "
            f"tokenizer's id while pooling keys on the config's, so every row would pool at the last "
            f"filler token instead of its last real one — a finite loss that learns nothing. Set "
            f"model.config.pad_token_id = tokenizer.pad_token_id (or pass --pad_token)."
        )
    return pad_id


def pooling_pad_id_for(trainer) -> int:
    """The pooling pad id for ``trainer``'s PP loss adapter, read off its model and tokenizer.

    Both pooled trainers resolve it identically at ``_setup_pipeline_parallel`` and assign it
    themselves; the two raises that guard it are in :func:`sequence_classification_pad_id`.
    """
    tokenizer = resolve_tokenizer(trainer.processing_class)
    return sequence_classification_pad_id(trainer.model.config, getattr(tokenizer, "pad_token_id", None))


def last_non_pad_positions(input_ids: torch.Tensor, pad_token_id: int) -> torch.Tensor:
    """Rightmost position whose token != ``pad_token_id``, per row.

    Transformers' exact sequence-classification pooling rule
    (``GenericForSequenceClassification``): handles left- and right-padding alike; an all-pad
    row yields position 0. It must match the model's own rule exactly, or PP would score different
    tokens than the unsplit model.
    """
    non_pad_mask = (input_ids != pad_token_id).to(torch.int32)
    token_indices = torch.arange(input_ids.shape[-1], device=input_ids.device, dtype=torch.int32)
    return (token_indices * non_pad_mask).argmax(dim=-1)


def encode_pooling_plane(input_ids: torch.Tensor, pad_token_id: int, values: int | torch.Tensor) -> torch.Tensor:
    """``[B, S]`` labels plane: ``LABEL_IGNORE_INDEX`` except ``values`` at each row's pooled position.

    The plane is the only per-row channel the pipeline runtime ships unconditionally, so encoding
    the pooling result into it hands the last stage both the position and the row's target with no
    side tensors, and makes the eval path's inert row padding self-identifying — which the raw
    ``input_ids`` are not, their filler carrying the tokenizer's pad id rather than the config's.
    """
    plane = torch.full_like(input_ids, LABEL_IGNORE_INDEX)
    rows = torch.arange(input_ids.size(0), device=input_ids.device)
    plane[rows, last_non_pad_positions(input_ids, pad_token_id)] = values
    return plane


def decode_pooling_plane(plane: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Inverse of :func:`encode_pooling_plane` — ``(valid_rows, positions, values)``.

    Correct on a whole batch or on one microbatch of it. Rows with no marker are the PP eval path's
    row padding and carry no loss.
    """
    positions = (plane != LABEL_IGNORE_INDEX).int().argmax(dim=-1)
    rows = torch.arange(plane.size(0), device=plane.device)
    return rows_with_labels(plane), positions, plane[rows, positions]


def pooled_outputs(per_token_outputs: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """One pooled vector per row, gathered from a ``[B, S, ...]`` per-token head output."""
    rows = torch.arange(per_token_outputs.size(0), device=per_token_outputs.device)
    return per_token_outputs[rows, positions]
