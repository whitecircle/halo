"""Tree-packing collator: packed rows of prefix-sharing trees as one flattened row plus the index
tensors that give every token exactly its own path as context."""

from itertools import chain
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from halo.data.collators.packing import DataCollatorWithPacking
from halo.data.tree_layout import DEPTH_COLUMN, POSITION_COLUMN, TRAINED_COLUMN
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.models.attention.tree_plan import (
    FULL_ATTENTION_FORMS,
    TREE_ATTENTION_KEY,
    FullAttentionForm,
    build_node_table,
    build_tree_attention_plan,
)


class DataCollatorWithTreePacking(DataCollatorWithPacking):
    """Collate rows packed from trees (:mod:`halo.data.pipeline.tree_rows`) into one ``[1, total]`` row whose
    labels are the trained tokens, plus the :func:`~halo.models.attention.tree_plan.build_tree_attention_plan`
    under ``tree_attention`` (skipped for plain documents unless ``conv_kernel`` layers or key-selecting
    ``sparse_attention`` layers read their segments off it).

    A packing collator by class, so every trainer gate that matches one applies; ``sliding_window``,
    ``full_form``, ``conv_kernel`` and ``sparse_attention`` are the plan's.
    """

    required_dataset_columns: tuple[str, ...] = (TRAINED_COLUMN, POSITION_COLUMN, DEPTH_COLUMN)

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        sliding_window: int | None,
        full_form: FullAttentionForm,
        conv_kernel: int | None = None,
        sparse_attention: bool = False,
    ):
        if full_form not in FULL_ATTENTION_FORMS:
            raise ValueError(f"full_form must be one of {FULL_ATTENTION_FORMS}, got {full_form!r}.")
        super().__init__(tokenizer=tokenizer)
        self.sliding_window = sliding_window
        self.full_form = full_form
        self.conv_kernel = conv_kernel
        self.sparse_attention = sparse_attention

    def torch_call(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.flatten_to_single_row:
            raise ValueError(
                "tree packing flattens the batch into one row and cannot keep packed rows apart, "
                "which pipeline parallelism requires. Disable tree packing under pipeline parallelism."
            )

        def column(name: str, dtype: torch.dtype) -> torch.Tensor:
            return torch.tensor(list(chain.from_iterable(example[name] for example in examples)), dtype=dtype)

        input_ids, position_ids = column("input_ids", torch.long), column(POSITION_COLUMN, torch.long)
        trained = column(TRAINED_COLUMN, torch.bool)
        table = build_node_table(position_ids, column(DEPTH_COLUMN, torch.long))
        if trained[table.starts].any():
            raise ValueError(
                "a tree node starts on a trained token: the loss pairs every label with the token "
                "before it in the row, which across a node boundary belongs to another branch."
            )
        labels = input_ids.masked_fill(~trained, LABEL_IGNORE_INDEX)
        batch = {"input_ids": input_ids[None], "labels": labels[None], POSITION_COLUMN: position_ids[None]}
        if table.has_branches or self.conv_kernel is not None or self.sparse_attention:
            batch[TREE_ATTENTION_KEY] = build_tree_attention_plan(
                table,
                sliding_window=self.sliding_window,
                full_form=self.full_form,
                conv_kernel=self.conv_kernel,
                sparse_attention=self.sparse_attention,
            )
        return batch
