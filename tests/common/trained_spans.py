"""What a completion mask trains, read back as text, and the tree-row map's mask of one conversation as labels."""

from halo.data.pipeline.tree_rows import build_tree_row_processor
from halo.data.tree_layout import TRAINED_COLUMN
from halo.kernels.logprobs import LABEL_IGNORE_INDEX


def trained_pieces(tokenizer, input_ids, labels) -> list[str]:
    """Each run of positions whose label is not the ignore index, decoded."""
    pieces, run = [], []
    for token, label in zip(input_ids, labels, strict=True):
        if label != LABEL_IGNORE_INDEX:
            run.append(token)
        elif run:
            pieces.append(tokenizer.decode(run))
            run = []
    return [*pieces, tokenizer.decode(run)] if run else pieces


def tree_row_labels(tokenizer, conversation, tools, *, eos_token_ids, marker) -> tuple[list[int], list[int]]:
    """``conversation``, alone in its tree, through the tree-row map: its token ids, and as labels the ids it trains."""
    processor = build_tree_row_processor(
        tokenizer,
        4096,
        conversation_field="messages",
        share_markers=(),
        share_requires=None,
        eos_token_ids=eos_token_ids,
        assistant_message_template=marker,
        train_on_last_assistant_only=False,
        tools_field="tools",
    )
    tree = processor({"messages": [conversation], "tools": [tools]})
    (input_ids,), (trained,) = tree["input_ids"], tree[TRAINED_COLUMN]
    return input_ids, [token if keep else LABEL_IGNORE_INDEX for token, keep in zip(input_ids, trained, strict=True)]
