"""Utilities for Vision-Language Model (VLM) processing."""

import base64
import io
import json
from typing import Any

from datasets import Features, Sequence, Value
from datasets import Image as ImageFeature
from PIL import Image

from src.data.pipeline.conversation import (
    IMAGE_PART_PAYLOAD_KEY,
    IMAGE_PART_TYPE,
    conversation_carries_images,
    fold_system_into_conversation,
)
from src.data.probe_consensus import agree_probe_across_ranks
from src.models.modality import is_vlm_model

# The stored schema of a preprocessed VLM dataset: pinned on the offline map (else an all-None batch
# infers null Arrow columns) and declared by :class:`~src.data.collators.vlm.PreprocessedVLMDataCollator`
# (else TRL's signature-column pruning drops the difference before collation).
VLM_OUTPUT_FEATURES = Features(
    {
        "input_ids": Sequence(Value("int64")),
        "attention_mask": Sequence(Value("int64")),
        "labels": Sequence(Value("int64")),
        "pixel_values": Value("binary"),
        "pixel_values_shape": Sequence(Value("int64")),
        "image_grid_thw": Sequence(Sequence(Value("int64"))),
    }
)
VLM_OUTPUT_COLUMNS = tuple(VLM_OUTPUT_FEATURES)

# The raw spellings a dataset ships images in, in resolution order (``images`` wins). TRL's own
# vision-dataset probe and the SMPO / reward vision routes read the same pair.
VLM_RAW_IMAGE_COLUMNS = ("images", "image")

# Every column that carries a run's images: the raw pair above plus ``pixel_values``, the pixels a
# preprocessed ``--vlm`` artifact stores (:data:`VLM_OUTPUT_COLUMNS`) for
# :class:`~src.data.collators.vlm.PreprocessedVLMDataCollator`.
VLM_IMAGE_COLUMNS = (*VLM_RAW_IMAGE_COLUMNS, "pixel_values")


def carried_image_columns(dataset) -> set[str]:
    """The :data:`VLM_IMAGE_COLUMNS` spellings present in any split of ``dataset``.

    Shared by the image-column guards (the text-only refusal, the offline bake's unconsumed-column
    refusal and the multimodality verdict) so all three read a dataset the same way. ``dataset`` may
    be a ``DatasetDict`` or a single split.
    """
    splits = dataset.values() if isinstance(dataset, dict) else [dataset]
    return {column for split in splits for column in split.column_names} & set(VLM_IMAGE_COLUMNS)


def vlm_row_tools(row: dict[str, Any]) -> list | None:
    """The row's tool schemas, decoded from the ``tools_json`` column the VLM map writes."""
    tools_json = row.get("tools_json")
    return json.loads(tools_json) if tools_json else None


def render_vlm_text(processor, history: list[dict[str, Any]], *, tools: list | None = None) -> str:
    """Chat-template one VLM conversation, shared by every VLM path.

    Takes the placeholder history :func:`process_vlm_conversation` produces, which
    ``apply_chat_template`` expands into the family's vision tokens. Runtime collation, the
    over-length pre-filter, the offline bake and the preference/reward renders (SMPO, the VLM
    preference map) all render here, so a preprocessed artifact, an untouched run and a preference
    pair tokenize the same conversation identically.
    """
    # ``is None``, not truthiness, and matching ``chat_template_kwargs`` on the text path: a row
    # declaring ``tools: []`` renders under a template branch that tests ``tools is defined``.
    template_kwargs = {} if tools is None else {"tools": tools}
    return processor.apply_chat_template(history, tokenize=False, add_generation_prompt=False, **template_kwargs)


def run_vlm_processor(processor, texts, images, **overrides):
    """Invoke a VLM processor on rendered chat-template text.

    ``add_special_tokens=False``: the rendered template already carries every special token;
    templates that emit BOS (LFM2.5-VL) would otherwise get a second one from the tokenizer
    post-processor. A ``StopIteration`` escaping the processor (image placeholder count vs provided
    images mismatch) is re-raised as ``ValueError``, since a ``StopIteration`` leaked from a collator
    ends the dataloader and training completes with zero steps and no error.
    """
    kwargs = {"text": texts, "return_tensors": "pt", "add_special_tokens": False, **overrides}
    if images:
        kwargs["images"] = images
    try:
        return processor(**kwargs)
    except StopIteration as exc:
        raise ValueError(
            "Processor raised StopIteration — the text's image placeholder count does not match the provided images."
        ) from exc


def raise_if_over_length(encoded, max_length: int | None, subject: str) -> None:
    """Raise when a processed VLM batch carries a sequence over the run's ``max_length``.

    Image placeholders expand at collation, so this is the only point that sees the real length, and
    neither remedy a text collator has is available here. ``None`` disables the check; ``subject``
    names the batch (the SDPG teacher and student batches are checked separately).
    """
    if max_length is None:
        return
    attention_mask = encoded.get("attention_mask")
    longest = int(attention_mask.sum(dim=-1).max()) if attention_mask is not None else encoded["input_ids"].shape[-1]
    if longest <= max_length:
        return
    raise ValueError(
        f"VLM {subject} contains a {longest}-token sequence, {longest - max_length} tokens over "
        f"max_length={max_length}, and image expansion happens at collation so a text-only pre-filter "
        f"could not see it. VLM sequences cannot be truncated at collation (cutting expanded image "
        f"placeholder tokens desyncs them from pixel_values/image_grid_thw), and a runtime batch cannot "
        f"drop rows without desyncing DP ranks. Raise max_length, pre-filter over-length rows (e.g. via "
        f"dataset preprocessing), or use smaller images."
    )


def process_image(image_data) -> Image.Image:
    """Process image data from various formats to PIL Image."""
    if isinstance(image_data, str):
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
            image = Image.open(io.BytesIO(base64.b64decode(image_data)))
        else:
            image = Image.open(image_data).convert("RGB")
    elif isinstance(image_data, Image.Image):
        image = image_data
    elif isinstance(image_data, bytes):
        image = Image.open(io.BytesIO(image_data))
    else:
        raise ValueError(f"Unsupported image format: {type(image_data)}")

    return image


def get_image_token_ids(tokenizer, processor=None) -> set:
    """Image/video placeholder token IDs to mask out of labels (they must not contribute to loss).

    Unions several sources: a missed placeholder produces no error, its vision-pad tokens are simply
    trained as text.
    """
    image_token_ids = set()

    for obj in (processor, getattr(processor, "tokenizer", None), tokenizer):
        token_id = getattr(obj, "image_token_id", None) if obj is not None else None
        if token_id is not None:
            image_token_ids.add(token_id)
        video_id = getattr(obj, "video_token_id", None) if obj is not None else None
        if video_id is not None:
            image_token_ids.add(video_id)

    vocab = tokenizer.get_vocab()
    common_image_tokens = [
        "<image>",
        "<img>",
        "<|image|>",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
        "<image_soft_token>",
    ]
    for token in common_image_tokens:
        if token in vocab:
            image_token_ids.add(vocab[token])

    for obj in (processor, tokenizer):
        token = getattr(obj, "image_token", None) if obj is not None else None
        if isinstance(token, str) and token in vocab:
            image_token_ids.add(vocab[token])

    return image_token_ids


def _list_element_feature(feature):
    """The element feature of a list-shaped features node, ``None`` when it is not list-shaped.

    Covers every spelling a conversation column arrives in: ``List``/``LargeList``/``Sequence``
    (all ``.feature``) and the plain-python ``[inner]`` shorthand a hand-written ``Features`` uses.
    """
    if isinstance(feature, (list, tuple)):
        return feature[0] if len(feature) == 1 else None
    return getattr(feature, "feature", None)


def _struct_fields(feature) -> dict | None:
    """The named fields of a struct features node, ``None`` when it is not a struct."""
    return feature if isinstance(feature, dict) else None


def _feature_carries_image(feature) -> bool:
    """Whether a features subtree declares a ``datasets.Image`` anywhere inside it."""
    if isinstance(feature, ImageFeature):
        return True
    fields = _struct_fields(feature)
    if fields is not None:
        return any(_feature_carries_image(field) for field in fields.values())
    element = _list_element_feature(feature)
    return element is not None and _feature_carries_image(element)


def _conversation_schema_declares_images(column_feature) -> bool:
    """Whether a conversation column's Arrow schema can hold image content parts.

    Row-independent: Arrow unions the parts struct over the whole split, so one image-carrying row
    anywhere puts the image field in every row's schema. Read only for the parts form, and only on
    the fields the VLM path treats as image payload
    (:data:`~src.data.pipeline.conversation.IMAGE_PART_PAYLOAD_KEY`, or a ``datasets.Image`` feature
    under any name). A JSON-string conversation column is covered by the first-row probe instead.
    """
    messages = _struct_fields(_list_element_feature(column_feature))
    if messages is None or "content" not in messages:
        return False
    parts = _struct_fields(_list_element_feature(messages["content"]))
    if parts is None:
        return False
    if IMAGE_PART_PAYLOAD_KEY in parts:
        return True
    return any(_feature_carries_image(field) for field in parts.values())


def _declares_images_locally(dataset, conversation_field: str | None) -> bool:
    """:func:`dataset_declares_images` on the local rank's slice, before the cross-rank agreement.

    A column in :data:`VLM_IMAGE_COLUMNS`, or images embedded in the conversation's content parts.
    The embedded form is read two ways so the verdict holds for a mixed dataset: the Arrow schema is
    exact wherever the column is parts-form struct data, and the first-row probe covers the
    JSON-string column, which has no schema.

    ``dataset`` may be a ``DatasetDict`` or a single split; ``None`` reads as no declaration.
    """
    if dataset is None:
        return False
    if carried_image_columns(dataset):
        return True
    return any(
        _column_embeds_images(split, conversation_field)
        for split in (dataset.values() if isinstance(dataset, dict) else [dataset])
        if conversation_field in split.column_names
    )


def _column_embeds_images(split, column: str) -> bool:
    """Whether ``column`` of ``split`` embeds image parts: by Arrow schema, else by its first row."""
    features = getattr(split, "features", None) or {}
    if _conversation_schema_declares_images(features.get(column)):
        return True
    return len(split) > 0 and conversation_carries_images(split[0][column])


def dataset_image_evidence(dataset) -> str | None:
    """Why ``dataset`` counts as image data on the local rank, or ``None`` when nothing declares any.

    The same two declarations :func:`dataset_declares_images` reads, but probing every column for
    embedded parts rather than one named ``conversation_field``, for a consumer that never learns
    the field name (a trainer handed the loaded dataset). ``dataset`` may be a ``DatasetDict`` or a
    single split; ``None`` reads as text. The returned text names the declaration found.
    """
    if dataset is None:
        return None
    columns = carried_image_columns(dataset)
    if columns:
        return f"the image column(s) {sorted(columns)}"
    for split in dataset.values() if isinstance(dataset, dict) else [dataset]:
        for column in split.column_names or ():
            if _column_embeds_images(split, column):
                return f"image parts embedded in the {column!r} column"
    return None


def dataset_declares_images(dataset, conversation_field: str | None = None) -> bool:
    """Whether a loaded dataset carries image data for the VLM path, agreed across ranks.

    Under :class:`~src.data.sources.sharded_dataset.ShardedDatasetLoader` every local term is
    rank-local, while the consumers branch a whole data pipeline on the answer and the two arms run
    different numbers of coordinated operations. One collective per call, on every input (``None``
    included), so the count cannot depend on the verdict itself.
    """
    local = _declares_images_locally(dataset, conversation_field)
    return agree_probe_across_ranks(local, "the loaded dataset", "dataset_declares_images")


def is_vlm_run(args, model_name_or_path: str, dataset=None, *, config=None, revision: str | None = None) -> bool:
    """Whether this run takes the VLM data path: a multimodal checkpoint plus image data to feed it.

    The checkpoint alone cannot decide it: every natively-multimodal family (Gemma 4, Qwen3.5/3.6) is
    a VLM by config while its text-only recipes are ordinary text runs. The run declares image data by
    naming the column (``images_field``) or carrying one; the model loads through its own multimodal
    class either way, so only the data prep, collator and eval split follow this verdict.

    Agreed across ranks once for the whole verdict rather than per term. ``config`` / ``revision`` pin
    the modality probe as in :func:`~src.models.modality.is_vlm_model`; pass the already-loaded
    ``model.config`` where there is one.
    """
    local = is_vlm_model(model_name_or_path, config=config, revision=revision) and bool(
        getattr(args, "images_field", None)
        or _declares_images_locally(dataset, getattr(args, "conversation_field", None))
    )
    return agree_probe_across_ranks(local, model_name_or_path, "is_vlm_run")


def process_vlm_conversation(
    conversation: list[dict[str, Any]],
    system_prompt: str | None = None,
    model_supports_system_role: bool = True,
) -> tuple[list[dict[str, Any]], list[Image.Image]]:
    """Extract images from a conversation, replacing them with bare image-type placeholders.

    Returns (processed_conversation, images), images separate for the processor.
    """
    # Without a system role the prompt is folded into the first turn, not emitted as one the template drops.
    conversation = fold_system_into_conversation(
        conversation, system_prompt, model_supports_system_role, demote_existing_system=False
    )

    processed_conversation = []
    images = []

    for msg in conversation:
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            processed_conversation.append({"role": role, "content": content})
        elif isinstance(content, list):
            new_content = []
            for item in content:
                if item.get("type") == IMAGE_PART_TYPE:
                    image_data = item.get(IMAGE_PART_PAYLOAD_KEY)
                    # Presence, not truthiness: a falsy-but-present payload (empty bytes, an empty
                    # container) has to reach process_image and raise. Skipping it while still
                    # appending the placeholder below builds a placeholder with no image behind it.
                    if image_data is not None:
                        image = process_image(image_data)
                        images.append(image)
                    new_content.append({"type": IMAGE_PART_TYPE})  # processor expects this placeholder
                else:
                    # Arrow unions the parts struct over the split, so a text part of an
                    # image-carrying dataset arrives padded with a null image key, and chat
                    # templates test for the key rather than the type (Qwen-VL: `'image' in
                    # content`), rendering a vision placeholder with no image behind it.
                    new_content.append({key: value for key, value in item.items() if value is not None})

            processed_conversation.append({"role": role, "content": new_content})
        else:
            raise ValueError(f"Unsupported message content type {type(content).__name__} for role '{role}'")

    # Content must be uniformly parts-form: mixed string/list makes Arrow type inference diverge across
    # .map workers (struct vs Json vs null). Processors render parts-form text byte-identically.
    history = [
        msg if isinstance(msg["content"], list) else {**msg, "content": [{"type": "text", "text": msg["content"]}]}
        for msg in processed_conversation
    ]

    return history, images
