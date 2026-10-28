"""Whether a run takes the VLM data path: the image-column spellings, the image evidence a loaded dataset
carries, and the cross-rank verdicts and refusals read off them."""

from datasets import Image as ImageFeature

from halo.data.conversation import IMAGE_PART_PAYLOAD_KEY, conversation_carries_images
from halo.models.modality import is_vlm_model
from halo.runtime.consensus import agree_probe_across_ranks

# The raw spellings a dataset ships images in, ``images`` first: TRL's vision probe reads the same pair.
VLM_RAW_IMAGE_COLUMNS = ("images", "image")

# Plus ``pixel_values``, what a preprocessed ``--is_vlm`` artifact stores its images as.
VLM_IMAGE_COLUMNS = (*VLM_RAW_IMAGE_COLUMNS, "pixel_values")


def _splits(dataset) -> list:
    """The splits of a ``DatasetDict``, or a single split as the only one."""
    return list(dataset.values()) if isinstance(dataset, dict) else [dataset]


def carried_image_columns(dataset) -> set[str]:
    """The :data:`VLM_IMAGE_COLUMNS` spellings present in any split of ``dataset`` (a ``DatasetDict`` or one
    split), as every image-column guard reads them."""
    return {column for split in _splits(dataset) for column in split.column_names} & set(VLM_IMAGE_COLUMNS)


def reject_image_columns(dataset, method: str) -> None:
    """Fail loud when a method with NO vision path is handed a dataset carrying images.

    Every such method prunes its dataset to the columns it names, so an image column is dropped
    without a word and the run trains on the rows' text alone at full cost. Read off the
    :data:`VLM_IMAGE_COLUMNS` spellings a VLM-capable method routes on.
    """
    carried = sorted(carried_image_columns(dataset))
    if not carried:
        return
    raise ValueError(
        f"{method} is text-only, but the dataset carries the image column(s) {carried}. They would "
        f"be pruned before the first batch and the run would train on the text alone. Drop the "
        f"column(s), or train this data with a method that has a vision path (SFT, distillation, "
        f"DPO, SMPO)."
    )


def _list_element_feature(feature):
    """The element feature of a list-shaped features node (``List``/``LargeList``/``Sequence`` or the
    ``[inner]`` shorthand), ``None`` when it is not list-shaped."""
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
    """Whether a parts-form conversation column's Arrow schema can hold image parts: an
    :data:`~halo.data.conversation.IMAGE_PART_PAYLOAD_KEY` field or a ``datasets.Image`` feature.
    Row-independent, since Arrow unions the parts struct over the whole split."""
    messages = _struct_fields(_list_element_feature(column_feature))
    if messages is None or "content" not in messages:
        return False
    parts = _struct_fields(_list_element_feature(messages["content"]))
    if parts is None:
        return False
    if IMAGE_PART_PAYLOAD_KEY in parts:
        return True
    return any(_feature_carries_image(field) for field in parts.values())


def _column_embeds_images(split, column: str) -> bool:
    """Whether ``column`` of ``split`` embeds image parts: by Arrow schema, else by its first row."""
    features = getattr(split, "features", None) or {}
    if _conversation_schema_declares_images(features.get(column)):
        return True
    return len(split) > 0 and conversation_carries_images(split[0][column])


def _declares_images_locally(dataset, conversation_field: str | None) -> bool:
    """:func:`dataset_declares_images` on the local rank's slice: an image column, or image parts embedded in
    the conversation (by Arrow schema, else by the first row of a JSON-string column); ``None`` declares none."""
    if dataset is None:
        return False
    if carried_image_columns(dataset):
        return True
    return any(
        _column_embeds_images(split, conversation_field)
        for split in _splits(dataset)
        if conversation_field in split.column_names
    )


def dataset_image_evidence(dataset) -> str | None:
    """The declaration that makes ``dataset`` image data on the local rank, or ``None``: the
    :func:`dataset_declares_images` reading, probing every column for a consumer that never learns the
    conversation field."""
    if dataset is None:
        return None
    columns = carried_image_columns(dataset)
    if columns:
        return f"the image column(s) {sorted(columns)}"
    for split in _splits(dataset):
        for column in split.column_names or ():
            if _column_embeds_images(split, column):
                return f"image parts embedded in the {column!r} column"
    return None


def dataset_declares_images(dataset, conversation_field: str | None = None) -> bool:
    """COLLECTIVE: whether a loaded dataset carries image data for the VLM path. A sharded load answers per
    rank while the two arms run different coordinated operations, so the verdict is agreed on every call."""
    local = _declares_images_locally(dataset, conversation_field)
    return agree_probe_across_ranks(local, "the loaded dataset", "dataset_declares_images")


def is_vlm_run(
    args,
    model_name_or_path: str,
    dataset=None,
    *,
    config=None,
    vlm_checkpoint: bool | None = None,
) -> bool:
    """Whether this run takes the VLM data path: a multimodal checkpoint plus image data to feed it,
    agreed across ranks once for the whole verdict.

    A natively multimodal family (Gemma 4, Qwen3.5/3.6) is a VLM by config while its text-only recipes
    are text runs, so the run must also name an ``images_field`` or carry image data. The checkpoint's
    side comes from the loaded ``config`` or from ``vlm_checkpoint``
    (:func:`~halo.model_loading.parallel.model_source.probe_checkpoint`); one of the two is required.
    """
    if vlm_checkpoint is None:
        if config is None:
            raise ValueError(
                f"is_vlm_run({model_name_or_path!r}) needs the checkpoint's config or its probe_checkpoint "
                f"verdict (vlm_checkpoint)."
            )
        vlm_checkpoint = is_vlm_model(model_name_or_path, config)
    local = vlm_checkpoint and bool(
        getattr(args, "images_field", None)
        or _declares_images_locally(dataset, getattr(args, "conversation_field", None))
    )
    return agree_probe_across_ranks(local, model_name_or_path, "is_vlm_run")
