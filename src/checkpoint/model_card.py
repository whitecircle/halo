"""The Hugging Face Hub tag every model Halo writes carries, and the ``README.md`` card that holds it.

Exported directories have their card tagged by :func:`tag_model_card`, which both export finalizers
run: the config finalizer every full-model writer ends with, and the non-weight copy every tool that
builds an export from a source directory runs. A writer that reaches neither calls it directly.
"""

import os
import shutil
from pathlib import Path

import yaml
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save

HALO_HUB_TAGS = ("halo",)


def with_halo_tags(tags: str | list[str] | None) -> list[str]:
    """``tags`` as a list, a bare string or ``None`` included, with :data:`HALO_HUB_TAGS` appended where absent."""
    tags = [tags] if isinstance(tags, str) else list(tags or [])
    return tags + [tag for tag in HALO_HUB_TAGS if tag not in tags]


def tag_model_card(output_dir: str) -> None:
    """Add :data:`HALO_HUB_TAGS` to the ``README.md`` card in ``output_dir``, creating it if absent.

    Only the ``tags`` entry of an existing card changes: its other metadata round-trips as the raw
    mapping (``model-index`` included), and its body and line endings are kept. A card that already
    carries every tag is not rewritten, and a fresh card holds the tags alone, so ``library_name``
    stays the owning library's. The write is staged beside the card and swapped in, which also
    replaces a symlinked card (a Hub-cache snapshot) instead of writing through it into the blob.

    Raises:
        ValueError: the card's metadata block is not a YAML mapping.
    """
    path = Path(output_dir) / REPOCARD_NAME
    exists = path.is_file()
    try:
        metadata = (metadata_load(path) if exists else None) or {}
    except (yaml.YAMLError, ValueError) as error:
        raise ValueError(
            f"The model card {path} has a metadata block that is not a YAML mapping ({error}). Repair or "
            f"remove it, then re-run."
        ) from error
    tags = with_halo_tags(metadata.get("tags"))
    if exists and tags == metadata.get("tags"):
        return
    metadata["tags"] = tags
    staged = path.with_name(f".{REPOCARD_NAME}.halo-staging")
    try:
        if exists:
            shutil.copyfile(path, staged)
        metadata_save(staged, metadata)
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
