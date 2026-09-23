"""The Hugging Face Hub tag every model Halo writes carries, and the ``README.md`` card that holds it.

Exported directories have their card tagged by :func:`tag_model_card`, which both export finalizers
run: the config finalizer every full-model writer ends with, and the non-weight copy every tool that
builds an export from a source directory runs. A writer that reaches neither calls it directly.
"""

import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import PeftType
from peft.utils import CONFIG_NAME as ADAPTER_CONFIG_NAME

HALO_HUB_TAGS = ("halo",)

# The staged card's name pattern: unique per write, and skipped by the non-weight copy should a
# crash leave one behind.
CARD_STAGING_PREFIX = f".{REPOCARD_NAME}."
CARD_STAGING_SUFFIX = ".tmp"
# A fresh card's mode: the staging file is created owner-only, which would hide the card from the
# other readers of a shared output filesystem.
_FRESH_CARD_MODE = 0o644
# The adapter types stock PEFT loads; the toolkit's native expert-LoRA types are outside it.
_STOCK_PEFT_TYPES = frozenset(peft_type.value for peft_type in PeftType)


def is_staged_card(name: str) -> bool:
    """Whether ``name`` is a card :func:`tag_model_card` staged and never swapped in."""
    return name.startswith(CARD_STAGING_PREFIX) and name.endswith(CARD_STAGING_SUFFIX)


def with_halo_tags(tags: str | list[str] | None) -> list[str]:
    """``tags`` as a list, a bare string or ``None`` included, with :data:`HALO_HUB_TAGS` appended where absent."""
    tags = [tags] if isinstance(tags, str) else list(tags or [])
    return tags + [tag for tag in HALO_HUB_TAGS if tag not in tags]


def tag_model_card(output_dir: str) -> None:
    """Add :data:`HALO_HUB_TAGS` to the ``README.md`` card in ``output_dir``, creating it if absent.

    Only the ``tags`` entry of an existing card changes: its other metadata round-trips as the raw
    mapping (``model-index`` included), and its body, line endings and mode are kept. A card that
    already carries every tag is not rewritten. A fresh card holds the tags, plus what
    :func:`_fresh_card_metadata` derives for an adapter directory. The write goes to a uniquely named
    file beside the card and is swapped in, which also replaces a symlinked card (a Hub-cache
    snapshot) instead of writing through it into the blob.

    Raises:
        ValueError: the card's metadata block is not a YAML mapping.
    """
    path = Path(output_dir) / REPOCARD_NAME
    exists = path.is_file()
    if exists:
        try:
            metadata = metadata_load(path) or {}
        except (yaml.YAMLError, ValueError) as error:
            raise ValueError(
                f"The model card {path} has a metadata block that is not a YAML mapping ({error}). Repair "
                f"or remove it, then re-run."
            ) from error
    else:
        metadata = _fresh_card_metadata(path.parent)
    tags = with_halo_tags(metadata.get("tags"))
    if exists and tags == metadata.get("tags"):
        return
    metadata["tags"] = tags
    with tempfile.NamedTemporaryFile(
        dir=output_dir, prefix=CARD_STAGING_PREFIX, suffix=CARD_STAGING_SUFFIX, delete=False
    ) as handle:
        staged = Path(handle.name)
    try:
        if exists:
            shutil.copyfile(path, staged)
            shutil.copymode(path, staged)
        else:
            staged.chmod(_FRESH_CARD_MODE)
        metadata_save(staged, metadata)
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _fresh_card_metadata(directory: Path) -> dict:
    """What a new card declares besides the tags: the stock-PEFT identity of an adapter directory.

    An adapter stock PEFT loads gets the ``library_name: peft`` and ``base_model`` PEFT's own card
    carries. A ``peft_type`` outside PEFT's registry (the native EP expert adapters) and a full-model
    directory declare no library, so the card claims no loader that would refuse the files.
    """
    adapter_config = directory / ADAPTER_CONFIG_NAME
    if not adapter_config.is_file():
        return {}
    config = json.loads(adapter_config.read_text())
    if config.get("peft_type") not in _STOCK_PEFT_TYPES:
        return {}
    metadata = {"library_name": "peft"}
    base_model = config.get("base_model_name_or_path")
    if base_model and _is_hub_repo_id(base_model):
        metadata["base_model"] = base_model
    return metadata


def _is_hub_repo_id(name: str) -> bool:
    """Whether ``name`` names a Hub repo: the Hub rejects a card whose ``base_model`` is a local path."""
    if os.path.isdir(name):
        return False
    try:
        validate_repo_id(name)
    except HFValidationError:
        return False
    return True
