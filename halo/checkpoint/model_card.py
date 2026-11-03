"""The Hugging Face Hub tag on every checkpoint Halo writes, and the ``README.md`` card that holds it."""

import logging
import os
import shutil
from pathlib import Path

import yaml
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import PeftType

from halo.checkpoint.atomic import staged_file
from halo.checkpoint.format import ADAPTER_CONFIG_FILE
from halo.checkpoint.reading import read_checkpoint_config
from halo.log import warn_once

logger = logging.getLogger(__name__)

HUB_TAGS = ("halo",)

# The adapter types stock PEFT loads; the toolkit's native expert-LoRA types are outside it.
_STOCK_PEFT_TYPES = frozenset(peft_type.value for peft_type in PeftType)
# Export cards already warned about, so the config finalizer that follows a copy does not repeat it.
_warned_export_cards: set[str] = set()


def _malformed_card_message(card: Path, reason: Exception) -> str:
    return f"The model card {card} has malformed metadata ({reason})."


class MalformedModelCardError(ValueError):
    """A ``README.md`` whose metadata is not a YAML mapping, or whose ``tags`` is neither a list nor a string."""

    def __init__(self, card: Path, reason: Exception):
        self.card = card
        self.reason = reason
        super().__init__(f"{_malformed_card_message(card, reason)} Repair or remove it, then re-run.")


def with_halo_tags(tags: str | list[str] | None) -> list[str]:
    """``tags`` as a list, with :data:`HUB_TAGS` appended where absent."""
    tags = [tags] if isinstance(tags, str) else list(tags or [])
    return tags + [tag for tag in HUB_TAGS if tag not in tags]


def _is_hub_repo_id(name: str) -> bool:
    """Whether ``name`` names a Hub repo: the Hub rejects a card whose ``base_model`` is a local path."""
    if os.path.isdir(name):
        return False
    try:
        validate_repo_id(name)
    except HFValidationError:
        return False
    return True


def _fresh_card_metadata(directory: Path) -> dict:
    """``library_name: peft`` and ``base_model`` for an adapter stock PEFT loads, else nothing, so the card claims no
    loader that would refuse the files."""
    config = read_checkpoint_config(directory, missing_ok=True, filename=ADAPTER_CONFIG_FILE)
    if config is None or config.get("peft_type") not in _STOCK_PEFT_TYPES:
        return {}
    metadata = {"library_name": "peft"}
    base_model = config.get("base_model_name_or_path")
    if base_model and _is_hub_repo_id(base_model):
        metadata["base_model"] = base_model
    return metadata


def tag_model_card(output_dir: str) -> None:
    """Add :data:`HUB_TAGS` to ``output_dir``'s ``README.md``, creating it if absent; a malformed card raises. Only
    ``tags`` changes (YAML comments are lost to the re-dump), and the staged swap replaces a symlinked Hub-cache card
    instead of writing through it into the blob."""
    path = Path(output_dir) / REPOCARD_NAME
    exists = path.is_file()
    if exists:
        try:
            metadata = metadata_load(path) or {}
        except (yaml.YAMLError, ValueError) as error:
            raise MalformedModelCardError(path, error) from error
        if not isinstance(metadata.get("tags"), str | list | None):
            reason = TypeError(f"tags is a {type(metadata['tags']).__name__}, not a list or a string")
            raise MalformedModelCardError(path, reason)
    else:
        metadata = _fresh_card_metadata(path.parent)
    tags = with_halo_tags(metadata.get("tags"))
    if exists and tags == metadata.get("tags"):
        return
    metadata["tags"] = tags
    with staged_file(path.parent, REPOCARD_NAME, durable=False) as staged:
        if exists:
            shutil.copyfile(path, staged)
        metadata_save(staged, metadata)
        if exists:
            # After the write, since a read-only source mode on the staged file would refuse metadata_save.
            shutil.copymode(path, staged)


def tag_exported_model_card(output_dir: str, *, source_dir: str | None = None) -> None:
    """:func:`tag_model_card` for an export, where a malformed card stays untagged with a warning, since the weights
    are already on disk; the warning names ``source_dir``'s card when it was copied from there."""
    try:
        tag_model_card(output_dir)
    except MalformedModelCardError as error:
        source_card = Path(source_dir) / REPOCARD_NAME if source_dir is not None else None
        card = source_card if source_card is not None and source_card.is_file() else error.card
        warn_once(
            logger,
            _warned_export_cards,
            os.path.realpath(error.card),
            f"{_malformed_card_message(card, error.reason)} The export keeps it verbatim, without the Halo "
            f"Hub tag; repair or remove {card}, then re-run to tag it.",
        )
