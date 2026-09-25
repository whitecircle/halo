"""The Hugging Face Hub tag on every checkpoint Halo writes, and the ``README.md`` card that holds it.

:func:`tag_model_card` tags a directory's card, creating it if absent. The two export finalizers run it
through :func:`tag_exported_model_card`: the config finalizer every full-model writer ends with, and
the non-weight copy every tool that builds an export from a source directory runs. The adapter savers
and the tools that reach neither finalizer call one of the two themselves. A card a library builds
from its own tag list gets the tag where that list comes from: the loaded model's ``model_tags``
(PEFT's adapter card, ``push_to_hub``), the trainer's ``create_model_card`` (:func:`with_halo_tags`),
and the embedding pipeline's card data.
"""

import json
import logging
import os
import shutil
import uuid
from pathlib import Path

import yaml
from huggingface_hub.constants import REPOCARD_NAME
from huggingface_hub.repocard import metadata_load, metadata_save
from huggingface_hub.utils import HFValidationError, validate_repo_id
from peft import PeftType
from peft.utils import CONFIG_NAME as ADAPTER_CONFIG_NAME

from src.log import warn_once

logger = logging.getLogger(__name__)

HUB_TAGS = ("halo",)

# The staged card's name pattern: unique per write, and skipped by the non-weight copy should a
# crash leave one behind.
CARD_STAGING_PREFIX = f".{REPOCARD_NAME}."
CARD_STAGING_SUFFIX = ".tmp"
# The adapter types stock PEFT loads; the toolkit's native expert-LoRA types are outside it.
_STOCK_PEFT_TYPES = frozenset(peft_type.value for peft_type in PeftType)
# Export cards already warned about, so the config finalizer that follows a copy does not repeat it.
_WARNED_EXPORT_CARDS: set[str] = set()


class MalformedModelCardError(ValueError):
    """A ``README.md`` whose tags cannot be rewritten: its metadata block is not a YAML mapping, or
    its ``tags`` entry is neither a list nor a string."""

    def __init__(self, card: Path, reason: Exception):
        self.card = card
        self.reason = reason
        super().__init__(f"{_malformed_card_message(card, reason)} Repair or remove it, then re-run.")


def is_staged_card(name: str) -> bool:
    """Whether ``name`` is a card :func:`tag_model_card` staged and never swapped in."""
    return name.startswith(CARD_STAGING_PREFIX) and name.endswith(CARD_STAGING_SUFFIX)


def with_halo_tags(tags: str | list[str] | None) -> list[str]:
    """``tags`` as a list, a bare string or ``None`` included, with :data:`HUB_TAGS` appended where absent."""
    tags = [tags] if isinstance(tags, str) else list(tags or [])
    return tags + [tag for tag in HUB_TAGS if tag not in tags]


def tag_model_card(output_dir: str) -> None:
    """Add :data:`HUB_TAGS` to the ``README.md`` card in ``output_dir``, creating it if absent.

    Only the ``tags`` entry of an existing card changes value: its other metadata round-trips as the
    raw mapping (``model-index`` included), keys and order kept, and its body, line endings and mode
    are kept. The metadata block is re-dumped, though, so its YAML comments and flow style are lost. A
    card that already carries every tag is not rewritten. A fresh card holds the tags, plus what
    :func:`_fresh_card_metadata` derives for an adapter directory, under the mode the umask gives any
    new file. The write goes to a uniquely named file beside the card and is swapped in, which also
    replaces a symlinked card (a Hub-cache snapshot) instead of writing through it into the blob.

    Raises:
        MalformedModelCardError: the card's metadata block is not a YAML mapping, or its ``tags``
            entry is neither a list nor a string.
    """
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
    staged = _create_staged_card(path.parent)
    try:
        if exists:
            shutil.copyfile(path, staged)
            shutil.copymode(path, staged)
        metadata_save(staged, metadata)
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def tag_exported_model_card(output_dir: str, *, source_dir: str | None = None) -> None:
    """:func:`tag_model_card` for an export, whose card nothing that loads the checkpoint reads.

    A card with malformed metadata must not fail an export whose weights are already on disk: it
    stays verbatim and untagged, and a warning names the file to repair, which is the card in
    ``source_dir`` when the export copied it from there (a re-run copies it again).
    """
    try:
        tag_model_card(output_dir)
    except MalformedModelCardError as error:
        source_card = Path(source_dir) / REPOCARD_NAME if source_dir is not None else None
        card = source_card if source_card is not None and source_card.is_file() else error.card
        warn_once(
            logger,
            _WARNED_EXPORT_CARDS,
            os.path.realpath(error.card),
            f"{_malformed_card_message(card, error.reason)} The export keeps it verbatim, without the Halo "
            f"Hub tag; repair or remove {card}, then re-run to tag it.",
        )


def _malformed_card_message(card: Path, reason: Exception) -> str:
    return f"The model card {card} has malformed metadata ({reason})."


def _create_staged_card(directory: Path) -> Path:
    """An empty file under a fresh staging name in ``directory``.

    Created with the mode ``open(path, "w")`` gives a file (``0o666`` under the umask), so a fresh
    card is as readable as the rest of the checkpoint and no more.
    """
    staged = directory / f"{CARD_STAGING_PREFIX}{uuid.uuid4().hex}{CARD_STAGING_SUFFIX}"
    os.close(os.open(staged, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666))
    return staged


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
