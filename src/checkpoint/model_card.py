"""The Hugging Face Hub tag every model Halo writes carries, and the ``README.md`` card that holds it.

A loaded model carries the tag in ``model_tags``, which the card writers of ``push_to_hub`` and of a
PEFT ``save_pretrained`` union in. An exported directory has its card tagged directly by
:func:`tag_model_card`, which both export finalizers run: the config finalizer every full-model
writer ends with, and the non-weight copy every tool that builds an export from a source runs.
"""

from pathlib import Path

from huggingface_hub import ModelCard, ModelCardData
from huggingface_hub.constants import REPOCARD_NAME

HALO_HUB_TAGS = ("halo",)


def tag_model_card(output_dir: str) -> None:
    """Add :data:`HALO_HUB_TAGS` to the ``README.md`` card in ``output_dir``, creating it if absent.

    An existing card (PEFT's, TRL's, sentence-transformers', or one carried over from the source
    checkpoint) keeps its body and metadata, and a card already carrying every tag is not rewritten.
    A fresh card holds the tags alone: ``library_name`` is left to the library that owns the
    directory's format, since a guess would be wrong for a sentence-transformers or adapter export.
    """
    path = Path(output_dir) / REPOCARD_NAME
    if not path.is_file():
        ModelCard(f"---\n{ModelCardData(tags=list(HALO_HUB_TAGS)).to_yaml()}\n---\n").save(path)
        return
    card = ModelCard.load(path)
    tags = list(card.data.tags or [])
    missing = [tag for tag in HALO_HUB_TAGS if tag not in tags]
    if missing:
        card.data.tags = tags + missing
        card.save(path)
