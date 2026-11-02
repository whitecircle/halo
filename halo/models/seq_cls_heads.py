"""Sequence-classification heads transformers 5.20 does not ship (Gemma 4, MoE Qwen3.5/3.6), registered
at import, and the live roster of multimodal families carrying a head. Each is the upstream Generic head
over the family's ``PreTrainedModel``, registered for both spellings a checkpoint can carry (the
composite wrapper and its text tower); a release that lands a native head takes precedence.
"""

from transformers import AutoModelForSequenceClassification
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
)
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4PreTrainedModel
from transformers.models.qwen3_5_moe import Qwen3_5MoePreTrainedModel, Qwen3_5MoeTextConfig

from halo.models.modality import config_declares_multimodality


def _register_sequence_classification_head(model_cls) -> None:
    """Register a seq-cls head transformers does not ship, under the config it declares.

    ``exist_ok=True`` is load-bearing: without it ``register()`` silently skips a config class living
    under ``transformers.*``, and resolution then fails far away as "Unrecognized configuration class".
    It would also let the head shadow a native one, so a family whose release ships its own head is
    skipped here.
    """
    config_cls = model_cls.config_class
    if config_cls.model_type not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
        AutoModelForSequenceClassification.register(config_cls, model_cls, exist_ok=True)


class Gemma4ForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """Pooled-score head over the composite Gemma 4 backbone."""


class Gemma4TextForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """Pooled-score head over the text-only backbone (``model_type: gemma4_text``).

    Its own registration: 5.20 gives the text tower its own config class, and a CausalLM SFT on a
    text-only artifact writes ``model_type: gemma4_text``, which the composite head cannot load.
    """

    config_class = Gemma4TextConfig


class Qwen3_5MoeForSequenceClassification(GenericForSequenceClassification, Qwen3_5MoePreTrainedModel):
    """Pooled-score head over the composite Qwen3.5/3.6-MoE backbone."""


class Qwen3_5MoeTextForSequenceClassification(GenericForSequenceClassification, Qwen3_5MoePreTrainedModel):
    """Pooled-score head over the text-only tower (``model_type: qwen3_5_moe_text``)."""

    config_class = Qwen3_5MoeTextConfig


# Driven off the classes above rather than a hand-kept pair list, so a head added to this module
# registers under the config class it already declares.
for _head_cls in GenericForSequenceClassification.__subclasses__():
    if _head_cls.__module__ == __name__:
        _register_sequence_classification_head(_head_cls)


def multimodal_sequence_classification_model_types() -> list[str]:
    """Model types with both a sequence-classification head (transformers' mapping, read live, which
    holds the toolkit's heads above) and a vision tower, by the
    :func:`~halo.models.modality.config_declares_multimodality` verdict the run-time probe reads, taken
    off each config class so the roster cannot classify a family differently from that probe."""
    return sorted(
        {cls.model_type for cls in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING if config_declares_multimodality(cls)}
    )
