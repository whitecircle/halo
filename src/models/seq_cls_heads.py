"""Sequence-classification heads transformers does not ship, registered at import.

transformers 5.16 maps no seq-cls head for Gemma 4 or for MoE Qwen3.5/3.6, so reward modeling and
classification fail at load on both families. Each head is the upstream Generic head over the
family's ``PreTrainedModel``, registered in both spellings a checkpoint can carry (the composite
wrapper and its text tower); a release that lands a native head takes precedence.
"""

from transformers import AutoModelForSequenceClassification
from transformers.modeling_layers import GenericForSequenceClassification
from transformers.models.auto.modeling_auto import MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4PreTrainedModel
from transformers.models.qwen3_5_moe import Qwen3_5MoePreTrainedModel, Qwen3_5MoeTextConfig


def register_sequence_classification_head(model_cls) -> None:
    """Register a seq-cls head transformers does not ship, under the config it declares.

    Not ``register()``: transformers >= 5.12 no-ops it for a config class living under
    ``transformers.*`` (``_LazyAutoMapping.register``'s anti-shadowing guard), and resolution then
    fails far away as "Unrecognized configuration class". ``_extra_content`` is the same storage,
    read identically by ``from_config`` / ``from_pretrained``. A family that gains a native head is
    already in the mapping, so this becomes a no-op.
    """
    config_cls = model_cls.config_class
    if config_cls.model_type not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
        AutoModelForSequenceClassification._model_mapping._extra_content[config_cls] = model_cls


class Gemma4ForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """Pooled-score head over the composite Gemma 4 backbone."""


class Gemma4TextForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """Pooled-score head over the text-only backbone (``model_type: gemma4_text``).

    Its own registration: 5.16 gives the text tower its own config class, and a CausalLM SFT on a
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
        register_sequence_classification_head(_head_cls)
