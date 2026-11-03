"""Sequence-classification heads transformers 5.20 does not ship (Gemma 4, MoE Qwen3.5/3.6), registered at import; a
release that lands a native head takes precedence."""

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
    """Without ``exist_ok=True``, ``register()`` silently skips a config class under ``transformers.*``. It would also
    shadow a native head, so a family whose release ships its own is skipped."""
    config_cls = model_cls.config_class
    if config_cls.model_type not in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES:
        AutoModelForSequenceClassification.register(config_cls, model_cls, exist_ok=True)


class Gemma4ForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """Pooled-score head over the composite Gemma 4 backbone."""


class Gemma4TextForSequenceClassification(GenericForSequenceClassification, Gemma4PreTrainedModel):
    """A CausalLM SFT on a text-only artifact writes ``model_type: gemma4_text``, which the composite head refuses."""

    config_class = Gemma4TextConfig


class Qwen3_5MoeForSequenceClassification(GenericForSequenceClassification, Qwen3_5MoePreTrainedModel):
    """Pooled-score head over the composite Qwen3.5/3.6-MoE backbone."""


class Qwen3_5MoeTextForSequenceClassification(GenericForSequenceClassification, Qwen3_5MoePreTrainedModel):
    config_class = Qwen3_5MoeTextConfig


for _head_cls in GenericForSequenceClassification.__subclasses__():
    if _head_cls.__module__ == __name__:
        _register_sequence_classification_head(_head_cls)


def multimodal_sequence_classification_model_types() -> list[str]:
    """Off each config class by the run-time probe's own verdict, so the roster cannot classify a family otherwise."""
    return sorted(
        {cls.model_type for cls in MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING if config_declares_multimodality(cls)}
    )
