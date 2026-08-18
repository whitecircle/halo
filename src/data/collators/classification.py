"""Classification batching: pad the tokenizer's model inputs plus the label, drop the rest."""

from typing import Any

from transformers import DataCollatorWithPadding


class ClassificationDataCollatorWithPadding(DataCollatorWithPadding):
    """Pad only the tokenizer's model inputs plus the label, dropping every other column.

    Classification rows keep their source columns after tokenization (raw ``text``, string labels,
    tool JSON) and :class:`~src.configs.classification_config.ClassificationConfig` defaults
    ``remove_unused_columns`` to False, so those columns reach the collator and ``tokenizer.pad``
    fails trying to tensorize a string. The model consumes only ``model_input_names`` and the label;
    the keep-set is derived from the tokenizer rather than hardcoded.
    """

    _LABEL_KEYS = ("label", "labels")

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        keep = set(self.tokenizer.model_input_names).union(self._LABEL_KEYS)
        return super().__call__([{k: v for k, v in feature.items() if k in keep} for feature in features])
