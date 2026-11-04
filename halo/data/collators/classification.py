from typing import Any

from transformers import DataCollatorWithPadding


class ClassificationDataCollatorWithPadding(DataCollatorWithPadding):
    """Pad only the model inputs plus the label: source columns reach the collator (``remove_unused_columns`` defaults
    to False) and ``tokenizer.pad`` fails tensorizing a string."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        keep = set(self.tokenizer.model_input_names).union(("label", "labels"))
        return super().__call__([{k: v for k, v in feature.items() if k in keep} for feature in features])
