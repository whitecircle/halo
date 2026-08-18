"""``DataCollatorForCompletionOnlyLM`` — masks prompt tokens so loss covers completions only."""

from typing import Any

from transformers import DataCollatorForLanguageModeling

from src.data.spans import (
    LABEL_IGNORE_INDEX,
    mask_batch_to_completion_spans,
    resolve_eos_token_ids,
    tokenize_response_template,
    warn_if_pad_equals_eos,
)


class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    """Completion-only collator: masks all non-assistant label tokens to ignore_index
    so loss is computed only on assistant completions.

    response_prompt_template (str or token IDs) marks the response start (e.g.
    '### Response:\n'); pass token IDs when the tokenizer encodes it context-
    dependently. train_on_last_assistant_only masks all but the LAST assistant
    message.
    """

    def __init__(
        self,
        response_prompt_template: str | list[int],
        *args,
        ignore_index: int = LABEL_IGNORE_INDEX,
        train_on_last_assistant_only: bool = False,
        eos_token_ids: frozenset[int] | None = None,
        **kwargs,
    ):
        super().__init__(*args, mlm=False, **kwargs)

        self.response_prompt_template = response_prompt_template
        self.train_on_last_assistant_only = train_on_last_assistant_only
        self.response_token_ids = tokenize_response_template(response_prompt_template, self.tokenizer)
        self.eos_token_ids = eos_token_ids if eos_token_ids is not None else resolve_eos_token_ids(self.tokenizer)

        warn_if_pad_equals_eos(self.tokenizer)

        self.ignore_index = ignore_index

    def torch_call(self, examples: list[list[int] | Any | dict[str, Any]]) -> dict[str, Any]:
        return mask_batch_to_completion_spans(
            super().torch_call(examples),
            self.response_token_ids,
            self.eos_token_ids,
            self.ignore_index,
            self.train_on_last_assistant_only,
            self.response_prompt_template,
            tokenizer=self.tokenizer,
        )
