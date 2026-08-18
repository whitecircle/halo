"""Text collator for SDPG self-distillation: the student batch plus a ``teacher_*`` branch whose last
user turn carries the privileged hint. The response tokens are byte-identical across the two.

The hint injection and confidence weighting here are also used by the VLM self-distillation collator.
"""

from typing import Any

import torch

from src.data.pipeline.rendered import render_conversation, tokenize_rendered
from src.data.spans import build_completion_only_labels, require_response_marker, resolve_eos_token_ids


def inject_privileged_hint(
    history: list[dict[str, Any]],
    hint_template: str,
    answer: Any,
    solution: Any,
) -> list[dict[str, Any]]:
    """Return a copy of ``history`` with the privileged hint appended to the last user turn."""
    hint = hint_template.format(
        answer="" if answer is None else answer,
        solution="" if solution is None else solution,
    )
    new_history = [dict(msg) for msg in history]
    for i in range(len(new_history) - 1, -1, -1):
        if new_history[i].get("role") != "user":
            continue
        content = new_history[i].get("content")
        if isinstance(content, str):
            new_history[i]["content"] = content + hint
        elif isinstance(content, list):
            new_history[i]["content"] = list(content) + [{"type": "text", "text": hint}]
        else:
            new_history[i]["content"] = hint
        break
    return new_history


def mean_normalized_confidence_weights(
    examples: list[dict[str, Any]],
    field: str,
    power: float,
) -> torch.Tensor:
    """Per-sample ``conf ** power`` weights, mean-normalized to preserve the effective LR."""
    confs = torch.tensor([float(ex.get(field) or 0.0) for ex in examples], dtype=torch.float32)
    weights = confs.clamp(min=0.0) ** power
    total = weights.sum()
    if total <= 1e-9:
        # All confidences zero or absent: uniform weights, else the batch scales to a zero gradient.
        return torch.ones_like(weights)
    return weights * (weights.numel() / total)


class SelfDistillTextCollator:
    """Text collator for SDPG-style self-distillation (arXiv:2606.04036).

    Reads raw conversation rows + privileged answer/solution fields, emitting the student batch and a
    teacher branch (``teacher_*``) whose last user turn carries the gold-answer hint. The assistant
    response is byte-identical across both sequences, so the teacher can supervise the student on the
    shared tokens. Optionally emits ``confidence_weights``.
    """

    def __init__(
        self,
        tokenizer,
        max_length: int = 2048,
        conversation_field: str = "messages",
        *,
        hint_template: str,
        answer_field: str = "answer",
        solution_field: str | None = "solution",
        confidence_field: str | None = None,
        confidence_power: float = 4.0,
        response_prompt_template: str | None = None,
        train_on_completions_only: bool = False,
        system_prompt: str | None = None,
        model_supports_system_role: bool = True,
        tools_field: str | None = None,
        interleaved_thinking: bool = False,
        model_config=None,
    ):
        require_response_marker(response_prompt_template, train_on_completions_only, "Self-distillation")
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversation_field = conversation_field
        self.hint_template = hint_template
        self.system_prompt = system_prompt
        self.model_supports_system_role = model_supports_system_role
        self.tools_field = tools_field
        self.interleaved_thinking = interleaved_thinking
        self.answer_field = answer_field
        self.solution_field = solution_field
        self.confidence_field = confidence_field
        self.confidence_power = confidence_power
        self.response_prompt_template = response_prompt_template
        self.train_on_completions_only = train_on_completions_only
        self.eos_token_ids = resolve_eos_token_ids(tokenizer, model_config)

    def cache_signature(self) -> dict[str, Any]:
        """Every knob this collator renders with, to thread through the audit map's cache key.

        The collator rides ``fn_kwargs`` (see :func:`audit_self_distill_row`), and a dataset-map
        fingerprint reads an object that is neither tokenizer nor processor as its class name alone,
        so without this the audit's verdict would survive a change to any knob below. Read off
        ``__dict__`` rather than a hand-listed subset, so a field added to ``__init__`` enters the
        key with it.
        """
        return dict(vars(self))

    def _render(self, history: list[dict[str, Any]], row: dict[str, Any]) -> str:
        """Chat-template one conversation through the shared text renderer, so the student and
        teacher branches honor the same knobs as the SFT text pipeline."""
        return render_conversation(
            self.tokenizer,
            history,
            row,
            conversation_field=self.conversation_field,
            system_prompt=self.system_prompt,
            model_supports_system_role=self.model_supports_system_role,
            interleaved_thinking=self.interleaved_thinking,
            tools_field=self.tools_field,
        )

    def _tokenize(
        self, histories: list[list[dict[str, Any]]], rows: list[dict[str, Any]], branch: str
    ) -> dict[str, torch.Tensor]:
        texts = [self._render(history, row) for history, row in zip(histories, rows, strict=True)]
        # No truncation, see the raise below. tokenize_rendered handles the rendered specials: a bare
        # add_special_tokens=True would emit a second BOS for templates that render one themselves.
        rows = [tokenize_rendered(self.tokenizer, text) for text in texts]
        encoded = self.tokenizer.pad(rows, padding=True, return_tensors="pt")
        longest = int(encoded["attention_mask"].sum(dim=-1).max())
        if longest > self.max_length:
            raise ValueError(
                f"Self-distillation {branch} branch contains a {longest}-token sequence, "
                f"{longest - self.max_length} tokens over max_length={self.max_length}. Truncating would "
                f"cut response tokens that must stay byte-identical between student and teacher (OPD row "
                f"alignment). Raise max_length (the teacher needs headroom for the privileged hint) or "
                f"drop over-length rows before training."
            )
        return encoded

    def _build_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mask pad tokens, and (optionally) everything outside assistant completions.

        Shared by the student batch and the teacher branch so both select the same response tokens.
        """
        return build_completion_only_labels(
            input_ids,
            self.tokenizer,
            self.response_prompt_template,
            self.train_on_completions_only,
            attention_mask=attention_mask,
            eos_token_ids=self.eos_token_ids,
        )

    def audit_row(self, example: dict[str, Any]) -> dict[str, Any]:
        """Run one raw row through the collate-time length contract; returns no columns.

        Mapped over the dataset at prep via :func:`audit_self_distill_row` (``num_proc`` maps reject
        bound methods), where the coordinated map machinery turns the raise into a world-uniform
        failure. The same raise at collate time is rank-local: only the rank whose batch drew the
        over-length row sees the error, while its peers block in the step's collectives until the
        NCCL watchdog fires.
        """
        self._tokenize([example[self.conversation_field]], [example], branch="student")
        self._tokenize(
            [
                inject_privileged_hint(
                    example[self.conversation_field],
                    self.hint_template,
                    example.get(self.answer_field),
                    example.get(self.solution_field) if self.solution_field else None,
                )
            ],
            [example],
            branch="teacher",
        )
        return {}

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        student_histories = [ex[self.conversation_field] for ex in examples]
        teacher_histories = [
            inject_privileged_hint(
                ex[self.conversation_field],
                self.hint_template,
                ex.get(self.answer_field),
                ex.get(self.solution_field) if self.solution_field else None,
            )
            for ex in examples
        ]

        student = self._tokenize(student_histories, examples, branch="student")
        teacher = self._tokenize(teacher_histories, examples, branch="teacher")

        batch = {
            "input_ids": student["input_ids"],
            "attention_mask": student["attention_mask"],
            "labels": self._build_labels(student["input_ids"], student["attention_mask"]),
            "teacher_input_ids": teacher["input_ids"],
            "teacher_attention_mask": teacher["attention_mask"],
            "teacher_labels": self._build_labels(teacher["input_ids"], teacher["attention_mask"]),
        }
        if self.confidence_field is not None:
            batch["confidence_weights"] = mean_normalized_confidence_weights(
                examples, self.confidence_field, self.confidence_power
            )
        return batch


def audit_self_distill_row(example: dict[str, Any], collator: SelfDistillTextCollator) -> dict[str, Any]:
    """Module-level entry point for mapping :meth:`SelfDistillTextCollator.audit_row` with
    ``num_proc`` workers, which reject bound methods. The collator rides ``fn_kwargs`` instead; its
    tokenizer and config pickle cleanly."""
    return collator.audit_row(example)
