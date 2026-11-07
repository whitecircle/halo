"""The SDPG self-distillation collators, text and VLM: the student batch plus a ``teacher_*`` branch whose last user
turn carries the privileged hint."""

import math
import string
from collections.abc import Mapping, Sequence
from typing import Any

import torch
from trl.trainer.utils import pad

from halo.data.chat_render import render_conversation, tokenize_rendered
from halo.data.collators.vlm import VLMDataCollator
from halo.data.completion_spans import SELF_DISTILL_SPAN_POLICY, build_training_labels
from halo.data.turn_markers import reject_unusable_response_marker, resolve_eos_token_ids
from halo.data.vlm_render import SEQUENCE_ALIGNED_VISION_KEYS
from halo.kernels.logprobs import LABEL_IGNORE_INDEX


def format_field_names(template: str) -> set[str]:
    """Names ``str.format`` looks up, nested specs included; a lone brace raises ``ValueError``, as it would there."""
    names: set[str] = set()
    for _, name, spec, _ in string.Formatter().parse(template):
        if name is not None:
            names |= {name, *format_field_names(spec or "")}
    return names


def hint_literal_bytes(template: str) -> int:
    """UTF-8 bytes of ``template`` outside its fields. A token spans at least one byte, so this bounds the tokens the
    literal text encodes to; a field's value is the dataset's and has no bound here."""
    return sum(len(literal.encode()) for literal, _, _, _ in string.Formatter().parse(template))


def privileged_hint(template: str, **slots: Any) -> str | None:
    """``None`` when a slot the template names is blank (it states its slots as fact); a slot the caller does not
    supply raises ``KeyError``."""
    named = {name: slots[name] for name in format_field_names(template)}
    if any(value is None or not str(value).strip() for value in named.values()):
        return None
    return template.format(**named)


def inject_privileged_hint(history: list[dict[str, Any]], hint: str | None) -> list[dict[str, Any]]:
    new_history = [dict(msg) for msg in history]
    if hint is None:
        return new_history
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


def teacher_history(
    history: list[dict[str, Any]],
    row: Mapping[str, Any],
    *,
    hint_template: str,
    answer_field: str | None,
    solution_field: str | None,
) -> list[dict[str, Any]]:
    """The teacher branch's conversation, for both collators and the VLM over-length filter."""
    hint = privileged_hint(
        hint_template,
        answer=row.get(answer_field) if answer_field else None,
        solution=row.get(solution_field) if solution_field else None,
    )
    return inject_privileged_hint(history, hint)


def require_aligned_responses(batch: Mapping[str, torch.Tensor]) -> None:
    """Raise unless each row's branches supervise the same tokens in order: OPD pairs the k-th supervised position."""
    student, teacher = batch["labels"][:, 1:], batch["teacher_labels"][:, 1:]
    for row in range(student.size(0)):
        student_tokens = student[row][student[row] != LABEL_IGNORE_INDEX]
        teacher_tokens = teacher[row][teacher[row] != LABEL_IGNORE_INDEX]
        if not torch.equal(student_tokens, teacher_tokens):
            raise ValueError(
                f"Self-distillation row {row}: the student branch supervises {student_tokens.numel()} tokens "
                f"and the hinted teacher branch {teacher_tokens.numel()}, not the same sequence. OPD pairs "
                f"them position by position, so every pair would compare different tokens. Supervise the "
                f"completions only (train_on_completions_only with assistant_message_template), and use a "
                f"chat template whose assistant turns render the same with the hint in the user turn."
            )


def _checked_confidences(split, field: str) -> torch.Tensor:
    """A split's confidences as float64, refused unless every value lies in ``[0, 1]``."""
    values = list(split[field])
    confidences = torch.tensor([math.nan if value is None else float(value) for value in values], dtype=torch.float64)
    outside = ~((confidences >= 0) & (confidences <= 1))
    if outside.any():
        first = int(outside.nonzero()[0])
        raise ValueError(
            f"confidence_field={field!r} holds {int(outside.sum())} value(s) outside [0, 1] (row {first}: "
            f"{values[first]!r}); each one weights its row's loss by conf ** confidence_power."
        )
    return confidences


def confidence_normalizer(train, test, field: str, power: float) -> float:
    """The train split's mean of ``conf ** power``, the one divisor every batch shares (a per-batch mean is 1.0 at
    batch size 1); ``test`` is range-checked too."""
    if test is not None:
        _checked_confidences(test, field)
    normalizer = float(_checked_confidences(train, field).pow(power).mean())
    if normalizer <= 0:
        raise ValueError(f"Every train confidence in {field!r} is 0, so every row would carry zero weight.")
    return normalizer


def confidence_weights(
    examples: Sequence[Mapping[str, Any]], field: str, power: float, normalizer: float
) -> torch.Tensor:
    confidences = torch.tensor([float(example[field]) for example in examples], dtype=torch.float32)
    return confidences**power / normalizer


class SelfDistillBranchMixin:
    """What both collators add to a student batch: privileged fields, hinted teacher history, confidence weights."""

    def _init_self_distill_fields(
        self,
        hint_template: str | None,
        answer_field: str | None,
        solution_field: str | None,
        confidence_field: str | None,
        confidence_power: float,
        confidence_normalizer: float | None,
    ) -> None:
        if confidence_field is not None and confidence_normalizer is None:
            raise ValueError(
                f"confidence_field={confidence_field!r} needs confidence_normalizer, the train split's mean of "
                f"conf ** confidence_power (confidence_normalizer()); a per-batch mean is 1.0 at batch size 1."
            )
        self.hint_template = hint_template
        self.answer_field = answer_field
        self.solution_field = solution_field
        self.confidence_field = confidence_field
        self.confidence_power = confidence_power
        self.confidence_normalizer = confidence_normalizer

    @property
    def builds_teacher_branch(self) -> bool:
        return self.hint_template is not None

    def cache_signature(self) -> dict[str, Any]:
        """Every knob this collator renders with, for the prep map's cache key: a map fingerprint reads a collator in
        ``fn_kwargs`` by class name alone, so a changed knob would serve stale rows."""
        return dict(vars(self))

    def _teacher_history(self, history: list[dict[str, Any]], example: dict[str, Any]) -> list[dict[str, Any]]:
        return teacher_history(
            history,
            example,
            hint_template=self.hint_template,
            answer_field=self.answer_field,
            solution_field=self.solution_field,
        )

    def _add_confidence_weights(self, batch: dict[str, torch.Tensor], examples: list[dict[str, Any]]) -> None:
        if self.confidence_field is not None:
            batch["confidence_weights"] = confidence_weights(
                examples, self.confidence_field, self.confidence_power, self.confidence_normalizer
            )


class SelfDistillTextCollator(SelfDistillBranchMixin):
    """Text collator (arXiv:2606.04036); ``hint_template=None`` builds no teacher branch. A row is rendered, tokenized
    and labelled once (:func:`encode_self_distill_row`, at prep), so collation pads; a raw row is encoded live."""

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        conversation_field: str,
        hint_template: str | None,
        train_on_completions_only: bool,
        answer_field: str | None,
        solution_field: str | None,
        confidence_power: float,
        confidence_field: str | None = None,
        confidence_normalizer: float | None = None,
        response_prompt_template: str | None = None,
        system_prompt: str | None = None,
        model_supports_system_role: bool = True,
        tools_field: str | None = None,
        interleaved_thinking: bool = False,
        model_config=None,
    ):
        reject_unusable_response_marker(
            tokenizer, response_prompt_template, train_on_completions_only, "Self-distillation"
        )
        self._init_self_distill_fields(
            hint_template, answer_field, solution_field, confidence_field, confidence_power, confidence_normalizer
        )
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.conversation_field = conversation_field
        self.system_prompt = system_prompt
        self.model_supports_system_role = model_supports_system_role
        self.tools_field = tools_field
        self.interleaved_thinking = interleaved_thinking
        self.response_prompt_template = response_prompt_template
        self.train_on_completions_only = train_on_completions_only
        self.eos_token_ids = resolve_eos_token_ids(tokenizer, model_config)

    def encode_row(self, example: dict[str, Any]) -> dict[str, list[int]]:
        return encode_self_distill_row(example, self)

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        rows = [example if "input_ids" in example else self.encode_row(example) for example in examples]
        batch = self._pad(rows, "")
        if self.builds_teacher_branch:
            batch.update(self._pad(rows, "teacher_"))
            require_aligned_responses(batch)
        self._add_confidence_weights(batch, examples)
        return batch

    def _pad(self, rows: list[dict[str, list[int]]], prefix: str) -> dict[str, torch.Tensor]:
        side = self.tokenizer.padding_side
        input_ids = [torch.tensor(row[f"{prefix}input_ids"]) for row in rows]
        return {
            f"{prefix}input_ids": pad(input_ids, padding_value=self.tokenizer.pad_token_id, padding_side=side),
            f"{prefix}attention_mask": pad([torch.ones_like(ids) for ids in input_ids], padding_side=side),
            f"{prefix}labels": pad(
                [torch.tensor(row[f"{prefix}labels"]) for row in rows],
                padding_value=LABEL_IGNORE_INDEX,
                padding_side=side,
            ),
        }


class SelfDistillVLMDataCollator(SelfDistillBranchMixin, VLMDataCollator):
    """VLM collator (arXiv:2606.04036). The text-only hint leaves the image grid unchanged, so the trainer reuses the
    student's image tensors; ``hint_template=None`` builds no teacher branch."""

    _COMPLETION_SPAN_POLICY = SELF_DISTILL_SPAN_POLICY

    def __init__(
        self,
        *args,
        hint_template: str | None,
        answer_field: str | None,
        solution_field: str | None,
        confidence_power: float,
        confidence_field: str | None = None,
        confidence_normalizer: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._init_self_distill_fields(
            hint_template, answer_field, solution_field, confidence_field, confidence_power, confidence_normalizer
        )

    def audit_row(self, example: dict[str, Any]) -> dict[str, Any]:
        """Refuse a row whose branches would supervise different tokens; returns no columns. Reads the text before
        image placeholders (all in the user turn) expand; mapped at prep, where the raise is world-uniform."""
        if not self.builds_teacher_branch:
            return {}
        history = example.get("history", [])
        labels = []
        for branch in (history, self._teacher_history(history, example)):
            text = self._render(branch, example)
            input_ids = torch.tensor([self.tokenizer(text, add_special_tokens=False)["input_ids"]])
            labels.append(self._build_labels(input_ids, torch.ones_like(input_ids)))
        require_aligned_responses({"labels": labels[0], "teacher_labels": labels[1]})
        return {}

    def __call__(self, examples):
        batch = super().__call__(examples)

        if self.builds_teacher_branch:
            # Never truncated: the hinted teacher is longer, and a cut would take response tokens the student keeps.
            teacher_batch = self._collate(
                examples,
                lambda example: self._teacher_history(example.get("history", []), example),
                "teacher branch (student history + privileged hint)",
            )
            batch["teacher_input_ids"] = teacher_batch["input_ids"]
            batch["teacher_attention_mask"] = teacher_batch["attention_mask"]
            batch["teacher_labels"] = teacher_batch["labels"]
            # Per-token tensors (GLM-4.6V mm_token_type_ids, a prefix LM's token_type_ids) must be the teacher's own.
            for key in SEQUENCE_ALIGNED_VISION_KEYS:
                if key in teacher_batch:
                    batch[f"teacher_{key}"] = teacher_batch[key]
            require_aligned_responses(batch)

        self._add_confidence_weights(batch, examples)
        return batch


def _encode_branch(
    collator: SelfDistillTextCollator, conversation: list[dict[str, Any]], row: dict[str, Any], branch: str
) -> tuple[list[int], list[int]]:
    """Token ids, refused rather than truncated past ``max_length``, and labels masked outside completions with
    ``train_on_completions_only``."""
    rendered = render_conversation(
        collator.tokenizer,
        conversation,
        row,
        conversation_field=collator.conversation_field,
        system_prompt=collator.system_prompt,
        model_supports_system_role=collator.model_supports_system_role,
        interleaved_thinking=collator.interleaved_thinking,
        tools_field=collator.tools_field,
    )
    # tokenize_rendered single-sources the specials: add_special_tokens=True double-BOSes templates emitting BOS.
    input_ids = torch.tensor([tokenize_rendered(collator.tokenizer, rendered)["input_ids"]])
    if input_ids.size(1) > collator.max_length:
        raise ValueError(
            f"Self-distillation {branch} branch contains a {input_ids.size(1)}-token sequence, "
            f"{input_ids.size(1) - collator.max_length} tokens over max_length={collator.max_length}. Truncating "
            f"would cut response tokens that must stay byte-identical between student and teacher (OPD row "
            f"alignment). Raise max_length (the teacher needs headroom for the privileged hint) or "
            f"drop over-length rows before training."
        )
    labels = build_training_labels(
        input_ids,
        collator.tokenizer,
        collator.response_prompt_template,
        collator.train_on_completions_only,
        attention_mask=torch.ones_like(input_ids),
        eos_token_ids=collator.eos_token_ids,
    )
    return input_ids[0].tolist(), labels[0].tolist()


def encode_self_distill_row(example: dict[str, Any], collator: SelfDistillTextCollator) -> dict[str, list[int]]:
    """``input_ids`` and ``labels`` per branch; an over-length branch or misaligned responses raise. Mapped at prep,
    where a raise is world-uniform. Module level, as ``num_proc`` map workers reject bound methods, and calling every
    helper by bare name, so the map's cache key folds in their source. The collator rides ``fn_kwargs`` for its knobs,
    which key the map through :meth:`~SelfDistillBranchMixin.cache_signature`."""
    history = example[collator.conversation_field]
    branches = [("student", "", history)]
    if collator.builds_teacher_branch:
        hinted = teacher_history(
            history,
            example,
            hint_template=collator.hint_template,
            answer_field=collator.answer_field,
            solution_field=collator.solution_field,
        )
        branches.append(("teacher", "teacher_", hinted))
    encoded = {}
    for branch, prefix, conversation in branches:
        encoded[f"{prefix}input_ids"], encoded[f"{prefix}labels"] = _encode_branch(
            collator, conversation, example, branch
        )
    if collator.builds_teacher_branch:
        require_aligned_responses({key: torch.tensor([value]) for key, value in encoded.items()})
    return encoded


def audit_self_distill_row(example: dict[str, Any], collator) -> dict[str, Any]:
    """Module-level seam for mapping ``audit_row`` with ``num_proc`` workers."""
    return collator.audit_row(example)
