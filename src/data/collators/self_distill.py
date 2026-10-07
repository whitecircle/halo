"""Text collator for SDPG self-distillation: the student batch plus a ``teacher_*`` branch whose last
user turn carries the privileged hint; the response tokens are byte-identical across the two.

The hint rendering, its injection, the response-alignment contract and the confidence weighting live
here rather than in a shared leaf: the VLM self-distillation collator and the on-policy SDPG trainer
are their only other callers.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from src.args.mixins import format_field_names
from src.data.pipeline.rendered import render_conversation, tokenize_rendered
from src.data.spans import (
    LABEL_IGNORE_INDEX,
    build_completion_only_labels,
    require_rendered_response_marker,
    resolve_eos_token_ids,
)


def privileged_hint(template: str, **slots: Any) -> str | None:
    """The hint ``template`` renders from one row's ``slots``, or ``None`` when a slot it names is blank.

    The template states its slots as fact, so a missing answer or solution would mislead the teacher
    rather than privilege it; no hint leaves it on the plain prompt. Slots the template does not name
    are never read; a named slot the arm does not supply raises ``KeyError``. Shared by both SDPG arms.
    """
    named = {name: slots[name] for name in format_field_names(template)}
    if any(value is None or not str(value).strip() for value in named.values()):
        return None
    return template.format(**named)


def inject_privileged_hint(history: list[dict[str, Any]], hint: str | None) -> list[dict[str, Any]]:
    """Return a copy of ``history`` with ``hint`` appended to the last user turn (unchanged for ``None``)."""
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
    """``history`` with ``row``'s privileged hint appended to its last user turn: the teacher branch's
    conversation, for both collators and the VLM over-length filter."""
    hint = privileged_hint(
        hint_template,
        answer=row.get(answer_field) if answer_field else None,
        solution=row.get(solution_field) if solution_field else None,
    )
    return inject_privileged_hint(history, hint)


def require_aligned_responses(batch: Mapping[str, torch.Tensor]) -> None:
    """Raise unless each row's student and teacher branches supervise the same tokens, in order.

    OPD pairs the k-th supervised position of one branch with the k-th of the other. The hint only
    lengthens the prompt, so a label span reaching into the prompt, or a chat template that renders a
    response token differently once hinted, pairs different tokens.
    """
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
    """The mean of ``conf ** power`` over the train split: the one divisor every batch's weights share.

    A per-batch mean is 1.0 at batch size 1, which would leave the weighting a no-op; dividing by the
    dataset mean keeps the effective learning rate at any batch size. ``test`` (or ``None``) is checked
    too, since eval rows are weighted by the same divisor.
    """
    if test is not None:
        _checked_confidences(test, field)
    normalizer = float(_checked_confidences(train, field).pow(power).mean())
    if normalizer <= 0:
        raise ValueError(f"Every train confidence in {field!r} is 0, so every row would carry zero weight.")
    return normalizer


def require_confidence_normalizer(confidence_field: str | None, normalizer: float | None) -> None:
    """Refuse a confidence column without its dataset normalizer (:func:`confidence_normalizer`)."""
    if confidence_field is not None and normalizer is None:
        raise ValueError(
            f"confidence_field={confidence_field!r} needs confidence_normalizer, the train split's mean of "
            f"conf ** confidence_power (confidence_normalizer()); a per-batch mean is 1.0 at batch size 1."
        )


def confidence_weights(
    examples: Sequence[Mapping[str, Any]], field: str, power: float, normalizer: float
) -> torch.Tensor:
    """Per-sample ``conf ** power / normalizer`` weights."""
    confidences = torch.tensor([float(example[field]) for example in examples], dtype=torch.float32)
    return confidences**power / normalizer


class SelfDistillBranchMixin:
    """What both self-distillation collators (text and VLM) add to a student batch: the privileged
    fields, the hinted teacher history and the per-row confidence weights."""

    def _init_self_distill_fields(
        self,
        hint_template: str | None,
        answer_field: str | None,
        solution_field: str | None,
        confidence_field: str | None,
        confidence_power: float,
        confidence_normalizer: float | None,
    ) -> None:
        require_confidence_normalizer(confidence_field, confidence_normalizer)
        self.hint_template = hint_template
        self.answer_field = answer_field
        self.solution_field = solution_field
        self.confidence_field = confidence_field
        self.confidence_power = confidence_power
        self.confidence_normalizer = confidence_normalizer

    @property
    def builds_teacher_branch(self) -> bool:
        """Whether batches carry the ``teacher_*`` branch the OPD term reads."""
        return self.hint_template is not None

    def cache_signature(self) -> dict[str, Any]:
        """Every knob this collator renders with, to thread through the audit map's cache key.

        The collator rides ``fn_kwargs`` (see :func:`audit_self_distill_row`), and a dataset-map
        fingerprint reads an object that is neither tokenizer nor processor as its class name alone
        — so without this the audit's verdict outlives a change to any knob below and a stale cache
        skips it entirely. Read off ``__dict__``, not a hand-listed subset: a field added to
        ``__init__`` enters the key with it.
        """
        return dict(vars(self))

    def _teacher_history(self, history: list[dict[str, Any]], example: dict[str, Any]) -> list[dict[str, Any]]:
        """``history`` with the row's privileged hint appended to its last user turn."""
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
    """Text collator for SDPG-style self-distillation (arXiv:2606.04036).

    Reads raw conversation rows + privileged answer/solution fields, emitting the student batch and a
    teacher branch (``teacher_*``) whose last user turn carries the gold-answer hint. The assistant
    response is byte-identical across both sequences (:func:`require_aligned_responses`), so the
    teacher can supervise the student on the shared tokens. ``hint_template=None`` builds no teacher
    branch (the OPD term is off). Optionally emits ``confidence_weights``.
    """

    def __init__(
        self,
        tokenizer,
        *,
        max_length: int,
        conversation_field: str,
        hint_template: str | None,
        train_on_completions_only: bool,
        answer_field: str | None = "answer",
        solution_field: str | None = "solution",
        confidence_field: str | None = None,
        confidence_power: float = 4.0,
        confidence_normalizer: float | None = None,
        response_prompt_template: str | None = None,
        system_prompt: str | None = None,
        model_supports_system_role: bool = True,
        tools_field: str | None = None,
        interleaved_thinking: bool = False,
        model_config=None,
    ):
        require_rendered_response_marker(
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

    def _render(self, history: list[dict[str, Any]], row: dict[str, Any]) -> str:
        """Chat-template one conversation through the shared text renderer, so the student and
        teacher branches honor exactly the knobs the SFT text pipeline does."""
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
        # NEVER truncate — see the raise below. tokenize_rendered single-sources the rendered specials:
        # a bare add_special_tokens=True double-BOSes templates that emit BOS themselves.
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

        Shared by the student batch and the teacher branch so both select the SAME response tokens.
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
        """Collate one raw row through the whole collate-time contract (length, response alignment,
        confidence); returns no columns.

        Mapped over the dataset at prep via :func:`audit_self_distill_row` (``num_proc`` maps
        reject bound methods), where the coordinated map machinery turns a raise into a
        world-uniform failure. The same raise at collate time is rank-local: only the rank whose
        batch drew the bad row reads the error while its peers sit in the step's collectives until
        the NCCL watchdog.
        """
        self([example])
        return {}

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        student = self._tokenize([ex[self.conversation_field] for ex in examples], examples, branch="student")
        batch = {
            "input_ids": student["input_ids"],
            "attention_mask": student["attention_mask"],
            "labels": self._build_labels(student["input_ids"], student["attention_mask"]),
        }
        if self.builds_teacher_branch:
            histories = [self._teacher_history(ex[self.conversation_field], ex) for ex in examples]
            teacher = self._tokenize(histories, examples, branch="teacher")
            batch["teacher_input_ids"] = teacher["input_ids"]
            batch["teacher_attention_mask"] = teacher["attention_mask"]
            batch["teacher_labels"] = self._build_labels(teacher["input_ids"], teacher["attention_mask"])
            require_aligned_responses(batch)
        self._add_confidence_weights(batch, examples)
        return batch


def audit_self_distill_row(example: dict[str, Any], collator) -> dict[str, Any]:
    """Module-level seam for mapping a SelfDistill collator's ``audit_row`` (text or VLM) with
    ``num_proc`` workers, which reject bound methods (every worker would pickle ``self``). The collator
    rides ``fn_kwargs`` instead — its tokenizer, processor and config pickle cleanly."""
    return collator.audit_row(example)
