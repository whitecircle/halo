"""The RLVR graders — strict ``\\boxed{}`` accuracy and a regex format check — and their reward terms."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from src.rewards.matching import extract_last_boxed
from src.rewards.spec import RewardTerm

DEFAULT_FORMAT_PATTERN = r"<think>.*?</think>\s*<answer>.*?</answer>"


@dataclass(frozen=True, kw_only=True)
class AccuracyTerm(RewardTerm):
    """The completion's last ``\\boxed{}`` equals the row's answer (1) or not (0)."""

    source: ClassVar[str] = "accuracy"

    name: str = "accuracy"


@dataclass(frozen=True, kw_only=True)
class FormatTerm(RewardTerm):
    """The completion matches ``pattern`` (1) or not (0); the pattern spans lines."""

    source: ClassVar[str] = "format"

    name: str = "format"
    pattern: str = DEFAULT_FORMAT_PATTERN

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            re.compile(self.pattern, re.DOTALL)
        except re.error as e:
            raise ValueError(f"format term {self.name!r}: invalid pattern {self.pattern!r}: {e}") from e

    @property
    def compiled(self) -> re.Pattern:
        return re.compile(self.pattern, re.DOTALL)


def completion_text(completion: Any) -> str:
    """Last assistant message content for a conversational completion, or the string itself."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def accuracy_reward(completions: Sequence[Any], answer: Sequence[Any], **kwargs: Any) -> list[float]:
    """1.0 where the completion's last ``\\boxed{}`` equals the ground truth, else 0.0.

    **Strict boxed exact-match**, deliberately not the environments' validated-answer chain
    (:func:`src.rewards.matching.validate_answer`, exact + numeric, which every environment grader
    uses). The two differ on purpose: this one strips a GSM8K ``####`` rationale prefix and ``,``/``$``
    from both sides and then requires string equality, so it grades only what the policy put in the
    box; the environment chain normalizes case, re-extracts a box from the *ground truth* as well,
    and accepts a numeric match within ``rtol=0.01``. Swapping in the chain would silently re-grade
    every shipped RLVR recipe (``0.5`` vs ``.5`` and ``7 \\boxed{42}`` vs ``42`` flip verdicts) —
    ``tests/cpu/grpo/test_rlvr_accuracy_reward.py`` pins the divergence.
    """
    # strict: a short answer column would otherwise truncate silently and return fewer rewards than
    # completions, scoring a different set of rows than was generated.
    rewards = []
    for completion, gt in zip(completions, answer, strict=True):
        extracted = (extract_last_boxed(completion_text(completion)) or "").strip()
        # GSM8K stores the rationale plus "#### <final>"; keep only the final answer.
        gt_normalized = str(gt)
        if "####" in gt_normalized:
            gt_normalized = gt_normalized.rsplit("####", 1)[-1]
        gt_normalized = gt_normalized.strip().replace(",", "").replace("$", "")
        extracted = extracted.replace(",", "").replace("$", "")
        # An empty extraction is never correct, even against a blank ground truth: a missing answer
        # column would otherwise pay full reward for every box-less completion.
        rewards.append(1.0 if extracted and extracted == gt_normalized else 0.0)
    return rewards


def format_reward(completions: Sequence[Any], pattern: re.Pattern, **kwargs: Any) -> list[float]:
    """1.0 where the completion matches the compiled ``pattern``, else 0.0."""
    return [1.0 if pattern.search(completion_text(completion)) else 0.0 for completion in completions]


def accuracy_grader(term: RewardTerm):
    return accuracy_reward


def format_grader(term: FormatTerm):
    compiled = term.compiled

    def grader(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return format_reward(completions, compiled)

    return grader


# Term type → grader builder, for the RLVR arm.
RLVR_GRADERS = {AccuracyTerm: accuracy_grader, FormatTerm: format_grader}
