"""The RLVR graders (strict ``\\boxed{}`` accuracy and a regex format check), their reward terms, and the reward
sources the RLVR arm admits."""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from halo.rewards.graders.matching import extract_last_boxed
from halo.rewards.terms import JudgeTerm, RewardModelTerm, RewardTerm, sources_of

_DEFAULT_FORMAT_PATTERN = r"<think>.*?</think>\s*<answer>.*?</answer>"


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
    pattern: str = _DEFAULT_FORMAT_PATTERN

    def __post_init__(self) -> None:
        super().__post_init__()
        try:
            compiled = re.compile(self.pattern, re.DOTALL)
        except re.error as e:
            raise ValueError(f"format term {self.name!r}: invalid pattern {self.pattern!r}: {e}") from e
        # Not a field: a config cannot set it, and the term compares by its pattern.
        object.__setattr__(self, "compiled", compiled)


def completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def accuracy_reward(completions: Sequence[Any], answer: Sequence[Any], **kwargs: Any) -> list[float]:
    """1.0 where the completion's last ``\\boxed{}`` equals the ground truth, else 0.0.

    Strict on purpose, unlike :func:`halo.rewards.graders.matching.is_correct_answer`: it strips a GSM8K ``####``
    rationale prefix and ``,``/``$`` from both sides and requires string equality. Swapping in that chain would
    silently re-grade every shipped RLVR recipe (``0.5`` vs ``.5`` and ``7 \\boxed{42}`` vs ``42`` flip verdicts).
    """
    # strict: a short answer column would otherwise score a different set of rows than was generated.
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


def _format_grader(term: FormatTerm):
    def grader(completions: Sequence[Any], **kwargs: Any) -> list[float]:
        return [1.0 if term.compiled.search(completion_text(completion)) else 0.0 for completion in completions]

    return grader


RLVR_GRADERS = {AccuracyTerm: lambda term: accuracy_reward, FormatTerm: _format_grader}
RLVR_REWARD_SOURCES = sources_of(AccuracyTerm, FormatTerm, JudgeTerm, RewardModelTerm)
