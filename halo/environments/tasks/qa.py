"""Question-answering task environments with rule-based rewards (no neural reward model)."""

import numbers
import re
from typing import Any

from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.messages import Trajectory
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.tools.definitions import NativeToolRegistry
from halo.environments.tools.toolsets import build_python_tools, build_search_tools
from halo.rewards.graders.matching import is_correct_answer

# MMLU-Pro tops out at 10 options. ``ExamQAEnvironment``'s system prompt is built from this range, so the instruction
# matches what the grader accepts.
MULTIPLE_CHOICE_LETTERS = "ABCDEFGHIJ"

_CHOICE_LETTER = f"[{MULTIPLE_CHOICE_LETTERS}]"
# A choice already led by its own letter ("B: Jupiter", "(B) Jupiter", "B. Jupiter") is shown as written.
_LETTER_LABEL = re.compile(rf"\(?({_CHOICE_LETTER})[).:]\s")

# Most specific first. In prose a choice is an uppercase letter standing alone: read case-insensitively or up to the
# next letter, "the answer is a tie", "the answer is definitely C" and "e.g." would each name a choice. A bracketed or
# bare letter is a choice in either case.
_MULTIPLE_CHOICE_PATTERNS = [
    re.compile(pattern)
    for pattern in (
        rf"(?i:answer\s+is)\s*:?\s*[(*]*({_CHOICE_LETTER})(?![A-Za-z])",
        rf"[\(\[]((?i:{_CHOICE_LETTER}))[\)\]]",
        rf"(?:^|\s)({_CHOICE_LETTER})[.):](?![A-Za-z])",
        rf"(?i:option|choice)\s+({_CHOICE_LETTER})(?![A-Za-z])",
        rf"^((?i:{_CHOICE_LETTER}))$",
    )
]

_QA_SEARCH_SYSTEM_PROMPT = (
    "You are a research assistant. Use the available tools to find "
    "information and answer the question accurately.\n\n"
    "When you have enough information, provide your final answer "
    "directly (without tool calls). Be concise -- state just the answer."
)


def render_choices(choices: list[Any]) -> str:
    """One line per choice, led by its letter (an already-labelled choice is kept): MMLU-style rows carry bare option
    texts, so without letters the model could not name a choice."""
    if len(choices) > len(MULTIPLE_CHOICE_LETTERS):
        raise ValueError(
            f"{len(choices)} choices, but only {MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]} "
            f"are gradable: the later choices could never be scored."
        )
    lines = []
    for letter, choice in zip(MULTIPLE_CHOICE_LETTERS, choices, strict=False):
        text = str(choice)
        label = _LETTER_LABEL.match(text)
        lines.append(text if label is not None and label.group(1) == letter else f"{letter}. {text}")
    return "\n".join(lines)


def multiple_choice_match(predicted: str, expected: str) -> bool:
    """``expected`` must already be a letter (:meth:`ExamQAEnvironment._expected_choice_letter` converts an index)."""
    pred = str(predicted).strip()
    exp = str(expected).strip().upper()

    if len(exp) != 1 or exp not in MULTIPLE_CHOICE_LETTERS:
        return False

    for pattern in _MULTIPLE_CHOICE_PATTERNS:
        match = pattern.search(pred)
        if match:
            return match.group(1).upper() == exp

    # No startswith fallback: "Although I'm not sure..." must not score as choice "A".
    return False


def build_qa_search_environment(
    search_backend: str | None = None,
    include_python_tools: bool = False,
    system_prompt: str | None = None,
    **kwargs,
) -> NativeToolUseEnvironment:
    """A factual-QA-with-search environment (SimpleQA/GAIA/TriviaQA/PopQA)."""
    registry = build_search_tools(backend=search_backend)
    if include_python_tools:
        registry.merge(build_python_tools())

    # A kwargs default: the registry forwards the whole env_config, so a keyword would make an explicit
    # ``requires_answer`` a TypeError. Required because the parent otherwise grades merely completing 1.
    if kwargs.get("requires_answer") is None:
        kwargs["requires_answer"] = True
    return NativeToolUseEnvironment(
        tool_registry=registry,
        system_prompt=system_prompt or _QA_SEARCH_SYSTEM_PROMPT,
        **kwargs,
    )


class ExamQAEnvironment(NativeToolUseEnvironment):
    """Exam-style QA (GPQA/MMLU/ARC), closed-book or open-book. ``choices`` in context triggers MC letter grading."""

    # An exam answer needs one response plus limited lookup, fewer turns than the generic tool loop.
    DEFAULT_MAX_TURNS = 8

    # Every grading path here scores the final response against the expected answer; without one the
    # parent's completion fallback grades any episode that answered at all 1.
    requires_answer = True

    # Interpolated from the grader's constant: a narrower instruction would steer the model away from valid options
    # (E-J on a 10-choice MMLU-Pro row).
    EXAM_SYSTEM_PROMPT = (
        "You are taking an exam. Answer the question{search_instruction}.\n\n"
        "For multiple-choice questions, clearly state your answer as a single letter from the "
        f"choices listed ({MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]}).\n"
        "For open-ended questions, provide a concise, accurate answer.\n\n"
        "When ready, provide your final answer directly."
    )

    def __init__(
        self,
        open_book: bool = False,
        search_backend: str | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ):
        if open_book:
            registry = build_search_tools(backend=search_backend)
            instruction = " using the available tools to search for information"
        else:
            if search_backend is not None:
                # Closed-book registers no search tool, so the backend name would reach nothing,
                # including a misspelled one that every other entry point rejects at construction.
                raise ValueError(
                    f"search_backend={search_backend!r} is set on a closed-book exam_qa, which "
                    f"registers no search tool, so it would never be used. Set open_book=true to "
                    f"give the model search, or drop search_backend."
                )
            registry = NativeToolRegistry()
            instruction = ""

        default_prompt = self.EXAM_SYSTEM_PROMPT.format(search_instruction=instruction)

        super().__init__(
            tool_registry=registry,
            system_prompt=system_prompt or default_prompt,
            **kwargs,
        )

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        traj = super()._reset_single(prompt, context)
        context = context or {}

        traj.info["expected_answer"] = context.get(ANSWER_KEY)
        traj.info["choices"] = context.get("choices")
        traj.info["is_multiple_choice"] = context.get("choices") is not None

        if traj.info["is_multiple_choice"] and traj.info["expected_answer"] is not None:
            traj.info["expected_answer"] = self._expected_choice_letter(
                traj.info["expected_answer"], traj.info["choices"]
            )

        if traj.info["choices"]:
            traj.append_to_last_user(f"\n\nChoices:\n{render_choices(traj.info['choices'])}")

        return traj

    @staticmethod
    def _expected_choice_letter(expected: Any, choices: Any) -> str:
        """A letter passes, a 0-based integer index becomes its letter, and any other shape raises at episode start, a
        digit string too, since ARC labels some rows ``"1"``-``"5"``, 1-based, so no one reading of it is safe."""
        # bool is an int subclass, so True would otherwise index choice "B".
        if not isinstance(expected, bool):
            text = str(expected).strip()
            if len(text) == 1 and text.upper() in MULTIPLE_CHOICE_LETTERS:
                return text.upper()
            if isinstance(expected, numbers.Integral):
                index = int(expected)
                if 0 <= index < min(len(choices), len(MULTIPLE_CHOICE_LETTERS)):
                    return MULTIPLE_CHOICE_LETTERS[index]
                raise ValueError(
                    f"multiple-choice answer index {index} does not address any of the {len(choices)} "
                    f"choices gradable as {MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]}."
                )
            if text.isdigit():
                raise ValueError(
                    f"multiple-choice answer {expected!r} is a digit string: as a 0-based index and as a 1-based "
                    "label (ARC's answerKey) it names different choices, so convert it to a choice letter "
                    f"({MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]}) or an int index in dataset "
                    "preparation."
                )
        raise ValueError(
            f"multiple-choice answer {expected!r} is neither a choice letter "
            f"({MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]}) nor a 0-based index into "
            f"'choices'; convert the answer column in dataset preparation."
        )

    def _scoring_reference(self, trajectory: Trajectory) -> Any:
        """The answer as the grader compares it: a multiple-choice row's letter, not its index."""
        return trajectory.info.get("expected_answer")

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        # Delegated first: the parent owns both the incomplete-episode and the no-expected-answer cases.
        expected = trajectory.info.get("expected_answer")
        if expected is None:
            return super()._grade_episode(trajectory, context)

        if not trajectory.info.get("completed"):
            return EpisodeGrade(0.0)

        final_response = trajectory.info.get("final_response", "")

        if trajectory.info.get("is_multiple_choice"):
            return EpisodeGrade(1.0 if multiple_choice_match(final_response, str(expected)) else 0.0)

        return EpisodeGrade(1.0 if is_correct_answer(final_response, expected) else 0.0)
