"""Question-answering task environments with rule-based rewards (no neural reward model)."""

import re
from typing import Any

from src.environments.base import Trajectory
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.rewards import compute_answer_reward
from src.environments.tools.definitions import NativeToolRegistry
from src.environments.tools.factories import create_native_python_tools, create_native_search_tools

# The choice letters :func:`multiple_choice_match` scores (MMLU-Pro tops out at 10 options).
# ``ExamQAEnvironment``'s system prompt is built from this same range, so the instruction the model
# reads matches what the grader accepts.
MULTIPLE_CHOICE_LETTERS = "ABCDEFGHIJ"

_CHOICE_LETTER = f"[{MULTIPLE_CHOICE_LETTERS}]"

# Priority order, most specific first: "The answer is A" / "(A)" / "A." / "Option A" / a bare letter.
_MULTIPLE_CHOICE_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        rf"(?:the\s+)?answer\s+is\s*:?\s*\(?({_CHOICE_LETTER})\)?",
        rf"[\(\[]({_CHOICE_LETTER})[\)\]]",
        rf"(?:^|\s)({_CHOICE_LETTER})[.):]",
        rf"(?:option|choice)\s+({_CHOICE_LETTER})",
        rf"^({_CHOICE_LETTER})$",
    )
]


def multiple_choice_match(predicted: str, expected: str) -> bool:
    """Extract a choice letter (:data:`MULTIPLE_CHOICE_LETTERS`) and compare to expected.

    Handles "A", "(A)", "A.", "A)", "The answer is A", "Option A", "Choice A", etc. ``expected`` must
    already be a letter; an index (MMLU's ``answer`` column is an int) always returns False, since the
    choice ordering needed to convert it lives in ``context["choices"]``, which this matcher does not
    see. That conversion is done by :meth:`ExamQAEnvironment._expected_choice_letter`.
    """
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


QA_SEARCH_SYSTEM_PROMPT = (
    "You are a research assistant. Use the available tools to find "
    "information and answer the question accurately.\n\n"
    "When you have enough information, provide your final answer "
    "directly (without tool calls). Be concise -- state just the answer."
)


def create_qa_search_environment(
    search_backend: str | None = None,
    include_python_tools: bool = False,
    system_prompt: str | None = None,
    **kwargs,
) -> NativeToolUseEnvironment:
    """Create a factual-QA-with-search environment (SimpleQA/GAIA/TriviaQA/PopQA).

    Sets ``require_tool_use``; final answer validated against ``context["answer"]`` by the parent reward.
    """
    registry = create_native_search_tools(backend=search_backend)
    if include_python_tools:
        registry.merge(create_native_python_tools())

    # setdefault, not a keyword argument: the registry forwards the whole env_config, so an explicit
    # ``require_tool_use`` in environment_kwargs must override this rather than raise a
    # duplicate-keyword TypeError.
    kwargs.setdefault("require_tool_use", True)
    return NativeToolUseEnvironment(
        tool_registry=registry,
        system_prompt=system_prompt or QA_SEARCH_SYSTEM_PROMPT,
        **kwargs,
    )


class ExamQAEnvironment(NativeToolUseEnvironment):
    """Exam-style QA (GPQA/MMLU/ARC), multiple-choice and open-ended.

    Closed-book (no tools) or open-book (search tools). ``choices`` in context triggers MC letter grading.
    """

    # An exam answer needs one response plus limited lookup, fewer turns than the generic tool loop.
    DEFAULT_MAX_TURNS = 8

    # The letter range is interpolated from the grader's constant rather than restated: an
    # instruction narrower than what multiple_choice_match scores would steer the model away from
    # valid options (E-J on a 10-choice MMLU-Pro row).
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
            registry = create_native_search_tools(backend=search_backend)
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

    @staticmethod
    def _expected_choice_letter(expected: Any, choices: Any) -> str:
        """Normalize a multiple-choice row's expected answer to a letter :func:`multiple_choice_match`
        scores: a letter passes through, a 0-based index into ``choices`` becomes its letter.

        MMLU/ARC ship ``answer`` as an int (occasionally a digit string), and the matcher rejects
        anything that is not a single letter, so an unconverted row would score ``failure_reward`` on
        every completion and leave its GRPO group with zero variance. Any other shape raises here, at
        episode start.
        """
        # bool is an int subclass, so True would otherwise index choice "B".
        if not isinstance(expected, bool):
            text = str(expected).strip()
            if len(text) == 1 and text.upper() in MULTIPLE_CHOICE_LETTERS:
                return text.upper()
            if isinstance(expected, int) or text.isdigit():
                index = int(text)
                if 0 <= index < min(len(choices), len(MULTIPLE_CHOICE_LETTERS)):
                    return MULTIPLE_CHOICE_LETTERS[index]
                raise ValueError(
                    f"multiple-choice answer index {index} does not address any of the {len(choices)} "
                    f"choices gradable as {MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]}."
                )
        raise ValueError(
            f"multiple-choice answer {expected!r} is neither a choice letter "
            f"({MULTIPLE_CHOICE_LETTERS[0]}-{MULTIPLE_CHOICE_LETTERS[-1]}) nor a 0-based index into "
            f"'choices'; convert the answer column in dataset preparation."
        )

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Store expected answer and append choices to prompt."""
        traj = super()._reset_single(prompt, context)
        context = context or {}

        traj.info["expected_answer"] = context.get("answer")
        traj.info["choices"] = context.get("choices")
        traj.info["is_multiple_choice"] = context.get("choices") is not None

        if traj.info["is_multiple_choice"] and traj.info["expected_answer"] is not None:
            traj.info["expected_answer"] = self._expected_choice_letter(
                traj.info["expected_answer"], traj.info["choices"]
            )

        if traj.info["choices"]:
            choices_text = "\n".join(traj.info["choices"])
            for msg in reversed(traj.messages):
                if msg.role == "user":
                    msg.content += f"\n\nChoices:\n{choices_text}"
                    break

        return traj

    def _compute_reward(
        self,
        trajectory: Trajectory,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Validate answer using MC match or shared validation."""
        # Delegated first: the parent handles both the incomplete-episode and the no-expected-answer
        # cases, and rebuilds the same shaped base itself.
        expected = trajectory.info.get("expected_answer")
        if expected is None:
            return super()._compute_reward(trajectory, context)

        base_reward = self._shaped_base_reward(trajectory)
        if not trajectory.info.get("completed"):
            return base_reward + self.failure_reward

        final_response = trajectory.info.get("final_response", "")

        if trajectory.info.get("is_multiple_choice"):
            if multiple_choice_match(final_response, str(expected)):
                return base_reward + self.success_reward
            return base_reward + self.failure_reward

        answer_reward = compute_answer_reward(
            predicted=final_response,
            expected=expected,
            success_reward=self.success_reward,
            failure_reward=self.failure_reward,
        )
        return base_reward + answer_reward
