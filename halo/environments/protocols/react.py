"""ReAct environment: Thought -> Action -> Observation, action parsed from text (https://arxiv.org/abs/2210.03629)."""

import ast
import contextlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from halo.config_validation import present, require_callable, require_finite
from halo.environments.environment_base import EnvironmentBase
from halo.environments.episode_keys import ANSWER_KEY, COMPLETED_KEY, CONTEXT_KEY, LAST_TURN_KEY
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.messages import Message, StepOutcome, Trajectory
from halo.environments.tool_accounting import (
    MAGNITUDE_WHY,
    initial_tool_counters,
    tool_call_outcome,
    validate_tool_budgets,
)
from halo.environments.tools.definitions import (
    MALFORMED_LITERAL_ERRORS,
    NativeToolRegistry,
    is_refused_unrun,
    parse_python_expression,
)
from halo.environments.tools.toolsets import build_math_tools, build_python_tools, build_search_tools
from halo.inference.response import ENGINE_CUT_FINISH_REASONS
from halo.rewards.graders.matching import is_correct_answer

logger = logging.getLogger(__name__)

# The ReAct turn grammar: a Thought, then an Action or a Final Answer (the answer wins when both appear).
_THOUGHT_RE = re.compile(
    r"(?:^|\n)\s*(?:Thought|THOUGHT|Think|THINK|Reasoning|REASONING)\s*:\s*(.+?)(?=\n\s*(?:Action|ACTION|Final|FINAL)|$)",
    re.IGNORECASE | re.DOTALL,
)
_FINAL_ANSWER_RE = re.compile(
    r"(?:^|\n)\s*(?:Final Answer|FINAL ANSWER|Answer|ANSWER)\s*:\s*(.+?)$", re.IGNORECASE | re.DOTALL
)
_ACTION_RE = re.compile(
    r"(?:^|\n)\s*(?:Action|ACTION)\s*:\s*(.+?)(?=\n\s*(?:Observation|OBSERVATION)|$)", re.IGNORECASE | re.DOTALL
)
# An action's spellings: ``tool_name(arg=...)``, ``tool_name: arg=...``, a bare ``tool_name``.
_CALL_ACTION_RE = re.compile(r"^(\w+)\s*\((.*)\)$", re.DOTALL)
_COLON_ACTION_RE = re.compile(r"^(\w+)\s*:\s*(.*)$", re.DOTALL)
_BARE_ACTION_RE = re.compile(r"^(\w+)$")
# One ``key=value`` argument: double- or single-quoted, a JSON object or list, or a bare token.
_ARGUMENT_RE = re.compile(r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\{[^}]*\})|(\[[^\]]*\])|([^,\s]+))')


@dataclass
class _ReActStep:
    thought: str | None = None
    action: str | None = None
    action_args: dict[str, Any] | None = None
    final_answer: str | None = None

    @property
    def has_action(self) -> bool:
        return self.action is not None

    @property
    def has_final_answer(self) -> bool:
        return self.final_answer is not None


def _parse_function_args(args_str: str) -> dict[str, Any]:
    """``arg1="val1", arg2=123``, as a Python call of literals first, so a triple-quoted or escaped code argument
    arrives whole, else by pattern."""
    args = {}
    if not args_str:
        return args
    with contextlib.suppress(*MALFORMED_LITERAL_ERRORS):
        call = parse_python_expression(f"_({args_str})")
        if isinstance(call, ast.Call) and not call.args and all(kw.arg for kw in call.keywords):
            return {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords}

    for match in _ARGUMENT_RE.finditer(args_str):
        key = match.group(1)
        # First MATCHED alternative by ``is not None``: truthiness would turn ``expression=""`` into None.
        value = next((g for g in match.groups()[1:] if g is not None), None)

        if value and (value.startswith("{") or value.startswith("[")):
            with contextlib.suppress(*MALFORMED_LITERAL_ERRORS):
                value = json.loads(value)
        elif value:
            with contextlib.suppress(ValueError, TypeError):
                value = float(value) if "." in value else int(value)

        args[key] = value

    return args


def _parse_action(action_text: str) -> tuple[str | None, dict[str, Any] | None]:
    """``tool_name(arg=...)``, JSON ``{"name":..., "arguments":...}``, ``tool_name: arg=...``, or bare name."""
    action_text = action_text.strip()

    if action_text.startswith("{"):
        try:
            data = json.loads(action_text)
            # Model-authored JSON: a non-dict ``function`` or ``arguments`` must degrade to an unknown-tool or
            # empty-args call the protocol penalizes, not an error that ends the episode.
            function = data.get("function")
            function = function if isinstance(function, dict) else {}
            name = data.get("name") or function.get("name")
            args = data.get("arguments") or function.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            return (name if isinstance(name, str) else None), (args if isinstance(args, dict) else {})
        except MALFORMED_LITERAL_ERRORS:
            pass

    call_match = _CALL_ACTION_RE.match(action_text) or _COLON_ACTION_RE.match(action_text)
    if call_match:
        return call_match.group(1), _parse_function_args(call_match.group(2).strip())

    name_match = _BARE_ACTION_RE.match(action_text)
    if name_match:
        return name_match.group(1), {}

    return None, None


def parse_react_output(text: str) -> _ReActStep:
    step = _ReActStep()

    thought_match = _THOUGHT_RE.search(text)
    if thought_match:
        step.thought = thought_match.group(1).strip()

    final_match = _FINAL_ANSWER_RE.search(text)
    if final_match:
        step.final_answer = final_match.group(1).strip()
        return step

    action_match = _ACTION_RE.search(text)
    if action_match:
        action_text = action_match.group(1).strip()
        step.action, step.action_args = _parse_action(action_text)

    return step


class ReActEnvironment(EnvironmentBase):
    """Actions are parsed out of the text, so no tool schema reaches the server; the tools are named in the system
    prompt. ``no_thought_penalty`` prices acting without a Thought."""

    # The default check grades against ``context["answer"]``; an ``answer_validator`` clears this per instance.
    requires_answer = True

    # Never ask for shorter reasoning: these texts are trained on wherever a recovery succeeds, so an instruction here
    # becomes a global lesson learned far outside the situation it was written for.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you produced an Action or a Final Answer, so nothing "
        "was recorded. Give your next Action now, or your Final Answer if you already have the "
        "solution."
    )
    EMPTY_TURN_NUDGE = (
        "Your previous turn ended without an Action or a Final Answer, so nothing was recorded. Give "
        "your next Action now, or your Final Answer if you already have the solution."
    )

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant that solves problems step by step.

Use the following format:

Thought: Think about what you need to do next
Action: tool_name(arg1="value1", arg2="value2")
Observation: [Result from the tool - this will be provided to you]

... (repeat Thought/Action/Observation as needed)

Thought: I now have the final answer
Final Answer: <your final answer>

Available tools:
{tools_description}

Always think before acting, and provide a Final Answer when you're done."""

    def __init__(
        self,
        tool_registry: NativeToolRegistry,
        system_prompt: str | None = None,
        thought_reward: float = 0.02,
        no_thought_penalty: float = 0.05,
        answer_validator: Callable[[Any, Any], bool] | None = None,
        tool_budgets: dict[str, int] | None = None,
        **kwargs,
    ):
        """``answer_validator`` replaces the default check; ``tool_budgets`` caps calls per tool per episode (a call
        past its cap is refused unrun)."""
        require_callable(type(self).__name__, **present(answer_validator=answer_validator))
        # Only the default check makes the answer column load-bearing. ``None`` is unset, so a YAML
        # ``requires_answer: null`` still lands on the class declaration.
        if kwargs.get("requires_answer") is None:
            kwargs["requires_answer"] = answer_validator is None
        super().__init__(**kwargs)

        require_finite(
            type(self).__name__,
            ge=0,
            why=MAGNITUDE_WHY,
            thought_reward=thought_reward,
            no_thought_penalty=no_thought_penalty,
        )

        self.registry = tool_registry
        self.tool_budgets = validate_tool_budgets(tool_budgets, tool_registry)
        self.thought_reward = thought_reward
        self.no_thought_penalty = no_thought_penalty
        self.answer_validator = answer_validator

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            tools_desc = self._format_tools_description()
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT.format(tools_description=tools_desc)

    def _format_tools_description(self) -> str:
        lines = []
        for tool in self.registry.list_tools():
            params = ", ".join(
                f"{p.name}: {p.type}" + (" (optional)" if not p.required else "") for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=self.system_prompt,
            extra_info={"final_answer": None, **initial_tool_counters(self.tool_budgets)},
        )

    def _step_single(self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None) -> StepOutcome:
        # A cut turn is a fragment and the base flags it untrainable, so an Action executed or a Final Answer graded
        # here would earn a reward on a turn the trainer never rewards.
        ctx = context or {}
        last_turn = bool(ctx.get(LAST_TURN_KEY))
        if ctx.get("finish_reason") in ENGINE_CUT_FINISH_REASONS:
            return self._recover_length_cutoff(trajectory, last_turn=last_turn)
        # Nothing to parse: not a format failure the hint below corrects, but a stop on nothing.
        if not action.strip():
            return self._recover_empty_turn(trajectory, last_turn=last_turn)

        step = parse_react_output(action)
        reward = 0.0
        if step.thought:
            reward += self.thought_reward
        elif step.has_action or step.has_final_answer:
            reward -= self.no_thought_penalty

        if step.has_final_answer:
            trajectory.info[COMPLETED_KEY] = True
            trajectory.info["final_answer"] = step.final_answer
            return StepOutcome(trajectory=trajectory, reward=reward, done=True, truncated=False, info={})
        if step.has_action:
            delta, info = self._act(trajectory, step)
            return StepOutcome(trajectory=trajectory, reward=reward + delta, done=False, truncated=False, info=info)

        hint = (
            "Please provide either:\n"
            "- An Action using one of the available tools, or\n"
            "- A Final Answer if you have the solution.\n\n"
            "Remember to include a Thought before your action."
        )
        trajectory.add_message(Message.user(hint))
        return StepOutcome(trajectory=trajectory, reward=reward, done=False, truncated=False, info={"no_action": True})

    def _act(self, trajectory: Trajectory, step: _ReActStep) -> tuple[float, dict[str, Any]]:
        """Returns the call's reward delta and the step info."""
        info = {}
        tool = self.registry.get(step.action)
        success = False
        fault = None
        if not tool:
            observation = self.registry.unknown_tool_message(step.action)
            info["tool_error"] = f"Unknown tool: {step.action}"
            # Trained only on a negative advantage, never rewarded, as under the native protocol.
            trajectory.last_assistant_message().calls_rejected = True
        else:
            try:
                args = self._admit_tool_call(tool, step.action_args or {}, trajectory)
                outcome = tool.execute(**args)
            except Exception as e:  # a tool fault is an observation, not an episode kill
                outcome = e
            observation, success, fault = tool_call_outcome(step.action, outcome)
            if not success:
                info["tool_error"] = str(outcome)
            if is_refused_unrun(outcome):
                trajectory.last_assistant_message().calls_rejected = True

        reward = self._book_tool_call(trajectory, step.action, success, fault)
        trajectory.add_message(Message.user(f"Observation: {self._truncate_observation(observation)}"))
        return reward, info

    def _final_answer(self, trajectory: Trajectory) -> str | None:
        return trajectory.info.get("final_answer") if trajectory.info.get(COMPLETED_KEY) else None

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        if not trajectory.info.get(COMPLETED_KEY):
            return EpisodeGrade(0.0)

        final_answer = trajectory.info.get("final_answer")
        ctx = context or trajectory.info.get(CONTEXT_KEY) or {}
        expected = ctx.get(ANSWER_KEY)

        if final_answer is None:
            return EpisodeGrade(0.0)

        # The validator is the grader wherever configured, so an answer-less row is its verdict to give.
        if self.answer_validator is not None:
            try:
                validated = self.answer_validator(final_answer, expected)
            except Exception as exc:
                # The grader failed, not the policy: no check stands in for it (the default one pays any
                # Final Answer on a row with no expected answer), so the episode leaves the baseline.
                logger.warning("answer_validator raised; scoring the episode invalid", exc_info=True)
                return self._invalid_grade(trajectory, f"answer_validator raised {type(exc).__name__}: {exc}")
            return EpisodeGrade(1.0 if validated else 0.0)

        if expected is None:
            # Presence, not value: a null ``answer`` cell is a data fault, an absent key an ungraded row.
            if ANSWER_KEY in ctx:
                return self._null_answer_grade(trajectory)
            # Reaching a Final Answer is the objective. ``requires_answer`` keeps an answer-graded run off this path.
            return EpisodeGrade(1.0)

        return EpisodeGrade(1.0 if is_correct_answer(final_answer, expected) else 0.0)


def _reject_preset_system_prompt(env_type: str, system_prompt: str | None) -> None:
    """The preset hardcodes the prompt stating the format its parser reads, so a configured value would be dropped."""
    if system_prompt is not None:
        raise ValueError(
            f"environment_type {env_type!r} hardcodes its ReAct system prompt, so environment_kwargs.system_prompt "
            f"would be ignored. Remove it, or pick an environment_type whose class takes a system_prompt "
            f"(e.g. native_math)."
        )


def build_react_math_environment(system_prompt: str | None = None, **kwargs) -> ReActEnvironment:
    _reject_preset_system_prompt("react_math", system_prompt)
    registry = NativeToolRegistry.combine(
        build_math_tools(),
        build_python_tools(),
    )

    system_prompt = """You are a math problem solver. Think step by step and use tools to calculate.

Format your response as:
Thought: <your reasoning about what to do next>
Action: calculate(expression="<math expression>") OR python(code="<python code>")
Observation: <you will see the result here>

When you have the final answer:
Thought: <summarize your solution>
Final Answer: <numeric answer>

Always show your reasoning in the Thought section."""

    return ReActEnvironment(
        tool_registry=registry,
        system_prompt=system_prompt,
        **kwargs,
    )


def build_react_search_environment(
    search_backend: str | None = None, system_prompt: str | None = None, **kwargs
) -> ReActEnvironment:
    """``search_backend``: see :func:`build_search_tools`; ``None`` auto-selects."""
    _reject_preset_system_prompt("react_search", system_prompt)
    registry = build_search_tools(backend=search_backend)

    system_prompt = """You are a research assistant that finds information to answer questions.

Format your response as:
Thought: <what do you need to find out>
Action: web_search(query="<search query>")
Observation: <search results>

When you have enough information:
Thought: <synthesize your findings>
Final Answer: <comprehensive answer>

Be thorough but concise in your final answer."""

    return ReActEnvironment(
        tool_registry=registry,
        system_prompt=system_prompt,
        **kwargs,
    )
