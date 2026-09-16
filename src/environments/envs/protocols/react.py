"""ReAct environment: Thought -> Action -> Observation, action parsed from text.

Paper: https://arxiv.org/abs/2210.03629
"""

import contextlib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.environments.base import (
    EPISODE_INVALID_KEY,
    EPISODE_TOOL_BUDGETS_KEY,
    TOOL_CALL_COUNTS_KEY,
    BaseEnvironment,
    EpisodeGrade,
    Message,
    Trajectory,
    require_magnitudes,
)
from src.environments.envs.protocols.native import validate_tool_budgets
from src.environments.tools.definitions import NativeToolRegistry, ToolArgumentError, ToolBudgetExhausted
from src.environments.tools.factories import (
    create_native_math_tools,
    create_native_python_tools,
    create_native_search_tools,
)
from src.inference.response import ENGINE_CUT_FINISH_REASONS
from src.rewards.matching import validate_answer

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
class ReActStep:
    """Parsed ReAct step from model output."""

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


def parse_react_output(text: str) -> ReActStep:
    """Parse ReAct output: a Thought followed by an Action (function-call/JSON/simple) or a Final Answer."""
    step = ReActStep()

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


def _parse_action(action_text: str) -> tuple[str | None, dict[str, Any] | None]:
    """Parse an action: ``tool_name(arg=...)``, JSON ``{"name":..., "arguments":...}``, ``tool_name: arg=...``, or bare name."""
    action_text = action_text.strip()

    if action_text.startswith("{"):
        try:
            data = json.loads(action_text)
            # Model-authored JSON: a non-dict ``function`` or non-dict ``arguments`` must degrade to an
            # unknown-tool / empty-args call the protocol penalizes, not an AttributeError/TypeError
            # that escapes the parser and errors the whole episode.
            function = data.get("function")
            function = function if isinstance(function, dict) else {}
            name = data.get("name") or function.get("name")
            args = data.get("arguments") or function.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            return (name if isinstance(name, str) else None), (args if isinstance(args, dict) else {})
        except json.JSONDecodeError:
            pass

    func_match = _CALL_ACTION_RE.match(action_text)
    if func_match:
        name = func_match.group(1)
        args_str = func_match.group(2).strip()
        args = _parse_function_args(args_str)
        return name, args

    simple_match = _COLON_ACTION_RE.match(action_text)
    if simple_match:
        name = simple_match.group(1)
        args_str = simple_match.group(2).strip()
        args = _parse_function_args(args_str)
        return name, args

    name_match = _BARE_ACTION_RE.match(action_text)
    if name_match:
        return name_match.group(1), {}

    return None, None


def _parse_function_args(args_str: str) -> dict[str, Any]:
    """Parse function-style arguments: arg1="val1", arg2=123"""
    args = {}
    if not args_str:
        return args

    for match in _ARGUMENT_RE.finditer(args_str):
        key = match.group(1)
        # First MATCHED alternative by ``is not None``: truthiness would turn ``expression=""`` into None.
        value = next((g for g in match.groups()[1:] if g is not None), None)

        if value and (value.startswith("{") or value.startswith("[")):
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(value)
        elif value:
            with contextlib.suppress(ValueError, TypeError):
                value = float(value) if "." in value else int(value)

        args[key] = value

    return args


class ReActEnvironment(BaseEnvironment):
    """ReAct environment: Thought / Action / Observation turns, action parsed from text.

    ``require_thought`` penalizes acting without a Thought. The action is read out of the assistant
    text, so no tool schema is advertised to the server and no server-side tool-call parser is
    involved — the tools are named in the system prompt.
    """

    # The default check grades the Final Answer against ``context["answer"]``; an ``answer_validator``
    # grades in its place and clears this per instance.
    requires_answer = True

    # Asks for the protocol's own next move (an Action or a Final Answer) and never for shorter
    # reasoning: the text is trained on wherever a recovery succeeds, so an instruction here becomes a
    # global lesson learned far outside the situation it was written for.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you produced an Action or a Final Answer, so nothing "
        "was recorded. Give your next Action now, or your Final Answer if you already have the "
        "solution."
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
        require_thought: bool = True,
        answer_validator: Callable[[Any, Any], bool] | None = None,
        tool_budgets: dict[str, int] | None = None,
        **kwargs,
    ):
        """Reward = per-step thought/tool deltas + the graded final answer (the objective term).

        Penalty knobs must be magnitudes (>= 0); minus is applied at the use site so a positive config
        value cannot farm the penalty as a bonus. The per-call knobs (``tool_success_reward``,
        ``tool_error_penalty``, ``tool_reward_cap``) are the base's. ``answer_validator`` overrides the
        default check. ``tool_budgets`` caps calls per tool per episode (``{tool_name: cap}``); a call
        past its cap is refused as a tool error and never runs.
        """
        # An answer_validator grades the final answer itself, so only the default check makes the
        # dataset's answer column load-bearing. An explicit requires_answer wins; ``None`` is unset,
        # so a YAML ``requires_answer: null`` still lands on the class declaration.
        if kwargs.get("requires_answer") is None:
            kwargs["requires_answer"] = not callable(answer_validator)
        super().__init__(**kwargs)

        require_magnitudes(thought_reward=thought_reward, no_thought_penalty=no_thought_penalty)

        self.registry = tool_registry
        self.tool_budgets = validate_tool_budgets(tool_budgets, tool_registry)
        self.thought_reward = thought_reward
        self.no_thought_penalty = no_thought_penalty
        self.require_thought = require_thought
        self.answer_validator = answer_validator

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            tools_desc = self._format_tools_description()
            self.system_prompt = self.DEFAULT_SYSTEM_PROMPT.format(tools_description=tools_desc)

    def _format_tools_description(self) -> str:
        """Format tool descriptions for system prompt."""
        lines = []
        for tool in self.registry.list_tools():
            params = ", ".join(
                f"{p.name}: {p.type}" + (" (optional)" if not p.required else "") for p in tool.parameters
            )
            lines.append(f"- {tool.name}({params}): {tool.description}")
        return "\n".join(lines)

    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        """Initialize episode with task prompt."""
        context = context or {}
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=self.system_prompt,
            extra_info={
                "expected_answer": context.get("answer"),
                # Presence, not value: a row whose ``answer`` cell is null is a data fault, an absent
                # key an ungraded episode, and the reward pays them differently. Read off the RESET
                # context, the only one that carries the row (a lost episode is graded with none).
                "_answer_in_context": "answer" in context,
                "thoughts": [],
                "actions": [],
                "observations": [],
                "final_answer": None,
                "total_tool_calls": 0,
                "successful_tool_calls": 0,
                "total_thoughts": 0,
                TOOL_CALL_COUNTS_KEY: {},
                EPISODE_TOOL_BUDGETS_KEY: dict(self.tool_budgets),
            },
        )

    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Process model output in ReAct format (Thought: <reasoning>; Action: <tool_call> OR Final Answer: <answer>)."""
        reward = 0.0
        info = {}

        # A turn the engine cut short is a fragment whatever the parser would salvage from it — the
        # base flags the message untrainable, so an Action executed or a Final Answer graded here would
        # earn a reward on a turn the trainer then excludes. Same rule as the native protocol.
        if (context or {}).get("finish_reason") in ENGINE_CUT_FINISH_REASONS:
            return self._handle_length_cutoff(trajectory)

        step = parse_react_output(action)

        if step.thought:
            trajectory.info["thoughts"].append(step.thought)
            trajectory.info["total_thoughts"] += 1
            reward += self.thought_reward
            info["has_thought"] = True
        elif self.require_thought and (step.has_action or step.has_final_answer):
            reward -= self.no_thought_penalty
            info["missing_thought"] = True

        if step.has_final_answer:
            trajectory.info["completed"] = True
            trajectory.info["final_answer"] = step.final_answer

            # Assistant message already appended by BaseEnvironment.step; re-adding doubles the turn.
            info["final_answer"] = step.final_answer
            return trajectory, reward, True, False, info

        if step.has_action:
            tool = self.registry.get(step.action)
            success = False

            if not tool:
                observation = self.registry.unknown_tool_message(step.action)
                info["tool_error"] = f"Unknown tool: {step.action}"
                # The turn accomplished nothing: skipped by the trainer, as under the native protocol.
                self._flag_calls_rejected(trajectory)
            else:
                try:
                    # Bind before spending the episode's budget: a call the handler cannot run is
                    # refused without being counted, as under the native protocol.
                    args = tool.bind(step.action_args or {})
                    cap = self._tool_budget_exhausted(trajectory, step.action)
                    if cap is not None:
                        raise ToolBudgetExhausted(tool.budget_exhausted_message(cap))
                    self._count_tool_call(trajectory, step.action)
                    observation = tool.execute(**args)
                    success = True
                    info["tool_success"] = True
                except (ToolBudgetExhausted, ToolArgumentError) as e:
                    # A refusal is expected control flow: charged like any tool error, logged without
                    # the traceback that a tool which actually broke gets below.
                    logger.debug("Tool %r refused the call: %s", step.action, e)
                    observation = f"Error: {e}"
                    info["tool_error"] = str(e)
                except Exception as e:
                    # Without this line the episode just grades 0 with nothing anywhere
                    # saying why: the observation carries the message, but the trajectory is not where
                    # a broken tool gets debugged.
                    logger.warning("Tool %r raised during execution", step.action, exc_info=True)
                    observation = f"Error: {str(e)}"
                    info["tool_error"] = str(e)

            reward += self._credit_tool_call(trajectory, success)
            observation = self._truncate_observation(observation)
            trajectory.info["actions"].append(
                {
                    "tool": step.action,
                    "args": step.action_args,
                }
            )
            trajectory.info["observations"].append(observation)

            observation_msg = f"Observation: {observation}"
            trajectory.add_message(Message.user(observation_msg))

            info["observation"] = observation
            return trajectory, reward, False, False, info

        hint = (
            "Please provide either:\n"
            "- An Action using one of the available tools, or\n"
            "- A Final Answer if you have the solution.\n\n"
            "Remember to include a Thought before your action."
        )
        trajectory.add_message(Message.user(hint))

        info["no_action"] = True
        return trajectory, reward, False, False, info

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """Grade the final answer: 1 when it validates against the expected one, else 0."""
        if not trajectory.info.get("completed"):
            return EpisodeGrade(0.0)

        final_answer = trajectory.info.get("final_answer")
        expected = trajectory.info.get("expected_answer")

        if final_answer is None:
            return EpisodeGrade(0.0)

        # Consulted before the expected answer is read: the validator is the grader wherever one is
        # configured, so an answer-less row is ITS verdict to give, not an automatic success.
        if callable(self.answer_validator):
            try:
                validated = self.answer_validator(final_answer, expected)
            except Exception:
                # Unwarned, an always-raising validator silently re-grades every episode by default.
                logger.warning("answer_validator raised; falling back to the default check", exc_info=True)
            else:
                return EpisodeGrade(1.0 if validated else 0.0)

        if expected is None:
            if trajectory.info.get("_answer_in_context"):
                # The row IS answer-graded and its cell is null: nothing was verified, so the
                # completion payout below would hand the full objective to any episode that answered
                # — and to its whole group, since every sibling answers just as easily. Drop it from
                # the baseline instead, the same contract the native protocol holds.
                logger.warning("Episode context carries a null 'answer'; scoring it invalid, not a success")
                trajectory.info[EPISODE_INVALID_KEY] = True
                return EpisodeGrade(0.0)
            # Nothing to grade against: reaching a Final Answer is the objective. ``requires_answer``
            # keeps an answer-graded run off this path rather than paying it the full objective.
            return EpisodeGrade(1.0)

        return EpisodeGrade(1.0 if validate_answer(final_answer, expected) else 0.0)


def create_react_math_environment(**kwargs) -> ReActEnvironment:
    """Create a ReAct environment for math problems (calculator + Python REPL tools)."""
    registry = NativeToolRegistry.combine(
        create_native_math_tools(),
        create_native_python_tools(),
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


def create_react_search_environment(**kwargs) -> ReActEnvironment:
    """Create a ReAct environment for search/QA tasks (web search + basic tools)."""
    registry = create_native_search_tools()

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
