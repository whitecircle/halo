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

from src.environments.base import BaseEnvironment, Message, Trajectory, require_magnitudes
from src.environments.rewards import compute_answer_reward
from src.environments.tools.definitions import NativeToolRegistry
from src.environments.tools.factories import (
    create_native_math_tools,
    create_native_python_tools,
    create_native_search_tools,
)
from src.inference.response import FINISH_REASON_LENGTH

logger = logging.getLogger(__name__)


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

    thought_match = re.search(
        r"(?:^|\n)\s*(?:Thought|THOUGHT|Think|THINK|Reasoning|REASONING)\s*:\s*(.+?)(?=\n\s*(?:Action|ACTION|Final|FINAL)|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    if thought_match:
        step.thought = thought_match.group(1).strip()

    # Final answer wins over an action if both are present.
    final_match = re.search(
        r"(?:^|\n)\s*(?:Final Answer|FINAL ANSWER|Answer|ANSWER)\s*:\s*(.+?)$", text, re.IGNORECASE | re.DOTALL
    )
    if final_match:
        step.final_answer = final_match.group(1).strip()
        return step

    action_match = re.search(
        r"(?:^|\n)\s*(?:Action|ACTION)\s*:\s*(.+?)(?=\n\s*(?:Observation|OBSERVATION)|$)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
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

    func_match = re.match(r"^(\w+)\s*\((.*)\)$", action_text, re.DOTALL)
    if func_match:
        name = func_match.group(1)
        args_str = func_match.group(2).strip()
        args = _parse_function_args(args_str)
        return name, args

    simple_match = re.match(r"^(\w+)\s*:\s*(.*)$", action_text, re.DOTALL)
    if simple_match:
        name = simple_match.group(1)
        args_str = simple_match.group(2).strip()
        args = _parse_function_args(args_str)
        return name, args

    name_match = re.match(r"^(\w+)$", action_text)
    if name_match:
        return name_match.group(1), {}

    return None, None


def _parse_function_args(args_str: str) -> dict[str, Any]:
    """Parse function-style arguments: arg1="val1", arg2=123"""
    args = {}
    if not args_str:
        return args

    pattern = r'(\w+)\s*=\s*(?:"([^"]*?)"|\'([^\']*?)\'|(\{[^}]*\})|(\[[^\]]*\])|([^,\s]+))'
    for match in re.finditer(pattern, args_str):
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

    # Asks for the protocol's own next move (an Action or a Final Answer) and never for shorter
    # reasoning: the text is trained on wherever a recovery succeeds, so an instruction here becomes a
    # global lesson learned far outside the situation it was written for.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off at its length limit before you produced an Action or a Final "
        "Answer, so nothing was recorded. Give your next Action now, or your Final Answer if you "
        "already have the solution."
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
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        tool_success_reward: float = 0.05,
        tool_error_penalty: float = 0.1,
        thought_reward: float = 0.02,
        no_thought_penalty: float = 0.05,
        require_thought: bool = True,
        answer_validator: Callable[[Any, Any], bool] | None = None,
        **kwargs,
    ):
        """Reward = per-step thought/tool deltas + terminal answer reward.

        Penalty knobs must be magnitudes (>= 0); minus is applied at the use site so a positive config
        value cannot farm the penalty as a bonus. ``answer_validator`` overrides the default check.
        """
        super().__init__(**kwargs)

        require_magnitudes(
            tool_success_reward=tool_success_reward,
            tool_error_penalty=tool_error_penalty,
            thought_reward=thought_reward,
            no_thought_penalty=no_thought_penalty,
        )

        self.registry = tool_registry
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.tool_success_reward = tool_success_reward
        self.tool_error_penalty = tool_error_penalty
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
                "thoughts": [],
                "actions": [],
                "observations": [],
                "final_answer": None,
                "total_tool_calls": 0,
                "successful_tool_calls": 0,
                "total_thoughts": 0,
            },
        )

    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Process model output in ReAct format (Thought: <reasoning>; Action: <tool_call> OR Final Answer: <answer>)."""
        reward = 0.0
        info = {}

        step = parse_react_output(action)

        # A turn the engine cut off before either terminator: a failed turn, not a formatting failure.
        # Checked after the parse, so a turn that got its Action or Final Answer out before the cap
        # takes the normal path, and unpriced in full — the thought bonus and the missing-thought
        # penalty both grade a turn the model never finished.
        if (
            not (step.has_action or step.has_final_answer)
            and (context or {}).get("finish_reason") == FINISH_REASON_LENGTH
        ):
            return self._handle_length_cutoff(trajectory)

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

            if not tool:
                observation = self.registry.unknown_tool_message(step.action)
                reward -= self.tool_error_penalty
                info["tool_error"] = f"Unknown tool: {step.action}"
            else:
                try:
                    args = step.action_args or {}
                    result = tool.execute(**args)
                    observation = result
                    reward += self.tool_success_reward
                    trajectory.info["successful_tool_calls"] += 1
                    info["tool_success"] = True
                except Exception as e:
                    # Without this line the episode just scores failure_reward with nothing anywhere
                    # saying why: the observation carries the message, but the trajectory is not where
                    # a broken tool gets debugged.
                    logger.warning("Tool %r raised during execution", step.action, exc_info=True)
                    observation = f"Error: {str(e)}"
                    reward -= self.tool_error_penalty
                    info["tool_error"] = str(e)

            observation = self._truncate_observation(observation)
            trajectory.info["total_tool_calls"] += 1
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

    def _compute_reward(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> float:
        """Compute final reward based on answer correctness."""
        base_reward = trajectory.total_reward

        if not trajectory.info.get("completed"):
            return base_reward + self.failure_reward

        final_answer = trajectory.info.get("final_answer")
        expected = trajectory.info.get("expected_answer")

        if expected is None:
            return base_reward + self.success_reward

        if final_answer is None:
            return base_reward + self.failure_reward

        if self.answer_validator:
            try:
                if self.answer_validator(final_answer, expected):
                    return base_reward + self.success_reward
                else:
                    return base_reward + self.failure_reward
            except Exception:
                # Unwarned, an always-raising validator silently re-grades every episode by default.
                logger.warning("answer_validator raised; falling back to the default check", exc_info=True)

        answer_reward = compute_answer_reward(
            predicted=final_answer,
            expected=expected,
            success_reward=self.success_reward,
            failure_reward=self.failure_reward,
        )
        return base_reward + answer_reward


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
