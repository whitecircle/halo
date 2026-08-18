# Custom Environments

Override three abstract methods on `BaseEnvironment` (`src/environments/base.py`):

```python
from src.environments.base import BaseEnvironment, Trajectory, Message

class CustomEnvironment(BaseEnvironment):
    def __init__(self, success_reward=1.0, **kwargs):
        super().__init__(**kwargs)
        self.success_reward = success_reward

    def _reset_single(self, prompt, context=None):
        """Initialize a single episode."""
        traj = Trajectory()
        traj.add_message(Message.user(str(prompt)))
        if context and "answer" in context:
            traj.info["expected"] = context["answer"]
        return traj

    def _step_single(self, trajectory, action, context=None):
        """Returns (trajectory, reward, done, truncated, info)."""
        done = "Final Answer:" in action
        return trajectory, 0.0, done, False, {}

    def _compute_reward(self, trajectory, context=None):
        expected = trajectory.info.get("expected", "")
        if expected in str(trajectory.messages[-1].content):
            return self.success_reward
        return 0.0
```

Register it so a training script can resolve it by name, then set `environment_type` in the YAML and launch `scripts/training/environmental_grpo.py` as usual:

```python
# my_envs.py (imported before training, e.g. via the registry module or a small launcher)
from src.environments.registry import register_environment

register_environment("my_env", lambda env_config: CustomEnvironment(**env_config))
```

Forward the whole `env_config`, as the built-in factories do: a factory that ignores it drops every YAML `environment_kwargs` key silently, and the strict-kwargs check below never runs because the keys never reach the constructor. Nothing is defaulted in the factory either — a turn budget that differs from the base 10 is declared on the class (`DEFAULT_MAX_TURNS`, and `DEFAULT_TOOL_SUCCESS_REWARD` / `DEFAULT_TOOL_ERROR_PENALTY` for the native tool protocol), so an explicit `max_turns` in the YAML still wins and the default is stated once. The registry's own `_without` helper exists only for a factory that binds a key itself and would otherwise raise on a duplicate keyword.

For fully programmatic use, `DistributedAsyncEnvironmentalGRPOTrainer` also accepts `environment_cls=CustomEnvironment` + `environment_kwargs=...` directly.

For I/O-bound steps (tool servers, HTTP APIs), subclass `AsyncBaseEnvironment` and override `_step_single_async` / `_reset_single_async` instead. The rollout driver detects an `AsyncBaseEnvironment` instance and drives it through `reset_async`/`step_async`.

For per-episode state, a `NativeToolUseEnvironment` subclass overrides two hooks instead of the execution or cleanup paths themselves: `_episode_binding(trajectory)` (a context manager entered around every tool batch, sync and async — set the episode's `ContextVar`s there so concurrent rollouts on one instance never share them) and `_release_episode(episode_id)` (called by `cleanup()` as it drops the episode — close its session or connection there).

A sync env with an async-only tool handler does *not* fail at step time: `NativeTool.execute` raises `NotImplementedError`, `_execute_tool_calls` catches it as an ordinary tool error, and every call lands as a failed tool result charged `tool_error_penalty`. The episode completes and the rewards are quietly wrong.

To surface task-specific diagnostics in the training logs, override `rollout_metrics(self, trajectory) -> dict[str, float]`. It runs per episode in the Ray actor (so it may read `trajectory.info`) and returns metrics keyed by their full path — `outcome/*` (task success), `episode/*` (behavior), `reward/*` (reward decomposition); the trainer mean-aggregates them.

The base emits `episode/tool_calls` (where the env counts tool calls) and always `episode/length_cutoff_turns`. `CodeContestsEnvironment` adds `outcome/solve_rate`, `outcome/test_pass_frac`, `episode/submission_rate`, `episode/test_calls`, `episode/tested_before_submission`, `episode/grading_infra_outage`, and the reward components. See [Logged metrics](../environmental-grpo.md#logged-metrics).

## Trajectory shape the trainer can tokenize

By default (`train_on_sampled_tokens`, on) the trainer builds **one row per assistant turn** from the engine's own sampled ids, taking the engine's `prompt_token_ids` as each row's prompt.

The re-tokenization fallback — `train_on_sampled_tokens: false`, or a trajectory whose sampled ids were not captured — instead renders the finished trajectory **once** through the serving chat template and locates each assistant turn's span inside that single render, because chat templates are not prefix-monotone and independently rendered per-turn prefixes cannot be diffed ([details](../environmental-grpo.md#the-re-tokenization-fallback-_tokenize_trajectory)). On that path a trajectory the template cannot decompose is recorded and raised on **every** rank, never trained on a guessed span.

Four rules keep an environment decomposable either way:

- **Let the base append assistant turns.** `_add_action_message` is the sole carrier of the engine's sampled ids, logprobs, routing mask and reasoning. Hand-appending an assistant message loses them, and two consecutive assistant messages have no locatable boundary.
- **Every advertised tool call needs a tool-result message.** Declare `max_tool_calls_per_turn` when the env executes fewer calls than the model may request — the base truncates the turn's `tool_calls` to it so the two counts match. `BaseEnvironment` declares it as a class attribute (default `None`, no truncation); the **native tool-use** envs (`NativeToolUseEnvironment` and its `mcp`/`swe` subclasses, default `5`) additionally take it as a constructor kwarg, so those are the ones an `environment_kwargs` entry can set. The ReAct protocol executes one parsed action per turn and takes no such kwarg — passing it there is refused by the base's unknown-kwarg check.
- **Return tool output as a message** — `Message.tool(content, tool_call_id, name)`, or a plain user turn (`Observation: …`) for a ReAct-style protocol. The loss mask comes from spans, not roles, so either shape trains correctly.
- **`tool_calls` reach the template verbatim.** An `arguments` value the template cannot handle (a JSON string where it expects a mapping) fails the render.

Constructor kwargs are strict: every `environment_kwargs` key must be a parameter of the resolved environment class, or construction raises.

## Registering custom tools

Use `NativeToolRegistry` for tools used with `NativeToolUseEnvironment`:

```python
from src.environments.tools.definitions import NativeToolRegistry, NativeTool, ToolParameter

registry = NativeToolRegistry()
registry.register(NativeTool(
    name="my_tool",
    description="Does something useful",
    parameters=[
        ToolParameter(name="param1", type="string", description="First parameter"),
        ToolParameter(name="param2", type="integer", description="Second parameter", required=False),
    ],
    handler=my_tool_function,  # called with keyword args: my_tool_function(param1="value", param2=42)
))
```

Parsing and serialization of OpenAI-format tool calls live on the data model (`src/environments/tools/definitions.py`): `NativeToolCall.from_openai_format(tool_call)` parses one call off the wire, and `NativeToolResult.to_message()` turns a result into the `tool` message the next turn conditions on. One path each — a second parser or serializer drifts from the one the rollout actually uses. The environment supplies tool schemas via `registry.to_openai_tools()`; choosing the server-side `--tool-call-parser` is the inference server's job.

A handler that raises marks the call failed (the episode pays `tool_error_penalty`), so raise for an infrastructure fault and return a string for a legitimate negative answer.

## Dataset format

```python
{
    "prompt": str | list[dict],  # task prompt or conversation
    "answer": Any,               # expected answer for reward computation
    ...                          # extra columns, each declared in context_fields
}
```

`prompt` is either a string (`{"prompt": "What is 25 * 4 + 10?", "answer": 110}`) or a conversation (`[{"role": "system", ...}, {"role": "user", ...}]`).

`answer` plus every column named in the script's `context_fields` reach the environment as the `context` parameter of `reset()` / `step()`. All other columns are **dropped** during preprocessing, and naming a column that isn't in the dataset raises at startup — to use `difficulty` and `category`, declare them: `context_fields: [difficulty, category]`.

A list-valued `prompt` survives whole only in the eval scripts. The trainer flattens it to the **last user message's content** (`_extract_prompts_and_contexts`), so a system message in the row is dropped — carry it as the environment's `system_prompt` instead.

| Environment | `answer` format | Example |
|-------------|----------------|---------|
| `react_math`, `native_math` | String or number | `"42"`, `"3.14"` |
| `qa_search` | Expected answer string | `"Sinclair Lewis"` |
| `exam_qa` (MC) | Choice letter, or a 0-based index into the `choices` column | `"B"`, `1` |
| `exam_qa` (open) | Answer string | `"Paris"` |
| `code_contests` / `codeforces` | JSON with test cases (+ optional `checker`, `time_limit`) | `'{"test_cases": [{"input": "5", "output": "10"}]}'` — see [Code Contests](code-contests.md#dataset-format) |

### Public datasets

| Task | Datasets | Schema notes |
|------|----------|--------------|
| Math (`react_math`, `native_math`) | `openai/gsm8k`, `lighteval/MATH`, `meta-math/MetaMathQA` | GSM8K answers arrive as `#### N`; MATH uses `\boxed{}` |
| Factual QA (`qa_search`) | `basicv8vc/SimpleQA`, `trivia_qa` (rc.nocontext), `gaia-benchmark/GAIA` | SimpleQA uses `problem`/`answer`; TriviaQA carries multiple valid answers in `answer.aliases` |
| Multiple choice (`exam_qa`) | `TIGER-Lab/MMLU-Pro`, `cais/mmlu`, `Idavidrein/gpqa`, `allenai/ai2_arc` (Challenge) | MMLU-Pro has 10 choices (A–J); MMLU's `answer` is an int index, converted to its letter at reset |
| Competitive programming | see [Code Contests](code-contests.md#dataset-format) | adapters and preparation live there |
