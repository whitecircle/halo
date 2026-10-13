# Custom Environments

Subclass `BaseEnvironment` (`src/environments/base.py`) and implement three methods, then register a factory for YAML or pass the class from Python.

```python
from src.environments.base import BaseEnvironment, EpisodeGrade

class GuessEnvironment(BaseEnvironment):
    DEFAULT_MAX_TURNS = 6
    requires_answer = True          # the grade reads context["answer"]

    def _reset_single(self, prompt, context=None):
        expected = (context or {}).get("answer")
        return self._init_trajectory(prompt, context, extra_info={"expected": expected})

    def _step_single(self, trajectory, action, context=None):
        done = "Final Answer:" in action
        trajectory.info["completed"] = done
        return trajectory, 0.0, done, False, {}

    def _grade_episode(self, trajectory, context=None):
        expected = trajectory.info.get("expected")
        answered = expected is not None and str(expected) in trajectory.messages[-1].content
        return EpisodeGrade(1.0 if answered else 0.0)
```

Rules that bite:

- **Grade, do not price.** `_grade_episode` returns an `EpisodeGrade`; a class that defines `_compute_reward` is refused at construction ([Environment arm](../../grpo/rewards/README.md#environment-arm)).
    - `objective` is the task grade in `[0, 1]`, priced by the reward's `environment` term (`reward/objective`).
    - `shaping` maps episode-level terms to values (`{"submission": 0.25}`), each added as `reward/<name>`. Declare the names in `SHAPING_COMPONENTS`, or the first [settled](../../grpo/rewards/README.md#settlement) episode fails. `objective` and `turn_shaping` are reserved, and no reward term may reuse a declared name.
    - Protocol-level shaping goes in `_episode_shaping(trajectory) -> dict`.
    - An episode whose grade carries no signal (a grader outage, a null `answer` cell) sets `trajectory.info[EPISODE_INVALID_KEY] = True`, its cause under `EPISODE_INVALID_REASON_KEY`, and grades 0, so it leaves the group baseline.
- **Hand a scorer what to read.** Override `_final_answer` and `_scoring_reference` for `judge` and `reward_model` terms ([What a scorer reads](../../grpo/rewards/README.md#what-a-scorer-reads)).
- **Forward `**kwargs` to `super().__init__`.** The base binds `rewards` (as `reward_terms`), `max_turns` and the [common knobs](README.md#common-knobs); a keyword no constructor binds raises `TypeError` at construction.
- **Declare differing defaults on the class** (`DEFAULT_MAX_TURNS`, `DEFAULT_TOOL_SUCCESS_REWARD`, `DEFAULT_TOOL_ERROR_PENALTY`, `requires_answer`), not in the factory, so a YAML `max_turns` still wins.
- **Build through `self._init_trajectory(...)`**, so the system turn, the task turn and the context land where the base expects them.
- **Route an unproductive turn, never grade it** ([Unproductive turns](../episodes.md#unproductive-turns)).
    - An engine cut arrives with `context["finish_reason"] in ENGINE_CUT_FINISH_REASONS` (`src/inference/response.py`): call `self._handle_length_cutoff(trajectory)`. A turn with neither visible content nor a tool call: call `self._handle_empty_turn(trajectory)`. Both nudge and retry within `max_turns` and the shared `max_length_cutoff_recoveries` cap.
    - Declare `LENGTH_CUTOFF_NUDGE` and `EMPTY_TURN_NUDGE`, or the base raises `NotImplementedError` on the first such turn.
    - A token-cap cut's unfinished tool calls arrive under `context[CUT_TOOL_CALLS_KEY]`, not `context["tool_calls"]`; never execute them. Pass `in_tool_call=bool(context.get(CUT_TOOL_CALLS_KEY))` to `_handle_length_cutoff` so the turn counts under `episode/length_cutoff_in_call_turns` and gets `LENGTH_CUTOFF_IN_CALL_NUDGE` (the plain nudge if unset).
    - A native-protocol environment with no tools sends the `TOOLLESS_*` nudges, which ask for the final answer. A subclass with its own wording declares those too.
- **For I/O-bound steps** subclass `AsyncBaseEnvironment` and override `_reset_single_async` / `_step_single_async`; free per-episode resources in `_release_episode`, which `cleanup` calls after every episode, error and timeout paths included.
- **Probe your own backend.** Extend `verify_backend` and call `super()`, so a bad URL fails the launch instead of every episode.

## Register it

```python
from src.environments.registry import register_environment

register_environment("guess", lambda env_config: GuessEnvironment(**env_config))
```

- Forward the whole `env_config`: a factory that ignores it drops every `environment_kwargs` key silently.
- A name resolves against the factories registered in the process that builds the environment. A training run builds it inside each Ray actor, where the registry holds only what `src/environments/registry.py` registers at import: add yours there (`_register_builtins`), then set `environment_type: guess`. A registration made only in the entry script passes the startup check and then fails every rollout with `Unknown environment type`.
- The eval scripts build the environment in their own process, so a module their wrapper imports is enough ([Test it](#test-it)).
- From Python, skip the registry: `DistributedAsyncEnvironmentalGRPOTrainer(..., environment_cls=GuessEnvironment, environment_kwargs={...})`; the class itself travels to the actors.

## A custom tool

```python
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

registry = NativeToolRegistry().register(NativeTool(
    name="lookup",
    description="Look up a record by id",
    parameters=[ToolParameter(name="record_id", type="string", description="Record id")],
    handler=lookup_record,          # called as lookup_record(record_id="...")
))
```

Pass the registry as `NativeToolUseEnvironment(tool_registry=registry, ...)`.

- Calls are bound against the declared parameters before the handler runs; a refused call is a tool error that spends no tool budget ([Argument binding](protocols/native-tool-use.md#argument-binding)). A tool with no declared parameters takes its arguments as given.
- A handler that raises fails the call and charges `tool_error_penalty`. Return a string for a legitimate negative answer.
- A sandbox-backed handler raises its sandbox fault instead ([Sandbox faults](sandbox.md#sandbox-faults)).

## Trainable trajectory shape

The trainer builds one row per assistant turn from the engine's sampled ids ([Training rows](../objective.md#training-rows)). Four rules keep that possible:

- **Let `step()` append the assistant turn.** Only it carries the sampled ids, logprobs and reasoning. One hand-appended assistant message sends the whole trajectory to a re-tokenization fallback that trains text the policy never emitted.
- **Give every advertised tool call a result message.** Set `max_tool_calls_per_turn` when the environment executes fewer calls than the model may request.
- **Return tool output as a message**: `Message.tool(content, tool_call_id, name)`, or an `Observation: …` user turn for a text protocol. The loss mask comes from spans, not roles.
- **Keep `tool_calls` renderable.** A trajectory the chat template cannot render, or whose turn spans cannot be located, is dropped as a masked row, not trained on a guess.

To log per-episode diagnostics, override `rollout_metrics(trajectory) -> dict[str, float]` and merge `super().rollout_metrics(trajectory)`; keys log under their full paths (`outcome/*`, `episode/*`). A string stamped under a slice name in `trajectory.info["slices"]` (`{"language": "cpp"}`) splits the `episode/*` keys as `<slice>/<value>/*` ([Logged metrics](../monitoring.md#logged-metrics)).

## Dataset

```python
{"prompt": str | list[dict], "answer": Any, "difficulty": "hard"}
```

- `answer` and the columns named in the script's `context_fields` reach `reset()` / `step()` in `context`; other columns are dropped. A `context_field` (or a renamed `answer_field`) that names no column raises at startup.
- `requires_answer = True` makes the trainer refuse a dataset without `answer` (default `False`; per-run override in `environment_kwargs`).
- A list-valued prompt reaches the environment as its **last `user` message** only, so put framing in the environment's system prompt (`system_prompt` on the protocol classes; `_init_trajectory(system_prompt=...)` on your own).

## Test it

A CPU test, in the shape of `tests/cpu/environments/test_environments.py`:

```python
import pytest

from src.configs.environment_config import EnvironmentConfig
from src.environments.registry import register_environment, resolve_environment

def test_guess_env_grades_a_correct_answer(isolated_registry):
    register_environment("guess", lambda c: GuessEnvironment(**c))
    env = resolve_environment("guess", EnvironmentConfig(environment_type="guess").to_env_config())

    ids, _ = env.reset(["What is 2 + 2?"], [{"answer": "4"}])
    step = env.step(ids, ["Final Answer: 4"], [{}])[0]
    assert step.done
    assert step.trajectory.total_reward == 1.0   # reward/objective 1.0 + reward/turn_shaping 0.0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

- The `isolated_registry` fixture unregisters the name afterwards. Run with `pytest tests/cpu/environments -m cpu`.
- With an external reward term, call `env.settle(ids)` before reading the reward.

Then run it end to end against a served model:

```bash
python scripts/environments/inference/run_env.py \
    --env_type guess --dataset <hf-id-or-dir> --split test \
    --base_url http://localhost:8000/v1 --model <served-model> \
    --num_examples 5 --save_trajectories /tmp/guess.jsonl
```

The script resolves `--env_type` through the same registry, so call its `main()` from a wrapper that imports your module first.

## Related pages

- [Environments](README.md) — registry, common knobs, actor runtime.
- [Native Tool-Use](protocols/native-tool-use.md) — the protocol most environments extend.
- [Async GRPO with Environments](../README.md) — the trainer and its dataset surface.
