# Custom Environments

Subclass `BaseEnvironment` (`src/environments/base.py`) and implement three methods, then either register a factory under a name for YAML, or hand the trainer `environment_cls` from Python.

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

- **Grade, do not price.** `_grade_episode` returns an `EpisodeGrade`: `objective` is the task grade in `[0, 1]`, which the reward's `environment` term prices as `weight × grade ^ exponent` (`reward/objective`); `shaping` is the environment's own episode-level terms by bare name (`{"submission": 0.25}`), each added to the reward and logged as `reward/<name>`. Declare the shaping names on the class, `SHAPING_COMPONENTS = ("submission", ...)` (the union over the class hierarchy is what an episode may carry; an undeclared name fails at the first settled episode, and a reward term may not take a declared name). `objective` and `turn_shaping` are reserved. A class that defines `_compute_reward` is refused at construction. Protocol-level shaping goes in `_episode_shaping(trajectory) -> dict` (the native protocol declares and returns `tool_shaping`). Mark an episode whose grade carries no signal — a grader outage, a null `answer` cell — with `trajectory.info[EPISODE_INVALID_KEY] = True` and grade it 0, so it leaves the group baseline ([Reward Terms](../rewards.md#environment-arm)).
- **Forward `**kwargs` to `super().__init__`.** The base binds the config's `rewards` list as `reward_terms`, plus `max_turns` and the shared knobs; any keyword no constructor in the chain binds raises `TypeError` at construction — that strictness is what turns a typo'd `environment_kwargs` key into an error instead of a silently ignored setting.
- **Declare differing defaults on the class** — `DEFAULT_MAX_TURNS`, `DEFAULT_TOOL_SUCCESS_REWARD`, `DEFAULT_TOOL_ERROR_PENALTY`, `requires_answer`, all declared on `BaseEnvironment` — never in the factory, so an explicit YAML `max_turns` still wins.
- **Build through `self._init_trajectory(...)`** so the system turn, the task and the context land where the base and the metrics expect them.
- **Route an unproductive turn, never grade it.** A turn the engine cut at its token cap or aborted arrives as `context["finish_reason"] in ENGINE_CUT_FINISH_REASONS` (`src/inference/response.py`) and carries a mid-sentence fragment; `_step_single` hands it to `self._handle_length_cutoff(trajectory)`, which nudges and retries it within `max_turns` and the episode's recovery cap. The class must declare `LENGTH_CUTOFF_NUDGE` — the message the model sees — or the base raises `NotImplementedError` there. Grading the fragment instead scores a cut-off attempt as a deliberate final answer ([Reasoning budget](../async-grpo/rollouts.md#reasoning-budget)). The other kind is a turn that ends with neither visible content nor a tool call: route it to `self._handle_empty_turn(trajectory)`, which flags the turn untrainable (`Message.empty`) and nudges the same way, and declare `EMPTY_TURN_NUDGE` or the base raises `NotImplementedError` there too. Both kinds spend the one `max_length_cutoff_recoveries` cap, and on the native protocol both are priced by `length_cutoff_penalty`.
- **For I/O-bound steps** subclass `AsyncBaseEnvironment` and override `_reset_single_async` / `_step_single_async`; free per-episode resources in `_release_episode`. The base's `verify_backend` probes every external reward term at launch; an environment with a backend of its own extends it and calls `super()`, so a bad URL fails the launch instead of every episode.

## Register it

```python
from src.environments.registry import register_environment

register_environment("guess", lambda env_config: GuessEnvironment(**env_config))
```

Forward the whole `env_config`: a factory that ignores it drops every `environment_kwargs` key silently. The registry only knows factories that have run, so register from a module the entry script imports. Then set `environment_type: guess` in the YAML. From Python, skip the registry: `DistributedAsyncEnvironmentalGRPOTrainer(..., environment_cls=GuessEnvironment, environment_kwargs={...})`.

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

Arguments are filtered to the declared parameters before the handler runs, so a hallucinated extra never reaches it and a missing required one is refused as a tool error without spending the episode's tool budget. A handler that raises marks the call failed and charges `tool_error_penalty` — raise for an infrastructure fault, return a string for a legitimate negative answer. Pass the registry as `NativeToolUseEnvironment(tool_registry=registry, ...)`.

## Trainable trajectory shape

The trainer builds one row per assistant turn from the engine's sampled ids, or falls back to a single render of the whole trajectory ([training on sampled tokens](../async-grpo/rollouts.md#training-on-sampled-tokens)). Four rules keep an environment decomposable either way:

- **Let `step()` append the assistant turn.** It is the only carrier of the engine's sampled ids, logprobs and reasoning; one hand-appended assistant message drops the whole trajectory onto the re-tokenization fallback and trains text the policy never emitted.
- **Give every advertised tool call a result message.** Set `max_tool_calls_per_turn` when the environment executes fewer calls than the model may request; the base trims the stored call list to match.
- **Return tool output as a message** — `Message.tool(content, tool_call_id, name)`, or a plain user turn (`Observation: …`) for a text protocol. The loss mask comes from spans, not roles.
- **Keep `tool_calls` renderable.** An `arguments` value the chat template cannot handle fails the render, and a trajectory whose turn spans cannot be located is dropped on every rank rather than trained on a guess.

Override `rollout_metrics(trajectory) -> dict[str, float]` to log per-episode diagnostics under their full paths (`outcome/*`, `episode/*`, `reward/*`); a string fact stamped in `trajectory.info["slices"]` slices them as `<slice>/<value>/*` ([metrics](../async-grpo/monitoring.md#logged-metrics)).

## Dataset

```python
{"prompt": str | list[dict], "answer": Any, "difficulty": "hard"}
```

`answer` and every column named in the script's `context_fields` reach `reset()` / `step()` in the `context` dict; all other columns are dropped, and a `context_field` — or an `answer_field` renamed away from `answer` — that names no column raises at startup.

Set `requires_answer = True` on the class when the reward grades against `context["answer"]`: the trainer then refuses a dataset without that column instead of scoring every episode of the run alike. A per-run override goes in `environment_kwargs` as `requires_answer`, and the default is `False` — completing the task is the objective.

A list-valued prompt reaches the environment as its **last `user` message** only, so put framing in the environment's `system_prompt`.

## Test it

A CPU test, in the shape of `tests/cpu/environments/test_environments.py`:

```python
import pytest

def test_guess_env_grades_a_correct_answer(isolated_registry):
    from src.configs.environment_config import EnvironmentConfig
    from src.environments.registry import register_environment, resolve_environment

    register_environment("guess", lambda c: GuessEnvironment(**c))
    env = resolve_environment("guess", EnvironmentConfig(environment_type="guess").to_env_config())

    ids, _ = env.reset(["What is 2 + 2?"], [{"answer": "4"}])
    step = env.step(ids, ["Final Answer: 4"], [{}])[0]
    assert step.done
    assert step.trajectory.total_reward == 1.0   # reward/objective 1.0 + reward/turn_shaping 0.0

if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
```

The suite's `isolated_registry` fixture drops the registered name again, so later tests resolve real environments. Run it with `pytest tests/cpu/environments -m cpu`. With a `judge` or `reward_model` term the episode dispatcher settles the episode; a test that steps the environment directly calls `env.settle(ids)` before reading the reward.

Then run the environment end to end against a served model on a handful of rows, which exercises the real tool calls and the reward:

```bash
python scripts/environments/inference/run_env.py \
    --env_type guess --dataset <hf-id-or-dir> --split test \
    --base_url http://localhost:8000/v1 --model <served-model> \
    --num_examples 5 --save_trajectories /tmp/guess.jsonl
```

The script resolves `--env_type` through the same registry, so run it from a wrapper that imports your module first and then calls its `main()`.

## Related pages

- [Environments](README.md) — registry, shared knobs, actor runtime.
- [Native Tool-Use](native-tool-use.md) — the protocol most environments extend.
- [Async GRPO with Environments](../async-grpo/README.md) — the trainer and its dataset surface.
