# SWE Environment

`SweEnvironment` (`src/environments/envs/tasks/coding/swe.py`) is a stateful, SWE-agent-style environment: read / edit / run-tests over a workspace that survives across turns. Registry name `swe`; `language` accepts `python` (default), `cpp`, or `c`.

Each episode gets its own persistent [`SandboxSession`](sandbox.md#sessions-persistent-multi-turn-state) — a real working directory — bound to the file and code tools through a `ContextVar`, so one environment instance serves concurrent rollouts safely. Sessions are closed by `cleanup()` after each episode.

## Tools

Four tools over that one workspace (`create_session_file_tools` / `create_session_code_tools`): `run_code` (compile if needed, then run; output via stdout), `write_file`, `read_file`, and `list_files` (optionally filtered by a path prefix).

Code runs in real OS isolation through a `SandboxExecutor`, not the in-process restricted REPL, so imports and the standard library are available. The backend is resolved once per environment from `sandbox_backend` / `sandbox_url`, falling back to `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` then local subprocess — see [Code Execution Sandboxes](sandbox.md).

## Usage

```python
from src.environments.envs.tasks.coding.swe import SweEnvironment

env = SweEnvironment(
    max_turns=20,
    success_reward=1.0,
    language="python",               # "python" (default), "cpp", or "c"
    code_timeout=15.0,               # per-run wall-clock cap; default SANDBOX_DEFAULT_TIMEOUT
    extra_tools=my_additional_tools, # extra tool registry, merged in without a subclass
    test_function=my_test_fn,        # callable graded on the finished trajectory
    sandbox_backend="bubblewrap",    # else HALO_SANDBOX_BACKEND, then "local"
)
```

## Reward

An incomplete episode scores `failure_reward`. A completed one is graded in priority order:

1. `test_function` → `success_reward` / `failure_reward`. A grader that raises is logged, scores failure, and marks the episode `episode_invalid`, keeping the forced failure out of the GRPO group baseline.
2. A `context["validator"]` or `context["answer"]` check, delegated to the parent.
3. Otherwise, completion with at least one *successful* tool call earns `success_reward`; completion with none scores `failure_reward` — a no-interaction answer is the worst behavior in a tool-use env, so it is not softened.

Per-turn tool deltas (`tool_success_reward` 0.05 added per successful call, `tool_error_penalty` 0.1 subtracted per failed one; magnitudes ≥ 0) accumulate on top, as does the inherited [per-episode shaping](native-tool-use.md#reward-knobs) — `no_tool_use_penalty`, `multi_turn_reward`, `turn_overflow_penalty`, all `0` unless set.

`SweEnvironment` is a minimal SWE-agent loop, not a full agentic coding harness. For hidden-test competitive-programming grading see [Code Contests](code-contests.md).
