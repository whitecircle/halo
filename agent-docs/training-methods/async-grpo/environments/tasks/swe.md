# SWE Environment

`SweEnvironment` (`src/environments/envs/tasks/coding/swe.py`, registry name `swe`) trains an edit-run-test loop over a workspace that persists across turns. It speaks native tool calls, so the server needs the family's tool-call parser ([Tool calls](../../episodes.md#tool-calls)).

- No third-party agent scaffold (SWE-agent, mini-SWE-agent, OpenHands, opencode) is wired in: the model gets the tools below and nothing else.
- Each episode gets its own sandbox session, so one instance serves concurrent rollouts.
- No shipped recipe targets `swe`; start from `examples/grpo/environmental/environmental-grpo-template.yaml`.

## Configuration

```yaml
environment_type: swe
max_turns: 20
environment_kwargs:
  language: python
  code_timeout: 15.0
```

| Knob | Default | Effect |
|---|---|---|
| `language` | `python` | What `run_code` runs; `cpp` and `c` need a compiling backend |
| `code_timeout` | `15.0` s | Wall-clock cap per `run_code` / `run_bash_command` call |
| `sandbox_backend` / `sandbox_url` | unset | Else `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL`, then local |

`max_turns` defaults to 20 ([common knobs](../README.md#common-knobs)). Reward knobs are the [native protocol's](../protocols/native-tool-use.md#configuration).

## Tools

- `run_code` — compiles if needed and runs a program in the workspace; stdout is the output.
- `run_bash_command` — runs one `bash` command in the workspace under the same `code_timeout`.
    - A zero exit returns **stdout only**; empty output reads `Code executed successfully (no output)`. Diagnostics need an explicit `2>&1`.
    - A non-zero exit leads with `Error:` and the killing signal or the tail of stderr, then stdout. It is an observation, not a tool error.
- `write_file` — writes a file later turns and `run_code` see.
- `read_file` — reads one back; a missing file is a message, not a tool error.
- `list_files` — lists the workspace, optionally filtered by a path prefix.

A sandbox fault on any tool ends the episode ([Sandbox faults](../sandbox.md#sandbox-faults)).

### Isolation

Code runs in a `SandboxExecutor` subprocess with imports and the standard library ([Choosing a backend](../sandbox.md#choosing-a-backend)).

| Backend | Confinement | Files a command creates |
|---|---|---|
| `local` | rlimits only; network and host filesystem (other episodes' workspaces included) open | Persist in the workspace |
| `bubblewrap` | Confined without `allow_network` | Persist in the workspace |
| `remote` | Confined | Not kept: a run sees only what `write_file` wrote |

`swe` [warns](../sandbox.md#confinement) when built on a backend that does not confine the program: `local`, or `bubblewrap` with `allow_network`. Run untrusted code and shell on `bubblewrap` without network, or on `remote`.

## Reward

An episode that never completed grades 0. A completed one is graded by the first that applies:

1. A `test_function` passed to the constructor: its verdict, 1 or 0. One that raises grades 0 and marks the episode invalid, out of the group baseline.
2. Else the row's `validator` callable, or its `answer` matched against the final reply (exact match, then numeric). The workspace is never inspected. A null `answer` cell marks the episode invalid.

There is no completion fallback:

- Without a `test_function`, `requires_answer` defaults on: the trainer and `run_env.py` refuse a dataset with no `answer` column, and the playground an empty Expected Answer box.
- `requires_answer: false` without a `test_function` raises at construction.
- A row carrying neither raises at grading.
- A judge-only reward (no `environment` term) takes no grade from the environment, so it needs neither, and a null `answer` or a raising `test_function` voids nothing.

The `environment` term prices the grade as `reward/objective` ([Reward Terms](../../../grpo/rewards/README.md#environment-arm)); the native protocol's shaping adds on top ([Native Tool-Use → Reward](../protocols/native-tool-use.md#reward)).

## Dataset

`{"prompt": ..., "answer": ...}`; the `answer` column is optional only under a `test_function` or a judge-only reward ([Reward](#reward)).

## Evaluation

```bash
python scripts/environments/inference/run_env.py --env_type swe \
    --dataset <hub-id-or-dir> --base_url http://localhost:8000/v1 \
    --model <served-name> --num_examples 20
```

Flags and the `--training_config` contract: [Evaluation](../../evaluation.md).

## From Python

`test_function` (grades the finished trajectory) and `extra_tools` (a registry merged in) are Python objects, so only the constructor sets them:

```python
env = SweEnvironment(language="python", test_function=my_test_fn, extra_tools=my_tools)
```

## Related pages

- [Code Execution Sandboxes](../sandbox.md) — backends, languages, limits.
- [Code Contests](code-contests/README.md) — hidden-test grading.
