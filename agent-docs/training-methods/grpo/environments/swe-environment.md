# SWE Environment

`SweEnvironment` (`src/environments/envs/tasks/coding/swe.py`, registry name `swe`) trains an
edit-run-test loop over a workspace that survives across turns: the model writes files, runs code,
and answers when done. It speaks native tool calls, so the server needs a tool-call parser for the
model family ([Rollout Configuration](../async-grpo/rollouts.md#tool-calls)).

No third-party agent scaffold is wired in — not SWE-agent, mini-SWE-agent, OpenHands or opencode.
The model gets the tool set below over one persistent sandbox workspace, and nothing else.

Each episode gets its own sandbox session — a real working directory, closed on cleanup — so one
instance serves concurrent rollouts. No shipped recipe targets `swe`; start from
`examples/grpo/environmental/environmental-grpo-template.yaml`.

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
| `max_turns` | `20` | Turn budget; an edit-run-test loop outlasts the generic 10 |

Reward knobs are the native protocol's ([Native Tool-Use](native-tool-use.md)).

## Tools

- `run_code` — compiles if needed and runs a program in the workspace; stdout is the output.
- `run_bash_command` — runs one shell command with `bash` in the workspace, under the same
  `code_timeout`. A zero exit returns **stdout only** — stderr is discarded, and empty output
  comes back as `Code executed successfully (no output)`, so diagnostics need an explicit `2>&1`.
  A non-zero exit returns stdout plus `Error: <last stderr line>`: an observation, not a tool
  error. Only a sandbox backend failure is booked as a failed tool call.
- `write_file` — writes a file later turns and `run_code` see.
- `read_file` — reads one back; a missing file is a message, not a tool error.
- `list_files` — lists the workspace, optionally filtered by a path prefix.

Code runs in OS isolation through a `SandboxExecutor`, so imports and the standard library work
([Sandboxes](sandbox.md#choosing-a-backend)). Files a command creates persist in the workspace on
`local` and `bubblewrap`; the `remote` service is stateless per call, so a run there sees only what
`write_file` wrote.

A shell command has the same exposure as `run_code`: the `local` backend bounds CPU, memory and
disk, but does not namespace the network or the host filesystem — other episodes' workspaces
included. Run untrusted shell on `bubblewrap` or `remote`.

## Reward

An episode that never completed grades 0. A completed one is graded in this order:

1. A `test_function` passed to the constructor: its verdict, 1 or 0. A grader that raises grades 0 and marks the episode invalid, keeping it out of the GRPO group baseline.
2. A `validator` or `answer` in the row's context: the protocol's answer grading.
3. Otherwise completion itself: one *successful* tool call grades 1, zero grades 0.

The reward's `environment` term prices the grade as `reward/objective` ([Reward Terms](../rewards.md#environment-arm)); the protocol's per-call and episode-level shaping add on top ([shared knobs](README.md#configuration)).

## Dataset

`{"prompt": ...}`, plus `{"answer": ...}` where the episode is graded against an expected answer.

## Evaluation

```bash
python scripts/environments/inference/run_env.py --env_type swe \
    --dataset <hub-id-or-dir> --base_url http://localhost:8000/v1 \
    --model <served-name> --num_examples 20
```

Flags and the `--training_config` contract: [Evaluating on an Environment](evaluation.md).

## From Python

`test_function` grades the finished trajectory; `extra_tools` merges a registry in:

```python
env = SweEnvironment(language="python", test_function=my_test_fn, extra_tools=my_tools)
```

## Related pages

- [Sandboxes](sandbox.md) — backends, languages, limits
- [Code Contests](code-contests.md) — hidden-test grading
