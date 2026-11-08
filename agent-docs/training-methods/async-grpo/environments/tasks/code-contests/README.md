# Code Contests Environment

`CodeContestsEnvironment` (`halo/environments/tasks/coding/code_contests.py`) trains competitive
programming. The model tries a solution in a scratchpad tool and submits it with `submit_solution`, which runs it
against the hidden tests in a [sandbox](../../sandbox.md). The reward's `environment` term pays 1 when the
submission passes every hidden test, else 0.

Two registry names share the class: `code_contests` (`output_comparison: exact`) and `codeforces` (token
comparison). The environment speaks native tool calls, so the server needs a
[tool-call parser](../../../episodes.md#tool-calls) for the model family.

Recipes ship under `examples/grpo/environmental/<family>/<backend>/`. The canonical one is
`examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-lora-ep1.yaml`.

## Configuration

```yaml
environment_type: codeforces
max_turns: 16
environment_kwargs:
  language: python           # or cpp / c, or a list ([python, cpp])
  timeout_per_test: 5
  max_grading_seconds: 150
  reasoning_effort: random
  reasoning_effort_profiles:   # reasoning_budget: an episode's free reasoning tokens; reasoning_price: per 1,000 past it
    low: {reasoning_budget: 40000, reasoning_price: 0.008, max_submissions: 1, max_test_calls: 2}
    medium: {reasoning_budget: 50000, reasoning_price: 0.006, max_submissions: 2, max_test_calls: 4}
    high: {reasoning_budget: 60000, reasoning_price: 0.004, max_submissions: 3, max_test_calls: 6}
rewards:
  - source: environment
  - source: judge            # the audit veto
    name: audit
    checks: environment      # the environment's rubric: Reward → Audit judge
```

| Knob | Default | Effect |
|---|---|---|
| [`language`](tools.md#languages) | `python` | `python`, `cpp`, `c`, or a list the model picks from |
| [`output_comparison`](grading.md#comparison) | `exact` (`tokens` under `codeforces`) | `exact`: trimmed equality, `\r\n` and `\r` read as `\n`. `tokens`: whitespace-token equality |
| [`verdict_detail`](grading.md#verdict-detail) | `outcome` | `outcome`: a failed test's verdict class alone. `full`: adds outputs, exit status and stderr |
| [`timeout_per_test`](grading.md#time-limits) | 15 s | Per-test cap when the problem declares none; the interpreted-language floor |
| `max_time_limit` | 15 s | Clamp on every per-test limit; below `timeout_per_test` it is refused |
| `compiled_time_limit_scale` | `1.0` | Multiplies a compiled language's limit |
| [`max_grading_seconds`](grading.md#grading-budget) | `None` | Wall-clock budget for one grade |
| `max_output_size` | 1 MB | Bytes, at least 1. Over-cap stdout is OUTPUT LIMIT EXCEEDED, not truncated. A test's cap rises to 4× its expected output |
| `stop_on_first_failure` | `false` | Stop at the first failing test. The grade is unchanged; `outcome/test_pass_frac` becomes a lower bound |
| `max_submissions` / `max_test_calls` | 2 / 5 | Per-episode tool budgets, overridable per effort level, and set here rather than in `tool_budgets`, which raises. The tuning lever; `max_turns` (15) is a backstop |
| [`eval_protocol`](evaluation.md#evaluation-protocols) | `harness` | `leaderboard` pins both tool budgets |

`timeout_per_test`, `max_time_limit` and `compiled_time_limit_scale` must be finite and > 0; a non-positive
`max_grading_seconds` raises too, at construction.

`sandbox_backend` / `sandbox_url` pick the [sandbox](../../sandbox.md#choosing-a-backend) both tools and the
grader run on. A backend that does not confine the program, `local` included, warns.

## Reasoning effort

`reasoning_effort` defaults to `medium` here. The class sets no per-level defaults: a profile's `reasoning_budget`
and `reasoning_price` are the [reasoning length terms](../../../reasoning.md#reasoning-length-reward), and the chat
template states the level alone ([Effort templates](../../../reasoning.md#effort-templates)).

Two profile keys make a higher level buy iteration, not only reasoning: `max_submissions` (int ≥ 1, graded
`submit_solution` calls) and `max_test_calls` (int ≥ 0, scratchpad runs).

- Both are bound at reset and enforced per call; where the level is undetermined at reset, the constructor's
  budgets stand. A value below its minimum raises at construction.
- Neither is stated to the model. A call past its budget is [refused](tools.md#refused-calls).

## Shipped recipes

Every recipe runs one economy, pinned by a CPU test ([Economy invariants](reward.md#economy-invariants)). The
curriculum recipes are the three Qwen3.6 vLLM stages
(`*-stage1-codeforces`, `*-stage2-hard`, `*-stage3-extra-hard`).

| Setting | Value |
|---|---|
| [`reasoning_budget` / `reasoning_price`](../../../reasoning.md#reasoning-length-reward) | low 40000 / 0.008, medium 50000 / 0.006, high 60000 / 0.004 per 1,000 tokens past the budget; budgets 100000 / 135000 / 170000 in stage 3 |
| [`reasoning_price_cap` / `reasoning_floor`](../../../reasoning.md#sizing) | 0.28 / 0.25 |
| [`max_submissions` / `max_test_calls`](#reasoning-effort) | low 1 / 2, medium 2 / 4, high 3 / 6; curriculum 1 / 6, 2 / 8, 3 / 8 |
| [`max_turns`](../../../episodes.md#trajectory-length) | 16; 18 in the curriculum recipes |
| [`rollout_max_tokens`](../../../episodes.md#trajectory-length) | 30000 per turn; 38192 in the curriculum recipes |
| [`rollout_max_answer_tokens`](../../../reasoning.md#answer-room) | 8192 on vLLM, so the engine closes a turn's reasoning at 21,808 tokens (30,000 in the curriculum recipes); unset on SGLang |
| [`rollout_max_episode_tokens`](../../../episodes.md#episode-output-budget) | 131072 over all turns; 128000 on gpt-oss, so the prompt and tool schema fit its 131072 window; 160000 in the curriculum recipes, 200000 in stage 3 |
| [`rollout_stop_tokens`](../../../episodes.md#tool-calls) | `<\|im_start\|>` on the Qwen3.6 vLLM recipes, so a model past a forced close cannot open a chat turn whose `<think>` restarts the reasoning cap; `<\|call\|>` on the gpt-oss recipes (the harmony tool-call terminator) |
| [`judge` term `audit`](reward.md#audit-judge) | `openai/gpt-6-luna`, `view: full`, `max_view_chars: 900000`, nine checks (six vetoes), `weight: -0.7`, `on_error: invalid` |
| [Shaping](reward.md#reward-ladder) (`environment_kwargs`) | `submission_reward` 0.1, `resubmission_penalty` 0.2, `no_tool_use_penalty` 0.2, `turn_overflow_penalty` 0.35, `length_cutoff_penalty` 0.08 with `max_length_cutoff_recoveries: 2`, `tool_error_penalty` 0.05 |
| [`language`](tools.md#languages) | `python`; `[python, cpp]` in the curriculum recipes |

## Pages

- [Tools](tools.md)
- [Grading](grading.md)
- [Reward](reward.md)
- [Datasets](datasets.md)
- [Evaluation](evaluation.md)
- [Sandboxes](../../sandbox.md)
