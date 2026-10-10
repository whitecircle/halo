# Code Contests Environment

`CodeContestsEnvironment` (`src/environments/envs/tasks/coding/code_contests.py`) trains competitive
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
  reasoning_effort_profiles:   # thinking_tokens: the level's per-turn reasoning budget
    low: {thinking_tokens: 8192, enforce_thinking_cap: false, max_submissions: 1, max_test_calls: 2}
    medium: {thinking_tokens: 12288, enforce_thinking_cap: false, max_submissions: 2, max_test_calls: 4}
    high: {thinking_tokens: 16384, enforce_thinking_cap: false, max_submissions: 3, max_test_calls: 6}
rewards:
  - source: environment
  - source: judge            # the audit veto
    name: audit
    checks: [...]            # Reward → Audit judge
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
| `max_output_size` | 1 MB | Over-cap stdout is OUTPUT LIMIT EXCEEDED, not truncated. A test's cap rises to 4× its expected output |
| `stop_on_first_failure` | `false` | Stop at the first failing test. The grade is unchanged; `outcome/test_pass_frac` becomes a lower bound |
| `max_submissions` / `max_test_calls` | 2 / 5 | Per-episode tool budgets, overridable per effort level. The tuning lever; `max_turns` (15) is a backstop |
| [`eval_protocol`](evaluation.md#evaluation-protocols) | `harness` | `leaderboard` pins both tool budgets |

`timeout_per_test`, `max_time_limit` and `compiled_time_limit_scale` must be finite and > 0; a non-positive
`max_grading_seconds` raises too, at construction.

`sandbox_backend` / `sandbox_url` pick the [sandbox](../../sandbox.md#choosing-a-backend) both tools and the
grader run on. A backend that does not confine the program, `local` included, warns.

## Reasoning effort

`reasoning_effort` defaults to `medium` here. The class ladder sets only `thinking_tokens`, the level's per-turn
[reasoning budget](../../../reasoning.md#reasoning-budget): low 4096, medium 8192, high 16384. It is enforced
unless the profile sets `enforce_thinking_cap: false`.

- `reasoning_effort_profiles` merges per level over the ladder, so a profile naming only tool budgets keeps the
  class budget.
- The chat template states the level to the model. The Qwen3.6 and Gemma 4 recipes pin the shipped
  [effort template](../../../reasoning.md#effort-templates), which also states the budget; gpt-oss's harmony template does not.

Two profile keys make a higher level buy iteration, not only longer reasoning: `max_submissions` (int ≥ 1, graded
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
| [Per-turn reasoning budget](../../../reasoning.md#reasoning-budget) | `thinking_tokens` low 8192, medium 12288, high 16384, all soft (`enforce_thinking_cap: false`): an enforced cap cuts reasoning mid-thought and teaches a policy to stop well short of the budget |
| [`max_submissions` / `max_test_calls`](#reasoning-effort) | low 1 / 2, medium 2 / 4, high 3 / 6; curriculum 1 / 4, 2 / 6, 3 / 8 |
| [`max_turns`](../../../episodes.md#trajectory-length) | 16; 18 in the curriculum recipes |
| [`rollout_max_tokens`](../../../episodes.md#trajectory-length) | 30000 per turn |
| [`rollout_max_answer_tokens`](../../../reasoning.md#answer-room) | 8192 on vLLM, so the engine closes a turn's reasoning at 21,808 tokens; unset on SGLang |
| [`rollout_max_episode_tokens`](../../../episodes.md#episode-output-budget) | 131072 over all turns |
| [`rollout_stop_tokens`](../../../episodes.md#tool-calls) | `<\|im_start\|>` on the Qwen3.6 vLLM recipes, so a model past a forced close cannot open a chat turn whose `<think>` starts a fresh budget; `<\|call\|>` on the gpt-oss recipes (the harmony tool-call terminator) |
| [`reasoning_floor`](../../../reasoning.md#reasoning-length-reward) | 0.10 on the Qwen3.6 vLLM recipes, 0.05 elsewhere |
| [`reasoning_price`](../../../reasoning.md#price-past-the-budget) with `reasoning_price_past_budget: true` | low 0.015, medium 0.01, high 0.005 per 1,000 tokens a turn reasons past its level's budget; `reasoning_price_cap: 0.1` an episode |
| [`judge` term `audit`](reward.md#audit-judge) | `openai/gpt-6-luna`, `view: full`, `max_view_chars: 900000`, eight checks (six vetoes), `weight: -0.1`, `on_error: invalid` |
| [Shaping](reward.md#reward-ladder) (`environment_kwargs`) | `submission_reward` 0.1, `resubmission_penalty` 0.2, `no_tool_use_penalty` 0.1, `turn_overflow_penalty` 0.1 (0.15 on Qwen3.6 vLLM), `length_cutoff_penalty` 0.08 with `max_length_cutoff_recoveries: 2`, `tool_error_penalty` 0.05 |
| [`language`](tools.md#languages) | `python`; `[python, cpp]` in the curriculum recipes |

## Pages

- [Tools](tools.md)
- [Grading](grading.md)
- [Reward](reward.md)
- [Datasets](datasets.md)
- [Evaluation](evaluation.md)
- [Sandboxes](../../sandbox.md)
