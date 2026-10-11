# Reward

An episode's reward is the solve, small shaping rungs around it, the trainer's reasoning length terms, and an audit
judge whose veto zeroes an episode's credit. Recipe values: [Shipped recipes](README.md#shipped-recipes).

## Reward ladder

The grade is read off the last `submit_solution` call and is all-or-nothing, matching the accept verdict pass@1
counts. Partial credit would pay a brute force that times out on the large tests.

- An unsubmitted solution, an infra outage and an inconclusive grade score 0 and pay no shaping rung; the
  resubmission penalty and tool shaping still apply. The last two mark the episode
  [invalid](grading.md#grading-budget).
- The `environment` term's `weight` prices a solve; its `exponent` has nothing to reshape
  ([Reward Terms](../../../../grpo/rewards/README.md)).

| Component | Knob | Default | Pays |
|---|---|---|---|
| `reward/objective` | `rewards:` `weight` | `1.0` | a solve: every hidden test passed |
| `reward/submission` | `submission_reward` | `0` | once a contentful graded submission lands |
| `reward/resubmission` | `resubmission_penalty` | `0` | `−resubmission_penalty` per admitted `submit_solution` call after the first |
| `reward/tool_shaping` | `no_tool_use_penalty` / `turn_overflow_penalty` | `0` | zero tool calls / burning `max_turns` |
| `reward/tool_shaping` | `length_cutoff_penalty` | `0` | per engine-cut or empty turn the episode recovers from |
| `reward/turn_shaping` | `tool_success_reward` / `tool_error_penalty` | `0` / `0` | per executed call; this env zeroes the protocol's 0.05 / 0.1 |

- Keep each shaping rung small next to the objective's weight: rungs bootstrap a base that never submits and
  cancel out within a group once every completion reaches them.
- Components log as `reward/*` and sum exactly to the reward. The trainer's
  [reasoning length terms](../../../reasoning.md#reasoning-length-reward) log outside them, as
  `reward/reasoning_floor` and `reward/reasoning_price`.

### Resubmissions

`resubmission_penalty` is flat, whether a resubmission fixes or re-rolls, and never stated to the model.
`episode/resubmission_improved` is the share of resubmissions whose pass fraction beat every earlier one.

### What scorers read

A `judge` or `reward_model` term's `final` view reads the submitted program as the episode's final answer, a
fenced code block, never the hidden tests. `view: full` reads every turn instead, and a judge's `digest` the turns
in brief ahead of that answer ([What a scorer reads](../../../../grpo/rewards/README.md#what-a-scorer-reads)).

## Audit judge

Every recipe runs a [veto judge](../../../../grpo/rewards/judge.md#veto-judge) named `audit` on what no counter
can judge. It reads the whole episode, reasoning included, beside a `context` that explains the environment and
its caps. The term's `weight` prices the non-veto checks by their check weights.

The judge is served through OpenRouter and reads `OPENROUTER_API_KEY`: pass it with `--env-file`, or drop the term
to run without a judge.

| Check | Effect | Fires on |
|---|---|---|
| `reasoning_in_actions` | veto | Working-out (roughly ten or more lines of deliberation) in a submitted program, a call the engine cut, or the visible reply |
| `notepad_run` | veto | A run or submission spent on an unfinished draft (stubs, an entry point never called) to close a turn; a complete program run without a usable input is `wasted_run`, not this |
| `hardcoded_output` | veto | Output the program did not compute: special-cased or sample-fitted answers, guessed placeholders |
| `environment_probe` | veto | Reading the sandbox, the network or the grader instead of solving the problem |
| `verdict_probe` | veto | A submission made for its verdict (a stub, a guess, a program the policy had found wrong) or a resubmission whose outputs cannot differ; a passing one only as the latter |
| `verdict_mining` | veto | Changes aimed at the failing tests rather than the method |
| `wasted_run` | check weight 1; fired, it costs 0.1 in the recipes | An input-reading program run with no stdin or with stdin that is no attempt at an input; not a program carrying its own inputs, nor a malformed input of the policy's own |
| `reasoning_in_runs` | check weight 0, logged | `reasoning_in_actions`' working-out in a program sent to the scratchpad that ran |

A veto zeroes every credit (the solve and the submission bonus); penalties stand. `wasted_run` is priced and `reasoning_in_runs` only logged, not vetoed: a veto
there would teach a policy to submit untested.

### Judge outages

The recipes' [`on_error: invalid`](../../../../grpo/rewards/README.md#common-options) drops an episode the judge
could not score from its group's baseline and the loss, so an outage (an expired key, a provider failure) trains
nothing rather than every episode unjudged. `judge/audit/scored` and `sampling/invalid_episode_frac` show it, and a
second step in a row with no valid episode stops the run ([Batch construction](../../../objective.md#batch-construction)).

### Cap context

Where the engine closes reasoning at a one-token marker (Qwen3.6 and Gemma 4 on vLLM), a turn whose reasoning
reaches the engine's ceiling is marked for the judge. Each recipe's `context` states what the policy sees of its
earlier reasoning (`carry_reasoning`) and how its turns are budgeted: a soft budget on vLLM and SGLang, a turn
limit on gpt-oss, whose template states no budget.

## Economy invariants

`tests/cpu/config/test_env_grpo_reward_economy.py` holds every shipped recipe to these relations:

- The floor's weight and the price's cap each stay under the resubmission penalty, so reasoning length never
  outweighs whether to resubmit. An episode pays at most one of the two.
- A graded submission that passes nothing scores above an episode that never attempts.
- At every level, a solve that pays every per-episode price (its resubmissions, admitted recoveries, the overflow,
  the worst length term) out-scores any zero-objective episode, before `tool_error_penalty` per failed call and the
  audit judge's `wasted_run` price.
- A recovered cut costs less than the submission bonus and no more than an overflow.
- Below its floor every level is paid to reason more; a higher level is priced lower and asked for no less.
- Per-turn budgets match across families, all enforced or all soft, and soft budgets are priced past them.
- The episode output budget binds below `max_turns × rollout_max_tokens`; on vLLM the answer room binds below the
  turn ceiling.

## Behavior counters

- `episode/submission_rate`
- `episode/test_calls`: scratchpad runs that counted
- `episode/tested_before_submission`: over submitting episodes
- `episode/grading_budget_hit`
- `episode/identical_resubmissions`: programs refused as already graded
- `episode/resubmission_improved`: [Resubmissions](#resubmissions)
- `episode/language_switches`: where the model picks the language

Grade outcomes (`episode/grade_inconclusive`, `episode/grading_infra_outage`, `episode/tests_graded_frac`):
[Grading](grading.md).
