# Reward

An episode's reward is the solve, small shaping rungs around it, the trainer's reasoning length terms (not charged in
a training group with no success), and an audit judge whose veto zeroes an episode's credit and whose weighted checks
charge a fixed price. Recipe values: [Shipped recipes](README.md#shipped-recipes).

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
| `reward/unrun_submission` | `unrun_submission_penalty` | `0` | once, in an episode that never passed, when a failed graded submission was a program the scratchpad never ran as submitted before its turn while a run was left (comments, whitespace and dropped output statements aside). The match is textual, so a tested program that a guard, a rename or a harness changed counts as unrun: the recipes leave it at `0` and price the failure through `untested_submission`, which reads the change |
| `reward/tool_shaping` | `no_tool_use_penalty` / `turn_overflow_penalty` | `0` | zero tool calls / ending truncated: `max_turns` or the output budget spent, or a cut it could not recover |
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
its caps. Its checks are the environment's own: `checks: environment` takes the rubric `CodeContestsEnvironment`
declares (`JUDGE_CHECKS`, built from `AUDIT_CHECKS` in `code_contests_prompts.py`), the run's scratchpad (`python_repl` or `run_code`) named
in it, so a recipe states only the judge's model, view, price and context. The context comes from a shared
`context_file` under `examples/grpo/environmental/code-contests-judge/`, one file per chat template, rollout
backend and scratchpad: `gptoss.txt` (gpt-oss on either engine), `gemma4-vllm.txt`, `gemma4-qwen3.6-sglang.txt`,
`qwen3.6-vllm.txt`, and `qwen3.6-vllm-run-code.txt` for the Python and C++ stages. The term's `weight` (−0.7) prices
the weighted checks by their check weights, each fired one adding its own price. The check weights are the
environment's (7 in all), so −0.7 is 0.1 per unit; a recipe of your own that lists `checks: environment` takes the
same check weights, so its unit is its `weight` / 7.

The judge is served through OpenRouter and reads `OPENROUTER_API_KEY`: pass it with `--env-file`, or drop the term
to run without a judge.

| Check | Effect | Fires on |
|---|---|---|
| `reasoning_in_actions` | veto | Working-out (roughly ten or more lines of deliberation) in a submitted program, a call the engine cut, a scratchpad call sent after a reply said the scratchpad budget was spent, or the visible reply |
| `notepad_run` | veto | A run or submission spent on an unfinished draft (stubs, an entry point never called) to close a turn; a complete program run without a usable input is `wasted_run`, not this |
| `hardcoded_output` | veto, check weight 3 (0.3) | Output the program did not compute: special-cased answers, exact for small inputs and guessed beyond, stock answers where the method fails, and on a failed submission rules fitted to the samples or its own brute force; not a capped search computed from the input, nor a rule found from small cases in a passing program |
| `environment_probe` | veto | Reading the sandbox, the network or the grader instead of solving the problem |
| `verdict_probe` | veto, check weight 2 (0.2) | A submission made for its verdict (a stub, a guess, a program the policy had shown or called wrong, sent to see) or a resubmission whose outputs cannot differ; a passing one only as the latter. A program sent after the scratchpad budget was spent is exempt, whatever the policy expected of it, unless it is such a resubmission |
| `verdict_mining` | veto | Changes aimed at the failing tests rather than the method |
| `wasted_run` | check weight 1 (0.1) | An input-reading program run with no stdin or with stdin that is no attempt at an input; not a program carrying its own inputs, nor a malformed input of the policy's own |
| `untested_submission` | check weight 1 (0.1) | In an episode that never passed, a failed submission whose logic no earlier run exercised (a change that cannot alter its output is no change) or that ran only on the samples, while runs remained; exempt by the environment (`unless: failed_with_no_runs_left`) when every failed submission came with no run left |
| `reasoning_in_runs` | check weight 0, logged | `reasoning_in_actions`' working-out in a program sent to the scratchpad that ran, failed to compile, or was the first call refused for a spent budget: the policy is not told its run budget, so the first refusal comes unannounced |

A veto zeroes every credit (the solve and the submission bonus); penalties stand, the veto's own price among them, so a guess or a probe
on a problem nobody solves still costs more than an honest failure. `wasted_run` is priced and `reasoning_in_runs` only logged, not
vetoed: a veto there would teach a policy to submit untested. `untested_submission` prices failed episodes only: the policy tests when
it is unsure, so within a group an outcome-only reward charges testing, and pricing the untested failure is what pays for testing
beyond the samples. Prices add when several checks fire. A call the environment refused unrun is marked so in the judge's view: it ran nothing and spent nothing.

### Judge outages

The recipes' [`on_error: invalid`](../../../../grpo/rewards/README.md#common-options) drops an episode the judge
could not score from its group's baseline and the loss, so an outage (an expired key, a provider failure) trains
nothing rather than every episode unjudged. `judge/audit/scored` and `sampling/invalid_episode_frac` show it, and a
second step in a row with no valid episode stops the run ([Batch construction](../../../objective.md#batch-construction)).

### Cap context

Where the engine closes reasoning at a one-token marker (Qwen3.6 and Gemma 4 on vLLM), a turn whose reasoning
reaches the engine's ceiling is marked for the judge. Each context file states what the policy sees of its
earlier reasoning (`carry_reasoning`) and how its turns are budgeted: a soft budget on vLLM and SGLang, a turn
limit on gpt-oss, whose template states no budget.

## Economy invariants

`tests/cpu/config/test_env_grpo_reward_economy.py` holds every shipped recipe to these relations:

- The floor's weight and the price's cap each stay under one more failed submission at its worst honest price
  (the resubmission penalty plus `unrun_submission_penalty` and `untested_submission`, which a program never run as
  submitted can both raise), so reasoning length never outweighs whether to resubmit. An episode pays the floor or the price, never both.
- A graded submission that passes nothing scores above an episode that never attempts; raising every audit process
  flag, it scores no lower at the same length terms.
- At every level, a solve that pays every per-episode price (its resubmissions, admitted recoveries, the worst
  length term, the process flags a solve can raise) out-scores any zero-objective episode, before
  `tool_error_penalty` per failed call. A solve never pays the overflow: the accept ends the episode.
- The overflow costs more than one more failed submission at its worst honest price (the same bound). A truncated
  episode never solves and always has a submission left, so running
  out is never the cheapest way to fail.
- A recovered cut costs less than the submission bonus and no more than an overflow.
- Below its floor every level is paid to reason more; a higher level is priced lower and asked for no less.
- Per-turn budgets match across families, all enforced or all soft, and soft budgets are priced past them.
- The episode output budget binds below `max_turns × rollout_max_tokens`; on vLLM the answer room binds below the
  turn ceiling.
- Each audit check fired alone costs what the [audit table](#audit-judge) states, so moving a judge `weight` or a
  check weight alone, which re-prices every weighted check, fails here; each shaping price is the one the
  [recipe table](README.md#shipped-recipes) states.

## Behavior counters

- `episode/submission_rate`
- `episode/test_calls`: scratchpad runs that counted
- `episode/tested_before_submission`: over submitting episodes
- `episode/grading_budget_hit`
- `episode/identical_resubmissions`: programs refused as already graded
- `episode/unrun_failed_submission`: a failed graded submission was a program never run as submitted while a run was left (priced as `reward/unrun_submission` where `unrun_submission_penalty` is set)
- `episode/resubmission_improved`: [Resubmissions](#resubmissions)
- `episode/language_switches`: where the model picks the language

Grade outcomes (`episode/grade_inconclusive`, `episode/grading_infra_outage`, `episode/tests_graded_frac`):
[Grading](grading.md).
