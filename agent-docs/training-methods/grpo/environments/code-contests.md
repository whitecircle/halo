# Code Contests Environment

`CodeContestsEnvironment` (`src/environments/envs/tasks/coding/code_contests.py`) trains
competitive programming: the model writes a solution, tries it in a scratchpad tool, and submits it
with `submit_solution`, which runs it against the hidden tests through a [sandbox](sandbox.md).
The grade is the fraction of tests passed, priced by the reward's `environment` term. Two registry names share the class — `code_contests`
(`output_comparison: exact`) and `codeforces` (token comparison).

It speaks native tool calls, so the server needs a tool-call parser for the model family
([Rollout Configuration](../async-grpo/rollouts.md#tool-calls)). Recipes ship under
`examples/grpo/environmental/<family>/<backend>/`; the canonical one is
`examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-lora-ep1.yaml`.

## Configuration

```yaml
environment_type: codeforces
max_turns: 14                # the shipped value; the class default is 15
rewards:
  - source: environment      # the pass fraction of the submitted solution
    exponent: 2.0            # convex partial credit: half-right earns a quarter
environment_kwargs:
  language: python           # or cpp / c, or a list ([python, cpp]) the model picks from
  timeout_per_test: 5
  max_grading_seconds: 150
  verdict_detail: outcome
  reasoning_effort: random
  reasoning_effort_profiles:
    low: {thinking_tokens: 8192, max_submissions: 2, max_test_calls: 2}
    medium: {thinking_tokens: 12288, max_submissions: 3, max_test_calls: 4}
    high: {thinking_tokens: 16384, max_submissions: 3, max_test_calls: 6}
```

| Knob | Default | Effect |
|---|---|---|
| `language` | `python` | `python`, `cpp`, `c`, or a list the model picks from |
| `output_comparison` | `exact` (`tokens` under `codeforces`) | `exact` is trimmed byte equality, `tokens` whitespace-token equality |
| `verdict_detail` | `full` | `full` shows a failed test's expected and produced output; `outcome` the verdict alone |
| `timeout_per_test` | 15 s | Per-test cap when the problem declares none; also the interpreted floor |
| `max_time_limit` | 15 s | Clamp on a declared limit; below `timeout_per_test` it is refused |
| `compiled_time_limit_scale` | `1.0` | Multiplies a compiled language's per-test limit; a non-finite or non-positive value raises at construction |
| `max_grading_seconds` | `None` | Wall-clock budget for one grade; a non-positive value raises at construction |
| `repl_timeout` | 15 s | Cap on one scratchpad run |
| `max_output_size` | 1 MB | Over-cap stdout is OUTPUT LIMIT EXCEEDED, not truncated |
| `stop_on_first_failure` | `false` | Stop at the first failing test; the pass fraction becomes a lower bound |
| `max_submissions` / `max_test_calls` | 2 / 5 | Per-episode tool budgets, overridable per effort level |
| `max_turns` | 15 | Backstop; the tool budgets are the tuning lever |

The objective's shape is the `environment` term's `exponent` in the top-level `rewards:` — above 1 it is convex, so half-right earns under half a solve ([Reward Terms](../rewards.md)).

### Reasoning effort

`reasoning_effort` defaults to `medium` here, and the class ladder sets `thinking_tokens` only: low
4096, medium 8192, high 16384. `reasoning_effort_profiles` merges per level over it, so a profile
naming only interaction keys keeps the class budget
([Reasoning budget](../async-grpo/rollouts.md#reasoning-budget)). The level and its budget reach the
model through the chat template; on Qwen3.6 that is the shipped effort template the recipes pin.

This environment adds three profile keys, bound per episode:

- `max_submissions` (int ≥ 1) and `max_test_calls` (int ≥ 0) — the episode's tool budgets, stamped at reset and stated in the task message. They make effort buy iteration, not just longer reasoning; without them the strategy collapses to submit-and-fix.
- `tested_submission_reward` (≥ 0) — paid once when a scratchpad run precedes the first submission. Unstated in the prompt: it steers through the gradient.

A value below its minimum raises at construction; where the level is undetermined at reset, the
constructor's budgets stand.

## Tools

- The scratchpad — `python_repl` when the run fixes `python`, else `run_code`. It runs a program through the grading sandbox, standard library included, on the `stdin` the call supplies (empty by default), so the model can feed it the statement's sample input or its own; it never sees the graded tests. Each call is one-shot — nothing a run writes survives into the next. Past `max_test_calls` a call is refused.
- `submit_solution` — grades a complete stdin/stdout program against the hidden tests. The only graded channel, with no fenced-code-block fallback. Reaching `max_submissions` ends the episode.

A refused call is a tool error: it pays `tool_error_penalty`, never `tool_success_reward`. With a
language list both tools take a required `language` argument enumerating the set, each program is
graded in the language its call names, and a foreign value is refused before admission. The episode
records the last language as its `language` slice, which the trainer slices metrics by
([Logged metrics](../async-grpo/monitoring.md#logged-metrics)).

## Reward

### Grading rules

Grading goes through `grade_solution` (`src/environments/envs/tasks/coding/grading.py`), shared with
the offline re-grader, so a checkpoint scores identically online and offline. The verdict lists
non-passing tests only, capped at five. One sandbox session serves the whole grade, so a compiled
submission builds once, reset after every test. A compile failure is graded once against the whole
pool.

- **Comparison.** Byte-exact equality spuriously fails correct Codeforces solutions, hence the `codeforces` preset. Token comparison accepts real-valued tokens within a 1e-6 relative tolerance, gated on a float-looking *expected* token, so integer answers stay exact.
- **Verdict detail.** Under `full`, a second submission turns the judge into a free test oracle — probing out-earns scratchpad testing within a group. The recipes use `outcome`.
- **Time limits.** The payload's `time_limit` is the per-test cap, else `timeout_per_test`. An interpreted language is floored at `timeout_per_test`, so a C++-tuned limit cannot fail a correct CPython solution; a compiled one is scaled by `compiled_time_limit_scale`. Both are clamped to `max_time_limit`, per graded language.
- **Grading budget.** Tests run sequentially, so a several-hundred-test problem stalls the round. `max_grading_seconds` is checked between tests and keeps the full pool as denominator — an ungraded test counts as failed, so size it for an honest solution (the recipes: 150 s). `episode/tests_graded_frac` shows a partial grade.
- **Special judges.** A per-problem `checker` (Python) in the payload overrides comparison: `python checker.py input.txt correct_output.txt solution_output.txt`, accepted only when it exits cleanly and its last stdout token is `1`. It runs at the 15 s infra default, never the solution's limit.
- **Infra errors.** A grade that hit a backend error with no test running cleanly or passing marks the episode invalid, so the trainer drops it from the group baseline rather than teaching a wrong answer (`episode/grading_infra_outage`).

### Reward ladder

The grade is the pass fraction `tests_passed / tests_total` of the submitted solution, priced by the
reward's `environment` term as `weight × fraction ^ exponent` (`rewards:` above). It is credited only
on `submit_solution`: an unsubmitted solution, a zero-test row and an infra outage all grade 0, and
the outage also marks the episode invalid. No shaping rung pays out on those either; the
resubmission penalty and the tool shaping still apply.

| Component | Knob | Default | Pays |
|---|---|---|---|
| `reward/objective` | `rewards:` `weight` / `exponent` | `1.0` / `1.0` | the pass fraction, priced by the environment term |
| `reward/submission` | `submission_reward` | `0` | once a contentful graded submission lands |
| `reward/execution` | `execution_progress_reward` | `0` | × the fraction of the whole test pool that ran cleanly |
| `reward/tested_submission` | `tested_submission_reward` (a profile key) | unset | once, if a scratchpad run preceded the first submission |
| `reward/resubmission` | `resubmission_penalty` | `0` | −1 × each admitted `submit_solution` call after the first |
| `reward/tool_shaping` | `multi_turn_reward` | `0` | >1 tool call and a real submission |
| `reward/tool_shaping` | `no_tool_use_penalty` / `turn_overflow_penalty` | `0` | zero tool calls / burning `max_turns` |
| `reward/turn_shaping` | `tool_success_reward` / `tool_error_penalty` | `0` / `0` | per executed call; this env zeroes the protocol's 0.05 / 0.1 |

The shaping rungs bootstrap a weak base that never submits, and self-neutralize within a group once
every completion reaches them — keep each small next to the objective's weight. The execution rung
is the anti-sparsity signal: where every completion fails, it separates runnable-but-wrong from
crashes. Components log as `reward/*` and sum exactly to the reward. A `judge` or `reward_model`
term reads the submitted program as a fenced code block, not the tool-call turn that carried it
([Reward Terms](../rewards.md#environment-arm)).

Behavior counters ride alongside: `episode/submission_rate`, `episode/test_calls`,
`episode/tested_before_submission` (over submitting episodes, the rate the tested-submission bonus
targets), `episode/grading_budget_hit`, and `episode/language_switches` where the model picks the
language.

## Dataset

`prompt` is the statement; `answer` the grading payload, a JSON string or dict — required
(`requires_answer`), since the payload IS the test set a submission is graded against. A bare
`{"test_cases": [...]}` and the full form are both accepted:

```json
{"prompt": "<problem statement>",
 "answer": "{\"tests\": [{\"input\": \"2 3\\n\", \"output\": \"5\\n\"}], \"checker\": null, \"time_limit\": 2.0}"}
```

Adapters (`src/environments/envs/tasks/coding/datasets.py`) map a source's rows into that shape:

| Adapter | Source | Role | Notes |
|---|---|---|---|
| `codeforces` | `open-r1/codeforces` | RL pool | `verifiable` config; `generated_checker` judges; interactive rows dropped; generated tests join via `--tests_table` |
| `hardtests` | `sigcp/hardtests_problems` + `_tests` | RL pool | Difficulty mapped to Codeforces ratings; needs `--tests_table`; judging function becomes the checker |
| `deepcoder` | `agentica-org/DeepCoder-Preview-Dataset` | RL pool | stdin/stdout tests; functional specs skipped; no report bucket |
| `livecodebench` | `livecodebench/code_generation_lite` | benchmark | Release `test*.jsonl` read directly, newest first; functional rows skipped |
| `icpc` | `RUC-AIBOX/ICPC-Eval` | benchmark | Streamed; `traditional` graded, `spj` skipped |
| `hlce` | `HumanLastCodeExam/icpc-world-finals` | benchmark | Streamed; stdin/stdout `test_cases` |

`scripts/environments/preparation/prepare_code_dataset.py` builds a training pool from the three RL
adapters:

```bash
python scripts/environments/preparation/prepare_code_dataset.py \
    --adapter hardtests --dataset sigcp/hardtests_problems --min_rating 800 \
    --tests_table "$HALO_DATA_ROOT/s3_datasets/hardtests-tests-compact" \
    --holdout_per_band 100 --push_to_hub org/hardtests-rl --push_bands
```

It composes the statement, packs the payload, and drops rows this environment cannot grade.
`--min_rating` / `--max_rating` bound difficulty, dropping unrated rows with them, `--exclude_keys`
removes listed ids, and `--holdout_per_band` carves a deterministic `test` split.

`--push_bands` publishes `full` plus one config per rating band (`medium` 1500-1999, `hard`
2000-2599, `extra-hard` 2600-3500) over a shared test split, which is how a curriculum stage selects
its pool (`org/name:hard`). `--verify_checkers`, on by default, drops a problem whose special judge
rejects its own reference output or accepts garbage; it runs them through a sandbox, so the
preparation host needs a backend.

A bulky test corpus goes through `compact_code_tests.py` first: it reduces open-r1's generated tests
or HardTests' encoded suites to one capped row per problem (40 tests within 256 KB, two of which may
reach 4 MB so a maximum-size input survives), which `--tests_table` joins by problem id.

## Evaluation

```bash
python scripts/environments/inference/run_code_contests.py --adapter codeforces \
    --dataset open-r1/codeforces --config verifiable --split test --language python \
    --base_url http://localhost:8000/v1 --model <served-name> \
    --num_examples 100 --num_samples 4 --reasoning_effort high
```

It buckets `success@1` / `success@k` by the adapter's field (rating here); at the default
`--success_threshold` a problem counts solved only when every test in the pool passes.

Without `--training_config` or `--max_tokens`, `--reasoning_effort` sets the generation budget: the
level's `thinking_tokens` plus 4096 tokens of solution headroom, which the served context window
must exceed. Grading knobs with no flag go through `--env_kwargs`, recorded in the trajectory meta.
Flags, output files and re-grading: [Evaluating on an Environment](evaluation.md).

## Related pages

- [Sandboxes](sandbox.md) — backends, languages, concurrency
- [Async GRPO with Environments](../async-grpo/README.md) — trainer and servers
