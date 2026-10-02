# Code Contests Environment

`CodeContestsEnvironment` (`src/environments/envs/tasks/coding/code_contests.py`) trains
competitive programming: the model writes a solution, tries it in a scratchpad tool, and submits it
with `submit_solution`, which runs it against the hidden tests through a [sandbox](sandbox.md).
The grade is 1 when the submitted solution passes every hidden test and 0 otherwise, priced by the
reward's `environment` term. Two registry names share the class — `code_contests`
(`output_comparison: exact`) and `codeforces` (token comparison).

It speaks native tool calls, so the server needs a tool-call parser for the model family
([Rollout Configuration](../async-grpo/rollouts.md#tool-calls)). Recipes ship under
`examples/grpo/environmental/<family>/<backend>/`; the canonical one is
`examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-lora-ep1.yaml`.

## Configuration

```yaml
environment_type: codeforces
max_turns: 16                # 16 in most recipes, 18 in the three curriculum recipes; the class default is 15
rewards:
  - source: environment      # 1 when the submitted solution passes every hidden test, else 0
environment_kwargs:
  language: python           # or cpp / c, or a list ([python, cpp]) the model picks from
  timeout_per_test: 5
  max_grading_seconds: 150
  reasoning_effort: random
  reasoning_effort_profiles:   # thinking_tokens is the episode's total under the scope below
    low: {thinking_tokens: 24576, max_submissions: 1, max_test_calls: 2}
    medium: {thinking_tokens: 32768, max_submissions: 2, max_test_calls: 4}
    high: {thinking_tokens: 36000, max_submissions: 3, max_test_calls: 6}
rollout_max_thinking_tokens: 18000     # the most one turn may reason of it
rollout_thinking_budget_scope: episode
```

| Knob | Default | Effect |
|---|---|---|
| `language` | `python` | `python`, `cpp`, `c`, or a list the model picks from |
| `output_comparison` | `exact` (`tokens` under `codeforces`) | `exact` is trimmed equality reading `\r\n` and `\r` as `\n` on both sides, `tokens` whitespace-token equality |
| `verdict_detail` | `outcome` | `outcome` states a failed test's verdict class alone (the compiler's first error too, on `bubblewrap`); `full` adds its expected and produced output, exit status or signal, output size and stderr |
| `timeout_per_test` | 15 s | Per-test cap when the problem declares none; also the interpreted floor. It and `max_time_limit` must be finite and > 0 |
| `max_time_limit` | 15 s | Clamp on a declared limit; below `timeout_per_test` it is refused |
| `compiled_time_limit_scale` | `1.0` | Multiplies a compiled language's per-test limit; a non-finite or non-positive value raises at construction |
| `max_grading_seconds` | `None` | Wall-clock budget for one grade; a non-positive value raises at construction |
| `max_output_size` | 1 MB | Over-cap stdout is OUTPUT LIMIT EXCEEDED, not truncated; a test's cap rises to 4× its expected output, so a large correct answer passes |
| `stop_on_first_failure` | `false` | Stop at the first failing test; the grade is unchanged, `outcome/test_pass_frac` becomes a lower bound |
| `max_submissions` / `max_test_calls` | 2 / 5 | Per-episode tool budgets, overridable per effort level |
| `max_starved_run_refunds` | 1 | Input-less scratchpad runs per episode that print nothing and are returned to the budget; later ones count. `0` returns none |
| `max_turns` | 15 | Backstop; the tool budgets are the tuning lever |
| `eval_protocol` | `harness` | Evaluation contract; `leaderboard` pins both tool budgets ([Evaluation protocols](#evaluation-protocols)) |

The grade is all-or-nothing, matching the accept verdict pass@1 counts: partial credit would pay a brute force that passes the small tests and times out on the large ones. The `environment` term's `weight` prices a solve; its `exponent` has nothing to reshape ([Reward Terms](../rewards.md)).

`sandbox_backend` / `sandbox_url` pick the [sandbox](sandbox.md#choosing-a-backend) both tools and the grader run on; one that does not confine the program, `local` included, [warns](sandbox.md#choosing-a-backend).

### Reasoning effort

`reasoning_effort` defaults to `medium` here, and the class ladder sets `thinking_tokens` only: low
4096, medium 8192, high 16384 — per turn, or the episode's total under
`rollout_thinking_budget_scope: episode`, which then scales the floor's reference
([Reasoning budget](../async-grpo/rollouts.md#reasoning-budget)). `reasoning_effort_profiles` merges
per level over it, so a profile naming only interaction keys keeps the class budget. The level and its budget reach the
model through the chat template; on Qwen3.6 that is the shipped effort template the recipes pin.

This environment adds two profile keys, bound per episode: `max_submissions` (int ≥ 1) and
`max_test_calls` (int ≥ 0), the episode's tool budgets, stamped at reset and stated in the task
message. They make effort buy iteration, not just longer reasoning; without them the strategy
collapses to submit-and-fix.

A value below its minimum raises at construction; where the level is undetermined at reset, the
constructor's budgets stand.

### Evaluation protocols

`eval_protocol` names the contract a run is scored under (`EVAL_PROTOCOLS` in `code_contests.py`):

- `harness` (default) pins nothing: the configured budgets stand. This is the agentic loop the recipes train, and its solve rate is attempts-until-accept within the budget.
- `leaderboard` pins `max_submissions: 1` and `max_test_calls: 0`: one graded program per sample, the scratchpad refused. The prompt, tool-call format and grader stay this environment's, so its numbers are not a benchmark's published ones.

A configured value that contradicts a pin raises at construction. An effort profile's
`max_submissions` / `max_test_calls` are validated, then give way to the pins (logged): the level
keeps its `thinking_tokens`, and a ladder that binds interaction still states the budgets in the task
message ("1 graded submission, 0 scratchpad runs").

## Tools

- The scratchpad — `python_repl` when the run fixes `python`, else `run_code`. It runs a program through the grading sandbox, standard library included, on the `stdin` the call supplies (empty by default), so the model can feed it the statement's sample input or its own; it never sees the graded tests. Each call is one-shot — nothing a run writes survives into the next. Past `max_test_calls` a call is refused, and the refusal names the graded submissions left.
- `submit_solution` — grades a complete stdin/stdout program against the hidden tests. The only graded channel, with no fenced-code-block fallback. Reaching `max_submissions` ends the episode.

A scratchpad run gets the per-test time limit its language is graded at ([Grading rules](#grading-rules)), and a timeout says so. Its reply leads with any error — the compiler's first diagnostics, or a crash's signal and stderr tail — ahead of the program's stdout ([Sandboxes](sandbox.md#using-it-from-python)). A run with no `stdin` that crashes adds a note naming the missing input, and one that exits cleanly having printed nothing is returned to the budget with a note saying so, since it told the model nothing, up to `max_starved_run_refunds` per episode; past that cap such a run counts and gets the missing-input note (neither note follows a compile failure or a timeout); every reply ends with the runs left while `max_test_calls` binds (`Scratchpad runs left: N of M.`).

A refused call is a tool error: it pays `tool_error_penalty`, never `tool_success_reward`. A
scratchpad run that ends on a sandbox fault ends the episode ([Sandbox faults](sandbox.md#sandbox-faults)). With a
language list both tools take a required `language` argument enumerating the set, each program is
graded in the language its call names, and a foreign value is refused before admission. The episode
records the last language as its `language` slice, which the trainer slices metrics by
([Logged metrics](../async-grpo/monitoring.md#logged-metrics)).

A final text answer ends the episode ungraded, so the recovery nudge after a cut or empty turn asks for a tool call and names `submit_solution`, where the protocol's empty-turn nudge offers a final answer. A turn that reaches its length limit while writing its call is told so and asked to make the call again with its reasoning kept out of the program's comments.

## Reward

### Grading rules

Grading goes through `grade_solution` (`src/environments/envs/tasks/coding/grading.py`), shared with
the offline re-grader, so a checkpoint scores identically online and offline. The verdict lists
non-passing tests only, one entry per distinct verdict with the tests that failed the same way folded
into it, capped at five. One sandbox session serves the whole grade, so a compiled submission builds
once, reset after every test. A compile failure is graded once against the whole pool and shows the
compiler's first error under `full`. Under `outcome` it shows it only on `bubblewrap`, whose build
runs once before any test, on an empty stdin, and whose program can force no rebuild. On `local` and
`remote` it shows the class alone: a `local` program can write a test's input to a host file and
remove its working directory, forcing a rebuild that includes the file, and a `remote` build shares
each test's request with its stdin.

- **Comparison.** `exact` comparison spuriously fails correct Codeforces solutions, hence the `codeforces` preset. Token comparison accepts real-valued tokens within `1e-6 × max(1, |expected|)` (absolute below 1, relative above), gated on a float-looking *expected* token, so integer answers stay exact.
- **Verdict detail.** `outcome` shows each failed test's verdict class (`FAIL`, `RUNTIME ERROR`, `TIME LIMIT EXCEEDED`, `OUTPUT LIMIT EXCEEDED`, `COMPILATION ERROR`, and `ERROR` for a test lost to infra, whose text goes to the log) and nothing beyond it (compile errors: above): stderr, an exit status or signal and an output size can each carry the hidden input the program read, and the output cap, which rises with the expected output, would reveal its size. Which tests fail, and with which class, still reaches the policy; `stop_on_first_failure` narrows that to the first failing test, the Codeforces contract. `full` adds them (a runtime error's signal or exit status, stderr as its tail, where a traceback names the exception, and an output-limit overrun's size against its cap), an infra error's text and a wrong answer's expected and produced output; a second submission then turns the judge into a free test oracle, and probing out-earns scratchpad testing within a group. Scratchpad runs on the model's own inputs show their output in both modes.
- **Time limits.** The payload's `time_limit` is the per-test cap, else `timeout_per_test`. An interpreted language is floored at `timeout_per_test`, so a C++-tuned limit cannot fail a correct CPython solution; a compiled one is scaled by `compiled_time_limit_scale`. Both are clamped to `max_time_limit`, per graded language (`GradingSpec.time_limit_for`, which the scratchpad shares). A compiled program's stack is raised to the memory limit, as on Codeforces, up to the process's hard stack limit ([Limits](sandbox.md#limits)).
- **Grading budget.** Tests run sequentially, so a several-hundred-test problem stalls the round. `max_grading_seconds` is checked between tests, and an ungraded test never counts as passed, so size it for an honest solution (the recipes: 150 s). A grade the budget stopped after a failed test is a wrong answer; one it stopped before any test failed says nothing about the code, so it grades 0 and marks the episode invalid, out of the group baseline (`episode/grade_inconclusive`). `episode/tests_graded_frac` shows a partial grade.
- **Special judges.** A per-problem `checker` (Python) in the payload overrides comparison: `python checker.py input.txt correct_output.txt solution_output.txt`, accepted only when it exits cleanly and its last stdout token is `1`. It runs at the 15 s infra default, never the solution's limit.
- **Infra errors.** A grade that lost tests to the backend with none passed and none failed marks the episode invalid, so the trainer drops it from the group baseline rather than teaching a wrong answer (`episode/grading_infra_outage`). Tests lost to the backend beside passes and no failure leave the grade inconclusive, handled like a budget stop. A build past the compile limit is a compile error, a program that replaces its working directory a runtime error, one that floods its output an output-limit or runtime error, and one that removes its working directory runs the next test in a fresh one: verdicts, not infra. The routes a program still has into an infra error are listed under [Sandbox faults](sandbox.md#sandbox-faults).

### Reward ladder

The grade is 1 when the submitted solution passes every hidden test and 0 otherwise, priced by the
reward's `environment` term (`rewards:` above). It is credited only on `submit_solution` and read off
the last submission: an unsubmitted solution, an infra outage and an inconclusive grade all grade
0, and the last two also mark the episode invalid. No shaping rung pays out on those
either; the resubmission penalty and the tool shaping still apply.

| Component | Knob | Default | Pays |
|---|---|---|---|
| `reward/objective` | `rewards:` `weight` | `1.0` | a solve: every hidden test passed |
| `reward/submission` | `submission_reward` | `0` | once a contentful graded submission lands |
| `reward/resubmission` | `resubmission_penalty` | `0` | `−resubmission_penalty` per admitted `submit_solution` call after the first |
| `reward/tool_shaping` | `no_tool_use_penalty` / `turn_overflow_penalty` | `0` | zero tool calls / burning `max_turns` |
| `reward/tool_shaping` | `length_cutoff_penalty` | `0` | per engine-cut or empty turn the episode recovers from |
| `reward/turn_shaping` | `tool_success_reward` / `tool_error_penalty` | `0` / `0` | per executed call; this env zeroes the protocol's 0.05 / 0.1 |

The shaping rungs bootstrap a weak base that never submits, and self-neutralize within a group once
every completion reaches them — keep each small next to the objective's weight. Components log as
`reward/*` and sum exactly to the reward. A `judge` or `reward_model`
term reads the submitted program as a fenced code block, not the tool-call turn that carried it
([Reward Terms](../rewards.md#environment-arm)).

Each graded submission after the first is a probe of the judge, priced flat whether it fixes or
re-rolls; where the effort profiles set the budgets, the task message states the price beside them,
so not resubmitting is an option the policy can weigh. `episode/resubmission_improved` is the share of
resubmissions whose pass fraction beat every earlier one.

The trainer's two effort length terms sit outside these components, as `reward/effort_length_penalty`
and `reward/effort_length_floor` ([Effort length reward](../async-grpo/rollouts.md#effort-length-reward));
the floor's reference is `effort_length_floor_budgets` × each level's own `thinking_tokens` (`0.75` by
default, `0.375` in the episode-scope Qwen3.6 vLLM recipes). The recipes keep their sum
under the resubmission price, so how long an episode reasons never outweighs whether it resubmits;
with `submission_reward` and `no_tool_use_penalty` at `0.1` each, a graded submission that passes
nothing still scores above an episode that never attempts, and at every level a solve that pays
every per-episode price (the level's resubmissions, the recoveries the cap admits, the overflow and
the length terms) out-scores any zero-objective episode; each failed tool call adds
`tool_error_penalty` on top.
`tests/cpu/config/test_env_grpo_reward_economy.py` holds the shipped recipes to those relations.

Behavior counters ride alongside: `episode/submission_rate`, `episode/test_calls` (runs that counted),
`episode/starved_test_runs` (runs returned for having no input and no output),
`episode/tested_before_submission` (over submitting episodes), `episode/grading_budget_hit`, and
`episode/language_switches` where the model picks the language.

## Dataset

`prompt` is the statement; `answer` the grading payload, a JSON string or dict — required
(`requires_answer`), since the payload IS the test set a submission is graded against. A bare
list, `{"test_cases": [...]}` and the full form are accepted. A payload that holds no tests (an empty
list, no non-empty `tests`/`test_cases`, unparseable JSON, a scalar) fails the episode at reset as a
rollout error, so the trainer drops it from the group baseline:

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
| `livecodebench` | `livecodebench/code_generation_lite` | benchmark | Release `test*.jsonl` read directly, newest file first; functional rows skipped; contest-date window and platform filter |
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

It buckets `success@1` / `success@k` by the adapter's field (rating here; neither is a benchmark's
mean-over-samples pass@1, see [Evaluating on an Environment](evaluation.md#running-an-evaluation)).
A problem counts solved when the submitted program passes every test in the pool — the environment's
verdict, not the shaped total, so the recipe's shaping under `--training_config` (a
`tool_error_penalty` on every refused scratchpad call under `leaderboard`, a submission bonus) moves
it neither way, and the coding CLI takes no `--success_threshold`. The verdict is the episode's last
graded submission and `success@1` each row's first scored sample, while the
[re-grader](evaluation.md#re-grading-recorded-trajectories) scores every episode on its first
submission (`s@1`) or any within its budget (`s@2`). The two score the same submission only on a
`leaderboard` run at `--num_samples 1`, where each row is one episode with one submission.

Without `--training_config` or `--max_tokens`, `--reasoning_effort` sets the generation budget: the
level's `thinking_tokens` as the environment binds it (a `reasoning_effort_profiles` override in
`--env_kwargs` included) plus 4096 tokens of solution headroom, which the served context window
must exceed. Every episode sends its level's thinking budget, so the vLLM server needs what the
[reasoning budget](../async-grpo/rollouts.md#reasoning-budget) needs, a reasoning parser among it;
without one vLLM rejects the request. A non-thinking model served with a think-tag parser gets its
whole answer back as reasoning (no end marker reads as all reasoning), so evaluate one with
`--reasoning_effort none`: no level, no budget, no parser needed, and a default `--max_tokens` of
32768, the training rollout's. Under `--training_config` the level defaults to the YAML's, where a
`reasoning_effort: null` is `none`; only a YAML without the key takes `medium`, as training does.
The eval knows the server is SGLang only from a `--training_config` naming `rollout_backend: sglang`,
which drops the budget as [training does](../async-grpo/rollouts.md#reasoning-budget).

`--eval_protocol` picks the [evaluation protocol](#evaluation-protocols), else the training
config's, else `harness`; the report title and the trajectory meta name it. A training config
written under another protocol gives up the `max_submissions` / `max_test_calls` the flag's
protocol pins (logged); a config that names the protocol itself, or `--env_kwargs`, contradicting a
pin raises. Grading knobs with no flag go through `--env_kwargs`, recorded in the trajectory meta.
Flags, output files and re-grading: [Evaluating on an Environment](evaluation.md).

### LiveCodeBench window

A LiveCodeBench release is cumulative — every problem since May 2023 — so most of it predates a
current model's training cutoff. Score a window after it:

```bash
python scripts/environments/inference/run_code_contests.py --adapter livecodebench \
    --dataset livecodebench/code_generation_lite --config release_v6 \
    --start_date 2025-01-01 --end_date 2025-04-30 --platform atcoder,codeforces \
    --eval_protocol leaderboard --base_url http://localhost:8000/v1 --model <served-name> \
    --num_examples 0 --num_samples 4
```

- `--start_date` / `--end_date` (`YYYY-MM-DD`) are both inclusive and compare the row's `contest_date` by calendar day; either may stay open.
- `--platform` takes the platforms this adapter grades, as the dataset spells them: `atcoder`, `codeforces`. LeetCode rows are functional, which the stdin/stdout environment cannot grade, so `leetcode` is refused.
- Rows come newest release file first, each file in its stored order, which is not newest-first (`test6.jsonl` opens on its 2025-01-04 contests). `--num_examples` (default 50) takes the first problems of the window in that order, not its newest; `0` scores the whole window.
- Any other date spelling, a start after the end, a platform outside that list, or a window or platform filter on an adapter that declares none exits before a row is read. A row without a parsable `contest_date` raises under a window.

The selection is recorded in the trajectory meta ([re-grading](evaluation.md#re-grading-recorded-trajectories)).
Only `livecodebench` declares a contest date and platform (`contest_date`, `platform_field` and
`platforms` on its `CodeDatasetAdapter`).

## Related pages

- [Sandboxes](sandbox.md) — backends, languages, concurrency
- [Async GRPO with Environments](../async-grpo/README.md) — trainer and servers
