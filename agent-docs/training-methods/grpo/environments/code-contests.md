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
environment_kwargs:
  language: python           # or cpp / c, or a list ([python, cpp]) the model picks from
  timeout_per_test: 5
  max_grading_seconds: 150
  reasoning_effort: random
  reasoning_effort_profiles:   # thinking_tokens is the level's per-turn reasoning cap (vLLM enforces it)
    low: {thinking_tokens: 8192, max_submissions: 1, max_test_calls: 2}
    medium: {thinking_tokens: 12288, max_submissions: 2, max_test_calls: 4}
    high: {thinking_tokens: 16384, max_submissions: 3, max_test_calls: 6}
rollout_max_tokens: 30000              # per-turn ceiling
rollout_max_answer_tokens: 8192        # the most a turn generates past its reasoning cap; never stated to the model
rollout_max_episode_tokens: 131072     # the most an episode may sample over all its turns; never stated to the model
reasoning_floor: 0.05                  # an episode reasoning under 0.75x its level's per-turn budget pays up to this
rewards:
  - source: environment      # 1 when the submitted solution passes every hidden test, else 0
  - source: judge            # a veto on what no counter can judge
    name: audit
    model: openai/gpt-6-luna
    view: full                 # every turn, the reasoning included
    max_view_chars: 900000
    context: "..."             # how the environment and its limits work, which the episode does not show
    checks:
      - {name: reasoning_in_actions, description: "...", veto: true}
      - {name: notepad_run, description: "...", veto: true}
      - {name: hardcoded_output, description: "...", veto: true}
      - {name: environment_probe, description: "...", veto: true}
      - {name: verdict_probe, description: "...", veto: true}
      - {name: verdict_mining, description: "...", veto: true}
```

| Knob | Default | Effect |
|---|---|---|
| `language` | `python` | `python`, `cpp`, `c`, or a list the model picks from |
| `output_comparison` | `exact` (`tokens` under `codeforces`) | `exact` is trimmed equality reading `\r\n` and `\r` as `\n` on both sides, `tokens` whitespace-token equality |
| `verdict_detail` | `outcome` | `outcome` states a failed test's verdict class alone (the compiler's first error too, on `bubblewrap`, and a Python syntax error's on `local` and `bubblewrap`); `full` adds its expected and produced output, exit status or signal, output size and stderr |
| `timeout_per_test` | 15 s | Per-test cap when the problem declares none; also the interpreted floor. It and `max_time_limit` must be finite and > 0 |
| `max_time_limit` | 15 s | Clamp on a declared limit; below `timeout_per_test` it is refused |
| `compiled_time_limit_scale` | `1.0` | Multiplies a compiled language's per-test limit; a non-finite or non-positive value raises at construction |
| `max_grading_seconds` | `None` | Wall-clock budget for one grade; a non-positive value raises at construction |
| `max_output_size` | 1 MB | Over-cap stdout is OUTPUT LIMIT EXCEEDED, not truncated; a test's cap rises to 4× its expected output, so a large correct answer passes |
| `stop_on_first_failure` | `false` | Stop at the first failing test; the grade is unchanged, `outcome/test_pass_frac` becomes a lower bound |
| `max_submissions` / `max_test_calls` | 2 / 5 | Per-episode tool budgets, overridable per effort level; enforced per call, never stated to the model |
| `max_turns` | 15 | Backstop; the tool budgets are the tuning lever |
| `eval_protocol` | `harness` | Evaluation contract; `leaderboard` pins both tool budgets ([Evaluation protocols](#evaluation-protocols)) |

The grade is all-or-nothing, matching the accept verdict pass@1 counts: partial credit would pay a brute force that passes the small tests and times out on the large ones. The `environment` term's `weight` prices a solve; its `exponent` has nothing to reshape ([Reward Terms](../rewards.md)).

`sandbox_backend` / `sandbox_url` pick the [sandbox](sandbox.md#choosing-a-backend) both tools and the grader run on; one that does not confine the program, `local` included, [warns](sandbox.md#choosing-a-backend).

Every shipped code-contests recipe sets `rollout_max_episode_tokens: 131072`, the most an episode may sample over
all its turns. Without it a recipe admits `max_turns × rollout_max_tokens` (16 × 30,000 = 480,000
tokens; 540,000 in the curriculum recipes at `max_turns: 18`)
([Trajectory length](../async-grpo/rollouts.md#trajectory-length)). Each vLLM recipe also sets
`rollout_max_answer_tokens: 8192`, so a turn runs at its reasoning cap plus 8,192 tokens (16,384–24,576 by level,
the retry reserve plus 8,192 on a retry): a turn the cap closes cannot carry its reasoning on in a program's comments
through the rest of a 30,000-token turn, and is cut instead ([Reasoning budget](../async-grpo/rollouts.md#reasoning-budget)).
The Qwen3.6 vLLM recipes also stop a turn at `<|im_start|>` (`rollout_stop_tokens`): past a forced close the model can
write a chat turn's start, and vLLM would read the `<think>` after it as a fresh reasoning budget.

### Reasoning effort

`reasoning_effort` defaults to `medium` here, and the class ladder sets `thinking_tokens` only: low
4096, medium 8192, high 16384, the level's per-turn reasoning cap
([Reasoning budget](../async-grpo/rollouts.md#reasoning-budget)); every shipped recipe sets
8192 / 12288 / 16384. `reasoning_effort_profiles` merges per level over the ladder, so a profile
naming only interaction keys keeps the class budget. The level and its per-turn budget reach the
model through the chat template; on Qwen3.6 and Gemma 4 that is the shipped effort template the
recipes pin.

This environment adds two profile keys, bound per episode: `max_submissions` (int ≥ 1) and
`max_test_calls` (int ≥ 0), the episode's tool budgets, stamped at reset and enforced per call. They
make effort buy iteration, not just longer reasoning; without them the strategy collapses to
submit-and-fix. Neither is stated to the model: what an episode may do is set by the level and
thinking budget the chat template states and by the engine's caps, and a call past its budget is
refused with what to do, never with a number ([Tools](#tools)).

A value below its minimum raises at construction; where the level is undetermined at reset, the
constructor's budgets stand.

### Evaluation protocols

`eval_protocol` names the contract a run is scored under (`EVAL_PROTOCOLS` in `code_contests.py`):

- `harness` (default) pins nothing: the configured budgets stand. This is the agentic loop the recipes train, and its solve rate is attempts-until-accept within the budget.
- `leaderboard` pins `max_submissions: 1` and `max_test_calls: 0`: one graded program per sample, the scratchpad closed. The prompt, tool-call format and grader stay this environment's, so its numbers are not a benchmark's published ones.

A configured value that contradicts a pin raises at construction. An effort profile's
`max_submissions` / `max_test_calls` are validated, then give way to the pins (logged); the level
keeps its `thinking_tokens`. The task message states no budget under either protocol: under
`leaderboard` the model learns the scratchpad is closed from its reply alone. A call to it is a tool error but no
refusal (`ToolDisabled`): its turn is not flagged and the turn after it gets no retry reserve, which would buy the
protocol more output per sample than the setting it reproduces ([Tools](#tools)).

## Tools

- The scratchpad — `python_repl` when the run fixes `python`, else `run_code`. It runs a program through the grading sandbox, standard library included, on the `stdin` the call supplies (empty by default), so the model can feed it the statement's sample input or its own; it never sees the graded tests. Each call is one-shot — nothing a run writes survives into the next. Past `max_test_calls` a call is refused: `Error: Not run: this task's scratchpad budget is spent. Submit your solution with submit_solution.`
- `submit_solution` — grades a complete stdin/stdout program against the hidden tests. The only graded channel, with no fenced-code-block fallback. A submission that passes every hidden test ends the episode, as reaching `max_submissions` does: past an accept a resubmission can only lose the solve, so one later in the same turn is not graded (`Not graded: an earlier submission already passed every test, …`), and one past the budget in the same turn is refused (`Error: Not graded: this task's submission budget is spent.`). Its description says a passing submission ends the task and the last graded one otherwise counts.

- `submit_solution` also refuses a program identical to one this episode already graded (comments, by the language's registered syntax, trailing whitespace and blank lines aside) unrun, the submission returned to its budget: `Not graded: this program is identical to one already graded and would receive the same verdict. Change it before submitting again.` It would only probe the judge; the turn is flagged like any refusal and the offline re-grader takes no slot for it (`episode/identical_resubmissions`).

Neither description states a budget, no reply counts what is left of one, and a refusal names what to do, never a number.

Both tool descriptions name the toolchain where the sandbox states it (`SandboxExecutor.toolchain`): on `local` and `bubblewrap` the registry's compile flags and the interpreter a Python program runs on (`Here python runs on CPython 3.12 and cpp is compiled with g++ -O2 -pipe -std=c++17.`); `remote` states none.

A scratchpad run gets the per-test time limit its language is graded at ([Grading rules](#grading-rules)), and a timeout says so. Its reply leads with any error — the compiler's first diagnostics, or a crash's signal and stderr tail — ahead of the program's stdout ([Sandboxes](sandbox.md#using-it-from-python)); a clean exit's reply is its stdout alone. A run with no `stdin`, or whitespace alone, adds a note on a line of its own that none was passed: neither a parse error, nor output computed from nothing, nor a clean exit with nothing on stdout names the cause. It spends its run and is booked like any other. A build failure — a compile error, or Python source that does not compile (on `local` and `bubblewrap`) — ran nothing and gets no note. An empty `stdin` is never refused: a self-test that embeds its input is a real use.

No tool refuses or prices a program for its comments, or a run for what it showed: reasoning carried into a program and scratchpad runs used as a notepad are the run's `judge` term's to handle ([Reward Terms](../rewards.md)).

Output that would push a reply past `max_observation_chars` is cut (`…[truncated N chars]`) so the notes after it survive the protocol's cap, which cuts from the end.

A refused call — past its budget, with arguments that do not bind, or an identical resubmission — is
a tool error: it pays `tool_error_penalty`, never `tool_success_reward`, and a turn whose every call was
refused or unknown is flagged untrainable, the turn after it running on the retry reserve. A call to a
tool whose budget is 0 (the `leaderboard` scratchpad, a level's `max_test_calls: 0`) gets the same reply and price
but is no refusal: its turn is not flagged. A
scratchpad run that ends on a sandbox fault ends the episode ([Sandbox faults](sandbox.md#sandbox-faults)). With a
language list both tools take a required `language` argument enumerating the set, each program is
graded in the language its call names, and a foreign value is refused before admission. A run that
fixes one language declares no `language` argument, so a call naming one is refused like any argument
a tool does not declare: the model reads `Error: <tool>: unknown argument 'language'; its arguments are …`,
and the call is a refused call as above, spending its turn but none of `max_test_calls` or
`max_submissions`; a refused `submit_solution` grades nothing. A list or object as `code` or `stdin` is
refused the same way (`Error: <tool>: code must be a string, got list`); a JSON number or boolean reads
as its Python string (`"stdin": 5` feeds the program `5`, `true` feeds `True`), and a null `stdin` is no stdin. The episode records the last language
as its `language` slice, which the trainer slices metrics by
([Logged metrics](../async-grpo/monitoring.md#logged-metrics)).

A call its handler returns to the budget — a program sent with the wrong language (below), a
submission after an accept — is neither paid nor charged: it counts in `total_tool_calls`
(`episode/tool_calls`) but not as a successful call, and earns no `tool_success_reward`.

With a language list, a program evidently written in another of its languages is not run
(`evident_language`): `Not run: this looks like cpp code sent with language "python"; send it again
with language "cpp". No scratchpad run was spent.` (`Not graded` and a submission for
`submit_solution`). The call is returned to the budget as above: it counts toward neither
`max_test_calls`, `max_submissions` nor the resubmission price, records no grade, and leaves the
`language` slice as it was. The check reads the text, so a program it misses runs as labelled:

- Sent as `python`, a program is pointed at a compiled language when it does not compile as Python and
  carries a C-family mark (an `#include` line, `int main(`, `using namespace`, `std::`, or `;` ending at
  least 30% of its non-blank lines) but no Python mark (`def`, `import`, `print(`). A C++ mark
  (`using namespace`, `std::`, a header without `.h` such as `<iostream>`, `<bits/stdc++.h>`,
  `cin`/`cout`, `template<`) names `cpp`, and none when the list has no `cpp`; a C header
  (`<stdio.h>`) without one names `c`; anything else takes the list's first compiled language.
- Sent as a compiled language, a program is pointed at `python` when it compiles as Python and
  carries a Python mark but no C-family one.

Each reading needs the parse and the marks to agree, so a valid program quoting the other language
in a string or a comment keeps its label, and broken Python with C-like semicolons keeps its syntax
error.

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
each test's request with its stdin. On `local` and `bubblewrap`, whose programs run on the grader's
own interpreter, a Python source that does not compile is graded the same way without a run and shows
its first error in both modes (`SyntaxError: invalid syntax (line 3)`): the interpreter stops on it
before the program reads anything, so it quotes the source alone. `remote` may run another Python
version, so there the program runs and its failure is graded as the service reports it.

- **Comparison.** `exact` comparison spuriously fails correct Codeforces solutions, hence the `codeforces` preset. Token comparison accepts real-valued tokens within `1e-6 × max(1, |expected|)` (absolute below 1, relative above), gated on a float-looking *expected* token, so integer answers stay exact.
- **Verdict detail.** `outcome` shows each failed test's verdict class (`FAIL`, `RUNTIME ERROR`, `TIME LIMIT EXCEEDED`, `OUTPUT LIMIT EXCEEDED`, `COMPILATION ERROR`, and `ERROR` for a test lost to infra, whose text goes to the log) and nothing beyond it (compile errors: above): stderr, an exit status or signal and an output size can each carry the hidden input the program read, and the output cap, which rises with the expected output, would reveal its size. Which tests fail, and with which class, still reaches the policy; `stop_on_first_failure` narrows that to the first failing test, the Codeforces contract. `full` adds them (a runtime error's signal or exit status, stderr as its tail, where a traceback names the exception, and an output-limit overrun's size against its cap), an infra error's text and a wrong answer's expected and produced output; a second submission then turns the judge into a free test oracle, and probing out-earns scratchpad testing within a group. Scratchpad runs on the model's own inputs show their output in both modes.
- **Time limits.** The payload's `time_limit` is the per-test cap, else `timeout_per_test`. An interpreted language is floored at `timeout_per_test`, so a C++-tuned limit cannot fail a correct CPython solution; a compiled one is scaled by `compiled_time_limit_scale`. Both are clamped to `max_time_limit`, per graded language (`GradingSpec.time_limit_for`, which the scratchpad shares). A language list mixing interpreted and compiled languages states the floor, the scale and the clamp in the system prompt; a payload carrying no `time_limit` gets the limit each test runs under appended to the task message (`Each test of this problem runs under 5 s in python, 10 s in cpp.`), worded to hold whatever the statement says. On `local` / `bubblewrap` a compiled program's stack is raised to the memory limit, as on Codeforces, up to the process's hard stack limit ([Limits](sandbox.md#limits)).
- **Grading budget.** Tests run sequentially, so a several-hundred-test problem stalls the round. `max_grading_seconds` is checked between tests, and an ungraded test never counts as passed, so size it for an honest solution (the recipes: 150 s). The verdict of a stopped grade says the time ran out, so the submission is not accepted, and that a program fast enough to finish every test within it is graded in full. A grade the budget stopped after a failed test is a wrong answer; one it stopped before any test failed says nothing about the code, so it grades 0 and marks the episode invalid, out of the group baseline (`episode/grade_inconclusive`). `episode/tests_graded_frac` shows a partial grade.
- **Special judges.** A per-problem `checker` (Python) in the payload overrides comparison: `python checker.py input.txt correct_output.txt solution_output.txt`, accepted only when it exits cleanly and its last stdout token is `1`. It runs at the 15 s infra default, never the solution's limit.
- **Infra errors.** A grade that lost tests to the backend with none passed and none failed marks the episode invalid, so the trainer drops it from the group baseline rather than teaching a wrong answer (`episode/grading_infra_outage`). Tests lost to the backend beside passes and no failure leave the grade inconclusive, handled like a budget stop. A build past the compile limit is a compile error, a program that replaces its working directory a runtime error, one that floods its output an output-limit or runtime error, and one that removes its working directory runs the next test in a fresh one: verdicts, not infra. The routes a program still has into an infra error are listed under [Sandbox faults](sandbox.md#sandbox-faults).

### Reward ladder

The grade is credited only on `submit_solution` and read off the last submission: an unsubmitted
solution, an infra outage and an inconclusive grade all grade 0, and the last two also mark the
episode invalid. No shaping rung pays out on those either; the resubmission penalty and the tool
shaping still apply.

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
`reward/*` and sum exactly to the reward. A `judge` or `reward_model` term reads the submitted
program as the episode's final answer, a fenced code block — never the tool-call turn that carried
it, and never the hidden tests, which reach no scorer; `view: full` or `digest` adds the turns
([Reward Terms](../rewards.md#what-a-scorer-reads)), and a judge's also shows the unrun partial call of a
turn cut while writing it, which a veto may quote ([What a judge reads](../rewards.md#what-a-judge-reads)).

Each graded submission after the first is a probe of the judge, priced flat whether it fixes or
re-rolls. The price, like the budgets, is never stated to the model: when not to resubmit is learned
from the reward. `episode/resubmission_improved` is the share of resubmissions whose pass fraction
beat every earlier one.

The trainer's reasoning floor sits outside these components, as `reward/reasoning_floor`
([Reasoning length reward](../async-grpo/rollouts.md#reasoning-length-reward)); its reference is
three quarters of each level's own per-turn `thinking_tokens`. Every code-contests recipe runs one
economy, with no per-token reasoning price: the level's thinking caps and interaction budgets, the
8,192-token answer room past each cap (vLLM recipes; SGLang forces no close at the cap), the 131,072-token episode output budget, the floor (`0.10` on the Qwen3.6 vLLM recipes, `0.05` elsewhere),
and a `judge` veto term ([Reward Terms](../rewards.md)); it reads `OPENROUTER_API_KEY`, so pass it with `--env-file`
or drop the term to run without a judge. The judge (`openai/gpt-6-luna`) reads the whole episode, every turn's
reasoning included, beside a `context` that tells it how the environment and its caps work, and each of its six checks
strips a solve's credit:

- `reasoning_in_actions` — working-out carried into a program it ran, submitted or was writing when cut (roughly ten
  or more lines of deliberation in one program), or into its visible reply;
- `notepad_run` — a run or submission spent on an unfinished draft (stubs, an entry point never called) to close a
  turn; a complete program run without its input is not one;
- `hardcoded_output` — output the program did not compute: special-cased or sample-fitted answers, guessed
  placeholders, on any submission;
- `environment_probe` — reading the sandbox, the network or the grader instead of solving the problem;
- `verdict_probe` — a graded submission made for its verdict: a stub, a guess, a program the policy had found wrong,
  or a resubmission whose outputs cannot differ;
- `verdict_mining` — changes aimed at the failing tests rather than the method.

The cap behind `reasoning_in_actions`: where the engine closes reasoning at a one-token marker (Qwen3.6 and Gemma 4 on
vLLM), a turn whose reasoning reaches its cap is marked for the judge, and a program written past it that carries the
reasoning on is the budget escape the check exists for. Each recipe's `context` tells the judge what its policy sees
of its earlier reasoning (`carry_reasoning`) and whether its turns run under a stated, closed cap.

The floor's weight stays under the resubmission price, so how long an episode
reasons never outweighs whether it resubmits; with `submission_reward`
and `no_tool_use_penalty` at `0.1` each, a graded submission that passes nothing still scores above
an episode that never attempts, and at every level a solve that pays every per-episode price (the
level's resubmissions, the recoveries the cap admits, the overflow and the floor) out-scores any
zero-objective episode; each failed tool call adds `tool_error_penalty` on top.
`tests/cpu/config/test_env_grpo_reward_economy.py` holds the shipped recipes to those relations.

Behavior counters ride alongside: `episode/submission_rate`, `episode/test_calls` (runs that counted),
`episode/tested_before_submission` (over submitting episodes), `episode/grading_budget_hit`,
`episode/identical_resubmissions` (programs refused as already graded), and `episode/language_switches` where the
model picks the language.

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
| `codeforces` | `open-r1/codeforces` | RL pool | `verifiable` config; `generated_checker` judges; interactive rows dropped; generated tests join via `--tests_table`; marks examples-only problems |
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
    --tests_table "$HALO_DATA_ROOT/hardtests-tests-compact" \
    --holdout_per_band 100 --push_to_hub org/hardtests-rl --push_bands
```

It composes the statement, packs the payload, and drops rows this environment cannot grade.
It also drops, counted per split, the problems an adapter marks examples-only: graded only on
their statement's example inputs, which the prompt shows with their answers, so a program printing
those answers passes. The `codeforces` adapter marks a row whose official tests are the statement's
samples and that no joined suite covers: in `open-r1/codeforces` `verifiable`, 219 of the 422 `test`
problems and 1,303 of the 8,299 `train` ones. `--include_examples_only` keeps them.
`--min_rating` / `--max_rating` bound difficulty, dropping unrated rows with them, `--exclude_keys`
removes listed ids, and `--holdout_per_band` carves a deterministic `test` split.

`--push_bands` publishes `full` plus one config per rating band (`medium` 1500-1999, `hard`
2000-2599, `extra-hard` 2600-3500) over a shared test split, which is how a curriculum stage selects
its pool (`org/name:hard`). `--verify_checkers`, on by default, drops a problem whose special judge
rejects its own reference output or accepts garbage; it runs them through a sandbox, so the
preparation host needs a backend.

A bulky test corpus goes through `compact_code_tests.py` first: it reduces open-r1's generated tests
or HardTests' encoded suites to one capped row per problem (40 tests within 256 KB, two of which may
reach 4 MB so a maximum-size input survives), which `--tests_table` joins by problem id. A table
may cover some splits only; one matching no row of any split exits before any split is filtered.

## Evaluation

```bash
python scripts/environments/inference/run_code_contests.py --adapter codeforces \
    --dataset open-r1/codeforces --config verifiable --split test \
    --training_config examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-full-ep1-stage1-codeforces.yaml \
    --base_url http://localhost:8000/v1 --model <served-name> \
    --num_examples 100 --num_samples 4 --reasoning_effort high
```

`--training_config` gives the eval the recipe's rollout settings and language list (`[python, cpp]` here), which a
policy trained on it names in every call; under `--language python` alone the tools declare no `language` argument
and refuse every such call.

The script buckets `success@1` / `success@k` by the adapter's field (rating here; neither is a benchmark's
mean-over-samples pass@1, see [Evaluating on an Environment](evaluation.md#running-an-evaluation)).
A problem counts solved when the submitted program passes every test in the pool — the environment's
verdict, not the shaped total, so the recipe's shaping under `--training_config` (a
`tool_error_penalty` on every scratchpad call under `leaderboard`, a submission bonus) moves
it neither way, and the coding CLI takes no `--success_threshold`. Examples-only problems
([Dataset](#dataset)) are left out and counted in the log unless `--include_examples_only`, which
the trajectory meta records with the selection. The verdict is the episode's last
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
32768, the training rollout's. It does not switch a thinking model's reasoning off: the request then
carries no `reasoning_effort`, so vLLM leaves `enable_thinking` unset and the template keeps its own
default — Qwen3/3.5/3.6 still think, and stop only on `enable_thinking: false` (the server's
`--default-chat-template-kwargs`, or `rollout_chat_template_kwargs` under `--training_config`). Under `--training_config` the level defaults to the YAML's, where a
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
