# Code Contests Environment

`CodeContestsEnvironment` (`src/environments/envs/tasks/coding/code_contests.py`) is a multi-turn, native-tool environment for competitive programming in Python (default), C++, or C. The model writes a solution, tries it with a REPL tool, and submits via `submit_solution`, which runs it against hidden tests through a [`SandboxExecutor`](sandbox.md). Grading is data-driven, so one class covers exact-match contest sets and Codeforces alike — `codeforces` is the registry preset that flips the default comparison to tokens.

```python
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment

env = CodeContestsEnvironment(
    max_turns=15,
    timeout_per_test=15,          # per-test cap when the problem declares none; also the interpreted-language floor
    max_grading_seconds=150,      # wall-clock budget per graded submission (default None = unbounded)
    language="python",            # "python" (default), "cpp", or "c"
    output_comparison="exact",    # "exact" (default) or "tokens" (Codeforces)
    stop_on_first_failure=False,  # True = stop at the first failing test; deflates the pass fraction (see Reward)
    max_submissions=2,            # graded submissions per episode; reaching the cap ends the episode
    max_test_calls=5,             # scratchpad test-tool calls per episode; beyond it the call is rejected
    reasoning_effort="medium",    # "low" | "medium" | "high" | "random"
)
```

## Tools

Two tools: a language-appropriate test tool (`python_repl` for `python`, `run_code` for `cpp`/`c`) plus `submit_solution`.

The test tool is a scratchpad. It runs whatever program the model passes through the same isolated `SandboxExecutor` that grades submissions, with the standard library available (the in-process restricted REPL blocks imports and cannot run real solutions). It gets no stdin and never sees the graded tests, so the model embeds its own inputs.

Beyond `max_test_calls` the call is rejected with a nudge to submit, but the episode continues. The rejection classifies as a tool **error** (charged `tool_error_penalty`, never paid `tool_success_reward`), as does an over-cap `submit_solution` call.

`submit_solution` grades a complete stdin/stdout program against the hidden tests and is the only graded channel — there is no fenced-code-block fallback, so the submission cap stays meaningful. The env binds each episode's tests to its trajectory via a `ContextVar`, so one instance grades concurrent rollouts correctly. `max_test_calls` and `max_submissions` bound the episode; `max_turns` is a backstop, not the tuning lever.

`CODE_SYSTEM_PROMPT` carries the solver's role and the task contract: submit to score, code belongs in tool calls and never in the message. The per-episode budgets travel in the tool descriptions, or in the task message when effort profiles bind them ([Reasoning effort](#reasoning-effort)).

## Reasoning effort

`reasoning_effort` (`"low"` / `"medium"` / `"high"` / `"random"`, default `"medium"`) reaches the model's chat template at generation time as the request's top-level `reasoning_effort` field — the mechanism gpt-oss uses to render `Reasoning: <effort>`. Training and eval send it through the same helper, in that one spelling ([Environmental GRPO](../environmental-grpo.md#chat-template-must-match-training)). It is not in the prompt; a model whose template ignores the key is unaffected. The knob and `"random"` are general to every environment (`BaseEnvironment`).

For this env each level is a **profile** (`reasoning_effort_profiles`, defaults `REASONING_EFFORT_PROFILES`) with five optional keys:

- **`thinking_tokens`** — hard per-turn CoT budget (low 4096, medium 8192, high 16384), sent as the vLLM `thinking_token_budget` and capped by the global `rollout_max_thinking_tokens`.
- **`max_submissions`** / **`max_test_calls`** — per-episode interaction budgets, stamped at reset and stated in the task message.
- **`tested_submission_reward`** — paid once per episode when a scratchpad run precedes the first submission (ordering-gated, contentful-grade-gated, not stated in the prompt). Logged as `reward/tested_submission`; `episode/tested_before_submission` tracks the rate.
- **`token_cost`** — a per-effort compute price in reward units per 1k generated tokens (both channels), charged against the episode's total generation and logged as `reward/token_cost`. A price, not a target: a length floor is farmable by padding, a cost is not.

Overrides merge per level over the defaults, so a profile that sets only interaction keys keeps the default thinking budget. Other environments leave the budget unset and fall back to the global cap.

Eval (`eval_runner.py`) binds the same per-level profile through `bind_episode_effort`, narrows its own `RolloutConfig` to the level's `max_tokens` / `max_thinking_tokens`, and sends the result through `generation_control_fields` — so the interaction budgets and the CoT cap both apply, and the resolved budget is recorded on every trajectory (`reasoning_budget`).

The interaction half is what makes effort buy **iteration**, not just longer CoT: the shipped configs scale both budgets by effort (2/3/3 submissions, 2/4/6 scratchpad runs), keeping the verdict→fix loop available at medium and high. Without an interaction limit the strategy collapses to submit-and-fix at every level. No shipped config sets `tested_submission_reward`, so the test-first bonus is off by default.

Interaction budgets apply only when the level is concrete at reset (a trainer-stamped group level, a non-`random` env setting, or eval's per-episode draw, which is stamped before reset — `BaseEnvironment.reset_effort_level`); an undetermined level keeps the class caps.

`"random"` samples a level uniformly, resolved once per **generation group** in training (`_stamp_group_efforts`) so all `num_generations` completions of a prompt share the level and the group-relative advantage compares like with like; eval draws per episode. Either way the level is fixed for every turn of an episode.

## Grading

All grading goes through `grade_solution` (`.../coding/grading.py`), shared by the environment and the offline re-grader, so a checkpoint scores identically in both. The verdict text lists **non-passing tests only**, capped at 5 with a "more omitted" line — the summary already carries the pass count.

It returns a `GradeResult` (`passed`, `total`, `details`, `ran_ok`, `graded`, `infra_errors`, `budget_hit` — a `NamedTuple`, so the order is the unpacking order). `ran_ok` counts tests whose code ran to completion **and** produced output, so a clean-exit stub that prints nothing scores 0 on the execution rung (a program that prints nothing still counts when the expected output is itself blank). `infra_errors` counts tests lost to the grading backend rather than to the program — a remote sandbox transport/backend failure, or a raising local executor (missing interpreter, absent `bwrap`, fork/fd exhaustion, ENOSPC).

- **Comparison** — `"exact"` (trimmed byte equality) or `"tokens"` (whitespace-token equality, tolerant of trailing spaces, blank lines, and `\r\n`). Byte-exact equality spuriously fails correct Codeforces solutions, hence the `codeforces` preset. Token comparison also accepts real-valued tokens within 1e-6 absolute-or-relative error, gated on a float-looking *expected* token so integer answers stay byte-exact.
- **Special judges** — a per-problem `checker` in the answer payload overrides comparison (`python checker.py input.txt correct_output.txt solution_output.txt` → `1` accept / `0` reject). It runs at the infra default timeout, never the solution's clamped limit.
- **Grading time budget** — `max_grading_seconds` (default `None` = unbounded) bounds the total wall clock of one submission grade. Each test is a sequential sandbox run bounded only by its own timeout, so a several-hundred-test problem otherwise grades for tens of minutes and stalls the whole rollout round behind one episode. The budget is checked between tests (an in-flight test finishes — the hard bound is the budget plus one per-test timeout) and at least one test always runs. A budget stop keeps the **full pool** as the denominator: an ungraded test counts as failed, so a correct but latency-bound solution scores below 1 and cannot outscore one that ran every test. Size the budget so an honest solution finishes. Every graded episode with tests logs `episode/tests_graded_frac` (tests judged ÷ pool, `1.0` on a full grade) and `episode/grading_budget_hit` (`0.0`/`1.0`), so a run's partial grading is visible instead of reading as wrong solutions.
- **Why the configs set 150** — a full honest grade of a several-hundred-test pool fits it (the sandbox fast path is ~0.02 s/test), while a TLE-prone solution stops after budget ÷ per-test limit timed-out tests (~30 at the training configs' `timeout_per_test: 5`).
- **Per-problem time limit** — a `time_limit` in the payload becomes the per-test cap (clamped to `max_time_limit`, default 15 s); otherwise `timeout_per_test` (default 15 s) applies. For interpreted languages the cap is floored at `timeout_per_test` so a C++-tuned limit (e.g. ICPC's 1–3 s) does not TLE a correct-but-slower CPython solution; compiled languages use the stated limit as-is. Lowering `timeout_per_test` therefore lowers that floor too.

### Reward

The objective is the fraction of hidden tests passed by the **submitted** solution, `(passed / total) * success_reward`; an unsubmitted solution scores `failure_reward`. `stop_on_first_failure` (default `false`) breaks out of grading at the first failing test, so `passed` and `ran_ok` become lower bounds against the full `total` and a partially-correct solution scores below its true pass fraction — it buys cheap all-pass checking at the cost of the dense signal both the objective and the execution rung depend on.

This env defaults `tool_success_reward` and `tool_error_penalty` to `0` (overriding the native per-call shaping), so the ladder below is the only extra signal unless a config re-enables them — the canonical config does, at `tool_error_penalty: 0.05`.

Every rung pays only on a **contentful** grade. A zero-test row, a submission whose arguments never parsed into runnable code, or a grading-backend outage pays nothing — such rows would otherwise be guaranteed-payout attractors.

`episode/grading_infra_outage` is logged as `0.0`/`1.0` on every graded episode that had tests, so its mean is the outage rate. An outage also marks the episode invalid, so the trainer drops it from the GRPO group baseline instead of scoring it as a wrong answer.

| Outcome | Reward contribution |
|---------|--------------------|
| plain-text giveup (0 tool calls) | `−no_tool_use_penalty` |
| a tool call but no submission | `0` |
| episode burned `max_turns` without terminating | `−turn_overflow_penalty` (on top of whatever it earned) |
| a contentful graded submission | `+submission_reward` |
| a scratchpad run before the first submission | `+tested_submission_reward` (from the effort profile; once per episode) |
| submitted AND used > 1 tool call | `+multi_turn_reward` |
| fraction of graded tests that **ran cleanly** (right or wrong) | `frac × execution_progress_reward` |
| fraction of hidden tests passed | `frac × success_reward` (dominant) |

All five shaping rungs are non-negative magnitudes defaulting to `0` (off). They bootstrap the tool-use loop a weak base model otherwise can't escape: answering in plain text and never submitting scores `failure_reward` every time, leaving GRPO no gradient toward "submit". Keep each small next to `success_reward` — a rung self-neutralizes within a GRPO group once all completions reach it, so it shapes early then fades.

The execution rung is the anti-sparsity signal: on a hard problem where every completion fails, it separates runnable-but-wrong from crashes and restores the within-group signal. `turn_overflow_penalty` prices the turn cap, which the reward is otherwise blind to.

The env logs the decomposition as `reward/objective`, `reward/submission`, `reward/tested_submission`, `reward/execution`, `reward/tool_shaping`, and `reward/turn_shaping`; the components sum exactly to the scalar reward, and the trainer's `reward/composition_residue` metric flags any channel that bypasses them ([metrics](../environmental-grpo.md#logged-metrics)).

C++/C need a compiling backend; see [Code Execution Sandboxes](sandbox.md).

## Dataset format

`answer` is a JSON string (or dict). Both the simple (`test_cases`) and full (`tests` + optional `checker` + optional `time_limit`) payloads are accepted:

```json
{
  "prompt": "<full problem statement>",
  "answer": "{\"tests\": [{\"input\": \"2 3\\n\", \"output\": \"5\\n\"}], \"checker\": null, \"time_limit\": 2.0}"
}
```

Adapters live in `.../coding/datasets.py` (`CODE_DATASET_ADAPTERS`). Training pools load with a plain `load_dataset`; benchmark adapters carry a custom `load` and are evaluated directly.

| Dataset | HuggingFace ID | Adapter | Role | Per-problem time limit | Notes |
|---------|---------------|---------|------|------------------------|-------|
| Codeforces | `open-r1/codeforces` | `codeforces` | RL pool | yes (`time_limit`, s) | `verifiable` config; token comparison + `generated_checker` special judges; interactive rows dropped |
| DeepCoder | `agentica-org/DeepCoder-Preview-Dataset` | `deepcoder` | RL pool | no | stdin/stdout tests; functional (`fn_name`) specs skipped; no report bucket |
| CodeContests | `deepmind/code_contests` | — | RL pool | no | exact-match; `answer = {"test_cases": [...]}`; no adapter, so `prepare_code_dataset.py` cannot build it (`--adapter` is `codeforces`/`deepcoder`) — hand-prepared only |
| LiveCodeBench | `livecodebench/code_generation_lite` | `livecodebench` | benchmark | no | release `test*.jsonl` read directly (its loader script datasets 4.x rejects), newest contests first; stdin problems graded, LeetCode `functional` skipped; bucketed by `difficulty` |
| ICPC-Eval | `RUC-AIBOX/ICPC-Eval` | `icpc` | benchmark | yes (`time_limit_ms`) | streamed (tests are multi-GB); `traditional` graded, `spj` (C++ special judge) skipped; bucketed by `source`, reported as "contest" |
| HLCE (ICPC WF) | `HumanLastCodeExam/icpc-world-finals` | `hlce` | benchmark | no | streamed; `test_cases` stdin/stdout; bucketed by `platform`, reported as "contest" |

The HuggingFace IDs are the datasets each adapter was written against, not a code-enforced binding — the id is the `--dataset` argument, and an adapter runs against any source with a matching row shape.

A missing per-problem limit falls back to `timeout_per_test` (see [Grading](#grading) for the interpreted-language floor and the `max_time_limit` clamp).

**Not adapted** — these need grading machinery this stdin/stdout env does not run: LiveOIBench (per-problem `grader_code` / `evaluation_script` with subtask scoring), the HLCE IOI subset (statement samples only, no hidden tests), and USACO (function/subtask graders or GitHub file-tree bundles). HLCE interactive problems need a back-and-forth manager and score zero here.

## Preparation

`scripts/environments/preparation/prepare_code_dataset.py` composes each problem's statement and packs its grading payload into the `answer` schema, filtering to gradable stdin/stdout rows:

```bash
python scripts/environments/preparation/prepare_code_dataset.py \
    --adapter codeforces --dataset open-r1/codeforces --config verifiable \
    --output_dir "$HALO_DATA_ROOT/s3_datasets/codeforces-verifiable-rl" --min_rating 800 --max_rating 2200

python scripts/environments/preparation/prepare_code_dataset.py \
    --adapter deepcoder --dataset agentica-org/DeepCoder-Preview-Dataset --config taco \
    --output_dir "$HALO_DATA_ROOT/s3_datasets/deepcoder-taco-rl"
```

`--min_rating` / `--max_rating` drop unrated problems along with out-of-band ones, so a rating-bounded pool carries no problems of unknown difficulty; on a dataset with no `rating` column (deepcoder) they are a no-op. The prepared `rating` column is `0` where the source had none.

One adapter per run, one directory per pool. The script writes no combined pool and none is needed: the shipped code-contests configs list both directories under `dataset:` and the loader concatenates them (`dataset_ratio` weights them; unset means all of each).

A pool keeps whatever splits its source shipped, and the script carves none of its own. DeepCoder's `taco` config ships `train` alone, so that pool contributes training rows and nothing to the held-out set — which is the Codeforces pool's own `test` split. A listed entry missing `train` is refused at load; to hold out rows from every entry instead, set `test_size` in the training config, which splits each one. The paths in those configs are the `/mnt` scratch of the machine they were written on; point them at the `--output_dir` you actually wrote, since `$HALO_DATA_ROOT` is a launch-time convention that a YAML value does not expand.

## Training and evaluation

```yaml
environment_type: codeforces  # or code_contests (exact-match default)
max_turns: 15
answer_field: answer
environment_kwargs:
  language: python            # or cpp / c
  timeout_per_test: 5         # the training configs' value; the class default is 15
  max_grading_seconds: 150
  reasoning_effort_profiles:  # effort = thinking budget + interaction + compute price (see Reasoning effort)
    low: {thinking_tokens: 4096, max_submissions: 2, max_test_calls: 2, token_cost: 0.05}
    medium: {thinking_tokens: 8192, max_submissions: 3, max_test_calls: 4, token_cost: 0.02}
    high: {thinking_tokens: 16384, max_submissions: 3, max_test_calls: 6}
  output_comparison: tokens   # codeforces preset default; "exact" for code_contests
  stop_on_first_failure: false
```

The canonical training config is `examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-lora-ep1.yaml`: gpt-oss-20b, `codeforces` preset, `reasoning_effort: random`, `max_grading_seconds: 150`, the shaping rungs (`submission_reward: 0.25`, `execution_progress_reward: 0.05`, `no_tool_use_penalty: 0.1`, `multi_turn_reward: 0.05`, `turn_overflow_penalty: 0.1`), the [tuned verifiable-reward objective](../online-grpo.md#grpo-objective-for-verifiable-rewards) and [chunked log-probs](../environmental-grpo.md#chunked-log-probs). It also re-enables the per-call `tool_error_penalty: 0.05` the env defaults off, leaving `tool_success_reward` at 0 — any per-call pay is farmable by duplicate re-runs.

It sets `episode_timeout: 2700`, which needs `DIST_NCCL_TIMEOUT_MINUTES=60` on the trainer ([Environmental GRPO — Troubleshooting](../environmental-grpo.md#troubleshooting)). Sibling variants in the same tree swap the backend (`sglang/`), adapter (`-full-`), or expert distribution (`-ep4` — one 4-rank DeepEP group; use it when expert weights or optimizer state are the memory pressure).

**Evaluation** — `scripts/environments/inference/run_code_contests.py` applies a dataset adapter, prompts in `--language`, and reports `success@1` / `success@k` bucketed by the adapter's group field, against a vLLM or OpenRouter endpoint:

```bash
python scripts/environments/inference/run_code_contests.py \
    --dataset open-r1/codeforces --config verifiable --split test --adapter codeforces \
    --base_url http://localhost:8000/v1 --model Qwen/Qwen3.6-35B-A3B \
    --num_examples 100 --num_samples 4 --reasoning_effort high
```

`--reasoning_effort` (low/medium/high) sets the chat-template effort and, unless `--max_tokens` is given, the generation budget — the level's `thinking_tokens` plus 4096 tokens of solution headroom, which the served context window must exceed. An episode that never calls `submit_solution` scores 0, not partial credit.

Raise `--max_workers` (default 16) for throughput and `--request_timeout` (default 180 s) to match — at high concurrency a single 32k-token generation takes minutes. The grading knobs the flags do not cover (`stop_on_first_failure`, `timeout_per_test`, `max_submissions`, `sandbox_backend`, …) go through `--env_kwargs '{"stop_on_first_failure": true}'` and are recorded in the trajectory meta.

To record episodes, [save trajectories](benchmarks.md#recording-trajectories); for a large parallel sweep, [grade offline](benchmarks.md#offline-re-grading-contention-free) instead of inline.
