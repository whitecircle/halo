# Evaluation

`scripts/environments/inference/run_code_contests.py` scores a served model on this environment. Flags, output
files and re-grading shared by every environment are on [Evaluation](../../../evaluation.md).

## Evaluation protocols

`eval_protocol` names the contract a run is scored under:

| Protocol | Pins | Measures |
|---|---|---|
| `harness` (default) | Nothing: the configured budgets stand | The agentic loop the recipes train: attempts-until-accept within the budget |
| `leaderboard` | `max_submissions: 1`, `max_test_calls: 0` | One graded program per sample, scratchpad closed. The prompt, tool-call format and grader stay this environment's, so its numbers are not a benchmark's published ones |

- A configured value that contradicts a pin raises at construction. An effort profile's tool budgets are
  validated, then give way to the pins (logged); the level keeps its `thinking_tokens`.
- Under `leaderboard` the model learns the scratchpad is closed only from its reply, a
  [budget-0 call](tools.md#refused-calls) that is not a refusal.

## Running an evaluation

```bash
python scripts/environments/inference/run_code_contests.py --adapter codeforces \
    --dataset open-r1/codeforces --config verifiable --split test \
    --training_config examples/grpo/environmental/qwen3_5/vllm/qwen3.6-35b-a3b-code-contests-full-ep1-stage1-codeforces.yaml \
    --base_url http://localhost:8000/v1 --model <served-name> \
    --num_examples 100 --num_samples 4 --reasoning_effort high
```

- `--training_config` supplies the recipe's rollout settings and language list. A policy trained on
  `[python, cpp]` names a language in every call, which `--language python` alone refuses.
- `--eval_protocol` picks the protocol, else the training config's, else `harness`; the report title and trajectory meta name it.
- A training config written under another protocol yields the tool budgets the flag's protocol pins (logged); one
  naming the protocol itself, or `--env_kwargs`, that contradicts a pin raises.
- Grading knobs with no flag go through `--env_kwargs`, recorded in the trajectory meta.

### Scoring

`success@1` / `success@k` are bucketed by the adapter's field (rating here); neither is a benchmark's
mean-over-samples pass@1 ([Running an evaluation](../../../evaluation.md#running-an-evaluation)).

- A problem counts solved when the episode's last graded submission passes every test. Shaping never moves it,
  and the coding CLI takes no `--success_threshold`.
- [Examples-only problems](datasets.md#examples-only-problems) are left out and counted in the log unless
  `--include_examples_only`, which the trajectory meta records with the selection. On an adapter that marks none,
  the flag exits before a row is read.
- `success@1` is each row's first scored sample. The
  [re-grader](../../../evaluation.md#re-grading-recorded-trajectories) scores each episode on its first submission
  (`s@1`) or any within its budget (`s@2`); the two agree only on a `leaderboard` run at `--num_samples 1`.

## Generation budget

- `--reasoning_effort` defaults to the training config's level, where `reasoning_effort: null` is `none`. A
  config without the key, or no config, takes `medium`, as training does.
- `--max_tokens` defaults to the training config's `rollout_max_tokens`. Without one it follows the level:
  enforced, the level's `thinking_tokens` (profile overrides in `--env_kwargs` included) plus 4096, which the
  served context window must exceed; soft, 32768 or that sum where larger; `none`, 32768.
- Any thinking cap needs a [reasoning parser](../../../reasoning.md#reasoning-budget) on the vLLM server, which
  otherwise rejects the request.
- Evaluate a non-thinking model with `--reasoning_effort none`; a think-tag parser reads its whole answer as
  reasoning. `none` does not switch a thinking model's reasoning off: the request carries no `reasoning_effort`, so
  the template keeps its own default. Qwen3/3.5/3.6 still think, and stop only on `enable_thinking: false` (the
  server's `--default-chat-template-kwargs`, or `rollout_chat_template_kwargs` under `--training_config`).
- The eval treats the server as SGLang only when `--training_config` names `rollout_backend: sglang`, and then
  drops the budget as training does.

## LiveCodeBench window

A LiveCodeBench release is cumulative (every problem since May 2023), so score a window after the model's
training cutoff:

```bash
python scripts/environments/inference/run_code_contests.py --adapter livecodebench \
    --dataset livecodebench/code_generation_lite --config release_v6 \
    --start_date 2025-01-01 --end_date 2025-04-30 --platform atcoder,codeforces \
    --eval_protocol leaderboard --base_url http://localhost:8000/v1 --model <served-name> \
    --num_examples 0 --num_samples 4
```

- `--start_date` / `--end_date` (`YYYY-MM-DD`) are inclusive, compare the row's `contest_date` by calendar day,
  and may each stay open.
- `--platform` takes `atcoder` and `codeforces`. `leetcode` is refused: its rows are functional, which a
  stdin/stdout environment cannot grade.
- Rows come newest release file first, each file in its stored order, which is not newest-first. `--num_examples`
  (default 50) takes the window's first problems in that order; `0` scores the whole window.
- Another date spelling, a start after the end, an unknown platform, or a window or platform filter on any adapter
  but `livecodebench` exits before a row is read. A row without a parsable `contest_date` raises under a window.

The selection is recorded in the trajectory meta
([re-grading](../../../evaluation.md#re-grading-recorded-trajectories)).
