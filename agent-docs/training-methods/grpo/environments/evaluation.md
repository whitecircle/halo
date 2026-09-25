# Evaluating on an Environment

Two CLIs run a model through an environment against an OpenAI-compatible endpoint (vLLM, SGLang,
OpenRouter) and report mean reward and success@k: `run_env.py` for the generic environments,
`run_code_contests.py` for competitive programming. Both drive training's own `reset`/`step` loop
(`src/environments/eval_runner.py`).

## Running an evaluation

```bash
python scripts/environments/inference/run_env.py --env_type qa_search \
    --dataset basicv8vc/SimpleQA --split test --prompt_field problem \
    --base_url http://localhost:8000/v1 --model <served-name> --num_examples 100
```

| Flag | Default | Effect |
|---|---|---|
| `--dataset` / `--config` / `--split` | — / none / `test` | Hub id or `save_to_disk` dir |
| `--base_url` / `--api_key` / `--model` | local vLLM / `$VLLM_API_KEY`, `$OPENAI_API_KEY`, else a placeholder / — | Endpoint and model name |
| `--training_config <yaml>` | none | Grade under a run's contract (below) |
| `--num_examples` / `--num_samples` | 100 (50 coding) / 1 | Rows scored; episodes per row |
| `--success_threshold` | `1.0` | `run_env.py` only: reward counting a sample as solved, for an environment that reports no solve verdict |
| `--max_workers` | 32 (16 coding) | Concurrent episodes |
| `--max_turns` / `--env_kwargs` | the env's own; coding 15 / `{}` | Turn cap override; JSON merged into the env config |
| `--temperature` / `--top_p` / `--max_tokens` / `--request_timeout` | 0.7 (0.2 coding) / 0.95 / 32768 (coding at a level: its effort budget) / 180 s | Sampling, HTTP timeout |

`run_env.py` reads `--prompt_field` / `--answer_field`, passes extra columns through
`--context_fields`, buckets by `--group_by` and names each example by `--id_field` (default `id`);
a field that names no column of the split exits before any row is read, except the default answer
and id columns, which a dataset may lack; like the trainer, it refuses a dataset without the answer
field before generating when the environment declares `requires_answer`. `run_code_contests.py`
instead takes `--adapter` (which fixes the bucket and id fields per benchmark), `--language`,
`--reasoning_effort` (`low` / `medium` / `high`, or `none` for no level; it also sets the default
`--max_tokens`), `--eval_protocol`, and `--start_date` / `--end_date` / `--platform` on a benchmark
that stamps contest dates ([Code Contests](code-contests.md#evaluation)). There an option with a
flag of its own (`--max_turns`, `--language`, `--eval_protocol`, `--reasoning_effort`) is refused in
`--env_kwargs`, which would otherwise override the flag.

`--training_config <yaml>` parses the YAML with the training script's own config classes: its
`RolloutConfig` (template variables, stop tokens, thinking budget, sampling) and environment config
(rewards, `max_turns`, `environment_kwargs`, `environment_type`) become the eval's. An explicit flag
wins over the YAML, the YAML over the default.

The training run's check of `episode_timeout` against the NCCL watchdog stays with training: the
eval joins no process group, so a recipe whose budget needs a raised `DIST_NCCL_TIMEOUT_MINUTES`
(the code-contests recipes' `episode_timeout: 2700`) evaluates without it.

To compare a checkpoint with its base, serve each in turn under the same `--served-model-name` and
run one command with `--training_config`, dataset, split and `--num_samples` fixed, so only the
weights differ.

The report logs mean reward, `success@1`, `success@k`, `invalid`, `generation_errors` and aggregate
telemetry, so token starvation reads differently from wrong answers. `success@1` is each row's first
scored sample and `success@k` whether any of its `--num_samples` succeeded; neither is the mean over
samples a benchmark's pass@1 reports. Reasoning models need a large `--max_tokens`: too low cuts the
chain of thought before any answer, scoring 0.

A sample counts as solved on the environment's own verdict where it reports one — the flag training
averages into `outcome/solve_rate` (code contests: every test in the pool passed) — whatever its shaped
reward: a tool-error or length-cutoff penalty cannot sink a solve, and a submission bonus cannot lift
a partial one. `--success_threshold` decides only for an environment without that verdict.

A turn runs under training's retry policy (`max_retries`, `retry_base_wait` from `--training_config`,
else 3 and 1 s): an engine fault the OpenAI client does not retry (vLLM's `not JSON compliant` 400) is
retried with backoff, and an engine abort is re-issued, never stepped. The client itself retries
transport failures. A generation that still fails ends the episode. When the episode's own length
caused it — a conversation over the served context (a client error whose message names the context
length), or a turn that outran `--request_timeout` on every client retry — the sample is graded on
what it earned, like a miss. Any other failure, a rejected key, an unknown model or route and a
malformed request among them, makes it a generation error: `reward` and `success` are null, it leaves
every score (the telemetry line still counts it), and `generation_errors` counts it. A score left with
no sample reads `nan`.

`invalid` counts the samples scored 0 with no signal, each carrying `error`: an invalid grade (a
grading or sandbox outage, a failed scorer, a null `answer`) or an episode whose run raised. Invalid
samples stay in the means, unlike in training, where the baseline drops them.

## Output files

`--output <path.json>` dumps per-example results: `group`, `id`, and per sample `reward`,
`success`, `stats`, `error` on a failed or signal-less grade, `generation_error` on a generation error —
plus the trajectory when one is recorded.

`--save_trajectories <path.jsonl>` records the full run; `--trajectory_dir <folder>` auto-names one
file per run instead (`<model>__<env_type>__<split>.jsonl`, or
`<model>__<adapter>__<split>__<language>.jsonl` for the coding script). The coding name adds
`__<eval_protocol>` for a protocol other than `harness` and a part naming a contest selection when
one is set, e.g. `__leaderboard__2025-01-01..2025-04-30_atcoder-codeforces`.

Line 1 is a `meta` record: model, env type, dataset/config/split, effective `max_turns`, the
generation contract (`rollout`), the `training_config`, `system_prompt` and tool schemas — for
coding, also the adapter, contest `selection`, language, `eval_protocol`, effort and the `GradingSpec`
(`env_grading`).

Each later line is an `episode`, addressed by `index` and `id`: `reward`, `success`,
`generation_error` (null on a scored sample), `stats`, the messages, `reasoning_effort` /
`reasoning_budget`, `info`. The answer key (`_`-prefixed `info`
fields), the `info` tool-call log, `context` and assistant chain-of-thought are stripped; each
message keeps its own `tool_calls`, which the re-grader replays.

## Re-grading recorded trajectories

A wall-clock grade inflates under host load, so a high-concurrency sweep reads a correct, fast
solution as TIME-LIMIT-EXCEEDED. Generate in parallel, re-grade in one bounded process where each
run gets a core:

```bash
python scripts/environments/inference/regrade_trajectories.py \
    "$HALO_DATA_ROOT/eval/trajectories"/*.jsonl --workers 64 --output regraded.jsonl
```

It rebuilds each problem's hidden tests by `index` under the meta line's contest `selection`, and
replays every recorded `submit_solution`, up to that episode's own budget, through `grade_solution`
under the meta line's `env_grading` contract. The meta's `eval_protocol` only rebuilds the
environment, whose `max_submissions` is the budget of an episode that stamped none. Grading stops at
the first failing test and `max_grading_seconds` does not apply. It reports, per file, the protocol
and, over the episodes that carry a verdict (`n`): `s@1`, the fraction whose first admitted submission
passes every test, and `s@2`, the fraction whose any submission within the episode's budget does. An
episode recorded with a `generation_error` leaves `n` and is counted in `generation_errors`. Keep
`--workers` at or below the core count.

Only `run_code_contests.py` stamps the meta a re-grade needs (`env_type`, `adapter`, `dataset`,
`model`, `language`); a `run_env.py` dump is refused.

## Smoke-testing a config

Point the eval at the training YAML and a few rows before a long run:

```bash
python scripts/environments/inference/run_env.py --training_config <config>.yaml \
    --dataset <the config's dataset> --model <served-name> \
    --num_examples 5 --max_workers 4 --save_trajectories /tmp/smoke.jsonl
```

It exercises what otherwise costs a whole run: the server's tool-call parser,
`rollout_stop_tokens` against the tokenizer (an unresolved name raises here), the chat template,
the sandbox or judge backend, and the dataset columns.

## Related pages

- [Code Contests](code-contests.md) — the coding environment and grading
- [QA Benchmarks](benchmarks.md) — `qa_search`, `exam_qa`
- [Metrics and Troubleshooting](../async-grpo/monitoring.md#evaluation) — eval rounds in training
