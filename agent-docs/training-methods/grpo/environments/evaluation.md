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
| `--success_threshold` | `1.0` | Reward counting a sample as solved |
| `--max_workers` | 32 (16 coding) | Concurrent episodes |
| `--max_turns` / `--env_kwargs` | the env's own; coding 15 / `{}` | Turn cap override; JSON merged into the env config |
| `--temperature` / `--top_p` / `--max_tokens` / `--request_timeout` | 0.7 (0.2 coding) / 0.95 / 32768 (coding: the effort budget) / 180 s | Sampling, HTTP timeout |

`run_env.py` reads `--prompt_field` / `--answer_field`, passes extra columns through
`--context_fields` and buckets by `--group_by`; `run_code_contests.py` instead takes `--adapter`,
`--language` and `--reasoning_effort`, which also sets the default `--max_tokens`
([Code Contests](code-contests.md)).

`--training_config <yaml>` parses the YAML with the training script's own config classes: its
`RolloutConfig` (template variables, stop tokens, thinking budget, sampling) and environment config
(rewards, `max_turns`, `environment_kwargs`, `environment_type`) become the eval's. An explicit flag
wins over the YAML, the YAML over the default.

A YAML whose `episode_timeout` exceeds the 30-min NCCL watchdog — the code-contests recipes'
`2700` — needs `DIST_NCCL_TIMEOUT_MINUTES=60` exported for the eval too: the contract builds the
run's `RolloutConfig`, and `get_rollout_config` runs the watchdog check even though the eval
forms no process group.

To compare a checkpoint with its base, serve each in turn under the same `--served-model-name` and
run one command with `--training_config`, dataset, split and `--num_samples` fixed, so only the
weights differ.

The report logs mean reward, `success@1`, `success@k` and aggregate telemetry, so token
starvation reads differently from wrong answers. Reasoning models need a large `--max_tokens`: too
low cuts the chain of thought before any answer, scoring 0.

## Output files

`--output <path.json>` dumps per-example results: `group`, `id`, and per sample `reward`,
`success`, `stats`, `error` on a failed or signal-less grade — plus the trajectory when one is
recorded.

`--save_trajectories <path.jsonl>` records the full run; `--trajectory_dir <folder>` auto-names one
file per run instead (`<model>__<env_type>__<split>.jsonl`, or
`<model>__<adapter>__<split>__<language>.jsonl` for the coding script).

Line 1 is a `meta` record: model, env type, dataset/config/split, effective `max_turns`, the
generation contract (`rollout`), the `training_config`, `system_prompt` and tool schemas — for
coding, also the adapter, language, effort and the `GradingSpec` (`env_grading`).

Each later line is an `episode`, addressed by `index` and `id`: `reward`, `success`, `stats`, the
messages, `reasoning_effort` / `reasoning_budget`, `info`. The answer key (`_`-prefixed `info`
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

It rebuilds each problem's hidden tests by `index` and replays every recorded `submit_solution`, up
to that episode's own budget, through `grade_solution` under the meta line's `env_grading` contract. Grading stops at the first failing test and `max_grading_seconds`
does not apply. Reports `s@1` / `s@2` per file; keep `--workers` at or below the core count.

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
