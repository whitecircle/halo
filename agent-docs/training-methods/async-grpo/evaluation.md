# Evaluation

A run evaluates in two ways: eval rounds inside training, and offline CLIs that drive a served model through an environment. Environment-specific commands live on each environment's page ([Code Contests](environments/tasks/code-contests/evaluation.md), [QA Benchmarks](environments/tasks/qa-benchmarks.md)).

## During training

Eval metrics carry the `eval_` prefix. Eval rounds run without prefetch, and a round waits for its slowest episode.

| Knob | Default | Effect |
|---|---|---|
| `num_generations_eval` | `None` → `num_generations` | The eval group size; `1` gives a fast pass@1 eval |
| `eval_rollout_batch_size` | unset: one round per eval batch | Rows (not prompts) per rank per eval round |

- `eval_rollout_batch_size` must be a multiple of `num_generations_eval`, at most `max_concurrent_rollouts`, and needs `dataloader_drop_last: false`; each raises at startup ([`eval_rollout_batch_size`](../../reference/configuration-reference.md#asynctrainingconfig)). Unset, `per_device_eval_batch_size` must hold whole groups instead; the loss forward chunks by it either way.
- Under `reasoning_effort: random` ([Effort levels](reasoning.md#effort-levels)) each problem's level is drawn from its text, here and under the offline `--training_config`, so checkpoints compare level for level.
- With `num_generations_eval` above 1 and a solve verdict, `eval_outcome/success@<k>` is the share of prompts any of their `k` valid samples solved; `eval_outcome/solve_rate` stays the per-sample mean.

### Padding and round means

Every rank runs the same rounds, so a short last round is padded with repeats of the split's first rows, never rolled out or counted. Two biases remain:

- TRL's loss-time eval metrics (`eval_entropy`, `eval_clip_ratio/*`, `eval_kl`) count a rank whose final round is all padding as 0.
- Every logged eval metric weighs rounds equally, however full.

For exact means, make `len(split) × num_generations_eval` a multiple of `eval_rollout_batch_size × data_parallel_size`, or, for the rollout metrics, run the split in one round.

## Running an evaluation

Two CLIs drive training's own `reset`/`step` loop (`src/environments/eval_runner.py`) against an OpenAI-compatible endpoint (vLLM, SGLang, OpenRouter): `run_env.py` for generic environments, `run_code_contests.py` for code contests.

```bash
python scripts/environments/inference/run_env.py --env_type qa_search \
    --dataset basicv8vc/SimpleQA --split test --prompt_field problem \
    --base_url http://localhost:8000/v1 --model <served-name> --num_examples 100
```

| Flag | Default | Effect |
|---|---|---|
| `--dataset` / `--config` / `--split` | — / none / `test` | Hub id or `save_to_disk` dir. A coding benchmark that ships one split reads its own (HLCE, the ICPC World Finals set: `train`) and refuses any other |
| `--base_url` / `--api_key` / `--model` | local vLLM / `$VLLM_API_KEY`, `$OPENAI_API_KEY`, else a placeholder / — | Endpoint and model name |
| `--training_config <yaml>` | none | Grade under a run's contract ([below](#grading-under-a-training-config)) |
| `--num_examples` / `--num_samples` | 100 (50 coding) / 1 | Rows scored; episodes per row |
| `--success_threshold` | `1.0` | `run_env.py` only: the reward that counts as solved without an environment solve verdict |
| `--max_workers` | 32 (16 coding) | Concurrent episodes |
| `--max_turns` / `--env_kwargs` | the env's own; coding 15 / `{}` | Turn cap override; JSON merged into the env config |
| `--temperature` / `--top_p` / `--max_tokens` / `--request_timeout` | 0.7 (0.2 coding) / 0.95 / 32768 (coding: [Generation budget](environments/tasks/code-contests/evaluation.md#generation-budget)) / 180 s | Sampling, HTTP timeout |

Every request also sends `top_k`, `min_p` and `repetition_penalty` (the run's `rollout_*` values under `--training_config`, else off), so a served model's `generation_config.json` never filters the eval.

A message-list `prompt` reaches the environment as its last `user` turn, as in training; a row with no `user` turn, or a non-text content part in it, is refused before any episode runs.

### Script-specific flags

`run_env.py` reads `--prompt_field` / `--answer_field`, passes extra columns through `--context_fields`, buckets by `--group_by` and names examples by `--id_field` (default `id`). A field naming no column exits before any row is read (the default answer and id columns may be absent). Like the trainer, it refuses a dataset without the answer field when the environment declares `requires_answer`.

`run_code_contests.py` takes instead `--adapter` (fixing the bucket and id fields per benchmark), `--language`, `--eval_protocol`, `--reasoning_effort` (`low` / `medium` / `high` / `none`), `--include_examples_only` and, on dated benchmarks, `--start_date` / `--end_date` / `--platform` ([Code Contests evaluation](environments/tasks/code-contests/evaluation.md)). An option with a flag of its own is refused in `--env_kwargs`.

### Grading under a training config

`--training_config <yaml>` parses the YAML with the training script's config classes, adopting its `RolloutConfig` (template variables, stop tokens, thinking and episode output budgets, sampling) and environment config (rewards, `max_turns`, `environment_kwargs`, `environment_type`). An explicit flag beats the YAML; the YAML beats the default.

- The YAML's dataset keys (`dataset`, `prompt_field`, `answer_field`, `context_fields`) are not read: pass the matching flags.
- The eval joins no process group, so a recipe whose `episode_timeout` needs a raised `DIST_NCCL_TIMEOUT_MINUTES` in training evaluates without it.
- To compare a checkpoint with its base, serve each under the same `--served-model-name` and rerun the same command.

### The report

Besides mean reward and aggregate telemetry, the report logs:

| Score | Meaning |
|---|---|
| `success@1` | Each row's first scored sample |
| `success@k` | Any of the row's `--num_samples` succeeded |
| `invalid` | Samples scored 0 with no signal, each carrying `error`: an invalid grade (a grading or sandbox outage, an inconclusive code grade, a failed scorer under `on_error: invalid`, a null `answer`) or a raised episode. Unlike in training, they stay in the means |
| `generation_errors` | Samples whose generation failed for a reason other than the episode's own length ([Failed generations](#failed-generations)) |

Neither success score is a benchmark's mean-over-samples pass@1. Reasoning models need a large `--max_tokens`, or the chain of thought is cut before any answer and scores 0.

A sample counts as solved on the environment's own verdict where it reports one (the flag behind `outcome/solve_rate`), whatever its shaped reward.

### Failed generations

The OpenAI client retries transport failures itself. Training's retry policy (`max_retries`, `retry_base_wait` from `--training_config`, else 3 and 1 s) covers what the client does not: vLLM's `not JSON compliant` 400 is retried and an engine abort re-issued. A generation that still fails ends the episode:

- **Its own length** (a conversation over the served context, or a turn that outran `--request_timeout` on every client retry): the sample is graded on what it earned, like a miss.
- **Any other cause** (a rejected key, an unknown model or route, a malformed request): a generation error. `reward` and `success` are null, the sample leaves every score but the telemetry line, and `generation_errors` counts it.

A score left with no sample reads `nan`.

## Output files

`--output <path.json>` dumps per-example results: `group`, `id`, and per sample `reward`, `success`, `stats`, `error` or `generation_error` where set, plus any recorded trajectory.

`--save_trajectories <path.jsonl>` records the full run; `--trajectory_dir <folder>` auto-names one file per run: `<model>__<env_type>__<split>.jsonl`, or for the coding script `<model>__<adapter>__<split>__<language>.jsonl` plus `__<eval_protocol>` (other than `harness`) and any contest selection.

Line 1 is a `meta` record of the run's contract: model, env type, dataset/config/split, effective `max_turns`, `rollout`, `training_config`, `system_prompt`, tool schemas, and for coding the adapter, contest `selection`, language, `eval_protocol`, effort and `env_grading`.

Each later line is an `episode` addressed by `index` and `id`: `reward`, `success`, `generation_error`, `stats`, the messages, `reasoning_effort` / `reasoning_budget` / `reasoning_budget_soft` and `info`.

- `context`, the `_`-prefixed `info` fields (code contests' hidden tests and checker) and assistant chain-of-thought are stripped; other `info` keys stay (`exam_qa` keeps `expected_answer`).
- Messages keep their `tool_calls`, which the re-grader replays; a call the engine cut stays as unrun `cut_tool_calls`, which it skips.
- Each assistant turn under a reasoning cap keeps its `thinking_cap`, the cap it ran under ([Thinking-cap metric](reasoning.md#thinking-cap-metric)).
- No turn records `reasoning_tokens`: the eval transport captures no sampled ids.

## Re-grading recorded trajectories

Only `run_code_contests.py` dumps carry the meta a re-grade needs; a `run_env.py` dump is refused. A wall-clock grade inflates under host load, so a high-concurrency sweep can read a correct, fast solution as TIME-LIMIT-EXCEEDED. Generate in parallel, then re-grade in one bounded process:

```bash
python scripts/environments/inference/regrade_trajectories.py \
    "$HALO_DATA_ROOT/eval/trajectories"/*.jsonl --workers 64 --output regraded.jsonl
```

It rebuilds each problem's hidden tests by `index` under the meta line's contest `selection` and replays every admitted `submit_solution` (a refused call, a mislabelled language or a repeat of a graded program takes no slot) under the recorded `env_grading` (the grading settings the meta line records):

- up to the episode's own submission budget (the rebuilt environment's `max_submissions` for an episode that stamped none);
- stopping at the first failing test, without `max_grading_seconds`.

A file is refused before any grade when:

- an episode's `index` lies past the rebuilt problems: the dataset is not the one the run scored;
- `env_grading` names a field the grading contract does not declare, or holds a value it refuses (an unknown `comparison` or `verdict_detail`, a non-positive time limit).

A `selection` that records no `include_examples_only` is read as including the examples-only problems.

Per file it reports the protocol and, over the `n` episodes with a verdict, `s@1` (the first admitted submission passes every test) and `s@2` (any submission within the budget does). An episode with a `generation_error` leaves `n` and counts in `generation_errors`. Keep `--workers` at or below the core count.

## Smoke-testing a config

Point the eval at the training YAML and a few rows before a long run:

```bash
python scripts/environments/inference/run_env.py --training_config <config>.yaml \
    --dataset <the config's dataset> --model <served-name> \
    --num_examples 5 --max_workers 4 --save_trajectories /tmp/smoke.jsonl
```

It checks what otherwise costs a whole run: the server's tool-call parser, `rollout_stop_tokens` against the tokenizer (an unresolved name raises), the chat template, the sandbox or judge backend, and the dataset columns.
