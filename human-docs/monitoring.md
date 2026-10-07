# Monitoring

## Console and log file

Everything a run prints — trainer logs, tqdm, native NCCL output — is
mirrored to `<output_dir>/log/run.log` on the logging rank. For a detached
run, that file is your console:

```bash
tail -f checkpoints/sft-qwen3-4b-ultrachat/log/run.log
```

The startup banner prints the parallelism layout (EP/CP/TP/DP sizes) and a
parameter breakdown, plus the detected GPU and precision when
`enable_efficiency_metrics: true`. Glance at it to confirm the run is shaped the
way you intended.

## Weights & Biases and ClearML

Tracking rides the standard HuggingFace integration:

```yaml
report_to: wandb          # or clearml, tensorboard, none
project_name: my-project  # becomes the W&B project / ClearML project
run_name: qwen3-sft-lr1e5 # optional; defaults to <script>-<output dir name>
```

Credentials come from your `.env` (`WANDB_API_KEY`; ClearML uses its usual
`clearml.conf` / `CLEARML_API_*` setup), which the container only sees via
`--env-file .env`. Halo sets `WANDB_PROJECT` and `CLEARML_PROJECT` from
`project_name` and `CLEARML_TASK` from the run name, which W&B takes through
`run_name`, so you don't juggle `WANDB_PROJECT` yourself.

When resuming a run and you want the curves to continue in the same W&B run,
export `WANDB_RUN_ID=<id>` and `WANDB_RESUME=allow` before relaunching;
otherwise the resume starts a fresh run.

## What gets logged

Loss, learning rate, and grad-norm come from the base trainer at every
`logging_steps` (most examples use `logging_steps: 1`). Halo adds opt-in
metric groups on top:

| Config field | Default | Adds |
| --- | --- | --- |
| `enable_efficiency_metrics` | off | step time, tokens/s per GPU and cluster-wide, allocated/peak GPU memory — the numbers [Performance](performance.md) quotes |
| `report_mfu_diagnostics` | off | logs MFU and achieved TFLOPS; needs `enable_efficiency_metrics` on. The S-MFU variants appear only for MoE, where plain MFU misreads sparse models |
| `enable_moe_metrics` | on | expert load balance averaged over the MoE layers: `moe/load_max`, `moe/load_cv`, `moe/dead_frac`, …, plus `moe/load_max_first` / `moe/load_max_last` for the first and last layer (no-op on dense models) |
| `generate_eval_examples` | on (off for SFT) | a table of sample generations at each evaluation (skipped under TP/CP) |
| `save_completions` (online / async GRPO) | on | writes each step's rollouts to `<output_dir>/completions/completions_<step>.parquet` (prompt, completion, reward, advantage), plus a `completions` table in W&B when `report_to` includes `wandb` |
| `log_completions` (online / async GRPO) | off | additionally prints the per-sample table to the console |

For async GRPO with environments, give `sampling/logratio_mean` a standing
dashboard panel: a steady negative drift means the weight sync to the rollout
server is broken, or, with `advantage/net_token_mass` staying negative and
`entropy` climbing after it, a KL-free run drifting, which `balance_token_mass` and the
[early stop](../agent-docs/training-methods/grpo/async-grpo/monitoring.md#early-stop) ↗
address. Online GRPO logs TRL's unsigned gap
`sampling/sampling_logp_difference/mean` instead, with the importance-sampling
correction on. Details on every callback:
[Callbacks](../agent-docs/training-methods/callbacks.md) ↗.

## Profiling

To see where the time goes, set `enable_torch_profiler: true` — it captures
a Chrome trace on rank 0 (`profiler_ranks: "all"` for every rank) with the MoE phases labeled. Feed them to
`halo run trace-report` for a compute/communication/idle breakdown.

Start at [Debugging](../agent-docs/reference/debugging.md) ↗.
