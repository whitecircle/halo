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
`--env-file .env`. Halo sets the project and run-name environment variables for
both backends from the fields above, so you don't juggle `WANDB_PROJECT`
yourself.

When resuming a run and you want the curves to continue in the same W&B run,
export `WANDB_RUN_ID=<id>` and `WANDB_RESUME=allow` before relaunching;
otherwise the resume starts a fresh run.

## What gets logged

Loss, learning rate, and grad-norm come from the base trainer at every
`logging_steps` (the examples use `logging_steps: 1`). Halo adds opt-in
metric groups on top:

| Config field | Default | Adds |
| --- | --- | --- |
| `enable_efficiency_metrics` | off | step time, tokens/s per GPU and cluster-wide, allocated/peak GPU memory |
| `report_mfu_diagnostics` | off | logs MFU and achieved TFLOPS (the callback computes them every step either way; needs `enable_efficiency_metrics` on). The S-MFU variants appear only for MoE, where plain MFU misreads sparse models |
| `enable_moe_metrics` | on | per-layer expert load balance: `moe/load_max`, `moe/load_cv`, `moe/dead_fraction`, … (no-op on dense models) |
| `generate_eval_examples` | on (off for SFT) | a table of sample generations at each evaluation (skipped under TP/CP) |
| `save_completions` (GRPO) | on | writes each step's rollouts to `<output_dir>/completions/completions_<step>.parquet` (prompt, completion, reward, advantage) plus a `completions` table on the tracking backend |
| `log_completions` (GRPO) | off | additionally prints the per-sample table to the console |

For environmental GRPO, give `sampling/logratio_mean` a standing dashboard
panel: a steady negative drift means the trainer→vLLM weight sync is broken.
Online GRPO does not emit it; there, watch reward and KL instead. Details on
every callback: [Callbacks](../agent-docs/training-methods/callbacks.md) ↗.

## Profiling

To see where the time goes, set `enable_torch_profiler: true` — it captures
a Chrome trace on rank 0 (`profiler_ranks: "all"` for every rank) with the MoE phases labeled. Feed them to
`halo run trace-report` for a compute/communication/idle breakdown.

Start at [Debugging](../agent-docs/reference/debugging.md) ↗.
