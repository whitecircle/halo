# Monitoring

## Console and log file

The logging rank (global rank 0, or each node's first rank on per-node disks)
copies its console output, including tqdm and NCCL messages, to
`<output_dir>/log/run.log`. Other ranks' output is not in it. For a detached
run, that file is your console:

```bash
tail -f checkpoints/sft-qwen3-4b-ultrachat/log/run.log
```

The startup banner prints the parallelism layout (EP/CP/TP/DP sizes) and a
parameter breakdown. With `enable_efficiency_metrics: true` it also prints the
detected GPU and precision. Check it to confirm the run has the shape you
intended.

## Weights & Biases and ClearML

Tracking uses the standard HuggingFace integration:

```yaml
report_to: wandb          # or clearml, tensorboard, none
project_name: my-project  # the W&B or ClearML project
run_name: qwen3-sft-lr1e5 # optional; defaults to <method>-<mode>-<output dir name>
```

Credentials come from your `.env`: `WANDB_API_KEY` for W&B, the usual
`clearml.conf` or `CLEARML_API_*` setup for ClearML. Halo does not load `.env`
itself; pass it to the container with `--env-file .env`.

Halo sets `WANDB_PROJECT` and `CLEARML_PROJECT` from `project_name`, and
`CLEARML_TASK` from the run name, so you do not set them yourself.

A resumed run starts a new W&B run. To continue the same curves, export
`WANDB_RUN_ID=<id>` and `WANDB_RESUME=allow` before relaunching.

## What gets logged

The base trainer logs loss, learning rate and grad norm every `logging_steps`
(most examples use `1`). Halo adds these metric groups:

| Config field | Default | Adds |
| --- | --- | --- |
| `enable_efficiency_metrics` | off | step time, tokens/s per GPU and cluster-wide, allocated and peak GPU memory |
| `report_mfu_diagnostics` | off | MFU and achieved TFLOPS; needs `enable_efficiency_metrics`. MoE runs also get the S-MFU variants, since plain MFU misreads sparse models |
| `enable_moe_metrics` | on | expert load balance averaged over MoE layers (`moe/load_max`, `moe/load_cv`, `moe/dead_frac`, …), plus `moe/load_max_first` and `moe/load_max_last`. Reports when router logits are on (`moe_balancing: aux_loss`); under `bias_update` the balancing callback logs the `moe/*` keys instead. No-op on dense models |
| `generate_eval_examples` (SFT, DPO, SMPO, offline GRPO) | on (off for SFT) | a table of sample generations at each evaluation; skipped under TP and CP |
| `save_completions` (online and async GRPO) | on | each step's rollouts in `<output_dir>/completions/completions_<step>.parquet` (`_eval` suffix for evaluation): prompt, completion, one column per reward function, advantage. Also a W&B `completions` table when `report_to` includes `wandb` |
| `log_completions` (online and async GRPO) | off | also prints the per-sample table to the console |

## RL health

For async GRPO with environments, keep `sampling/logratio_mean` on a dashboard
panel. A steady negative drift means one of two things:

- the weight sync to the rollout server is broken;
- a KL-free run is drifting, if `advantage/net_token_mass` stays negative and
  `entropy` climbs after it. `balance_token_mass` and the
  [early stop](../agent-docs/training-methods/grpo/async-grpo/monitoring.md#early-stop) ↗
  address this.

Online GRPO logs TRL's unsigned gap `sampling/sampling_logp_difference/mean`
instead, with the importance-sampling correction on. Every callback and metric:
[Callbacks](../agent-docs/training-methods/callbacks.md) ↗.

## Profiling

Set `enable_torch_profiler: true` to see where the time goes. It captures a
Chrome trace on rank 0, with the EP phases (dispatch, expert compute, combine)
labeled; `profiler_ranks: "all"` captures every rank. `halo run trace-report`
turns the traces into a compute, communication and idle breakdown.

Hangs, memory snapshots and the rest of the toolbox:
[Debugging](../agent-docs/reference/debugging.md) ↗.
