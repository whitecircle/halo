# Troubleshooting

What people actually hit. The exhaustive symptom→cause list is
[Troubleshooting](../agent-docs/reference/troubleshooting.md) ↗ in the
reference.

## First things to check

- You are **inside the Docker image**, and it's the right one: `halo:blackwell`
  for B200/B300, `halo:hopper` for H100/H200. The host has no usable Python;
  import errors for `flash_attn` or `deep_ep` almost always mean the wrong image.
- The container got `--gpus all` and `--env-file .env`. Missing
  `HF_TOKEN`/`WANDB_API_KEY` failures mean the second one — `.env` is never
  loaded automatically.
- Caches point at a disk with space: `df -h` on whatever `HF_HOME`,
  `HF_DATASETS_CACHE`, `TMPDIR`, and `HALO_DATA_ROOT` resolve to.

## Single-node

| Symptom | Cause → fix |
| --- | --- |
| CUDA out of memory | Activations dominate. In order: `gradient_checkpointing: true` (~19% slower), then lower `per_device_train_batch_size` / `max_length`, then shard — TP for dense, EP or ETP for MoE, or LoRA/QLoRA. |
| Config rejected at startup (`must divide`, `not supported`, …) | Working as intended: the validator refuses shapes that would hang or crash mid-run. The message names the rule; valid combinations are in [Parallelism](parallelism.md). |
| `ep_size=N on a single M-GPU NVLink domain forms K concurrent >2-rank DeepEP dispatch groups` | The rejected middle ground for single-node EP. Use `ep_size=2`, `ep_size` = the GPU count, or `ep4 + etp2` for a 4-way expert split on 8 GPUs. `ep4 + tp2` hits the same rejection. |
| Missing dataset column | Each method needs fixed columns ([Datasets](data.md)); combining sources keeps only columns common to all of them. |
| CPU RAM spike or OOM while loading the model | Default loads half the node's ranks concurrently, capped at 4. Set `max_concurrent_loading: 1`. |
| Loss degrades only past ~2048 tokens | TF32 rounding corrupting long-context RoPE. The image pins fp32 matmuls to full precision, so you only see this after setting `HALO_FP32_MATMUL_PRECISION=high` — unset it. |
| Garbage output / index crash right after dataset mapping | A map whose output depends on a closure value the fingerprint can't hash. Grep the log for `Dataset-map cache fingerprint` and thread that value through `cache_key_extras`; clearing `HF_DATASETS_CACHE` is the blunt fallback. |
| `cuDNN` error in backward on Zaya | Zaya doesn't support gradient checkpointing; set `gradient_checkpointing: false`. |
| `fp32 training is not supported under Expert Parallelism` at model load | `torch_dtype: float32` with `expert_parallel_size > 1` — DeepEP's dispatch buffer is sized for 2-byte tokens. Train in bf16, or keep fp32 masters via `fp32_non_ep_params` / `fp32_experts`. Dense fp32 and pure ETP are unaffected. |
| Run dies mid-run with a write error | A cache or output landed on the small root filesystem after all. Recheck the four cache variables. |

## Multi-node and clusters

| Symptom | Cause → fix |
| --- | --- |
| `ProcessGroup not initialized` / nodes never join | The torchrun arguments differ across nodes, a node started late (10-minute rendezvous window), the master address isn't fabric-routable, or a stale process holds the port (`pkill -f torchrun`). |
| Hang at step 0, watchdog timeout minutes later | A rank never reached a collective. Grab all-rank stacks (below); the rank *not* in a collective is the culprit. |
| Every GPU at 100% util but idle power draw, no error | A data-dependent backward graph: a row disconnected from the loss on one rank prunes its gradient collective. Keep every row connected to the loss — [Debugging](../agent-docs/reference/debugging.md) ↗. |
| Timeout during dataset prep or a huge checkpoint save | Legitimate slow work outlasting the watchdog. Raise `DIST_NCCL_TIMEOUT_MINUTES` (default 30). |
| "Checkpoint not found" on resume, or duplicate per-node saves | The filesystem flag doesn't match reality: `DIST_SHARED_FILESYSTEM` is `1` for a shared FS, `0` for per-node disks. |
| `OSError: Stale file handle` while loading a model or dataset cache | A cross-node read-after-write on NFS/EFS. Set `DIST_INPUT_SHARED_FILESYSTEM=0` and leave the output side shared — see [Clusters](clusters.md). |
| A rank waits hours then aborts during a download or corpus pack | The rank going first outlasted `DIST_STORE_TIMEOUT_HOURS` (default 4). Raise it. |
| Slow cross-node traffic on AWS | EFA needs opt-in env (`FI_PROVIDER=efa`, …) — see [Clusters](clusters.md). Those same vars degrade an InfiniBand cluster if left set. |
| `Xid 145` NVLink messages flooding dmesg | Usually benign FEC churn. `halo run nvlink-health` exits non-zero only on real faults — trust it, not dmesg volume. |

## RL runs (vLLM / SGLang)

| Symptom | Cause → fix |
| --- | --- |
| Weight-sync group never forms ("1/2 clients joined") | Both containers must run `network_mode: host`; the sync rendezvouses on an ephemeral port a bridge network won't publish. Under SGLang, check the server came from this repo's `Dockerfile.sglang` — the upstream image ships a different NCCL. |
| Startup rejection under `rollout_backend: sglang` | SGLang refuses distributed experts, `rollout_max_thinking_tokens`, and every MoE family but GPT-OSS ([Supported Matrix](supported-matrix.md#rollout-engines)). Drop the knob it names, or use `rollout_backend: vllm`. |
| SGLang step time far above the same config on vLLM | Its cross-container sync forces process-global socket NCCL, so FSDP2 re-gathers the whole model over loopback TCP once per grad-accum microstep. Set `fsdp_reshard_after_backward: false` (the shipped SGLang configs do); rejected under TP. The step still runs slower than vLLM's. |
| `routing_replay: rollout` captures nothing on SGLang | The server needs `--enable-return-routed-experts --moe-runner-backend triton`; the fused runners bypass the capture hook. Keep `--enable-torch-compile` off. |
| `GENERATION is wedged` at startup | A previous trainer died attached to the vLLM engine. Restart the vLLM container before relaunching. |
| Rewards fine, policy silently degrades | Under environmental GRPO, watch `sampling/logratio_mean` — a steady negative drift means broken weight sync. Also serve MoE models with `--moe-backend triton`; the auto-selected backends silently corrupt synced expert weights. |

## Getting eyes on a hung run

Attach from a second shell in the same container (needs `--cap-add=SYS_PTRACE`):

```bash
halo run py-spy-diag -- dump                 # all-rank Python stacks
halo run py-spy-diag -- record --duration 30 # flame graphs (dataloader stalls)
halo run nvlink-health                       # NVLink preflight / verdict
```

For where-does-the-time-go questions, `enable_torch_profiler: true` and
`halo run trace-report` — see [Monitoring](monitoring.md). The full toolbox,
including the NCCL flight recorder for mismatched collectives, is in
[Debugging](../agent-docs/reference/debugging.md) ↗.
