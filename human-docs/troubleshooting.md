# Troubleshooting

The failures people hit most, with the fix. The full symptom → cause list is
[Troubleshooting](../agent-docs/reference/troubleshooting.md) ↗ in the
reference.

## First things to check

- **You are inside the right Docker image.** `halo:blackwell` for B200/B300,
  `halo:hopper` for H100/H200. The host has no usable Python, and an import
  error for `flash_attn` or `deep_ep` almost always means the wrong image.
- **The container got `--gpus all` and `--env-file .env`.** Halo never loads
  `.env` itself. The GPU `make` targets (`train`, `test-gpu-*`) and the compose
  `training` service pass it; a container you start by hand needs the flag. A
  missing `HF_TOKEN` or `WANDB_API_KEY` usually means it is missing.
- **Caches point at a disk with space.** Run `df -h` on wherever `HF_HOME`,
  `HF_DATASETS_CACHE`, `TMPDIR` and `HALO_DATA_ROOT` resolve.

## Single-node

| Symptom | Cause → fix |
| --- | --- |
| CUDA out of memory | Activations dominate. In order: `gradient_checkpointing: true` (adds about 20–30% to step time), a smaller `per_device_train_batch_size` or `max_length`, then sharding (TP for dense, EP or ETP for MoE) or LoRA/QLoRA. |
| Config rejected at startup (`must divide`, `not supported`, …) | The validator refuses shapes that would hang or crash mid-run. The message names the rule; valid shapes are in [Parallelism](parallelism.md). |
| `expert_parallel_size=N on a single M-GPU NVLink domain forms K concurrent >2-rank DeepEP dispatch groups` | EP between 2 and the NVLink domain size on a single-domain job. Use `ep_size=2`, `ep_size` = the domain size, or `ep4 + etp2` for a 4-way split on 8 GPUs. `ep4 + tp2` is rejected the same way. |
| Missing dataset column | Each method reads fixed columns ([Datasets](data.md)). Mixing sources keeps only the columns common to all of them. |
| Host RAM spike or OOM while loading the model | By default half the node's ranks load at once, capped at 4. Set `max_concurrent_loading: 1`. |
| Loss degrades only past ~2048 tokens | TF32 rounding corrupts long-context RoPE. Halo pins fp32 matmuls to full precision at model load; you see this only after setting `HALO_FP32_MATMUL_PRECISION` to `high` or `medium`. Unset it. |
| Garbage output or an index crash right after dataset mapping | A map whose output depends on a value the cache fingerprint cannot hash. Grep the log for `Dataset-map cache fingerprint`. In code, pass that value through `cache_key_extras`; from outside, clear `HF_DATASETS_CACHE`. |
| Zaya raises when gradient checkpointing is on | Zaya cannot train with it (the recompute faults in cuDNN). Set `gradient_checkpointing: false`. |
| `fp32 training is not supported under Expert Parallelism` at model load | `bf16: false` without `fp16` means fp32, and DeepEP's buffer is sized for 2-byte tokens. Train in bf16, or keep fp32 masters with `fp32_non_ep_params` / `fp32_experts` (Gemma 4 refuses `fp32_non_ep_params` under EP). |
| A write error mid-run | A cache or output landed on the small root filesystem. Recheck the four cache variables. |
| `Resume checkpoint(s) … are incomplete on at least one node` | A save stopped partway, so `trainer_state.json` (written last) is missing. Resume from an earlier checkpoint or remove the incomplete one ([Checkpoints](checkpoints.md#resume)). |
| `optimizer_meta.pt is missing beside the per-rank optimizer shards` (or `is unreadable`) at resume | The optimizer half of an interrupted save. Resume from an earlier checkpoint, or set `allow_optimizer_warm_restart: true` to keep the weights and start fresh optimizer moments. |

## Multi-node and clusters

| Symptom | Cause → fix |
| --- | --- |
| Nodes never join, or the rendezvous times out | The torchrun arguments differ across nodes, a node started late, the master address is not routable on the fabric (a TCPStore connect timeout naming the host and port), or a stale process holds the port (`pkill -f torchrun`). |
| Hang at step 0, then a watchdog timeout | A rank never reached a collective. Dump every rank's stack ([below](#getting-eyes-on-a-hung-run), once per node); the rank that is not in a collective is the culprit. |
| Every GPU at 100% utilization but idle power draw, no error | A row disconnected from the loss on one rank pruned its gradient collective. Keep every row connected to the loss ([Debugging](../agent-docs/reference/debugging.md) ↗). |
| Watchdog timeout during a large gathered checkpoint save | Slow work outlasted the watchdog. Raise `DIST_NCCL_TIMEOUT_MINUTES` (default 30). |
| A rank waits hours, then aborts during a download, a dataset map or pack, or a queued model load | The rank going first outlasted `DIST_STORE_TIMEOUT_HOURS` (default 4). Raise it. |
| `Output filesystem is declared SHARED but …` (or `… declared PER-NODE …`) at startup | The filesystem flag contradicts the mount. Set `DIST_OUTPUT_SHARED_FILESYSTEM` (or the `DIST_SHARED_FILESYSTEM` umbrella) to `1` for a shared FS or `0` for per-node disks, or move `output_dir`. |
| `OSError: [Errno 116] Stale file handle` while loading a model or dataset cache | A cross-node read-after-write on NFS/EFS. Set `DIST_INPUT_SHARED_FILESYSTEM=0` and keep the output side shared ([Clusters](clusters.md#storage)). |
| `Resolving the checkpoint '<path>' failed on K of N rank(s)`, or `The checkpoint '<repo>' each rank resolved differs across ranks` | Some nodes cannot see the model source, or their Hub caches hold different commits. Put the checkpoint or `HF_HOME` on a shared mount, or set `DIST_INPUT_SHARED_FILESYSTEM=0`, and pin `model_revision` to one commit. |
| `Dataset <path>: the ranks … that must hold the same rows loaded different ones`, or `The shard index of <path> differs across ranks` | Nodes read different copies of the dataset. Re-sync or delete the stale copy (`$HALO_DATA_ROOT/s3_datasets/<md5>` for an S3 source) and relaunch. |
| Slow cross-node traffic on AWS | EFA needs its own environment ([Clusters](clusters.md#network-fabric)). The same variables slow down an InfiniBand cluster if left set. |
| `Xid 145` NVLink messages flood dmesg | Usually benign FEC churn. `halo run nvlink-health` exits non-zero only on real faults; trust it over dmesg volume. |

## RL runs (vLLM / SGLang)

Server setup is in [Rollout Servers](rollout-servers.md), which also covers
tool parsers, thinking budgets and port binding.

| Symptom | Cause → fix |
| --- | --- |
| The weight-sync group never forms | Both containers need `network_mode: host`; a bridge network hides the ports the server dials back on. Under SGLang, serve from this repo's `Dockerfile.sglang` image. |
| Trainer exits at start with `Errno 98` on the weight-sync port | The default port `51216` (plus one per extra server) sits in Linux's ephemeral range. Reserve it in `net.ipv4.ip_local_reserved_ports` or set `group_port` outside `32768–60999`. |
| Weight sync hangs at the first collective after the group formed | The containers run different NCCL transports or fabric builds. Keep `NCCL_SOCKET_IFNAME` off Docker's `veth` interfaces (the compose default), serve from Halo's images with the same fabric setup on both ends, and check with `halo run weight-sync-transport --server-url http://<server>:8000 --expect efa` ([Rollout Servers](rollout-servers.md#across-nodes)). |
| Rewards look fine but the policy degrades | Watch `sampling/logratio_mean` ([Monitoring](monitoring.md#rl-health)). Serve MoE models with `--moe-backend triton`; the auto-selected backends silently corrupt synced expert weights. |
| Rollouts much slower in the run than on the server alone | The vLLM engine core is one CPU thread, and the trainer and sandboxes compete for it. Give each server its own cores (`--cpuset-cpus`) and run one engine per GPU. |
| Startup rejection under `rollout_backend: sglang` | Weight sync is supported per family and per engine ([Supported Matrix](supported-matrix.md#rollout-engines)). `rollout_max_thinking_tokens` and `carry_reasoning` are vLLM-only. Drop the knob the message names, or use `rollout_backend: vllm`. |
| `ncclP2pImportShareableBuffer ... invalid argument` in the SGLang log on the first update | cuMem differs between the containers. Keep the compose default `NCCL_CUMEM_ENABLE=1` on the server and restart it, since it holds a half-written model. |
| `routing_replay: rollout` captures nothing on SGLang | Start the server with `--enable-return-routed-experts --moe-runner-backend triton` and without `--enable-torch-compile`. The trainer refuses capture for Gemma 4 and Zaya. Bailing raises at the first capture on SGLang: serve it without `SGLANG_ENABLE_R3`. |
| Rollout server restarts forever with `VLLM_ENABLE_R3` or `SGLANG_ENABLE_R3` set | Routed-experts capture is MoE-only; the engine exits on a dense model and the compose `restart` policy loops it. Unset the variable for a dense model. |
| SGLang exits at start with "Unsupported head dimensions" | GLM-4 MoE Lite on Blackwell. Start the server with `--attention-backend triton` (`SGLANG_ATTENTION_BACKEND=triton` for compose). |
| `GENERATION is wedged` at startup | A previous trainer died attached to the vLLM engine. Restart the vLLM container before relaunching. |
| "Excluded N of M `lora_target_modules` matches from PEFT injection" at startup | Expected. Those modules have no stock LoRA fit (a Gemma 4 vision-tower projection wrapper, DeepSeek-V4's grouped `o_a_proj`); everything else is adapted. |
| `a vLLM thinking budget can be enforced but the importance-sampling correction is off` | Only the correction keeps the budget's forced tokens out of the loss. Turn `train_on_sampled_tokens` and `vllm_importance_sampling_correction` back on (both default on). |
| A `bubblewrap` sandbox fails to start (`Can't mount proc on /newroot/proc`, `loopback: Failed RTM_NEWADDR`) | The container lacks the rights the sandbox needs. Run the trainer as root in a container started with `--init` and either `--privileged` or `--cap-add SYS_ADMIN --security-opt seccomp=unconfined --security-opt apparmor=unconfined`; with the `--cap-add` set, and in any GPU container, also mount a clean `/proc` ([Async GRPO](training-methods/async-grpo-environments.md#the-environments)). |

## Getting eyes on a hung run

Attach from a second shell in the same container:

```bash
halo run py-spy-diag dump                 # Python stacks of every rank on this node
halo run py-spy-diag record --duration 30 # flame graphs (dataloader stalls)
halo run nvlink-health                    # NVLink preflight and verdict
```

py-spy needs the container started with `--cap-add=SYS_PTRACE` when the host's
`kernel.yama.ptrace_scope` is 1 or 2; at 3 it cannot attach. Start every
training container with the flag. The `make` targets do not add it.

For where the time goes, use `enable_torch_profiler: true` and
`halo run trace-report` ([Monitoring](monitoring.md#profiling)). The full
toolbox, including the NCCL flight recorder for mismatched collectives, is
[Debugging](../agent-docs/reference/debugging.md) ↗.
