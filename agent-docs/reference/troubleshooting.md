# Troubleshooting

Symptom→fix lookup for OOM, NCCL hangs, DeepEP faults, parallelism-config rejections, NaN loss, and
attention-backend mismatches. Run `/debug` to route a live symptom to the exact helper.
A step that is slow rather than broken is a bottleneck question: read power, not util %
([GPU Training Theory §11](gpu-training-theory.md#watch-power-not-utilization)).

## First things to check

Most "it won't start" reports are environment, not code:

| Check | Why it bites |
|---|---|
| **Right image for the GPU.** `halo:blackwell` on B200 (SM100) / B300 (SM103), `halo:hopper` on H100/H200 (SM90). | The host has **no usable Python** — PyTorch, DeepEP, and Flash Attention live only inside the image. FA3 is Hopper-only, FA4 Blackwell-only; the wrong image gives import errors or kernel hangs. |
| **`--env-file .env` passed to `docker run`.** | The code does not auto-load `.env`. Without it, `WANDB_API_KEY` / `HF_TOKEN` / `AWS_DEFAULT_REGION` are unset, so a run that reports to W&B, pulls a gated repo, or reads an `s3://` dataset fails at that step — each needs your own account. Start from `.env.example`. |
| **No `poetry run` / `uv run` prefix.** | The image installs deps into the system interpreter, so `python` / `torchrun` / `accelerate` / `pytest` run directly. |
| **The scratch mount is actually large.** | The root filesystem is often small and shares the Docker image pool; a path named `/mnt` is not guaranteed to be a separate volume. Check `findmnt` / `df -h`, then point `HF_DATASETS_CACHE`, `TMPDIR`, `HALO_DATA_ROOT`, and logs at whichever mount has capacity. |

## Symptom → cause → fix

| Symptom | Cause | Fix | Deep page |
|---|---|---|---|
| `torch.OutOfMemoryError: CUDA out of memory` | Activations + weights + optimizer states exceed VRAM. | Gradient checkpointing on, lower `per_device_train_batch_size` / `max_length`, or move to a sharded mode. See [OOM](#oom). | [Debugging §3](debugging.md#3-gpu-memory-profiling) |
| Job **hangs at step 0** or first cross-rank sync, no error, then a watchdog timeout | Collective mismatch — one rank issued a collective another never reached (divergent control flow, a straggler still in the dataloader). | Dump every rank's stack; the rank *not* in a collective is the culprit. See [NCCL hang](#nccl-timeout-hang). | [Debugging §4](debugging.md#4-diagnosing-multi-node-hangs) |
| Every GPU at **100% utilization but idle power**, all ranks spinning a CPU core, no NCCL error | A per-rank backward graph: a masked or empty row left disconnected from the loss let autograd prune that row's backward on one rank, so its FSDP2/EP collective never fired. | Keep every row connected to the loss (a value masked downstream is fine). Confirm with the NCCL flight recorder — mismatched collectives at one `collective_seq_id`. | [Debugging §4](debugging.md#4-diagnosing-multi-node-hangs) |
| `ValueError: ep_size=N on a single M-GPU NVLink domain forms K concurrent >2-rank DeepEP dispatch groups` at startup | Racy single-domain multi-group EP: one NVLink domain, `ep_size > 2`, and `ep_group_size` smaller than the domain (e.g. `ep_size=4` on 8 GPUs → two 4-rank groups). The combine barriers race FSDP2's DP-wide collectives — the `legacy` buffer deadlocks, the `elastic` default faults with `CUDA error: Invalid access of peer GPU memory over nvlink`, both with GC on or off. `CUDA_DEVICE_MAX_CONNECTIONS=1` (baked into the images) does not cover it. | Use `ep_size=2` or `ep_size = domain` — the validated shapes. `ParallelismConfig` rejects the racy shape at config time. Attention TP leaves `ep_group_size` untouched, so `ep4 + tp2` hits the same rejection; `ep4 + etp2` raises `ep_group_size` to the domain and is accepted, despite forming the same two 4-rank dispatch groups. | [DeepEP](#deepep-build-runtime-faults) |
| **`EP capacity dedup: a MoE layer dispatched N tokens/rank, over the capacity C cached at forward generation G`** on one rank, its peers stuck in the dispatch | The forward reuses the first MoE layer's all-reduced capacity for every later layer, and a layer outgrew it: either the model's later MoE layers dispatch more tokens than its first, or this forward entered the backbone directly and opened no capacity scope of its own. The raising rank leaves its peers in the dispatch collective until `HALO_DEEPEP_GPU_TIMEOUT_SECONDS`, so it reads as a hang plus one traceback. | A caller that peels the backbone off the wrapper calls `bump_forward_generation()` once per forward (TRL's chunked log-prob path does). Otherwise `HALO_EP_CAPACITY_DEDUP=0` sizes every layer with its own all-reduce — at one arena per layer. | [DeepEP](../infrastructure/deepep.md) |
| `ImportError` / `NVSHMEM` / undefined symbol on `import deep_ep` | Wrong NVSHMEM package, or DeepEP used without a GPU / the right image. | On Blackwell + PyTorch 2.11+cu130, NVSHMEM ships with `nvidia-nvshmem-cu13` — do **not** install `nvidia-nvshmem-cu12` (clobbers headers). Run inside the image. | [DeepEP](../infrastructure/deepep.md) |
| **`FlashAttention-4 produces NaN gradients for this model's head_dim-256 partial-rotary attention`** at load — Qwen3.5 / Qwen3.6 / Qwen3-Next MoE or GLM-4 MoE Lite | FA4's backward goes NaN on head_dim-256 + partial-rotary attention (QK-norm + partial rotary + output gate) and on GLM-4 MoE Lite's MLA (256-wide qk/v, 64-dim rope split). Both families log the same line. | Auto-handled: `resolve_attn_implementation` demotes them to SDPA and says so. The predicate keys on the model alone, so an explicitly forced `attn_implementation: flash_attention_4` is demoted too — nothing to do. On a path that bypasses the resolver, set `attn_implementation: sdpa` yourself. | [Qwen3.5](../models/qwen3_5.md), [GLM-4](../models/glm4.md) |
| **NaN loss**, other models, mid-run | LR too high for bf16, or a precision/grad-reduce mismatch. | AdamWBF16 with stochastic rounding is auto-on under `bf16: true` (FSDP/EP/TP). Lower the LR; set `fp32_grad_reduce: true` for a tighter grad reduce. | [BF16 Optimizer](../optimization/bf16-optimizer.md) |
| **Per-token loss degrades sharply past ~2048 tokens**, or logprobs differ between the image and a host Python install | The NGC image defaults fp32 matmuls to TF32; TF32's 10-bit mantissa collapses adjacent RoPE token positions past 2048, corrupting the positional encoding on every model. | Auto-handled: `configure_float32_matmul_precision` pins fp32 matmuls to `highest` at model load and the image sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0`. Opt into TF32 with `HALO_FP32_MATMUL_PRECISION=high`. | [Configuration](configuration-reference.md) |
| **OOM on Gemma 4 at long sequence**, or FA2 rejects the model | Gemma 4 has `head_dim=512`, above FA2's supported head dim; cuDNN SDPA rejects it too and the math kernel OOMs on the full score matrix. | Auto-routed to memory-efficient SDPA with a manual KV repeat, which is what unlocks seq > 20k. Set `attn_implementation: sdpa` explicitly if needed. | [Gemma 4](../models/gemma4.md) |
| **Hang inside `ptxas`** at build/first kernel, Hopper, building FA2 | FA2's split-K forward kernels hang `ptxas` under CUDA 13.2 on SM90. | The Hopper image strips every `flash_fwd_split_*` source and links 24 throw-stubs in their place; CP prefers FA3 on Hopper for the same reason. Use the prebuilt image; building FA2 for Hopper outside Docker needs the same stub or CUDA 13.1. | [Flash Attention](../optimization/flash-attention.md) |
| `gradient_checkpointing_enable` raises on Zaya | cuDNN Conv1d backward-recompute fault under CUDA 13.2; the toolkit clears the model's GC support flag so it fails at enable time rather than deep in the first backward. | Run Zaya as plain FSDP2 **without GC**, EP without GC, or EP+ETP without GC — the supported configs. TP and CP are unsupported for Zaya. | [Zaya](../models/zaya.md) |
| `fp32_non_ep_params=True cannot combine with fsdp_shard_ep1_experts=True` at model load (MoE, `ep_size: 1`) | The upcast skips every EP-wrapper parameter, so FSDP-managed replicated experts would stay bf16 inside an fp32 shard group and FSDP2 asserts a uniform original parameter dtype at the first forward. Rejected off `config.json` instead, before the process groups and the meta shell — and rank-symmetrically, since under PP a hybrid stack can leave one stage with no MoE layer. | Set `fsdp_shard_ep1_experts: false`, or drop `fp32_non_ep_params`. Under PP/TP/CP the first remedy is itself refused, so dropping `fp32_non_ep_params` is the only one; the message names whichever applies. | [Bailing/Ling](../models/bailing.md#ling-30) |
| `fp32_non_ep_params=True is not supported for gemma4/gemma4_text under Expert Parallelism` at model load (`ep_size > 1` only) | Gemma 4's norms re-emit activations at weight dtype, so the upcast feeds fp32 tokens into DeepEP's 2-byte transport. The gate turns that into a named refusal instead of a raw C++ assert at the first dispatch, after the whole multi-GPU load. | Train Gemma 4 experts-distributed in plain bf16; `fp32_experts` still applies. | [Gemma 4](../models/gemma4.md) |
| `ValueError: ... must divide ...` / `Node-local EP group size (N) cannot exceed the NVLink domain (M)` at startup | `ParallelismConfig` validation rejected the layout. | Fix the degrees so they divide `world_size`; keep TP and node-local EP within one NVLink domain. See [Config rejections](#parallelism-config-rejections). | [Parallelism](../parallelism/index.md) |
| **QLoRA rejected** at model load under EP / TP / PP / grouped-GEMM MoE | The EP lazy loader, the grouped-GEMM MoE loader and TP DTensor sharding all materialize plain de-quantized weights, so bitsandbytes `Params4bit` are lost and PEFT's 4-bit dispatch fails; PP rejects PEFT outright. An `ep_size=1` MoE run hits this through the grouped-GEMM wrappers alone. | QLoRA on standard DDP/FSDP (`accelerate launch`), or CP on a dense model; plain LoRA (no quantization) for EP. TP and EP+TP reject every adapter shape, native expert LoRA included. | [Scale & Limits](scale-and-limitations.md) |
| **`use_hsdp` / `fsdp_reshard_after_forward` / `fsdp_reshard_after_backward` rejected** at trainer construction under QLoRA | `fully_shard` cannot wrap bitsandbytes `Params4bit`, so QLoRA skips FSDP2 and syncs gradients with a post-backward all-reduce instead. The knobs shape a wrap that never happens — allowed through, a multi-node run asking for `use_hsdp` would silently train on flat replicated gradients. | Remove the flag(s), or drop the 4-bit quantization (plain LoRA or full fine-tuning) to get the sharding the config names. | [PEFT](../optimization/peft.md) |
| **Embedding index / OOB crash** right after the dataset map, wrong-model token IDs | A map whose output depends on a closure value the cache fingerprint cannot hash. Tokenizer identity and chat-template hash are already in the key, so a plain model swap cannot collide. | Look for the `Dataset-map cache fingerprint ... skips a value of type ... it cannot fingerprint` warning and thread that value through `cache_key_extras`; clear `HF_DATASETS_CACHE` only as a blunt fallback. | [Data Loading](../parallelism/data-loading.md) |
| **`NVRM: Xid ... 145, RLW_RXPIPE Nonfatal` flooding `dmesg`**, thousands per hour | A physically marginal NVLink lane. Forward error correction absorbs the errors, so the link keeps full bandwidth and no data is at risk — the Xid line is informational. | Run `python scripts/profiling/nvlink_health.py`: it exits 1 only on errors FEC did **not** absorb and reports correctable churn as `marginal`. Escalate for a physical reseat only if a link turns unhealthy or codewords reach the deep FEC bins. | [Debugging](debugging.md) |
| **`N model PARAMETER(s) reached device placement still on the meta device`** | The checkpoint load lost those tensors — a key the loader's disk→model mapping did not resolve, or state the checkpoint does not carry that nothing initialized. They hold no weights, and filling them would train uninitialized memory that differs on every rank. | Check the named keys against the checkpoint's index; load with `ep_lazy_loading=False` to route through `from_pretrained`. | [Expert Parallelism](../parallelism/expert-parallelism.md#model-loading) |
| **`N model BUFFER(s) reached device placement still on the meta device`** | Same gate, buffer side. No supported path should reach it: the lazy shell builds under `init_empty_weights(include_buffers=False)`, so every buffer computes normally in `__init__` — including a config-less rotary whose `inv_freq` cannot be recomputed later (Qwen VL vision). Reaching this gate means a loader lost the buffer, not that the model or its EP config is unsupported. | Report it with the named keys rather than working around it. A zero `inv_freq` degenerates RoPE to NoPE at a plausible loss, which is why the gate is fatal. | [Checkpoints](checkpoints.md) |
| **`Lazy load: tensor from checkpoint key(s) ... has shape ... but model tensor ... expects ...`**, or **`global expert index(es) ... are absent`** at an EP/PP lazy load | The checkpoint and the config disagree: a `patch_vocab`-shrunk checkpoint paired with the base `config.json`, a changed `intermediate_size` or expert count, a truncated per-expert upload, or a stale hub rename. `from_pretrained` catches these itself; the lazy path re-implements the same gates. | Fix the pairing — load the checkpoint with the `config.json` it was saved with, or re-fetch a torn checkpoint. The raise is world-uniform (rank-consensus fenced), so the first error names the real cause. | [Expert Parallelism](../parallelism/expert-parallelism.md#model-loading) |
| **`N checkpoint key(s) align to no model tensor`** at an EP lazy load | Checkpoint keys the loader's mapping never claimed — the same causes as the row above, most often a stale hub rename. This one is a **warning**, not a raise, and it is not rank-fenced: those tensors are skipped and the run proceeds. | Read the named keys before ignoring it. Genuinely surplus keys (a checkpoint carrying extra state this architecture drops) are fine; a renamed weight here means the model tensor was filled by something else or left to the load-coverage gate. | [Expert Parallelism](../parallelism/expert-parallelism.md#model-loading) |
| **`N tensor(s) of <ModelClass> are absent from the checkpoint and were randomly initialized`** at model load | The checkpoint does not carry weights the live model needs — truncated, partially uploaded, or written for a different architecture. Without the gate the run would train part-pretrained, part-noise with no other symptom. | Check the named keys against the checkpoint's index and re-fetch it. A task head the architecture adds on top of the backbone, a tied `lm_head` and the class's own `_keys_to_ignore_on_load_missing` are excused already; `HALO_ALLOW_MISSING_CHECKPOINT_KEYS=1` accepts the rest deliberately. | [Checkpoints](checkpoints.md#load-coverage-gate) |
| **CPU RAM spikes / OOM per node at model load** | Batched per-node loading. Left unset it adapts to the node — `min(4, max(1, local_world_size // 2))`, so 4 ranks at once on an 8-GPU node and 2 on a 4-GPU tray. | Set `max_concurrent_loading` explicitly: toward `1` (fully sequential) on CPU-RAM-constrained machines; `0` (all-parallel) only with ample RAM. | [Configuration](configuration-reference.md) |
| **`The launcher declares world_size=N but ['MASTER_ADDR', 'MASTER_PORT'] is unset`** at startup | A bare `srun --ntasks-per-node=N <script>`: SLURM declares the world through `SLURM_NTASKS` but supplies no `env://` rendezvous, so no process group can be built. | Launch one `torchrun` per node — `srun --ntasks-per-node=1 torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_NODEID …` — or export `MASTER_ADDR`/`MASTER_PORT` identically on every task. | [Launch Recipes](../parallelism/launch-recipes.md#slurm) |
| **`Output filesystem is declared SHARED but N of M ranks … cannot see a file global rank 0 wrote`**, or the PER-NODE mirror image, at startup | The multi-node startup probe wrote a sentinel under `output_dir` from global rank 0 and the declaration contradicts what the ranks see. Declared shared but invisible: only rank 0 writes `trainer_state.json`, so every other node would resume at step 0. Declared per-node but visible everywhere: every node's local rank 0 would write the same checkpoint paths. | Match the declaration to the mount with `DIST_OUTPUT_SHARED_FILESYSTEM` (or the `DIST_SHARED_FILESYSTEM` umbrella), or move `output_dir` — onto the shared mount, or one per node. | [Filesystem Handling](../data/filesystem-handling.md) |
| **`Multi-Node NVLink prerequisite check … failed on N of M rank(s)`** at config time | `NVLINK_DOMAIN_SIZE > gpus_per_node` declares node-local groups that span OS nodes over NVLink, and some rank has no IMEX channels at `/dev/nvidia-caps-imex-channels`, reports a fabric registration other than `COMPLETED`, or sees no fabric clique at all. | Bring up NVIDIA Fabric Manager plus the IMEX service with matching channels on every node (NCCL >= 2.25.2), or drop `NVLINK_DOMAIN_SIZE` to `gpus_per_node` and keep the groups within one OS node. | [Multi-Node](../parallelism/multi-node.md) |
| **env-GRPO steps several× slower on `rollout_backend: sglang` than vLLM**, trainer GPUs "100% util" at idle-class power through backward, invariant to any serving change | SGLang's cross-container sync forces `NCCL_P2P_DISABLE`/`NCCL_SHM_DISABLE` process-global, and FSDP2 reshards after every microstep's backward — one full-model re-all-gather per grad-accum microstep over loopback TCP (~15s each at 20B scale). | `fsdp_reshard_after_backward: false` — keeps params unsharded across the window's microsteps (one bf16 param copy per GPU), leaving one gather per optimizer step instead of one per microstep. | [Data Parallelism](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward), [Rollout Servers](../infrastructure/rollout-servers.md) |
| **`save_sharded_ep=True` rejected at trainer construction** | One of nine shapes, all refused up front so a run cannot train into unmergeable shards: multiple EP groups (`ep_group_size != world_size` — replicas would merge as duplicated experts), `expert_tp_size > 1`, context parallelism, native expert LoRA, `merge_expert_lora_on_save`, a `model_type` no EP layer class claims (no merge transform exists), a family that exports the hub namespace through transformers' save-side revert (Step-3.7 Flash — the merge streams key by key and cannot apply it), a multi-node job whose output filesystem is not shared (per-rank shards scatter across nodes' local disks where `merge_ep_shards.py` never sees a complete set), or a run with no EP layers at all. | Use the gathered save (`save_sharded_ep: false`) — it streams, so it costs no more host memory, and it needs no merge. | [Checkpoints](checkpoints.md#expert-parallelism-ep) |
| **Exact resume warm-restarted although `world_size` is unchanged** | A fingerprint field other than `world_size` changed — any of `ep_size`, `expert_tp_size`, `cp_size`, `tp_size`, `pp_size`, `fsdp_shard_ep1_experts`, `optimizer_class`, `ep_scope`, `use_grouped_gemm`, `hsdp`, `nvlink_domain_size`, `expert_replica_size`. | The warning names every differing field with `saved=`/`current=` — restore it, or accept the warm restart (weights and LR schedule still resume). | [Checkpoints](checkpoints.md#warm-restart-vs-exact-resume-torchrun) |

## OOM

Peak memory is dominated by saved activations, not weights. Cut it in this order:

1. **Gradient checkpointing on** — trades recompute for activation memory. Turn it off only when the
   batch already fits; the recompute costs wall-clock.
2. **Lower `per_device_train_batch_size` or `max_length`** — both scale activations linearly.
3. **Move to a sharded mode.** Dense model that OOMs in FSDP2 → add node-local TP. MoE model that
   OOMs → EP (experts across ranks) or pure ETP (`ep_size=1`, expert FFN sharded).

To find *what* holds memory, dump a CUDA snapshot and drag it onto
[memory_viz](https://pytorch.org/memory_viz) ([Debugging §3](debugging.md#3-gpu-memory-profiling)).

## NCCL timeout / hang {#nccl-timeout-hang}

A hang is almost always a collective mismatch: one rank issues an all-reduce / all-gather / barrier
that another never reaches, so NCCL blocks until the watchdog fires.

```bash
python scripts/profiling/py_spy_diag.py dump   # py-spy stacks for EVERY rank on the node
```

The container needs `--cap-add=SYS_PTRACE` to attach. Dumps land in `$TMPDIR/halo_diag_stacks`, and
the rank *not* in a collective is the culprit.
For the definitive "which collective did rank K miss?", enable the NCCL flight recorder
(`TORCH_NCCL_TRACE_BUFFER_SIZE`, `TORCH_NCCL_DUMP_ON_TIMEOUT`).

One class of hang is refused up front instead: a job whose ranks disagree on a toolkit EP or timeout
knob (`HALO_EP_CAPACITY_DEDUP`, the DeepEP wire parameters, `HALO_GRAD_BUCKET_MB`,
`DIST_NCCL_TIMEOUT_MINUTES`, `DIST_STORE_TIMEOUT_HOURS`) raises `Rank-uniform toolkit environment` in
distributed setup, before the weight load — a per-node `--env-file` rollout that reached only some
nodes would otherwise diverge the collective counts from the second MoE layer on, naming nothing.

Raise the watchdog for large cross-node all-to-all or gathered saves — a gathered EP/TP save holds
non-writers in its trailing barrier while the writer serializes the checkpoint. For 100B+ gathered
saves, size the timeout to checkpoint size ÷ filesystem write speed with margin. Main-first work
(model downloads, dataset load, tokenize/pack) is exempt: the peers wait on the c10d store
(`DIST_STORE_TIMEOUT_HOURS`, default 4 h), not inside a collective.

```bash
export DIST_NCCL_TIMEOUT_MINUTES=60   # default 30; PyTorch's own default is 10
```

Full procedure: [Debugging §4](debugging.md#4-diagnosing-multi-node-hangs).

## DeepEP build / runtime faults {#deepep-build-runtime-faults}

DeepEP is required for EP — there is no NCCL fallback. Two failure modes dominate:

- **`import deep_ep` fails (NVSHMEM / undefined symbol).** On Blackwell + PyTorch 2.11+cu130,
  NVSHMEM ships as `nvidia-nvshmem-cu13`; installing `nvidia-nvshmem-cu12` clobbers the headers. Use
  the prebuilt image.
- **`Assertion exception ... != NCCL_GIN_TYPE_NONE` / "NCCL GIN is unavailable" at buffer init on
  cross-node EP (EFA).** Proxy GIN could not come up on at least one node. Check, on **every** node:
  the `gdrdrv` kernel module is loaded (`ls /dev/gdrdrv` — the host must install the gdrcopy
  driver), the container was started with `--device /dev/gdrdrv`, and `NCCL_GIN_TYPE=2` is exported.
  One node missing any of the three fails the whole job with this assertion while node-local EP on
  the same machine runs fine. See [DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa).
- **`CUBLAS_STATUS_EXECUTION_FAILED` / DeepEP `combine.hpp` `CUDA_ERROR_LAUNCH_FAILED` + Xid 43 on
  several ranks at once, single-node EP.** Before reading it as a kernel or hardware fault, grep the
  full log for `OutOfMemoryError` — a rank-local OOM inside a distributed step abandons the DeepEP
  barrier its peers are waiting in (`DeepEP NVLink barrier timeout` lines), and as its context tears
  down the peers die on launch failures inside whatever kernel is on-stream. The OOMing rank's
  traceback is the primary failure and is easily buried under the collateral; the trainer logs a
  `RANK n: ... THIS RANK IS THE PRIMARY FAILURE` banner for it, and warns after the first optimizer
  step when a rank's peak sits above ~92% of its device. On MoE the usual cause is routing skew — a
  cold (unbalanced) router concentrates dispatch buffers and expert activations on hot ranks, and
  under gradient checkpointing every MoE layer's saved dispatch/combine results scale with it; under
  `bias_update` the skew (watch `moe/load_max`) falls over the first few hundred steps, so a batch
  shape that OOMs at cold start can fit when resumed from a balanced checkpoint. Cold-start at the
  smaller batch, or lower `per_device_train_batch_size` / `max_length`.
- **`Dispatch CPU wait ... received count 0`, or Xid 109 `CTX SWITCH TIMEOUT` → Xid 43 /
  `unspecified launch failure`, on cross-node EP (EFA).** The dispatch exceeded the proxy-GIN
  per-rank payload ceiling and wedged in transit — the Xid 43 / launch-failure form surfaces in
  whatever kernel is on-stream (often the expert GEMM) and reads like a compute bug, but the
  trigger is the oversized dispatch. The dispatcher rejects `> 8192` tokens/rank at buffer sizing
  (`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`); if this fires at runtime instead, the guard was
  disabled or raised. Lower `per_device_train_batch_size`/`max_length`, or use `ep_scope=node`
  with DP across nodes. See [DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa).

`CUDA_DEVICE_MAX_CONNECTIONS=1` is baked into the images as `ENV` (the driver latches it at the
`deep_ep` import's `cuInit`, so a Python `os.environ` write is too late). Free default — neutral on
dense/ep2, **+9.7%** on ep8. Details: [DeepEP](../infrastructure/deepep.md).

## Parallelism config rejections

Which axis combinations may run is an **allowlist** (`SUPPORTED_AXIS_SETS` in
`src/distributed/parallelism_config.py`), checked before any rank math — anything unlisted is
rejected at startup with the reason rather than deadlocking later. The permitted sets: plain data
parallelism, each axis alone (EP, ETP, TP, CP, PP), and **EP+TP**, **EP+CP**, **EP+ETP**, **PP+EP**,
**PP+ETP** — the PP sets not yet runnable this release, since `pipeline_parallel_size > 1` is
rejected at config time ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)).

| Combo | Status | Use instead |
|---|---|---|
| `TP + CP` | Not supported | DTensor mesh conflicts with the CP process groups — pick one |
| `ETP + CP` | Not supported | ETP sub-EP groups break CP sequence reconstruction |
| `TP + ETP` (`ep_size=1`) | Not supported | Pure ETP, or EP+TP |
| `EP + TP + ETP` | Not supported | EP+TP, or pure ETP — not both |
| `PP + TP` · `PP + CP` · `PP + EP + TP` · `PP + EP + CP` · `PP + EP + ETP` | Not supported | PP composes with the expert axes only — EP, or pure ETP (`ep_size=1`), never both |
| Racy single-domain multi-group EP (`ep_size > 2` with `ep_group_size` below the NVLink domain) | Rejected at config time | `ep_size=2` or `ep_size = domain`. Raising `ep_size × expert_tp_size` to the domain (`ep4 + etp2` on 8) passes the gate — the only 4-way expert split left on one node |
| `PP` + `gradient_checkpointing_kwargs: {use_reentrant: true}` | Rejected at config time, EP or not (a reentrant forward runs under `no_grad`, so FSDP2 registers no pre-backward hooks) | Drop the kwarg — the default `use_reentrant: false` is the supported setting, and PP+EP selects it silently when the kwarg is unset |
| `PP` + expert-LoRA · `PP` + HSDP · `PP` + `fsdp_reshard_after_forward` · `PP` + `lowp_precision != bf16` · `PP` + `fsdp_shard_ep1_experts: false` | Rejected at config time. The last fires only where expert wrappers would apply — `use_grouped_gemm` on (the default) or `ep_size > 1`; setting `use_grouped_gemm: false` at `ep_size: 1` passes. The check runs before any model config, so it does not distinguish dense from MoE | Drop the flag, or drop PP |
| `PP` + PEFT/LoRA | Rejected at trainer construction | Adapter injection resolves module names on the full model tree, which a stage cannot satisfy. Full fine-tune under PP, or LoRA without PP |
| `PP` + a `tie_word_embeddings` checkpoint | Rejected at model split | `embed_tokens` lands on stage 0 and `lm_head` on the last stage, so each copy gets only its own path's gradient and they diverge from step 1 — `reconcile_tie_word_embeddings` needs both keys in one rank's state dict. Use an untied checkpoint, or drop PP |
| `PP` + `moe_balancing` resolving to `aux_loss` with `router_aux_loss_coef > 0` | Rejected at model split | A stage runs the backbone and applies the head itself, so the `*ForCausalLM.forward` that adds the aux loss never executes — routing collapses silently. Use `moe_balancing: bias_update` (family permitting), `bias_update_transient` on families with no exportable slot, or `none` |
| `moe_balancing: aux_loss` on a model whose `forward` omits `output_router_logits`, with `router_aux_loss_coef > 0` | Rejected when the perf callbacks build | HF's config fallback lives on that parameter, so the flag never reaches the aux term while still switching router-logit recording on — a `[tokens, num_experts]` plane per MoE layer, every forward. Multimodal wrappers (`Qwen3_5MoeForConditionalGeneration`) are the case; use `bias_update_transient` (those architectures have no exportable bias slot, so `auto` resolves to `none` there) |
| `moe_balancing: bias_update` rejected: "would train a routing bias that NO EXPORT CARRIES" | Rejected when the perf callbacks build | The family's architecture has no checkpoint slot for a selection bias (Qwen3, Qwen3.5/3.6, Mistral4, Cohere2 MoE) — the exported model would silently serve without the bias it trained with. Opt in deliberately with `bias_update_transient`, or use `none` |
| `PP` + a model declaring multi-token-prediction tail layers | Rejected at model split | MTP layers re-embed `input_ids` inside the backbone forward — a second cross-stage input stream a hidden-state cut cannot carry. Disable the MTP head, or drop PP |
| Expert LoRA + `expert_tp_size > 1` | Rejected at config time by `ParallelismConfig`, before the checkpoint downloads (`EPConfig` repeats the identical raise at group construction) | The replicated adapter half gets partial gradients under expert TP and drifts across ranks. Use EP without ETP for expert adapters, or drop expert projections from `lora_target_modules` |
| `use_peft: true` with no attention target AND no EP layer that built expert adapters | Rejected at PEFT setup | No adapter would be created and the base stays fully trainable — a full fine-tune at the LoRA learning rate. A genuine expert-only run is fine: its list is empty only after the peel. Name the modules to adapt, or set `use_peft: false` |
| Expert projections in `lora_target_modules` + `use_dora` | Rejected before model load, at the expert peel | DoRA's magnitude decomposition has no grouped implementation, so it would apply to the attention half only. `use_rslora` **is** carried (expert scaling `alpha/sqrt(r)`). Drop `use_dora`, or drop the expert projections |
| `lora_target_parameters` under EP | Rejected at PEFT setup | PEFT would wrap the EP layer itself and swap it into its parent — adapters outside the EP gradient sync and outside both EP validators. Name the expert projections in `lora_target_modules` instead. Allowed without EP wrappers, where it is ordinary PEFT |
| Expert-only `lora_target_modules` + `lora_modules_to_save` | Rejected at PEFT setup | The model is never PEFT-wrapped, so those modules would stay frozen. Add an attention target (they route to stock PEFT, which owns `modules_to_save`), or drop `lora_modules_to_save`. Either way `merge_expert_lora_on_save` still produces a merged servable checkpoint — it folds the attention half of a mixed run too |
| Resuming expert adapters no EP layer can receive | Rejected in `apply_ep_lora_adapters` | EP off, `use_grouped_gemm: false`, or the expert projections dropped — every saved expert delta would be discarded while the restore reported success. Resume with the run's expert configuration |
| `PP` + `activation_offloading` | Rejected at trainer construction | TRL applies it by wrapping `training_step`, which the PP schedule-driven step bypasses — it would silently never engage. Drop the flag |
| LoRA/PEFT + TP (TP-only or EP+TP) | Rejected at construction | The adapter is not in the TP DTensor graph and trains rank-inconsistent. Native EP expert adapters are counted by their own check (`has_ep_lora`), so an **expert-only** EP+TP run is refused too. LoRA works under FSDP/CP/EP/ETP |
| QLoRA + EP / TP / PP / grouped-GEMM MoE | Rejected at load | QLoRA on DDP/FSDP, or CP on a dense model; plain LoRA for EP |
| CP + a trainer using `logits_to_keep` / global log-prob sums / full-sequence pooling / dual models | Rejected at trainer construction (`<TrainerClass> does not support Context Parallelism (CP)`) | The trainer's `_supports_cp` class attribute is the gate, and it defaults off — nothing inspects the loss for CP-safety. Only SFT and SMPO declare it |
| PP + a trainer needing a live reference model or a second forward | Rejected at construction | DPO/KTO are precompute-only under PP; a `kl_beta > 0` offline-GRPO reference would be scored once before training by a pipeline sweep |
| Any `pipeline_parallel_size > 1` | Rejected at config time | Pipeline parallelism is not yet available in this release |

TP and node-local EP must stay within one NVLink domain. EP can cross domains under
`ep_scope: global`. The full matrix is in [Parallelism](../parallelism/index.md) and the per-mode
guides: [Expert](../parallelism/expert-parallelism.md) ·
[Tensor](../parallelism/tensor-parallelism.md) · [Context](../parallelism/context-parallelism.md) ·
[Expert-Tensor](../parallelism/expert-tensor-parallelism.md) ·
[Pipeline](../parallelism/pipeline-parallelism.md).
