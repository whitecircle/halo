# Configuration Reference

Every training script takes a YAML config file as its first positional argument. Config classes extend HuggingFace `TrainingArguments`, so all its parameters (learning rate, batch size, scheduler, …) are valid everywhere. Any field can be overridden on the command line; CLI wins:

```bash
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml \
    --learning_rate=0.00001 --num_train_epochs=2 --per_device_train_batch_size=2
```

Working configs per method and model family live under `examples/`.

---

## Sequence length: caps vs generation budgets

Three knob names carry the length settings. They mean the same thing everywhere; only the *mechanism* that enforces them differs per trainer, because some data can be safely truncated and some cannot.

| Knob | Meaning |
|---|---|
| `max_length` | Total tokenized sequence budget (prompt + completion). `null` or any non-positive value resolves to the **model's context window** at launch (`resolve_length_to_context`, which reads `max_position_embeddings` / `max_seq_length` / `n_positions` off the **text** sub-config so composite VLM configs resolve too, then falls back to `tokenizer.model_max_length`). Also becomes `tokenizer.model_max_length` for the run. |
| `max_prompt_length` | Prompt share of the budget. |
| `max_completion_length` | Completion share of the budget, or — under RL — the number of tokens the policy may generate. Environmental GRPO overwrites it with `rollout_max_tokens` at construction (the rollout budget is the generation cap there). |

*Omitting* a field falls back to the dataclass default in the tables below. Writing `max_length: null` opts into the model's context window; `null` on `max_prompt_length` / `max_completion_length` means whatever the per-trainer row below says (no cap, no prompt filtering, or a derived share of `max_length`) — never the context window.

### Per-trainer mechanism

| Trainer (config) | `max_length` | `max_prompt_length` | `max_completion_length` |
|---|---|---|---|
| SFT (`SFTConfig`) | **drop**: the chat processor tokenizes untruncated and discards any conversation over the budget (cutting one mid-turn corrupts it). Under `packing` it is also the fixed pack size, which must be explicit — the script raises on `null` | — | — |
| DPO (`DPOConfig`) | TRL truncates the **concatenated** prompt + completion `keep_start` | — (no training-side cap; see below) | — |
| KTO (`KTOConfig`) | TRL truncates the assembled prompt + completion to it, **keeping the start** (TRL's own field help saying "from the left" is stale — the code slices `[:max_length]`) | — | — |
| SMPO (`SmoothMarginPOConfig`) | total budget, split into the two shares below (`resolve_length_budget`) | truncate per `truncation_mode`; `null` → **half of `max_length`** | truncate (keeping the terminal EOS); `null` → the `max_length − max_prompt_length` remainder. The two shares must fit inside `max_length` |
| Reward (`RewardConfig`) | TRL **filters**: pairs whose chosen *or* rejected exceeds it are dropped, not truncated | — | — |
| Classification (`ClassificationConfig`) | **truncate** at tokenization on the script path; a dataset handed straight to the trainer as raw `text`/`label` is tokenized untruncated and **filtered** instead | — | — |
| Distillation (`DistillationConfig`) | over-length conversations are **dropped**, not truncated | — | — |
| Embedding (`EmbeddingConfig`) | installed as SentenceTransformers' `max_seq_length` (truncates) | — | — |
| Offline GRPO (`OfflineGRPOConfig`) | pipeline-parallel fixed shape only (`null` → the two shares' sum; **rejected** outside PP) | **left**-truncate (keep the tokens nearest the completion) | truncate the stored completion (EOS only within the budget); also the `dr_grpo` loss normalizer |
| Online GRPO — RLVR (`GRPOConfig` + script args) | — | dataset **filter**: over-length prompts are dropped, never truncated | **generation budget** (TRL `max_new_tokens` / vLLM `SamplingParams`) |
| Environmental GRPO (`AsyncTrainingConfig` + script args) | — | dataset **filter** (as above) | not a knob — the script overwrites it with `rollout_max_tokens` (a YAML value is discarded), leaving it only as TRL's `dr_grpo` normalizer |

Two knobs are *not* a plain cap:

- **DPO has no training-side prompt cap.** TRL truncates the concatenated sequence `keep_start`, so an over-long prompt eats its own completion — filter them in the dataset. `DPOScriptArguments.generation_max_prompt_length` bounds only the eval-time generation dataset.
- **Environmental GRPO never truncates a trajectory.** `rollout_max_tokens` is the per-turn generation budget; the multi-turn trajectory accumulates across turns and is bounded only by the served model's context window — one that exceeds it fails ([trajectory length](../training-methods/grpo/environmental-grpo.md#trajectory-length)).

### The tokenizer's window

Every training script routes its `tokenizer.model_max_length` write through one of two seams in `src/training/script_runner.py`:

- `apply_max_length` — every single-budget script (SFT, DPO, KTO, SMPO, reward, classification, distillation). It resolves `max_length` against the context window, writes the resolved number back onto the config, and pins the same number, so collators, length filters and packers all read one already-resolved value.
- `apply_prompt_completion_window` — the GRPO family, which has no single `max_length`. The window is `max_prompt_length + max_completion_length`, and it is pinned **only when both halves are bounded**. HF resolves any `truncation=True, max_length=None` call against `model_max_length`, so pinning a partial sum would turn the *unbounded* half into a silent cap at the other half's budget; an unbounded half therefore leaves the tokenizer at its own value. On the on-policy scripts the completion half is a generation budget, which has no unbounded setting — an unset one is rejected here.

Boundedness is one predicate everywhere (`is_bounded_length`): a positive int bounds, while `null` and any non-positive value mean unset.

The pin is **run-scoped and never exported**. `save_pretrained` writes the live `model_max_length` into `tokenizer_config.json`, so an unrestored pin would ship the training budget as the served context — a run at `max_length: 40000` capping a 262k-context model. `setup_model_and_tokenizer` records the tokenizer's own bound and `DistributedTrainerMixin.save_model` runs every writer under `pristine_model_max_length`, which serves that bound for the duration of the save and puts the pin back afterwards.

Two training paths sit outside the two seams. Environmental GRPO pins the raw context window (`get_model_context_window`) — the same hard limit vLLM enforces during rollout, which the trainer fails against rather than truncating. Embedding training pins nothing: its length is installed as SentenceTransformers' `max_seq_length`. The inference scripts (`scripts/inference/`) take a raw CLI number and resolve nothing.

The YAML parser migrates no spelling: a config still naming TRL's retired `max_seq_length` hits the unknown-key raise (see the [Configuration Guide](../getting-started/configuration.md)).

## Performance & balancing flags {#performance-balancing-flags}

Every standard training script inherits these from `CommonScriptArguments`; they wire the observability and balancing callbacks via `build_perf_callbacks` (`src/callbacks/wiring.py`).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `enable_efficiency_metrics` | `bool` | `false` | Construct [EfficiencyCallback](../training-methods/callbacks.md#efficiencycallback). Off by default: multi-sequence trainers (DPO/SMPO/Reward forward chosen + rejected; Distillation also forwards a teacher) report a misleading utilization. Sets `include_num_input_tokens_seen="all"` when on — the HF field is tri-state (`"no" \| "all" \| "non_padding"`), not a bool. |
| `enable_moe_metrics` | `bool` | `true` | Construct [MoEMetricsCallback](../training-methods/callbacks.md#moemetricscallback): `moe/load_{max,min,cv}`, `moe/dead_frac`, `moe/load_max_{first,last}`, emitted every `logging_steps`. No-op for dense models. It turns `output_router_logits` on itself only under `moe_balancing: aux_loss`, so asking for metrics never adds the router-logit tensor or the aux term as a side effect. Not wired under `bias_update` or pipeline parallelism, where `RouterBiasBalancingCallback` emits the same keys from its load counter instead. It reports nothing under **GRPO** (backbone-only log-probs) or where the EP wrapper bypasses HF's router recorder (DeepSeek-V4), warning loudly rather than going silent. |
| `moe_balancing` | `str` | `"auto"` | Router balancing: `auto` (default), `none`, `aux_loss`, `bias_update`, `bias_update_transient`. `auto` resolves per model — `bias_update` where the aux loss cannot reach the loss **and** the bias lands in checkpoint-exported state, `none` + warning where only a transient bias would be possible (Mistral4, Cohere2 MoE, multimodal Qwen3.5/3.6) or where the forward takes no `output_router_logits` and nothing accepts a bias (no balancing route at all — Gemma 4, and the wrapper-signal families launched without EP wrappers), `aux_loss` for other MoE, `none` for dense. An explicit mode **raises** where it would misstate reality (`bias_update` with no bias acceptor or on a family whose bias no export carries, `bias_update_transient` on a family whose bias exports natively, `aux_loss` on a forward that never takes `output_router_logits`), and **warns and stays off** where the term exists but cannot reach the loss (no usable `router_aux_loss_coef`, or an EP wrapper that severs the aux path). Under a GRPO trainer `aux_loss` is inert, and the on-policy weight-sync scripts (online / environmental GRPO) downgrade both bias modes to `none` — those runs have no router balancing at all. Full resolution rules, per-family support and the mode table: [MoE balancing modes](../training-methods/callbacks.md#moe-balancing-modes). |
| `router_balancing_rate` | `float` | `1.0e-3` | Sign-step magnitude (γ) for `RouterBiasBalancingCallback` when a bias-update mode is active. |
| `num_full_model_params` | `float \| null` | `null` | Total param count across all EP/TP ranks. When set, `EfficiencyCallback` computes `distributed_efficiency = params_ratio * mfu`. |
| `enable_torch_profiler` | `bool` | `false` | Construct [TorchProfilerCallback](debugging.md#1a-torchprofiler--gpuoperator-trace--flame-graph) — captures a step window, writes per-rank Chrome trace + flame-graph stacks + memory timeline. |
| `profiler_output_dir` | `str` | `$HALO_DATA_ROOT/profiling/torch` | Output dir for torch.profiler artifacts (derives from `HALO_DATA_ROOT`; set explicitly to override). |
| `profiler_wait` / `profiler_warmup` / `profiler_active` | `int` | `5` / `1` / `3` | torch.profiler schedule: skip `wait`, warm up `warmup`, record `active` steps (one-shot). |
| `profiler_ranks` | `str` | `"0"` | Which global ranks profile: `"0"`, `"all"`, or a comma list like `"0,8"`. |
| `profiler_record_memory_snapshot` | `bool` | `false` | Also dump a CUDA memory snapshot (`.pickle`) over the active window for [memory_viz](https://pytorch.org/memory_viz). |

---

## Optimizer selection

`optim` is HuggingFace's field, extended at import time with two toolkit values (`src/optimizers/registry.py` registers everything in `NAMED_OPTIMIZER_BUILDERS`) so they pass `TrainingArguments` validation.

| `optim` | Optimizer | Notes |
|---|---|---|
| `adamw_torch_fused` (HF default) / `adamw_torch` | AdamW, or **AdamWBF16** when auto-enabled | With `bf16: true` this resolves to [AdamWBF16](../optimization/bf16-optimizer.md) (6 B/param, stochastic rounding) everywhere except an accelerate-managed DDP launch, which is outside the validated SR matrix — set `bf16_optimizer: true` to opt in there. Mixed dtypes are handled internally: bf16 params take the SR kernel, fp32 params (`fp32_router`, `fp32_experts`, `fp32_non_ep_params`) take standard in-place AdamW. |
| `muon` | [Muon](../optimization/muon-optimizer.md) | Newton-Schulz orthogonalization on 2-D params; embeddings and the LM head are excluded. |
| `flash_adamw` | [FlashAdamW](../optimization/flash-adamw.md) | Quantized optimizer states, ~5 B/param. |

`bf16_optimizer: true` together with `optim: muon` or `optim: flash_adamw` **raises** — both select an optimizer and the bf16 path would silently win (`DistributedTrainerMixin._configure_mixed_precision`). Any other explicit `optim` (e.g. `adamw_bnb_8bit`) suppresses the AdamWBF16 auto-enable.

---

## DistributedArguments (model & from-scratch) {#distributedarguments-model-from-scratch}

Inherited by the EP/CP/TP scripts from `src/args/distributed_args.py` (parallelism sizes are under [ParallelismConfig](#parallelismconfig)).

| Parameter | Type | Default | Description |
|---|---|---|---|
| `init_from_scratch` | `bool` | `false` | Build the model with **random weights** from `model_name_or_path`'s config instead of loading pretrained weights. SFT script only (other scripts reject it); dense DP / FSDP2 only — raises for EP/TP/CP/ETP/PP. See [Pre-training](../training-methods/pretraining.md). |
| `reset_sinks` | `bool` | `true` | GptOss only: neutralize the attention sinks for SFT — `dtype.min` and frozen (removed under FA2). `false` keeps the pretrained sinks live and **frozen**, the on-policy RL setting (the trainer scores with the sinks the rollout engine serves). Live sinks need a sink-carrying attention implementation — FA4 on Blackwell, `flex_attention` or `eager` on Hopper; FA2 and SDPA are rejected — and are refused under Context Parallelism. With live sinks a `beta != 0` non-PEFT run is also rejected, in every parallelism mode: TRL's implicit reference model is not sink-restricted, so the KL would be biased on every token. Auto-skipped under `init_from_scratch`. See [GPT-OSS](../models/gpt-oss.md#attention-sinks). |
| `train_sinks` | `bool` | `false` | GptOss only: keep the live sinks **trainable** (requires `reset_sinks: false`; the contradiction is rejected). Full fine-tuning under FA4 (the loader installs an exact sink-less + `sigmoid(lse - sink)` rescale on both kernel entry points, since the fused backward emits no sink gradient) or `eager`; `flex_attention` and FA3 are rejected (no usable sink gradient), as are adapter runs and weight-sync RL. Counts as live for every sink gate. |
| `text_only_model` | `bool` | `false` | Load a multimodal checkpoint through its text-only CausalLM class instead of the vision-bearing wrapper (Qwen3.5/3.6: `Qwen3_5MoeForCausalLM`). Drops the vision tower and any MTP tail from the build **and the export** (a text-only artifact: no `processor_config.json`, no vision token ids), and restores the config-honored `output_router_logits` path, so `moe_balancing: aux_loss` works where the wrapper's forward never consults the flag. Image-bearing datasets are refused loudly. Honored by the `load_model_for_training` scripts (SFT, DPO/SMPO/KTO, distillation) and the GRPO family; scripts that pin their own model class (reward/classification/embedding) warn and ignore it. Serving the text-only export on the pinned vLLM/SGLang images needs `scripts/after_training/reattach_vision_tower.py` (their registries carry only the multimodal class). |
| `save_sharded_ep` | `bool` | `false` | Save EP checkpoints as per-rank shards (every rank writes its own slice in parallel) instead of gathering to one rank — write bandwidth, not host memory, since the gathered path streams. The result loads nowhere until `merge_ep_shards.py` runs, and the merge carries no optimizer state, so pair it with `save_only_model`. The default gathered path writes an HF-standard sharded checkpoint (`model-XXXXX-of-YYYYY.safetensors` + index, `max_shard_size` 5 GB); resume is sharded-aware. Rejected under pipeline parallelism: the shard format keys tensors by unsplit-model names with no stage layer offset, so shards from different stages would collide. |
| `save_max_shard_size` | `str` | `"5GB"` | Max size of one safetensors shard written by the distributed gathered (EP/TP/FSDP2/CP) and PP stage save paths. HF's own `save_pretrained` keeps its own default. A sharded EP save writes one file per rank by design and reports that it does not apply. |
| `overwrite_output_dir` | `bool` | `false` | Allow starting a run in a non-empty `output_dir`. Off by default so a fresh run cannot interleave its checkpoints with an existing one's. See [Checkpoints](checkpoints.md#checkpoint-detection). |
| `merge_expert_lora_on_save` | `bool` | `false` | Fold the adapter delta into the base on save, so the checkpoint is a fully-merged, servable HF model. Covers both halves of a mixed run: the expert deltas inside each family's gather, and any attention adapters via `merge_adapter` held open over the write (undone afterwards, so training continues). Requires native grouped expert adapters — rejected when the run built none, under accelerate-managed FSDP, and with `save_sharded_ep: true` (merging is only wired through the gathered EP save). `false` writes a standalone adapter file and leaves the base unchanged. |

---

## Environment variables

Read from the environment (not the YAML); set in the launch command / `.env`. Toolkit-owned knobs only — third-party credentials (`WANDB_API_KEY`, `HF_TOKEN`, `OPENAI_API_KEY`, the web-search backend keys) are listed in [Dev Environment](../contributing/development-environment.md).

| Variable | Default | Description |
|---|---|---|
| `DIST_SHARED_FILESYSTEM` | `1` | Umbrella for both sides: `1` = shared FS (only global rank 0 writes downloads/checkpoints); `0` = per-node local disk (each node's local rank 0 writes). Must be identical on every rank — the flags select the coordination scope, so `init_distributed()` broadcasts rank 0's *resolved* values for all three and warns any rank whose env disagreed. See [Filesystem Handling](../data/filesystem-handling.md). |
| `DIST_INPUT_SHARED_FILESYSTEM` | umbrella | Read side only (model/dataset downloads, dataset map/pack, HF caches): picks the `fs_aware_main_first` coordination scope and `fs_aware_load_rank()`. Falls back to the umbrella while unset, and wins over it once set. Set `0` with a shared umbrella when a multi-node NFS/EFS mount returns stale file handles on cross-node cache reads. |
| `DIST_OUTPUT_SHARED_FILESYSTEM` | umbrella | Write side only (checkpoints, `run.log`, dumped artifacts): picks `fs_aware_save_rank()`. Falls back to the umbrella while unset, and wins over it once set. Non-shared also forces `save_on_each_node` and rejects a multi-node sharded-EP save (per-rank shards scatter across nodes with no gather path). Multi-node runs probe the declaration against the filesystem at startup and raise on a contradiction in either direction. |
| `DIST_NCCL_TIMEOUT_MINUTES` | `30` | NCCL collective-watchdog timeout, applied by `init_distributed` (PyTorch's own default is 10) and pinned as the default for every later `dist.new_group()`, so EP/CP/TP subgroup collectives inherit it. **Does not bound DeepEP V2's dispatch/combine** (its own `ElasticBuffer` barrier over the NCCL Gin backend). Raise for slow cross-node all-to-all or large gathered saves. Must be `>= 1` — a non-positive value warns and falls back to the default. Rank-uniform (`verify_rank_uniform_env`): a node left at a smaller budget aborts its process group first, and the survivors then die on the *next* collective and blame it. |
| `DIST_STORE_TIMEOUT_HOURS` | `4` | Wall-clock bound for the c10d-store coordination waits — `fs_aware_main_first` (main-first downloads, dataset load, packing), `store_reject_across_ranks` (the coordinated map/filter joins, the sharded-load join), and `sequential_load_within_node` (the `max_concurrent_loading` throttle). These wait out ONE rank's work rather than a collective, so they are hours-scale and independent of `DIST_NCCL_TIMEOUT_MINUTES`. Must be `>= 1` (same fallback rule) and rank-uniform — the rank with the smaller budget gives up on a join its peers are still serving. See [Filesystem Handling](../data/filesystem-handling.md). |
| `NVLINK_DOMAIN_SIZE` | `gpus_per_node` | The **locality unit** for node-local EP/CP/TP/ETP grouping — validation checks NVLink-locality against this, not `gpus_per_node`. Set to `72` on GB200/GB300 NVL72 to run NVLink-wide across OS nodes; leave unset on ≤8-GPU nodes. Also settable as `--nvlink_domain_size`. See [NVL72](../parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink). |
| `SGLANG_GROUP_HOST` | loopback (local server) / default-route NIC | The `VLLM_GROUP_HOST` equivalent for `rollout_backend: sglang`. Separate per engine because the two servers can sit on different hosts. |
| `VLLM_GROUP_HOST` | loopback (local server) / default-route NIC | Trainer interface advertised to vLLM workers for the NCCL weight-sync group. Set on multi-homed hosts where the default-route NIC is unreachable from a remote vLLM. Precedence: explicit `group_host` arg → this var → loopback when the server is local → default-route NIC. |
| `HALO_SCRATCH` | `/mnt` | Host-side scratch root consumed by the Makefile and the vLLM compose file / devcontainer (never read by Python): bind-mounted into containers and deriving the in-container `HALO_DATA_ROOT`. |
| `HALO_DATA_ROOT` | `~/.cache/halo` | Toolkit scratch root — the single knob for toolkit-owned caches and outputs. The S3 dataset cache (`<root>/s3_datasets`) and profiler artifacts (`<root>/profiling`) derive from it. Point it at a verified large volume. HuggingFace caches are governed separately by `HF_HOME` / `HF_DATASETS_CACHE`. |
| `HALO_S3_DATASET_CACHE_DIR` | `$HALO_DATA_ROOT/s3_datasets` | Explicit override for the S3 dataset cache. |
| `HALO_S3_DEFAULT_BUCKET` | `my-bucket` (placeholder) | Bucket used when a dataset key is given without one; set it to your own bucket. |
| `HALO_S3_MAX_FOLDER_CONCURRENCY` | `16` | Files transferred in parallel per S3 folder operation, clamped to `>= 1`. Bounded so concurrency × each file's multipart connections stays inside the client's connection pool. |
| `HALO_S3_CACHE_LOCK_TIMEOUT_SECONDS` | `DIST_STORE_TIMEOUT_HOURS × 3600` | How long a rank waits for the peer downloading the same S3 cache entry. It derives from the store budget because that is what bounds the waiting peers — a lower value turns a legitimately slow multi-GB fetch into a rank-local timeout. |
| `HALO_DATASET_NUM_PROC` | `max(1, min(cpu_count // 4, 4))` | Cluster-wide pin for the HF `dataset.map` `num_proc`. HF keys its map cache on `num_proc`, so a heterogeneous cluster where nodes compute different CPU-based values misses the writer rank's cache and re-runs the whole map — pin it. Must be `>= 1` (below 1 fails loud); `1` disables map multiprocessing. A config `dataset_num_proc` overrides it; resolved once at import. |
| `HALO_TP_CONSISTENCY_CHECK` | off | Makes an added `assert_consistent` raise instead of warn. **Manual instrumentation** — the toolkit ships no call sites, so the flag alone produces nothing. See [Debugging](debugging.md). |
| `HALO_EP_PERF_PROFILE` | off | EP dispatch/expert-compute/combine timing via CUDA-event timers; syncs per phase, so diagnostic runs only. Off, the same spans still appear as `ep.*` ranges in torch.profiler traces at no cost. |
| `HALO_EP_SHARED_OVERLAP` | off | `1` runs the shared-expert FFN on a side stream concurrent with the routed dispatch all-to-all (shared-expert families only). See [DeepEP](../infrastructure/deepep.md). |
| `HALO_EP_CAPACITY_DEDUP` | `1` | `0` restores the per-MoE-layer DeepEP buffer-capacity all-reduce instead of reusing the first layer's capacity for the whole forward. One capacity per forward is also what lets every MoE layer share one `ElasticBuffer`, so `0` additionally gives each layer its own arena — multiplying it by the MoE layer count ([DeepEP](../infrastructure/deepep.md)). |
| `HALO_GRAD_BUCKET_MB` / `HALO_GRAD_BUCKET_MAX_INFLIGHT` | `256` / `2` | Flat-buffer size and concurrency of the bucketed gradient reduction — the deferred cross-replica EP sweep, the TP replicated / per-head-norm sweep, and the QLoRA sweep. (The per-parameter EP grad hooks reduce one tensor at a time and ignore both.) `HALO_GRAD_BUCKET_MB` sets the chunk boundaries every rank must agree on, so it belongs to the rank-uniform set verified in distributed setup ([DeepEP](../infrastructure/deepep.md)); `MAX_INFLIGHT` is rank-local and sets how many bucket collectives run at once, so each one's latency covers the next bucket's flatten. Peak transient is that many flat buffers, and under `fp32_grad_reduce` each also holds an fp32 copy — ~3× the bucket size from bf16. |
| `HALO_DEEPEP_GPU_TIMEOUT_SECONDS` | `100` | Device-side spin budget for the DeepEP dispatch/combine NVLink barrier (DeepEP's own default). The clock starts when a rank **enters** the barrier, so it bounds rank **skew**, not idle time between steps — a rank traps if a peer is still this far behind when it arrives. Save paths fence with a host-side barrier first, which absorbs skew under `DIST_NCCL_TIMEOUT_MINUTES` instead. Raising it also delays how fast a real hang surfaces. Elastic backend only — the `legacy` V1 buffer takes no timeout and warns when the parsed value is not the default (an explicit `100`, or an empty pass-through, is not a request). See [DeepEP](../infrastructure/deepep.md). |
| `HALO_DEEPEP_NUM_SMS` / `HALO_DEEPEP_NUM_QPS` | auto | Pin the DeepEP dispatch/combine SM count and RDMA queue-pair count. `NUM_SMS` applies to both buffer backends (legacy needs an even count); `NUM_QPS` is elastic-only and also sizes the buffer's QP allocation. A/B them on EFA proxy Gin; see [DeepEP](../infrastructure/deepep.md). |
| `HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK` | `8192` | Cross-node (Gin) dispatch payload ceiling, tokens/rank. Above the measured ~8k boundary an EFA proxy-GIN dispatch wedges instead of erroring (receive counts never arrive; Xid 109→43), so the dispatcher rejects it at buffer sizing. `0` disables; raise only after validating end-to-end on your fabric. Intra-node dispatch is unaffected. See [DeepEP → AWS EFA](../infrastructure/deepep.md#expert-parallelism-over-aws-efa). |
| `HALO_PP_FUSED_HEAD_LOSS` | `1` | Fold `lm_head` + cross-entropy into the last pipeline stage's forward in token chunks, so the `[microbatch, seq, vocab]` logits plane never exists. Applies only to the causal-LM pipeline contract; `0` restores the logits path. Reads only under pipeline parallelism, which is [not yet available in this release](../parallelism/pipeline-parallelism.md). |
| `EP_DISABLE_GIN` / `EP_SUPPRESS_NCCL_CHECK` | dispatcher-set / `1` | DeepEP-owned. The dispatcher sets `EP_DISABLE_GIN` before buffer creation (`0` inter-node, `1` intra-node) and honors an explicit value. `EP_SUPPRESS_NCCL_CHECK` is latched at `import deep_ep`, so it must be in the process environment (both images bake it); the dispatcher can only warn when it is unset. Documented in [DeepEP](../infrastructure/deepep.md). |
| `HALO_DEEPGEMM_NATIVE` / `HALO_DEEPGEMM_MIN_TOKENS_PER_EXPERT` / `HALO_DEEPGEMM_MIN_N` | off / `1024` / `4096` | Opt into the native DeepGEMM fp8/fp4 grouped kernel above a shape floor (tokens per expert, output width). The `deep_gemm` wheel is built into the Blackwell image only (its kernels are SM100 block-scaled), so on Hopper the flag warns once and stays inert — and where it does run it measured net-slower than bf16 at the toolkit's MoE shapes ([Low-Precision MoE](../optimization/low-precision-moe-kernels.md)). |
| `HALO_LOWP_COMPILE` / `HALO_LOWP_WEIGHT_CACHE` | `1` / `1` | `0` forces the low-precision quantize/dequantize round-trip to eager (bit-identical, slower) and disables the per-step expert-weight quantization cache. |
| `HALO_ALLOW_MOCK_SEARCH` | off | Makes the fabricated-results `mock` search backend selectable. Off, naming it raises: mock snippets pay `tool_success_reward` exactly like a real search, so a training run that reached them would reward invented evidence. For UI and test runs. |
| `HALO_SANDBOX_BACKEND` / `HALO_SANDBOX_URL` / `HALO_SANDBOX_MAX_CONCURRENCY` | `local` / unset / `cpu_count()` | RL code-execution sandbox: backend (`local` / `bubblewrap` / `remote`), the SandboxFusion endpoint required by `remote`, and the process-wide concurrent-execution cap. See [Sandboxes](../training-methods/grpo/environments/sandbox.md). |
| `HALO_NCCL_SYNC_TIMEOUT_SECONDS` | 120 s (warm-up) / 600 s (per broadcast) | Deadline for the vLLM weight-sync stream drain; on expiry the communicator aborts with the no-IB recovery hint. Raise for very large policies over TCP. |
| `HALO_VLLM_GENERATION_TIMEOUT_SECONDS` | 80% of the NCCL watchdog (`1440` at the default `DIST_NCCL_TIMEOUT_MINUTES`) | Per-request HTTP timeout (seconds) for a vLLM generation call from the NCCL weight-sync client — **online GRPO only**. The call blocks every peer at the next collective, so the default tracks `DIST_NCCL_TIMEOUT_MINUTES` and stays under it. Env-GRPO generates from the Ray actors instead, bounded by `request_timeout` and `episode_timeout`. |
| `HALO_WEIGHT_SYNC_MEM_LOG` | off | `1` brackets each RL weight sync with a per-rank CUDA-memory line. The post−pre delta isolates the gathers from the training step, and the per-rank spread exposes the forwarding rank's extra full-tensor cost — `reserved` far above `peak_alloc` there means stranded allocator pools ([Rollout Servers](../infrastructure/rollout-servers.md)). Off by default: it is one INFO line per rank per sync. |
| `HALO_FP32_MATMUL_PRECISION` | `highest` | fp32 matmul precision applied at model load (`highest`/`high`/`medium`). The default overrides the NGC image's TF32 preference, whose 10-bit mantissa collapses adjacent RoPE token positions past 2048 and corrupts long context on every model. bf16 matmuls are unaffected. `torch.backends.cudnn.allow_tf32` follows the same knob, so fp32 convolutions (Zaya's Conv1d) match the matmuls. Set `high` to opt back into TF32 for an fp32-heavy job. |
| `HALO_ALLOW_MISSING_CHECKPOINT_KEYS` | off | `1` downgrades the model-load coverage gate to a warning. The gate raises when a checkpoint does not carry a tensor the live model needs — see [Checkpoints](checkpoints.md#load-coverage-gate). A task head added on top of a base checkpoint, a tied `lm_head` and the class's own `_keys_to_ignore_on_load_missing` are excused without the flag. |
| `HALO_TORCH_NUM_THREADS` | `1` | CPU threads per rank, applied with `torch.set_num_threads` (one process per GPU, so 1 avoids contention). PyTorch's own knob is `OMP_NUM_THREADS`. |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` (baked into both images) | Driver-owned, latched at `deep_ep`'s `cuInit`, so a Python write is too late. `1` serializes work submission and is worth +9.7% on ep8; the trainer reads it only to warn when it is not `1`. |
| `VLLM_DISABLE_PYNCCL` / `VLLM_NCCL_SO_PATH` | off / `libnccl.so.2` | Vendored from vLLM. `VLLM_NCCL_SO_PATH` is honored by the **trainer-side** weight-sync client; `VLLM_DISABLE_PYNCCL` set in the trainer process is **refused** (RuntimeError) — it belongs to the vLLM server only, and honoring it would silently disable the weight-sync broadcast. It is parsed the way vLLM parses it — `1`/`true` only, not the toolkit's `yes`/`on` — so the trainer's verdict matches the server's. |
| `VLLM_USE_V2_MODEL_RUNNER` | unset | Read by the **vLLM server**, not the toolkit: must be `0` for any run setting `rollout_max_thinking_tokens` — Model Runner V2 answers `thinking_token_budget` with a 400 on every request. Set it in the server's environment (compose passes it through). |
| `VLLM_API_KEY` | `EMPTY` | OpenAI-compatible API key the `scripts/inference/` and `scripts/environments/` CLIs send to a self-hosted rollout server, matching the server's own `--api-key` convention. Falls back to `OPENAI_API_KEY`, then to the `EMPTY` placeholder a keyless local server accepts; each CLI's key flag overrides both. |
| `WANDB_PROJECT` | `project_name` arg | Overwritten unconditionally by the trainer setup before wandb initializes (an exported value does not survive). The run NAME does not travel through `WANDB_NAME`: transformers passes `run_name` straight to `wandb.init`, which wins over the environment, so the setup shortens `run_name` to its last path segment (else `<script>-<output_dir basename>`) and writes it back onto the training config instead. |
| `WANDB_RUN_ID` | `md5(output_dir:launch_time)[:8]` | Resume identity. An exported value is honored on global rank 0 and broadcast to every rank, so export it on the whole job (or nowhere) and reuse it to continue the same wandb run across restarts. |
| `WANDB_RESUME` | unset | Consumed by the wandb SDK, not by this toolkit: set `allow` alongside a fixed `WANDB_RUN_ID` to append to that run instead of starting a new one. |
| `TOKENIZERS_PARALLELISM` | `false` | Set with `setdefault` at package import (`src/__init__.py`), before tokenizers loads: the Rust tokenizer's thread pool deadlocks against the dataset-map worker processes that fork around it. An exported value wins, so a single-process tool can opt back into parallel encoding. |
| `CLEARML_PROJECT` / `CLEARML_TASK` | `project_name` arg / `run_name` basename | Set by the same trainer setup, same unconditional overwrite, and read by the transformers ClearML integration when `report_to` includes `clearml` — which is why the task name IS an env var here and is not one for wandb. |

---

## Trainer-to-config quick reference

Each trainer combines a **trainer config** (hyperparameters) with **script arguments** (dataset, tokenizer, infra). All configs also accept standard `TrainingArguments` fields. Every script additionally parses TRL's `ModelConfig` — `model_name_or_path`, `model_revision`, `dtype`, `attn_implementation`, and the LoRA fields ([PEFT](../optimization/peft.md)) — and [DistributedArguments](#distributedarguments-model-from-scratch); all four dataclasses draw from the one YAML.

| Trainer | Config Class | Script Arguments | Source |
|---------|-------------|-----------------|--------|
| `DistributedSFTTrainer` | [SFTConfig](#trl-trainer-configs) (TRL) | [SFTScriptArguments](#sftscriptarguments) | `sft.py` |
| `SmoothMarginPOTrainer` | [SmoothMarginPOConfig](#smoothmarginpoconfig) | [SMPOScriptArguments](#other-script-arguments) | `preference/smpo.py` |
| `DistributedDPOTrainer` | [DPOConfig](#trl-trainer-configs) (TRL) | [DPOScriptArguments](#other-script-arguments) | `preference/dpo.py` |
| `DistributedKTOTrainer` | [KTOConfig](#trl-trainer-configs) (TRL) | [KTOScriptArguments](#other-script-arguments) | `preference/kto.py` |
| `OfflineGRPOTrainer` | [OfflineGRPOConfig](#offlinegrpoconfig) | [OfflineGRPOScriptArguments](#other-script-arguments) | `offline_grpo.py` |
| `DistributedGRPOTrainer` (RLVR) | [GRPOConfig](#trl-trainer-configs) (TRL) | [RLVROnlineGRPOScriptArguments](#rlvronlinegrposcriptarguments) | `online_grpo/rlvr.py` |
| `DistributedSDPGTrainer` (online SDPG) | [GRPOConfig](#trl-trainer-configs) (TRL) | [RLVROnlineGRPOScriptArguments](#rlvronlinegrposcriptarguments) (`--use_sdpg=true`) | `online_grpo/rlvr.py` |
| `DistributedAsyncEnvironmentalGRPOTrainer` | [GRPOConfig](#trl-trainer-configs) (TRL) + [EnvironmentConfig](#environmentconfig) + [AsyncTrainingConfig](#asynctrainingconfig) | [EnvironmentalGRPOScriptArguments](#environmentalgrposcriptarguments) | `environmental_grpo.py` |
| `DistributedRewardTrainer` | [RewardConfig](#trl-trainer-configs) (TRL) | [RMScriptArguments](#other-script-arguments) | `preference/rewards.py` |
| `ClassificationTrainer` | [ClassificationConfig](#classificationconfig) | [CLFScriptArguments](#other-script-arguments) | `classification.py` |
| `DistributedDistillationTrainer` | [DistillationConfig](#distillationconfig) | [DistillScriptArguments](#distillscriptarguments) | `distillation/teacher_distill.py` |
| `DistributedSelfDistillationTrainer` (offline SDPG) | [SFTConfig](#trl-trainer-configs) (TRL) | [SelfDistillationArguments](#other-script-arguments) | `distillation/self_distill.py` |
| `EmbeddingTrainer` | [EmbeddingConfig](../training-methods/embedding.md) (extends `SentenceTransformerTrainingArguments`) | [EmbeddingScriptArguments](#other-script-arguments) | `embedding.py` |

All distributed trainers also accept the [ParallelismConfig](#parallelismconfig) parameters, in the YAML or on the CLI.

---

## TRL trainer configs

These configs come from [TRL](https://huggingface.co/docs/trl) and are used directly. Only the commonly tuned fields are listed; all extend `TrainingArguments`.

### SFTConfig — `DistributedSFTTrainer` ([TRL docs](https://huggingface.co/docs/trl/sft_trainer#trl.SFTConfig))

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | `int \| None` | `1024` | Total sequence budget. Over-length conversations are dropped by the chat processor; under `packing` it is the fixed pack size and must be explicit. `null` → model context window. |
| `packing` | `bool` | `False` | Pack multiple sequences into one example. |
| `eval_packing` | `bool \| None` | `None` | Packing for eval. Defaults to `packing`. |
| `padding_free` | `bool` | `False` | Flatten the batch into one varlen sequence instead of padding it. Needs a varlen Flash kernel (FA2/FA3/FA4) and **raises** on any other resolved `attn_implementation` — the `cu_seqlens` it exists to emit go unread there. See [Padding-Free](../optimization/padding-free-collator.md). |
| `dataset_text_field` | `str` | `"text"` | Column with pre-formatted text (skips chat template). |
| `dataset_kwargs` | `dict \| None` | `None` | Kwargs passed to dataset processing. |
| `bf16` | `bool \| None` | `True` | Toolkit default `True` (TRL leaves it `None`); yields to an explicitly-enabled `fp16`. `mixed_precision` is re-derived after toolkit defaults and CLI overrides. |
| `use_liger_kernel` | `bool` | `True` | Liger Triton kernels. Toolkit default `True` (TRL default `False`). |
| `logging_nan_inf_filter` | `bool` | `False` | Toolkit default `False` (upstream `True`). Upstream's filter reads the loss scalar back to the host on **every** micro-batch — `gradient_accumulation_steps` stalls per optimizer step, and at large world size every rank then waits on the slowest — and replaces a NaN/Inf loss with the running average, hiding a diverged run from its own loss curve. |

### DPOConfig — `DistributedDPOTrainer` ([TRL docs](https://huggingface.co/docs/trl/dpo_trainer#trl.DPOConfig))

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `0.1` | KL penalty coefficient. Higher = more conservative updates. |
| `loss_type` | `list[str]` | `["sigmoid"]` | `sigmoid`, `hinge`, `ipo`, `exo_pair`, `nca_pair`, `robust`, `bco_pair`, `sppo_hard`, `aot`, `aot_unpaired`, `discopop`, `apo_zero`, `apo_down`, `sft`, `sigmoid_norm` (a list — losses can be combined). |
| `max_length` | `int \| None` | `1024` | Total budget (prompt + completion). TRL truncates the **concatenated** sequence `keep_start`, so an over-long prompt eats its own completion — `DPOConfig` has no `max_prompt_length` / `max_completion_length`. Filter over-long prompts upstream. `null` → model context window. |
| `label_smoothing` | `float` | `0.0` | Label smoothing for DPO loss. |
| `truncation_mode` | `str` | `"keep_start"` | `keep_start`, or the upstream-deprecated `keep_end`. |
| `disable_dropout` | `bool` | `True` | Disable dropout. |

### KTOConfig — `DistributedKTOTrainer`

Unpaired `{prompt, completion, label}` preference optimization. `max_length` (`int | None`, default `1024`) truncates the assembled sequence **keeping the start**; `loss_type` is `kto` (default) or `apo_zero_unpaired` — the only one the (not-yet-available) pipeline-parallel gate accepts, and only with `precompute_ref_log_probs: true`. `beta` defaults to `0.1`.

### GRPOConfig — `DistributedGRPOTrainer`, `DistributedAsyncEnvironmentalGRPOTrainer` ([TRL docs](https://huggingface.co/docs/trl/grpo_trainer#trl.GRPOConfig))

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_generations` | `int \| None` | `8` | Completions per prompt. |
| `max_completion_length` | `int \| None` | `256` | Generation budget (online / vanilla GRPO; TRL `max_new_tokens`). Set it explicitly. **Env-GRPO does not use it** — it generates with `rollout_max_tokens` and is bounded by the model context (see [Sequence length](#sequence-length-caps-vs-generation-budgets)). |
| `beta` | `float` | `0.0` | KL penalty coefficient (`0.0` = no reference model). |
| `temperature` | `float` | `1.0` | Sampling temperature. |
| `top_k` | `int` | `0` | Top-k sampling (`0` = disabled). Online GRPO only — **env-GRPO ignores it** and warns; its rollouts sample from the rollout server config. |
| `top_p` | `float` | `1.0` | Top-p sampling. Online GRPO only — **env-GRPO ignores it** (use `rollout_top_p`) and warns. |
| `num_iterations` | `int` | `1` | Policy update iterations per batch (mu-GRPO). |
| `epsilon` | `float` | `0.2` | Clipping epsilon. |
| `loss_type` | `str` | `"dapo"` | `grpo`, `dapo` (default), `bnpo`, `dr_grpo`, plus TRL's `cispo`, `sapo`, `vespo`, `luspo` — the toolkit adds no gate on this path. Offline GRPO accepts only `grpo`, `bnpo`, `dr_grpo`. |
| `scale_rewards` | `str` | `"group"` | `"group"`/`True` (per-group std), `"batch"`, `"none"`/`False`. |
| `use_vllm` | `bool` | `False` | Use vLLM for generation. The toolkit's online GRPO trainer **requires `true`** — it raises on `false` (in-process HF generation is slow and desyncs ranks under FSDP2/EP). |
| `vllm_mode` | `str` | `"colocate"` | `server` (external) or `colocate` (managed). The toolkit's online GRPO trainer supports only `server` — set it in the config (it raises on `colocate`). |
| `vllm_server_host` / `vllm_server_port` | `str` / `int` | `"0.0.0.0"` / `8000` | vLLM server address/port. |
| `vllm_group_port` | `int` | `51216` | NCCL group port for weight sync. In multi-server mode it is the **base** port: a `rollout_server_configs` entry without its own `group_port` binds `vllm_group_port + index`. |
| `log_completions` | `bool` | `False` | Print the rich per-sample completions **console** table (TRL). Decoupled from the durable parquet record — see `save_completions`. |
| `save_completions` | `bool` | `True` | Halo `CommonScriptArguments` field, not TRL — the durable parquet record, see [CommonScriptArguments](#commonscriptarguments). |

### RewardConfig — `DistributedRewardTrainer` ([TRL docs](https://huggingface.co/docs/trl/reward_trainer#trl.RewardConfig))

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | `int \| None` | `1024` | Total sequence budget, applied as a **filter**: TRL drops pairs whose chosen *or* rejected exceeds it (it does not truncate, despite TRL's field help). `null` → model context window. |
| `center_rewards_coefficient` | `float \| None` | `None` | Reward-centering loss coefficient. Helps prevent reward hacking. |
| `disable_dropout` | `bool` | `True` | Disable dropout. |

---

## ParallelismConfig

Unified config for Expert (EP), Context (CP), Tensor (TP), Expert-Tensor (ETP), and Pipeline (PP) Parallelism. Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md) — the `pp_*` fields exist, but `parallelism_config_from_args` rejects `pipeline_parallel_size > 1` at config time. The class itself is built at construction; users set it through `DistributedArguments` (`src/args/distributed_args.py`), whose names go in the YAML or on the CLI and differ from the field names below: `ep_size` ← `expert_parallel_size`, `cp_size` ← `context_parallel_size`, `tp_size` ← `tensor_parallel_size`, `expert_tp_size` ← `expert_tensor_parallel_size`, `pp_size` ← `pipeline_parallel_size`, `pp_schedule` ← `pipeline_schedule`, `pp_microbatches` ← `pipeline_microbatches`, `pp_split` ← `pipeline_split`, `ep_fp32_router` ← `fp32_router`, `ep_fp32_experts` ← `fp32_experts`. Every other field below keeps its name, except `world_size` / `gpus_per_node` (auto-detected, not settable), `expert_lora`, and `merge_expert_lora_on_save` (documented under `DistributedArguments` above).

**Source:** `src/distributed/parallelism_config.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `world_size` / `gpus_per_node` | `int` | `0` | `0` = auto-detect from the launcher environment. |
| `ep_size` | `int` | `1` | Expert parallel group size. >1 enables EP for MoE. |
| `cp_size` | `int` | `1` | Context parallel group size. >1 enables sequence splitting. |
| `tp_size` | `int` | `1` | Tensor parallel size. >1 shards model weights. |
| `expert_tp_size` | `int` | `1` | Expert Tensor Parallel size. Shards expert FFN weights within EP groups (MoE-only, experimental). |
| `pp_size` | `int` | `1` | Pipeline stages. Only `1` is accepted in this release — larger values are rejected at config time, and the three PP-only knobs below are rejected at `pp_size == 1`. Outermost axis by design — it carves the world first (`stage_world_size = world_size / pp_size`). See [Pipeline Parallelism](../parallelism/pipeline-parallelism.md). |
| `pp_schedule` | `"1f1b" \| "gpipe"` | `"1f1b"` | `1f1b` (lowest activation memory, needs `pp_microbatches >= pp_size`) or `gpipe` (all forwards then all backwards, no microbatch-count constraint). |
| `pp_microbatches` | `int` | `0` | Microbatches per optimizer step — the pipeline's own accumulation. `0` = auto: `gradient_accumulation_steps`, raised to `pp_size` when `1f1b` requires it. Setting it alongside `gradient_accumulation_steps > 1` raises (the microbatches *are* the accumulation); `per_device_train_batch_size` must divide by it. |
| `pp_split` | `list[int] \| None` | `None` | Per-stage decoder-layer **counts** (one entry per stage, each ≥ 1, summing to the model's layer count, with boundaries on the model's `layer_types` period). `None` = the head-weighted default, which shrinks the last stage's budget by the `lm_head`'s layer-equivalent cost. |
| `ep_scope` | `"auto" \| "node" \| "global"` | `"auto"` | `auto` resolves to node-local if `ep_group_size <= nvlink_domain_size`, else global. `node` keeps groups within each NVLink domain. `global` spans domains (requires RDMA). |
| `nvlink_domain_size` | `int` | `0` | Locality unit for node-local EP/CP/TP/ETP grouping. `0` = auto: reads `NVLINK_DOMAIN_SIZE`, else `gpus_per_node`. Set to the rack size (e.g. `72`) on NVL72/MNNVL. CLI: `--nvlink_domain_size`. |
| `ep_fp32_router` | `bool` | `False` | Run MoE router in FP32. Implied by `fp32_non_ep_params`: that upcast skips the router, and a bf16 router among fp32 dense params trips FSDP2's uniform-dtype check. |
| `ep_fp32_experts` | `bool` | `False` | Store EP expert weights in FP32 for stable optimizer updates; compute stays BF16 under autocast. Skipped with a warning when FSDP2 manages the experts (`ep_group_size==1` at the default `fsdp_shard_ep1_experts`), where it would do nothing. |
| `fp32_non_ep_params` | `bool` | `False` | Store non-expert params in FP32 (compute stays BF16 via autocast). Implies fp32 reduction for the **FSDP2 `reduce_dtype` only** — the EP, TP and QLoRA bucketed reductions read `fp32_grad_reduce` directly, so set that explicitly to cover them. |
| `fp32_grad_reduce` | `bool` | `False` | Reduce gradients across ranks in FP32 while storing BF16 (FSDP2 `reduce_dtype=fp32` for dense params + EP router/expert grad-sync hooks). Keeps BF16 master + AdamWBF16 (6 B/param); ~2x bandwidth on the reduce. See [BF16 Optimizer](../optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options). |
| `use_grouped_gemm` | `bool` | `True` | `torch.nn.functional.grouped_mm` for MoE expert compute on SM90+ (no-op on older HW). MoE layers are wrapped even standalone (`ep_size=1`, no EP), so an MoE under any `accelerate launch` is rejected at load — launch with `torchrun` or set `false`. Inert for dense models (they run under accelerate normally). |
| `fsdp_reshard_after_forward` | `bool` | `False` | FSDP2 resharding: `False` = ZeRO-2 analog (SHARD_GRAD_OP, params stay gathered, faster, higher peak memory); `True` = ZeRO-3 analog (FULL_SHARD, lower peak memory). **Allowed only where `ep_group_size==1`** (pure DP, pure TP with dp=1, CP, `ep_size==1` MoE without expert TP); rejected with `ep_size>1`, Expert-TP, TP+DP (dp>1), or PP — the backward re-gather races the DeepEP combine, or all-gathers TP-sharded DTensors with no sharding strategy. See [ZeRO-2 vs ZeRO-3](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward). |
| `fsdp_reshard_after_backward` | `bool` | `True` | `False` keeps params unsharded across a grad-accum window's microsteps (torch `set_reshard_after_backward`, re-armed for the window's last backward): drops the full-model re-all-gather FSDP2 otherwise issues after every microstep's backward to one per optimizer step — negligible over NVLink, ~15s × `gradient_accumulation_steps` per step when NCCL is forced onto sockets (`rollout_backend: sglang`). Saves nothing at `gradient_accumulation_steps=1`. Costs one unsharded bf16 param copy per GPU. Plain-DP/CP/EP torchrun path only; rejected with `fsdp_reshard_after_forward=True`, TP, or PP. See [Data Parallelism](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward). |
| `fsdp_shard_ep1_experts` | `bool` | `True` | Applies only when experts are truly replicated — `ep_group_size==1` (`ep_size==1` **and** `expert_tp_size==1`): FSDP-shards the replicated MoE experts so their reduce-scatter becomes the sole gradient sync. Default `True` frees memory that grows with DP (gpt-oss-20b −19%/−37% at 2/8 GPU), throughput-neutral and grad-equivalent; `False` keeps a full replicated copy per rank (max throughput). No effect when `ep_group_size>1`. Composes with `fsdp_reshard_after_forward` for full ZeRO-3 experts, and works for online/environmental GRPO (the weight-sync gather `full_tensor`-s the sharded experts first). `False` raises at config time under TP or CP (those paths shard ep1 experts unconditionally, so the flag would be ignored) and under PP whenever the EP wrappers are active (`use_grouped_gemm`, the default — `ParallelismConfig` has no model knowledge, so this catches dense runs too): a stage holding plain replicated experts beside a dense stage issues a different collective program and deadlocks at the gradient norm. |
| `use_hsdp` | `bool` | `False` | Hybrid Sharded DP on the standard DP path (pure DP or CP, no EP). Shards non-expert params within each NVLink domain and replicates across domains, so only one gradient all-reduce crosses RDMA per step. No-op on a single domain. **Rejected with any EP (including EP+CP), TP, Expert-TP, or PP** — EP already shards over the EP group, so HSDP would be a no-op or race the DeepEP combine. See [HSDP](../parallelism/data-parallelism.md#hsdp-hybrid-sharded-data-parallel). |
| `lowp_precision` | `"bf16" \| "fp8" \| "fp4" \| "mxfp4"` | `"bf16"` | Low-precision matmul compute — bf16/fp32 master weights, low-precision GEMM operands; params and checkpoints unchanged. `bf16` (off), `fp8` (mxfp8), `fp4` (nvfp4, most accurate), `mxfp4` (fast fp4). All are the block-scale **fake-quant** QAT oracle and are **slower than bf16** (they quantize operands each step). The native DeepGEMM fp8/fp4 kernel is opt-in (`HALO_DEEPGEMM_NATIVE=1`), net-slower at every training shape, and never auto-selected. Applies to dense MLP + MoE expert GEMMs; attention/embeddings/lm_head/norms stay bf16. **SFT script only** — every other script rejects a non-`bf16` `lowp_precision` and any non-default `lowp_apply_*` / `lowp_keep_*`; PP rejects it too. See [Low-Precision Kernels](../optimization/low-precision-moe-kernels.md). |
| `lowp_apply_dense_mlp` | `bool` | `True` | Apply `lowp_precision` to dense MLP projections (gate/up/down). |
| `lowp_apply_moe_experts` | `bool` | `True` | Apply `lowp_precision` to MoE expert grouped GEMMs. |
| `lowp_keep_first_blocks` / `lowp_keep_last_blocks` | `int` | `0` | Leading/trailing transformer blocks kept in bf16 (NVFP4 recipe keeps the most precision-sensitive ends, ~8 trailing). |
| `bf16_optimizer` | `bool \| None` | `None` | Use `AdamWBF16` (bf16 master weights + stochastic rounding, 6 B/param). `None` = auto (on under `bf16` with the default AdamW, off under replicated DDP); `True` forces it on. `False` forces full fp32 master weights, which builds the stock fused/foreach AdamW — refused **at optimizer build** wherever plain-tensor experts sit beside FSDP2 DTensor params (`ep_group_size > 1`, or EP-wrapped experts at `ep_size==1` with `fsdp_shard_ep1_experts: false`), since `aten._fused_adamw_` cannot mix the two. An `ep_size==1` MoE at the default `fsdp_shard_ep1_experts` is uniformly DTensor and steps fine; `fp32_non_ep_params` gives fp32 masters on the non-expert params only. See [BF16 Optimizer](../optimization/bf16-optimizer.md). |
| `fp32_output_conversion` | `bool` | `False` | Keep accelerate's fp32 output conversion wrapper. Off by default: the upcast of the `[B, S, V]` logits plane OOMs on long sequences, and a bf16 model needs no fp32 outputs. Ignored under fp16, where `native_amp` also owns GradScaler unscaling (warned). |
| `ep_lazy_loading` | `bool` | `True` | Lazy safetensors loading for EP, EP+CP, EP+TP, and pure ETP — each rank reads only its expert slice (parallel, low CPU RAM). Covers every MoE family except Cohere2 MoE, across fused and per-expert safetensors layouts — the gate is the layer class's `_supports_lazy_loading` ([per-family EP restrictions](../parallelism/expert-parallelism.md#per-family-ep-restrictions)), not a key list; a checkpoint whose expert layout the loader cannot address, or a non-local one, falls back to sequential CPU-staged loading. TP-only MoE ignores it and always uses `from_pretrained` + patching. |
| `ep_buffer_backend` | `"auto" \| "elastic" \| "legacy"` | `"auto"` | DeepEP all-to-all transport. `auto`/`elastic`: V2 ElasticBuffer (NCCL Gin, cross-node capable, streams arbitrary sequence length — the only guarded limit is the 32-bit wire index at ~175k tokens/rank). `legacy`: V1 CUDA-IPC Buffer, numerically identical but intranode-only and rejected for any cross-node EP group. See [Transport backend](../infrastructure/deepep.md#transport-backend). |
| `max_concurrent_loading` | `int \| None` | `null` | Max ranks loading simultaneously per node. Unset resolves node-width-aware to `min(4, max(1, local_world_size // 2))` — 4 on an 8-GPU node, 2 on a 4-GPU tray. **Every** explicit value is honored verbatim, `4` included: `0` = all-parallel, a value at or above the node width disarms the throttle, `1` is fully sequential for CPU-RAM-constrained machines. Governs the CPU-staged loaders (DDP, CP, TP-MoE, grouped-GEMM, non-lazy EP fallback); the lazy-EP path ignores it, and so does HF-native dense TP, which streams each rank's shard straight to its GPU and ends in a tie-equality all-reduce on the default process group that a rank-serialized region would deadlock. |
| `expert_lora` | `ExpertLoraSpec \| None` | `None` | Native grouped LoRA on EP experts. No CLI flag — it is built from the PEFT `LoraConfig` and passed at construction so it cannot bypass validation. Carries `r` / `alpha` / `dropout` / `projections` / `use_rslora`; knobs it cannot honor (`use_dora`, `lora_target_parameters`) are rejected rather than applied to the attention half alone. See [PEFT](../optimization/peft.md#moe-models-expert-targets-and-full-trained-modules). |

Besides these, `num_nodes`, `num_nvlink_domains`, `ep_group_size`, `data_parallel_size`, `stage_world_size`, `pp_rank`, and the stage-local rank coordinates are computed in `__post_init__`. Two more carry no `DistributedArguments` spelling and are derived from the training config by `parallelism_config_from_args` (`src/training/parallelism_args.py`): `ep_rows_per_device` (the rows one MoE forward carries per device — the trainer's rows-per-example × `per_device_train_batch_size`) and `ep_declared_max_length`, which the config-time dispatch-ceiling gate multiplies out into a per-rank token budget before any weight is read. A trainer whose config declares no `max_length` field stamps neither and leaves the ceiling to the dispatcher's runtime backstop; `max_length: null` still stamps the rows, and the gate resolves the length against the model's context window.

Four knobs are implemented inside the mixin's own FSDP2 wrap and are therefore **ignored under `accelerate launch`** (accelerate owns the wrap): `use_hsdp`, `fsdp_reshard_after_forward`, `fsdp_reshard_after_backward`, `fp32_grad_reduce`. Setting any of them on an accelerate launch logs one warning — use `torchrun`.

### Data parallel size

EP is always orthogonal to data parallelism. PP carves the world first (`stage_world_size = world_size / pp_size` — every rank of one pipeline chain consumes the same batch), then TP, CP, and ETP reduce what is left:

```text
data_parallel_size = (world_size / pp_size) / max(tp_size, cp_size, expert_tp_size)
```

So standard DDP and EP-only give `world_size`; TP/CP/ETP and EP+TP/EP+CP divide by the respective size; PP and PP+EP divide by `pp_size`. Note the divisor is a **max**, not a product.

> **Pre-sharded datasets:** a per-DP-rank pre-split dataset must have `num_shards >= data_parallel_size`, else a DP rank holds zero examples and training raises `ValueError`. Re-preprocess with a larger `--num-shards`, or use a non-sharded dataset.

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
    --expert_parallel_size=8 --context_parallel_size=8 --ep_scope=node
```

---

## OfflineGRPOConfig

`OfflineGRPOTrainer` — group-relative policy optimization on pre-computed rewards. **Source:** `src/configs/offline_grpo_config.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_prompt_length` | `int \| None` | `512` | Prompt budget; a longer prompt is **left**-truncated (the tokens nearest the completion are kept). `null` = no cap, and is rejected under pipeline parallelism (which needs a fixed shape). |
| `max_length` | `int \| None` | `None` | **PP only** — and PP is [not yet available](../parallelism/pipeline-parallelism.md), so setting it is always **rejected at trainer construction** today; cap the run with `max_prompt_length` / `max_completion_length` instead. (Under a pipeline it would be the fixed total length every batch pads to; `None` = the two shares' sum.) |
| `max_completion_length` | `int \| None` | `None` | Completion budget; a longer stored completion is truncated from the end. `null` (default) = no cap. Also the `loss_type='dr_grpo'` loss normalizer and required under pipeline parallelism — both reject `null` rather than normalize by a default nobody chose. |
| `kl_beta` | `float` | `0.0` | KL divergence loss coefficient. |
| `disable_dropout` | `bool` | `True` | Disable dropouts. |
| `padding_value` | `int \| None` | `None` | Override padding value (defaults to tokenizer `pad_token_id`). |
| `model_init_kwargs` | `dict \| None` | `None` | On every entry-script path these are **model-config overrides**, written onto the loaded config's fields before the load: a key that config does not declare **raises**, and so does `dtype`/`torch_dtype` (the run's precision comes from `bf16` / `fp32_*`). They are model-loading kwargs only where a trainer is constructed programmatically with the model as a path string; beside an already-built model, a non-`null` value raises. |
| `dataset_num_proc` | `int \| None` | `None` | Dataset preprocessing processes. |
| `advantage_method` | `str` | `"quantile_norm"` | `z_norm`, `minmax`, `quantile_norm`, `quantile_uniform`, `robust` (median/IQR, outlier-resistant). |
| `best_completion_emphasis` | `float \| str` | `0.0` | Boost for the best completion(s) per group: `0.0` = off, a float **> 1.0** = fixed boost (`1.5` = +50%), `"auto"` = std-adaptive. A value in `(0.0, 1.0]` **raises** — the consumer applies the factor only above 1.0, so it would be a silent no-op. |
| `min_log_prob` | `float \| None` | `-3.0` | Min log prob for negative-advantage tokens (numerical stability). |
| `initial_min_log_prob` | `float \| None` | `None` | If set, linearly schedules `min_log_prob` from here to the final value. |
| `loss_type` | `str` | `"bnpo"` | `grpo` (per-sequence avg then weighted mean across groups), `bnpo` (global weighted token avg), `dr_grpo` (normalized by effective batch size + max completion length). |
| `policy_gradient_formulation` | `str` | `"prob_weighted"` | `prob_weighted` — `L = -(π(a\|s) * advantage)`, probability-weighted REINFORCE, more conservative offline. `reinforce` — `L = -(log π(a\|s) * advantage)`, equal weight, more aggressive but less stable offline. |
| `drop_degenerate_groups` | `bool` | `False` | Drop groups whose completions all scored **exactly** the same reward (or with < 2 completions) at tokenization — their advantages are all 0, so they spend forward compute only to dilute the loss normalizer. Near-ties are kept (the rank methods train them at full scale). Raises if the whole dataset would be dropped. |
| `use_chunked_grpo_logprobs` | `bool` | `False` | Vocab-chunked completion log-probs instead of full `[B, T, vocab]` logits — the shared `ChunkedLogprobsArguments` switch; covers the policy and the `kl_beta > 0` reference forward. Inert under PP (the last stage still materializes its own logits plane) — the trainer warns at construction rather than leaving it to a surprise OOM. See [Offline GRPO](../training-methods/grpo/offline-grpo.md#memory-chunked-log-probs). |

---

## SmoothMarginPOConfig

`SmoothMarginPOTrainer` (SMPO) — reference-model-free, margin-based preference optimization. **Source:** `src/configs/smpo_config.py`

The class also overrides four `TrainingArguments` defaults: `learning_rate=1e-6`, `logging_steps=10`, `gradient_checkpointing=True`, and `bf16` (declared `bool | None`, resolved in `__post_init__` to `not fp16`).

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `beta` | `float` | `1.2` | Temperature controlling loss sensitivity. Higher = stronger preference signal. |
| `target_margin` | `float` | `0.35` | Target margin for the chosen/rejected log-prob ratio. |
| `loss_type` | `str` | `"smooth_lower_bound"` | `sigmoid` (margin violation), `hinge` (hard margin), `ipo` (squared), `smooth_lower_bound` (squared hinge, smooth near boundary; recommended). |
| `chosen_sft_ratio` | `float` | `0.8` | Weight of chosen in the SFT loss component. `1.0` = chosen only, `0.0` = rejected only. |
| `use_margin_schedule` | `bool` | `True` | Linearly raise `target_margin` from `initial_margin`. |
| `initial_margin` | `float` | `0.01` | Starting margin when scheduling. |
| `lower_clip_percentile` | `float \| None` | `0.02` | Clip low token log probs in rejected (recommended 0.01–0.05). |
| `upper_clip_percentile` | `float \| None` | `None` | Clip high token log probs in chosen. |
| `min_log_prob` | `float \| None` | `-2.3` | Absolute min log-prob threshold for rejected tokens. |
| `padding_free` | `bool` | `False` | Flatten the batch into one varlen sequence instead of padding it. Needs a varlen Flash kernel (FA2/FA3/FA4) and raises on any other — including the `sdpa` the script defaults to under `reset_sinks: true`, which is what makes it need an explicit `attn_implementation` ([SMPO](../training-methods/preference/smpo.md)). **Incompatible with CP, VLM mode, and PP.** |
| `max_length` | `int \| None` | `1024` | Total budget (prompt + completion), split into the two shares below by `resolve_length_budget()`. `null` → model context window. |
| `max_prompt_length` | `int \| None` | `None` | Prompt share; longer prompts truncate per `truncation_mode`. `null` (default) = **half of `max_length`**, so the split scales with a context-resolved budget. VLM prompts are never truncated — one that expands past this raises at collation. |
| `max_completion_length` | `int \| None` | `None` | Completion share; longer completions truncate from the end, keeping the terminal EOS. `null` (default) = the `max_length − max_prompt_length` remainder. An explicit value must still fit that remainder (else the config raises). |
| `truncation_mode` | `str` | `"keep_end"` | `keep_end` or `keep_start`. |
| `label_pad_token_id` | `int` | `-100` | Mask token for prompt tokens in labels. |
| `disable_dropout` | `bool` | `True` | Disable dropout. |
| `dataset_num_proc` | `int \| None` | `None` | Dataset preprocessing processes. |
| `model_init_kwargs` | `dict \| None` | `None` | Model-config overrides applied before the load — see [OfflineGRPOConfig](#offlinegrpoconfig). |

---

## DistillationConfig

`DistributedDistillationTrainer`. **Source:** `src/configs/distillation_config.py`. The teacher and the conversation-rendering fields live in `DistillScriptArguments` below.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `distill_loss` | `str` | `"kl_divergence"` | `kl_divergence`, `mse`, `soft_cross_entropy`, `cosine_similarity`, `jensen_shannon`, `earth_mover_distance`, `alpha_beta_divergence`, `slim`. |
| `distill_temperature` | `float` | `1.0` | Softmax temperature — forwarded only to the losses that declare it (`kl_divergence`, `soft_cross_entropy`, `jensen_shannon`, `slim`). Ignored by `mse`, `cosine_similarity`, `earth_mover_distance`, `alpha_beta_divergence`. Must be `> 0`. |
| `distill_alpha` | `float` | `1.0` | Weight of distillation vs CLM loss. `1.0` = distillation only. Must be in `[0, 1]`. |
| `apply_hard_labels` | `bool` | `False` | Apply hard-labels coefficient to distillation loss. |
| `use_clm_loss` | `bool` | `True` | Add the auxiliary CLM (SFT) loss alongside distillation. `False` = distillation only. |
| `max_length` | `int \| None` | `2048` | Maximum tokenized sequence length. Over-length conversations are **dropped**, not truncated; `null` → the student's context window. |
| `dataset_num_proc` | `int \| None` | `None` | Dataset preprocessing processes. |

### DistillScriptArguments

**Source:** `src/args/distill_args.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `teacher_model` | `str \| None` | `None` | Teacher model name or path. Required — an unset or blank value raises at parse time, not at model load. |
| `teacher_model_revision` | `str \| None` | `None` | Hub revision for the teacher. It is usually a different repo from the student, so it cannot share `model_revision` (one repo's commit sent to another 404s). `None` = the teacher repo's `main`. |
| `conversation_field` | `str \| None` | `"messages"` | Dataset field with conversations. |
| `images_field` | `str \| None` | `None` | VLM only: image column injected into the first user turn (same semantics as the [SFT field](#sftscriptarguments)). |
| `system_prompt` | `str \| None` | `None` | System prompt if none in dialogue. |
| `model_supports_system_role` | `bool` | `True` | If `False`, system prompt is prepended to the first user message. |
| `train_on_completions_only` | `bool` | `True` | Mask prompt tokens, train only on assistant completions (text and VLM paths alike). |
| `assistant_message_template` | `str \| None` | `None` | The rendered assistant-turn prefix of the model's chat template; required when `train_on_completions_only` is on (no default fits every template — a missing or mismatched marker raises at startup). |
| `interleaved_thinking` | `bool` | `False` | Pass `clear_thinking=False` to `apply_chat_template` (GLM-family templates). Text-only: the VLM path raises (no supported VLM template has the switch). |

---

## ClassificationConfig

`ClassificationTrainer`. **Source:** `src/configs/classification_config.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_length` | `int \| None` | `1024` | Truncation length at tokenization (`scripts/training/classification.py`). A raw `text`/`label` dataset passed straight to the trainer is instead tokenized untruncated and filtered to it. Batches pad to the batch longest, not to this. `null` → model context window. |
| `disable_dropout` | `bool` | `True` | Disable dropout. |
| `dataset_num_proc` | `int \| None` | `None` | Preprocessing processes. |
| `remove_unused_columns` | `bool` | `False` | Set `True` only for pre-tokenized datasets. |
| `loss_type` | `str` | `"cross_entropy"` | `cross_entropy`, `focal`, `label_smoothing_ce`. `label_smoothing_ce` is single-label only (softmax CE) and raises on a multi-label dataset — use `focal` or `class_weights` there. |
| `focal_gamma` | `float` | `2.0` | Focusing parameter (focal loss). Ignored unless `loss_type="focal"`. |
| `focal_alpha` | `float \| None` | `None` | Positive/negative balancing factor for focal loss, applied as `alpha_t = alpha*y + (1-alpha)*(1-y)`. **Multi-label (sigmoid) heads only** — on a single-label softmax head every element has exactly one target, so a scalar alpha would be a uniform loss rescale rather than balancing, and it is rejected there. Use `class_weights` / `derive_class_weights` for per-class balancing. |
| `label_smoothing` | `float` | `0.0` | Smoothing epsilon (typical 0.05–0.1). Ignored unless `loss_type="label_smoothing_ce"`. Not to be confused with HF's inherited `label_smoothing_factor`, which this trainer **rejects**: it builds its own loss and never consults HF's `label_smoother`. |
| `class_weights` | `list[float] \| None` | `None` | Per-class weights by label ID. Applied to every `loss_type` (CE, focal, label-smoothed CE) and as `pos_weight` on the multi-label BCE. |
| `derive_class_weights` | `bool` | `False` | Auto-compute balanced class weights from label counts all-reduced across the world group (presharded/TP-safe). Classes absent from training keep weight `1.0`. Mutually exclusive with `class_weights` — setting both raises, as does using it on a multi-label dataset (pass `class_weights` explicitly there). |
| `multi_label_threshold` | `float` | `0.5` | Sigmoid threshold for multi-label predictions. |
| `compute_per_class_metrics` | `bool` | `False` | Log per-class precision/recall/F1. Single-label only — ignored on a multi-label dataset. |
| `compute_auc_roc` | `bool` | `False` | Compute AUC-ROC (multiclass = one-vs-rest macro). |
| `compute_mcc` | `bool` | `True` | Compute Matthews Correlation Coefficient. Single-label only — ignored on a multi-label dataset. |

---

## EnvironmentConfig

Environment selection for Environmental GRPO. **Source:** `src/configs/environment_config.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `environment_type` | `str` | `"react_math"` | Registry name: `react_math`, `react_search`, `native_math`, `native_coding`, `native_combined`, `swe`, `mcp`, `qa_search`, `code_contests`, `codeforces`, `exam_qa`. |
| `success_reward` | `float` | `1.0` | Reward for correct final answer. |
| `failure_reward` | `float` | `0.0` | Reward for incorrect answer. |
| `max_turns` | `int \| null` | `null` | Max environment turns per episode, `>= 1`. `null` keeps the environment class's own default (`code_contests`/`codeforces` 15, `swe` 20, `exam_qa` 8; every other environment 10). |
| `environment_kwargs` | `dict` | `{}` | Env-specific kwargs passed to the registry factory. Merged with core settings in `to_env_config()`. A key the target environment's constructor chain does not bind raises `TypeError`. |

Common `environment_kwargs` keys: `search_backend` (`qa_search` / open-book `exam_qa` — setting it with `open_book: false` is refused at construction: `duckduckgo`, `serper`, `brave`, `tavily`; `mock` only with `HALO_ALLOW_MOCK_SEARCH=1`), `open_book` (`exam_qa`, default `false`), `timeout_per_test` (`code_contests`, seconds/test, default `15`), `max_grading_seconds` (`code_contests`, wall-clock budget per submission grade, default `None` = unbounded — see [Code Contests](../training-methods/grpo/environments/code-contests.md#grading)), `reasoning_effort_profiles` (`code_contests`, effort → `{thinking_tokens, max_submissions, max_test_calls, tested_submission_reward, token_cost}` merged over defaults — see [Code Contests](../training-methods/grpo/environments/code-contests.md#reasoning-effort)), `mcp_server` (`mcp`, e.g. `filesystem`), `include_python_tools` (`qa_search`, default `false`).

```yaml
environment_type: qa_search
environment_kwargs:
  search_backend: duckduckgo
  include_python_tools: true
```

---

## AsyncTrainingConfig

`DistributedAsyncEnvironmentalGRPOTrainer` — Ray workers, vLLM connections, weight sync, rollout generation, async prefetch. **Source:** `src/configs/async_training_config.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `num_rollout_workers` | `int` | `64` | Ray environment actors, **per training rank**, `>= 1`. Async, so not a concurrency limit — size it to the env's blocking per-episode cost: CPU for sandboxed envs (bounded by the sandbox gate), remote latency for search/MCP envs. An actor needs 1 free CPU at placement only, none for its lifetime ([Ray Cluster](../infrastructure/ray.md#pool-sizing)). With `ray_address` set (shared cluster) the per-rank count becomes `this // world_size` (floored at 1), so the cluster-wide pool approximates `this`. |
| `max_concurrent_rollouts` | `int \| None` | `None` | Per-rank in-flight cap — the throughput throttle. Size to one generation cycle's per-rank demand (`per_device_train_batch_size × gradient_accumulation_steps`) with ~2× prefetch headroom; server-pool load is `this × data_parallel_size ÷ servers`. Raising it past the actual rollout count does nothing. `>= 1` when set; `null` derives 4 × this rank's share of the rollout workers (`num_rollout_workers ÷ world_size` on a shared Ray cluster, all of them locally), clamped to at least that share. |
| `ray_address` | `str \| None` | `None` | Ray cluster address (`head-node:6379`). `None` = per-rank local instances; setup and sizing on [Ray Cluster](../infrastructure/ray.md). |
| `rollout_backend` | `"vllm" \| "sglang"` | `"vllm"` | Rollout engine for generation and weight sync. `sglang` supports `routing_replay: rollout` (serve with `--enable-return-routed-experts --moe-runner-backend triton`), rejects `rollout_max_thinking_tokens` at config time, must be served from the NCCL-aligned `Dockerfile.sglang` image, and wants `fsdp_reshard_after_backward: false` on the trainer (its socket-global NCCL requirement otherwise makes FSDP2's per-microstep reshard the dominant step cost) — a plain-DP/CP/EP lever, rejected under TP or PP — see [Rollout Servers](../infrastructure/rollout-servers.md). |
| `rollout_server_url` | `str` | `"http://localhost:8000"` | Primary rollout-server URL (whichever engine `rollout_backend` selects) for weight sync + generation. |
| `rollout_connection_timeout` | `float` | `120.0` | Seconds to wait for the rollout server. |
| `rollout_server_configs` | `list \| None` | `None` | Multi-server configs (each a dict with `url` and optional `group_port` / `group_host` — the weight-transfer master address the serving node dials back to). Overrides `rollout_server_url`. |
| `sync_weights_every_n_steps` | `int` | `1` | Sync weights every N training steps. |
| `rollout_temperature` | `float` | `0.7` | Rollout generation temperature. Overwrites `GRPOConfig.temperature` (logged) so log-probs are scored at the sampling temperature the IS ratio assumes. |
| `rollout_top_p` | `float` | `0.95` | Rollout top-p. |
| `rollout_max_tokens` | `int` | `32768` | Env-GRPO's per-turn generation budget (one `/chat/completions` call). The multi-turn trajectory accumulates across turns and is bounded by the served model's context window — not truncated (a trajectory exceeding it fails). |
| `rollout_max_thinking_tokens` | `int \| None` | `None` | Per-turn reasoning-token budget for reasoning models (vLLM `thinking_token_budget`): caps the chain-of-thought, then forces the answer out of the rest of `rollout_max_tokens`. Needs a reasoning parser on the server (`--reasoning-parser qwen3` for Qwen3.x, the `openai_gptoss` plugin for gpt-oss) **and** `VLLM_USE_V2_MODEL_RUNNER=0`: Model Runner V2 answers `thinking_token_budget` with a 400 on every request, so rollouts error instead of generating. `None` = unbounded reasoning. Rejected under `rollout_backend: sglang` for every model — the trainer wires neither of SGLang's budget mechanisms, and harmony models have none server-side. |
| `rollout_stop_tokens` | `list[str]` | `[]` | Special-token strings that end a turn (resolved to vLLM `stop_token_ids`). Set to the model's tool-call terminator so a turn stops when the model emits its call — needed for a terminator that is not an eos (e.g. gpt-oss `<\|call\|>` under harmony-disabled serving). Empty = only eos stops a turn. |
| `train_on_sampled_tokens` | `bool` | `True` | Train on the server's actual sampled token ids (each assistant turn a training row sharing the trajectory advantage) instead of re-tokenizing a chat-template re-render — eliminates re-tokenization mismatch. On vLLM this needs the server's `--return-tokens-as-token-ids` (passed by `docker-compose.vllm.yml`); on SGLang the capture is per-request and needs no server flag. Falls back to a single re-tokenized row for any trajectory with an uncaptured turn. |
| `drop_degenerate_groups` | `bool` | `True` | Drop GRPO groups whose completions all scored the same reward (advantage already 0). Restores effective batch size on sparse verifiable rewards; logged as `sampling/degenerate_group_frac`. |
| `advantage_mode` | `str` | `"mean"` | Group-baseline / negative-side advantage surgery: `mean` (plain GRPO baseline), `qae` (per-group quantile baseline), `asymmetric` (scale positive/negative advantages), `neg_mask_hard` (zero negatives in groups where no member's objective reward reached `advantage_hard_group_threshold`). See [stability knobs](../training-methods/grpo/environmental-grpo.md#off-policy-mismatch-and-stability-knobs). |
| `advantage_quantile` | `float` | `0.4` | Baseline quantile for `advantage_mode: qae`. |
| `advantage_pos_scale` / `advantage_neg_scale` | `float` | `1.0` / `0.4` | Positive / negative advantage multipliers for `advantage_mode: asymmetric` (`neg_scale: 0` = full negative mask). |
| `advantage_hard_group_threshold` | `float` | `0.5` | `neg_mask_hard`: a group is hard (negatives zeroed) when no member's objective reward component reaches this value. |
| `isr_band_min` / `isr_band_max` | `float \| None` | `None` | Token band on the vLLM→trainer IS ratio: a corrected token whose raw ratio leaves the band is masked, not just truncated. Set both to activate. Requires the IS correction (raises otherwise), like every `isr_*` knob below. |
| `isr_geo_band_min` / `isr_geo_band_max` | `float \| None` | `None` | Trajectory geometric-mean band: mask a whole trajectory when `exp(mean log-ratio)` over its corrected tokens leaves the band. Set both to activate. |
| `isr_veto_min` | `float \| None` | `None` | Catastrophic-token veto: mask a whole trajectory when any corrected token's raw ratio falls below this. |
| `isr_opsm_delta` | `float \| None` | `None` | Off-policy sequence masking: mask negative-advantage trajectories whose \|mean log-ratio\| exceeds this many nats; positives never masked. |
| `skip_update_masked_frac` | `float \| None` | `None` | Trust-region circuit breaker: zero the step's policy gradient when more than this fraction of IS-corrected trajectories are fully masked by the stages above (the survivors are a selection-biased sample). Any KL term still applies. Logged as `sampling/update_skipped`. Requires the IS correction **and** at least one mask stage (`isr_band`/`isr_geo_band`/`isr_veto`/`isr_opsm`) — raises otherwise, since nothing would ever be masked. |
| `scale_rewards_std_floor` | `float` | `0.0` | Floor (reward units) on the advantage-scaling std divisor: divide by `max(std, floor)`, bounding degenerate-batch noise amplification. `0` = off. |
| `routing_replay` | `str` | `"none"` | MoE routing replay. `recompute` (R2) captures each EP MoE layer's top-k selection in the no-grad logprob-recompute pass and replays it in the update/GC forwards, re-deriving gate weights from live router scores. `rollout` (R3) replays the rollout engine's per-turn routing to close the cross-engine gap; it needs `train_on_sampled_tokens` and a capture-capable server — vLLM ≥0.22 with `--enable-return-routed-experts` and a non-FlashInfer MoE backend (`--moe-backend triton`), or SGLang with `--enable-return-routed-experts` and `--moe-runner-backend triton` (the `triton_kernel`/flashinfer runners bypass the capture hook). EP-wrapped MoE only; Gemma4 and Zaya are rejected. |
| `episode_timeout` | `float` | `1200.0` | Wall-clock deadline (seconds) for one whole rollout episode — generation + tool execution + grading — unlike `request_timeout`, which bounds a single HTTP call. A timed-out episode is cancelled and counted in `episode/error_rate`, and a straggler holds its peers at the per-step collective for up to this long. It must stay under the NCCL watchdog (`DIST_NCCL_TIMEOUT_MINUTES × 60`): a larger value **raises when training starts**, and ≥80% of it warns. The default sits at two thirds of the 30-min watchdog, leaving ~10 min of margin; raise `DIST_NCCL_TIMEOUT_MINUTES` before raising it. |
| `enable_prefetch` | `bool` | `True` | Overlap rollout collection with training (multi-server only; auto-disabled for one server). The resulting one-sync off-policy staleness is corrected by the truncated vLLM importance-sampling ratio built from the batch's sampling logprobs (clamped by TRL's `vllm_importance_sampling_clip_max`, default 3.0). |
| `num_prefetch_batches` | `int` | `1` | Batches to prefetch ahead. Must be `>= 1`; set `enable_prefetch: false` to turn prefetching off (`0` would make the queue unbounded). |
| `use_chunked_grpo_logprobs` | `bool` | `False` | Compute completion log-probs from the hidden state + a vocab-chunked softmax instead of full `[B, T, vocab]` logits — bounds the loss-forward peak by chunk size, not `B·T·vocab`. For large-vocab models on long trajectories. See [Environmental GRPO](../training-methods/grpo/environmental-grpo.md#chunked-log-probs). |
| `reasoning_compliance_weight` | `float` | `0.0` | Weight of the asymmetric reasoning-budget calibration reward (`0` = off). Active when the episode has a CoT budget — the env's per-effort budget when `reasoning_effort` is set, otherwise `rollout_max_thinking_tokens`: no penalty inside `[0.3, 0.9]×` the budget, mild penalty for under-use, strong penalty for over-use or truncation (down to `-weight`). `~0.15` shapes without dominating the task reward. |
| `model_name` | `str \| None` | `None` | Model name for `/v1/chat/completions`. Optional — vLLM uses the loaded model when omitted. |
| `request_timeout` | `float` | `120.0` | HTTP timeout per vLLM request (seconds). |
| `max_retries` | `int` | `3` | Max retries for transient vLLM failures (exponential backoff). |
| `retry_base_wait` | `float` | `1.0` | Base wait (seconds) for backoff. |

---

## Script arguments

### CommonScriptArguments

Base shared by all training scripts. **Source:** `src/args/common_script_args.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `dataset` | `str \| list[str]` | `"/path/to/dataset"` | An `org/name[:config][@split]` hub id, an `s3://` URI, a `save_to_disk` directory, or a single `.jsonl`/`.json`/`.parquet`/`.arrow`/`.csv` file. A list of any of these for multi-dataset training. Placeholder default with no validation; an unset value fails at dataset load. |
| `dataset_ratio` | `float \| None` | `None` | Fraction of each dataset to take (0–1). A YAML **list** gives one ratio per `dataset` entry; that form is YAML-only (the field is annotated `float` so `HfArgumentParser` can build, so a list cannot be spelled on the CLI). |
| `test_size` | `float \| None` | `None` | Test split proportion (e.g. `0.05`). Leave empty if the dataset already has a test split. |
| `project_name` | `str` | per-script | wandb/clearml project. `__post_init__` refuses `null` and blank, then replaces the `"default-project"` sentinel with the script's own (e.g. SFT → `sft-tuning`). |
| `pad_token` / `bos_token` / `eos_token` | `str \| None` | `None` | Override special tokens. |
| `chat_template` | `str \| None` | `None` | Path to a `.jinja`/`.jinja2`/`.j2` file, or template string. |
| `force_chat_template` | `bool` | `False` | Force custom template even if the tokenizer has one. |
| `added_special_tokens` | `list[str] \| None` | `None` | Additional special tokens, added to the tokenizer's existing extra-special set rather than replacing it (transformers' own default replaces, which would drop every control token the checkpoint shipped from `all_special_ids` and from the exported `tokenizer_config.json`). Re-listing a token already present is a no-op. Applied before the `pad_token`/`bos_token`/`eos_token` roles, which read an id back. |
| `tokenizer_backend` | `str` | `"hf"` | Text→ids backend: `"hf"` or `"gigatoken"` (optional extra; token IDs verified identical at startup). Resolved in `setup_model_and_tokenizer`, so every training method honors it; embedding rejects it (SentenceTransformer owns tokenization) — see [Dataset Pre-Processing](../data/dataset-preparation.md). |
| `tools_field` | `str \| None` | `None` | Dataset field with tool definitions passed to `apply_chat_template`. Reward modeling aliases it onto `tools`, the one column TRL's `RewardTrainer` templates. Everywhere the render cannot carry it, it is **rejected** rather than parsed and dropped: outright by KTO and embedding; by environmental GRPO, whose rollout schema comes from the environment's own tool registry; and on the vision arms of SMPO, DPO and reward modeling, whose pair render templates without `tools=`. |
| `unfreeze_layers_patterns` / `freeze_layers_patterns` | `list[str] \| None` | `None` | Layer-name patterns to unfreeze / freeze (freeze applied after unfreeze). Under **pipeline parallelism** a pattern that pins a decoder-layer index **raises**: each stage holds only its own layers, re-based to index 0, so a global index selects nothing on most stages and the wrong layer on the rest. Any segment following `layers`/`h` containing a digit counts, including glob character classes (`model.layers.[6-8][0-9].*`); index-free patterns (`*.self_attn.sinks`, `*.mlp.experts.0.*`) pass. A pattern that matches nothing **raises** on either knob — `unfreeze` would leave the model fully frozen, `freeze` would leave what it named training. They match different things: `unfreeze_layers_patterns` is fnmatch against full **module** names, `freeze_layers_patterns` against full **parameter** names. |
| `enable_efficiency_metrics` / `enable_moe_metrics` / `moe_balancing` / `router_balancing_rate` / `num_full_model_params` | — | — | See [Performance & balancing flags](#performance-balancing-flags). |
| `report_mfu_diagnostics` | `bool` | `False` | Add MFU / S-MFU / achieved-TFLOPS to the headline log. Values are computed every step regardless; this only controls headline visibility. tokens/s/GPU is the reported throughput metric. |
| `save_completions` | `bool` | `True` | GRPO family (online / env): write per-step completions and trajectories to `<output_dir>/completions/completions_<step>.parquet` (+ a backend `completions` table). Text is rendered from detokenized message content. Independent of TRL's console-only `log_completions`; ignored by non-generating trainers (SFT, offline GRPO). |
| `log_decoded_samples` | `bool` | `False` | Write the first few decoded train/eval samples (`skip_special_tokens=False`) to `<output_dir>/log/{train,eval}_sample.txt`. Written on the FS-aware save rank (same gate as `run.log`: global rank 0 on a shared output FS, each node's local rank 0 when `DIST_OUTPUT_SHARED_FILESYSTEM` — or the umbrella it falls back to — is `0`); datasets without `input_ids` are skipped. |

### SFTScriptArguments

**Source:** `src/args/sft_args.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `conversation_field` | `str \| None` | `"prompt"` | Dataset field with conversations. |
| `images_field` | `str \| None` | `None` | VLM only: image column (HF `Image`, single or list) injected into the first user turn — hub datasets that keep images outside the conversation (FineVision/the_cauldron/Docmatix) train without preprocessing. Setting it declares the run VLM on a multimodal checkpoint, before the dataset is probed ([declaration rules](../data/dataset-formats.md#sft-vlm)). |
| `system_prompt` | `str \| None` | `None` | System prompt if none in dialogue. |
| `train_on_completions_only` | `bool` | `True` | Train only on assistant completions, masking prompt tokens. |
| `train_on_last_assistant_only` | `bool` | `False` | Train only on the last assistant message. Requires `train_on_completions_only=True`. |
| `generate_eval_examples` | `bool` | `False` | Generate text examples during eval. |
| `assistant_message_template` | `str \| None` | `None` | The rendered assistant-turn prefix of the model's chat template; required when `train_on_completions_only` is on. |
| `num_eval_examples` | `int` | `50` | Examples to generate during eval. |
| `model_supports_system_role` | `bool` | `True` | If `False`, system prompt is prepended to the first user message. |
| `interleaved_thinking` | `bool` | `False` | Pass `clear_thinking=False` to `apply_chat_template`. Needed for templates with a `clear_thinking` switch (e.g. GLM) when the rollout backend kept historical assistant reasoning — the template default otherwise strips `<think>…</think>` from history. Text-only: the VLM path raises. |

### RLVROnlineGRPOScriptArguments

RLVR (verifiable-reward) Online GRPO with rule-based rewards. **Source:** `src/args/rlvr_online_grpo_args.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_prompt_length` | `int \| None` | `None` | Prompt budget applied as a dataset **filter**: rows above it are dropped, never truncated. `null` = no filtering. |
| `prompt_field` | `str` | `"prompt"` | Field with the prompt (string or conversation list). |
| `answer_field` | `str` | `"answer"` | Field with the ground-truth answer. |
| `system_prompt` | `str \| None` | `None` | System prompt to prepend. |
| `reasoning_effort` | `str \| None` | `None` | Chat-template reasoning-effort steer: `low`/`medium`/`high`, `random` (sampled per prompt), or `None`. Passed via `chat_template_kwargs`; needs a template that reads `reasoning_effort` (e.g. gpt-oss harmony). |
| `use_accuracy_reward` | `bool` | `True` | Accuracy reward: `\boxed{}` content matches ground truth. |
| `use_format_reward` | `bool` | `False` | Format reward: completion matches `format_pattern`. |
| `format_pattern` | `str` | `<think>.*?</think>\s*<answer>.*?</answer>` | Regex for format reward. |
| `accuracy_reward_weight` | `float` | `1.0` | Accuracy reward weight. |
| `format_reward_weight` | `float` | `0.5` | Format reward weight. |
| `use_chunked_grpo_logprobs` | `bool` | `False` | Vocab-chunked completion log-probs instead of full `[B, T, vocab]` logits — for large-vocab models on long completions. See [Online GRPO](../training-methods/grpo/online-grpo.md#chunked-log-probs). |

The same args class also carries the `use_rlrr` (relative-reward advantage shaping) and `use_sdpg` (privileged-teacher OPD on positive-advantage rollouts) toggles and their tuning fields — see [RLVR Online GRPO](../training-methods/grpo/online-grpo.md).

It also inherits `AdvantageShapingArguments` (`src/args/mixins.py`): `advantage_mode`, `advantage_quantile`, `advantage_pos_scale`, `advantage_neg_scale`, `advantage_hard_group_threshold`, `scale_rewards_std_floor`, `drop_degenerate_groups` — same semantics as the [AsyncTrainingConfig](#asynctrainingconfig) rows, except `drop_degenerate_groups` defaults **`false`** here (it is `true` in env-GRPO, whose sparse verifiable reward makes dead groups dominate the batch).

### EnvironmentalGRPOScriptArguments

**Source:** `src/args/environmental_grpo_args.py`

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_prompt_length` | `int \| None` | `None` | Prompt budget applied as a dataset **filter**: rows above it are dropped, never truncated. `null` = no filtering. |
| `prompt_field` | `str` | `"prompt"` | Field with the prompt or question. |
| `answer_field` | `str \| None` | `"answer"` | Field with the expected answer (reward computation). |
| `context_fields` | `list[str] \| None` | `None` | Additional fields passed as context to the environment. |

There is no `system_prompt`: the environment builds the rollout conversation from its OWN system prompt plus the dataset's user turn, so the key is not declared here and a config setting it raises. Configure it on the environment instead (`environment_kwargs: {system_prompt: ...}`).

### Other script arguments

These extend `CommonScriptArguments` with minimal additions:

| Class | Source | Extra Fields |
|-------|--------|-------------|
| `SMPOScriptArguments` | `src/args/smpo_args.py` | `generate_eval_examples` (`True`), `num_eval_examples` (`50`) |
| `DPOScriptArguments` | `src/args/dpo_args.py` | `generate_eval_examples` (`True`), `num_eval_examples` (`50`), `generation_max_prompt_length` (`512`) — bounds the eval-time generation dataset only; DPO has no training-side prompt cap — and `images_field` (`None`) — VLM only: image column renamed to the `images` spelling TRL's vision probe reads, ahead of the dispatch |
| `OfflineGRPOScriptArguments` | `src/args/offline_grpo_args.py` | `generate_eval_examples` (`True`), `num_eval_examples` (`50`) |
| `KTOScriptArguments` | `src/args/kto_args.py` | `completion_field` (`"completion"`), `label_field` (`"label"`) — KTO uses an *unpaired* `{prompt, completion, label}` dataset — and `images_field` (`None`) — VLM only: renamed to TRL's `images` spelling; the vision path refuses `precompute_ref_log_probs` and paired columns |
| `RMScriptArguments` | `src/args/reward_args.py` | `images_field` (`None`) — VLM only: image column, renamed to `images` and merged into the shared prompt conversation; declares the run VLM on a score-headed multimodal checkpoint ([reward VLMs](../training-methods/preference/reward-modeling.md#vision-language)) |
| `CLFScriptArguments` | `src/args/classification_args.py` | `text_field` (`None`) — raw-text column wrapped as one user turn when the dataset has no `prompt` conversation (e.g. text/label sets like imdb) |
| `EmbeddingScriptArguments` | `src/args/embedding_args.py` | *None (inherits only)* |
| `SelfDistillationArguments` | `src/args/self_distill_args.py` | Extends `SFTScriptArguments` with privileged-context + OPD-loss fields — see [Self-Distillation](../training-methods/distillation/self-distillation.md) |

---

## YAML shape {#yaml-config-examples}

A config is the method's fields plus any `TrainingArguments` field. `bf16: true`,
`use_liger_kernel: true` and `logging_nan_inf_filter: false` are toolkit defaults and need not be
written.

```yaml
model_name_or_path: "meta-llama/Llama-3.1-8B-Instruct"
dataset: "s3://bucket/sft-dataset"
conversation_field: "conversation"
train_on_completions_only: true
max_length: 16000
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
learning_rate: 0.00004
gradient_checkpointing: true
output_dir: "checkpoints/sft-run"
```

Working configs for every method and model family live under `examples/<method>/<family>/`; each
method page links its own.
