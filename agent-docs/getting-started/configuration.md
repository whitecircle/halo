# Configuration Guide

Every training script takes a YAML config as its first positional argument, parsed into HuggingFace `TrainingArguments` (or a method-specific subclass) plus toolkit fields.

```bash
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

## YAML parser

`H4ArgumentParser` (`src/training/parser.py`) extends `HfArgumentParser` with:

- **Toolkit defaults**, applied unless explicitly set in the YAML or on the CLI: `use_liger_kernel: true`, `bf16: true`, `logging_nan_inf_filter: false`. Each is a value the YAML did not write, handed to every config declaring the field before it is built, so `TrainingArguments` derives `mixed_precision` from it. The `bf16` default yields to an explicitly-enabled `fp16`.

    Upstream's `logging_nan_inf_filter: true` reads a device scalar back to the host per micro-batch and logs the running average in place of a NaN loss: a per-step sync plus a hidden divergence.

- **Literal validation** — fields annotated `Literal[...]` (e.g. `advantage_method`) reject out-of-set values at parse time, in YAML and `--key=value` alike. Mixed unions (`float | Literal["auto"]`) are not validated.
- **Boolean spellings** — YAML 1.2 booleans are unquoted `true`/`false`; the 1.1 spellings `yes`/`no`/`on`/`off` parse as (truthy) strings, so a string value on a bool field is rejected at parse time instead of silently inverting `packing: no`.
- **No field renames.** Nothing is migrated and nothing is stripped: every key no config declares raises and names the field, ecosystem spellings this toolkit does not accept included (TRL's own `max_seq_length` — use `max_length`).
- **`output_dir` strftime expansion** — `%<letter>` directives expand from one shared timestamp, so `checkpoints/sft-%Y-%m-%dT%H-%M-%S` becomes a timestamped path; `%%` collapses to a literal `%`, and any other `%` (`sft-100%-data`) survives byte-identical.
- **Distributed init happens first.** `parse()` calls `init_distributed()` before building the dataclasses: constructing a `TrainingArguments` touches `self.device`, which lets accelerate create the default process group *without* `device_id`. Losing that binding costs a `new_group` per mesh dimension instead of one `ncclCommSplit` and leaves every barrier guessing its device.

## CLI overrides

Any YAML field can be overridden as `--param=value` (dashed spellings normalize to the underscore
field name); CLI wins. The `=` is required: a space-separated `--key value` raises "CLI overrides
must be in --key=value form".

An override matching no field on any of the script's config dataclasses fails loudly, as does an
unknown YAML key. `--field=None`, `--field=null` and `--field=none` all clear any Optional field
instead of setting the literal string, container unions included. Three carve-outs: a `Literal`
whose choices include the string `"none"` gets the string (`--moe_balancing=none`); a field whose
default spells "no value" as a string of its own gets that string — `--report_to=none` reaches
transformers as `"none"`, no integrations, the standard way to silence logging on a smoke run, where
`None` would come out of its `__post_init__` as `[None]`, which the Trainer refuses; and an optional
bool refuses the spelling outright (`--bf16=none` raises — clearing a precision flag silently is
exactly the failure the parser exists to prevent).

Setting a value on a field with no confident string cast — dict-typed fields, lists of containers
(`rollout_server_configs: list[dict]`), and unions with a container member (`dataset: str | list[str]`,
`--report_to=wandb`) — still requires the YAML.

Each override is cast to its field's type and merged over the YAML, and every config dataclass is
built once from the merged values: each `__post_init__` — transformers', TRL's and the toolkit's —
derives its state from, and validates, the final configuration. `--gradient_accumulation_steps=1`
over a YAML's `8` therefore also re-derives GRPO's `steps_per_generation` and
`generation_batch_size`, and a CLI value meets exactly the guards a YAML value does.

```bash
python scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml \
    --learning_rate=0.00001 --num_train_epochs=2 --output_dir=checkpoints/experiment-v2
```

Without a YAML, the whole config is flags parsed by argparse: `--key value` works and an unknown flag
is a usage error. Each flag reaches every config dataclass that declares it, as a YAML key does
(`pad_token` sits on the script args and on TRL's config), and an unset field keeps each dataclass's
own default.

## Liger kernels

`use_liger_kernel: true` (default) enables fused Triton kernels (cross-entropy, RMSNorm, SwiGLU, RoPE); Qwen3 MoE, GLM-4.7-Flash, GPT-OSS and Gemma 4 fill the RMSNorm role with torch's fused `F.rms_norm`. Four safety filters (`liger_parallelism_overrides` and `liger_routed_expert_overrides`, `src/kernels/liger/orchestrator.py`) override the defaults, applied by one sanitizer (`sanitize_liger_config`) at model load and again when the trainer re-sanitizes the config HF Trainer re-applies:

- **Wrapped MoE experts** — `swiglu`/`geglu` fusion off, because the EP or grouped-GEMM wrapper replaces the expert FFN Liger would swap.

    The trigger is `liger_ep_disables_fused_glu` (`src/kernels/liger/orchestrator.py`): the run needs EP wrappers (`ep_size > 1`, `expert_tensor_parallel_size > 1` including pure ETP, or `use_grouped_gemm`), the model has experts, the family has a registered EP layer class, **and** the applier that owns the GLU swap is the one the wrapper replaces.

    Dense runs keep fused SwiGLU, as does a family whose toolkit Liger spec patches the dense and shared-expert MLPs the wrappers adopt unchanged.

- **Unwrapped MoE experts** — `swiglu` off, even when requested, where Halo does not wrap the routed experts (`ep_size: 1` with `use_grouped_gemm: false`, or a family with no EP layer class) and upstream liger-kernel holds the flag, which its MoE appliers use to install `LigerExperts` (input gradient wrong on Blackwell in the pinned release). The flag goes whole, so upstream's dense, shared-expert and vision SwiGLU on such a model run eager too ([Routed experts](../optimization/liger-kernels.md#routed-experts)).

- **TP** (`tp_size > 1`) — `cross_entropy` and `fused_linear_cross_entropy` off; the `lm_head` logits are DTensor-sharded across the vocab dim, so a fused softmax would see a partial vocab.
- **CP** (`cp_size > 1`) — same two off: the CP wrapper computes the loss outside the model's forward, so the fused path never fires and its memory saving does not exist.

`fused_linear_cross_entropy` is otherwise opt-in, defaulting on only for DeepSeek-V4, GLM-4 MoE Lite, and Zaya. Override individual kernels with `liger_kernel_config`:

```yaml
liger_kernel_config:
  cross_entropy: false
  fused_linear_cross_entropy: true    # fuses lm_head + CE, never materializes the logits tensor
```

`cross_entropy` and `fused_linear_cross_entropy` cannot both be `true`.

## Launcher selection

`halo launch <method> <config>` (`src/cli.py`) resolves the method to a script under
`scripts/training/` and picks the launcher: `accelerate launch` whenever `-a launcher-configs/accelerate/<config>.yaml`
is given (there `-n N` becomes `--num_processes`), else `torchrun` when `-n N` sets more than one
process (required for EP/CP/TP/ETP), else plain Python. A plain-Python run trains on the one GPU it binds however many are visible: `init_training_script` pins `n_gpu` to 1, so HF never wraps the model in `nn.DataParallel` (whose loss and gradients scale with the GPU count). Every flag the launcher does not own reaches the trainer, so overrides follow the config directly (`halo launch sft cfg.yaml -n 8 --learning_rate=1e-5`); a standalone `--` is needed only before a flag that collides with the launcher's own (`--help`, `--port`, `--dry-run`, ...).
`--list` prints the method names, `--dry-run` prints the command, `-p <port>` sets the rendezvous port
so concurrent launches do not collide (it raises on a single-process launch). A missing config path
fails before any process starts.

A launch runs **from the repository root**, so relative paths in the config (and in CLI overrides)
resolve there, not against the caller's directory.

`halo run <tool>` indexes the top-level `scripts/` subtrees other than `training/` and `diagrams/`
(post-training, data prep, inference, environments, profiling) and launches any script there carrying
a `__main__` guard. A tool keeps the **caller's** working directory: its relative path flags mean what
they would had the script been run directly.

EP/CP/TP/ETP/PP under `accelerate launch` **raise** at startup for any `distributed_type` (pure ETP folds into the EP check, which keys on `ep_group_size > 1`). The guard (`is_accelerate_launch`, `src/env.py`) keys on `ACCELERATE_MIXED_PRECISION` — set by the launcher for every config — or `ACCELERATE_USE_FSDP`. Use `torchrun`. An MoE model raises under `accelerate launch` too unless `use_grouped_gemm: false`: the grouped-GEMM expert wrappers (on by default) need `torchrun`.

| Scenario | Command |
|---|---|
| Data parallel (FSDP2) | `torchrun --nproc_per_node=8 script.py config.yaml` |
| Data parallel via accelerate | `accelerate launch --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml script.py config.yaml` |
| EP | `torchrun --nproc_per_node=8 script.py config.yaml --expert_parallel_size=8` |
| TP | `torchrun --nproc_per_node=8 script.py config.yaml --tensor_parallel_size=8` |
| CP | `torchrun --nproc_per_node=4 script.py config.yaml --context_parallel_size=4` |
| ETP | `torchrun --nproc_per_node=8 script.py config.yaml --expert_tensor_parallel_size=8` |
| PP | — not yet available in this release ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)) |
| EP+CP / EP+TP | add both flags |
| Multi-node | add `--nnodes`, `--node_rank`, `--master_addr`, `--master_port` |

EP+ETP (`ep_size>1` and `expert_tensor_parallel_size>1`) is supported but experimental: the expert-TP reduction runs in token space so the coupled DeepEP dispatch groups don't deadlock the combine barrier under FSDP2. Its expert-TP groups stay NVLink-local — across domains it runs as one EP group with exactly one ETP group per domain — and it cannot combine with attention TP. See [ETP validation rules](../parallelism/expert-tensor-parallelism.md#validation-rules).

FSDP2 (`fully_shard`) is applied automatically for all `torchrun` modes, with gradients and optimizer states sharded across DP ranks; the EP modules stay outside FSDP2 except at `ep_group_size == 1`, where `fsdp_shard_ep1_experts` (default `true`) hands the experts to FSDP2 as well ([Data Parallelism](../parallelism/data-parallelism.md)).

Three `torchrun`-only knobs tune it: `fsdp_reshard_after_forward` (ZeRO-3), `fsdp_reshard_after_backward` (keep params unsharded across an accumulation window) and `fsdp_defer_grad_sync` (reduce once per optimizer step). Each row in [ParallelismConfig](../reference/configuration-reference.md#parallelismconfig) states its refusals; the measured trade-offs are in [ZeRO-2 vs ZeRO-3](../parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward) and [Deferred gradient reduce](../parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync).

## Example SFT config

```yaml
model_name_or_path: Qwen/Qwen3-8B      # HF ID, S3 path, or local dir
# attn_implementation unset -> auto-detected (see below)

dataset:                                 # HF Hub, S3 URI, or local; list = concat
  - s3://bucket/chat_sft_data
  - s3://bucket/quality_sft_data
conversation_field: conversation         # JSON key holding the conversation list
train_on_completions_only: true          # loss on assistant turns only
assistant_message_template: "<|im_start|>assistant\n"   # ChatML — required with train_on_completions_only; no default fits every template
test_size: 0.02                          # eval holdout fraction

per_device_train_batch_size: 1           # micro batch / GPU; keep 1 for long seq
gradient_accumulation_steps: 32          # eff batch = bs x accum x num_gpus
learning_rate: 7.0e-06
lr_scheduler_type: cosine
warmup_steps: 15
num_train_epochs: 1
max_length: 20000                        # over-length conversations are dropped, not truncated
gradient_checkpointing: true
optim: adamw_torch_fused
# bf16: true, use_liger_kernel: true     # enabled by default

output_dir: checkpoints/sft-qwen3-8b
save_strategy: steps
save_steps: 50
save_total_limit: 3
save_only_model: true
report_to: wandb                         # wandb | clearml | none
logging_steps: 1

use_peft: false                          # true + lora_* for PEFT
pad_token: "<|endoftext|>"               # model-specific special tokens
eos_token: "<|im_end|>"
```

Leave `attn_implementation` unset — a manual pin is arch-specific. `_detect_attention_impl`
(`src/models/patches/attention.py`) picks `flash_attention_4` on Blackwell (SM 10.0+),
`flash_attention_3` on Hopper when installed, else `flash_attention_2`.

The loader then drops off FlashAttention where it cannot serve: Gemma 4's `head_dim=512` and the FA4
backward-NaN families (`qwen3_5*`, `qwen3_next*`, `glm4_moe_lite`) fall to `sdpa`; fp32 training
falls to `sdpa`, or `flex_attention` for a sinks model; DeepSeek-V4 is forced to `eager` from any
implementation. See [Flash Attention](../optimization/flash-attention.md).

`dataset_ratio` is a per-dataset keep-fraction applied before concatenation (a single float applies to all).

`moe_balancing` (default `auto`) and `router_balancing_rate` are `CommonScriptArguments` fields every script takes; the modes are `auto | none | aux_loss | bias_update | bias_update_transient`. See [Callbacks](../training-methods/callbacks.md).

Method-specific fields (`beta`, `loss_type`, `advantage_method`, `environment_type`, …) are listed per method in the [Configuration Reference](../reference/configuration-reference.md) and on each method's page.

## Config file locations

Configs live under `examples/<method>/<model-family>/`: `sft/`, `preference/`, `grpo/{offline,online,environmental}/`, `reward/`, `classification/`, `embedding/`, `distillation/`. SFT families are `cohere2_moe, deepseek_v4, gemma4, glm4, glm5_next, gptoss, inkling, laguna, lfm2, ling_mini_2, mistral4, qwen3, qwen3_5, step3p7, zaya`.

Async GRPO with Environments adds a rollout-backend level below the family — `environmental/<family>/{vllm,sglang}/`, with `sglang` files for gpt-oss, Qwen3.5/3.6 and Gemma 4. The GRPO templates (`examples/grpo/online/rlvr-online-grpo-template.yaml`, `examples/grpo/environmental/environmental-grpo-template.yaml`) sit at the top of their method folder.

A family directory is a snake_case family token (`qwen3_5`, `deepseek_v4`, `ling_mini_2`). File names start with the model's name (`gptoss-20b-…`, `glm-4.7-flash-…`), behind a method prefix outside `sft/` and the Async GRPO recipes (`dpo-gemma4-26b-a4b-…`, `offline-grpo-qwen3.6-…`). `qwen3_5/` also holds the `qwen3.6-*` configs, since Qwen3.6 ships under the Qwen3.5 model types.

## Accelerate configs

| File | Type | Sharding | When |
|---|---|---|---|
| `launcher-configs/accelerate/fsdp2_gradop_config.yaml` | FSDP v2 | grad/optimizer (`fsdp_reshard_after_forward: false`) | Default. Plain data parallelism only |
| `launcher-configs/accelerate/fsdp2_full_config.yaml` | FSDP v2 | full (`fsdp_reshard_after_forward: true`) | Large models, lower peak memory |
| `launcher-configs/accelerate/multigpu_dp_config.yaml` | DDP (`MULTI_GPU`) | none (replicated) | Online GRPO with LoRA, or when FSDP breaks |

Start with the gradop config; if OOM, try the full one. A hand-written accelerate FSDP **v1** config triggers a state-dict corruption warning — use the `fsdp2_*` files.

## Optimizer selection

| `optim` | Optimizer | Memory/param | Best for |
|---|---|---|---|
| `adamw_torch_fused` | PyTorch fused AdamW | 12 B over fp32 params (`bf16: false`) | `bf16: false` runs |
| (auto with `bf16: true`) | AdamWBF16 (stochastic rounding) | 6 B | The default under `bf16: true` |
| `muon` | Muon (Newton-Schulz) | ~4 B on 2D params | Orthogonalized updates on matrix params |
| `flash_adamw` | FlashAdamW (quantized states) | ~5 B | Maximum memory savings, drop-in AdamW |

AdamWBF16 replaces `adamw_torch_fused`/`adamw_torch` automatically when `bf16: true`, except under accelerate-managed DDP. `bf16_optimizer` (default `null` = that auto rule) overrides it either way: `true` is the opt-in under DDP, `false` runs the stock AdamW over the params as loaded (bf16 masters with round-to-nearest under `bf16: true`, not fp32 — fp32 masters come from `fp32_non_ep_params` (non-expert params), `fp32_router` and `fp32_experts` (EP routers and experts), or `bf16: false`). The stock AdamW is refused where plain-tensor experts sit beside FSDP2 DTensors (any `ep_group_size > 1`, or EP-wrapped experts at `ep_size: 1` with `fsdp_shard_ep1_experts: false`), unless `fp32_non_ep_params` is set, which routes to a per-tensor-type grouped AdamW with fp32 masters on the non-expert params only. `true` alongside `optim: muon` or `flash_adamw` raises — both name an optimizer, and one would silently win. FlashAdamW ships in both training images; on a bare host install the `flash-optimizers` extra (`uv pip install -e ".[flash-optimizers]"`). See [BF16 Optimizer](../optimization/bf16-optimizer.md#compatibility), [Muon](../optimization/muon-optimizer.md), [FlashAdamW](../optimization/flash-adamw.md).

## Low-precision compute

`lowp_precision: fp8 (mxfp8) | fp4 (nvfp4) | mxfp4` runs matmuls in a block-scaled low-precision format while keeping bf16/fp32 master weights and the checkpoint unchanged — quantization-aware training. It requires a bf16/fp32 master (rejected with `fp16` or a `quantization_config`).

It is **not** a throughput win: the default simulated fake-quant path re-quantizes operands every step, and the native DeepGEMM kernel (`HALO_DEEPGEMM_NATIVE=1`) is also net-slower, so `bf16` stays the production default. The real win is inference memory — convert the master with `scripts/after_training/quantize_to_lowp.py`. See [Low-Precision MoE](../optimization/low-precision-moe-kernels.md).

It is also **SFT-only**: every other training script builds its `ParallelismConfig` with `allow_low_precision=False`, which rejects a non-`bf16` `lowp_precision` and any stray `lowp_apply_*` / `lowp_keep_*` knob rather than parsing it and doing nothing.

## Parallelism parameters

CLI flags under `torchrun` (also settable in YAML):

| Parameter | Description |
|---|---|
| `--expert_parallel_size` | GPUs for MoE expert distribution |
| `--context_parallel_size` | GPUs for sequence splitting (Ulysses attention) |
| `--tensor_parallel_size` | GPUs for DTensor weight sharding |
| `--expert_tensor_parallel_size` | GPUs for expert FFN sharding (experimental) |
| `--pipeline_parallel_size` | Pipeline stages over the decoder layers. Only `1` (the default) is accepted in this release — larger values are rejected at config time, and the other `--pipeline_*` knobs are rejected at `pipeline_parallel_size == 1` ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md)) |
| `--pipeline_microbatches` / `--pipeline_schedule` / `--pipeline_split` | PP-only knobs (microbatch count; `1f1b`/`gpipe`; per-stage layer counts) — see above |
| `--nvlink_domain_size` | GPUs reachable over NVLink — the locality unit for node-local EP/CP/TP/ETP grouping (default `0` = auto: the `NVLINK_DOMAIN_SIZE` env var, else `gpus_per_node`) |
| `--ep_lazy_loading` | Lazy safetensors loading for EP paths (default `true`); each rank reads only its expert slice |
| `--max_concurrent_loading` | Ranks loading per node. Left unset it adapts to the node — `min(4, max(1, local_world_size // 2))`, so 4 on an 8-GPU node and 2 on a 4-GPU tray; any explicit value is used verbatim (`1` for CPU-RAM-constrained hosts, `0` for all-parallel) |

EP is orthogonal to data parallelism; only TP, CP, and ETP reduce it — `data_parallel_size = world_size / max(tp_size, cp_size, expert_tp_size)`. See [Parallelism](../parallelism/README.md) for supported and rejected combinations.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DIST_SHARED_FILESYSTEM` | `1` | Set `0` if nodes do not share a filesystem |
| `DIST_NCCL_TIMEOUT_MINUTES` | `30` | NCCL collective-watchdog timeout; inherited by every EP/CP/TP subgroup. Does **not** bound DeepEP's own dispatch/combine barrier |
| `NVLINK_DOMAIN_SIZE` | `gpus_per_node` | Env-var form of `--nvlink_domain_size` above; applies to every entrypoint, not just the training scripts. Raise it to the rack width on GB200/GB300 NVL72 |
| `HF_HOME` / `HF_DATASETS_CACHE` | `~/.cache/huggingface[/datasets]` | HuggingFace model / dataset cache |
| `HALO_DATA_ROOT` | `~/.cache/halo` | Toolkit scratch root (S3 dataset cache, profiler artifacts) |
| `TMPDIR` | `/tmp` | Temp files |

Point the cache/scratch/temp vars at a **verified** large mounted volume (`df -h` / `findmnt` — a `/mnt` path is not always a big array). Secrets (`WANDB_API_KEY` / `HF_TOKEN` / `AWS_*`) come from `.env` and are never auto-loaded. Full catalog: [Configuration Reference](../reference/configuration-reference.md#environment-variables).
