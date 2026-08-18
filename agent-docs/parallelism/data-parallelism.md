# Data Parallelism

PyTorch FSDP2 (`fully_shard`) handles gradient synchronization for `torchrun` multi-GPU training,
applied automatically by `DistributedTrainerMixin` from the active parallelism mode. For training
without EP/CP/TP, `accelerate launch` with a pre-built config provides FSDP v2 or plain DDP
instead.

The wrap decision keys on the **wrap width** each mode passes — the rank block, or the EP group under
deferred DP — not on `data_parallel_size`. A single GPU and pure TP skip the wrap, QLoRA takes
post-accumulate all-reduce hooks instead, and a CP run at `data_parallel_size == 1` is still wrapped
over the whole rank block.

## Two launchers

`torchrun` uses FSDP2 exclusively and is required for any EP/CP/TP run — the mixin coordinates FSDP2
with parallelism-specific gradient hooks ([How the mixin manages FSDP](#how-the-mixin-manages-fsdp)).

`accelerate launch` reads an FSDP config YAML and is for standard multi-GPU without EP/CP/TP. Both
paths produce equivalent results there. Dense models run under `accelerate launch` (FSDP or DDP) even
at the default `use_grouped_gemm: true` — the grouped-GEMM expert wrappers only activate for MoE
models. An **MoE** model with `use_grouped_gemm: true` is rejected at load under any `accelerate
launch`: the wrappers require the mixin-managed FSDP2 path. Launch with `torchrun`, or set
`use_grouped_gemm: false` to train the MoE under accelerate.

## FSDP2 strategy by mode

The mixin applies `fully_shard` (`reshard_after_forward=False` by default, configurable via
`ParallelismConfig.fsdp_reshard_after_forward`) wherever the mode leaves a DP dimension to shard
over. What gets sharded varies:

| Mode | What's sharded | Note |
|------|----------------|------|
| No parallelism / CP | All params | CP attention coordinates the sequence split independently |
| EP / ETP / EP+CP / EP+TP | Non-EP params only | EP modules (router + experts) excluded via `ignored_params`; their gradients sync through the EP layer's own post-accumulation grad hooks (`src/distributed/expert_parallel/grad_sync.py`), or — at `num_ep_groups > 1`, single-node shapes included — through one post-backward sweep instead. **Multi-group EP across nodes** (`is_deferred_dp`) additionally shards non-EP params over the EP group; `ep_size==1` MoE stays on plain world-wide FSDP — see [Multi-Node → deferred cross-replica sync](multi-node.md#deferred-cross-replica-sync) |
| TP (DP > 1) | Per-layer, 2D `(dp, tp)` mesh | DTensor-compatible grad sync across the DP dim |
| TP (DP == 1, pure TP) | None | Nothing to sync across DP |
| Single GPU | None | — |

FSDP2 shards params, gradients, and optimizer states across the DP ranks, so per-rank optimizer-state
memory is ~`dp_size` smaller than DDP's full per-rank replication. Setup lives in
`src/distributed/fsdp.py`; `IdentityParamSet` backs `ignored_params` with `id()`-based
membership, because a plain set's membership test triggers tensor `__eq__` on a hash collision and
raises when comparing EP's fused 3D expert weights against standard 2D weights.

Two setups **raise** rather than wrap:

- Parameters on multiple devices (e.g. `device_map="auto"` under `torchrun`), on the DP and TP paths
  alike. The check cannot skip: it is rank-local while the mesh construction after it is collective,
  so one rank bailing out would hang its peers and train unsharded with no gradient sync at all. Use
  `device_map=None` (or `load_distributed_model()`).
- A generative decoder whose layer list the backbone probe cannot reach (`DECODER_LAYER_LIST_ATTRS` —
  `.layers`, `.h`); the message names the class. The alternative is one root shard group all-gathered
  for the whole forward — the per-rank memory ceiling FSDP2 exists to remove — reported as a
  successful wrap. Models that carry no such list by construction (a SentenceTransformer, a
  BERT-family classification backbone) are still wrapped at the
  root; the fix for a decoder is to add its spelling to `DECODER_LAYER_LIST_ATTRS`
  (`src/models/structure.py`).

## ZeRO-2 vs ZeRO-3 (`reshard_after_forward`)

`fsdp_reshard_after_forward` chooses how FSDP2 holds parameters between forward and backward; both
modes shard gradients and optimizer states across the DP ranks.

- **`false` (default) — ZeRO-2 analog (SHARD_GRAD_OP):** params stay gathered after the forward, so backward skips a re-gather. Faster, higher peak memory.
- **`true` — ZeRO-3 analog (FULL_SHARD):** params reshard after the forward and re-gather in backward. Lower peak memory, one extra all-gather per layer.

Set it with `--fsdp_reshard_after_forward`. **Allowed only where the backward re-gather is a plain
all-gather** — pure DP, CP, TP at `data_parallel_size==1`, and an `ep_group_size==1` MoE (its no-op
EP issues no collectives, so the re-gather races nothing). `ParallelismConfig` rejects it
whenever `is_ep_mode` (`ep_group_size>1`), because the re-gather can race the DeepEP combine (real
EP) or the Expert-TP reduce (pure ETP); **TP with `data_parallel_size>1` is also rejected** (a plain
all-gather on TP-sharded DTensor params has no registered sharding strategy); and under PP (the
schedule pins each stage unsharded). Lower peak memory there with
[HSDP](#hsdp-hybrid-sharded-data-parallel) or activation checkpointing instead.

`fsdp_reshard_after_backward: false` additionally keeps parameters **unsharded across a
gradient-accumulation window's microsteps** (torch `set_reshard_after_backward`). Even under
SHARD_GRAD_OP, FSDP2 reshards each module after its backward and re-all-gathers it on the next
microstep's forward — one full param re-gather per grad-accum microstep for weights that did not
change in between. Over NVLink that traffic is negligible; when NCCL is forced onto sockets
process-global (`rollout_backend: sglang`, whose cross-container weight-sync group requires
`NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1`) it measures ~15s per re-gather at gpt-oss-20b scale —
~6 minutes of every optimizer step at `gradient_accumulation_steps: 24`.

The window's **last** backward still reshards: the trainer arms the flag per microstep from
`accelerator.sync_gradients` in `src/trainers/mixins/base.py`. That leaves one re-gather per
optimizer step instead of one per microstep, so the saving scales with `gradient_accumulation_steps`
and is nil at 1. The last reshard is also mandatory — FSDP2's `post_backward` clears the unsharded
parameters' `.grad` before reduce-scattering onto the sharded DTensors, so a module left unsharded
hands `model.parameters()` grad-less tensors the optimizer never captured (grad norm 0, nothing
clipped) while `unshard()` no-ops on it, hiding the optimizer's update from the next forward.

The cost is one unsharded bf16 param copy per GPU held for the whole run. Plain-DP/CP/EP torchrun
path only; rejected with `fsdp_reshard_after_forward: true` (contradicts FULL_SHARD's purpose), TP,
or PP.

### EP1 expert sharding

At `ep_group_size==1` (`ep_size==1` AND `expert_tp_size==1`) the MoE experts are replicated and the
DeepEP dispatch is a no-op. By default (`fsdp_shard_ep1_experts: true`) FSDP shards them, with its
reduce-scatter as their sole gradient sync — throughput-neutral (the expert all-gather overlaps),
grad-equivalent, and freeing memory that scales with DP (gpt-oss-20b −19% peak at 2 GPU, −37% at
8 GPU). Set it `false` to keep a full replicated copy on every DP rank (EP modules become FSDP
`ignored_params`) — the max-throughput choice when memory is not the bottleneck. No effect when
`ep_group_size>1`. `false` raises at config time under TP, CP, or PP: those setup paths FSDP-shard
ep1 experts unconditionally, so the flag is honored only on the pure-DP path.

The two flags compose into the EP1 sharding matrix (MoE, `ep_group_size==1`):

| `fsdp_shard_ep1_experts` | `fsdp_reshard_after_forward` | experts | non-expert params |
|---|---|---|---|
| `true` (default) | `false` (default) | ZeRO-2 (sharded) | ZeRO-2 |
| `true` | `true` | ZeRO-3 | ZeRO-3 |
| `false` | `false` | replicated | ZeRO-2 |
| `false` | `true` | replicated | ZeRO-3 |

All four cells work for both SFT and online/environmental GRPO. At `ep_group_size==1` the MoE is
still EP-wrapped (grouped-GEMM path) while `is_ep_mode` is `False`, so the GRPO vLLM weight-sync
gather keys the EP expert reshape on the model carrying EP wrappers, not on `is_ep_mode`, and
materializes the FSDP-sharded experts (`materialize_dtensor`) before reshaping to
vLLM's checkpoint layout. Without that the experts reach vLLM in the EP-internal layout, the server
rejects them, and the weight-sync NCCL broadcast hangs.

## HSDP (Hybrid Sharded Data Parallel)

**1D full-shard works on multi-node and is the default — HSDP is a bandwidth optimization, not a
requirement.** The default DP path 1D-full-shards non-expert params across **every** DP rank, so on
a multi-node job every per-layer all-gather and reduce-scatter crosses the inter-node fabric, an
order of magnitude below NVLink
([bandwidth ladder](../reference/gpu-training-theory.md#interconnect-tiers)). HSDP shards within each
NVLink domain and **replicates** across domains, keeping the bandwidth-heavy collectives on NVLink so
only one gradient all-reduce crosses RDMA per step.

Enable with `--use_hsdp`. The layout is derived from topology — no shard-size knob: shard width =
`nvlink_domain_size`, replica count = `num_nvlink_domains`. `setup_fsdp2_for_dp()` applies FSDP2 over
the 2D `(dp_replicate, dp_shard)` mesh built by `create_dp_mesh` (`src/distributed/mesh.py`);
grad-norm sums shard norms over the `dp_shard` sub-group only.

- **Scope:** pure DP and CP only; every other mode rejects `use_hsdp` at config time
  (`ParallelismConfig._validate_hsdp`, `_validate_pipeline_parallel`). TP / EP+TP / Expert-TP build
  their own `(dp, tp)` mesh the 2D HSDP mesh is not wired into; multi-group EP already shards over
  the EP group, and a single global EP group must keep 1D FSDP so its backward collectives share the
  DeepEP combine's membership; under PP the 2D mesh is built by `init_device_mesh` over the whole
  world and cannot be restricted to a stage's rank block, so every rank would silently get the first
  stage's ranks.
- **Trade-off:** one param replica per domain, so it costs memory vs 1D full-shard. Use it when
  inter-node DP bandwidth, not per-GPU memory, is the bottleneck.
- **Single domain:** a no-op; `is_hsdp` stays False and the mesh falls back to 1D.

```bash
# Multi-node pure-DP with HSDP: non-expert params shard within each node
# and replicate across nodes.
torchrun --nnodes=2 --nproc_per_node=8 \
    scripts/training/sft.py config.yaml --use_hsdp=true
```

## Accelerate launch configs

For standard multi-GPU without EP/CP/TP. Only the FSDP v2 configs shard; `multigpu_dp_config.yaml` is plain DDP.

| Config file | Strategy | Use case |
|-------------|----------|----------|
| `fsdp2_gradop_config.yaml` | FSDP2, `reshard_after_forward=false` | **Recommended default** — same FSDP2 as the torchrun path; shards optimizer states + gradients |
| `fsdp2_full_config.yaml` | FSDP2, `reshard_after_forward=true` | Full param sharding for models too large to replicate on one GPU; multi-node |
| `multigpu_dp_config.yaml` | DDP (`MULTI_GPU`) | Standard DDP, full replication, no memory savings. `mixed_precision: 'no'` — the toolkit's `bf16: true` still applies |

```bash
accelerate launch \
    --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
    scripts/training/sft.py \
    examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

The two FSDP v2 configs differ in `fsdp_reshard_after_forward`; the full config also sets
`fsdp_cpu_ram_efficient_loading: true`. They have no hybrid-shard option — for that use `torchrun`
FSDP2 with [`--use_hsdp`](#hsdp-hybrid-sharded-data-parallel).

!!! warning "Avoid accelerate FSDP v1"
    FSDP v1 SHARD_GRAD_OP and FULL_SHARD have a known PyTorch bug that can corrupt model state after
    checkpoint saves during training. The mixin warns whenever accelerate is launched with an FSDP v1
    sharding strategy. Use the FSDP v2 configs or DDP. The `torchrun` path is FSDP2-only and
    unaffected.

## Data parallel size

`(world_size / pp_size) / max(tp_size, cp_size, expert_tp_size)`; EP is orthogonal and does not
reduce it, and a whole pipeline chain consumes one batch. The
per-mode breakdown lives in [Distributed Data Loading](data-loading.md#data-parallel-size); the
support matrix in [Supported combinations](index.md#supported-combinations).

## How the mixin manages FSDP

Under `torchrun`, `DistributedTrainerMixin` (`src/trainers/mixins/base.py`) owns FSDP2:
`_setup_distributed_modes()` dispatches to the per-mode setup after `super().__init__()`. HF's own
FSDP wiring is disabled ahead of it — `apply_distributed_trainer_config`
(`src/training/script_runner.py`) blanks `fsdp` on every distributed trainer's config.
Accelerate's DDP wrapping is skipped whenever the mixin syncs gradients
(`_should_skip_ddp_wrapping()`). `use_grouped_gemm` does not count as custom parallelism; it only
activates for MoE models during loading, and an MoE under `accelerate launch` is rejected at load
([Two launchers](#two-launchers)).

Mixed precision is auto-detected from training args by `create_mixed_precision_policy_v2`
(`src/distributed/fsdp.py`). With `fp32_non_ep_params: true`, non-expert params are
stored fp32 while compute/comm use bf16 — an alternative to `AdamWBF16` when exact fp32 updates beat
stochastic rounding (12 vs 6 B/param). The lighter `fp32_grad_reduce` sets `reduce_dtype=fp32`
**without** moving storage, keeping BF16 masters while summing grads in fp32; this matters as world
size grows.
See [BF16 Optimizer](../optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options).

## Limitations

**Trainers.** All of them — plain DP is the fallback every trainer runs on, and no trainer declares
a DP restriction.

**Models.** All of them, dense and MoE. The one model-shaped rejection is a MoE with
`use_grouped_gemm: true` under `accelerate launch` ([Two launchers](#two-launchers)).

**Axis combinations.** DP is the residual width, not an axis in the
[allowlist](index.md#supported-combinations) — EP/CP/TP/ETP each carve their groups first and
FSDP2 shards over what is left. HSDP is the exception with a scope of its own: pure DP and CP only.

**Knobs.** Everything below raises unless the verdict says otherwise.

| Knob | Under plain DP | Gate |
|---|---|---|
| `use_grouped_gemm: true` + MoE + `accelerate launch` | rejected — the wrappers need the mixin-managed FSDP2 path. Use `torchrun`, or `use_grouped_gemm: false` | `_validate_gmm_launch_method` |
| multi-device `device_map` (e.g. `"auto"`) under `torchrun` | rejected — the FSDP2 setup cannot skip a rank-local bail-out before a collective mesh build | `src/distributed/fsdp.py` |
| `bf16_optimizer: false` on a MoE with `fsdp_shard_ep1_experts: false` | rejected — fused AdamW cannot mix the unsharded plain expert tensors with FSDP2 DTensors. At the default `fsdp_shard_ep1_experts: true` the experts are DTensors too and it is allowed | `mixins/base.py` |
| `use_hsdp`, `fsdp_reshard_after_forward`, `fsdp_reshard_after_backward`, `fp32_grad_reduce` under `accelerate launch` | warned and ignored — accelerate owns the wrap | `_ACCELERATE_UNSUPPORTED_KNOBS` |
| `bf16_optimizer` auto-enable under accelerate DDP | warned and skipped — replicated DDP is outside the validated stochastic-rounding matrix; set it explicitly to override | `mixins/base.py` |
| accelerate FSDP v1 sharding strategy | warned — a known PyTorch bug can corrupt model state after a save. Use the FSDP2 configs or DDP | `mixins/base.py` |
| `use_hsdp` on a single NVLink domain | warned — no-op; the replica axis engages once the job spans domains | `_validate_hsdp` |
| QLoRA / `load_in_4bit` | supported on a **dense** model. On a MoE the grouped-GEMM loader takes over and rejects a quantized base (`use_grouped_gemm` is on by default) — use `use_grouped_gemm: false`, or `accelerate launch` | `model_loading.py` |
| `use_peft` / LoRA, `packing`, `padding_free`, `torch_compile`, `init_from_scratch`, `gradient_checkpointing` | supported and ungated — plain DP is the mode with the widest knob surface | — |
| `lowp_precision != "bf16"` | SFT only | `parallelism_config_from_args` |

## Common issues

- **FSDP + Accelerate config conflict** — wrong FSDP strategy or double-wrapping under `torchrun`,
  caused by HuggingFace FSDP fields (`fsdp`, `fsdp_config`) in the training YAML. Do not set them
  when using `torchrun` with parallelism.
- **Gradient checkpointing with EP or CP** — the mixin forces `use_reentrant=True` before
  `super().__init__()`, even when the config sets `false` (the GRPO templates commonly do); the
  recompute replays the checkpoint frame's DeepEP dispatch instead of issuing a second one.
