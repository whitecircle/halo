# Data Parallelism

PyTorch FSDP2 (`fully_shard`) handles gradient synchronization for `torchrun` multi-GPU training,
applied automatically by `DistributedTrainerMixin` from the active parallelism mode. For training
without EP/CP/TP/ETP/PP, `accelerate launch` with a pre-built config provides FSDP v2 or plain DDP
instead.

The wrap decision keys on the **wrap width** each mode passes: the rank block for plain DP, CP and
EP, the EP group under deferred DP, and the mesh's DP dimension (`data_parallel_size`) only under TP.
A CP run at `data_parallel_size == 1` is still wrapped over the whole rank block; a single GPU and
pure TP skip the wrap. QLoRA skips FSDP2, starts every floating-point parameter from rank 0's
copy, and averages its trainable gradients in one bucketed all-reduce sweep per optimizer step
(`HALO_GRAD_BUCKET_MB`, [Configuration](../reference/configuration-reference.md#environment-variables)).

## Two launchers

`torchrun` uses FSDP2 exclusively and is required for any EP/CP/TP/ETP/PP run — the mixin coordinates
FSDP2 with parallelism-specific gradient hooks ([How the mixin manages FSDP](#how-the-mixin-manages-fsdp)).

`accelerate launch` reads an accelerate config YAML (FSDP or DDP) and is for standard multi-GPU
without EP/CP/TP/ETP/PP; those modes raise under it (`reject_custom_parallelism_under_accelerate`). Both paths produce
equivalent results there. Dense models run under `accelerate launch` (FSDP or DDP) even
at the default `use_grouped_gemm: true`: the grouped-GEMM EP layers only activate for MoE
models.

An **MoE** model with `use_grouped_gemm: true` is rejected at load under any `accelerate launch`:
the EP layers require the mixin-managed FSDP2 path. Launch with `torchrun`, or set
`use_grouped_gemm: false` to train the MoE under accelerate.

## FSDP2 strategy by mode

The mixin applies `fully_shard` (`reshard_after_forward=False` by default, configurable via
`ParallelismConfig.fsdp_reshard_after_forward`) wherever the mode leaves a DP dimension to shard
over. What gets sharded varies:

| Mode | What's sharded | Note |
|------|----------------|------|
| No parallelism / CP | All params | CP attention coordinates the sequence split independently |
| EP / ETP / EP+CP / EP+TP | Non-EP params only | EP modules (router + experts) excluded via `ignored_params`; with one EP group the router syncs through the EP layer's own post-accumulation grad hooks (`halo/distributed/expert_parallel/moe_layer/grad_sync.py`) and the experts through one post-backward sweep, which at `num_ep_groups > 1`, single-node shapes included, takes the router too. **Multi-group EP across nodes** (`is_deferred_dp`) additionally shards non-EP params over the EP group; `ep_size==1` MoE stays on plain world-wide FSDP — see [Multi-Node → deferred cross-replica sync](multi-node.md#deferred-cross-replica-sync) |
| TP (DP > 1) | Per [shard group](#shard-groups), 2D `(dp, tp)` mesh | DTensor-compatible grad sync across the DP dim |
| TP (DP == 1, pure TP) | None | Nothing to sync across DP |
| PP (alone, or with EP / pure ETP) | The stage's params, over its rank block | As the rows above, scoped to the stage; the schedule reduces once per optimizer step ([PP + EP](pipeline-parallelism.md#pp--ep-moe)) |
| Single GPU | None | — |

Plain FSDP2 and DDP load the whole model onto every GPU before the wrap, whatever
`fsdp_reshard_after_forward` says, so a dense model must fit on one GPU at its load dtype. Past one GPU,
TP's HF-native `tp_plan` load shards it as it loads ([Tensor Parallelism](tensor-parallelism.md)), and
across NVLink domains PP's stage loader reads only each stage's layers
([Pipeline Parallelism → Loading](pipeline-parallelism.md#loading)).

The plain-DP, CP, EP and EP+TP wraps also leave out the frozen parameters whose dtype no trainable
parameter shares (a base under differently typed adapters), one parameter at a time, so a trainable child
of their module stays in its shard group (`_fsdp_excluded_params` in `halo/trainers/parallelism/mode_setup.py`). A
trainable parameter left out of every group has no reduce-scatter, and only an EP layer can leave one out;
unless its hooks or the deferred sweep average it, trainer construction raises on every rank rather than
let it drift across DP ranks (`reject_unsynced_trainable_params`).

FSDP2 asserts one original dtype per shard group at the first forward. A decoder layer whose trainable
parameters mix dtypes (the fp32 router `fp32_router` keeps at `ep_group_size == 1` under
`fsdp_shard_ep1_experts`, or its `modules_to_save` copy beside bf16 adapters) first shards the smallest
submodule holding each set of the other dtype as a nested group. The layer's forward pre-hook unshards
those groups too: the LFM2, DeepSeek-V4 and Inkling EP layers, and GPT-OSS under `bias_update`, read the
router's weight without calling the router, and FSDP2's root post-backward callback then reduces its
gradient. A module that itself owns trainable parameters of two dtypes cannot be split, and the wrap
raises. A layer whose trainable parameters share one dtype gets no extra group.

On gpt-oss-20b at `ep_size: 1` (2× B300, seq 4096, batch 1, Halo 1.1.0 or earlier) the fp32 router runs 10,647 tok/s/GPU against
10,883 with a bf16 router, at the same 104 GiB peak. In that run set, the fp32 router over replicated
experts (`fsdp_shard_ep1_experts: false`, which keeps the router out of FSDP2) ran 8,770–9,060 at 153 GiB.

FSDP2 shards params, gradients, and optimizer states across the DP ranks, so per-rank optimizer-state
memory is ~`dp_size` smaller than DDP's full per-rank replication. Setup lives in
`halo/distributed/fsdp.py`; `IdentityParamSet` backs `ignored_params` with `id()`-based
membership, because a plain set's membership test triggers tensor `__eq__` on a hash collision and
raises when comparing EP's fused 3D expert weights against standard 2D weights.

### Shard groups

Under ZeRO-2 (the default) each shard group is one unit of two consecutive decoder layers (`LAYERS_PER_UNIT`,
`halo/distributed/fsdp.py`), beside one group for the module that embeds the input ids and one for the root. A
unit all-gathers its parameters in one collective and reduce-scatters its gradients in one, so it issues half the
collectives, copy launches and FSDP2 hook calls of one-layer groups, the host work every group adds to a
host-bound step. A unit ends early where the layers run out (a pipeline stage's odd count included) or where the
next layer's trainable dtype, outside its nested groups, differs from the unit's. FSDP2 shards each parameter on
its own, so state-dict keys, shapes and placements, the saved weights and optimizer shards, and the weight sync
are the same at any unit size, and a run resumes across unit sizes, a switch between ZeRO-2 and ZeRO-3 included.
Against one-layer groups on 8× B300 at s4096 (1.2.0, three rounds): gpt-oss-20b ep1 **+11.3%**, Qwen3-8B CP2 +8.2%,
gpt-oss-20b ep4 × etp2 +3.2%, Qwen3-8B TP2 +2.1%, gpt-oss-20b ep2 +1.4%, Qwen3-8B FSDP2 +0.8%, gpt-oss-20b ep8 −0.5%.

ZeRO-2 keeps every layer's parameters gathered from its forward to its backward at any unit size, so a unit adds
gradient memory: until its first layer's backward ends it holds its later layer's unreduced gradients and
gathered parameters, which a one-layer group would have reduced and freed, and its reduce-scatter buffer, in the
reduce dtype, spans both layers, which adds 0.36 GiB on Qwen3-8B FSDP2 and 1.54 GiB on gpt-oss-20b ep1 at s4096, about
one decoder layer of bf16 weights. ZeRO-3 (`fsdp_reshard_after_forward: true`) keeps one decoder layer per group:
FSDP2 holds the running group and the one its prefetch gathers, so units of two would gather four layers at the
backward's peak instead of two, the memory ZeRO-3 is chosen to save.

torch starts a multi-module group at whichever module runs first, and a per-layer reentrant checkpoint
recomputes a unit's last layer first: the unit would reduce and reshard before its other layers ran their
backward. The wrap replaces torch's hooks with its own at the unit's two ends (`_install_unit_hooks`,
`halo/distributed/fsdp.py`), under which a unit reduces from its post-backward hook inside the backward as a
one-layer group does, with checkpointing off, non-reentrant or reentrant, on every layer or every
`every_n_layers`, and each layer casts its inputs as its own group would. `tests/cpu/parallelism/test_fsdp2_layer_units.py`
holds unit sizes 2 and 3 bit for bit to one-layer groups across those shapes, several forwards per backward,
both ZeRO modes, LoRA, ep1 EP layers, a pipeline stage, HSDP and TP, and pins the FSDP2 members the hooks drive. One
combination is equal only to rounding: with some layers checkpointed and several forwards per backward, a unit
reduces each forward's gradients inside the backward where a one-layer group folds the later ones into one
reduce after it, the same sum added in another order.

`fully_shard` unshards a group from a forward pre-hook, so `embed_tokens` has to sit in a group whose
`__call__` runs before the lookup: on a multimodal wrapper that is the composite `model.model`, which
builds `inputs_embeds` itself before handing them to the decoder backbone below it
(`input_embedding_backbone`, `halo/models/module_tree.py`). A composite with no `.model` that holds
`language_model` itself (Llama 4) embeds at its root, so the group holding the embedding is the wrap's root,
the module whose `__call__` a wrapper above the composite cannot bypass. The rule is ancestry — `model.model` is
taken whenever the layer backbone sits under it, however the text stack is spelled — and the probe
reads the plain transformers tree under every wrapper: a `PeftModel`'s attribute forwarding answers
with the head module, whose `__call__` the adapter's tuner bypasses, and the CP wrapper's `.model` is
the head too. A tied embedding is reserved for the root group and does not depend on any of this; a
classification head (no `lm_head`), a `modules_to_save` copy or a second embedding table (Gemma 4's
per-layer inputs) is untied and does.

On a composite the embedding and root groups also hold what only an image row reaches, so a rank whose batch has no image would enter their reduce-scatter with a shorter list than its peers'. Before reducing, those groups agree across their process groups which parameters any rank holds a gradient for and zero-fill the rest (`_agree_unsharded_grad_presence`); what no rank reached keeps `grad is None`, so a text-only run never writes its tower. That costs one small all-reduce per group and backward, two under HSDP. Mixed image and text rows therefore train under `torchrun` without `freeze_vision_tower`.

Two setups cannot pair those reduces, so a rank's training forward holding no image raises at a data-parallel size above 1, where the step would otherwise hang to the collective timeout (`_refuse_a_batch_holding_no_image`, `halo/entrypoints/data_setup.py`):

- Accelerate-managed FSDP2, which takes no such wrap. Launch with `torchrun`.
- A frozen input embedding beside a trainable tower with gradient checkpointing off, once a decoder layer trains too or under ZeRO-3 (`unfreeze_layers_patterns` matching the tower but not the embedding). A text row's first trained layer gets no gradient-carrying input, so that layer's group reduces at the end of the backward on a text rank and in order on an image rank, and under ZeRO-3 the tower's group re-gathers at its backward on an image rank alone. Checkpointing's input-grad hook gives the embedding's output a gradient on every row, so the same setup pairs with checkpointing on. Train the embedding or turn checkpointing on.

Both also take `freeze_vision_tower: true` or a dataset with no text-only rows. A run that holds an image on every rank pairs in either setup. At ZeRO-2 with no layer training, only end-of-backward groups reduce, so a tower beside a trained head pairs. A tower training alone leaves a text row's loss with no gradient, and autograd raises at that row's backward.

An adapter run takes the same refusal: its embedding is frozen unless `modules_to_save` trains it, and `run_trainer` carries the hook onto the `PeftModel` the loop calls (`carry_image_refusal_above_adapters`), whose `.forward()` runs no pre-hook below it. The message names the parameter the adapter reached. Keep `lora_target_modules` off the tower, as `freeze_vision_tower` is refused there, or take a remedy above.

With checkpointing on those adapters' reduces pair, and a tower block takes the same gradient under reentrant checkpointing as under non-reentrant ([Tower blocks under reentrant checkpointing](#tower-blocks-under-reentrant-checkpointing)).

Two setups **raise** rather than wrap:

- Parameters on multiple devices (e.g. `device_map="auto"` under `torchrun`), on the DP and TP paths
  alike. Use `device_map=None` (or `load_distributed_model()`).

    The check cannot skip: it is rank-local while the mesh construction after it is collective, so
    one rank bailing out would hang its peers and train unsharded with no gradient sync at all.

- A generative decoder whose layer list the backbone probe cannot reach (`DECODER_LAYER_LIST_ATTRS` —
  `.layers`, `.h`, `.layer`); the message names the class. The fix for a decoder is to add its spelling to
  `DECODER_LAYER_LIST_ATTRS` (`halo/models/module_tree.py`).

    The alternative is one root shard group all-gathered for the whole forward, the per-rank memory
    ceiling FSDP2 exists to remove, reported as a successful wrap. Models that carry no such list by
    construction (a SentenceTransformer, a BERT-family classification backbone) are still wrapped at
    the root.

## ZeRO-2 vs ZeRO-3 (`reshard_after_forward`)

`fsdp_reshard_after_forward` chooses how FSDP2 holds parameters between forward and backward; both
modes shard gradients and optimizer states across the DP ranks.

- **`false` (default) — ZeRO-2 analog (SHARD_GRAD_OP):** params stay gathered after the forward, so backward skips a re-gather. Faster, higher peak memory.
- **`true` — ZeRO-3 analog (FULL_SHARD):** params reshard after the forward and re-gather in backward. Lower peak memory, one extra all-gather per layer, each decoder layer its own [shard group](#shard-groups).

Set it with `--fsdp_reshard_after_forward`. **Allowed only where the backward re-gather is a plain
all-gather**: pure DP, CP, TP at `data_parallel_size==1`, and an `ep_group_size==1` MoE (its no-op
EP issues no collectives, so the re-gather races nothing).

`ParallelismConfig` rejects it whenever `has_ep_group` (`ep_group_size>1`), because the re-gather can
race the DeepEP combine (real EP) or the Expert-TP reduce (pure ETP). **TP with
`data_parallel_size>1` is also rejected** (a plain all-gather on TP-sharded DTensor params has no
registered sharding strategy). Lower peak
memory there with [HSDP](#hsdp-hybrid-sharded-data-parallel) or activation checkpointing instead.

`fsdp_reshard_after_backward: false` additionally keeps parameters **unsharded across a
gradient-accumulation window's microsteps** (torch `set_reshard_after_backward`). Even under
SHARD_GRAD_OP, FSDP2 reshards each module after its backward and re-all-gathers it on the next
microstep's forward: one full param re-gather per grad-accum microstep for weights that did not
change in between.

Over NVLink, turning the re-gathers off gains about 4–10% throughput (Halo 1.1.0 or earlier): on 8× B300, dense Qwen3-8B SFT
runs +3.7% (packed `sft.py`, batch 2, `gradient_accumulation_steps: 8`) to +9.7% (4k tokens, batch 1,
GA 8) faster with `false`, and gpt-oss-20b EP8 +3.6%. Under the default ZeRO-2 wrap
(`reshard_after_forward=False`) the parameters are already unsharded through each forward and backward,
so the measured peak is unchanged. With the trainer's NCCL on TCP sockets (the no-fabric compose recipe,
`NCCL_NET=Socket`) it measures ~15 s per re-gather at gpt-oss-20b scale, ~6 minutes of every optimizer
step at `gradient_accumulation_steps: 24`.

The window's **last** backward still reshards: the trainer arms the flag per microstep from
`accelerator.sync_gradients` (`_set_window_end`, `halo/trainers/parallelism/grad_sync.py`). That leaves one
re-gather per optimizer step instead of one per microstep, so the saving scales with
`gradient_accumulation_steps` and is nil at 1.

The last reshard is also mandatory. FSDP2's `post_backward` clears the unsharded parameters' `.grad`
before reduce-scattering onto the sharded DTensors, so a module left unsharded
hands `model.parameters()` grad-less tensors the optimizer never captured (grad norm 0, nothing
clipped) while `unshard()` no-ops on it, hiding the optimizer's update from the next forward.

The cost is one unsharded param copy per GPU (at the FSDP param dtype), held between a window's
microsteps and freed at its last backward; under ZeRO-2 each forward/backward already holds it, so
peak memory is unchanged. Plain-DP/CP/EP torchrun path only; rejected with
`fsdp_reshard_after_forward: true` (contradicts FULL_SHARD's purpose), TP or PP (whose schedule already
pins each stage unsharded).

### Deferred gradient reduce (`fsdp_defer_grad_sync`)

FSDP2 reduce-scatters every module's gradients after each microstep's backward (HSDP adds the
cross-domain all-reduce), so a window pays `gradient_accumulation_steps` full-gradient reductions.
`fsdp_defer_grad_sync: true` turns them off for microsteps 1..n-1 (torch
`set_requires_gradient_sync`, toggled from `accelerator.sync_gradients` alongside the backward
reshard): autograd accumulates the unsharded gradients locally and the window's last backward
reduces them once. The sum is the same, taken in a different order (microsteps before ranks), so
bf16 runs match the default to rounding: across the DP, HSDP, TP+DP, CP+DP and MoE (ep1, EP,
multi-group EP) rows of `tests/gpu/trainers/sft/test_sft_fsdp_defer_grad_sync.py`, per-step loss
stays within about 4e-4 of the default and the final weights within about 2% of their movement,
under the 1.2e-3 loss spread between DP and HSDP (both correct) on the same data.

Measured on Halo 1.1.0 or earlier with packed `sft.py` (4k tokens, batch 2, gradient checkpointing, B300; tok/s/GPU,
peak allocated in GiB). The two GA 4 rows on one 8-GPU node are single runs from a separate run set, whose tok/s the trainer logs to
four significant figures; the other rows are the mean of 3 runs unless noted:

| Run | GA | default | `fsdp_defer_grad_sync` | + `fsdp_reshard_after_backward: false` | peak allocated |
|---|---|---|---|---|---|
| Qwen3-8B, 8 GPUs, NVLink | 4 | 15,670 | 16,460 (+5.0%) | 17,130 (+9.3%) | 32.6 → 45.9 |
| Qwen3-8B, 8 GPUs, NVLink | 8 | 16,458 | 16,985 (+3.2%) | 17,673 (+7.4%) | 32.6 → 46.0 |
| Qwen3-8B, 2 nodes × 2 GPUs, EFA | 4 | 14,794 | 16,107 (+8.9%) | — | 40.2 → 51.7 |
| same, `use_hsdp` (1 run) | 4 | 14,945 | 16,910 (+13.1%) | — | 55.5 → 63.1 |
| same, NCCL on TCP sockets (1 run) | 4 | 2,024 | 3,084 (+52%) | — | 40.2 → 51.7 |
| Qwen3-30B-A3B, `ep_size: 1`, 8 GPUs, NVLink | 4 | 12,100 | 13,600 (+12.4%) | — | 95.3 → 145.0 |

Over NVLink the dense reduce-scatters mostly overlap the backward, so the gain is a few percent; it
grows with the gradient bytes per token (an `ep_size==1` MoE reduces every expert for 3B active
params) and with the fabric's cost, and it pairs with `fsdp_reshard_after_backward: false`, which
removes the matching per-microstep all-gathers. The three-run spread stays within 2.4% of the mean
(1.3% on one node).

The cost is one unsharded gradient copy per GPU held across the window, at the reduce dtype: 2
B/param, or 4 B/param when the reduce runs in fp32 (`fp32_grad_reduce`, `fp32_non_ep_params` or
`bf16: false`; FSDP2 upcasts the accumulator). The default holds the sharded gradient anyway, so at
equal dtypes the peak rises by (1 − 1/shard width) of that copy: the table's peaks grow by 7/8 of the
bf16 gradient on 8 GPUs, 3/4 on 2 × 2, and 1/2 under `use_hsdp`, whose shard width is the 2-GPU node.
An `ep_size==1` MoE pays it for every expert, since FSDP2 owns them; at `ep_size>1` the experts are
FSDP-ignored and only the non-expert params pay it. It saves nothing at
`gradient_accumulation_steps: 1`. Use it when the reduction is a visible share of the step (large DP
width, slow inter-node fabric, small microbatches) and the memory is there.

The in-backward EP router hooks gate on the same `sync_gradients`, and the deferred EP sweep, the TP
replicated-gradient sweep and the gradient clip run after the window's last backward, so all of them
read the same reduced gradients as with the flag off. Rejected under PP (the schedule
already reduces once per step), TP at `data_parallel_size==1` (no FSDP2 wrap to defer), QLoRA (no
wrap; its sweep already runs once per step) and `fsdp_reshard_after_forward: true` (the held unsharded
gradient is the state ZeRO-3 exists to shard; defer under ZeRO-2 instead). Under `accelerate launch`
it is warned and ignored: accelerate's `no_sync` already skips the reduce on non-final microsteps.

### EP1 expert sharding

At `ep_group_size==1` (`ep_size==1` AND `expert_tp_size==1`) the MoE experts are replicated and the
DeepEP dispatch is a no-op. By default (`fsdp_shard_ep1_experts: true`) FSDP shards them, with its
reduce-scatter as their sole gradient sync: grad-equivalent, freeing memory that scales with DP, and
faster than replicated experts on gpt-oss-20b at batch 1 and 4
([Throughput Benchmarks](../optimization/throughput-benchmarks.md#ep-only-batch-scaling)).

Set it `false` to keep a full replicated copy on every DP rank (EP modules become FSDP
`ignored_params`) — what `fp32_non_ep_params` and `fp32_experts` need at ep1 (the first raises on a
MoE at `true`, the second is skipped with a warning). No effect when `ep_group_size>1`. The wrap
broadcasts those copies, the router and native expert LoRA adapters included, from DP rank 0, so
adapters each rank drew from its own RNG stream start identical.

`false` raises at config time under TP or CP, whose setup paths FSDP-shard ep1 experts
unconditionally. The flag is honored only on the pure-DP path.

The two flags compose into the EP1 sharding matrix (MoE, `ep_group_size==1`):

| `fsdp_shard_ep1_experts` | `fsdp_reshard_after_forward` | experts | non-expert params |
|---|---|---|---|
| `true` (default) | `false` (default) | ZeRO-2 (sharded) | ZeRO-2 |
| `true` | `true` | ZeRO-3 | ZeRO-3 |
| `false` | `false` | replicated | ZeRO-2 |
| `false` | `true` | replicated | ZeRO-3 |

All four cells are admitted for SFT and online/async GRPO; GPU rows run the first three. At `ep_group_size==1` the MoE is
still EP-wrapped (grouped-GEMM path) while `has_ep_group` is `False`, so the GRPO weight-sync gather
keys the EP expert reshape on the model's live EP layers, not on `has_ep_group`.

The layer's `gather_expert_state_dict` materializes the FSDP-sharded experts (`materialize_dtensor`)
before reshaping them to the hub checkpoint layout the rollout engine loads. Without that the experts
reach the server in the EP-internal layout, the server rejects them, and the weight-sync NCCL
broadcast hangs.

## HSDP (Hybrid Sharded Data Parallel)

**1D full-shard works on multi-node and is the default — HSDP is a bandwidth optimization, not a
requirement.** The default DP path 1D-full-shards non-expert params across **every** DP rank, so on
a multi-node job every shard group's all-gather and reduce-scatter crosses the inter-node fabric, an
order of magnitude below NVLink
([bandwidth ladder](../reference/gpu-training-theory.md#interconnect-tiers)).

HSDP shards within each NVLink domain and **replicates** across domains, keeping the bandwidth-heavy
collectives on NVLink so only one gradient all-reduce crosses RDMA per backward — per optimizer step
with [`fsdp_defer_grad_sync`](#deferred-gradient-reduce-fsdp_defer_grad_sync).

Enable with `--use_hsdp`. The layout is derived from topology — no shard-size knob: shard width =
`nvlink_domain_size`, replica count = `num_nvlink_domains`. `apply_fsdp2_for_dp()` applies FSDP2 over
the 2D `(dp_replicate, dp_shard)` mesh built by `build_dp_mesh` (`halo/distributed/mesh.py`);
grad-norm sums shard norms over the `dp_shard` sub-group only.

The wrap then gives every replica the first replica's shard of every parameter in FSDP2's shard
groups, frozen ones included, and DP rank 0's copy of every FSDP-ignored parameter, each a whole copy
on every rank since HSDP runs no expert distribution. FSDP2, unlike DDP, broadcasts no module state,
and the replicas share only their gradient average, so anything a rank drew from its own RNG stream
(LoRA adapters, a resized vocabulary's new rows, a fresh head) would otherwise stay different on each
replica for the whole run while every loss looks normal.

- **Scope:** pure DP and CP only; every other mode rejects `use_hsdp` at config time
  (`parallelism_rules._reject_hsdp`).

    TP / EP+TP / Expert-TP build their own `(dp, tp)` mesh the 2D HSDP mesh is not wired into.
    Multi-group EP already shards over the EP group, and a single global EP group must keep 1D FSDP
    so its backward collectives share the DeepEP combine's membership.

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

For standard multi-GPU without EP/CP/TP/ETP/PP. Only the FSDP v2 configs shard; `multigpu_dp_config.yaml` is plain DDP.

| Config file | Strategy | Use case |
|-------------|----------|----------|
| `fsdp2_gradop_config.yaml` | FSDP2, `reshard_after_forward=false` | **Recommended default** — same FSDP2 as the torchrun path; shards optimizer states + gradients |
| `fsdp2_full_config.yaml` | FSDP2, `reshard_after_forward=true` | Full param sharding (ZeRO-3) for lower steady-state memory, multi-node included; the model must still fit on one GPU at load ([above](#fsdp2-strategy-by-mode)) |
| `multigpu_dp_config.yaml` | DDP (`MULTI_GPU`) | Standard DDP, full replication, no memory savings. `mixed_precision: 'no'` — the toolkit's `bf16: true` still applies |

```bash
accelerate launch \
    --config_file launcher-configs/accelerate/fsdp2_gradop_config.yaml \
    scripts/training/sft.py \
    examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

The two FSDP v2 configs differ in `fsdp_reshard_after_forward`; the full config also sets
`fsdp_cpu_ram_efficient_loading: true`. With it, `from_pretrained` reads the checkpoint on every rank and then
zero-fills the weights off each node's local rank 0, leaving them to the broadcast at `accelerator.prepare`, which a
frozen reference or teacher never gets and which the policy gets only once training starts. The toolkit's loads
therefore keep the full weights on every rank (`full_weights_on_every_rank` in
`halo/model_loading/checkpoint_coverage.py`), as under the gradop config, at the same host memory.

Neither config has a hybrid-shard option; for that use `torchrun` FSDP2 with
[`--use_hsdp`](#hsdp-hybrid-sharded-data-parallel).

> [!WARNING]
> **Avoid accelerate FSDP v1**
>
> FSDP v1 SHARD_GRAD_OP and FULL_SHARD have a known PyTorch bug that can corrupt model state after
> checkpoint saves during training. The mixin warns when accelerate is launched with FSDP v1 at
> either strategy. Use the FSDP v2 configs or DDP. The `torchrun` path is FSDP2-only and
> unaffected.

## Data parallel size

Derived once in [Stacking the dimensions](README.md#stacking-the-dimensions), per mode in
[Supported combinations](README.md#supported-combinations).

## How the mixin manages FSDP

Under `torchrun`, `DistributedTrainerMixin` (`halo/trainers/distributed_trainer.py`) owns FSDP2:
`_configure_distributed_modes()` runs the mode ladder (`ModeSetupMixin._wrap_for_parallel_mode`) after
`super().__init__()`. HF's own
FSDP wiring is disabled ahead of it: `_init_distributed_config` blanks `fsdp` on every distributed
trainer's config.
Accelerate's DDP wrapping is skipped whenever the mixin syncs gradients
(`AcceleratorMixin._inner_training_loop`).

Anything that reads parameters outside a forward — the optimizer build (`create_optimizer`), every
checkpoint writer, `save_model`, RL weight sync — calls `reshard_fsdp2_modules` (`halo/distributed/fsdp.py`)
first. A forward leaves the transient unsharded params registered (`reshard_after_forward=False`; the root
never reshards after its forward), so a read would see objects the optimizer never steps: param groups
built over them update orphans, a writer saves state one step stale, and the PEFT save's DTensor probe
mis-routes.

### Tower blocks under reentrant checkpointing

A reentrant checkpoint runs its first pass under `no_grad`, and its outputs carry a gradient only if a positional input does. transformers' input-grad hook reaches a tower only through a declared `get_input_embeddings`: SigLIP, SigLIP 2, Pixtral and Llama 4's vision model declare one, and Qwen3-VL, Qwen3.5/3.6, Qwen3.8-Flash-Next, GLM-5 Next, MiniMax-M3-VL, Step-3.7, Gemma 4 and Nemotron-H do not. Inkling's tower has no checkpointed layer, and Llama 4's blocks take their hidden state by keyword, which a checkpoint's graph does not see. So a block holding an adapter or a trained parameter behind a frozen patch embedding would stay at no gradient, on a MoE above all, where reentrant is forced.

`GradientCheckpointingLayer.__call__` closes this for every layer it runs, tower or not (`_with_gradient_carrying_input`, `halo/models/gradient_checkpointing.py`). Under a reentrant checkpoint with no gradient-carrying positional input, a keyword hidden state moves to its positional slot, and a layer holding a trainable parameter, or taking a gradient in another argument, takes a detached alias of its first floating input. That covers adapters, a fine-tune of some blocks and a frozen block fed by a trained module beside its hidden state, and each later block already has its input. A layer that holds no float for this raises at its forward.

A block with nothing trainable and no gradient arriving takes no new input, and neither does any layer under non-reentrant checkpointing or with checkpointing off, so a frozen tower or a run whose adapters stay off it steps bit for bit as under transformers' own call. torch's `None of the inputs have requires_grad=True` warning fires on those blocks and names no fault. A frozen tower of Llama 4's shape recomputes its blocks, as its hook-given input reaches them, at the same loss and gradients.

The tower's gradient on an image rank then reaches the reduce through the [agreement](#shard-groups), which is why these runs pair under `torchrun` and why `accelerate launch`, with no agreement, is refused with checkpointing on too. `tests/cpu/parallelism/test_reentrant_tower_gradient.py` holds a reentrant step to a non-reentrant one on each composite, in one process and on gloo ranks across ZeRO-2 and ZeRO-3, HSDP, accumulation windows, a TP text stack, and expert layers with fused and split experts.

### Layers recomputed more than once per backward

Reentrant gradient checkpointing, forced for EP, CP and every MoE
([rule](../optimization/flash-attention.md#usage)), wraps each decoder layer's `__call__`, so FSDP2's
hooks run inside the checkpoint. A loss that runs a layer more than once before one backward recomputes
it once per run: env-GRPO scores each turn row in its own forward, and the GRPO log-prob scorers run one
backbone call per row or chunk. torch's FSDP2 hooks only the first recompute the backward reaches. The
others find the layer's state mid-backward, re-gather its parameters and accumulate into their
unsharded gradients, and the root's final callback skips a one-layer group as already reduced. Left alone,
those contributions never reach the sharded gradients the clip and the optimizer read, the group stays
gathered past the optimizer step (the next forward reads pre-step weights), and its sharded gradient
survives HF's `model.zero_grad()` into the next step. A ZeRO-2 [unit](#shard-groups) of two layers sits
in the callback's state list once per layer, so its later visit finds the group idle and reduces the
leftovers itself; the one-layer groups left are every decoder-layer group under ZeRO-3, a ZeRO-2 unit cut
short to one layer (at the end of an odd layer count, or where the next layer's trainable dtype differs), and
the nested dtype groups.

After every microstep's backward, `training_step` runs `complete_fsdp2_post_backward`
(`halo/distributed/fsdp.py`) over every shard group: FSDP2's own `post_backward` on each group left
gathered or holding an unsharded gradient, then the compute stream waits on the reductions it issued,
as the root's final callback does. It reads each group's window flags: a `fsdp_defer_grad_sync`
microstep folds the leftovers into the deferred accumulator with no collective, and a
`fsdp_reshard_after_backward: false` microstep keeps the group gathered. A layer run once per
backward leaves nothing to complete, and the step is bit-identical. The cost is one extra
reduce-scatter per recomputed layer group per backward, or per window under `fsdp_defer_grad_sync`,
issued after the backward with no compute left to overlap it. Memory inside the backward is torch's:
from its second recompute until the backward returns, each recomputed group holds its gathered
parameters and an unsharded gradient, under ZeRO-3 too. `reject_unfinished_fsdp2_backward` then raises
if a group whose reduce is armed still holds an unsharded gradient, or one whose reshard is armed is
still gathered. Which groups reduce is read off each rank's own gradients, so every rank must run each
layer the same number of times per backward (env-GRPO pads its row count across ranks).

The completion reads FSDP2 internals (`FSDPParamGroup`); `tests/cpu/parallelism/test_fsdp2_post_backward_internals.py`
pins every member it touches, and the defect, against the installed torch. Non-reentrant checkpointing,
and a checkpoint inside the module's `forward` (FSDP2's hooks outside it), are unaffected.

`use_grouped_gemm` does not count as custom parallelism; it only activates for MoE models during
loading, and an MoE under `accelerate launch` is rejected at load ([Two launchers](#two-launchers)).

The trainer resolves the compute dtype from the training args (`bf16` / `fp16`) and hands it, with
`fp32_grad_reduce`, to the FSDP2 entry points, which build the policy (`mixed_precision_policy`,
`halo/distributed/fsdp.py`). With `fp32_non_ep_params: true`, non-expert params are
stored fp32 while compute and the param all-gather run in bf16 and gradients reduce in fp32 — an
alternative to `AdamWBF16` when exact fp32 updates beat stochastic rounding (12 vs 6 B/param). The
lighter `fp32_grad_reduce` sets `reduce_dtype=fp32` **without** moving storage, keeping BF16 masters
while summing grads in fp32; this matters as world size grows.
See [BF16 Optimizer](../optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options).

## Limitations

**Trainers.** All of them — plain DP is the fallback every trainer runs on, and no trainer declares
a DP restriction.

**Models.** All of them, dense and MoE. The MoE-only rejections are the table rows below that name a
MoE.

**Axis combinations.** DP is the batch count, not an axis in the
[allowlist](README.md#supported-combinations); FSDP2's shard group per mode is in
[Stacking the dimensions](README.md#stacking-the-dimensions). HSDP is the exception with a scope of its own: pure DP and CP only.

**Knobs.** Everything below raises unless the verdict says otherwise.

| Knob | Under plain DP | Gate |
|---|---|---|
| `use_grouped_gemm: true` + MoE + `accelerate launch` | rejected — the EP layers need the mixin-managed FSDP2 path. Use `torchrun`, or `use_grouped_gemm: false` | `reject_unsupported_model` |
| multi-device `device_map` (e.g. `"auto"`) under `torchrun` | rejected — the FSDP2 setup cannot skip a rank-local bail-out before a collective mesh build | `halo/distributed/fsdp.py` |
| `bf16_optimizer: false` with a stock AdamW `optim` on a MoE with `fsdp_shard_ep1_experts: false` | rejected — fused AdamW cannot mix the unsharded plain expert tensors with FSDP2 DTensors. At the default `fsdp_shard_ep1_experts: true` the experts are DTensors too and it is allowed | `reject_stock_adamw_on_mixed_params` |
| `fp32_non_ep_params` on a MoE at the default `fsdp_shard_ep1_experts: true` | rejected — the FSDP-managed experts stay bf16 beside fp32 dense params in one shard group, which FSDP2 refuses. Set `fsdp_shard_ep1_experts: false` | `reject_unsupported_model` |
| `fsdp_defer_grad_sync: true` or `fsdp_reshard_after_backward: false` with `fsdp_reshard_after_forward: true` | rejected — each holds unsharded state across the window ZeRO-3 exists to shard; use them under ZeRO-2 | `_reject_fsdp_settings` |
| `use_hsdp`, `fsdp_reshard_after_forward`, `fsdp_reshard_after_backward`, `fsdp_defer_grad_sync`, `fp32_grad_reduce` under `accelerate launch` | warned and ignored — accelerate owns the wrap | `AcceleratorMixin._warn_knobs_unapplied_under_accelerate` |
| `bf16_optimizer` auto-enable under accelerate DDP | warned and skipped — replicated DDP is outside the validated stochastic-rounding matrix; set it explicitly to override | `resolve_bf16_optimizer` |
| accelerate FSDP v1 at `SHARD_GRAD_OP` / `FULL_SHARD` | warned — a known PyTorch bug can corrupt model state after a save. Use the FSDP2 configs or DDP | `halo/trainers/parallelism/mode_setup.py` |
| `use_hsdp` on a single NVLink domain | warned — no-op; the replica axis engages once the job spans domains | `_reject_hsdp` |
| QLoRA / `load_in_4bit` | supported on a **dense** model. On a MoE the grouped-GEMM loader takes over and rejects a quantized base (`use_grouped_gemm` is on by default) — set `use_grouped_gemm: false`, under either launcher. QLoRA skips FSDP2, so `use_hsdp`, `fsdp_reshard_after_forward`, `fsdp_reshard_after_backward` and `fsdp_defer_grad_sync` raise under it | `model_loading/parallel/load_rules.py`, `reject_fsdp_knobs_under_qlora` |
| `use_peft` / LoRA, `packing`, `padding_free`, `torch_compile`, `gradient_checkpointing` | supported and ungated — plain DP is the mode with the widest knob surface | — |
| `lowp_precision != "bf16"` | SFT only | `parallelism_config_from_args` |
| `init_from_scratch` | SFT only; refuses QLoRA and every EP/CP/TP/ETP/PP mode, so plain DP is the one mode it runs in | `parallelism_config_from_args`, `model_loading/parallel/load_rules.py` |

## Common issues

- **HuggingFace FSDP fields in the training YAML** (`fsdp`, `fsdp_config`) do nothing for a
  multi-GPU `torchrun` run: the training scripts blank `fsdp`, and the mixin's own accelerator
  carries no HF FSDP plugin. Leave them out and use the knobs on this page.
- **Gradient checkpointing with EP, CP or a MoE** goes reentrant even when the config sets
  `use_reentrant: false` (several example configs do); the rule and its PP exception:
  [Flash Attention](../optimization/flash-attention.md#usage).
