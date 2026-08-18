# Tensor Parallelism (TP)

TP shards large weight matrices across GPUs. Use it when a dense model's layers don't fit on one
GPU. `data_parallel_size = world_size / tp_size`; all ranks participate in gradient sync
(`ParallelismConfig`, `src/distributed/parallelism_config.py`). Why its all-reduce has to stay on
NVLink: [GPU Training Theory §9](../reference/gpu-training-theory.md#collectives).

!!! warning "TP buys little memory on a fine-grained MoE"
    On MoE models TP shards **attention only**. The experts stay replicated (EP owns them), and
    `embed_tokens`/`lm_head` stay replicated deliberately. On a fine-grained MoE those are nearly all
    the parameters, so raising `tp_size` costs collectives and saves little — and every TP rank still
    holds the full expert set, so a full fine-tune of a 35B-A3B under `tp_size=2` needs ≈280 GB per rank
    (weights + AdamW-bf16 states + grads), over one B300; that shape runs under EP. The useful MoE shapes
    are **EP**, **EP+ETP** (expert memory), and **EP+TP** (only when attention itself is the
    bottleneck).

Two sharding mechanisms are in play — HF-native TP (`tp_plan="auto"`, dense models) and the
toolkit's selective attention TP — and both place their sharded params as **DTensors** on the TP
mesh, so one `full_tensor()` walk reconstructs either. The exception is the params sliced by hand
(GptOss `sinks`), which stay plain tensors and are load-bearing for gradients and checkpoints. See
[Gradient sync](#gradient-sync-and-the-grad-norm).

## Supported models

**Native HF TP** (dense models with `base_model_tp_plan`, via `tp_plan="auto"` — shards attention,
MLP, and the vocab dim of a [tied embedding/head pair](#tied-embeddings-under-hf-native-tp)): Llama,
Mistral, any dense HF model declaring the plan.

**Selective TP** (attention only, `apply_tp_to_attention_only()`; MoE handled by EP,
embedding/lm_head replicated):

| Model | Notes | Page |
|---|---|---|
| GPT-OSS | Attention sinks sharded as non-DTensor, all-gathered at save | [gpt-oss.md](../models/gpt-oss.md) |
| Qwen3 dense / Qwen3 MoE | ColwiseParallel(Q/K/V) + RowwiseParallel(O) | [qwen3.md](../models/qwen3.md) |
| Qwen3.5 / Qwen3.6 dense / MoE | Double-width `q_proj` shards uniformly | [qwen3_5.md](../models/qwen3_5.md) |
| GLM-4 MoE Lite | MLA — of the colwise projections only `q_b_proj` / `kv_b_proj` shard (`o_proj` is still rowwise) | [glm4.md](../models/glm4.md) |
| LFM-2 MoE | Standard GQA pattern | [lfm2.md](../models/lfm2.md) |
| Mistral4 | MLA — same as GLM-4 | [mistral4.md](../models/mistral4.md) |
| Qwen3-VL (text tower) | `Qwen3VLTextAttention`; the vision tower stays replicated | [qwen3.md](../models/qwen3.md#qwen3-vl) |
| Cohere2 MoE | Standard GQA pattern; the tied lm_head stays replicated | [cohere2-moe.md](../models/cohere2-moe.md) |

Every MoE under **pure** TP routes to `_load_tp_moe_model` rather than `tp_plan="auto"`, because
HF's auto plan wrongly shards MoE expert biases. Supported attention classes:
`src/distributed/tensor_parallel/module_types.py`.

**Not supported:** every family absent from the table above. None of their attention classes is in
the registry, so a `tp_size > 1` run shards zero layers and raises rather than silently leaving
every weight replicated. Zaya trips the earlier head-divisibility gate at `tp_size > 2`
(`num_key_value_heads` is 2). Behind the missing entries: Gemma 4's KV-shared layers plus
`attention_k_eq_v`, Zaya's CCA/Conv1d front-end, which has no DTensor sharding primitive,
Step-3.7 Flash's per-layer head counts, which fit no uniform q/k/v plan, and GLM-5 Next's DSA
indexer plus KDA projections, which have no sound sharding ([glm5-next.md](../models/glm5-next.md#limitations)).
Use EP for their experts.

## Load paths

Three paths in `load_distributed_model` (`src/distributed/loading/model_loading.py`):

| Path | When | Mechanism |
|---|---|---|
| `_load_tp_model` | Pure TP, dense, `ep_size=1` | `from_pretrained(distributed_config=DistributedConfig(tp_plan="auto"))` — HF-native |
| `_load_tp_moe_model` | Pure TP, MoE, `ep_size=1` | `apply_tp_to_attention_only()`; experts replicated |
| `_load_ep_tp_model` | EP+TP | `apply_tp_to_attention_only()`; EP owns experts |

The HF-native path validates the load's materialization, not just its plan: a plan that shards zero
parameters raises (the run would be `tp_size` replicas), and a plan-sharded param that materialized
as a **plain** tensor raises too (`reject_plan_sharded_plain_params`,
`src/distributed/tensor_parallel/state_dict.py`) — such a bare slice is indistinguishable from a
replica downstream, and the replicated-grad AVG would mix disjoint slices across the TP group.
Hand-sliced params recorded in `_tp_sharded_non_dtensor` are exempt.

### Tied embeddings under HF-native TP

`_load_tp_model` shards a tied embedding/head pair as **one vocab-parallel weight**. A tied config
injects `embed_tokens: embedding_rowwise` into the backbone `base_model_tp_plan`, the ForCausalLM's
`lm_head: colwise_gather_output` agrees with it on `Shard(0)` of the vocab dim, and transformers
shards module by module and ties *afterwards* — so both ends land on the same DTensor: one weight,
one gradient, one exported key, and half the embedding + head memory per rank at `tp_size=2`.

Sharding the pair makes the vocabulary a TP dimension: `colwise_gather_output` refuses a
`vocab_size` that `tp_size` does not divide (`ColwiseParallel.validate_param`, loud at load, naming
the layer), where a replicated pair did not care. Every tied dense model in the roster divides
evenly (Qwen3 151936, Qwen3.5 248320); a vocab-patched checkpoint that does not must drop `tp_size`
to a divisor.

An architecture planning only **one** end of the pair (a multimodal wrapper class declaring no
`lm_head` entry; a backbone shipping no `base_model_tp_plan`) cannot do that — the tie would hand the
unplanned end a weight of the other kind and the first forward dies on mixed plain/DTensor operands.
`consistent_tied_tp_plan` (`src/distributed/tensor_parallel/tie_plan.py`) drops that lone entry for
the load so the pair stays replicated instead, warning that those weights cost a full copy per rank.
`validate_tied_pair_consistent` then fails the
load loud if the loaded pair is untied, or if its sharding disagrees with the applied plan.

For a checkpoint carrying **both** tied keys on disk (the Qwen3-0.6B/1.7B export shape) transformers
compares them before tying, which on two DTensors is an **all-reduce on the default process group** —
every rank must reach it. `_load_tp_model` therefore does not throttle its ranks with
`max_concurrent_loading`: a rank-serialized load blocks the loading rank inside that collective while
its peers wait their turn, and the job hangs until the store timeout. Nothing is lost — transformers
streams the checkpoint key by key and places each rank's shard straight on its GPU, so host RAM
never holds the model.

## The selective-TP plan

`apply_tp_to_attention_only` (`src/distributed/tensor_parallel/parallelize_attention.py`) walks the
model and builds a per-module DTensor plan:

| Module | Strategy | Notes |
|--------|----------|-------|
| `self_attn.{q,k,v}_proj` | `ColwiseParallel` | per-head |
| `self_attn.{q_b,kv_b}_proj` | `ColwiseParallel` | MLA: only the head-dim expansion shards |
| `self_attn.{o,out}_proj` | `RowwiseParallel` | the all-reduce point |
| `self_attn.sinks` | sliced (`shard_sinks_param`) | GptOss, one slice per head shard |
| per-head norms (`q_norm`/`k_norm`, LFM2 `q_layernorm`/`k_layernorm`) | replicated + grad all-reduce | see below |
| `embed_tokens` / `lm_head` | replicated | the plan never visits them: DTensor-sharding them faults cuBLAS under EP+TP |

Head divisibility is validated before sharding: `num_attention_heads % tp_size == 0` always, and
`num_key_value_heads % tp_size == 0` for standard GQA. `ColwiseParallel` splits Q/K/V with no
knowledge of heads, so an indivisible count would split *within* a head; the guard raises instead.
MLA models skip the KV-head check — they shard by query head and keep the KV compression replicated.

**Per-head norm gradients.** A norm applied *after* a colwise projection is a replicated
`(head_dim,)` parameter shared across heads. `ColwiseParallel` defaults `use_local_output=True`, so
the DTensor graph ends at the projection and each rank's gradient covers only its own heads — the
true gradient is the **SUM** over the TP group, not the average a replica gets.
`apply_tp_to_attention_only` finds those norms structurally (any unplanned attention child with
trainable parameters, never a name list), records their names on `model._tp_per_head_norm_params`,
and `_sync_tp_replicated_grads` sums them once per optimizer step. A name missing from that set is
AVG-reduced as a plain replica, i.e. `1/tp_size` of its true gradient.

MLA families are exempt: their `q_a`/`kv_a` norms sit *before* the colwise expansion, where
DTensor's `Replicate` backward already all-reduces.

Transformers' own `ReplicatedWithGradAllReduce` is deliberately **not** used: its
`full_backward_hook` re-reduces whatever sits in `.grad` on every backward, so under gradient
accumulation each earlier micro-batch's contribution is multiplied by `tp_size` again per
micro-step. On the HF-native `tp_plan="auto"` path `retarget_hf_replicated_grad_hooks` strips those
hooks into the same step-time SUM, and raises if the plan asks for that reduction and no hook is
found. Guarded by `tests/gpu/parallelism/tp/test_tp_attention_norm_grad.py`.

MoE layers are never in this plan. In EP+TP the experts belong to EP; in pure-TP MoE they stay
replicated.

### Identical-input requirement

Every TP rank must call `model.forward` with **identical** `input_ids` / `attention_mask` shapes
*and* values — column/row-parallel collectives assume each rank computes a partial result for the
same hidden states. Mismatched inputs do not error early; they deadlock at the first `o_proj`
all-reduce (visible in py-spy as ranks stuck at different sub-modules of one forward).

This affects trainers pulling per-rank data from an external service. `DistributedGRPOTrainer`
overrides `_generate_single_turn` to broadcast the vLLM rollout tuple from the TP-group leader. For
custom trainers, add an `assert_consistent` / `assert_tensor_shape_consistent` call
(`src.diagnostics.debugging`) at the suspect tensor — the toolkit ships no call sites, so
`HALO_TP_CONSISTENCY_CHECK=1` on its own produces nothing; it only turns those checks' warning into
a raise.

## Quick start

```bash
# Pure TP=4 on 4 GPUs (no data parallelism)
torchrun --nproc_per_node=4 scripts/training/sft.py \
    examples/sft/qwen3/qwen3-4b-ultrachat.yaml \
    --tensor_parallel_size=4
```

`--tensor_parallel_size` defaults to 1 (TP off) and is the only TP flag — there is no
`--tensor_parallel_plan`. Dense TP always passes `distributed_config=DistributedConfig(tp_plan="auto")`;
`tp_plan` and `device_map` are mutually exclusive, so do not set `device_map`.

## TP + data parallelism

2D `(dp, tp)` mesh: TP within a node, FSDP2 across replicas. DTensor gradients cannot be reduced
with plain `dist.all_reduce` or DDP — FSDP2's `fully_shard` understands DTensor semantics. The
loader builds the mesh and attaches it to the model; the trainer wraps over the DP dimension via
`setup_fsdp2_for_tp` (`src/distributed/fsdp.py`), reusing that mesh. Pure TP (DP=1)
skips FSDP. Prefer `tp_size = world_size` when the model fits the TP group's memory.

## Gradient sync and the grad norm

A TP run holds four kinds of parameter, and each needs a different reduction. Getting the kind
wrong is silent — the loss stays finite and the objective is just different:

| Kind | Produced by | Grad sync | Grad-norm bucket |
|---|---|---|---|
| DTensor shard | either mechanism — HF's `tp_plan` styles, selective attention TP | DTensor / FSDP2 | summed over the axes its mesh names |
| **Plain slice** | hand-sliced GptOss `sinks` (`model._tp_sharded_non_dtensor`) | none — the slices are already disjoint | summed over the TP group |
| Replica | `embed_tokens`/`lm_head` under selective TP, router, norms outside attention | `_sync_tp_replicated_grads` averages over the TP group (the local `dp` shards once FSDP2 wraps them — TP siblings hold the same shard) | counted once, no TP reduce |
| Per-head attention norm | a norm after a colwise projection (`q_norm`/`k_norm`) | `_sync_tp_replicated_grads` **sums** over the TP group | counted once, no TP reduce |

The trap is the plain slice. It is indistinguishable from a replica by tensor type, and once FSDP2
wraps the run for DP **both** become 1-D `dp` DTensors, so the mesh cannot separate them either.
`_tp_sharded_plain_param_ids()` (`src/trainers/mixins/grad_sync.py`) is the single source of truth:
it reads the `_tp_sharded_non_dtensor` registry — the `(suffix, shard_dim)` pairs recorded when a
param is sliced by hand — and both `_sync_tp_replicated_grads` and `_sharded_grad_bucket` key off
it. Everything a TP plan shards is a DTensor whose gradient reduces itself, so the plan contributes
no entries here.

Classifying a slice as a replica costs twice: the TP-group average mixes another rank's heads into
every slice's gradient, and the norm counts only the local slice, making the global norm
rank-dependent so TP peers clip the same parameter by different coefficients. Guarded by
`tests/gpu/parallelism/tp/test_tp_correctness.py` and `test_tp_dp_correctness.py`, both against an
unsharded single-GPU grad norm.

## EP+TP mode

EP+TP shards attention with TP and distributes MoE experts with EP on the same GPU groups
(`_load_ep_tp_model`). Router, `embed_tokens`/`lm_head`, and norms stay replicated. `ep_size` must be
a multiple of `tp_size`, TP stays NVLink-local, and DP = `world_size / tp_size`. Locality rules, the
EP/TP/DP example shapes, and the gradient-sync groups: [multi-node.md](multi-node.md#eptp-mode).

For MoE where only expert memory is the bottleneck, shard the expert FFN with
[ETP](expert-tensor-parallelism.md) instead. Attention TP and expert TP are mutually exclusive.

**Numerics.** EP+TP in bf16 carries a deterministic offset against an unsplit reference: TP reorders
the attention reductions feeding the router, which flips a handful of near-tied top-k expert picks
and moves whole-expert mass. Gradient scale and direction still match — this is bf16 reduction
order, not a sync bug. Measured **0.060** loss offset on gpt-oss-20b (top-4-of-32), so the
correctness gate uses `TOL.router_pick_flip_loss_abs` (0.1, shared with the EP+ETP gate) instead of
the generic parallel-vs-baseline bound; the rotated-expert negative control sits at 0.344 with
gradient cosine 0.16, keeping health and breakage more than 3× apart.

## Limitations

**Trainers.** `_supports_tp` defaults `True` on the mixin and no trainer overrides it, so no trainer
rejects TP; the
only trainer-level TP gate that fires is the LoRA one below. Matrix:
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility).

**Models.** There is no per-family "supports TP" flag. Selective TP shards exactly the attention
classes in `TP_SHARDABLE_ATTENTION_CLASSES` (`src/distributed/tensor_parallel/module_types.py`), and
a `tp_size > 1` run that shards **zero** layers raises, naming the model's attention classes — that
is how the [unsupported families](#supported-models) are rejected. A run that shards *some* layers
(hybrid stacks — Qwen3.5/3.6 `GatedDeltaNet`, LFM-2 short-conv) only **warns**: numerics stay
correct, but the unsharded layers keep a full replica per rank, so per-rank memory falls by far less
than `1/tp_size`. Head divisibility is checked first — `num_attention_heads % tp_size == 0` always,
`num_key_value_heads % tp_size == 0` for non-MLA GQA. The dense HF-native `tp_plan="auto"` path gets
the same gate: `ParallelismConfig.validate_against_model_config` runs `validate_tp_head_divisibility`
on every load path, off `config.json` before any weight is read. Transformers validates no head count
of its own, so without it the run dies on the first forward's reshape — after the whole checkpoint
has been pulled and placed on every rank.

**Axis combinations.** TP composes with EP only. TP+CP, TP+ETP, EP+TP+ETP and PP+TP are refused by
the [allowlist](index.md#supported-combinations); multi-domain multi-group EP+TP is rejected
separately (`_validate_tp`).

**Knobs.** Everything below raises unless the verdict says otherwise.

| Knob | Under TP | Gate |
|---|---|---|
| `use_peft` / LoRA (incl. EP+TP) | rejected — PEFT adapters are plain tensors outside the TP graph, so the replicated half diverges and the sharded half is corrupted by the replicated-grad sync. Use LoRA with FSDP/DP, CP, or pure ETP | `_validate_lora_tp_compatibility` |
| native EP expert LoRA | rejected — the grouped expert adapters ride the EP-distributed expert weights, which both TP gates skip by param identity, so no adapter shape under EP+TP is gradient- or save/merge-covered. Train expert LoRA under EP without TP | same |
| QLoRA / `load_in_4bit` | rejected — the TP loaders materialize plain de-quantized weights | `model_loading.py` |
| `fsdp_reshard_after_forward` | rejected at `data_parallel_size > 1` — a plain all-gather on TP-sharded DTensor params has no registered sharding strategy | `_validate_fsdp_settings` |
| `fsdp_shard_ep1_experts: false` | rejected — the TP path shards `ep1` experts unconditionally | same |
| `use_hsdp` | rejected — TP builds its own `(dp, tp)` mesh | `_validate_hsdp` |
| `init_from_scratch` | rejected — no sharded random init | `model_loading.py` |
| `accelerate launch` | rejected — TP requires `torchrun` | `model_loading.py`, `ParallelismValidationMixin` |
| `ref_model` (explicit) | rejected — the reference is never parallelized, so its log-probs would not match the policy | `_validate_reference_model` |
| `load_best_model_at_end` | supported under pure TP — the reload distributes each checkpoint tensor into the live placements; rejected as a full fine-tune under TP+DP and under a MoE carrying EP/grouped-GEMM wrappers | `_validate_load_best_model_reloadable` |
| `use_liger_kernel` | supported; `cross_entropy` and `fused_linear_cross_entropy` are forced off (warned when explicitly enabled) | `kernels/liger/orchestrator.py` |
| `added_special_tokens` that grow the vocab | unsupported on a dense **tied** model, whose embedding is a vocab-sharded DTensor: `resize_token_embeddings` dies in its mean-resizing Cholesky, which has no DTensor sharding strategy. Patch the vocab offline (`scripts/before_training/patch_vocab.py`) instead | — (raises in transformers) |
| `gradient_checkpointing`, `packing`, `padding_free`, `lowp_precision`, `moe_balancing` | not gated under TP | — |
| `torch_compile` | not gated, and no meaningful speedup — DTensor's sharded-op dispatch breaks the graph at every sharded op. Use Liger | — |

## Multi-node

TP requires an NVLink-bandwidth all-reduce after every layer, so it must stay **within a domain**;
cross-node TP over InfiniBand would bottleneck every pass. The pattern is TP within nodes, FSDP
across nodes. `nvlink_domain_size % tp_size == 0` is enforced at config time.

```bash
# Each node (set --node_rank per node)
torchrun --nnodes=2 --node_rank=0 --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    scripts/training/sft.py --tensor_parallel_size=4
```

`DIST_NCCL_TIMEOUT_MINUTES` (default 30) is pinned onto the default group and every EP/CP/TP subgroup
via `apply_default_pg_timeout`. DeepEP's dispatch/combine runs behind its own `ElasticBuffer`
GPU-side barrier, not the PyTorch watchdog, so this does **not** bound it.

## Checkpoint saving

`save_tp_model` (`src/distributed/tensor_parallel/checkpoint.py`) drives TP saving, and it always
gathers: all TP ranks reconstruct full tensors and one save rank writes a standard HF checkpoint, put
through `normalize_gathered_state_dict` (hub expert layout, save dtype) like the FSDP2 and CP
writers. There is no per-rank TP save. The write streams through `stream_gathered_checkpoint`, so
the save rank's host RAM is bounded by one chunk plus one pending shard rather than the whole model.

`chunked_saveable_tensors` reconstructs every sharded param with `DTensor.full_tensor()` — one walk
covers both mechanisms — and `gather_tp_sharded_non_dtensor_params` all-gathers the hand-sliced
GptOss `sinks` after it. Both are **collective**: every TP-mesh rank must drive them or the gather
hangs.

On resume every TP shape is Path B: the training scripts repoint `model_name_or_path` at the
checkpoint, so the weights load at construction and `CheckpointLoader._load_tp` skips the re-read.
Where it does read (a best-model reload, or a model built from elsewhere), each rank streams the
checkpoint's full tensors and `distribute_tensor`s them into the live DTensor placements. TP+DP is
the exception: FSDP2 over TP stacks a strided `dp` shard on the `tp` shard, a 2-D placement
`distribute_tensor` does not invert for packed projections, so it refuses every reload but the
constructed-from-checkpoint skip. See
[Checkpoints](../reference/checkpoints.md#resume-by-parallelism-mode).

## Troubleshooting

- **"tp_plan and device_map are mutually exclusive"** — drop `device_map`.
- **Quiet non-main ranks** — expected: transformers drops to ERROR off the main rank, and the script logger follows `log_level_replica` (HF default `warning`).
- **DTensor all_reduce error with DDP** — DTensors cannot be used with DDP; use FSDP2.
- **OOM with TP + FSDP** — enable gradient checkpointing.
