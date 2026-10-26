# Context Parallelism (CP)

CP splits long sequences across GPUs using Ulysses sequence parallelism with all-to-all
communication in attention layers. Use it when a sequence is too long to fit in single-GPU
activation memory. CP runs standalone or combined with EP (MoE models). Why sequence length, not
parameter count, drives that memory:
[GPU Training Theory §8](../reference/gpu-training-theory.md#activations-the-memory-sharding-doesnt-touch).

`data_parallel_size = world_size / cp_size`; gradient sync always spans all `world_size` ranks.
CP is **not compatible with TP** — both partition the same contiguous rank blocks, and TP has
already cut this rank's head count that Ulysses redistributes over.

## How Ulysses works

Naive sequence chunking breaks cross-chunk attention. Ulysses instead redistributes tensors around
the attention kernel: each GPU holds `S/cp_size` tokens but all heads, all-to-alls swap that to
the full `S` tokens but `H/cp_size` heads for the attention compute (one for Q, one for the fused K
and V), and a third swaps the output back. The backward runs the same three in reverse.
Multi-head attention is independent across heads, which is what makes this exact.

![Ulysses attention: RoPE runs on the local chunk, all-to-alls on Q and the fused K|V scatter heads and gather the sequence so each rank attends over the full sequence with 16 of 64 heads, and an all-to-all on the output restores the per-rank chunk layout](../assets/diagrams/ulysses_attention_flow.png)

![An SFT training step under CP: the wrapper narrows the batch to one contiguous sequence chunk per rank, embeddings and every decoder layer run on that chunk (Ulysses attention, then the MLP or MoE), the logits stay sharded, and the loss is scaled by cp_size over the CP group's all-reduced token count](../assets/diagrams/cp_training_step.png)

## Implementation

`UlyssesModel` (`src/distributed/context_parallel/ulysses_model.py`) is the entry point: the CP
loaders construct it over the loaded model and the trainer adopts its `CPConfig`. The all-to-all
(`UlyssesAllToAll`, `all_to_all.py`) is a custom autograd function: `sequence_to_heads` scatters heads
and gathers the sequence, `heads_to_sequence` swaps back. The per-family attention wrappers in
`families.py` subclass `UlyssesAttention` (`ulysses_attention.py`), whose one forward path runs RoPE
(and any position-dependent scale) on the local chunk before the all-to-all, so no position gather,
then Flash Attention with native GQA in `[B, S, H, D]`, with zero transposes. The MLA families
(GLM4 MoE Lite, Mistral4) share `MLAUlyssesAttention`: RoPE on the rope half of each head, and an
`_attend` that right-pads V to `qk_head_dim` where `v_head_dim` differs, runs flash, and crops back.

### Loss computation

CP computes loss per chunk with two corrections:

1. **Boundary tokens** — the causal label shift (`logits[:-1]` predicts `labels[1:]`) loses one
   prediction per chunk. Non-final ranks use their last logit to predict the first token of the next
   chunk.
2. **Globally-normalized sum loss** — completion-only training distributes tokens unevenly across
   chunks, so per-rank `reduction='mean'` biases gradients. Each rank computes
   `loss_i = cp_size * local_sum / global_tokens` with `global_tokens` the all-reduced non-masked
   token count. After FSDP2's `1/cp_size` gradient averaging every token contributes equally.

That form is rank-varying and exists for the gradient path only. Under `torch.no_grad()` (the eval
loop) the wrapper returns the rank-uniform group value instead — the CE sum and the aux term are
all-reduced over the CP group — because HF's DP-scoped metric gather keeps a single CP sibling's
copy as `eval_loss`, and a chunk-partial copy would bias it by that chunk's share of loss tokens.

The MoE router aux loss takes **no** `cp_size` factor. Each rank's `load_balancing_loss_func` sees
only its chunk's router logits, so the term is chunk-local — the load and the mean router
probability of that chunk — and the FSDP average over CP ranks trains the mean of the per-chunk
terms at `router_aux_loss_coef`. That is not the whole-sequence term (a product of means does not
split across chunks); a `cp_size` factor would only scale it `cp_size`×.

The GRPO scoring seam calls `forward_hidden_states` on the CP wrapper, which returns this rank's
backbone hidden states without constructing vocabulary-wide logits. `cp_shift_against_full_labels`
pairs them with the full row's next-token labels, and the vocab-chunked output head scores those
targets. Offline GRPO's per-row token-objective sums (and SMPO's per-sequence sums) reduce through
`cp_sum_rows` in `src/distributed/context_parallel/sequence_chunks.py`, an autograd-aware SUM whose
backward supplies the CP factor canceled by world-wide mean gradient synchronization. They must
**not** also use the SFT loss's explicit `cp_size` multiplier. Offline GRPO uses this path for
training, evaluation and reference scoring; online and environmental GRPO do not declare CP. Its
reference and resume contract is in
[Offline GRPO](../training-methods/grpo/offline.md#reference-model).

A model that returns an `aux_loss` while its config declares no `router_aux_loss_coef` **raises** — a
stand-in weight would train a different objective than the same config without CP. Set the field
through `model_init_kwargs`, or turn the aux loss off for that family with `moe_balancing`.

## Requirements

**NVLink-local topology** — `cp_size` must divide the NVLink domain (`gpus_per_node` on a standard
node, the rack on NVL72) and cannot exceed it; Ulysses' all-to-alls per attention layer must stay
on NVLink. `cp_size=3` on an 8-GPU domain is rejected at config time, before the model loads
(`_reject_cp_locality`).

**Attention impl** — a real Flash Attention is required. `reject_unsupported_ulysses_model` raises
`CPConfigError` for any `_attn_implementation` outside `SUPPORTED_ATTN_IMPLEMENTATIONS` (FA2,
FA3, FA4, community kernel paths): `eager` and `sdpa` are rejected, and `flex_attention` is
auto-switched to FA4/FA2 by `load_distributed_model` when CP is active. A wrapper declaring
`REQUIRES_FLASH_ATTN_LABEL = False` waives the check for modeling code that cannot carry a flash
label at all (Bailing; see [bailing.md](../models/bailing.md#cp-wrapper)).

That check reads the *declared* implementation; the kernel CP actually calls is resolved separately,
arch-matched with an FA4 veto for the FA4-backward-NaN families
([Flash Attention → Supported backends](../optimization/flash-attention.md#supported-backends)). Leave
`attn_implementation` at its auto-detected default. Bailing is the exception — it needs an explicit
`sdpa`, since transformers refuses the auto-detected FA4 at model build.

**Head divisibility** — `cp_size` must divide both the Q and the KV head count. GPT-OSS (64 Q, 8 KV):
CP=8 → 8 Q / 1 KV; CP=4 → 16 Q / 2 KV; CP=3 rejected.

**Sequence length** must be a multiple of `cp_size`. `scripts/training/sft.py` sets `pad_to_multiple_of=cp_size`
and the collator factory routes to a padding collator that rounds each batch up — no manual collator
construction. A ragged batch that reaches SFT's `compute_loss` anyway is right-padded there, which needs
the tokenizer's `pad_token_id`: a `processing_class` without one raises rather than padding with
vocabulary token 0.

**Right padding only.** The Ulysses path ignores `attention_mask`: it calls dense flash attention
with `causal=` and has no varlen path, so it tolerates only padding a causal mask already ignores.
A left-padded batch is rejected on every forward. Every real token would attend the leading pads
and the loss would silently differ from the same batch without CP.

SMPO's collator left-pads prompts; its chosen|rejected concat flushes every row left, so every pad
trails the completion and any batch size runs.

Offline GRPO's CP collator right-pads each joined prompt+completion row
([Offline GRPO](../training-methods/grpo/offline.md)).

### Supported model architectures

Source of truth: `ULYSSES_ATTENTION_MAP` in `src/distributed/context_parallel/families.py`, built
by walking the `UlyssesAttention` subclass tree and reading each wrapper's `HF_MODULE_NAMES` (a
duplicate HF name raises). The validator and the patcher both read that map — there is no separate
accept list to keep in sync.

| Attention class | Wrapper | Family specifics | Page |
|---|---|---|---|
| `GptOssAttention` | `GptOssUlyssesAttention` | split-halves RoPE; sinks must be reset | [gpt-oss.md](../models/gpt-oss.md) |
| `Qwen3MoeAttention` (Qwen3 MoE) | `Qwen3MoeUlyssesAttention` | Q/K RMSNorm | [qwen3.md](../models/qwen3.md#qwen3-moe) |
| `Qwen3Attention` (Qwen3 dense) | `Qwen3MoeUlyssesAttention` (reused) | Q/K RMSNorm | [qwen3.md](../models/qwen3.md#qwen3-dense) |
| `Qwen3VLTextAttention` (Qwen3-VL text branch) | `Qwen3MoeUlyssesAttention` (reused) | Q/K RMSNorm | [qwen3.md](../models/qwen3.md#qwen3-vl) |
| `Glm4MoeLiteAttention` | `Glm4MoeLiteUlyssesAttention` | DeepSeek-V3 MLA: separate qk/v head dims | [glm4.md](../models/glm4.md) |
| `Mistral4Attention` | `Mistral4UlyssesAttention` | MLA: separate qk/v head dims, llama-4 scaling | [mistral4.md](../models/mistral4.md) |
| `Qwen3_5MoeAttention` / `Qwen3_5Attention` | `Qwen3_5MoeUlyssesAttention` | post-attention sigmoid gate + partial RoPE | [qwen3_5.md](../models/qwen3_5.md#cp-wrapper) |
| `BailingMoeV2Attention` / `…FlashAttention2` / `…SdpaAttention` (Ling 2.0) | `BailingMoeV2UlyssesAttention` | fused `query_key_value`, `dense` output projection, Q/K norm before partial RoPE | [bailing.md](../models/bailing.md#cp-wrapper) |
| `Cohere2MoeAttention` | `Cohere2MoeUlyssesAttention` | interleaved fp32 rotary on sliding layers only; full-attention layers are NoPE | [cohere2-moe.md](../models/cohere2-moe.md) |

**Not supported on production checkpoints:**

- **Qwen3.5 / Qwen3.6** — the full-attention wrapper exists, but every released checkpoint also ships `Qwen3_5MoeGatedDeltaNet` linear-attention layers (sequence-axis Conv1d + recurrent scan) that can't be sharded. Validation rejects the `GatedDeltaNet` classes by name, and any config whose `layer_types` contains `"linear_attention"`. See [qwen3_5.md — Why CP is blocked on real checkpoints](../models/qwen3_5.md#why-cp-is-blocked-on-real-checkpoints).
- **Ring-mini-linear-2.0** (`bailing_moe_linear`) — Lightning Attention-2 in most layers; validation rejects `BailingMoeV2LinearAttention` by name. Ling 2.0 itself is supported (table above).
- **Ling 3.0** (`bailing_hybrid`) — 3 of every 4 layers are `BailingMoeV3KimiDeltaAttention`, a KDA linear recurrence, and the MLA layers carry no wrapper; validation rejects the model as having no supported attention module.
- **LFM-2** — hybrid short-convolution layers mix tokens along the sequence axis, so a Ulysses split severs the conv receptive field.
- **Gemma 4** — KV-shared layers + `attention_k_eq_v`.
- **Laguna** — `LagunaAttention` has no Ulysses wrapper registered.
- **Inkling** — `InklingShortConvolution` runs a depthwise causal `Conv1d` over the sequence axis, and position enters as an additive relative-logits bias `flash_attn_func` cannot take (the family declares `_supports_flash_attn = False`). Validation rejects the conv class. Use EP (optionally + ETP) instead — see [Inkling](../models/inkling.md).
- **Zaya** — the CCA front-end runs depthwise + grouped `Conv1d` (kernel size 2) along the sequence axis and concatenates a delayed `v_proj_delayed(h_{t-1})` into V; both cross Ulysses chunk boundaries and would need a per-layer halo exchange the path does not provide.
- **DeepSeek-V4** — the CSA/HCA compressors (and the Lightning Indexer scoring against them) pool non-overlapping token windows along the sequence axis, so a CP shard would compress incomplete windows at every chunk boundary. Validation rejects the classes. Use EP instead — see [DeepSeek-V4](../models/deepseek-v4.md).
- **GLM-5 Next** (`glm5_next`) — 34 of 45 layers are `Glm5NextTextLinearAttention`, a KDA linear recurrence (sequence-axis conv1d k=4 + delta-rule scan); validation rejects the class by name (and the `"linear_attention"` entries in `layer_types`). Use EP instead — see [GLM-5 Next](../models/glm5-next.md).
- **Step-3.7 Flash** — `Step3p7Attention` has no Ulysses wrapper registered, so validation rejects the model as having no supported attention module. Nothing architectural blocks a wrapper (plain full/sliding GQA; 64/96 heads and 8 KV heads divide cp 2/4/8). Use EP instead — see [Step-3.7 Flash](../models/step3p7.md).

## CP with EP

EP is orthogonal to DP; CP reduces it. So `data_parallel_size = world_size / cp_size` whether EP is
on or off. CP does its all-to-all in attention and EP its DeepEP dispatch/combine in the MoE
block — different points in the forward pass, no conflict.

**Orthogonal mode** (`cp_size == world_size`): one CP group over all ranks, DP=1. **Partial mode**:
`world_size / cp_size` CP groups, each on a different batch.

CP always requires gradient sync when `world_size > 1`, even at DP=1 — each rank holds partial
gradients from its chunk. `DistributedTrainerMixin` applies FSDP2 across all ranks; for EP+CP the
same wrapper runs with EP params in `ignored_params`.

EP+CP requires node-local EP with `ep_group_size == nvlink_domain_size`
(`parallelism_rules._reject_ep_cp` rejects cross-domain EP under CP). Supported 8-GPU shapes:
ep8/cp2 → DP4, ep8/cp4 → DP2, ep8/cp8 → DP1.

**Pure CP on a MoE still gets expert wrappers.** `load_cp_model` builds an `ep_size == 1` EP
config whenever `needs_ep_wrappers` holds and the model declares experts, so the
MoE blocks run the [grouped-GEMM](../optimization/grouped-gemm.md) expert path rather than the stock
per-expert loop — `needs_ep_wrappers` forces Liger's routed-expert swiglu/geglu off either way, so
without it the model would pay that cost for no speedup. Dense models under CP get no EP config.

## Usage

```bash
# CP-only (any supported model, long sequences)
torchrun --nproc_per_node=4 scripts/training/sft.py \
    examples/sft/qwen3/qwen3-4b-ultrachat.yaml --context_parallel_size=4 --packing=false

# EP+CP (MoE, node-local EP with ep_group_size equal to the NVLink domain)
torchrun --nproc_per_node=8 scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
    --expert_parallel_size=8 --ep_scope=node --context_parallel_size=2 --packing=false
```

Both example configs set `packing: true` (refused under CP), and the gpt-oss one sets
`ep_scope: global` (refused under CP), so the commands override them.

Programmatic: `parallelism_config=ParallelismConfig(ep_size=8, cp_size=8)`.

### Checkpoint saving

`save_cp_checkpoint` serves **dense** CP runs only. An MoE carries EP wrappers (grouped GEMM is
on by default), and the saver ladder checks for EP layers first, so EP+CP and even `ep_size=1`
MoE take the EP save — which strips the `.original_attention.` prefix itself.

`trainer.save_model(output_dir)` gathers on every rank and the FS-aware save rank writes.
`_find_cp_wrapper()` finds the wrapper on the compile-unwrapped model, directly or under PEFT. The
wrapper's `state_dict()` remaps `.original_attention.` keys to standard paths and drops the
duplicate dense `.mlp.{gate,up,down}_proj` keys only on layers carrying routed experts — genuinely
dense layers keep their dense MLP weights.

On resume CP is a **Path B** mode: the trainer skips the checkpoint weight-reload (the CP wrapper
changes the module tree). The training scripts repoint `model_name_or_path` at the checkpoint so
`load_distributed_model` loads the trained weights at construction; a model not constructed from
the checkpoint raises rather than silently continuing on its current weights.

Configured FP32 masters are restored before the CP wrapper is installed
([Load precision](../models/README.md#load-precision)).

Trainer state is restored, and so are LoRA adapters: the saved CP-normalized keys are remapped back
onto the live wrapped names, and a wholesale key miss raises.

`load_best_model_at_end` is refused at construction for CP full fine-tunes (base weights only load
at construction, so the best checkpoint cannot be reloaded in place). See
[Checkpoints & Resume](../reference/checkpoints.md).

## Optimizations

Native GQA (no `repeat_kv`) saves on the order of 6 GB per forward on a 20B-class GQA model, and the
`[B, S, H, D]` layout drops the transposes. The all-to-alls run in native BFloat16, with K and V fused
into one.

`torch.compile` gives no benefit: the all-to-all breaks the graph at every attention layer (±0.0% on
Qwen3-8B, CP=2, seq 16384, 2 GPUs (Blackwell)). Use Liger (default on).

## Limitations

**Trainers.** CP is declare-to-enable (`SUPPORTS_CP`, default `False`): only
`DistributedSFTTrainer`, `SmoothMarginPOTrainer` and `OfflineGRPOTrainer` declare it. Offline GRPO
under CP requires full fine-tuning: it rejects PEFT and native expert LoRA at any `kl_beta`, and an
explicit `ref_model`, since it scores the run-start reference through the distributed policy
(`reject_offline_cp_options`). Every other trainer raises at construction, and its entry script
rejects `--context_parallel_size > 1` earlier through
`parallelism_config_from_args(..., trainer_cls=...)`, which reads the same flag. Full matrix:
[Trainer Compatibility](../reference/trainer-architecture.md#trainer-compatibility). Nothing
inspects a trainer's loss for CP-safety, so a new trainer must verify its own objective before
declaring the flag.

**Models.** Only the wrappers in the [table above](#supported-model-architectures); anything else
raises `CPConfigError`. Linear-attention, sequence-axis convolution and window-pooling blocks
are rejected by class name even when the surrounding attention is wrapped.

**Generation.** `UlyssesModel.generate()` raises: each rank holds one sequence chunk and the
Ulysses attention has no KV-cache path. Generate from a saved checkpoint without CP.

**Axis combinations.** CP composes with EP only; TP+CP and ETP+CP are refused by the
[allowlist](README.md#supported-combinations). EP alone OOMs at long sequences (it shards experts,
not activations) — that is the case CP exists for.

**Knobs.** Everything below raises unless the verdict says otherwise.

| Knob | Under CP | Gate |
|---|---|---|
| `packing`, `padding_free` | rejected — the Ulysses path runs a dense causal kernel with no per-document boundaries | `src/data/collators/factory.py`; `reject_cp_packed_collator` re-checks a hand-built collator |
| left-padded batches | rejected on **every** forward (not cached — a prompt-left-padding collator pads only its ragged batches) | `context_parallel/ulysses_model.py` |
| `attn_implementation` | must resolve to FA2/FA3/FA4 or a community flash kernel; `flex_attention` is auto-switched with a warning, `eager`/`sdpa` rejected — except for a wrapper declaring `REQUIRES_FLASH_ATTN_LABEL = False` (Bailing), which runs on the `sdpa` label its model build forces | `SUPPORTED_ATTN_IMPLEMENTATIONS`, `reject_unsupported_ulysses_model` |
| `init_from_scratch` | rejected — no sharded random init | `model_loading/parallel/load_rules.py` |
| `reset_sinks: false` on GPT-OSS | rejected at model load — the CP attention kernels drop the sink column, so live sinks would misnormalize the softmax in every layer | `reject_unsupported_model`, re-checked in `GptOssUlyssesAttention` |
| `label_smoothing_factor > 0`, `loss_type: dft` | rejected — the Trainer pops `labels` and pairs full labels with this rank's chunk logits | `reject_cp_trainer_args` |
| `compute_metrics`, `preprocess_logits_for_metrics` | rejected whenever set, not gated on `eval_strategy` (a direct `evaluate()`/`predict()` reaches the same misaligned chunk-logits path) — eval under CP is loss-only | same |
| CP's own metrics (`mean_token_accuracy`, `entropy`, `aux_loss`, `num_attended_tokens_seen`) | accumulated locally per micro-batch and reduced once per log, so the metric path adds one collective per log instead of five to six per micro-batch | `src/trainers/sft.py` (`_drain_cp_metrics`) |
| multimodal inputs (`pixel_values`) | rejected — text-only | `context_parallel/ulysses_model.py` |
| `fsdp_shard_ep1_experts: false` | rejected — the CP path shards `ep1` experts unconditionally, so the flag would be a silent no-op | `_reject_fsdp_settings` |
| `accelerate launch` | rejected — CP requires `torchrun` | `reject_unsupported_run` and `reject_accelerate_launch` |
| `gradient_checkpointing` | supported; `use_reentrant` is forced to `True` (warned when the config sets `false`) | `DistributedTrainerMixin._apply_training_args_overrides` |
| `use_liger_kernel` | supported; `cross_entropy` and `fused_linear_cross_entropy` are forced off (warned when explicitly enabled) | `src/model_loading/liger/dispatch.py` |
| `use_peft` / LoRA | supported for SFT and SMPO — CP leaves attention unsharded, so adapters stay replicated; offline GRPO rejects adapters | `reject_offline_cp_options` (offline GRPO) |
| QLoRA (`load_in_4bit`) | supported for SFT and SMPO on a **dense** model — CP keeps the standard loader and preserves `Params4bit`. On an MoE the grouped-GEMM loader takes over and rejects a quantized base (`use_grouped_gemm` is on by default); offline GRPO rejects adapters | `model_loading/parallel/load_rules.py`, `reject_offline_cp_options` |
| `use_hsdp`, `fsdp_reshard_after_forward` | supported — CP is one of HSDP's two accepted paths (pure DP, CP) and one of ZeRO-3's EP-free shapes | `_reject_hsdp`, `_reject_fsdp_settings` |
| `torch_compile` | not gated, and no measured benefit — the all-to-all breaks the graph at every attention layer | — |
| `save_sharded_ep` | rejected on a CP run — per-rank shards carry `.original_attention.` keys the merge script cannot remap | `reject_unmergeable_ep_sharded_save` |

## Adding a new model

1. Subclass `UlyssesAttention` in `families.py`. The default `_project_qkv` reads plain
   `q_proj`/`k_proj`/`v_proj` into `[B, S, H, D]`, the default `_apply_rotary` is rotate-half
   (`apply_rope`) and the output projection is `OUTPUT_PROJECTION = "o_proj"`; override only what the
   family does differently (GptOss split-halves rotary, Cohere2 interleaved fp32, Bailing's fused
   projection and `dense`).

    For an MLA family subclass `MLAUlyssesAttention` instead: it owns the geometry, the rope-half
    `_apply_rotary` and the V-padding `_attend`.

2. Declare the HF attention class name in the wrapper's `HF_MODULE_NAMES`. `ULYSSES_ATTENTION_MAP`
   is read off the subclass tree, so defining the class is the whole registration.
3. Test:

```bash
torchrun --nproc_per_node=2 \
    tests/gpu/parallelism/cp/test_cp_correctness.py
# EP+CP, if MoE (2-GPU minimal repro: EP=CP=world=2):
torchrun --nproc_per_node=2 \
    tests/gpu/parallelism/combined/test_ep_cp_correctness.py
```

Further CP tests live in `tests/gpu/parallelism/cp/`.

Background: [DeepSpeed-Ulysses](https://arxiv.org/abs/2309.14509).
