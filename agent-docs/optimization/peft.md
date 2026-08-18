# PEFT (LoRA) Fine-Tuning

LoRA trains small rank-decomposed adapter matrices while the base model stays frozen, eliminating optimizer
states for frozen parameters.

It runs under DDP/FSDP, EP, CP and pure ETP; **TP**, **EP+TP** and **PP** reject it, and QLoRA runs only
under DDP/FSDP and CP-on-dense (full matrix and reasons: [Parallelism compatibility](#parallelism-compatibility)).
Under EP, LoRA targets both attention (via PEFT) and the MoE experts (via native grouped adapters).

Most trainers take it. The embedding trainer
accepts plain LoRA only ([below](#embedding-models)).

## Supported methods

Uses the [PEFT library](https://github.com/huggingface/peft) (v0.18+). LoRA is the only method with tested
parallelism support; other PEFT methods (AdaLoRA, Prompt Tuning, (IA)3) reach the trainer through a custom
`peft_config` but are untested with it.

| Method | Enable | Notes |
|---|---|---|
| LoRA | `use_peft: true` | Default low-rank adaptation |
| QLoRA | LoRA + BitsAndBytes 4-bit/8-bit base | LoRA on a quantized base |
| DoRA | `use_dora: true` | Weight-decomposed (magnitude + direction). Attention only — **rejected** alongside expert projections, which have no grouped DoRA |
| rsLoRA | `use_rslora: true` | `alpha/sqrt(r)` scaling, for high ranks. Applies to attention *and* native expert adapters |

## Configuration

Fields come from TRL's `ModelConfig` and are converted to a `LoraConfig` via `get_peft_config()`:

```yaml
use_peft: true
lora_r: 64                    # Rank (TRL default 16)
lora_alpha: 128               # Scaling factor (alpha/r applied to adapter output)
lora_dropout: 0.05
lora_task_type: CAUSAL_LM     # CAUSAL_LM | SEQ_CLS | FEATURE_EXTRACTION
lora_target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
```

`lora_target_modules` is tri-state: unset (`None`) uses the architecture defaults, and a populated list
selects those modules. Expert-LoRA-only means listing **only** expert projections — the peel then empties
the list, which is what disables attention PEFT. Writing `lora_target_modules: []` yourself names nothing
at all and is rejected: with no adapter to create, the run would full-finetune at the LoRA learning rate.

A one-entry list is collapsed to a bare string by TRL's `ModelConfig`, so `lora_target_modules: [experts]`
arrives as `"experts"`. The expert peel handles that; PEFT does not (it reads a lone string as a regex
matched against full module paths), so an attention-only single target needs `all-linear` or two entries.

### Hyperparameters

- **Rank (`lora_r`):** 8–32 for 1B–3B, 32–96 for 7B–14B, 64–384 for 27B+. With a tuned learning rate,
  performance is largely rank-independent; rank matters most when the LR is suboptimal.
- **Alpha:** `alpha = rank` (conservative) or `2 * rank` (aggressive). For rank ≥ 64, prefer `use_rslora`.
- **Target modules:** all-linear for best quality; attention-only when memory-constrained. On MoE models
  `all-linear` does not reach the fused expert tensors — name the expert projections explicitly (see
  [MoE models](#moe-models-expert-targets-and-full-trained-modules)).
- **Learning rate:** LoRA needs 5–10× the full-FT rate.
- **Dropout:** honored on SFT. The preference, reward and offline-GRPO trainers run TRL's
  `disable_dropout_in_model` after the adapter wrap (`disable_dropout` defaults to `True`), zeroing PEFT's
  `lora_dropout` along with the model's own for reference-vs-policy determinism. A warning fires when a
  configured dropout was zeroed; `disable_dropout: false` keeps it.

| Trainer | Full-FT band | LoRA start |
|---|---|---|
| SFT | `2e-6` – `2e-5` | `1e-4` |
| SMPO / DPO | `5e-7` – `5e-6` | `5e-5` |
| GRPO | `1e-6` – `5e-6` | `1e-5` – `5e-5` |

The effective-batch formula is unchanged by LoRA — see
[SFT](../training-methods/sft.md#learning-rate-and-global-batch-size).

Ready configs: `examples/sft/qwen3/qwen3-4b-ultrachat-lora.yaml` (and its `-qlora` counterpart, which adds
`load_in_4bit: true`, `bnb_4bit_quant_type: nf4`, `use_bnb_nested_quant: true`).

## MoE models — expert targets and full-trained modules {#moe-models-expert-targets-and-full-trained-modules}

MoE experts are fused `nn.Parameter` tensors (`GptOssExperts.gate_up_proj` / `down_proj`, and the
equivalents on other families), not `nn.Linear`, so PEFT's `all-linear` cannot reach them; on gpt-oss it
resolves to the attention `q/k/v/o_proj` only. Name the expert projections explicitly instead.
`split_expert_lora_targets` (`src/distributed/loading/peft_setup.py`) peels those entries out of the list and
routes them to native grouped-LoRA built inside each EP layer (`_init_expert_lora` in
`src/distributed/expert_parallel/base_layer.py`); the remaining attention names stay on stock PEFT.

| `lora_target_modules` entry | Requests |
|---|---|
| `gate_proj` / `up_proj` / `down_proj` | that logical projection |
| `gate_proj_gmm` / `up_proj_gmm` | the same, in the grouped-GEMM storage spelling |
| `gate_up_proj` | the fused gate+up projection |
| `experts` / `mlp.experts` | all three expert projections |

Requests are **logical**; what gets built follows the family's storage. One adapter is created per stored
3-D expert tensor that any requested projection touches. On a family that stores the fused
`gate_up_proj` — every family except Qwen3 and Bailing — asking for `gate_proj` alone adapts the whole
`[E, H, 2M]` tensor: gate and up share one rank-`r` subspace. On the two separate-storage families it
adapts gate alone, with its own rank `r`.

GptOss picks its layout at runtime: with grouped-GEMM (SM90+, the default) it stores
`gate_proj_gmm`/`up_proj_gmm` separately, without it a fused interleaved tensor. The same YAML therefore
yields a different adapter, and resume across that switch is refused.

The peel is **name-based, not module-based**. `gate_proj`/`up_proj`/`down_proj` also spell the plain
`nn.Linear` MLPs of a hybrid MoE — dense prefix layers (`first_k_dense_replace`) and shared experts — and
those are adapted by neither half once the name is peeled. A warning names what moved.

Shared-expert LoRA has no supported route under EP: named generically it is peeled; named explicitly
(`shared_experts.down_proj`) PEFT wraps a Linear inside the EP layer and `_validate_lora_ep_compatibility`
rejects it.

Do not reach for TRL's `lora_target_parameters` on an MoE model — it is **rejected** on any model carrying
EP layers (`ep_size > 1`, or the default `use_grouped_gemm`), whatever it targets, so an attention-only run
is rejected too. It is PEFT's own mechanism for fused `nn.Parameter` experts, and it wraps the module
*owning* the parameter, which sits above the EP layer and so escapes both EP validators.

Weights PEFT can't LoRA are **full-trained** (a trainable copy, not a low-rank adapter) via
`lora_modules_to_save`: `embed_tokens` (an `nn.Embedding`), `lm_head` (excluded by `all-linear`), and the
MoE `router` / `gate`. An **expert-only** run builds no `PeftModel`, so `lora_modules_to_save` raises there
rather than leaving those modules silently frozen — add an attention target, or drop the field.

### Router training under EP

The router/gate is absorbed into the EP module for every family but Gemma4, whose router is a sibling module
the wrapper never owns. EP modules are FSDP-ignored, so their gradients are DP-averaged by a hook the EP
layer registers at construction on the *original* router param — one that never sees the trainable copy
`modules_to_save` creates. `EPMoELayerBase.reattach_router_grad_sync` re-attaches it to the live copy
automatically, so `router` in `lora_modules_to_save` stays DP-consistent in every EP mode; the ep1
FSDP-sharded path (`fsdp_shard_ep1_experts`) and the multi-node deferred sweep already cover it.

`modules_to_save` on any **other** EP-internal submodule is unsupported:
`_validate_ep_peft_trainable_params_synced` raises naming any trainable param left inside an EP module with
no gradient sync — it would otherwise silently drift across DP ranks. Its sibling
`_validate_lora_ep_compatibility` (`src/trainers/mixins/validation.py`) rejects stock-PEFT LoRA layers placed
inside EP modules.

### Example: gpt-oss LoRA under EP

```yaml
lora_target_modules:            # attention → PEFT; expert projections → native grouped-LoRA
- q_proj
- k_proj
- v_proj
- o_proj
- gate_up_proj                  # fused gate+up expert projection
- down_proj
lora_modules_to_save:           # full-trained (PEFT can't LoRA these)
- embed_tokens
- lm_head
- router                        #   DP-sync hook re-attached automatically under EP
```

`examples/grpo/environmental/gptoss/vllm/gptoss-20b-code-contests-lora-ep4.yaml` is a full
Environmental-GRPO version.

**Adapter dtype under FSDP:** `get_peft_model` upcasts adapters (and `modules_to_save` copies) to fp32, but
FSDP2 requires a uniform dtype per shard group. `_cast_peft_params_to_compute_dtype` casts trainable fp32
PEFT params to the compute dtype before wrapping (to fp32 instead under `fp32_non_ep_params`). EP params are
excluded by identity, before the dtype test: they are FSDP-ignored, so the one-dtype-per-group rule never
applies, and downcasting a deliberately-fp32 router or expert would negate `fp32_router` / `fp32_experts`.

## Parallelism compatibility

| Mode | LoRA | QLoRA | Target restriction | Notes |
|---|---|---|---|---|
| DDP / FSDP | Yes | Yes | None | Standard path |
| EP | Yes | No | Attention + experts | Attention via PEFT; expert FFNs via native grouped adapters. QLoRA rejected: EP loaders materialize plain de-quantized weights, losing `Params4bit` |
| CP | Yes | Yes* | None | CP shards the sequence, not the weights — adapters stay replicated; standard `from_pretrained` preserves `Params4bit`. *QLoRA only on a **dense** model: an MoE takes the grouped-GEMM loader, which rejects a quantized base |
| TP | **No** | No | — | Rejected — adapters are plain tensors outside the TP DTensor graph |
| EP+CP | Yes | No | Attention + experts | Both active |
| EP+TP | **No** | No | — | Both adapter kinds rejected: attention LoRA as under TP, native expert LoRA by the gate's `has_ep_lora` arm |
| ETP | Yes | No | Attention only | Expert adapters rejected at config time by `ParallelismConfig` (`expert_tp_size > 1` gives the replicated adapter half a partial, never-synced gradient) |
| PP | **No** | No | — | Attention PEFT rejected at trainer construction (a stage cannot resolve full-tree module names); expert LoRA rejected earlier by `ParallelismConfig` (the adapter save/merge paths would record stage-local layer indices) |

**EP.** The expert names [above](#moe-models-expert-targets-and-full-trained-modules) route to **native grouped
LoRA** — grouped `[E_local, K, r]`/`[E_local, r, N]` adapters stored alongside each expert weight, applied in
the grouped-GEMM compute and gradient-synced across the EP group like the experts themselves. `lora_B` is
zero-initialized, so the initial delta is zero. The frozen base experts stay bf16.

The default save writes a standalone adapter; `merge_expert_lora_on_save: true` folds the delta into the
base for a servable HF checkpoint instead ([Merging adapters](#merging-adapters)).

**TP / EP+TP.** For a colwise-sharded base, `lora_B` becomes a per-rank output shard while `lora_A` stays
replicated. Nothing broadcasts the replicated matrix (it diverges from init) and nothing distinguishes the
sharded one from a replica (the TP replicated-grad sync averages and corrupts it), so the trained adapter is
rank-inconsistent and will not reload onto a non-TP model. Use FSDP/DP, CP, or pure ETP instead.

The gate is `_validate_lora_tp_compatibility` (`src/trainers/mixins/validation.py`) and it refuses both
adapter kinds: a `PeftModel` or any adapter outside the EP layers, and — checked first, via `has_ep_lora` —
the native grouped expert adapters, which every other TP gate skips by param identity. `expert_tp_size > 1`
rejects expert LoRA earlier still, at config time in `ParallelismConfig`, before the checkpoint downloads.

## Measured cost

Dense — 1× B300 (SM103), `DistributedSFTTrainer`, AdamWBF16, Liger, FA4, Qwen3-8B, seq 16384, BS=1, GC,
10 steps / 3 warmup:

| Config | Trainable | tokens/s/GPU | Peak memory |
|---|---|---|---|
| Full fine-tuning | 8,191M (100%) | 17,342 | 64.6 GB |
| LoRA r=64, attn only | 61M (0.7%) | 16,824 | 34.6 GB |
| LoRA r=64, all linear | 175M (2.1%) | 11,717 | 35.9 GB |
| QLoRA r=64, all linear (NF4 base) | 175M (3.6%) | 15,802 | 25.3 GB |

Attention-only nearly matches full-FT throughput; all-linear is ~32% slower, since adapter matmuls run on
every MLP layer. Throughput is **rank-invariant** within a variant (attn-only ~16.8k, all-linear ~11.9k
across r=16/64/128) — the frozen base forward/backward dominates the step, and only memory grows with rank.
QLoRA saves ~10 GB more than bf16 all-linear LoRA and is ~33% faster, because the 4-bit base cuts weight
bandwidth on the bandwidth-bound MLP matmuls; it is the path onto consumer GPUs, since plain LoRA needs
~34 GB even at the minimum rank.

MoE — same setup on 8× B300, `gpt-oss-20b` (32 experts, top_k=4) at EP=2, seq 4096, r=64:

| Config | Trainable | tokens/s/GPU | Peak memory |
|---|---|---|---|
| Full fine-tuning | 11,388M (100%) | 7,896 | 77.3 GB |
| LoRA r=64, attn only (PEFT) | 32M (0.28%) | 9,632 | 28.4 GB |
| LoRA r=64, experts only (grouped) | 425M (3.60%) | 9,919 | 32.0 GB |
| LoRA r=64, attn + experts | 457M (3.86%) | 7,586 | 32.0 GB |

**LoRA under EP is faster *and* leaner than full fine-tuning.** The frozen base experts carry no optimizer
state and skip the EP gradient all-to-all, so attention-only and experts-only both run ~1.25× full-FT
throughput at ~⅓ the memory — experts-only ties attention-only because the grouped expert adapters fold into
the grouped-GEMM compute. Attn + experts is slower than either alone, on par with full FT. On
`qwen3-30b-a3b` (128 experts) experts-only r=64 is 9.39% trainable at 5,034 tok/s/GPU and 46.8 GB. At batch 1
the step is communication-bound, so tok/s/GPU varies ±10% run-to-run.

## Reference model handling

Offline GRPO and Environmental GRPO disable the adapter (`disable_adapter()`) to compute reference log-probs
from the frozen base instead of a second model copy. Under EP the mixin patches that context
(`make_disable_adapter_ep_aware`) so it reverts the native EP expert adapters too, giving a true frozen-base
reference for both adapter halves; the patch also covers TRL's `use_adapter(None)` for online GRPO / DPO / KTO.

A reference pass must open that context **after** some other forward has unsharded the FSDP2 parameters —
both trainers do, behind the policy forward and behind the no-grad log-prob recompute respectively. peft
clears `requires_grad` on the adapter tensors a module holds at entry and restores it on the ones it holds at
exit; the forward in between swaps them, and this toolkit's `reshard_after_forward=False` keeps the transient
unsharded copies registered afterwards. Open the context first and the restore lands on those copies, leaving
every sharded adapter frozen — measured: the next training step then raises, with no `grad_fn` on the loss or,
under gradient checkpointing, a recompute-metadata mismatch.

An **explicit** `ref_model` is rejected under EP and TP (it is never parallelized, so its log-probs would not
match the policy's): use LoRA with `ref_model=None`, or `precompute_ref_log_probs=True`. Under TP, LoRA is
rejected too, so DPO/KTO there must precompute. SMPO is reference-free.

Where no adapter wraps the model — a full fine-tune, or an expert-only LoRA run, which builds no
`PeftModel` — TRL builds its own reference model when `beta != 0`: an unparallelized replica per rank.
`_validate_implicit_reference_model` warns about that under EP, and **raises** whenever the policy carries
live attention sinks (`reset_sinks: false`), where the two models would compute different log-probs for
identical tokens. Set `beta: 0`, `use_peft: true`, or precompute.

## Online RL — rollout-server weight sync

Online and Environmental GRPO generate rollouts from a separate rollout server — vLLM or SGLang
([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)) — which serves the **plain base
model** with no adapter. Before each NCCL weight sync the trainer merges the adapter into the base, forwards
the merged weights under base-model param names (PEFT prefixes stripped, `lora_*` params skipped), then
unmerges to keep training. Without the merge, the server would generate from the un-adapted base. Both
trainers share this path (`gather_and_send_weights` in `src/trainers/grpo/rollout/weight_sync.py`), which
also folds in the EP / FSDP2 / TP gathers; the merge is a collective on all ranks under FSDP2.

## Embedding models

The embedding trainer is SentenceTransformer-based. `use_peft: true` injects LoRA into the underlying
transformer via `peft.inject_adapter_in_model` (not `SentenceTransformer.add_adapter`) and freezes every
non-adapter param. Plain LoRA only — 4-bit QLoRA is rejected on the ST loader. Runs under standard / FSDP2
data parallelism only; EP and TP are rejected at trainer construction (the EP save path has no adapter-merge
step, so the checkpoint would carry adapter keys that reload as random base weights). In-place
injection never reads `lora_task_type`.

## Quantized training (QLoRA)

QLoRA combines LoRA with 4-bit/8-bit base quantization under DDP/FSDP, and under CP on a dense model. CP
keeps the standard `from_pretrained` loader, which preserves `Params4bit`. AdamWBF16 optimizes only the bf16
adapter params; the frozen quantized base is skipped.

`load_distributed_model` raises for EP, TP, PP, and the grouped-GEMM MoE loader — pure ETP included, since
the gate reads `ep_group_size = ep_size × expert_tp_size`. Those loaders materialize plain de-quantized
weights, so `Params4bit` are lost and PEFT's 4-bit adapter dispatch fails (PP rejects PEFT outright anyway).

The 4-bit **compute** dtype follows the run's own precision (`bf16`/`fp16` on the training config), not
TRL's `ModelConfig.dtype` — whose `"float32"` default nothing else here reads, and which would otherwise
dequantize and compute every 4-bit matmul in fp32.

QLoRA is SFT/offline territory: the online and environmental GRPO trainers reject a quantized base at
construction (`validate_weight_sync_support`). The NCCL weight sync forwards raw parameter storage under
base-weight names, so a bnb-packed 4-bit base would ship non-floating-point tensors that corrupt the served
policy, and a per-sync merge/unmerge round-trip through 4-bit weights is lossy. Plain LoRA is the supported
RL adapter path — see [Online GRPO](../training-methods/grpo/online-grpo.md#lora).

Under `torchrun` data parallelism the 4-bit base cannot be FSDP2-sharded (`fully_shard` rejects the non-float
`Params4bit`), so the trainer skips FSDP2 and all-reduces the (small) adapter gradients across ranks, with
the frozen base replicated per rank. Nothing wraps the model, so the FSDP2 sharding knobs cannot take
effect: a non-default `use_hsdp`, `fsdp_reshard_after_forward` or `fsdp_reshard_after_backward` is rejected
at trainer construction rather than silently ignored, naming the offending flags — QLoRA under CP takes the
same gate. Under `accelerate launch`, the other supported QLoRA launcher, accelerate owns the wrap and those
knobs only warn.

Pass a `BitsAndBytesConfig` through `load_distributed_model(..., quantization_config=bnb_config)`, or pass a
pre-loaded quantized model plus `peft_config` to the trainer. For 8-bit optimizer states without a 4-bit
base, set `optim: adamw_8bit`.

### FusedLinearCrossEntropy for long sequences

[Liger FLCE](liger-kernels.md) fuses `lm_head` with the loss instead of materializing the
`batch × seq × vocab` logits. Qwen3-8B QLoRA r=32 all-linear at 32k (BS=1, GC, Liger, FA4) needs **43.5 GB
with plain CE and 20.9 GB with FLCE**, so it fits a 24 GB consumer GPU only with FLCE. FLCE is SFT-only and
disables entropy logging.

## Checkpoint saving

`save_model()` saves only adapter weights (~MB). `PeftAdapterSaver` (`src/distributed/checkpoint/peft.py`)
handles every mixin-managed mode. Two routes bypass it: an accelerate-managed FSDP run, where the base
Trainer's save owns the sharded layout, and `merge_expert_lora_on_save`, which goes through the gathered EP
save instead. Per mode:

- **EP:** attention adapters are replicated (rank 0 saves); native expert adapters are rank-local and
  gathered across the EP group into the same adapter file. An **expert-only** run has no PEFT model at
  all — `save_ep_checkpoint` writes the adapter with a synthesized config instead: `peft_type:
  EXPERT_LORA`, the base model path, and every field the delta depends on (`r`, `lora_alpha`,
  `lora_dropout`, `use_rslora`, `expert_projections`). (Pure ETP carries attention adapters only —
  expert LoRA raises at `expert_tp_size > 1`.)
- **CP:** adapter keys normalized to strip CP wrapper paths (`.original_attention.`, extra `.model.`) so they
  load onto non-CP models; DTensor reconstruction also applied.
- **FSDP2:** DTensor adapter params reconstructed via `full_tensor()` (a collective) before save; global rank
  0 saves on a shared FS, rank 0 per node otherwise.
- **EP+CP:** combines EP rank selection with CP key normalization.

**Reloading elsewhere.** Attention adapters carry standard PEFT keys and load onto a stock
`PeftModel.from_pretrained`. Native expert adapters do not — grouped `[E, K, r]` tensors have no PEFT
representation, and PEFT drops unrecognized adapter keys without a warning (`load_adapter` filters
`missing_keys` and never reads `unexpected_keys`). Every directory carrying them is therefore labeled with a
`peft_type` PEFT does not know, so an external load raises on the label rather than returning a half-adapted
model: `EXPERT_LORA` for an expert-only save, `LORA_WITH_EP_EXPERT_LORA` for a mixed attention+expert one.
The mixed label keeps every `LoraConfig` field beside it, plus an `ep_expert_lora` block recording the expert
half's `r`/`lora_alpha`/`lora_dropout`/`use_rslora`/`expert_projections`; resume reads the tensors directly
and never parses the file. The repo's merge and convert scripts check those markers
(`assert_no_expert_lora_adapter`) and fall through to a tensor-key scan when the config is absent or carries
a stock `peft_type`, so an unmarked directory holding `.experts.<attr>.lora_{A,B}` keys is refused too.

**Resume:** EP/CP rebuild the base with zero-initialized adapters at init, so trained adapters are restored
from the checkpoint's `adapter_model.safetensors` (not from the base reload) by
`restore_adapters` (`src/distributed/checkpoint/peft.py`), which `CheckpointLoader` calls. Resuming expert adapters into a run that does not build them (EP off,
`use_grouped_gemm: false`, or the expert projections dropped from `lora_target_modules`) raises rather than
discarding them.

Under CP the saved (normalized) keys are mapped back onto live wrapper spelling by
`remap_cp_adapter_keys_to_live`, through the shared `strip_peft_adapter_segment`, which drops the
adapter-name segment in all three spellings PEFT uses (`lora_A.default.weight`,
`modules_to_save.default.weight`, and the trailing `ParameterDict` form `lora_embedding_A.default`). Both
adapter loaders drop an unmatched key as unexpected, so a spelling it missed would resume that adapter at
initialization, silently. See [Checkpoints](../reference/checkpoints.md#resume-by-parallelism-mode).

## Merging adapters

```bash
python scripts/after_training/merge_peft_adapters.py \
    --adapter_dir path/to/adapter/checkpoint \
    --output_dir path/to/merged/model
```

Add `--task classification --num_labels N` for classification models. Handles causal LM, sequence
classification, multimodal/VLM, and trust_remote_code architectures. Use `--device_map auto` for large
models.

The merge re-applies the run's training sidecars to the merged model and copies both files into the output:
the GptOss sinks policy recorded in `training_provenance.json`, and `router_balancing_biases.pt` into the
family's native routing-bias slots, so the merged model routes as it trained.

The script **refuses native EP expert-LoRA adapters** (`peft_type` of `EXPERT_LORA` or
`LORA_WITH_EP_EXPERT_LORA`, or tensor keys ending at `.experts.<attr>.lora_{A,B}`), because
`PeftModel.merge_and_unload` silently drops the grouped expert deltas. A stock PEFT adapter on a per-expert
`nn.Linear` MoE keys as `.experts.<i>.<proj>.lora_A.default.weight` and merges normally; only the grouped
namespace is refused.

`merge_expert_lora_on_save: true` produces the merged servable checkpoint at training time instead, for
expert-only **and** mixed runs alike — without it a mixed adapter is resumable by this toolkit but foldable
by no tool, since saving the adapters and merging afterwards hits exactly the refusal above. It needs native
grouped expert adapters to exist: `lora_target_modules` must name at least one expert projection, or
`_validate_merge_expert_lora_save` raises. Both halves are folded — expert deltas inside each family's
`gather_expert_state_dict`, attention deltas via a `merge_adapter` held across the write and unmerged after,
so training continues unchanged. The fold happens in the gathered EP save under mixin-managed FSDP2
(torchrun); it is rejected under accelerate-managed FSDP and with `save_sharded_ep: true`.

## Tests and benchmarks

```bash
# LoRA + CP (trains) and LoRA + TP (asserts rejection)
torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_cp_tp.py
# LoRA + EP with MoE
torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_ep.py

# Adapters on an on-policy run, asserted on the SERVED policy (needs a live vLLM server):
# attention LoRA under EP and pure ETP, native grouped expert LoRA under EP
torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_online_grpo_vllm_moe_e2e.py \
    --trainer online --mode expert_lora_ep2
# attention LoRA under FSDP2 DP, and the TP refusal at trainer construction
torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_online_grpo_vllm_dense_e2e.py \
    --trainer online --mode lora_tp2_rejected

# Dense LoRA/QLoRA benchmark
torchrun --nproc_per_node=1 tests/gpu/profiling/benchmark_sft_dense.py \
    --model qwen3-8b --seq 16384 --lora_r 64 --lora_all_linear --qlora
# Expert-targeting LoRA under EP
torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_sft_ep.py \
    --ep 2 --seq 4096 --lora_r 64 --lora_experts
```

Omit `--lora_all_linear` for attention-only; on the EP benchmark it means attention + experts, while
`--lora_experts` is the native grouped adapters alone.

## Related pages

- [Configuration Reference](../reference/configuration-reference.md) — full `lora_*` / `use_peft` parameters
- [Expert Parallelism](../parallelism/expert-parallelism.md) · [Tensor Parallelism](../parallelism/tensor-parallelism.md)
- [BF16 Optimizer](bf16-optimizer.md) — composes with LoRA
- [LoRA](https://arxiv.org/abs/2106.09685) · [QLoRA](https://arxiv.org/abs/2305.14314) · [DoRA](https://arxiv.org/abs/2402.09353)
