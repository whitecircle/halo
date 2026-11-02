# BF16 Master Weights with Stochastic Rounding

`AdamWBF16` keeps both model weights and optimizer states in bf16 (6 B/param), matching pure-bf16 memory while avoiding the "stale weights" problem via stochastic rounding (SR). A fused Triton kernel makes the step faster than `adamw_torch_fused`. Owning files: `halo/optimizers/adamw_bf16.py` (`AdamWBF16`, and `adamw_bf16_optimizer_from_args`, which builds it from the training args) and `halo/optimizers/stochastic_rounding.py` (the SR seeds and the bf16 rounding step it shares with Muon); auto-enable lives in `halo/trainers/optimizer_selection.py`. What stays fp32 in a bf16 run, and where master weights fit: [GPU Training Theory §7](../reference/gpu-training-theory.md#bf16-numerics-what-stays-fp32).

## The problem: stale weights in bf16

BF16 has ~0.8% relative precision (7-bit mantissa). When the update `lr * adam_step` is smaller than this threshold relative to the weight magnitude, nearest rounding truncates it to zero, and ~67% of parameters stop updating after the first few steps. The standard fix is fp32 master weights (cast to bf16 for forward/backward), but it doubles optimizer memory:

| Approach | Weight | States | Total/param |
|---|---|---|---|
| Pure bf16 (67% stale) | 2B | 4B bf16 | **6B** |
| FP32 master weights | 4B fp32 | 8B fp32 | **12B** |
| **bf16 + SR (AdamWBF16)** | **2B bf16** | **4B bf16** | **6B** |

At 20B params, that is 120 GB vs 240 GB of optimizer memory.

## Stochastic rounding

SR rounds up or down probabilistically based on the truncated bits (carry probability = `dropped_bits / 2^16`), making the rounding error unbiased so small updates accumulate over many steps.

The two moments round differently:

- **`exp_avg` (first moment) — nearest rounding.** Signed, ~zero-mean EMA, so truncation is already unbiased; SR would only add noise.
- **`exp_avg_sq` (second moment) — SR.** A non-negative accumulator near the bf16 underflow floor; nearest rounding there biases the running variance by tens of percent (the sign depends on the gradient regime; `tests/cpu/optimizers/test_optimizer_sr.py` pins a >15% bias against fp32), which moves `sqrt(v)` and the effective step away from `lr`. SR keeps it unbiased.

The kernel computes both EMAs and the weight update in fp32, truncating only on store-back, so the update always sees the exact fp32 second moment.

On sm_100 and sm_103 (B200, B300) one launch steps up to 256 bf16 parameters that share a device, dtypes and launch specialization, so a step costs a few launches whatever the parameter count; a launch goes out as soon as its 256 are collected, while the host builds the rest. On any other GPU, Hopper included, each launch steps one parameter, a table of one that compiles to a lone tensor's per-element code. A per-step table, copied to the device pinned and non-blocking, holds one row per tensor: its local-shard offsets, element count, scalars, seed and first program. Each program steps up to 16 consecutive 1024-element blocks of one tensor: it finds that tensor by binary search over the first-program row (8 probes at the cap), loads its row once, and runs the per-block code of a lone launch for each block, with the block's index within the tensor keying the Philox counter. So every element draws the noise a launch over its tensor alone would, and the result is bit-identical to stepping each tensor in its own launch. A block costs about the instructions a lone launch's program does, and the kernel is capped at that kernel's 32 registers, so an SM holds as many programs. The specialization is what Triton's launcher would give the tensor alone: whether its element count is a multiple of 16, and which of its four local tensors start 16-byte aligned. The launch hints exactly that, so each tensor compiles to its lone launch's per-element code, with the same vector widths and the same register layout. The compiled FMA association of an element depends on its register, so a launch hinted with any other alignment can change the element's last bit. Most LLM parameters and FSDP2 shards are aligned throughout; an FSDP2 grad view shifted off the boundary keeps its parameter and moments vectorized. The step refuses a local shard that is not contiguous, which the kernel's flat indexing would misread.

**Gradient clipping inside the step.** `defer_grad_scale(coef)` hands the step a device scalar that the kernel multiplies into each gradient as it reads it. The EP clip (`GradientClippingMixin._install_ep_clip`) uses it whenever the optimizer exposes the hook and its parameters with gradients are exactly the clipped parameters with gradients, instead of rescaling every gradient with `_foreach_mul_` (whose fp32-scalar-on-bf16 form runs one unvectorized kernel per parameter). The gradients stay unscaled until the step; `zero_grad` drops a scale no step consumed. So anything reading `.grad` between the clip and the step (a `TrainerCallback.on_pre_optimizer_step`, which Hugging Face documents as running after clipping) sees the unclipped gradients; the reported `grad_norm` is the pre-clip norm either way.

**Skipping a step on the device.** `skip_next_step_if(flag)` (`DeviceStepSkipMixin`, `halo/optimizers/step_skip.py`) hands the next step a 0-dim device bool. Set, the fused kernel's programs step no block and the eager bf16 path keeps every old value; the fp32 path carries the flag in its own ops (decay factors of one, an infinite `eps`, the gradient selected to -0.0: one select per parameter), so its weights and moments keep their values (a -0.0 weight can turn +0.0). The step's host bookkeeping (`state["step"]`, first-step state) still moves until `settle_step(skipped)` rolls it back, and drops the copy of the old state whether the step was skipped or not. Unset, the step writes what a step without the flag does, bit for bit. The [non-finite step guard](../reference/trainer-architecture.md#non-finite-steps) skips this way, so it never waits for its verdict.

SR seeds are drawn from no generator: `sr_seed_pair` (`halo/optimizers/stochastic_rounding.py`) hashes the parameter's optimizer step (`state["step"]`) and its name (`sr_param_index`, over the `param_names` each group carries) under a per-optimizer key. Replicas that hold the same parameter and receive the same averaged gradient — HSDP `dp_replicate` groups, DDP, EP replica groups, expert-TP partners — round it identically and stay bit-for-bit in sync, even where the ranks' optimizers hold different parameter lists (GPT-OSS keeps an expert's `down_proj_bias` on expert-TP rank 0 only). The optimizer refuses param groups built without names. A resumed run restores `step` with the optimizer state, so it rounds exactly as the uninterrupted run would have; there is no generator state to checkpoint, and the zero-LR step torch runs to materialize state before a restore consumes nothing. An element's noise is keyed by its offset in the rank's local shard, so it reproduces for one sharding layout (the one an optimizer-shard resume must keep), not across layouts.

## Benchmarks

**GPT-OSS-20B MoE (24 layers, 32 experts, 20.7B params, ~14B trainable with first 8 layers frozen), single B300 (SM103), Blackwell image:** AdamWBF16 (Triton) steps in 31.8 ms vs 62.0 ms for `adamw_torch_fused` (**−49%**), at identical 134.2 GB peak and identical bf16 (4B) state dtype.

The kernel is faster because it fuses state EMA + weight update + SR into one memory pass (14 B/element). Each lane owns four consecutive elements and one `tl.randint4x` Philox call, whose 4 × 32 bits give every element its two independent 16-bit SR draws: 6.5 TB/s effective against 3.9 TB/s with one Philox call per element, on a 254M-element expert tensor (one B300). This row compares two bf16-state optimizers; the 6-vs-12 B/param memory win is against fp32-state AdamW and is not visible here.

**Loss quality (4-layer FFN, 50M params, 200 steps):** AdamWBF16 (SR on weight + `exp_avg_sq`) reaches a gap to fp32-master of ~0.00008 — about 5× tighter than weight-only SR (~0.0004), because the unbiased second moment keeps the effective LR at nominal `lr`. Pure bf16 (fused) lags by 0.27.

## Usage

AdamWBF16 is auto-enabled when `bf16: true` and `optim` is the default AdamW. No separate flag, no special launch flags:

```bash
torchrun --nproc_per_node=8 \
    scripts/training/sft.py \
    examples/sft/gptoss/gptoss-20b-multinode-ep.yaml \
    --expert_parallel_size=8
```

```yaml
bf16: true
learning_rate: 1e-4
adam_beta2: 0.999
optim: adamw_torch  # adamw_torch / adamw_torch_fused both auto-enable AdamWBF16 under bf16: true; any other optim (e.g. adamw_bnb_8bit) suppresses it
```

`bf16_optimizer` (a `ParallelismArguments` field, so every training script parses it) overrides that
resolution: `true` forces AdamWBF16 on where the auto path would decline (an `optim` other than
`adamw_torch` / `adamw_torch_fused` — except `muon` and `flash_adamw`, which raise, below — and replicated DDP), `false` forces the stock AdamW over the parameters
as loaded. Under `bf16: true` those are bf16, so `false` keeps bf16 master weights and moments with
round-to-nearest updates (the stall above), not fp32 ones; fp32 masters come from `fp32_non_ep_params`
(non-expert params), `fp32_router` and `fp32_experts` (EP routers and experts), or `bf16: false`.

`false` is rejected where the run mixes plain-tensor experts with FSDP2 DTensors: `ep_group_size` (`ep_size × expert_tp_size`) above 1, or `ep_group_size == 1` with `fsdp_shard_ep1_experts: false`, unless `fp32_non_ep_params` is set, which routes to a per-tensor-type grouped AdamW with fp32 masters on the non-expert params. The raise lands when the optimizer is built, not at config time. Dense runs and MoE at `ep_size == expert_tp_size == 1` with the default `fsdp_shard_ep1_experts: true` are allowed.

Combining `bf16_optimizer: true` with `optim: muon` or `optim: flash_adamw` **raises**: both select an optimizer and the bf16 path would silently win, so pick one.

Direct use: `AdamWBF16(model.named_parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)`, or the named decay / no-decay groups `decay_groups` builds (`halo/optimizers/param_groups.py`). Pass `use_triton=False` for the eager PyTorch fallback (functionally equivalent, slower — multiple memory passes instead of the one fused kernel; it also engages automatically without CUDA). The eager path seeds its SR noise from the same `sr_seed_pair`, so replica bit-identity and exact resume hold there too.

## Master-weight and grad-reduce options

AdamWBF16 auto-detects dtype per param: bf16 params take the fused Triton SR path, fp32 params take standard in-place AdamW (DTensor-aware). It composes with `fp32_router` / `fp32_experts` (those params stay fp32 and get standard AdamW) and with `fp32_non_ep_params`.

`fp32_non_ep_params: true` moves non-expert (dense) params to fp32 storage (12 B/param) for exact updates; experts stay bf16 + SR. Use it for headroom on large-vocab `lm_head`/embeddings. On a MoE at `ep_size: 1` it needs `fsdp_shard_ep1_experts: false` — otherwise the FSDP-managed experts would stay bf16 inside an fp32 shard group and `ParallelismConfig` refuses the pair ([Precision control](../parallelism/expert-parallelism.md#precision-control)).

**Measured (gpt-oss-20b EP=2, seq 4096, batch 4, 8× B300, FA4, Blackwell image):**

| master-weight regime | tok/s/GPU | peak mem | notes |
|----------------------|:---------:|:--------:|-------|
| **full bf16** (AdamWBF16, default) | 20,554 | 85.2 GiB | 6 B/param; production path |
| `fp32_non_ep_params` | 19,993 | 86.6 GiB | non-EP params fp32, experts bf16; +1.3 GiB only (experts dominate, stay bf16) |
| `fp32_non_ep_params` + `fp32_grad_reduce` | 18,741 | 86.5 GiB | fp32 grad reduction on top of the row above (−6%: 2× bandwidth on the grad all-reduce) |
| **stock AdamW** (`bf16_optimizer=False`) | — | — | **rejected at `ep_group_size > 1`, and at `ep_group_size == 1` with `fsdp_shard_ep1_experts: false`** — fused AdamW cannot mix the plain-tensor expert FFN (EP rank-local experts, or the grouped-GEMM `gate_proj_gmm`/`up_proj_gmm` split at ep1) with FSDP2 DTensors (`aten._fused_adamw_ got mixed torch.Tensor and DTensor`); raised when the optimizer is built (still before the first step) |

`bf16_optimizer=False` builds on dense models and on `ep_group_size == 1` MoE with the default FSDP-sharded experts. The `fp32_non_ep_params` delta is small for gpt-oss because its non-expert params are a minor fraction; high-vocab or attention-heavy models cost more.

`fp32_grad_reduce: true` upcasts gradients to fp32 for every cross-rank reduction the mixin owns (FSDP2 `reduce_dtype=fp32` for dense params, the EP router grad-sync hooks and deferred sweep, the TP replicated-grad sync, the QLoRA adapter AllReduce), then stores the averaged result bf16. It keeps bf16 master weights (6 B/param) — unlike `fp32_non_ep_params` it changes only the reduction, not storage. An fp32 reduction is the TorchTitan, Megatron and DeepSpeed default.


NCCL's bf16 all-reduce accumulates in bf16, so error grows with world size. On 8× B300 with real bf16 gradients, fp32 reduce is ~2.2× tighter (0.17% vs 0.37% of signal), and the gap widens with scale. The cost is ~2× bandwidth on that collective only.

The bucketed sweeps (`reduce_grads_bucketed`: the deferred cross-replica EP sweep, the TP replicated / per-head-norm sweep, the QLoRA sweep) hold **3× `HALO_GRAD_BUCKET_MB` per in-flight bucket** under `fp32_grad_reduce`, not 1×. Each chunk keeps its bf16 flat buffer alive for the scatter-back alongside the fp32 upcast it reduces.

At the defaults (256 MB, 2 in flight) that is ~1.5 GB allocated post-backward, while every gradient is still live. That is the worst memory moment of the step, and exactly the multi-node EP path where 100–400B models run. Halve `HALO_GRAD_BUCKET_MB` there if the step OOMs at the sweep.

Default `false`. `fp32_non_ep_params` implies it only for the FSDP2 reduce dtype — the EP router hooks and deferred sweep, the TP and QLoRA sweeps read `fp32_grad_reduce` directly, so set it explicitly to cover them. Enable for many-rank / multi-node runs. It applies only on the mixin-managed (torchrun) path — an `accelerate launch` that manages FSDP/DDP itself takes its reduce dtype from accelerate's plugin, and the trainer warns that the knob is unapplied.

## Compatibility

Every distributed trainer supports `bf16_optimizer`; resolution lives in `resolve_bf16_optimizer` (`halo/trainers/optimizer_selection.py`), which every trainer's construction runs. It auto-enables under FSDP, EP, TP, CP and their combinations (rank-local experts and per-rank DTensor shards alike).

The one exception is accelerate-managed replicated DDP, where the auto-enable is skipped as a conservative default outside the validated FSDP/EP/TP/HSDP matrix, not a correctness limit. SR is replica-safe (the step-and-name seed rounds shared params identically), so `bf16_optimizer: true` opts in.

Checkpoints use standard PyTorch `state_dict()` / `load_state_dict()`, round-tripping all state in bf16.

## Tests

```bash
# Unit tests (single GPU)
torchrun --nproc_per_node=1 tests/gpu/optimizers/test_adamw_bf16.py

# Fused launch vs per-tensor launches, bit for bit, incl. FSDP2 shards, resume and Qwen3-8B scale (2 GPUs)
torchrun --nproc_per_node=2 tests/gpu/optimizers/test_adamw_bf16_fused.py

# Launch tables, the kernel comparison under Triton's interpreter, the per-element code compiled for sm_90, sm_100 and sm_103 (CPU)
python tests/cpu/optimizers/test_adamw_bf16_fused_launch.py

# The device skip flag on every path of AdamWBF16 and Muon, interpreted for the Triton ones (CPU)
python tests/cpu/optimizers/test_device_step_skip.py

# EP end-to-end + SR correctness under EP=8 (8 GPUs)
torchrun --nproc_per_node=8 \
    tests/gpu/optimizers/test_bf16_optimizer_ep.py

# Scale benchmark on the 20B MoE (single GPU, ~140 GB; --freeze-layers N trims it)
CUDA_VISIBLE_DEVICES=0 python tests/gpu/optimizers/bench_adamw_bf16.py
```

`test_adamw_bf16.py` covers the single-GPU claims: SR statistics (weight and `exp_avg_sq`, both paths), mixed dtypes, decay groups, state-dict round-trip, a restored AdamWBF16 and Muon rounding bit-identically to the uninterrupted run, and that every shipped optimizer advances `param._version` across a step (the low-precision weight cache keys on it).

`test_bf16_optimizer_ep.py` runs full + LoRA training under EP=8 and asserts the two distributed SR claims: `exp_avg_sq` on a local expert shard tracks an fp32 reference (de-bias), and the SR-rounded weight is bit-identical across the EP replicate group.

## References

- [Stochastic Rounding for BF16 Training](https://arxiv.org/abs/2010.06192)
- [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437) — bf16 optimizer states at 671B scale
