# BF16 Master Weights with Stochastic Rounding

`AdamWBF16` keeps both model weights and optimizer states in bf16 (6 B/param), matching pure-bf16 memory while avoiding the "stale weights" problem via stochastic rounding (SR). A fused Triton kernel makes the step faster than `adamw_torch_fused`. Owning file: `src/optimizers/adamw_bf16.py`; auto-enable lives in `src/trainers/mixins/base.py`. What stays fp32 in a bf16 run, and where master weights fit: [GPU Training Theory §7](../reference/gpu-training-theory.md#bf16-numerics-what-stays-fp32).

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
- **`exp_avg_sq` (second moment) — SR.** A non-negative accumulator near the bf16 underflow floor; nearest rounding there biases the running variance upward (measured ~+50% at small gradients), inflating `sqrt(v)` and silently shrinking the effective step below `lr`. SR keeps it unbiased.

The kernel computes both EMAs and the weight update in fp32, truncating only on store-back, so the update always sees the exact fp32 second moment.

SR seeds come from a dedicated, rank-synchronized RNG (`_SR_RNG`, one per optimizer module) kept separate from the global `random` module, so nothing in the data path can desync the noise across ranks. Replicas that hold the same parameter and receive the same averaged gradient — HSDP `dp_replicate` groups, DDP — round it identically and stay bit-for-bit in sync.

## Benchmarks

**GPT-OSS-20B MoE (24 layers, 32 experts, 20.7B params, ~14B trainable with first 8 layers frozen), single B300 (Blackwell, SM100):** AdamWBF16 (Triton) steps in 51.6 ms vs 62.1 ms for `adamw_torch_fused` (**−17%**), at identical 134.2 GB peak and identical bf16 (4B) state dtype. The kernel is faster because it fuses state EMA + weight update + SR into one memory pass (14 B/element) and draws both SR noise streams from one `tl.randint4x` Philox call. This compares two bf16-state optimizers; the 6-vs-12 B/param memory win is vs fp32-state AdamW, not visible in this row.

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
optim: adamw_torch  # adamw_torch / adamw_torch_fused both auto-enable AdamWBF16 under bf16: true; a non-adamw optim (e.g. adamw_bnb_8bit) suppresses it
```

`bf16_optimizer` (a `DistributedArguments` field, so every training script parses it) overrides that
resolution: `true` forces AdamWBF16 on where the auto path would decline (a non-AdamW `optim`, replicated
DDP), `false` forces full fp32 master weights.

`false` is rejected only where the run mixes plain-tensor experts with FSDP2 DTensors — `ep_group_size`
(`ep_size × expert_tp_size`) above 1, or `ep_group_size == 1` with `fsdp_shard_ep1_experts: false` — and the
raise lands when the optimizer is built, not at config time. Dense runs and MoE at
`ep_size == expert_tp_size == 1` with the default `fsdp_shard_ep1_experts: true` are allowed.
Combining `bf16_optimizer: true` with `optim: muon` or `optim: flash_adamw` **raises**: both select an
optimizer and the bf16 path would silently win, so pick one.

Direct use: `AdamWBF16(model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-8)`, with the standard HF decay / no-decay param groups. Pass `use_triton=False` for the eager PyTorch fallback (functionally equivalent, slower — multiple memory passes instead of the one fused kernel; it also engages automatically without CUDA). The eager path seeds its SR noise from the same rank-synchronized `_SR_RNG`, so replica bit-identity holds there too.

## Master-weight and grad-reduce options

AdamWBF16 auto-detects dtype per param: bf16 params take the fused Triton SR path, fp32 params take standard in-place AdamW (DTensor-aware). It composes with `fp32_router` / `fp32_experts` (those params stay fp32 and get standard AdamW) and with `fp32_non_ep_params`.

`fp32_non_ep_params: true` moves non-expert (dense) params to fp32 storage (12 B/param) for exact updates; experts stay bf16 + SR. Use it for headroom on large-vocab `lm_head`/embeddings. On a MoE at `ep_size: 1` it needs `fsdp_shard_ep1_experts: false` — otherwise the FSDP-managed experts would stay bf16 inside an fp32 shard group and `ParallelismConfig` refuses the pair ([Precision control](../parallelism/expert-parallelism.md#precision-control)).

**Measured (gpt-oss-20b EP=2, seq 4096, batch 4, 8× B300, FA4):**

| master-weight regime | tok/s/GPU | peak mem | notes |
|----------------------|:---------:|:--------:|-------|
| **full bf16** (AdamWBF16, default) | 17,875 | 91.4 GB | 6 B/param; production path |
| `fp32_non_ep_params` | 17,409 | 92.7 GB | non-EP params fp32, experts bf16; +1 GB only (experts dominate, stay bf16) |
| `+ fp32_grad_reduce` | 16,117 | 92.7 GB | bf16 master, fp32 grad reduction (~−9%: 2× bandwidth on the grad all-reduce) |
| **full fp32** (`bf16_optimizer=False`) | — | — | **rejected at `ep_group_size > 1`, and at `ep_group_size == 1` with `fsdp_shard_ep1_experts: false`** — fused AdamW cannot mix the plain-tensor expert FFN (EP rank-local experts, or the grouped-GEMM `gate_proj_gmm`/`up_proj_gmm` split at ep1) with FSDP2 DTensors (`aten._fused_adamw_ got mixed torch.Tensor and DTensor`); raised when the optimizer is built (still before the first step) |

Full fp32 master (`bf16_optimizer=False`) is supported on dense models and on `ep_group_size == 1` MoE with the default FSDP-sharded experts. The fp32 deltas are small for gpt-oss because its non-expert params are a minor fraction; high-vocab or attention-heavy models cost more.

`fp32_grad_reduce: true` upcasts gradients to fp32 for every cross-rank reduction the mixin owns (FSDP2 `reduce_dtype=fp32` for dense params, the EP router/expert grad-sync hooks, the TP replicated-grad sync, the QLoRA adapter AllReduce), then stores the averaged result bf16. It keeps bf16 master weights (6 B/param) — unlike `fp32_non_ep_params` it changes only the reduction, not storage.

**Under pipeline parallelism** ([not yet available in this release](../parallelism/pipeline-parallelism.md)) it would also change storage: a pipeline schedule turns FSDP gradient sync off for the whole microbatch loop and reduce-scatters once per optimizer step, so each stage would hold a full *unsharded* fp32 gradient (4 B/param) for that loop instead of a bf16 one, on top of the unsharded params `fsdp_reshard_after_forward=False` already pins — `2×P_stage` (params) + `4×P_stage` (grads) per rank, not shrinking with DP width. `ParallelismConfig` warns when the two are combined.

NCCL's bf16 all-reduce accumulates in bf16, so error grows with world size. On 8× B300 with real bf16 gradients, fp32 reduce is ~2.2× tighter (0.17% vs 0.37% of signal), and the gap widens with scale. The cost is ~2× bandwidth on that collective only.

The bucketed sweeps (`reduce_grads_bucketed` — the deferred cross-replica EP sweep, the TP replicated / per-head-norm sweep, the QLoRA sweep) hold **3× `HALO_GRAD_BUCKET_MB` per in-flight bucket** under `fp32_grad_reduce`, not 1×: each chunk keeps its bf16 flat buffer alive for the scatter-back alongside the fp32 upcast it reduces. At the defaults (256 MB, 2 in flight) that is ~1.5 GB allocated post-backward, while every gradient is still live — the worst memory moment of the step, and exactly the multi-node EP path where 100–400B models run. Halve `HALO_GRAD_BUCKET_MB` there if the step OOMs at the sweep.

Default `false`. `fp32_non_ep_params` implies it only for the FSDP2 reduce dtype — the EP router/expert, TP and QLoRA hooks read `fp32_grad_reduce` directly, so set it explicitly to cover them. Enable for many-rank / multi-node runs. It applies only on the mixin-managed (torchrun) path — an `accelerate launch` that manages FSDP/DDP itself takes its reduce dtype from accelerate's plugin, and the trainer warns that the knob is unapplied.

## Compatibility

Every distributed trainer supports `bf16_optimizer` — resolution lives in `DistributedTrainerMixin._configure_mixed_precision`, which all of them run — and it auto-enables under FSDP, EP, TP, CP and their combinations (rank-local experts and per-rank DTensor shards alike). The one exception is accelerate-managed replicated DDP, where the auto-enable is skipped as a conservative default outside the validated FSDP/EP/TP/HSDP matrix, not a correctness limit: SR is replica-safe (the rank-synchronized RNG rounds shared params identically), so `bf16_optimizer: true` opts in.

Checkpoints use standard PyTorch `state_dict()` / `load_state_dict()`, round-tripping all state in bf16.

## Tests

```bash
# Unit tests (single GPU, no torchrun)
CUDA_VISIBLE_DEVICES=0 python tests/gpu/optimizers/test_adamw_bf16.py

# EP end-to-end + SR correctness under EP=8 (8 GPUs)
torchrun --nproc_per_node=8 \
    tests/gpu/optimizers/test_bf16_optimizer_ep.py

# Scale benchmark on the 20B MoE (single GPU, ~140 GB; --freeze-layers N trims it)
CUDA_VISIBLE_DEVICES=0 python tests/gpu/optimizers/bench_adamw_bf16.py
```

`test_adamw_bf16.py` covers the single-GPU claims — SR statistics (weight and `exp_avg_sq`, both paths), mixed dtypes, decay groups, state-dict round-trip, and that every shipped optimizer advances `param._version` across a step (the low-precision weight cache keys on it). `test_bf16_optimizer_ep.py` runs full + LoRA training under EP=8 and asserts the two distributed SR claims: `exp_avg_sq` on a local expert shard tracks an fp32 reference (de-bias), and the SR-rounded weight is bit-identical across the EP replicate group.

## References

- [Stochastic Rounding for BF16 Training](https://arxiv.org/abs/2010.06192)
- [DeepSeek V3 Technical Report](https://arxiv.org/abs/2412.19437) — bf16 optimizer states at 671B scale
