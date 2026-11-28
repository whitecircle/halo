# Optimization

Memory reduction, throughput, and hardware utilization during training. The [GPU training theory](../reference/gpu-training-theory.md) page explains why each lever works; the pages below toggle them.

## Foundations

- [GPU training theory](../reference/gpu-training-theory.md): the roofline, the limits a training step hits on one GPU and across many, and why each kernel exists; each number is marked as measured on one B300, a specification or computed.
- [Throughput benchmarks](throughput-benchmarks.md): 1.2.0 against 1.1.0 on twelve shapes, the comparison with TRL, NeMo AutoModel and Megatron Bridge, GRPO against Miles, then tokens/s/GPU and achieved TFLOPS on B300 across parallelism modes, sequence lengths, and model sizes.
- [Halo vs stock TRL](halo-vs-stock-trl.md): Halo 1.2.0 against `trl.SFTTrainer` 1.17.1 at matched model, rows and kernels, covering layouts, ratios, TRL's expert parallelism and gpt-oss long context.

## Memory

- [PEFT (LoRA)](peft.md): train a few percent of parameters at about half the memory, across EP and CP (attention-only under ETP; rejected under TP, EP+TP and PP), with merge-back to a standalone checkpoint.
- [BF16 optimizer](bf16-optimizer.md): AdamW with bf16 masters and stochastic rounding at 6 bytes per parameter. Auto-enabled with `bf16: true` on FSDP/EP/TP.
- [FlashAdamW](flash-adamw.md): quantized 8-bit states and 24-bit masters (about 5 bytes per parameter) with AdamW-matching convergence. Set `optim: flash_adamw`.
- [Low-precision compute (FP8/FP4)](low-precision-moe-kernels.md): FP8/FP4 matmul over bf16 masters and checkpoints, as a QAT oracle plus a block-scaled weight export (`scripts/after_training/quantize_to_lowp.py`). The same page holds the fused dequantization of the frozen NVFP4/MXFP8 expert banks Inkling 2 trains over ([Frozen expert banks](low-precision-moe-kernels.md#frozen-expert-banks)).
- [Allocator](throughput-benchmarks.md#allocator-expandable-segments): `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` lowers peak reserved memory on dense runs at unchanged throughput; the dense SFT, DPO and SMPO examples launch with it, an EP run sets it only to fit.

![Memory before and after a lever, each row its own run: fsdp_shard_ep1_experts 148.3 to 60.3 GiB on gpt-oss-20b ep1, the length-ordered deal 210.25 to 183.35 GiB on an async GRPO row, the CP fused head loss 53.30 to 46.99 GiB on GLM-4.7-Flash, and expandable segments 13–14% off reserved memory on dense SFT, DPO and SMPO](../assets/benchmarks/memory_levers.png)

## Throughput

- [Padding-free collator](padding-free-collator.md): concatenates sequences into Flash-Attention blocks instead of padding, with boundary masking against cross-sample leakage.
- [MTP heads](mtp-heads.md): train the multi-token-prediction head a family's release ships, jointly, detached or over a frozen trunk, with a hard or a soft target. Set `mtp_loss_weight`, `mtp_train_scope` and `mtp_loss_type`.
- [Tree packing](tree-packing.md): rows that share a prefix store it once and branch after it; each example trains as it does packed flat, at fewer tokens.
- [Flash attention](flash-attention.md): auto-detected per architecture, with FA4 on Blackwell (B200/B300), FA3 on Hopper (H100/H200), FA2 on Ampere and Ada, and SDPA on an older GPU.
- [Masked matmul attention](flash-attention.md#masked-matmul-attention): GLM-5 Next's sparse-attention layers on a tiled matmul with a closed-form backward, about 5× over a forward, its recompute and a backward, on by default on Blackwell. Set `HALO_DSA_MATMUL=0` to leave them on stock SDPA.
- [Block-sparse QSA attention](flash-attention.md#block-sparse-attention-qwen38-flash-next): Qwen3.8-Flash-Next's sparse-attention layers on a runtime-length Triton kernel over the selected blocks, one compile for every pack length. `HALO_QSA_KERNEL=auto` by default; `off` keeps the tiled path.
- [Inkling 2 relative-logits attention](../models/inkling2.md#architecture): Inkling 2's attention on a compiled FlexAttention kernel that applies the learned relative logits and the sinks. A full-attention layer's forward and backward at 8,192 tokens take 41 ms against 166 ms eager on one B300. On by default on CUDA; `attn_implementation: eager` keeps the eager path.
- [Hyper-connection kernels](hyper-connection-kernels.md): GLM-5 Next's and DeepSeek-V4's hyper-connections and mixes on one launch per step, at about a sixth of the time. Set `HALO_HC_KERNELS=0` to leave upstream's forwards. The same page holds Qwen3.8-Flash-Next's gated residual, on by default; `HALO_GATED_RESIDUAL_KERNELS=0` leaves upstream's forward.
- [Grouped GEMM](grouped-gemm.md): batched MoE expert matmul via `torch.nn.functional.grouped_mm`, default on SM90+. Home to the atomic-free permute and MoE throughput playbook.
- [Liger kernels](liger-kernels.md): Triton kernels for a cross-entropy that keeps the logits, RMS normalization, SwiGLU and RoPE (Qwen3 MoE, GLM-4.7-Flash, GPT-OSS, Gemma 4, Nemotron-H and Inkling 2 run their norms on torch's fused `F.rms_norm`). On by default; set `use_liger_kernel: false` to disable.
- [torch.compile](torch-compile.md): Inductor compilation of the spans between MoE/attention graph breaks (`torch_compile: true`), at most 1% on EP MoE and slower in `reduce-overhead` mode.
- [Checkpointing plan](../training-methods/callbacks.md#gradientcheckpointingplancallback): `gradient_checkpointing_plan: auto` measures one fully checkpointed step and checkpoints only the decoder layers the memory budget needs (opt-in, packed SFT rows).
- [Muon optimizer](muon-optimizer.md): Newton-Schulz orthogonalization for matrix parameters, with fused Triton momentum and update kernels. Set `optim: muon`.
- [NUMA binding](throughput-benchmarks.md#numa-binding): `torchrun --numa-binding node` runs each rank on its GPU's NUMA node, which helps overhead-bound rows (TP, small-batch EP) on a multi-socket host. `torchrun` launches only.

## Levers that do not help at these shapes

Each of these ran on B300 and did not pay off; its page holds the shapes.

- Halo's simulated fp8/fp4 path serves quantization-aware training and its export, and bf16 stays the default. Its whole steps at ep8 ran at 0.10–0.45× bf16 on B300 with the eager activation round trip, and simulated mxfp4 with the fused one runs at 0.93–0.99× on H200 ([Why bf16 is the default](low-precision-moe-kernels.md#why-bf16-is-the-default)).
- The DeepGEMM adapter ran whole steps at 0.37–0.54× bf16, since casts and row copies cost more than the kernel saves ([DeepGEMM](low-precision-moe-kernels.md#deepgemm-the-native-fp8fp4-path)).
- `torch.compile` on EP MoE gains at most 1%, and `reduce-overhead` is slower ([Benchmark results](torch-compile.md#benchmark-results)).
- Master weights never go below bf16, so Halo has no sub-bf16 master; bf16 with stochastic rounding is the floor ([BF16 optimizer](bf16-optimizer.md)).
- Where FA4 runs the model, FA2 on Blackwell is the slow outlier against FA4 and SDPA at head_dim 128 ([FA4 vs FA2 vs SDPA](flash-attention.md#fa4-vs-fa2-vs-sdpa-on-blackwell)). The defaults land on FA2 on purpose for GLM-4.7-Flash, Gemma 2/3 at head_dim 256 and softcapped rows that reach FA4's varlen kernels; leave those on it. Packed head_dim-256 GatedDeltaNet rows (Qwen3.5/3.6, Qwen3-Next) default to FA2 as well, where FA4 gains under 5% ([Model-specific handling](flash-attention.md#model-specific-handling)).
- MoE throughput measured at batch 1 is noisy from run to run; measure at batch ≥ 4 ([Liger benchmarks](liger-kernels.md#benchmarks)).
