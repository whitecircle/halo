# FlashAdamW (Quantized Optimizer States)

FlashAdamW is a drop-in replacement for AdamW that cuts per-parameter optimizer memory ~57% via 8-bit quantized Adam moments and a 24-bit master weight, with quantize/dequantize fused into the update by Triton kernels. Convergence matches AdamW.

Per-param state: AdamW 12 B (4B exp_avg + 4B exp_avg_sq + 4B master, all fp32) vs FlashAdamW ~5 B (1B int8 exp_avg + 1B int8 exp_avg_sq + 3B master = bf16 param as the high bits + an 8-bit low-bit correction). At 20B params: 240 GB vs ~100 GB optimizer state.

## Usage

```yaml
optim: flash_adamw
learning_rate: 1e-4
weight_decay: 0.01
```

```bash
torchrun --nproc_per_node=8 \
    scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

Both images install `flashoptim` directly (`Dockerfile`, pinned, `--no-deps` so it cannot re-resolve torch and downgrade NCCL), so `optim: flash_adamw` works out of the box. On a bare host, install the extra (resolves from PyPI):

```bash
uv pip install -e ".[flash-optimizers]"   # or: pip install flashoptim
```

Without the package, `optim: flash_adamw` raises an import error stating what to install.

The YAML/trainer path (`build_flash_adamw_optimizer` in `src/optimizers/flash_adamw.py`) wires `learning_rate`, `weight_decay`, `betas` (from `adam_beta1`/`adam_beta2`), and `eps` (from `adam_epsilon`). Only `master_weight_bits` is not read from `TrainingArguments`, so over YAML it stays at its default. To override it, construct directly: `create_flash_adamw_optimizer(model, lr=..., weight_decay=..., master_weight_bits=24)` in `src/optimizers/flash_adamw.py`. The wrapper splits parameters into weight-decay groups (`decay_parameters` get `weight_decay`, the rest `0.0`).

`create_flash_adamw_optimizer` defaults: `lr=1e-3`, `betas=(0.9, 0.999)`, `eps=1e-8`, `weight_decay=0.01`, `master_weight_bits=24` (24 = bf16 + 8-bit correction, 32 = full fp32, `None` = no master weights).

## Benchmark

The value is memory, and it scales with model size — tens of GB at 70B+ params, where it is the lever.

Optimizer micro-benchmark on B300 (`tests/gpu/optimizers/bench_muon.py --hidden 4096 --layers 8`, synthetic FFN): FlashAdamW peak memory is **−8.6%** vs AdamW (fused), at a ~5 ms optimizer-step overhead from state quant/dequant. For the AdamW / AdamWBF16 / Muon comparison at the default shape, see [Muon](muon-optimizer.md#benchmark).

That overhead amortizes against fwd+bwd on most models, but it is not free — on an attention-bound long-context step the per-step cost can outweigh the memory saving (GLM-4.7-Flash EP8 32k: ~50 s/step vs ~34 s/step for fused AdamW).

A real-model benchmark lives in `tests/gpu/optimizers/bench_muon_qwen3_5.py` (Qwen3.5-2B). Correctness is covered by `tests/gpu/optimizers/test_flash_adamw.py` — creation, loss descent, decay filtering, memory savings, and a bit-exact state-dict round-trip.

## When to use which optimizer

| Optimizer | Memory | Convergence | Best for |
|---|---|---|---|
| `adamw_torch_fused` | 12B/param | Baseline | Default, unlimited memory |
| AdamWBF16 (auto) | 6B/param | Same as FP32 | Standard bf16 training |
| `flash_adamw` | ~5B/param | Same as AdamW | Maximum memory savings |
| `muon` | ~4–6B/param | Faster convergence | When convergence speed matters — see [Muon](muon-optimizer.md#benchmark) |

## Compatibility

`optim: flash_adamw` routes through `create_optimizer` in `src/trainers/mixins/base.py` for every distributed trainer, the same path as Muon and AdamWBF16, and composes with gradient checkpointing and FSDP2. `flashoptim` steps per parameter and unwraps DTensors to their local shard before the kernel runs, so a weight-decay group mixing FSDP2/TP DTensors with the plain rank-local tensors EP and `fp32_non_ep_params` produce is handled, not rejected — `examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml` ships this combination at `expert_parallel_size: 4`.

Nothing gates `flash_adamw` on a parallelism mode. The only guard is the shared one in `_configure_mixed_precision`: `bf16_optimizer: true` together with a named `optim` raises rather than silently discarding one of them. Correctness coverage is single-GPU (`tests/gpu/manifest.py` registers `test_flash_adamw.py` at `nproc=1`).

The wrapper registers a step post-hook that advances each stepped param's version counter — `flashoptim`'s fused Triton update stores through raw pointers ATen never sees, and the low-precision weight-quant cache keys on that counter (see [Low-Precision MoE](low-precision-moe-kernels.md)).

**Checkpoint limit: unevenly sharded params.** `flashoptim`'s `state_dict` refuses any FSDP2 DTensor param whose sharded dim does not divide the mesh, and the trainer's all-or-nothing shard save then skips optimizer state on every checkpoint — resume warm-restarts the optimizer. Training itself is unaffected. `_warn_if_uneven_shards` (`src/optimizers/flash_adamw.py`) says so at optimizer build; pick dims divisible by the shard world or another `optim` if exact optimizer resume matters.

## References

- `flashoptim` — Databricks' implementation; `pyproject.toml` allows `>=0.1.3,<0.2.0`, the images pin `0.1.4`
- [8-Bit Optimizer Paper](https://arxiv.org/abs/2110.02861) — dynamic quantization for Adam states
