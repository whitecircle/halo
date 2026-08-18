# Muon Optimizer (Newton-Schulz Orthogonalization)

Muon projects each gradient matrix onto the nearest orthogonal matrix via Newton-Schulz iteration and uses that as the update direction, reaching a lower loss than Adam in the same step budget on matrix-shaped parameters. Fused Triton kernels do the momentum/nesterov and weight-decay/update math per parameter; the orthogonalization itself runs the Newton-Schulz kernel (quack/cutlass, or a pure-torch fallback). Owning file: `src/optimizers/muon.py`.

## Usage

```yaml
optim: muon
learning_rate: 3e-4
weight_decay: 0.1
```

```bash
torchrun --nproc_per_node=8 \
    scripts/training/sft.py examples/sft/qwen3/qwen3-4b-ultrachat.yaml
```

`optim: muon` routes through `build_muon_optimizer` (`src/optimizers/muon.py`), which forwards `learning_rate`, `weight_decay`, `adam_epsilon`, and the decay-parameter set.

`adam_beta1` / `adam_beta2` reach Muon's internal scalar AdamW only when the config sets them to something other than the HF default; Muon's tuned `DEFAULT_SCALAR_BETAS = (0.9, 0.95)` (`src/optimizers/muon.py`) otherwise stands, since HF's `adam_beta2` default of `0.999` would silently retune the scalar optimizer of every Muon config that never mentions betas. Everything else takes the `create_muon_optimizer` default: `momentum=0.95`, `nesterov=True`, `scalar_lr=lr`, `ns_algorithm="gram_newton_schulz"`, `ns_use_kernels=True`. To change those, call `create_muon_optimizer` directly:

```python
from src.optimizers.muon import create_muon_optimizer
opt = create_muon_optimizer(model, lr=3e-4, weight_decay=0.1, ns_algorithm="standard_newton_schulz")
```

## Parameter splitting

`create_muon_optimizer` splits parameters by shape and module role:

- Hidden 2D+ weight matrices (Linear, conv) → Muon with Newton-Schulz.
- Everything below 2D (biases, norms) and the token embedding / `lm_head` → an internal scalar optimizer (`scalar_lr`, betas `(0.9, 0.95)`).

The embedding and head are 2D but excluded: their rows index the vocabulary, where Newton-Schulz is ill-suited and empirically harmful. They are identified from the module tree the model declares — `get_input_embeddings()` / `get_output_embeddings()` plus a pooled `score`/`classifier` head — so a family spelling its embedding `tok_embeddings` or its head `output` is routed correctly. `_is_embedding_or_head`'s name-marker list is a second, always-active exclusion on top of the accessors, catching models that expose none.

The scalar optimizer is [AdamWBF16](bf16-optimizer.md), not `torch.optim.AdamW`: Muon requires bf16 hidden weights, so the scalar params are bf16 too and a nearest-rounding write would truncate every sub-ULP update to zero (norm weights, value ~1 and ULP 3.9e-3, would never move). Being per-parameter, it also steps a mix of sharded DTensors and plain tensors under EP/TP, which a foreach/fused AdamW rejects.

Memory per param: ~4 B for 2D (bf16 weight + bf16 momentum), ~6 B for 1D (bf16 weight + bf16 AdamW states), plus transient Newton-Schulz buffers. Same-shape params are stacked for the NS kernel in chunks of 32 (`_GNS_CHUNK_SIZE`), which bounds the stacked NS input — not the whole transient: a bf16 update buffer for every param plus the per-shape concat of all NS outputs still materialize, tens of GB extra on a 100B+ MoE. Budget headroom for the step, and expect the same transient once on resume (the restore primes state with a zero-LR step).

**Muon-routed weights must be bf16.** Construction raises when any 2D+ param it would orthogonalize is not
bf16, naming the offender — so the fp32 master-weight knobs (`fp32_non_ep_params`, `fp32_router`,
`fp32_experts`) are incompatible with `optim: muon`. It also raises if the model has no 2D+ parameter at all.
`state_dict`/`load_state_dict` are overridden to merge the scalar optimizer's moments into the standard
per-param index space — upstream Muon serializes only its own state, which would reset them on resume.
Setting `bf16_optimizer: true` alongside `optim: muon` raises — both select an optimizer.

## Stochastic rounding on the bf16 weight write

`_weight_decay_update_kernel` computes the update in fp32 and rounds the new weight to bf16 with stochastic rounding (same technique as [AdamWBF16](bf16-optimizer.md)). The orthogonalized update has magnitude ~`lr` after RMS-norm scaling, which at typical fine-tune LRs is below the bf16 weight ULP — nearest rounding would truncate the entire update to zero. SR keeps the write unbiased so small updates accumulate. SR seeds come from a dedicated RNG seeded identically on every rank, so replicas stay bit-identical.

## Newton-Schulz kernel backend

The orthogonalization matmuls can run on the `quack-kernels` / `nvidia-cutlass-dsl` CUDA backend (`ns_use_kernels`, on by default). `create_muon_optimizer` probes the kernel on a tiny tensor at construction; if the call raises — missing package, import error, or ABI mismatch — it warns and falls back to a numerically-equivalent pure-torch path that is slower.

The `Dockerfile` force-installs the ABI-compatible pair (`quack-kernels==0.5.0`, `nvidia-cutlass-dsl==4.5.2`) over `uv.lock`'s `0.4.1` / `4.4.2`, so the probe passes and the kernel path is the active one in the images. A bare host on the lockfile versions takes the fallback.

## Benchmark

Optimizer micro-benchmark on B300 (`tests/gpu/optimizers/bench_muon.py`, synthetic FFN, median of 20 timed steps), isolating the optimizer step:

| Optimizer | Optimizer step | Peak mem vs AdamW | Best loss (40 steps) |
|---|:---:|:---:|:---:|
| AdamW (fused) | 7.4 ms | — | 0.13 |
| AdamWBF16 (SR) | 7.0 ms | −0.0% | 0.13 |
| Muon (Gram–Newton–Schulz) | 102 ms | +68% | 0.10 |
| Muon (standard NS) | 144 ms | +68% | 0.098 |

Muon reaches the lowest loss in the step budget at the cost of a far more expensive optimizer step. This FFN is worst-case: it is all matrix params, so the Newton-Schulz cost is unamortized. On a real transformer, Newton-Schulz runs on smaller per-layer matrices amortized over a larger fwd+bwd, so the end-to-end slowdown is far smaller — measure on your model. The fused Triton kernels replace upstream Muon's per-batch Python loop + `torch.compile`, holding the Muon step at ~200 ms on Qwen3.5-2B versus ~1000 ms for the upstream path.

Run `tests/gpu/optimizers/bench_muon.py` or `tests/gpu/optimizers/bench_muon_qwen3_5.py` (real Qwen3.5-2B) for tables on your hardware.

## Compatibility

Works with FSDP2, gradient checkpointing, and all parallelism modes — the kernels operate on the local shard after `to_local` (`src/distributed/runtime.py`) unwraps DTensors.

That implies an approximation: Newton-Schulz orthogonalization runs on each rank's local dim-0 shard, not the full matrix, so the Muon update depends on the sharding layout and world size — the same model sharded differently takes a different update. Shard-local NS is the accepted trade-off, keeping the step communication-free, not equivalence with single-GPU Muon; re-tune Muon hyperparameters when the world size changes materially. 3D expert tensors are exempt: they shard on the expert dim, so each expert matrix orthogonalizes whole, and the approximation bites only dense 2D weights under FSDP2/TP.

Two further consequences: NS normalizes its input, so `max_grad_norm` clipping constrains only the scalar/embedding AdamW leg, not the Muon-routed 2D updates; and the NS-kernel availability probe is an all-ranks-agree reduction, so a rank that cannot JIT the kernels drops every rank to the pure-torch path — replica updates stay bitwise identical.

## References

- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) — weight decay + per-parameter update-scale adjustment for Muon at scale
- [gram-newton-schulz](https://github.com/Dao-AILab/gram-newton-schulz) — upstream implementation (git pin in `pyproject.toml`)
