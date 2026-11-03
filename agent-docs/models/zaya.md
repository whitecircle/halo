# Zaya (Zyphra/ZAYA1)

`ZayaForCausalLM` — Zyphra's 8.84B-parameter MoE (16 routed experts, top-1; ~760M active) with CCA (Compressed Convolutional Attention), cross-layer EDA (Evolving Decision Adapter) router state, and a discard-bucket routing slot. Transformers ships `transformers.models.zaya` natively, so the family loads with no remote code and no revision pin. The most constrained model in Halo.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). Every mode runs without gradient checkpointing ([Limitations](#limitations)).

40 `ZayaDecoderLayer`, each one CCA attention plus a `ZayaSparseMoeBlock` — every layer is MoE. `layer_types` picks `hybrid` or `hybrid_sliding` per layer, which selects the mask, the per-layer `sliding_window` kernel argument, and the RoPE parameters. `rope_parameters` is a per-layer-type dict; a flat spelling is stored as-is and dies at model build with a bare `KeyError`.

`num_experts_per_tok` must be `1` — the config raises otherwise, as it does for a `hybrid_sliding` entry with no `sliding_window`. Config defaults: hidden 2048, `moe_intermediate_size` 2048, `router_hidden_size` 256, vocab 262272, 8 attention heads / 2 KV heads, head_dim 128, `cca_time0` = `cca_time1` = 2, `tie_word_embeddings: true`.

FSDP2 needs `cast_forward_inputs=False` in its mixed-precision policy — Zaya carries an fp32 inter-layer residual, and the default bf16 cast at the layer boundary mismatches the recompute dtype. `_should_cast_forward_inputs` (`halo/distributed/fsdp.py`) reads the `_fp32_interlayer_residual` class attribute the load patch stamps on the Zaya modeling class — no model-type list.

## Checkpoint format

Hub `Zyphra/ZAYA1-8B` `main` is the native format: 40 layers, fused `model.layers.{N}.mlp.experts.gate_up_proj` `[E, 2M, H]` and `down_proj` `[E, H, M]`. Load it directly.

`Zyphra/ZAYA1-8B-legacy` preserves the pre-native layout — 80 alternating layers, per-expert `zaya_block.experts.local_experts.{i}.linear_fc{1,2}` — which transformers **cannot** read. Derive any vocab-patched training base from `main` instead (`scripts/before_training/patch_vocab.py`).

## EP layer

`EPZayaMoELayer` (`halo/distributed/expert_parallel/families/zaya.py`) wraps `ZayaSparseMoeBlock` (router + experts).

- Routing: `ZayaRouter` with cross-layer EDA state — layer N's router output feeds layer N+1's. The forward returns `(expert_output, prev_router_hidden_states)` with the EDA state **attached**, as `ZayaSparseMoeBlock` does.

    That state is the only gradient path from layer N+1's routing loss to layer N's `router.down_proj` and `router_states_scale`, so detaching it silently trains the EP path on a different router objective than plain FSDP2. A severed edge moves those gradients by ~1e-2 relative, against the ~1e-7 floor the fp32 dispatch boundary leaves. Pinned in `tests/cpu/parallelism/test_zaya_ep_eda_gradient.py`.

- **Discard slot**: the router emits `num_experts + 1` logits, the extra one a learned "send to nowhere" bucket. `ZayaRouter.forward` masks tokens routed to it (weight → 0, index → 0) before returning. Under EP (`ep_size > 1`) the EP layer dispatches those zero-weight picks as `-1`, DeepEP's "no expert", so they never ride the all-to-all to expert 0's rank; at ep1 nothing is dispatched and the upstream masking stands.
- Topology: top-1 only, enforced by the config.
- Storage: fused `gate_up_proj [E, H, 2M]` and `down_proj [E, M, H]` in matmul convention (the checkpoint is `[E, 2M, H]` / `[E, H, M]`, transposed on load). SwiGLU, Grouped GEMM compute.
- Loading and saving both use the base fused path: lazy safetensors loading is supported (each rank reads only its expert slice), and the gathered save emits the two native fused tensors per layer, which `from_pretrained` reads back. The loader's structural probe declines a per-expert checkpoint in the `-legacy` layout.
- Routing replay is unsupported (`supports_routing_replay=False` on `ZAYA`) — the EDA state makes a replayed forward non-reproducible.

vLLM 0.32.0 ships no **native** Zaya implementation. An exported checkpoint still resolves through vLLM's transformers backend, whose generation quality for this family the toolkit does not validate.

RL weight sync is refused at construction on both engines, each for its own loader gap: vLLM has no native served model for the broadcast to land in, and SGLang 0.5.22's `zaya` loader reads the pre-transformers-5.14 per-expert checkpoint (`zaya_block.experts.local_experts.N.linear_fc1`), not the native fused layout the trainer holds ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

## Limitations

**Gradient checkpointing** is unavailable in every mode. Upstream declares support, but `patch_zaya_gradient_checkpointing_refusal` (`halo/models/zaya.py`) clears `ZayaPreTrainedModel.supports_gradient_checkpointing`, so `gradient_checkpointing_enable` raises rather than failing deep in the first backward. The underlying fault is environmental: on the CUDA 13.2 image, backward recompute through CCA's `nn.Conv1d` pair (`conv_qk_depthwise`, `conv_qk_grouped`) hits `cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`.

**EP + per-layer GC** is refused separately, by `ZAYA.supports_gradient_checkpointing=False`: the cross-layer `prev_router_hidden_states` plus a per-layer checkpoint re-wrap produce a polynomial backward recompute cascade.

**TP** — two blockers: `num_key_value_heads=2` means only `tp_size=2` divides evenly, and CCA replaces QKV with `q_proj` / `k_proj` / `v_proj_current` / `v_proj_delayed` plus the `nn.Conv1d` stack, which has no DTensor sharding primitive. Upstream ships no `base_model_tp_plan` at all, so the toolkit's raise is the only clear rejection. EP already gives 16× scaling on the dominant expert parameters, so there is no TP wrapper.

**CP** — the CCA convolutions run over the sequence axis and the delayed `v_proj_delayed` shifts each token's value to the previous timestep. Both break Ulysses partitioning at chunk boundaries: the conv receptive field crosses them with no handshake, and the delay would pull a sequence element from another rank.

**PP** — refused (`ZAYA_PP`): the fp32 residual stream and the accumulating EDA router state cross every layer boundary, a two-tensor, mixed-dtype boundary the single-activation pipeline contract does not carry ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md#model-support)).

**ETP** is supported via the shared fused-GLU helper: gate/up halves store as separate shards at `expert_tp_size > 1` so each rank holds matching intermediate positions, while the router (owning the EDA state) and the discard-slot masking stay replicated and FSDP-managed. EP+ETP without GC carries the same constraints as plain EP.

## Document isolation under packing

Every mixer is isolated, on every backend. `ZayaModel.forward` declares `position_ids` but never forwards them to the layers, so `patch_zaya_packed_document_isolation` (`halo/models/zaya.py`) stashes them on each attention module and each CCA projection for the duration of the model forward:

- **Attention**: the dense backends build a packed mask from `position_ids` at the model level; on flash the patch's registry view re-injects them into the flash interface, without which a packed row runs as one causal sequence.
- **CCA**: upstream runs the q/k conv pair (depthwise then grouped, kernel 2 each) as two `nn.Conv1d` over the whole row and shifts the delayed value by one token, so a document's first tokens read the previous document's last ones. The patched projection folds the two conv weights into one grouped map per input shift, applies each to the q/k states shifted by that much with the tokens before the document start zeroed, and resets the delayed value to its row-start state at every document start: each document reads as it would alone. On a single-document row it is upstream's arithmetic (equal in fp64, values and gradients); in bf16 it replaces the upstream conv with three small GEMMs at the same error against fp64 (6.8e-3 relative on a ZAYA1-8B projection, as the upstream conv, both measured on CPU), and keeps the two shifted q/k copies for backward: about 3 KiB more activation per token per layer than the upstream conv pair (7.9 against 5 KiB).

Cached decoding keeps upstream's path; a cached prefill takes the patched one and writes the same conv and recurrent state. An adapter on `conv_qk_depthwise` / `conv_qk_grouped` would be bypassed, so the patched forward raises on one.

## Router balancing

`ZayaForCausalLM` computes no aux loss. `ZayaRouter` instead carries a persistent fp32 `balancing_biases` buffer of width `num_experts + 1`, added to the **detached** gating probabilities before top-1 selection, so the bias steers which expert is chosen while the routed weight (and its gradient) comes from the unbiased path. Being persistent, it round-trips with every checkpoint. Without a periodic update the biases stay at init.

`moe_balancing: auto` resolves to `bias_update` off that buffer, and [`BalancingBiasCallback`](../training-methods/callbacks.md#balancingbiascallback) applies the DeepSeek-V3 auxiliary-loss-free update `b_i ← b_i + γ·sign(mean_count − count_i)`.

`count_i` is tokens routed to expert *i* over the last optimizer step, summed across DP/EP ranks. The gate masks discarded picks to index 0 with zero weight, and the load recorder re-points them to the counter's last slot, so discards inflate neither expert 0 nor the real-expert mean. `mean_count` averages over the real experts, the discard-slot bias is clamped, and γ is sign-only and tiny.

The counts come from `patch_zaya_router_load_recording` (`halo/models/zaya.py`), which scatter-adds the router's own indices inside `ZayaRouter.forward` — one mechanism for EP and plain FSDP2, since both keep the gate intact.

```yaml
moe_balancing: auto              # resolves to bias_update for Zaya
router_balancing_rate: 1.0e-3    # gamma; only used when bias_update is active
```

## Kernels

- **Attention** — FA2/FA3/FA4 all supported. The shipped configs leave it unset, so the loader picks FA4 on Blackwell and FA3 on Hopper; packed documents stay isolated on each label ([Document isolation under packing](#document-isolation-under-packing)).
- **Grouped GEMM** (`F.grouped_mm`) — default on SM90+; disable with `use_grouped_gemm: false`.
- **Liger** — upstream has no `zaya` applier; the toolkit covers it from its own spec (`halo/model_loading/liger/family_specs.py`). It defaults to the fused loss plus Liger RMSNorm; CE is off when the fused loss is on.

    The fused loss skips the `[B*S, 262272]` logits plane (~17 GB at batch 1, `S=32k`), which fits `b=1,S=32k` / `b=2,S=16k`. RoPE is not swapped (partial rotary: `rope_parameters` carries `partial_rotary_factor: 0.5` per layer type), and no GLU is swapped (the EP layer replaces the whole MoE block).

- **`torch.compile`** — untested, default off; expect graph breaks at DeepEP dispatch boundaries.

## Configs

| Config | EP | Notes |
|---|:--:|---|
| `examples/sft/zaya/zaya-1-8b-ultrachat.yaml` | — | Plain FSDP2, 16K packed rows |
| `examples/sft/zaya/zaya-1-8b-ultrachat-ep.yaml` | 8 (16 experts → 2/rank) | 16K packed rows |

All of them set `gradient_checkpointing: false` and `output_router_logits: false`, and pack 16,384-token rows at `gradient_accumulation_steps: 4` (65,536 tokens per rank per optimizer step). A row holds about 50 GB of activations: beside EP8's ~11 GB of state that fits an 80 GB GPU. Plain FSDP2 on eight GPUs (~27 GB of state) needs B200/B300 at 16K; since checkpointing is refused, on 80 GB GPUs it takes half the `max_length` at twice the accumulation. On one GPU the ~72 GB of state alone leaves no 80 GB fallback: a single-GPU run needs a B200/B300 (about 102–124 GB at 16K). Three steps on eight B300s at 16K peaked at 66 GiB reserved per GPU under EP8 and 80 GiB under FSDP2; the other fits are computed, not measured. For long-context EP runs raise `max_length` further (e.g. 32K at `gradient_accumulation_steps: 2`) — FLCE skips the logits tensor.

## Checkpoint save

General save/merge/resume mechanics are in [Checkpoints & Resume](../reference/checkpoints.md). Zaya needs no family-specific conversion: the EP gather reconstructs the two fused expert tensors per layer in the on-disk convention, which is exactly what the hub serves and `from_pretrained` reads.

`save_sharded_ep: true` works — `merge_ep_shards.py` reconstructs the same layout from the shards. The family-independent limits on sharded saves (no ETP, no CP, no expert LoRA, one EP group) are in [Checkpoints & Resume](../reference/checkpoints.md).
