# Laguna

Poolside's `LagunaForCausalLM` (`poolside/Laguna-S-2.1`, `poolside/Laguna-XS-2.1`) — a MoE with a sigmoid top-k router, fused 3D expert storage, and one shared expert on every token. Both released sizes route to 256 experts: **S** is 48 layers / hidden 3072 / top-10 at 1M positions, **XS** is 40 layers / hidden 2048 / top-8 at 256K.

Transformers ships `transformers.models.laguna` natively, and the released checkpoints also load through `auto_map`, so both implementations are reachable: the shipped configs set `trust_remote_code: true` with a pinned `model_revision` (the revisions the EP path was validated against); with it off, the same checkpoints load into the in-library classes, which compute the same forward. The EP layer claims both by class name and by `model_type`, and Liger patches both ([Configs](#configs)).

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). Pipeline parallelism cuts on whole four-layer `layer_types` periods ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md#model-support)). Under it `aux_loss` with a nonzero coefficient is refused; the released checkpoints ship `0.0`.

## EP layer

`EPLagunaMoELayer` (`halo/distributed/expert_parallel/families/laguna.py`) subclasses GLM-4 MoE Lite's `EPGlm4MoELayer` and claims `LagunaSparseMoeBlock` / `model_type: laguna`. The two families share the whole contract (sigmoid routing, fused `gate_up_proj`/`down_proj` 3D storage, a shared expert on every token, and the per-expert hub checkpoint layout), so the gather and the sharded merge are inherited unchanged.

`EPLagunaMoELayer` declares four deltas of its own: the top-k weight default, the optional-routing-knob opt-out, the served-bias warning, and the hub↔module key renames. Those, plus the two inherited GLM-4 behaviors that decide a Laguna load:

- **Shared-expert attribute** (inherited). Both in-library blocks name it `shared_experts`; `shared_expert` is the hub (and remote-code) spelling. The wrapper registers the submodule under whichever name its block actually used and records it in `_shared_expert_attr`.

    That name becomes the export key in `replicated_named_params`. `EPSharedExpertsMoELayerBase` reads the same attribute in its `forward`, so the shared leg runs for both.

- **Top-k weight normalization.** `LagunaTopKRouter.forward` divides the gathered sigmoid scores by their sum **unconditionally**, and `LagunaConfig` declares no `norm_topk_prob` at all (`modular_laguna` deletes the inherited field), so nothing on the in-library block, gate or config can supply it.

    `EPLagunaMoELayer` therefore opts the knob out (`_OPTIONAL_ROUTING_KNOBS`) and declares `_NORM_TOPK_PROB_DEFAULT = True`. GLM-4 MoE Lite, whose gate and config both declare it, requires it and raises on an upstream rename. Skipping the division scales every token's routed output by the sum of its top-k scores (~0.9-2.0x, data dependent) with no shape or dtype signal.

- **Where the routing knobs live.** In-library GLM-4 hangs `routed_scaling_factor` / `num_group` / `topk_group` / `norm_topk_prob` on the router and leaves `n_routed_experts` / `n_group` on the config; the Laguna remote-code revisions put `norm_topk_prob` on the router and declare no groups.

    `EPGroupLimitedMoELayerBase._adopt_routing` resolves each knob off the block, then the gate, then the config, accepting both the `n_group` and `num_group` spellings. An explicit declaration always wins over the family default, and with no group knob at all group-limited selection degenerates to a plain top-k. Both families list exactly the knobs they lack in `_OPTIONAL_ROUTING_KNOBS`; any other missing knob raises.

- **A missing router bias** (inherited). A checkpoint without `e_score_correction_bias` would leave it on meta under lazy loading; the EP layer materializes it at zero (a no-op additive bias) instead of letting the first routing matmul fault. The released Laguna checkpoints carry it.
- **Hub vs module key spelling.** Laguna is the only family whose spec declares `export_key_renames`: module `mlp.shared_experts.` is hub `mlp.shared_expert.`, and module `mlp.gate.e_score_correction_bias` is hub `mlp.experts.e_score_correction_bias`.

    Transformers declares both as `WeightRenaming` entries, which `from_pretrained` and the lazy loader apply on read. `LAGUNA.export_key_renames` mirrors them for the three save-side paths that bypass transformers: the gather, the RL weight sync and `merge_ep_shards.py`. Without them the export writes keys vLLM silently skips.

    The pairs fire only when the live name actually differs, so a remote-code revision already spelling `shared_expert` is untouched. See [Checkpoints](../reference/checkpoints.md#expert-parallelism-ep-eptp-epcp).

Routing is otherwise unchanged: sigmoid the logits, add the correction bias (and any `moe_balancing: bias_update` bias) for *selection* only, and gate with the unbiased scores. `LagunaTopKRouter.forward` returns `(logits, weights, indices)`; the EP layer keeps only the logits and recomputes weights and indices itself, since the cached pair predates routing replay and balancing.

Gathered saves restore the hub layout `experts.{i}.{gate,up,down}_proj.weight` through the spec's `FUSED_SPLIT_PER_EXPERT` `ExpertLayout`.

Balancing is where the two families diverge. `LagunaForCausalLM.forward` declares `output_router_logits`, so `moe_balancing: auto` resolves to `aux_loss` — GLM-4 MoE Lite, which does not, lands on `bias_update`. The released checkpoints set `router_aux_loss_coef: 0.0`, so that term stays off, with a warning, unless the run sets a coefficient through `model_init_kwargs`. That term is transformers' shared switch-style aux loss: it softmaxes the raw router logits and counts their top-k, so it balances the unbiased routing, not the selection `e_score_correction_bias` shifts.

The inherited `SUPPORTS_BIAS_BALANCING` still accepts an explicit `bias_update` ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)). Under it the sign-updates land in the gate's own `e_score_correction_bias` (the inherited native-slot adoption), so the trained bias is part of the checkpoint and a transformers reload routes exactly as training did.

vLLM 0.32.0 does not: its Laguna loader registers that tensor under a different internal name and drops the exported `mlp.experts.e_score_correction_bias` key silently, so a served copy never routes on the trained bias (by the loader's code, the served bias keeps its zero initialization). The balancing enable path warns about this at run start.

RL weight sync runs on vLLM only. SGLang 0.5.22's Laguna loader asserts every routed-expert tensor of every sparse layer in each `load_weights` call, which a chunked online update cannot satisfy, so `rollout_backend: sglang` is refused at trainer construction ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

## Fused SwiGLU

Laguna's experts use a standard SwiGLU, so the base combine latch runs the fused Triton kernel whenever the resolved activation really is SiLU. Both the kernel and its behavioral gate are documented on the base class's page — [GLM-4 MoE Lite → Fused SwiGLU](glm4.md#fused-swiglu).

## Why CP and TP are unavailable

Both are registry-gated, and Laguna is in neither registry:

- **CP** needs a Ulysses attention wrapper. `LagunaAttention` is absent from `ULYSSES_ATTENTION_MAP` — the registered wrappers in `halo/distributed/context_parallel/families.py` — so CP, and therefore EP+CP, is rejected.
- **TP** on a MoE model takes the selective attention-only path (HF's `tp_plan="auto"` mis-shards expert weights), gated on `TP_SHARDABLE_ATTENTION_CLASSES` (`halo/distributed/tensor_parallel/parallelize_attention.py`). `LagunaAttention` is not listed, so there is nothing for the DTensor path to shard.

ETP is mechanically reachable (the experts use the shared fused-GLU storage, so `_init_fused_glu_params` handles `expert_tp_size > 1`) and tiny-model verified only: pure ETP and EP+ETP train, save and resume (`tests/gpu/parallelism/ep/test_ep_preference_precompute_resume.py --family laguna --mode etp2` and `--mode ep2etp2`); the released checkpoints are not run under ETP.

## Configs

`examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml` (EP=8 on eight GPUs, 256 experts → 32/rank, packed 8,192-token rows) and `examples/sft/laguna/laguna-xs-2.1-ultrachat.yaml`. The S config fills the node with 8,192 tokens per micro-batch, the same 65,536 tokens per optimizer step as EP4 at 2,048 with accumulation 8, and spreads the per-layer dispatch and host cost over four times the tokens. It trains without gradient checkpointing: a row holds about 114 GB of activations beside about 75 GB of EP8 state, a B300 fit that is computed, not measured; smaller GPUs pass `--gradient_checkpointing=true`. A four-GPU launch passes `--expert_parallel_size=4`: `ep_size=4` is one dispatch group on 4 GPUs, but two racy 4-rank groups on 8, which `ParallelismConfig` rejects at config time. Two more settings are load-bearing:

- `attn_implementation: sdpa`: the pinned revision's modeling code never passes `sliding_window` to the attention function, so a flash backend would run its sliding-window layers as full attention; SDPA applies the sliding mask. The loader holds this for remote code on its own: left unset, it takes SDPA with a warning, and a pinned flash label raises. SDPA is not a varlen backend, so the collator factory ([Collators](../data/collators.md)) rejects `padding_free` outright. Through the pinned remote code the model runs plain SDPA; the in-library classes (`trust_remote_code: false`) are built as `sdpa_flex_sliding`, FlexAttention on the sliding layers and on packed full layers ([Flash Attention](../optimization/flash-attention.md#sliding-window-and-wide-head-layers-on-sdpa)).

    The configs use `packing: true`, which keeps documents isolated but materializes a dense mask over the flattened batch (side up to `per_device_train_batch_size * max_length`) instead of consuming `cu_seqlens`.

- `pad_token: "〈|PAD|〉"` / `eos_token: "〈|EOS|〉"` — Laguna's vocabulary really does use the CJK angle brackets U+3008/U+3009, not ASCII `<`/`>`. Substituting ASCII silently adds new tokens instead of resolving the existing ones.

[Liger](../optimization/liger-kernels.md)'s `laguna` spec patches both implementations: the in-library classes, and, through `remote_classes`, the `LagunaRMSNorm` / `LagunaMLP` / `LagunaForCausalLM` of the pinned hub modeling file, which are the in-library classes verbatim. Under either load that is RMSNorm, the fused SwiGLU on the dense and shared-expert MLPs, cross-entropy and an opt-in fused loss. RoPE stays eager on either path — Laguna's full-attention layers rotate half the head and its sliding layers all of it, through one shared function.

CPU parity coverage lives in `tests/cpu/parallelism/test_laguna_ep.py`, built against the real `transformers.models.laguna` block and covering both it and a remote-code-shaped one.
