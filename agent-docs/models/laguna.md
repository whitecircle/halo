# Laguna

Poolside's `LagunaForCausalLM` (`poolside/Laguna-S-2.1`, `poolside/Laguna-XS-2.1`) — a MoE with a sigmoid top-k router, fused 3D expert storage, and one shared expert on every token. Both released sizes route to 256 experts: **S** is 48 layers / hidden 3072 / top-10 at 1M positions, **XS** is 40 layers / hidden 2048 / top-8 at 256K.

Transformers ships `transformers.models.laguna` natively, and the released checkpoints still load through `auto_map`, so both implementations are reachable: the shipped configs set `trust_remote_code: true` with a pinned `model_revision` (the revisions the EP path was validated against), while a checkpoint converted to library format loads the in-library classes. The EP wrapper claims both by class name and by `model_type`.

| | EP | CP | TP | ETP | PP | EP+CP | EP+TP | LoRA |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Laguna S / XS 2.1 | Yes | **No** | **No** | untested | — ¹ | **No** | **No** | Yes |

¹ Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md). Laguna-S-2.1's `layer_types` repeat with period 4, so the shipped PP split contract binds stage boundaries to multiples of 4.

## EP wrapper

`EPLagunaMoELayer` (`src/distributed/expert_parallel/layers/laguna.py`) subclasses GLM-4 MoE Lite's `EPGlm4MoELayer` and claims `LagunaSparseMoeBlock` / `model_type: laguna`. The two families share the whole contract — sigmoid routing, fused `gate_up_proj`/`down_proj` 3D storage, a shared expert on every token, and the per-expert hub checkpoint layout — so the gather and the sharded merge are inherited unchanged. `EPLagunaMoELayer` declares four deltas of its own — the top-k weight default, the optional-routing-knob opt-out, the served-bias warning, and the hub↔module key renames. Those, plus the two inherited GLM-4 behaviors that decide a Laguna load:

- **Shared-expert attribute** (inherited). Both in-library blocks name it `shared_experts`; `shared_expert` is the hub (and remote-code) spelling. The wrapper registers the submodule under whichever name its block actually used and records it in `_shared_expert_attr`, because that name becomes the export key in `replicated_named_params`. `EPSharedExpertsMoELayerBase` reads the same attribute in its `forward`, so the shared leg runs for both.
- **Top-k weight normalization.** `LagunaTopKRouter.forward` divides the gathered sigmoid scores by their sum **unconditionally**, and `LagunaConfig` declares no `norm_topk_prob` at all (`modular_laguna` deletes the inherited field), so nothing on the in-library block, gate or config can supply it. `EPLagunaMoELayer` therefore opts the knob out (`_OPTIONAL_ROUTING_KNOBS`) and declares `_NORM_TOPK_PROB_DEFAULT = True`, where GLM-4 MoE Lite — whose gate and config both declare it — requires it and raises on an upstream rename. Skipping the division scales every token's routed output by the sum of its top-k scores (~0.9-2.0x, data dependent) with no shape or dtype signal.
- **Where the routing knobs live.** In-library GLM-4 hangs `routed_scaling_factor` / `num_group` / `topk_group` / `norm_topk_prob` on the router and leaves `n_routed_experts` / `n_group` on the config; the Laguna remote-code revisions put `norm_topk_prob` on the router and declare no groups. `EPGroupLimitedMoELayerBase._init_routing` resolves each knob off the block, then the gate, then the config — accepting both the `n_group` and `num_group` spellings — so an explicit declaration always wins over the family default, and with no group knob at all group-limited selection degenerates to a plain top-k. Both families list exactly the knobs they lack in `_OPTIONAL_ROUTING_KNOBS`; any other missing knob raises.
- **A missing router bias** (inherited). The pinned remote-code revision omits `e_score_correction_bias` (the in-library format always writes it), so lazy loading leaves it on meta there. The wrapper materializes it at zero (a no-op additive bias) instead of letting the first routing matmul fault.
- **Hub vs module key spelling.** Laguna is the only family on the roster whose checkpoint names differ from its in-library module names: module `mlp.shared_experts.` is hub `mlp.shared_expert.`, and module `mlp.gate.e_score_correction_bias` is hub `mlp.experts.e_score_correction_bias`. Transformers declares both as `WeightRenaming` entries and applies them only inside `from_pretrained`, so `EPLagunaMoELayer._EXPORT_KEY_RENAMES` mirrors them for the three paths that bypass it — the gather, the RL weight sync, and the lazy loader (which applies the inverse on read). Without them the export writes keys vLLM silently skips and the loader leaves that submodule randomly initialized. The pairs fire only when the live name actually differs, so a remote-code revision already spelling `shared_expert` is untouched. See [Checkpoints](../reference/checkpoints.md#expert-parallelism-ep).

Routing is otherwise unchanged: sigmoid the logits, add the correction bias (and any `moe_balancing: bias_update` bias) for *selection* only, and gate with the unbiased scores. `LagunaTopKRouter.forward` returns `(logits, weights, indices)`; the wrapper keeps only the logits and recomputes weights and indices itself, since the cached pair predates routing replay and balancing.

Gathered saves restore the hub layout `experts.{i}.{gate,up,down}_proj.weight` through `_PER_EXPERT_UNFUSED_KEYS`.

Balancing is where the two families diverge. `LagunaConfig` ships `router_aux_loss_coef: 0.001` and `LagunaForCausalLM.forward` declares `output_router_logits`, so `moe_balancing: auto` resolves to `aux_loss` — GLM-4 MoE Lite, which has neither, lands on `bias_update`.

The inherited `_supports_bias_balancing` still accepts an explicit `bias_update` ([Callbacks](../training-methods/callbacks.md#moe-balancing-modes)). Under it the sign-updates land in the gate's own `e_score_correction_bias` — the inherited native-slot adoption — so the trained bias is part of the checkpoint and a transformers reload routes exactly as training did. vLLM 0.26.0 does not: its Laguna loader registers that tensor under a different internal name and drops the exported `mlp.experts.e_score_correction_bias` key silently, so a served copy routes on the pretrained bias. The balancing enable path warns about this at run start.

## Fused SwiGLU

Laguna's experts use a standard SwiGLU, so the base combine latch runs the fused Triton kernel whenever the resolved activation really is SiLU. Both the kernel and its behavioral gate are documented on the base class's page — [GLM-4 MoE Lite → Fused SwiGLU](glm4.md#fused-swiglu).

## Why CP and TP are unavailable

Both are registry-gated, and Laguna is in neither registry:

- **CP** needs a Ulysses attention wrapper. `LagunaAttention` is absent from `CP_SUPPORTED_ATTENTION_CLASSES` — the registered wrappers in `src/distributed/context_parallel/layers/` — so CP, and therefore EP+CP, is rejected.
- **TP** on a MoE model takes the selective attention-only path (HF's `tp_plan="auto"` mis-shards expert weights), gated on `TP_SHARDABLE_ATTENTION_CLASSES` (`src/distributed/tensor_parallel/module_types.py`). `LagunaAttention` is not listed, so there is nothing for the DTensor path to shard.

ETP is mechanically reachable (the experts use the shared fused-GLU storage, so `_init_fused_glu_params` handles `expert_tp_size > 1`) but has not been validated on Laguna.

## Configs

`examples/sft/laguna/laguna-s-2.1-ultrachat-ep.yaml` (EP=4, 256 experts → 64/rank) and `examples/sft/laguna/laguna-xs-2.1-ultrachat.yaml`. Launch the EP config at `--nproc_per_node=4`: `ep_size=4` is one dispatch group on 4 GPUs, but two racy 4-rank groups on 8, which `ParallelismConfig` rejects at config time. Three more settings are load-bearing:

- `attn_implementation: sdpa` — the pinned hub revision provides no Flash-Attention path. SDPA is not a varlen backend, so the collator factory ([Collators](../data/collators.md)) rejects `padding_free` outright; the configs use `packing: true`, which keeps documents isolated but materializes a dense mask over the flattened batch (side up to `per_device_train_batch_size * max_length`) instead of consuming `cu_seqlens`.
- Liger covers `laguna` (RMSNorm, the fused SwiGLU on the dense and shared-expert MLPs, cross-entropy, and an opt-in fused loss); the shipped configs run it. RoPE stays eager — Laguna's full-attention layers rotate half the head and its sliding layers all of it, through one shared function.
- `pad_token: "〈|PAD|〉"` / `eos_token: "〈|EOS|〉"` — Laguna's vocabulary really does use the CJK angle brackets U+3008/U+3009, not ASCII `<`/`>`. Substituting ASCII silently adds new tokens instead of resolving the existing ones.

CPU parity coverage lives in `tests/cpu/parallelism/test_laguna_ep.py`, built against the real `transformers.models.laguna` block and covering both it and a remote-code-shaped one.
