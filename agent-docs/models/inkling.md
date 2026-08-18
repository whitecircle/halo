# Inkling

Thinking Machines' `InklingForConditionalGeneration` (`thinkingmachines/Inkling-Small`, 276B total / 12B active) — a multimodal MoE routing to 256 experts with top-6 selection, 2 always-on shared experts, and 40 MoE layers (of 42 — `dense_mlp_idx: 2` keeps the first two dense) at hidden 4096.

Attention is hybrid sliding (5 sliding : 1 full, period 6) with **no RoPE** — position enters as an additive relative-logits bias — plus depthwise causal short convolutions (`InklingShortConvolution`) on the projected K/V and around each decoder sublayer.

Transformers ships `transformers.models.inkling` natively (the image pins 5.16.1), so the family loads with no remote code. The composite config's `model_type` is `inkling_mm_model`, the text config's `inkling_text` (there is no bare `inkling` spelling).

| | EP | CP | TP | ETP | EP+CP | EP+TP | LoRA |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Inkling-Small | Yes | **No** ¹ | **No** ¹ | Yes | **No** ¹ | **No** ¹ | untested |

¹ Architectural, not a missing registration — see [Why CP and TP are out](#why-cp-and-tp-are-out). PP is not yet available in this release — see [Pipeline parallelism](#pipeline-parallelism).

## EP wrapper

`EPInklingMoELayer` (`src/distributed/expert_parallel/layers/inkling.py`) claims `InklingMoE` / `model_type: inkling_mm_model`, `inkling_text`. Fused `gate_up_proj`/`down_proj` routed experts in the GLM-4 layout, so the base fused-GLU storage, gather, and sharded merge apply unchanged — including ETP, whose split-shard path and token-space partial-sum reduce are validated by `tests/gpu/parallelism/combined/test_ep_etp_inkling.py` (pure ETP on 2 GPUs, EP+ETP on 4; the 2 shared experts stay replicated on every ETP rank). Two family quirks force a standalone wrapper rather than a GLM-4 subclass:

- **Joint routed+shared normalization.** `InklingTopkRouter` emits `n_routed_experts + n_shared_experts` logits from one projection and normalizes the routed top-k and the shared experts **jointly** (`logsumexp` over `top_k + n_shared` logits), so the shared experts compete for probability mass. The shared FFN is scaled by its resulting share (`gammas`) — an extra argument `EPSharedExpertsMoELayerBase`'s shared-expert call cannot pass, so the wrapper runs the shared leg itself.
- **Selection vs gating split.** Selection runs on `sigmoid(routed_logits) + e_score_correction_bias` (a learned routing bias the family ships); `moe_balancing: bias_update` adds its balancing bias on top for selection only, while the returned weights stay derived from the unbiased logits. The causal-LM loss never reads `router_logits`, so there is no aux-loss path and `moe_balancing: auto` resolves to `bias_update`.

One contract fails closed: `_supports_weight_sync = False` refuses online/environmental GRPO — an inference server loading hub names would silently skip every module-spelled tensor the sync sends.

## Loading

The hub checkpoint stores the expert width as `intermediate_size: 2048`, but transformers defaults `moe_intermediate_size: 3072` and its compat shim maps only `dense_intermediate_size` — without `model_init_kwargs: {moe_intermediate_size: 2048}` every expert tensor is built 3072-wide and the load hard-fails.

The checkpoint ships Thinking Machines' original namespace (`model.llm.*`, `attn.wq_du`, interleaved `experts.w13_weight`), which transformers converts inside `from_pretrained` (`transformers/conversion_mapping.py`, keyed `inkling_mm_model`).

The lazy loaders read the same declarative entries through the family's `_HUB_CONVERSION_KEYS` declaration, replayed by `src/distributed/expert_parallel/hub_conversion.py`: renames plus a closed op vocabulary (de-interleave, chunk) applied at materialization, after the ranged per-expert read. The default lazy path therefore loads the hub checkpoint directly, each rank reading only its expert slice, pinned bitwise against `from_pretrained` by `tests/gpu/parallelism/ep/test_lazy_load_inkling.py`. The `from_pretrained` fallback still works but materializes the full checkpoint per concurrently-loading rank — 532 GB for Inkling-Small, so `max_concurrent_loading: 4` ≈ 2.1 TB peak host RAM.

`from_pretrained` honors the family's fp32 pin, which covers the **short convolutions** (`_keep_in_fp32_modules_strict`: `k_sconv`/`v_sconv`/`attn_sconv`/`mlp_sconv`); `load_ep_model` then re-casts parameters to the run dtype, since FSDP2 rejects mixed-dtype parameters in one shard group — the sconvs train in bf16 here, a deliberate trade validated by the multi-node runs below.

## Why CP and TP are out

- **Packing** — the attention layers isolate packed documents, but the four depthwise convolutions per layer cross boundaries: the modeling reads `seq_idx` yet its conv call sites never forward kwargs, so no collator emission can reach them ([Document isolation](../data/collators.md#document-isolation-under-packing)).
- **CP** — `InklingShortConvolution` runs over the sequence axis (a CP shard severs its receptive field at every chunk boundary — the LFM-2/Zaya blocker), and the additive relative-logits bias cannot pass through `flash_attn_func`, which the Ulysses path is built on (`_supports_flash_attn = False` upstream). Validation rejects the conv class with a precise error.
- **TP** — the selective-TP planner shards q/k/v/o structurally; Inkling's attention carries per-layer head geometry (`swa_*` on sliding layers), sequence convolutions on the projected K/V, and a per-head `rel_logits_proj`, none of which the planner can shard. The zero-shard raise names the class.

## Pipeline parallelism

Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md).
The shipped seams target the **text decoder** (`InklingForCausalLM`): given a text-only
`config.json` (`InklingTextConfig` fields, `architectures: ["InklingForCausalLM"]`) beside the hub
weights, the stage loader reads the TM-namespace safetensors through the conversion entry and drops
the tower/MTP keys. The generic VLM gate refuses only a run that feeds images, so the composite
class is admitted text-only as well. Layer types repeat with period 6 on Inkling-Small (42 layers),
so the split contract binds stage boundaries to multiples of 6, and EP must fit inside one stage.

## Multimodal training

The composite class trains under EP: patching finds `InklingMoE` under `model.language_model`, the
vision/audio towers survive as replicated (FSDP-managed) modules, and an image-carrying batch —
pixel features masked-scattered at `image_token_id` placeholders — backpropagates into the vision
tower alongside the expert shards (`tests/gpu/parallelism/ep/test_ep_vlm_inkling.py`, EP=2 vs the
undistributed composite reference). Image-text SFT rides the VLM data path (`VLMDataCollator` +
`processing_inkling`); text-only data through the same class is what the multi-node runs below
validated. CP rejects the class outright and PP refuses any image-carrying run — multimodal is
EP/ETP-only.

## Multi-node EP

532 GB of bf16 weights do not fit node-local EP=8 alongside full-FT optimizer states — Inkling-Small trains at `ep_size: 16` / `ep_scope: global` on 2×8 B300. Measured on that topology:

- Keep `per_device_train_batch_size × max_length` within the cross-node Gin dispatch cap (`HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`, 8192 — [DeepEP](../infrastructure/deepep.md)); the shipped config runs mbs 2 × 4096 = 8192, validated exactly at the cap.
- Grouped GEMM and the auto SM count (the 24-SM zero-division fallback) run clean at Inkling's shapes. If the first cross-node combine times out intermittently, pin `HALO_DEEPEP_NUM_SMS=16` — fewer channels shrink the barrier race window.
- Measured: 3.7-4.1 s/step at mbs 2 × 4096 (2×8 B300, per-batch padding), ~9,400 attended tok/s global (~590/GPU), peak ~246 GB of the 288 GB card; DeepEP arena grows are exercised and stable.

## Configs

`examples/sft/inkling/inkling-small-multinode-ep16.yaml` (EP=16 across 2 nodes, 256 experts → 16/rank). Load-bearing settings beyond the loading overrides above: `attn_implementation: sdpa` (no Inkling FA dispatch; the Blackwell auto-select would fall back per-rank), and `pad_token: "<|unused|>"` (the tokenizer ships no pad token; id 199998 is a dedicated special that never appears in text). Liger covers Inkling's RMSNorm and cross-entropy (the config runs it). Its MLP and head stay eager — `InklingMLP` scales its output by a trained `global_scale`, and the head divides by `logits_mup_width_multiplier` and truncates to `unpadded_vocab_size` before the loss. There is no rotary to fuse: position enters as a learned relative-logit bias.

Inkling loads as a `ConditionalGeneration` class, but the data path follows the run: text-only SFT rows take the text pipeline, so `packing` is available here ([SFT — VLMs](../training-methods/sft.md#vision-language-models)). The binding limit at EP=16 is the cross-node dispatch cap above — `per_device_train_batch_size × max_length ≤ 8192` tokens/rank, however the rows are formed.

The shipped config sets `packing: false`: full rows attend 8192 tokens/rank/step against the ~2.3k the measurement above ran at, on a peak of 246 of 288 GB, and the depthwise convs cross packed documents either way.

CPU coverage: `tests/cpu/models/test_inkling_support.py` (registration and router parity, joint normalization included). GPU gates: `tests/gpu/parallelism/ep/test_ep_vs_reference_inkling.py`, `tests/gpu/parallelism/combined/test_ep_etp_inkling.py`, `tests/gpu/parallelism/ep/test_ep_vlm_inkling.py`.
