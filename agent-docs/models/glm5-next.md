# GLM-5 Next

`Glm5NextForConditionalGeneration` (`zai-org/GLM-5.3-Flash`) — 321B total / 18B active composite VLM, native in transformers 5.16 (`model_type` `glm5_next`, text tower `glm5_next_text`; no remote code). 45 text layers mix KDA linear attention with DeepSeek sparse attention, hyper-connections widen the cross-layer residual 4×, and a 288-expert sigmoid-routed MoE carries most of the parameters.

| | EP | CP | TP | ETP | PP | FSDP2 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| GLM-5 Next | Yes | **No** | **No** | Yes | — ¹ | Yes |

¹ Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md) — the shipped split contract for the family is under [Limitations](#limitations).

## Architecture

- **Attention interleave** — 34 `Glm5NextTextLinearAttention` layers (KDA: depthwise conv1d k=4 + delta-rule scan) and 11 `Glm5NextTextAttention` layers (`deepseek_sparse_attention`: MLA with `q_lora_rank` 1536 / `kv_lora_rank` 512, qk 256 **NoPE** — RoPE exists only inside the top-2048 `Glm5NextTextIndexer`), 1-in-4 from index 3. `layer_types` spells `"linear_attention"`, which is what the CP gate reads.
- **Hyper-connections** (`hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`) — per-layer `hc_attn_*` / `hc_ffn_*` tensors route a 4×-widened residual state across layers; the mHC mapping (Sinkhorn-projected combine matrix) computes in fp32 by upcasting its inputs and `fn` inside the forward, and the model declares `_is_stateful = True`.
- **MoE** — 288 routed experts, sigmoid top-8 `noaux_tc` (`n_group` 1 → plain top-k), `norm_topk_prob: true`, `routed_scaling_factor: 2.5`, 1 shared expert, `moe_intermediate_size` 2048; the first 3 layers are dense. `swiglu_limit: 10.0` clamps the GLU; the EP layer runs its own fused Triton clamped SwiGLU (`fused_clamped_silu_mul`, bound as a runtime kernel argument), latched into the generic fused-GLU seam the same way DeepSeek-V4 does ([GLM-4 → Fused SwiGLU](glm4.md#fused-swiglu)).
- **MTP** — the checkpoint ships one MTP layer (index 45); transformers drops it on load (`_keys_to_ignore_on_load_unexpected`), so every export ships without it.

## Checkpoint conversion

The only release is fp8-e4m3 block-quantized (`quantization_config`: 128×128 blocks, fp32 per-block `*_scale_inv` sidecars; the KDA stack, norms, router, hyper-connection tensors and vision tower stay unquantized). EP requires plain BF16 experts, so convert once:

```bash
HF_HOME=/mnt/hf python scripts/before_training/convert_glm5_bf16.py \
    --model_id zai-org/GLM-5.3-Flash --output_dir /mnt/models/GLM-5.3-Flash-BF16
```

Budget ~330 GB download cache + ~650 GB output; the conversion streams shard-by-shard, so host RAM stays bounded by `--max_shard_size`. Unquantized tensors (the KDA stack, norms, router, hyper-connection tensors, vision tower) keep their stored dtype, and the emitted `config.json` drops its `quantization_config`. The family's fp32 pins are transformers' `_keep_in_fp32_modules_strict` — `e_score_correction_bias`, `conv1d`, `dt_bias`, `A_log` — which the load upcasts and every toolkit save keeps at their trained dtype (`save_dtype_caster`); the `hc_*` tensors are not among them.

## Model loading

The checkpoint is a composite VLM with **no text-only CausalLM sibling** in transformers, so `text_only_model` is refused for this family. The composite class loads through the shared VLM path; a text-only dataset still takes the text data path ([SFT — VLMs](../training-methods/sft.md#vision-language-models)).

## EP wrapper

`EPGlm5NextMoELayer` (`src/distributed/expert_parallel/layers/glm5_next.py`) replaces `Glm5NextTextMoE`.

- Routing: sigmoid scores plus the fp32 `gate.e_score_correction_bias` on **selection only** (the routed weight comes from the unbiased scores), top-8, `norm_topk_prob`, then `routed_scaling_factor` — fp32 top-k weights, expert load recorded for balancing.
- Experts: fused `Glm5NextTextExperts.gate_up_proj [E, 2M, H]` / `down_proj [E, H, M]` in-module; the hub spells them per-expert (`experts.{i}.gate_proj/up_proj/down_proj`) and transformers' weight converter restores the fused layout on load and, on a wrapper-less `save_pretrained`, the per-expert spelling on save. An EP-gathered save keeps the module spelling (the fused pair, `attn_hc.*`, `forget_gate.*`, the fused KDA `conv1d`) — the layout the lazy loaders and `from_pretrained` both read, so the two artifacts reload identically while differing on disk.
- Shared expert: one `Glm5NextTextMLP`, replicated per rank, added after DeepEP combine.
- The 3 dense leading layers keep their plain MLP — EP swaps MoE blocks only.

ETP uses the shared fused-GLU helper: at `expert_tp_size > 1` the gate/up halves store as separate shards ([ETP](../parallelism/expert-tensor-parallelism.md#limitations)).

EP loads take the lazy safetensors path: the hub checkpoint keeps a vendor namespace (hc tensors, the KDA `forget_gate` family, a three-source `q/k/v_conv1d → conv1d` merge, per-expert projections), and the lazy loaders replay transformers' `glm5_next` conversion per key (`_HUB_CONVERSION_KEYS`) while fusing only this rank's experts — bit-exact against `from_pretrained` (`tests/gpu/parallelism/ep/test_lazy_load_converted.py`), with no full-model host residency at 321B ([host RAM under the lazy path](../parallelism/large-scale-scenarios.md#other-300b-class-checkpoints)) — ~628 GB of page cache node-wide at ep8 (the bf16 conversion minus the MTP tail, shared by the 8 ranks), up to ~95 GB of file-backed RSS per rank while its shard handles are open. The per-expert gate+up fusion's 2× transient (~1.2 GB per layer) is on the GPU. RL weight sync is off (`_supports_weight_sync`): a sync into a serving engine reading hub names would land nowhere, and no pinned rollout engine loads `glm5_next`.

## Router balancing

The router carries a native fp32 `e_score_correction_bias` buffer — GLM-4's exact slot — and the forward honors `output_router_logits`, so `moe_balancing: auto` resolves to `aux_loss` (the Laguna case). Explicit `bias_update` adopts the native buffer instead: [RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback) sign-updates it, and the buffer exports with every checkpoint, so a served copy routes as trained.

## Attention

Upstream declares `_supports_flash_attn = False`; SDPA is the only fast backend — pin `attn_implementation: sdpa` (the resolver diverts any flash request there itself). The KDA layers run their own linear-attention kernels regardless of the backend choice.

## Limitations

- **CP** — 34 of 45 layers are a KDA linear recurrence over the sequence axis; validation rejects both the `Glm5NextTextLinearAttention` module and any `layer_types` containing `"linear_attention"` ([Context Parallelism](../parallelism/context-parallelism.md#supported-model-architectures)).
- **TP** — no shard plan: the DSA indexer and the KDA projections have no sound sharding, so a `tensor_parallel_size > 1` run is rejected (zero shardable layers).
- **PP** — [not yet available in this release](../parallelism/pipeline-parallelism.md). The shipped `Glm5NextPPSpec` split contract carries the 4×-widened hyper-connection stream as the stage boundary, keeps `hc_head` on the last stage, and refuses a stage that begins on a `shared` DSA indexer layer (GLM-5.3-Flash ships all `full`). The family ships only the composite `Glm5NextForConditionalGeneration`, no text-only CausalLM; the multimodal gate admits it only for a run that feeds no images — the vision tower and projector are held by no stage, stashed on the save rank for re-emission in every checkpoint, so an export reloads as the composite class — and refuses image data.
- **Packing** — the KDA conv/scan crosses packed document boundaries on every backend, the same accepted mixer class as Zaya and Ling ([Collators](../data/collators.md#document-isolation-under-packing)); pack only where a small amount of cross-document mixing is acceptable.

## Configs

| Config | Topology | Notes |
|---|---|---|
| `examples/sft/glm5_next/glm-5.3-flash-ultrachat-ep-lora.yaml` | EP=8, 1×8 | LoRA + expert-LoRA — the only shape expected to fit one 8×Blackwell node (~110 GB static of a 275 GB card, unmeasured; full FT puts ~38B of experts on each rank, ~304 GB at 8 B/param) |

The config points `model_name_or_path` at the BF16 conversion output above. Full fine-tuning takes cross-node EP (`ep_scope: global`, e.g. EP=16 across 2×8), where the Gin dispatch ceiling caps `per_device_train_batch_size × max_length` at 8192 tokens/rank ([DeepEP](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)).
