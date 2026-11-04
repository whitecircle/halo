# GLM-5 Next

`Glm5NextForConditionalGeneration` (`zai-org/GLM-5.3-Flash`) — 321B total / 18B active composite VLM, native in transformers (`model_type` `glm5_next`, text tower `glm5_next_text`; no remote code). 45 text layers mix KDA linear attention with DeepSeek sparse attention, hyper-connections widen the cross-layer residual 4×, and a 288-expert sigmoid-routed MoE carries most of the parameters.

Per-axis support: [Compatibility matrix](README.md#compatibility-matrix). Pipeline parallelism takes the text tower only (`GLM5_NEXT_PP`): a text-only run of the composite is admitted and re-emits the vision tower in every checkpoint, image data is refused, and no stage may begin on a `shared` DSA indexer layer ([Pipeline Parallelism](../parallelism/pipeline-parallelism.md#model-support)).

## Architecture

- **Attention interleave** — 34 `Glm5NextTextLinearAttention` layers (KDA: depthwise conv1d k=4 + delta-rule scan) and 11 `Glm5NextTextAttention` layers (`indexed_attention`: MLA with `q_lora_rank` 1536 / `kv_lora_rank` 512, qk 256 **NoPE**), 1-in-4 from index 3.

    No text layer applies a position encoding: the config refuses a non-zero `qk_rope_head_dim`, and the top-2048 `Glm5NextTextIndexer` applies no rotary either, so position reaches a DSA layer only through the indexer's pool boundaries. `layer_types` spells `"linear_attention"`, which is what the CP gate reads.

- **Hyper-connections** (`hc_mult: 4`, `hc_sinkhorn_iters: 20`, `hc_eps: 1e-6`) — per-layer `hc_attn_*` / `hc_ffn_*` tensors route a 4×-widened residual state across layers; the mHC mapping (Sinkhorn-projected combine matrix) computes in fp32 by upcasting its inputs and `fn` inside the forward, and the model declares `_is_stateful = True`.
- **MoE** — 288 routed experts, sigmoid top-8 `noaux_tc` (`n_group` 1 → plain top-k), `norm_topk_prob: true`, `routed_scaling_factor: 2.5`, 1 shared expert, `moe_intermediate_size` 2048; the first 3 layers are dense.

    `swiglu_limit: 10.0` clamps the GLU; the EP layer runs its own fused Triton clamped SwiGLU (`GluCombine.clamped_silu`, bound as a runtime kernel argument), latched into the generic fused-GLU seam the same way DeepSeek-V4 does ([GLM-4 → Fused SwiGLU](glm4.md#fused-swiglu)).

- **MTP** — the checkpoint ships one MTP layer (index 45); transformers drops it on load (`_keys_to_ignore_on_load_unexpected`), so every export ships without it. `mtp_loss_weight` [refuses the family](../optimization/mtp-heads.md): its decoder class builds `attn_hc` and `ffn_hc` layers the release's tail lacks, and the release stores the tail in fp8.

## Checkpoint conversion

The only release is fp8-e4m3 block-quantized (`quantization_config`: 128×128 blocks, fp32 per-block `*_scale_inv` sidecars; the KDA stack, norms, router, hyper-connection tensors and vision tower stay unquantized). EP requires plain BF16 experts, so convert once:

```bash
D=/path/to/large/volume   # verified with df -h / findmnt
HF_HOME=$D/hf python scripts/before_training/convert_glm5_bf16.py \
    --model_id zai-org/GLM-5.3-Flash --output_dir $D/models/GLM-5.3-Flash-BF16
```

Budget ~330 GB download cache + ~650 GB output; the conversion streams shard-by-shard, so host RAM stays bounded by `--max_shard_size`. Unquantized tensors keep their stored dtype, and the emitted `config.json` drops its `quantization_config`.

The family's fp32 pins are transformers' `_keep_in_fp32_modules_strict` (`e_score_correction_bias`, `conv1d`, `dt_bias`, `A_log`); the `hc_*` tensors are not among them. Every training loader casts the pinned parameters to the run dtype unless `fp32_non_ep_params` keeps them as fp32 masters ([Load precision](README.md#load-precision)), while the `e_score_correction_bias` buffer stays fp32, and every toolkit save keeps each at its trained dtype (`save_dtype_caster`).

## Model loading

The checkpoint is a composite VLM with **no text-only CausalLM sibling** in transformers, so `text_only_model` is refused for this family. The composite class loads through the shared VLM path; a text-only dataset still takes the text data path ([SFT — VLMs](../training-methods/sft.md#vision-language-models)).

## EP layer

`EPGlm5NextMoELayer` (`halo/distributed/expert_parallel/families/glm5_next.py`) replaces `Glm5NextTextMoE`.

- Routing: sigmoid scores plus the fp32 `gate.e_score_correction_bias` on **selection only** (the routed weight comes from the unbiased scores), top-8, `norm_topk_prob`, then `routed_scaling_factor` — fp32 top-k weights, expert load recorded for balancing.
- Experts: fused `Glm5NextTextExperts.gate_up_proj [E, 2M, H]` / `down_proj [E, H, M]` in-module. The hub spells them per-expert (`experts.{i}.gate_proj/up_proj/down_proj`); transformers' weight converter restores the fused layout on load and, on a `save_pretrained` with no EP layer, the per-expert spelling on save.

    An EP-gathered save writes the hub's vendor namespace (`hc_attn_*`, the KDA `f_a_proj`/`A_log` family, the split `q/k/v_conv1d`) and keeps the experts as the fused pair; `unfuse_moe_experts.py` turns it into the hub checkpoint. The lazy loaders and `from_pretrained` read either layout.

- Shared expert: one `Glm5NextTextMLP`, replicated per rank, added after DeepEP combine.
- The 3 dense leading layers keep their plain MLP — EP swaps MoE blocks only.

ETP uses the shared fused-GLU helper: at `expert_tp_size > 1` the gate/up halves store as separate shards ([ETP](../parallelism/expert-tensor-parallelism.md#limitations)).

EP loads take the lazy safetensors path. The hub checkpoint keeps a vendor namespace (hc tensors, the KDA `forget_gate` family, a three-source `q/k/v_conv1d → conv1d` merge, per-expert projections), and the lazy loaders plan their keys on transformers' `glm5_next` conversion, as `from_pretrained` does, while fusing only this rank's experts, bit-exact against `from_pretrained` (`tests/gpu/parallelism/ep/test_lazy_load_converted.py`).

There is no full-model host residency at 321B ([host RAM under the lazy path](../parallelism/large-scale-scenarios.md#other-300b-class-checkpoints)): ~628 GB of page cache node-wide at ep8 (the bf16 conversion minus the MTP tail, shared by the 8 ranks), up to ~95 GB of file-backed RSS per rank while its shard handles are open. The per-expert gate+up fusion's 2× transient (~1.2 GB per layer) is on the GPU.

RL weight sync is off (the spec's `weight_sync_refusal`): both pinned rollout engines register `glm5_next`, but their expert loaders take only per-expert keys where the gather emits the fused pair, and no online update into either has been validated.

## Router balancing

The router carries a native fp32 `e_score_correction_bias` buffer — GLM-4's exact slot — and the forward honors `output_router_logits`, so `moe_balancing: auto` resolves to `aux_loss` (the Laguna case), whose switch-style aux loss softmaxes the raw router logits and so balances the unbiased routing, not the selection `e_score_correction_bias` shifts. Explicit `bias_update` adopts the native buffer instead: [BalancingBiasCallback](../training-methods/callbacks.md#balancingbiascallback) sign-updates it, and the buffer exports with every checkpoint, so a served copy routes on it.

## Attention

Upstream declares `_supports_flash_attn = False`; SDPA is the only fast backend — pin `attn_implementation: sdpa` (the resolver diverts any flash request there itself). The KDA layers run their own linear-attention kernels regardless of the backend choice, and their autotune picks can move a run's logits across fresh caches ([Docker](../infrastructure/docker.md#baked-in-environment-variables)); they reach `fla` through the hub-kernel funnel the toolkit repairs, and on Blackwell share the delta-rule state kernel the toolkit keeps on `num_warps=2` ([Qwen3.5 → Linear attention](qwen3_5.md#linear-attention-qwen3_5moegateddeltanet)) — without the repair the funnel silently captures the torch scan.

## Limitations

- **CP** — 34 of 45 layers are a KDA linear recurrence over the sequence axis; validation rejects both the `Glm5NextTextLinearAttention` module and any `layer_types` containing `"linear_attention"` ([Context Parallelism](../parallelism/context-parallelism.md#supported-model-architectures)).
- **TP** — no shard plan: the DSA indexer and the KDA projections have no sound sharding, so a `tensor_parallel_size > 1` run is rejected (zero shardable layers).
- **Packing** — flat packing isolates neither layer type: both receive only the 2D padding mask, so the DSA layers attend across the documents of a packed row and the KDA conv/scan carries state across them ([Collators](../data/collators.md#document-isolation-under-packing)). The family runs no varlen backend, so `dense_packing_leaks` on its spec refuses flat `packing` whatever `allow_cross_document_linear_attention` says, and `padding_free` is refused on SDPA as well; so is packing under pipeline parallelism ([PP packing](../parallelism/pipeline-parallelism.md#packing)). [Tree packing](#tree-packing) isolates both layer types; otherwise train on padded rows.

## Tree packing

`tree_packing: true` ([Tree Packing](../optimization/tree-packing.md)) leaves the resolved backend (SDPA) in place for plan-less forwards; under a plan neither layer type reaches the attention interface, so `tree_full_attention` is refused.

- **KDA layers** take the GatedDeltaNet treatment: the conv runs each node behind the last 3 tokens of its path, and the KDA scan (fla `chunk_kda`) runs one tree depth at a time from the parent's final state.
- **DSA layers** select along each token's own path: the indexer's 4-token pools are a token and its path predecessors, and a query's candidates are the pools whose last token precedes it on its path, plus its incomplete tail pool. Attention over the selected keys runs on the 512-wide MLA latent with `kv_b_proj` folded in, not on a dense `[T, T]` mask; under autocast it reads the folded query and the latent in the autocast dtype. An adapter on `kv_b_proj`, a quantized load (`load_in_4bit`/`load_in_8bit`, or the fp8 release before [conversion](#checkpoint-conversion)) and a training-time `attention_dropout` are refused before the model loads (a regex target at the first tree batch): drop `kv_b_proj` from the [LoRA example](#configs)'s `lora_target_modules`.

The plan comes with every batch, so plain documents packed into one row are isolated in both layer types as well. `jinja-templates/glm/glm5-chat.jinja` renders every turn byte for byte as the hub template renders GLM-5's own non-thinking turn (`<|assistant|><think></think>` and the answer), minus the hub's `Reasoning Effort` system line and plus `<|endoftext|>` on the final turn (`tests/cpu/data/test_glm5_chat_template.py`, against the hub template vendored at one revision). Its generation prompt closes the empty block, so a model served with it answers directly; the hub template's opens `<think>`. It refuses tool definitions, tool calls and tool results instead of dropping them, since it does not render the hub's tool format. On top of the options tree packing requires (`packing`, `tree_share_markers`, `train_on_completions_only`), a GLM-5 run sets:

```yaml
attn_implementation: sdpa
tree_packing: true
chat_template: jinja-templates/glm/glm5-chat.jinja
force_chat_template: true
assistant_message_template: "<|assistant|>"
```

## Configs

`examples/sft/glm5_next/glm-5.3-flash-ultrachat-ep-lora.yaml` trains LoRA + expert-LoRA at EP=8 on one
8-GPU node, the only shape expected to fit one 8×Blackwell node (~110 GB static of a 288 GB card,
unmeasured). Full fine-tuning puts ~38B of experts on each rank, ~304 GB at 8 B/param.

The config points `model_name_or_path` at the BF16 conversion output above. Full fine-tuning takes cross-node EP (`ep_scope: global`, e.g. EP=16 across 2×8), where the Gin dispatch ceiling caps `per_device_train_batch_size × max_length` at 8192 tokens/rank ([DeepEP](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)).
