# Step-3.7 Flash

`Step3p7ForConditionalGeneration` (`stepfun-ai/Step-3.7-Flash`) — 198B total / ~11B active composite VLM (196B language backbone + 1.8B vision encoder), native in transformers 5.16 (`model_type` `step3p7`, text config `step3p5`; no remote code). 45 text layers interleave 12 full-attention and 33 sliding-window layers, every head carries a sigmoid attention gate, and a 288-expert sigmoid-routed MoE carries most of the parameters.

| | EP | CP | TP | ETP | PP | FSDP2 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| Step-3.7 Flash | Yes | **No** | **No** | Yes | — ¹ | Yes |

¹ Pipeline parallelism is [not yet available in this release](../parallelism/pipeline-parallelism.md) — the shipped split contract for the family is under [Limitations](#limitations).

## Architecture

- **Attention interleave** — `layer_types` repeats full, sliding, sliding, sliding from index 0: 12 full-attention layers (64 heads, half-RoPE — `partial_rotary_factor` 0.5 — at θ 5e6 with llama3 scaling, factor 2 over 131072 original positions → 262144 max) and 33 sliding-window layers (96 heads, window 512, full RoPE at θ 10000). All layers are GQA with 8 KV heads — the per-layer head count lives in `per_layer_config`, and a bare `config.num_attention_heads` read **raises**, so head counts resolve through `get_config_field`'s per-layer seam — with QK-norm and a per-head sigmoid gate on the attention output (`g_proj`; applied after the attention kernel, so it rides every backend).
- **Norms** — Gemma-style RMSNorm: normalize in fp32, scale by `weight + 1`.
- **MoE** — layers 3–44 (the first 3 are dense MLP): 288 routed experts, sigmoid top-8 with a selection-only correction bias (`gate.e_score_correction_bias`, added to fp32 sigmoid scores for top-k; the combine weights are the unbiased scores, renormalized), `routed_scaling_factor` 3.0, one shared expert (`share_expert_dim` 1280), `moe_intermediate_size` 1280. Layers 43–44 clamp the routed SwiGLU at 7 (`swiglu_limits`, applied **after** the activation) and the shared expert at 16 (`swiglu_limits_shared`); every other layer runs unclamped.
- **MTP** — the checkpoint ships 3 MTP tail layers (indices 45–47); transformers never builds them (their keys are ignored on load), so every export ships without them. The config field `num_nextn_predict_layers: 3` survives as metadata; the PP gate rejects only live MTP layers, so it passes.

## Checkpoint

The release is plain bf16 (~403 GB of shards) — train from the repo id directly, no conversion step. The sibling `-FP8` / `-NVFP4` repos are serving artifacts, not training inputs.

The hub spells the experts as per-layer fused 3D stacks — `moe.gate_proj [E, M, H]` / `moe.up_proj` / `moe.down_proj` — which transformers' weight converter concatenates into the native fused `experts.gate_up_proj [E, 2M, H]` on load and splits back on save, alongside the renames (`moe.router_bias` ↔ `gate.e_score_correction_bias`, `share_expert.*` ↔ `mlp.shared_experts.*`, the vendor-namespace vision tower). Embeddings are untied (a separate `lm_head.weight`).

An EP-gathered save writes its **weights** in the hub layout the serving engines read: the family declares `_EXPORTS_HUB_NAMESPACE`, so the gathered save runs transformers' own save-side conversion revert on every streamed chunk (prefix renames back, `moe.gate.weight` / `moe.router_bias` / `share_expert.*`, the fused `gate_up_proj` split back into `moe.gate_proj` + `moe.up_proj`, the vendor-namespace vision tower). The EP SFT round-trip test pins the on-disk keys against a plain `save_pretrained` of the same config, the expert halves bit-exact against the live gathered tensor, the fp32 `moe.router_bias`, and reload-loss equality. Nothing else could restore that layout from a module-spelled save — `scripts/after_training/unfuse_moe_experts.py` refuses the family (its hub layout stores no per-expert tensors), and a transformers load + `save_pretrained` of a module-layout source reverts nothing (the load consumed no conversions) — which is also why sharded EP saves (`save_sharded_ep: true`) are refused for the family: the offline merge streams key by key and cannot apply the same revert.

The `config.json` beside those weights is the **source checkpoint's own**, not transformers'. The serving engines have no `step3p7` config class (vLLM 0.26.0 ships transformers 5.14) and read the family only through the release's `config.json` and the modules its `auto_map` names, which spell it in the vendor's keys — `moe_num_experts`, `moe_top_k`, `moe_layers_enum`, `attention_other_setting`, per-layer `rope_theta`. transformers 5.16 absorbs those at load (`Step3p7TextConfig.attribute_map` plus the `kwargs.pop` derivations in `__post_init__`) and re-emits only the native ones, so its own serialization is unservable and nothing on the save side can invert it. The family therefore declares `_EXPORTS_SOURCE_CONFIG_SCHEMA`, and every toolkit config write carries the source's config and remote-code modules forward with this run's changed values applied under the source's spellings ([Checkpoints](../reference/checkpoints.md#what-gets-saved)) — a `patch_vocab.py` `vocab_size` included. Training is unaffected: this transformers accepts those spellings, so the export reloads to exactly the config that was trained. A source that itself declares no `auto_map` (a synthetic checkpoint) has no schema to hand on; the export says so and keeps the native one.

## Model loading

The checkpoint is a composite VLM with **no text-only CausalLM sibling** in transformers, so `text_only_model` is refused for this family — and the refusal sees through the hub config's poisoned `auto_map`, whose `AutoModelForCausalLM` entry is the remote-code *conditional-generation* class itself, not a text-only sibling (`resolve_auto_model_class` rejects a masquerading entry). The composite class loads through the shared VLM path; a text-only dataset still takes the text data path ([SFT — VLMs](../training-methods/sft.md#vision-language-models)).

Native `AutoProcessor` is broken for the repo (it ships no `preprocessor_config.json`; the only processor is remote code). Text-only training needs neither: the text path uses the native `AutoTokenizer`, which carries the chat template (`<|im_start|>`/`<|im_end|>` roles, a `reasoning_effort` template variable, and a generation prompt that forces `<think>`). Two tokenizer facts to hold: `tokenizer_config.json` pins `model_max_length` 131072 while the model accepts 262144 — set `max_length` explicitly for runs past 128k — and the 128896-row embedding is deliberately wider than the 128815-entry tokenizer, so do not resize.

## EP wrapper

`EPStep3p7MoELayer` (`src/distributed/expert_parallel/layers/step3p7.py`) replaces `Step3p7SparseMoeBlock`.

- Routing: fp32 sigmoid scores plus `gate.e_score_correction_bias` on **selection only**, plain top-8 (no group limiting); the combine weights are the unbiased scores renormalized, with the block's `routed_scaling_factor` folded into them (equivalent — the combine is linear in the expert outputs).
- Experts: fused `Step3p7Experts.gate_up_proj [E, 2M, H]` / `down_proj [E, H, M]` in-module. The clamped layers run a fused Triton **post-activation** clamped SwiGLU — SiLU first, then clamp the activated gate (`fused_silu_then_clamp_mul`, `src/kernels/fused_glu.py`; the bound is a runtime kernel argument, so per-layer limits share one kernel), the mirror of DeepSeek-V4/GLM-5's pre-activation clamp and not interchangeable with it; unclamped layers take the base fused-SiLU combine.
- Shared expert: the block's own `Step3p7MLP`, adopted by reference (it carries its own `swiglu_limits_shared` clamp), added after DeepEP combine. The 3 dense leading layers keep their plain MLP — EP swaps MoE blocks only.

ETP uses the shared fused-GLU helper: at `expert_tp_size > 1` the gate/up halves store as separate shards ([ETP](../parallelism/expert-tensor-parallelism.md#limitations)).

EP loads take the lazy safetensors path: the hub checkpoint keeps a vendor namespace (the top-level prefix renames, `moe.*` → `mlp.*`, the two-source `moe.gate_proj + moe.up_proj → gate_up_proj` concatenate, the Step-3.5 vision tower's own chunked, RoPE-permuted `in_proj`), and the lazy loaders replay transformers' `step3p7` and `step3p5_vision` conversions per key (`_HUB_CONVERSION_KEYS`; the vision entries scoped under the tower's module path, as transformers scopes them) while reading only this rank's expert slice through both fan-in sources — bit-exact against `from_pretrained` (`tests/gpu/parallelism/ep/test_lazy_load_converted.py --family step3p7`), with no full-model host residency at 198B ([host RAM under the lazy path](../parallelism/large-scale-scenarios.md#other-300b-class-checkpoints)) — ~398 GB of page cache node-wide at ep8 (the checkpoint minus the MTP tail, shared by the 8 ranks), up to ~65 GB of file-backed RSS per rank while its shard handles are open. The fan-in's 2× transient (~0.8 GB per layer) is on the GPU.

## Serving for GRPO (vLLM)

Online and environmental GRPO sync into vLLM 0.26.0, which serves the family from the **hub** namespace (`model.layers.N.moe.gate_proj` / `up_proj` / `down_proj` stacks, `moe.gate.weight`, `share_expert.*`, the vendor-namespace vision tower) and silently drops any name it does not map. The sync forwards that namespace: the same save-side revert the gathered save runs (`_EXPORTS_HUB_NAMESPACE`) rewrites every forwarded key — the fused `experts.gate_up_proj` gather is split into the two hub stacks, the vision q/k/v are fused back into the un-permuted `in_proj`, and the `model.language_model.*` / `mlp.*` spellings return to the hub's — so the served model receives exactly the tensors a gathered checkpoint would carry (`tests/cpu/grpo/test_weight_sync_hub_namespace.py` pins the two against `save_pretrained`; `tests/gpu/trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py` proves the server's logprobs follow the trainer's, per weight group, on a live server). Serve the hub repo or a toolkit export directly, and note:

- `--trust-remote-code` — the vLLM image's transformers has no native `step3p7`; the composite config is the repo's own `configuration_step3p7.py` (the text tower resolves to vLLM's `step3p5`). A toolkit export ships that module and the ones it imports itself ([Checkpoint](#checkpoint)), so it needs the same flag and nothing else.
- `--moe-backend triton` and the layerwise-reload patch, as for every MoE family ([Rollout Servers](../infrastructure/rollout-servers.md#weight-sync)); the experts are a `FusedMoE`, which the patch already excludes from the reload lifecycle.
- The router's selection bias (`moe.router_bias`) is a buffer and the sync ships parameters only: it stays at its checkpoint value on both sides, which is exact because weight-sync RL runs `moe_balancing: none` (bias-update balancing is downgraded before any bias state exists).
- vLLM's clamped fused-MoE kernel accepts `swiglu_limits` of exactly 7 (the release value); any other clamp fails at server start, not silently.
- SGLang is refused for the family at construction: its loader needs the fused pair, which only GptOss gathers.

## Router balancing

The router carries a native `e_score_correction_bias` buffer that exports with every checkpoint (hub key `moe.router_bias`), and the modeling has no aux-loss machinery at all — no `router_aux_loss_coef`, no `output_router_logits` parameter — so under the EP wrappers `moe_balancing: auto` resolves to `bias_update`: [RouterBiasBalancingCallback](../training-methods/callbacks.md#routerbiasbalancingcallback) sign-updates the gate's own slot (selection only) and a served copy routes as trained. On the wrapper-less tree (`ep_size=1` with `use_grouped_gemm: false`) nothing carries the bias and no aux term can reach the loss, so `auto` resolves to `none` with a warning.

## Attention

Upstream declares `_supports_flash_attn = False`; SDPA is the only fast backend — pin `attn_implementation: sdpa` (the resolver diverts any flash request there itself).

## Limitations

- **CP** — `Step3p7Attention` has no Ulysses wrapper registered, so validation rejects the model as having no supported attention module ([Context Parallelism](../parallelism/context-parallelism.md#supported-model-architectures)). Nothing architectural blocks a wrapper: both head counts (64 full / 96 sliding) and the 8 KV heads divide cp 2/4/8.
- **TP** — the per-layer head counts fit no uniform q/k/v shard plan, so `Step3p7Attention` is outside the selective-TP accept-list and `tensor_parallel_size > 1` is rejected (zero shardable layers).
- **PP** — [not yet available in this release](../parallelism/pipeline-parallelism.md). The shipped contract admits the composite class (the only class the family ships) only for a run that feeds no images — the vision tower and projector are held by no stage, stashed for re-emission in every checkpoint — and refuses image data. The text tower splits with untied embeddings and a hidden-states-only residual; the split contract's stage offsets follow the period-4 `full,s,s,s` layer list.
- **Packing** — isolated: the layers are plain full/sliding attention (no conv or linear-attention mixers) and the forward feeds `position_ids` into both mask constructions, verified bit-exact through dense layers on SDPA and eager; through MoE layers doc-B drift is expert-summation reduction noise (~1e-7 fp32). The isolation holds on the training path only (`use_cache=False` — a live cache suppresses the packed mask, as DeepSeek-V4). See [Collators](../data/collators.md#document-isolation-under-packing).

## Configs

| Config | Topology | Notes |
|---|---|---|
| `examples/sft/step3p7/step-3.7-flash-ultrachat-ep-lora.yaml` | EP=8, 1×8 | LoRA + expert-LoRA — the single-node shape with headroom (~65 GB static of a 275 GB Blackwell card, expected); full FT at ep8 is ~190 GB of experts at 8 B/param, ~213 GB static — marginal and unmeasured |

The config points `model_name_or_path` at the hub repo directly (bf16 release). Full fine-tuning takes cross-node EP (`ep_scope: global`, e.g. EP=16 across 2×8), where the Gin dispatch ceiling caps `per_device_train_batch_size × max_length` at 8192 tokens/rank ([DeepEP](../infrastructure/deepep.md#expert-parallelism-over-aws-efa)).
