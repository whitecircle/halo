# Adding a New Model

| Goal | Track |
|---|---|
| MoE model, need EP | [Add EP support](#add-ep-support) |
| Long-context dense or MoE, need CP | [Add CP support](#add-cp-support) |
| Large model, want TP beyond what `tp_plan` covers | [Add TP support](#add-tp-support) |
| Model not in transformers yet | [Vendoring a model](#vendoring-a-model) |
| Family the auto-detected attention backend cannot run | [Attention backend](#attention-backend) |

For plain FSDP, no work is needed — any `AutoModelForCausalLM` works.

Multimodal models work too: the loaders resolve the class through `resolve_auto_model_class()` (`src/models/loading/model_preparation.py`), so any `model_type` in transformers' image-text-to-text mapping loads via `AutoModelForImageTextToText` rather than dropping the vision tower. The training scripts detect a VLM from its config and route the load through `load_distributed_model`, so a MoE VLM gets EP/CP/TP on the multimodal wrapper (`config_has_experts` finds the experts on the nested `text_config`). No EP-wrapper work is VLM-specific.

## Attention backend

Auto-detection covers a family whose attention the installed kernels already run. Where it does not — a head dim no flash kernel accepts, a NaN-prone backward, attention sinks, position IDs the modeling never forwards — the two seams are in `resolve_attn_implementation` (`src/models/patches/attention.py`), which narrows the backend from the family's own capabilities, and `apply_family_attention_patches` (`src/models/loading/model_preparation.py`), which applies the family's patches to trainable and frozen loads alike. Add the family predicate beside the existing ones and wire it into whichever seam applies; the per-family matrix is in [Flash Attention](../optimization/flash-attention.md#model-specific-handling).

## Add EP support

EP mechanics are in the [Expert Parallelism guide](../parallelism/expert-parallelism.md). A new family is **one wrapper class** under `src/distributed/expert_parallel/layers/` — there is no registry file to edit. `MOE_LAYER_MAP` is built by walking the `EPMoELayerBase` subclass tree, and `layers/roster.py` imports every module in the package, so the wrapper registers by existing (a duplicate HF class name raises at import).

**1. Write the wrapper.** Subclass `EPMoELayerBase` — or `EPSharedExpertsMoELayerBase` when the forward is the standard fp32-router → route → shared-experts shape (LFM-2, Cohere2 MoE), `EPGroupLimitedMoELayerBase` on top of it when routing also carries the six DeepSeek-V3 group-limited knobs (GLM-4, Laguna, Mistral4, GLM-5 Next, Step-3.7), or `EPSeparateGluMoELayerBase` when the experts are stored as separate gate/up/down tensors and exported per expert (Qwen3, Bailing). Crib the closest layout:

| Layout | Closest wrapper |
|---|---|
| Per-expert `nn.ModuleList`, fused to 3D at init | `EPBailingMoELayer` |
| Pre-fused `gate_up_proj [E, 2M, H]`, split into gate/up at init | `EPQwen3MoELayer` |
| Pre-fused `gate_up_proj`, kept fused (contiguous `[gate; up]` halves) | `EPQwen3_5MoELayer`, `EPGlm4MoELayer`, `EPDeepseekV4MoELayer` |
| Interleaved `[g0, u0, g1, u1, …]` (+ expert biases) | `EPGptOssMoELayer` |
| Sigmoid routing, group mask | `EPGlm4MoELayer` |
| Sigmoid routing, expert bias | `EPLfm2MoELayer` |
| Router outside the MoE block (pre-weighted indices arrive) | `EPGemma4MoELayer` |
| Hash + top-k router variants, clamped SwiGLU | `EPDeepseekV4MoELayer` |
| Top-1 + discard slot, cross-layer router state | `EPZayaMoELayer` |
| Top-k on raw logits then activate selected, averaged shared expert | `EPCohere2MoELayer` |
| Identical to a shipped family bar a routing default or a key spelling | subclass that wrapper — `EPLagunaMoELayer(EPGlm4MoELayer)` |

The base owns dispatch→compute→combine, GC dispatch replay, grad-sync hooks, grouped-GEMM/loop compute, fused-GLU param init — and `__init__` itself. That template resolves the router and the expert container off the declarations below, builds the EP state, adopts the router, then calls one hook per construction step; a family overrides a hook, never the sequence. `self.top_k` and the shared expert are set for you. The wrapper supplies `forward` (FP32 `topk_weights` — a DeepEP requirement) plus whichever hooks differ:

| Hook | Default | Overridden by |
|---|---|---|
| `_detect_hidden_dim(router, experts)` | the router projection's input width, else `gate_up_proj`'s | nothing — Gemma4 and Zaya reach the fallback by having no bare router projection |
| `_init_routing(block)` | `self.top_k = self._find_top_k(block)` | every family with more knobs: a selection function, a correction-bias buffer. Group-limited families inherit `EPGroupLimitedMoELayerBase`, whose `_init_routing` resolves the six DeepSeek-V3 knobs off block → gate → config |
| `_init_shared_experts(block)` | adopt `_SHARED_EXPERT_ATTRS` | Cohere2 (combination strategy), Qwen3.5/3.6 (a second gate module) |
| `_init_expert_compute(block, experts)` | `_resolve_activation` off the container | families owning their combine: GptOss (`alpha`/`limit`), DeepSeek-V4, GLM-5, Step-3.7 (clamp bounds), Bailing (activation on the per-expert module) |
| `_init_expert_params(experts, weights_already_sharded)` | `_init_fused_glu_params` | Qwen3 (split), Bailing (stack), GptOss (interleave) |
| `_init_summary_extras(block)` | `()` | any family with a field worth printing at construction |

A hook may read only the wrapped module and the `EPConfig`. **No hook may issue a collective** — some load paths patch layers before every rank has reached group setup, so a collective here deadlocks the job rather than failing it (DeepEP buffer creation is deferred to `create_ep_buffers()` for that reason) — and none may allocate state sized by `world_size`, which is invisible at 8 GPUs and an OOM at 512. Both are pinned per family in `tests/cpu/parallelism/test_ep_prologue_parity.py`.

**2. Declare the contract.** These class attributes are what the machinery reads — `EPDeepseekV4MoELayer` exercises most of them.

- `HF_MODULE_NAMES` — the HF MoE block class name(s) this wrapper replaces. The sole registration point.
- `_NUM_EXPERTS_ATTR_PATHS` — the family's expert-count attribute paths on the live HF module, resolved in order by the base `detect_num_experts()` before its generic container probe. Every wrapper declares or inherits one (Laguna takes GLM-4's). The config-level spellings are separate and central — `ROUTER_EXPERT_COUNT_FIELDS` / `ROUTER_TOPK_FIELDS` in `src/models/moe_balancing.py`; a family whose config invents another name extends the registry there rather than teaching one consumer about it.
- `_EXPERTS_CONTAINER_ATTRS` / `_HF_FUSED_EXPERT_KEYS` / `HF_FUSED_EXPERT_CONTRACTION_AXIS` — the attribute the block holds its experts under, the fused 3D tensor names as the *checkpoint* spells them, and which axis of those tensors contracts. `_EXPERTS_CONTAINER_ATTRS` defaults to the single `experts` every current HF block uses — declare every spelling your family accepts if it differs across revisions (GLM-4 also serves `routed_experts`); `_HF_FUSED_EXPERT_KEYS` defaults to the common fused names; the axis defaults to the base export's `F.linear` convention `[E, N, K]` and only GptOss overrides it. Getting the axis wrong block-scales the output dim on a low-precision export and corrupts every dequantized expert.
- `_EXPERT_WEIGHT_ATTR_ROOTS` — every attribute the family may store as a direct expert `nn.Parameter`, across **all** config branches (fused vs separate/ETP). The base default covers `gate_up_proj`/`gate_proj`/`up_proj`/`down_proj`; extend it when names differ (GptOss). Highest blast radius: it feeds `expert_named_params()` (grad-sync hooks, FP32 upcast, checkpoint skip-set) and the name-only weight-sync classifier. An undeclared root leaves a trainable param with no gradient sync — the construction-time scan in `src/trainers/mixins/base.py` raises on it.
- `_PER_EXPERT_UNFUSED_KEYS` **or** a `gather_expert_state_dict` override — when the hub checkpoint or vLLM loader stores one tensor per expert, declare the `(gate, up, down)` key names and the base gather splits the fused export automatically (GLM4, LFM2). Declaring both raises in `__init_subclass__`. Genuinely distinct layouts override the gather instead: GptOss re-interleaves gate/up, Gemma4 strips the `experts.` prefix.
- `_HUB_PER_EXPERT_KEYS` — the same triple, for a family whose hub checkpoint is per-expert while its gather does *not* do the split, which is exactly what `__init_subclass__` forbids combining with `_PER_EXPERT_UNFUSED_KEYS`. Two cases: families storing gate/up/down separately, which subclass `EPSeparateGluMoELayerBase` for the `_gather_individual_glu_state_dict` / `_merge_individual_glu_shards` pair (Bailing, Qwen3), and families whose gather writes the fused tensor transformers reverts on save (Qwen3.5/3.6, DeepSeek-V4, Cohere2 MoE, GLM-5 Next). `EPMoELayerBase.hub_per_expert_keys()` unions the two; a family answering `None` there is one `unfuse_moe_experts.py` refuses.
- `gather_fused_expert_state_dict` — optional, and needed only for RL against a rollout engine that loads experts the way transformers stores them (`experts.gate_up_proj` / `experts.down_proj`), i.e. `rollout_backend: sglang`. The base default returns `{}`, and a family that keeps it is refused at trainer construction under that backend, naming the class, rather than gathering a layout the engine drops or rejects mid-update. GptOss is the only family that implements it — the only one SGLang 0.5.17 loads fused. A GLU family adding one would build it on `_gather_separate_glu_full`, but only against an engine loader that verifiably consumes the fused pair: Qwen3 MoE deliberately does not, since SGLang's `qwen3_moe` loader maps per-expert names and would drop the fused keys silently. vLLM needs nothing here — it takes whatever layout `gather_expert_state_dict` already produces.
- `_EXPORT_KEY_RENAMES` — `(module spelling, hub spelling)` pairs for a family whose in-library module names differ from its checkpoint's. Transformers applies its own `WeightRenaming` entries only inside `from_pretrained`; the gather, the RL weight sync and the lazy loader all bypass that. Undeclared, the export writes keys vLLM silently skips and the lazy loader leaves that submodule randomly initialized (Laguna's `shared_expert` ↔ `shared_experts`).
- `_supports_weight_sync` / `_supports_gradient_checkpointing` / `_supports_lazy_loading` — all default `True`. Set one `False` and the owning gate rejects loudly instead of corrupting silently. Which family switches off which flag is published, pinned against the classes, in [Per-family EP restrictions](../parallelism/expert-parallelism.md#per-family-ep-restrictions).
- `_supports_bias_balancing` — `True` when routing *selection* happens in-layer: add `self._balancing_bias(scores)` before top-k, gather gate weights from the **unbiased** scores, and call `self._record_expert_load(indices)` (`_deepseek_biased_route` does the whole pattern for logit-routed families). A layer can refuse per-instance by overriding `enable_bias_balancing` (DeepSeek-V4 hash layers). Leave `False` when the router sits outside the wrapper (Gemma4) or the family's own gate owns a native balancing buffer (Zaya). An explicit `bias_update` on a model where no layer accepts the bias raises. Declaring support is also one of the three signals `moe_balancing: auto` reads — [`auto` resolution per family](../training-methods/callbacks.md#auto-resolution-per-family).
- `_NATIVE_BALANCING_BIAS_ATTR` — dotted path to the checkpoint-persistent tensor the family's own selection consults, so the sign-updates land in exported state. Strict `bias_update` refuses a family without one, since the export would silently serve without the trained bias (`bias_update_transient` is the user's explicit opt-in). A config-gated slot pairs the declaration with `_NATIVE_BALANCING_CONFIG_FLAG` + the `_materialize_native_balancing_slot` hook (LFM-2).
- `_ep_severs_aux_loss` — `True` when the wrapper bypasses the HF router *module* entirely, so `outputs.router_logits` never populates and the HF aux-loss path is dead under EP. `moe_balancing: auto` then resolves the family to `bias_update` (`resolve_balancing_mode` in `src/models/moe_balancing.py`, applied by `src/distributed/expert_parallel/balancing_strategy.py`).
- `_supports_routing_replay` — defaults `True`; set `False` where the layer cannot re-derive gate weights at a forced selection (Gemma4's sibling router, Zaya's EDA cross-layer state). The replay injector also rejects any layer that never set `self.top_k`.
- `_SHARED_EXPERT_ATTRS` / `_SHARED_EXPERT_REQUIRED` — the attribute name(s) the block may hold its shared expert under, tried in order; the first the block carries is adopted and becomes `_shared_expert_attr`, which is the *export* key. Declare every spelling the family accepts across revisions (GLM-4 serves both `shared_experts` and the singular `shared_expert`), and a family whose block always builds one sets `_SHARED_EXPERT_REQUIRED = True` so a rename raises instead of dropping the FFN from every output. `replicated_named_params()` is derived from the adopted name, so a family with the ordinary single shared expert writes no override; replicated submodules are FSDP-ignored and DP-averaged by the router hook, and an undeclared one trips the same construction-time scan. Override `replicated_named_params()` only when the replicated set is a different *shape* — Qwen3.5/3.6, whose gate is a second module.
- `_ROUTER_ATTR` / `_DEFAULT_ACTIVATION` — the attribute the router hangs off on both the block and the wrapper (default `gate`; GptOss says `router`), and the activation name the resolution chain falls back to when neither the container nor the block names one (default `silu`; Gemma4 says `gelu_pytorch_tanh`, since its config spells the field `hidden_activation`). `_ROUTER_ATTR = None` declares the router *external* to the wrapper — Gemma4's sibling module: nothing is adopted, no routing knob is read, and only expert grad-sync hooks register.
- `_fused_glu_mul` — the single seam for the GLU combine. `_resolve_activation` latches it from the resolved activation; a family with a non-standard combine (DeepSeek-V4 and GLM-5 Next's clamped SwiGLU, Step-3.7's post-activation clamp) rebinds it in `_init_expert_compute` — a `functools.partial` over the kernel with the family's bound, or a bound method for a gate no kernel implements. Latch it rather than overriding `_glu_combine`: the construction summary names the latched callable, so an override makes the reported combine and the running one two declarations that can disagree.
- `_OPTIONAL_ROUTING_KNOBS` — for a `EPGroupLimitedMoELayerBase` family only: the routing knobs (any spelling) its block, router and config genuinely do not declare, so the shared `_init_routing` may substitute a neutral default. Everything else is required — an upstream rename raises instead of silently under-scaling every routed weight. Laguna and Step-3.7 carry no group limiting and no `norm_topk_prob`; no family opts out of `routed_scaling_factor`.

**3. Expert LoRA.** A new stored weight root needs a `LORA_PROJECTION_COVERAGE` entry (`src/distributed/expert_parallel/config.py`) mapping it to the logical `gate`/`up`/`down` projection — without it `expert_lora` silently never adapts that weight (`tests/cpu/peft/test_expert_lora_projection_coverage.py` fails on any registered root missing an entry).

**4. Sharded-save merge.** `save_sharded_ep=True` needs the class to declare `HF_MODEL_TYPES` — every `config.model_type` spelling its checkpoints carry, including text-tower and remote-code variants. That union is what resolves a checkpoint back to its layer class (`resolve_ep_merge_layer_class`, `src/distributed/expert_parallel/expert_weights.py`); a `model_type` no registered family claims **raises** at trainer construction (`validate_ep_sharded_save`). Drop `save_sharded_ep` and take the gathered save, which needs no transform.

The merge itself is `merge_shards_to_hf` on the class. The base fused-GLU implementation covers families that keep the base gather; a family with a distinct layout overrides it **alongside** `gather_expert_state_dict`, since they are inverses and `__init_subclass__` raises if only one is overridden. Add a roundtrip test in `tests/cpu/checkpoint/test_merge_ep_shards.py`.

**5. Buffers.** `finalize_loaded_model()` (`src/models/patches/buffer_fixes.py`) is the post-load repair every load path runs: it walks two fixer chains — `_ROTARY_FIXERS` for `inv_freq`, `_NON_PERSISTENT_FIXERS` for the rest — and re-ties shared weights. transformers 5 re-materializes every non-persistent buffer as `torch.empty_like`, so add a fixer to the matching chain for any the family carries; an uncovered one reaches device placement still on meta, where the trainer raises rather than training on uninitialized memory.

**6. Tests.** The DeepSeek-V4 trio is the template: an EP-vs-FSDP equivalence test (`tests/gpu/parallelism/ep/test_ep_vs_fsdp_deepseek_v4.py`), a trainer test that saves via the gathered EP path and reloads the checkpoint as a plain HF model (`tests/gpu/trainers/sft/test_sft_deepseek_v4_moe.py`), both registered in `tests/gpu/manifest.py`, plus a CPU test covering registration, gather/merge layout, and balancing resolution (`tests/cpu/models/test_deepseek_v4_support.py`).

**7. Wire the docs.** A supported family gets `agent-docs/models/<name>.md`, a matrix row in [Supported Models](README.md), an entry in the models `README.md`, the `CLAUDE.md` model lists, and an `examples/sft/<family>/` config.

## Add CP support

Full procedure in [Context Parallelism — Adding a new model](../parallelism/context-parallelism.md#adding-a-new-model). The machinery lives in `src/distributed/context_parallel/`.

1. **Write a wrapper** under `layers/` and declare `HF_MODULE_NAMES`. Two templates: subclass `UlyssesAttentionBase` for the optimized path (RoPE before all-to-all, native GQA — copy `Qwen3MoeUlyssesAttention`), or `MLAUlyssesAttentionBase` when Q/K and V head dims differ (copy `Glm4MoeLiteUlyssesAttention`). There is no accept list to edit: `layers/registry.py` imports every module in the package and derives `WRAPPER_CLASS_MAP` — and `CP_SUPPORTED_ATTENTION_CLASSES` from it — by walking the `UlyssesAttentionBase` subclass tree, so the wrapper registers by existing and a duplicate HF name raises.
2. **Runtime requirements**: a real Flash Attention impl (`SUPPORTED_ATTN_IMPLEMENTATIONS` in `validation.py`) unless the wrapper declares `REQUIRES_FLASH_ATTN_LABEL = False` (modeling code that cannot carry a flash label, e.g. Bailing), and both `num_attention_heads` and `num_key_value_heads` divisible by `cp_size`. Override `_resolve_scaling` / `_output_projection` when the module computes its scale inline or names its output projection something other than `o_proj`.
3. **Test** with `tests/gpu/parallelism/cp/test_cp_correctness.py`.

The wrapper must split Q/K/V into the local-rank slice before all-to-all, apply RoPE on the correct side of it for the layout, run the output projection locally, finish with a final all-to-all, pass position IDs through the chunk boundary, and slice any per-head buffers (e.g. GPT-OSS sinks) to this rank's head range. CP runs only on full-attention blocks — linear-attention and gated-delta layers are rejected by `_UNSUPPORTED_SEQUENCE_AXIS_LAYERS`.

## Add TP support

- **Model already has `base_model_tp_plan`** — `tp_plan="auto"` does everything (Llama, Mistral, most Qwen variants). Nothing to add.
- **Selective TP for EP+TP** — add the attention class to `TP_SHARDABLE_ATTENTION_CLASSES` in `src/distributed/tensor_parallel/module_types.py`. The selective path only visits attention submodules, so MoE blocks need no opt-out; `embed_tokens` and `lm_head` stay replicated. If the attention has a non-standard shape (GLM-4's LoRA-style compression, GPT-OSS's sinks), add a case in `parallelize_attention.py`; a non-shardable attention (DeepSeek-V4's shared-KV MQA) stays off the list, and `apply_tp_to_attention_only` raises when it ends up patching zero modules.
- MLA attention (a `q_b_proj` / `kv_b_proj` pair) is handled by that plan already: only the expansions shard, the KV-head divisibility check is skipped, and `register_mla_rope_grad_reduction` sums the per-rank-partial rope rows of `kv_a_proj_with_mqa` in backward. A hybrid stack shards its full-attention layers only, so its per-rank footprint falls by far less than `1/tp_size`.

## Vendoring a model

If a model isn't in transformers yet, or its `trust_remote_code` conflicts with the v5 pin, vendor it — and remove the vendoring at step 5 once upstream ships the family.

1. **Copy `configuration_*.py` and `modeling_*.py`** into `src/models/<name>/`.
2. **Register in a module**, not in the package `__init__.py` — every package init under `src/` carries a docstring only (`tests/cpu/conventions/test_package_inits.py`). Put the calls in `src/models/<name>/registration.py`:

    ```python
    AutoConfig.register("<name>", YourConfig)
    AutoModelForCausalLM.register(YourConfig, YourForCausalLM)
    ```

3. **Trigger registration** by adding a module-level side-effect import of that module to `src/models/loading/model_preparation.py`, which every loader imports before it resolves an `Auto*` class (that is how the toolkit's seq-cls heads in `src/models/seq_cls_heads.py` register). `apply_remote_code_compat_shims()` restores removed transformers APIs (`is_torch_fx_available`, `ROPE_INIT_FUNCTIONS['default']`), binds names remote files call without importing (`DynamicLayer`), and patches every loaded remote module for dispatchless eager attention and legacy tied-weight keys.
4. **Add a CPU test** — `tests/cpu/models/test_<name>_registration.py` verifying `AutoModelForCausalLM.from_pretrained` resolves to your class.
5. **Remove the vendoring** once upstream lands — the directory, its registration import, and the registration test all go, and the EP wrapper, Liger spec and any patches re-key to the upstream module.

HF models that only need `trust_remote_code=True` (Bailing/Ling, Laguna) take the shims path with no vendoring.

## Declare Liger coverage

Upstream Liger's `MODEL_TYPE_TO_APPLY_LIGER_FN` doesn't cover every supported model, and covers some only in part. A family that neither registry covers warns at model load that the run trains unfused, and refuses an explicit `liger_kernel_config` key. Add one `LigerFamilySpec` to `src/kernels/liger/families.py` — there is no per-family applier module:

1. **Name the classes that fill each role** — `rms_norm`, `gated_rms_norm` (a linear-attention block's `norm(x) * w * act(gate)`, served by `fla`), `glu_mlp`, `causal_lm` — plus the variant parameters (`rms_norm_offset` / `rms_norm_casting_mode` for a Gemma-style `(1 + w)` norm, `logit_scale_attr` for a head that scales its logits, `rope=True` only for a full-width `rotate_half` rotary). A `trust_remote_code` family sets `remote_classes` instead of `modeling_module`; its patch fires when transformers loads the modeling file.
2. **Declare a role only where the kernel is EXACT.** A clamped GLU, a trained output scale, a weightless or gated norm, a partial or interleaved rotary, a head that truncates the vocab — each would compute a different function silently. Leave it off and say why in the spec comment.
3. **Set `flce_default=True`** only when the `[seq, vocab]` logits plane is the family's binding memory limit; the generic default keeps logits for metrics.
4. **Set `delegates_to_upstream=True`** when upstream already covers the family and the spec only ADDS to it (Qwen3.5/3.6, Qwen3-Next). Upstream's applier runs first with every flag it declares; the spec names only the roles it leaves eager — a `gated_rms_norm`, and a `glu_mlp` where upstream's `swiglu` swaps the routed-experts class but not the dense or shared-expert MLPs. Three derived guards hold it together: the upstream applier is resolved from liger-kernel's own registry by `model_types`, so a family it stops covering, aliases that stop sharing one applier, or a flag the added role rides that left its signature all raise at import; the spec must add a role of its own to a native modeling module; and a named class upstream has already replaced raises at patch time, read off the class's own `__module__`, which is what keeps the two swaps apart. `cross_entropy` there is upstream's process-wide rebind, not the toolkit's scoped patch.
5. **The CPU suite gates it**: `tests/cpu/kernels/test_liger_family_coverage.py` checks every roster family resolves an applier, that each declared class exists, that each declared norm is what the chosen Liger variant computes, and that each declared MLP and head is the canonical body; `test_liger_upstream_delegation.py` pins the delegation seam. Add the GPU numerics case to `tests/gpu/kernels/test_liger_family_kernels.py`.
