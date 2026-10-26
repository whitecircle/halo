# Adding a New Model

| Goal | Track |
|---|---|
| MoE model, need EP | [Add EP support](#add-ep-support) |
| Long-context dense or MoE, need CP | [Add CP support](#add-cp-support) |
| Large model, want TP beyond what `tp_plan` covers | [Add TP support](#add-tp-support) |
| Model not in transformers yet | [Vendoring a model](#vendoring-a-model) |
| Family the auto-detected attention backend cannot run | [Attention backend](#attention-backend) |
| Forward scales, caps or cuts the logits around `lm_head` | [Declare the head transform](#declare-the-head-transform) |

For plain FSDP, no work is needed — any `AutoModelForCausalLM` works.

Pipeline parallelism is an allowlist: a new backbone splits under PP only once its class is in a `validated=True` `PPFamilySpec` in `PP_FAMILY_SPECS` (`src/distributed/pipeline_parallel/families.py`; `STOCK_FORWARD_PP` when the upstream forward splits as-is), added after its PP GPU rows pass. Until then `reject_unsupported_pp_structure` refuses it by name.

Multimodal models work too: the loaders resolve the class through `resolve_auto_model_class()` (`src/model_loading/model_preparation.py`), so any `model_type` in transformers' image-text-to-text mapping loads via `AutoModelForImageTextToText` rather than dropping the vision tower. The training scripts detect a VLM from its config and route the load through `load_distributed_model`, so a MoE VLM gets EP/CP/TP on the multimodal wrapper (`has_routed_experts` finds the experts on the nested `text_config`). No EP-wrapper work is VLM-specific.

## Attention backend

Auto-detection covers a family whose attention the installed kernels already run. It falls short on a head dim no flash kernel accepts, a NaN-prone backward, attention sinks, or position IDs the modeling never forwards.

Two seams handle those. `resolve_attn_implementation` (`src/models/attention/selection.py`) narrows the backend from the family's own capabilities; `apply_family_patches` (`src/model_loading/model_preparation.py`) applies the family's patches, keyed on that backend, to trainable and frozen loads alike, and returns the implementation both loaders build with (the backend, or a variant of it such as Gemma 4's `sdpa_flex_sliding`).

Add the family predicate beside the existing ones and wire it into whichever seam applies. The per-family matrix is in [Flash Attention](../optimization/flash-attention.md#model-specific-handling).

## Declare the head transform

The chunked GRPO log-probs (`use_chunked_grpo_logprobs`) compute logits from the backbone's hidden state instead of calling `*ForCausalLM.forward`, as does the PP stage seam. Whatever that forward applies around `lm_head` — a hidden-state scale, a logit scale or division, a softcap, a vocabulary cut — has to be declared, or the chunked path refuses the family.

Declare it with a `HeadTransformSpec` subclass in `src/models/head_transform.py`: claim the causal-LM class names in `HF_MODULE_NAMES` and build the `HeadTransform` from the config in `transform`. Resolution walks the class's MRO. A family with no spec gets the base, which applies `final_logit_softcapping` where the config sets it.

The declaration is verified, never trusted: `verify_head_transform` runs the family's own forward on a meta-device shell whose backbone emits a fixed hidden state and whose output embedding is a small stand-in, and raises when the declaration does not reproduce the logits. `tests/cpu/models/test_head_transform.py` runs that check over every roster class, the MoE ones derived from the EP layer registry, and compares the chunked sweep's log-probs and gradients with each listed tiny model's forward; add the family's tiny model there, and a dense family to the dense roster.

## Add EP support

EP mechanics are in the [Expert Parallelism guide](../parallelism/expert-parallelism.md). A new family is three pieces:

- a frozen `MoEFamilySpec` constant in `src/models/moe/family_specs.py`, listed in `MOE_FAMILY_SPECS` — the family's facts, which offline tools, config writers, weight sync and the loaders read by `config.model_type` without importing EP;
- one wrapper class in `src/distributed/expert_parallel/families/<name>.py` declaring `FAMILY = <SPEC>`;
- one import line in `families/roster.py`, which builds `MOE_LAYER_MAP` from the imported classes' `HF_MODULE_NAMES` (a duplicate HF class name raises at import).

`tests/cpu/models/test_moe_family_specs.py` holds every spec to exactly one family class, and the spec index refuses a `model_type` two specs claim.

**1. Write the wrapper.** Subclass `EPSharedExpertsMoELayerBase` when the forward is the routed template — route the flattened tokens, then routed plus shared experts — and supply `route_tokens_to_experts` over the fp32 router logits (LFM-2, Cohere2 MoE) or override `_route_hidden_states` where the router routes itself (Qwen3, Qwen3.5, Bailing, DeepSeek-V4's hash layers). `EPGroupLimitedMoELayerBase` sits on top of it when routing carries the six DeepSeek-V3 group-limited knobs (GLM-4, Laguna, Mistral4, GLM-5 Next, Step-3.7). Subclass `EPMoELayerBase` directly only for a forward of another shape (GPT-OSS, Gemma4, Inkling, Zaya). Crib the closest layout:

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

The base (`moe_layer/layer.py`, with its compute, routing, grad-sync, expert-LoRA and balancing-bias mixins) owns dispatch→compute→combine, GC dispatch replay, grad-sync hooks, grouped-GEMM/loop compute, fused-GLU param init — and `__init__` itself. That template resolves the router and the expert container off the declarations below, builds the EP state, adopts the router, then calls one hook per construction step. A family overrides a hook, never the sequence.

`self.top_k` and the shared expert are set for you. The wrapper supplies `forward` (FP32 `topk_weights` — a DeepEP requirement) plus whichever hooks differ:

| Hook | Default | Overridden by |
|---|---|---|
| `_detect_hidden_dim(router, experts)` | the router projection's input width, else `gate_up_proj`'s | nothing — Gemma4 and Zaya reach the fallback by having no bare router projection |
| `_adopt_routing(block)` | `self.top_k = self._find_top_k(block)` | every family with more knobs: a selection function, a correction-bias buffer. Group-limited families inherit `EPGroupLimitedMoELayerBase`, whose `_adopt_routing` resolves the six DeepSeek-V3 knobs off block → gate → config |
| `_adopt_shared_experts(block)` | adopt `FAMILY.shared_expert_attrs` | Cohere2 (combination strategy), Qwen3.5/3.6 (a second gate module) |
| `_adopt_expert_compute(block, experts)` | `_resolve_activation` off the container | families owning their combine: GptOss (`alpha`/`limit`), DeepSeek-V4, GLM-5, Step-3.7 (clamp bounds), Bailing (activation on the per-expert module) |
| `_adopt_expert_params(experts, weights_already_sharded)` | `_init_fused_glu_params` | Qwen3 (split), Bailing (stack), GptOss (interleave) |
| `_init_summary_extras(block)` | `()` | any family with a field worth printing at construction |

A hook may read only the wrapped module and the `EPConfig`. **No hook may issue a collective**: some load paths patch layers before every rank has reached group setup, so a collective here deadlocks the job rather than failing it (DeepEP builds its buffer lazily, at the first dispatch, for that reason).

No hook may allocate state sized by `world_size` either — invisible at 8 GPUs, an OOM at 512. Both rules are pinned per family in `tests/cpu/parallelism/test_ep_prologue_parity.py`.

**2. Declare the spec.** The `MoEFamilySpec` fields every family-keyed reader consults; a field left out takes the default most families share.

- `model_types` — every `config.model_type` its checkpoints carry, the family's own first, then the composite wrappers, text towers and remote-code spellings. That union resolves a checkpoint back to its family offline (sharded merge, lazy load, quantize, unfuse, config export).
- `expert_weight_roots` — every attribute the family may store as a direct expert `nn.Parameter`, across **all** config branches (fused vs separate/ETP); the default `GLU_PROJECTIONS` covers `gate_up_proj`/`gate_proj`/`up_proj`/`down_proj`, GptOss extends it.

    Highest blast radius: it feeds `expert_named_params()` (grad-sync hooks, FP32 upcast, checkpoint skip-set) and the name-only weight-sync classifier. An undeclared root leaves a trainable param with no gradient sync — the construction-time scan (`reject_unsynced_trainable_params`, `src/trainers/parallelism/mode_setup.py`) raises on it.

- `experts_container_attrs` / `fused_contraction_axis` — the attribute the block holds its experts under (default `experts`; declare every spelling across revisions, as GLM-4's `routed_experts`), and which axis of the fused 3D tensors contracts (the base export's `F.linear` `[E, N, K]`; only GptOss says `1`). The wrong axis block-scales the output dim on a low-precision export and corrupts every dequantized expert.
- `shared_expert_attrs` — the attribute name(s) the block may hold its shared expert under, tried in order; the first the block carries becomes `_shared_expert_attr`, the *export* key. Declare every spelling across revisions (GLM-4 serves `shared_experts` and `shared_expert`).
- `export_key_renames` — `(module spelling, hub spelling)` pairs for a family whose in-library module names differ from its checkpoint's inside an EP layer. The gathered save and the RL weight sync revert transformers' own `WeightRenaming` entries off the load's record, but `merge_ep_shards.py`, which streams key by key with no model behind it, reads only these pairs, as does the EP layers' own gather. Undeclared, a merged save keeps the module spelling, which serving engines skip (Laguna's `shared_expert` ↔ `shared_experts`).
- `balancing_slot` — the checkpoint-exported routing-bias slot, every spelling (live first, then the hub respelling) plus the config flag that gates it (LFM-2 `use_expert_bias`). Strict `bias_update` refuses a family without one.
- `hub_conversion_keys` — transformers' conversion-mapping keys for a vendor-namespace hub checkpoint. Each must resolve to entries in the pinned transformers, or the lazy load refuses; a text-only class transformers maps no conversions for plans with them.
- `exports_source_config_schema` — exports carry the source repo's `config.json` schema and `auto_map` modules, for a family no pinned rollout server parses through transformers' own class ([Checkpoints](../reference/checkpoints.md#exported-configjson)).
- `supports_lazy_loading` / `supports_gradient_checkpointing` / `supports_routing_replay` / `supports_fp32_non_ep_params` — all default `True`; set one `False` and the owning gate rejects loudly instead of corrupting silently. `weight_sync_refusal` — `None`, or the reason no end-to-end weight sync into the family's served model is validated. Which family switches off which is published, pinned against the specs, in [Per-family EP restrictions](../parallelism/expert-parallelism.md#per-family-ep-restrictions).

**3. Declare the class contract.** These class attributes are what the layer machinery reads — `EPDeepseekV4MoELayer` exercises most of them.

- `HF_MODULE_NAMES` — the HF MoE block class name(s) this wrapper replaces. The sole registration point.
- `_NUM_EXPERTS_ATTR_PATHS` — the family's expert-count attribute paths on the live HF module, resolved in order by the base `detect_num_experts()` before its generic container probe. Every wrapper declares or inherits one (Laguna takes GLM-4's).

    The config-level spellings are separate and central: `resolve_expert_count` / `resolve_router_topk` in `src/models/moe/config_fields.py`. A family whose config invents another name extends the spellings there rather than teaching one consumer about it.

- `MoEFamilySpec.expert_layout` — the family's one `ExpertLayout` (`src/models/moe/expert_layout.py`), which the EP gather, the RL weight sync and `merge_ep_shards.py` all apply through `to_hub_expert_tensors`:

    | Kind | Hub layout | Families |
    |---|---|---|
    | `FUSED` (default) | fused `experts.gate_up_proj` / `down_proj`, `F.linear` convention; `key_prefix=""` drops `experts.` where the wrapper replaces the container itself | Cohere2 MoE, DeepSeek-V4, GLM-5 Next, Qwen3.5/3.6, Mistral4, Inkling, Step-3.7, Zaya; Gemma4 bare |
    | `FUSED_SPLIT_PER_EXPERT` | the fused bank split into `experts.{i}.<gate\|up\|down>.weight` | GLM-4, Laguna, LFM-2 (`w1`/`w3`/`w2`) |
    | `SEPARATE_GLU` | gate/up/down stored apart, written per expert | Qwen3, Bailing |
    | `INTERLEAVED_GLU` | GPT-OSS's interleaved `[g0, u0, ...]` pair with biases, matmul convention | GPT-OSS |

    `per_expert_keys` names the `(gate, up, down)` per-expert spelling. A `FUSED` family declares it where transformers also loads that spelling — its hub's for DeepSeek-V4, Cohere2 MoE and GLM-5 Next, the one `unfuse_moe_experts.py` writes for Qwen3.5/3.6; a family without one is what that tool refuses. A runtime tensor a layout has no branch for raises rather than dropping out of the export.

- **Rollout-engine support is not a family declaration.** Both pinned engines read the layout the family's `ExpertLayout` writes, so nothing here is per engine.

    Which families an engine can take an *online* update for lives on its weight-sync client (`UNSERVABLE_MODEL_TYPES` in `src/weight_sync/clients/`), one loader fact per `model_type`. Add an entry there when the engine's loader cannot take the chunked update; `reject_unsyncable_model` then refuses the pair at trainer construction quoting it ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)).

- `SUPPORTS_BIAS_BALANCING` — `True` when routing *selection* happens in-layer. Select through the shared helpers, not a hand-added `self._balancing_bias(...)`, which is `None` when balancing is off and in native-adoption mode (the adopted slot already sits in the selection): score-routed layers take top-k over `self._selection_scores(scores)` (the group-limited base, DeepSeek-V4, Inkling, LFM-2); logit-routed ones use `self._biased_topk(logits)` (Qwen3, Cohere2 MoE) or `_deepseek_biased_route`, which also returns the unbiased gate (GPT-OSS, Qwen3.5). Gate weights come from the **unbiased** scores, and the caller records the load with `self._record_expert_load(indices)`.

    A layer can refuse per-instance by overriding `enable_bias_balancing` (DeepSeek-V4 hash layers). Leave `False` when the router sits outside the wrapper (Gemma4) or the family's own gate owns a native balancing buffer (Zaya). An explicit `bias_update` on a model where no layer accepts the bias raises.

    Declaring support is also one of the three signals `moe_balancing: auto` reads — [`auto` resolution per family](../training-methods/callbacks.md#auto-resolution-per-family).

- `MoEFamilySpec.balancing_slot` — the family's exported balancing slot: its first spelling is the live-tree path the layer adopts, so the sign-updates land in exported state (`bias_update_transient` is the user's explicit opt-in where there is none), and its `config_flag` the field flipped for a slot the config gates. A config-gated slot also implements the `_materialize_native_balancing_slot` hook (LFM-2).

- `SEVERS_AUX_LOSS` — `True` when the wrapper bypasses the HF router *module* entirely, so `outputs.router_logits` never populates and the HF aux-loss path is dead under EP. `moe_balancing: auto` then resolves the family to `bias_update` (`resolve_balancing_mode` in `src/models/moe/balancing.py`, applied by `src/callbacks/balancing.py`).
- `_SHARED_EXPERT_REQUIRED` — `True` for a family whose block always builds its shared expert, so a rename raises instead of dropping the FFN from every output.

    `replicated_named_params()` is derived from the adopted name, so a family with the ordinary single shared expert writes no override. Replicated submodules are FSDP-ignored and DP-averaged by the router hook, and an undeclared one trips the same construction-time scan. Override `replicated_named_params()` only when the replicated set is a different *shape* — Qwen3.5/3.6, whose gate is a second module.

- `_ROUTER_ATTR` / `_DEFAULT_ACTIVATION` — the attribute the router hangs off on both the block and the wrapper (default `gate`; GptOss says `router`), and the activation name the resolution chain falls back to when neither the container nor the block names one.

    That activation default is `silu`. Gemma4 says `gelu_pytorch_tanh`, since its config spells the field `hidden_activation`.

    `_ROUTER_ATTR = None` declares the router *external* to the wrapper — Gemma4's sibling module. Nothing is adopted, no routing knob is read, and only expert grad-sync hooks register.

- `_fused_glu_mul` — the single seam for the GLU combine. `_resolve_activation` latches it from the resolved activation.

    A family with a clamped SwiGLU (DeepSeek-V4 and GLM-5 Next clamp before the activation, Step-3.7 after it) rebinds it in `_adopt_expert_compute` to `resolve_clamped_glu_mul` (`src/kernels/fused_glu.py`): the fused kernel for a SiLU gate, the same clamps eagerly for any other.

    Latch it rather than overriding `_glu_combine`. The construction summary names the latched callable, so an override makes the reported combine and the running one two declarations that can disagree, and the fused `[gate | up]` path runs a latch's packed form without calling `_glu_combine` at all.

- `_OPTIONAL_ROUTING_KNOBS` — for a `EPGroupLimitedMoELayerBase` family only: the routing knobs (any spelling) its block, router and config genuinely do not declare, so the shared `_adopt_routing` may substitute a neutral default.

    Everything else is required — an upstream rename raises instead of silently under-scaling every routed weight. Laguna and Step-3.7 carry no group limiting and no `norm_topk_prob`; no family opts out of `routed_scaling_factor`.

**4. Expert LoRA.** A new stored weight root needs a `LORA_PROJECTION_COVERAGE` entry (`src/models/moe/expert_lora_spec.py`) mapping it to the logical `gate`/`up`/`down` projection — without it `expert_lora` silently never adapts that weight (`tests/cpu/peft/test_expert_lora_projection_coverage.py` fails on any registered root missing an entry).

**5. Sharded-save merge.** `merge_ep_shards.py` resolves a checkpoint to its spec through `find_moe_family_spec(model_type)` and writes the spec's `ExpertLayout`, the same transform the gathered save applies, so a family needs nothing more; a `model_type` no spec claims **raises** at trainer construction under `save_sharded_ep=True` (`reject_unmergeable_ep_sharded_save`). Add the family to `test_full_merge_matches_gathered_save` in `tests/cpu/checkpoint/test_merge_ep_shards.py`.

**6. Buffers.** `finalize_loaded_model()` (`src/model_loading/buffer_repair.py`) is the post-load repair every load path runs: it walks two fixer chains — `_ROTARY_FIXERS` for `inv_freq`, `_NON_PERSISTENT_FIXERS` for the rest — and re-ties shared weights. transformers 5 re-materializes every non-persistent buffer as `torch.empty_like`, so add a fixer to the matching chain for any the family carries; an uncovered one reaches device placement still on meta, where the trainer raises rather than training on uninitialized memory.

**7. Tests.** The DeepSeek-V4 trio is the template: an EP=2-vs-stock row (`tests/gpu/parallelism/ep/test_ep_family_vs_reference.py --family deepseek_v4`, generated from `TINY_MOE_FAMILIES`; family-specific checks go in its `FAMILY_CHECKS`), a trainer test that saves via the gathered EP path and reloads the checkpoint as a plain HF model (`tests/gpu/trainers/sft/test_sft_ep_roundtrip.py --family deepseek_v4`, a thin subclass of `EPSftRoundTrip` in `tests/common/ep_sft_roundtrip.py` — add one beside it and override only the family hooks), both registered in `tests/gpu/manifest.py`, plus a CPU test covering registration, gather/merge layout, and balancing resolution (`tests/cpu/models/test_deepseek_v4_support.py`). The family's tiny model also goes into `TINY_MOE_FAMILIES` (`tests/common/tiny_models.py`), keyed by its `model_type`, and its name into `_TINY_MOE_FAMILIES` in the manifest: the merge-on-save and precompute-resume sweeps run every entry, and `tests/cpu/conventions/test_tiny_family_roster.py` fails on a registered EP family missing from either. `tests/cpu/parallelism/test_ep_family_instantiation.py` builds every registered family's EP layer on its tiny model, and `tests/cpu/parallelism/test_ep_gc_replay.py` replays its dispatch under gradient checkpointing for every family whose layer supports it.

**8. Wire the docs.** A supported family gets `agent-docs/models/<name>.md`, an `examples/sft/<family>/` config, and a row in each family roster. Three are pinned to the registries and fail `tests/cpu/parallelism/test_docs_limitation_tables.py` until the row lands: the [compatibility matrix](README.md#compatibility-matrix), the [EP supported-models table](../parallelism/expert-parallelism.md#supported-models) and the human [Supported Matrix](../../human-docs/supported-matrix.md#model-families). The rest are kept by hand: the per-family list in [Supported Models](README.md#per-family-pages), the human guide's [Supported Models](../../human-docs/models.md) table, and the `CLAUDE.md` family list and Models index.

## Add CP support

Full procedure in [Context Parallelism — Adding a new model](../parallelism/context-parallelism.md#adding-a-new-model). The machinery lives in `src/distributed/context_parallel/`.

1. **Write a wrapper** in `families.py` and declare `HF_MODULE_NAMES`. Two templates: subclass `UlyssesAttention` for the optimized path (RoPE before all-to-all, native GQA — copy `Qwen3MoeUlyssesAttention`), or `MLAUlyssesAttention` when Q/K and V head dims differ (copy `Glm4MoeLiteUlyssesAttention`).

    There is no accept list to edit. `ULYSSES_ATTENTION_MAP` is read off the `UlyssesAttention` subclass tree, so the wrapper registers by existing and a duplicate HF name raises.

2. **Runtime requirements**: a real Flash Attention impl (`SUPPORTED_ATTN_IMPLEMENTATIONS` in `model_rules.py`) unless the wrapper declares `REQUIRES_FLASH_ATTN_LABEL = False` (modeling code that cannot carry a flash label, e.g. Bailing), and both `num_attention_heads` and `num_key_value_heads` divisible by `cp_size`.

    Override `_resolve_scaling` / `OUTPUT_PROJECTION` when the module computes its scale inline or names its output projection something other than `o_proj`.

3. **Test** with `tests/gpu/parallelism/cp/test_cp_correctness.py`.

The wrapper must split Q/K/V into the local-rank slice before all-to-all, apply RoPE on the correct side of it for the layout, run the output projection locally, finish with a final all-to-all, pass position IDs through the chunk boundary, and slice any per-head buffers (e.g. GPT-OSS sinks) to this rank's head range. CP runs only on full-attention blocks — linear-attention and gated-delta layers are rejected by `_UNSUPPORTED_SEQUENCE_AXIS_LAYERS`.

## Add TP support

- **Model already has `base_model_tp_plan`** — `tp_plan="auto"` does everything (Llama, Mistral, most Qwen variants). Nothing to add.
- **Selective TP for EP+TP** — add the attention class to `TP_SHARDABLE_ATTENTION_CLASSES` in `src/distributed/tensor_parallel/parallelize_attention.py`. The selective path only visits attention submodules, so MoE blocks need no opt-out; `embed_tokens` and `lm_head` stay replicated.

    If the attention has a non-standard shape (GLM-4's LoRA-style compression, GPT-OSS's sinks), add a case in `parallelize_attention.py`. A non-shardable attention (DeepSeek-V4's shared-KV MQA) stays off the list, and `apply_tp_to_attention_only` raises when it ends up patching zero modules.

- MLA attention (a `q_b_proj` / `kv_b_proj` pair) is handled by that plan already: only the expansions shard, the KV-head divisibility check is skipped, and `register_mla_rope_grad_reduction` sums the per-rank-partial rope rows of `kv_a_proj_with_mqa` in backward.

    A hybrid stack shards its full-attention layers only, so its per-rank footprint falls by far less than `1/tp_size`.

## Vendoring a model

If a model isn't in transformers yet, or its `trust_remote_code` conflicts with the v5 pin, vendor it — and remove the vendoring at step 5 once upstream ships the family.

1. **Copy `configuration_*.py` and `modeling_*.py`** into `src/models/<name>/`. Seed
   `HARDCODED_CONFIG_FOR_MODELS["<name>"]` (`transformers.utils.auto_docstring`) before importing the
   modeling module, or `@auto_docstring` raises "Config not found". A hub checkpoint that stores experts
   per expert while the module fuses them gets no entry in transformers' conversion mapping, so declare
   the layout on the spec (`expert_layout` with its `per_expert_keys`), which the gather and the lazy
   loader read.
2. **Register in a module**, not in the package `__init__.py` — every package init under `src/` but the `src/__init__.py` bootstrap carries a docstring only (`tests/cpu/conventions/test_package_inits.py`). Put the calls in `src/models/<name>/registration.py`:

    ```python
    AutoConfig.register("<name>", YourConfig)
    AutoModelForCausalLM.register(YourConfig, YourForCausalLM)
    ```

3. **Trigger registration** by adding a module-level side-effect import of that module to `src/model_loading/model_preparation.py`, which every loader imports before it resolves an `Auto*` class (that is how the toolkit's seq-cls heads in `src/models/seq_cls_heads.py` register).

    `apply_remote_code_compat_shims()` restores removed transformers APIs (`is_torch_fx_available`, `ROPE_INIT_FUNCTIONS['default']`), binds names remote files call without importing (`DynamicLayer`), and patches every loaded remote module for dispatchless eager attention and legacy list-form tied-weight keys, which convert to the v5 dict form; `reject_untieable_list_ties` refuses a tied config that carried one, since `{}` ties nothing.

4. **Add a CPU test** — `tests/cpu/models/test_<name>_registration.py` verifying `AutoModelForCausalLM.from_pretrained` resolves to your class.
5. **Remove the vendoring** once upstream lands — the directory, its registration import, and the registration test all go, and the EP wrapper, Liger spec and any patches re-key to the upstream module. Name the removal trigger (the upstream PR) in the docstring of `src/models/<name>/__init__.py`.

HF models that only need `trust_remote_code=True` (Bailing/Ling, Laguna) take the shims path with no vendoring.

A node's ranks share the module cache transformers copies remote code into (`HF_MODULES_CACHE`), and transformers rewrites a module there in place. `src/model_loading/remote_code.py`, which every loader, training entry point and `from_pretrained_verified` caller imports, makes each copy land whole through the checkpoint staging writes (the remote code a save writes beside a checkpoint too), so a rank never imports a module a peer is still writing.

## Declare Liger coverage

Upstream Liger's `MODEL_TYPE_TO_APPLY_LIGER_FN` doesn't cover every supported model, and covers some only in part. A family that neither registry covers warns at model load that the run trains unfused, and refuses an explicit `liger_kernel_config` key. Add one `LigerFamilySpec` to `src/model_loading/liger/family_specs.py` — there is no per-family applier module:

1. **Name the classes that fill each role** — `rms_norm`, `gated_rms_norm` (a linear-attention block's `norm(x) * w * act(gate)`, served by `fla`), `glu_mlp`, `causal_lm`.

    Then the variant parameters: `rms_norm_offset` / `rms_norm_casting_mode` for a Gemma-style `(1 + w)` norm or an fp32 weight multiply, `rms_norm_kernel="native"` to serve an offset-free norm with torch's fused `F.rms_norm` instead of Liger's, `router_aux_loss_in_head` for a head that adds the router aux loss after the projection, `rope=True` only for a full-width `rotate_half` rotary.

    A `trust_remote_code` family sets `remote_classes` (every class a role names) instead of `modeling_module`; its patch fires when transformers loads the modeling file. A native family whose hub repos also ship their own modeling file through `auto_map` (Laguna) sets both.

    List a multimodal wrapper's `model_type` only when its text tower can be nothing but this family. Otherwise the dispatch resolves it through `text_config`, and the fused loss is forced off there because the wrapper's own head runs.

2. **Declare a role only where the kernel is EXACT.** A clamped GLU, a trained output scale, a weightless or gated norm, a partial or interleaved rotary, a head that truncates the vocab — each would compute a different function silently.

    Leave it off and say why in the spec comment.

3. **Set `flce_default=True`** only when the `[seq, vocab]` logits plane is the family's binding memory limit; the generic default keeps logits for metrics.
4. **Set `delegates_to_upstream=True`** when upstream already covers the family and the spec only ADDS to it (dense Qwen3.5/3.6), or takes over a role upstream gets wrong (GptOss's norm casting, Gemma 4's EP-surviving dense MLP, the Qwen MoE `swiglu` that would also swap the routed experts — the withheld flag goes in `upstream_off` and the spec must fill that role).

    A MoE family's routed experts never run Liger's fused MoE kernel (`LigerExperts`): where upstream's `swiglu` could swap them and the spec does not withhold it, the dispatch forces the flag off whenever Halo does not wrap the experts ([Routed experts](../optimization/liger-kernels.md#routed-experts)).

    Upstream's applier runs first with every other flag it declares; the spec names only the roles it leaves eager or takes over.

    Three derived guards hold it together. The upstream applier is resolved from liger-kernel's own registry by `model_types`, so a family it stops covering, aliases that stop sharing one applier, or a flag the added role rides that left its signature all raise at import.

    The spec must add a role of its own to a native modeling module. And a named class upstream has already replaced raises at patch time, read off the class's own `__module__`, which is what keeps the two swaps apart.

    `cross_entropy` is never handed to upstream (its branch rebinds `F.cross_entropy` process-wide): the toolkit's scoped patch serves it for every family.

5. **The CPU suite gates it.** `tests/cpu/model_loading/test_liger_family_coverage.py` checks every roster family resolves an applier, that each declared class exists, that each declared norm is what the chosen Liger variant computes, and that each declared MLP and head is the canonical body.

    `test_liger_upstream_delegation.py` pins the delegation seam. Add the GPU numerics case to `tests/gpu/kernels/test_liger_family_kernels.py`.
