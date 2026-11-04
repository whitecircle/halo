# Adding a New Model

| Goal | Track |
|---|---|
| MoE model, need EP | [Add EP support](#add-ep-support) |
| Long-context dense or MoE, need CP | [Add CP support](#add-cp-support) |
| Large model, want TP beyond what `tp_plan` covers | [Add TP support](#add-tp-support) |
| Model not in transformers yet | [Vendoring a model](#vendoring-a-model) |
| Family the auto-detected attention backend cannot run | [Attention backend](#attention-backend) |
| Forward scales, caps or cuts the logits around `lm_head` | [Declare the head transform](#declare-the-head-transform) |
| Release stores a multi-token-prediction tail, or the class ignores one at load | [Declare the MTP head](#declare-the-mtp-head) |
| Family built from `qwen4_exp`'s blocks (hyper-connections, QSA, the PLE n-gram table) | [Qwen4-lineage checklist](qwen4_exp.md#qwen4-lineage-checklist) |
| transformers' conversion mapping misplaces or misses the hub's keys | [Correct the conversion mapping](#correct-the-conversion-mapping) |
| Multimodal composite whose CausalLM transformers does not map, or whose tower stores keys elsewhere | [Text-only load of a composite](#text-only-load-of-a-composite) |
| A forward the toolkit replaces, or one walking a row's documents | [Replacing an upstream forward](#replacing-an-upstream-forward) |
| New release of a family the toolkit already supports | [New release of a supported family](#new-release-of-a-supported-family) |

For plain FSDP, no work is needed — any `AutoModelForCausalLM` that fits one GPU works ([Data Parallelism](../parallelism/data-parallelism.md#fsdp2-strategy-by-mode)).

Pipeline parallelism is an allowlist: a new backbone splits under PP only once its class is in a `validated=True` `PPFamilySpec` in `PP_FAMILY_SPECS` (`halo/distributed/pipeline_parallel/families.py`; `STOCK_FORWARD_PP` when the upstream forward splits as-is), added after its PP GPU rows pass (a MoE joins `_PP_MOE_FAMILIES` in `tests/gpu/manifest.py`). Until then `reject_unsupported_pp_structure` refuses it by name.

Multimodal models work too: the loaders resolve the class through `resolve_auto_model_class()` (`halo/model_loading/model_preparation.py`), so any `model_type` in transformers' image-text-to-text mapping loads via `AutoModelForImageTextToText` rather than dropping the vision tower. The training scripts detect a VLM from its config and route the load through `load_distributed_model`, so a MoE VLM gets EP/CP/TP on the multimodal wrapper (`has_routed_experts` finds the experts on the nested `text_config`). No EP-layer work is VLM-specific. A prefix-LM family, whose decoder reads the prefix/suffix split from `token_type_ids` (PaliGemma), is declared in `PREFIX_LM_MODEL_TYPE_PREFIXES` (`halo/models/attention/prefix_lm.py`), which keeps it off flash kernels and has the VLM collators write that split ([SFT → Vision-language models](../training-methods/sft.md#vision-language-models)).

A composite's towers need no declaration of their own. FSDP2 groups its input embedding with the module whose forward reads it, under `.model` or at the wrap's root for a composite holding `language_model` itself ([Data Parallelism](../parallelism/data-parallelism.md#fsdp2-strategy-by-mode)). A tower's checkpointed blocks train under reentrant checkpointing whether or not it declares `get_input_embeddings` ([Tower blocks](../parallelism/data-parallelism.md#tower-blocks-under-reentrant-checkpointing)). A lazy load draws a checkpoint-absent tensor with the `_init_weights` of the nearest model above it, and the coverage gate reads the `_keys_to_ignore_on_load_missing` the root aggregates from its children.

## Attention backend

Auto-detection covers a family whose attention the installed kernels already run. It falls short on a head dim no flash kernel accepts, a kernel without the deterministic backward the run asks for, attention sinks, or position IDs the modeling never forwards.

Two seams handle those. `resolve_attn_implementation` (`halo/models/attention/selection.py`) narrows the backend from the family's own capabilities; `apply_family_patches` (`halo/model_loading/model_preparation.py`) applies the family's patches, keyed on that backend, and returns the implementation the model is built with (the backend, or a variant of it such as Gemma 4's `sdpa_flex_sliding`). The policy and frozen loaders and the reward-model scorer on a CUDA device call it before their build; the embedding SentenceTransformer load and a trainer's path-string model load (`resolve_trainer_model`) call it on every load, with `None` where no backend is pinned, which applies only the patches no backend keys.

Add the family predicate beside the existing ones and wire it into whichever seam applies. The per-family matrix is in [Flash Attention](../optimization/flash-attention.md#model-specific-handling).

## Correct the conversion mapping

transformers' `conversion_mapping` both spells a checkpoint's keys at load and reverts them at save, so a wrong entry shows up as randomly initialized weights or an export no engine reads. A family corrects it with one `ConversionOverride` in `CONVERSION_OVERRIDES` (`halo/models/conversion_overrides.py`), never inside `apply_family_patches`, which a merge tool or stock `from_pretrained` never calls. The leaf registers at import, and `checkpoint_keys.py` and `checkpoint_coverage.py` both import it, so every load, revert and tool reads the corrected mapping.

An entry names `key` (a `model_type` or class name), `upstream` (transformers' own entries as `(sources, targets)`, empty where it ships none) and `rewrite` (from those entries to the ones to register). When transformers' entries stop matching `upstream`, the import raises: a rewrite of moved entries would misplace weights with no other symptom, and it is cheapest to catch at the pin bump.

`get_model_conversion_mapping` hands registered entries to remote-code classes of the same `model_type` too, and a hub's own modeling file may register with `overwrite=True`, replacing the entry for the whole process. Run such a family's tool rows with `--trust_remote_code` on and off.

## Text-only load of a composite

`text_only_model` needs a text class to build. Where transformers' CausalLM mapping takes the composite's `model_type`, the Auto factory narrows the config itself and nothing is declared. A composite it does not map declares `text_only_layout` on its spec: `resolve_text_only_config` then hands the load its text sub-config, whose `model_type` must have a CausalLM entry. Its conversion mapping also needs the composite level dropped ([above](#correct-the-conversion-mapping)).

`CompositeTextLayout` says where the composite stores what the export keeps at the root, which `reattach_vision_tower.py` reads: its defaults are the wrapper layout (`model.*` to `model.language_model.*`, `lm_head.*` in place). A tensor under neither `composite_tower` nor `composite_head` (an MTP tail) is carried from the base.

A family that also declares `exports_source_config_schema` keeps transformers' own schema on a text-only export, since its source is the composite.

## Replacing an upstream forward

Guard each upstream signature a replacement is written for with `verify_upstream_parameters` (`halo/models/upstream_guard.py`), so a release that moved it fails at load. Where the forward calls a module-level helper, replace the helper instead of copying the forward, as `halo/models/nemotron_h.py` does for the Mamba scan. A forward walking a packed row reads `row_documents` (`halo/models/row_documents.py`), which gives each document's span and its unpadded tokens.

## Declare the head transform

The chunked GRPO log-probs (`use_chunked_grpo_logprobs`) compute logits from the backbone's hidden state instead of calling `*ForCausalLM.forward`, as does the PP stage seam. Whatever that forward applies around `lm_head` — a hidden-state scale, a logit scale or division, a softcap, a vocabulary cut — has to be declared, or the chunked path refuses the family.

Declare it with a `HeadTransformSpec` subclass in `halo/models/head_transform.py`: claim the causal-LM class names in `HF_MODULE_NAMES` and build the `HeadTransform` from the config in `transform`. Resolution walks the class's MRO. A family with no spec gets the base, which applies `final_logit_softcapping` where the config sets it.

The declaration is verified, never trusted: `verify_head_transform` runs the family's own forward on a meta-device shell whose backbone emits a fixed hidden state and whose output embedding is a small stand-in, and raises when the declaration does not reproduce the logits. `tests/cpu/models/test_head_transform.py` runs that check over every roster class, the MoE ones derived from the EP layer registry, and compares the chunked sweep's log-probs and gradients with each listed tiny model's forward; add the family's tiny model there, and a dense family to the dense roster.

## Declare the MTP head

A family whose release stores an MTP tail takes one entry in `halo/models/mtp/spec.py`: `MTP_HEAD_SPECS` where the tail is the family's own decoder layers plus a projection and norms, `MTP_REFUSALS` with the reason otherwise. `tests/cpu/models/test_mtp_head_spec.py` fails a class naming a tail in `_keys_to_ignore_on_load_unexpected` with neither.

A spec declares only what a release spells its own way: placement (`root` for `mtp.*`, `list` for the decoder list's entry at `num_hidden_layers`), the projection and norm paths, trunk copies the release stores, and a drafter that zeroes the first embedding. Layer class, index and norm class are read off the trunk.

For a spec:

- [ ] Add the release's config and tail keys to `tests/common/mtp_release_tails.json`, read from its safetensors header, so a save's keys are held to them.
- [ ] Add the tiny family to `MTP_TINY_FAMILIES` (`tests/common/mtp_heads.py`). A new placement or a family with its own head chain takes a `--mode ep` row in `test_sft_mtp_head.py` and in `test_sft_mtp_head_checkpoint.py`; a new layer class reuses the rows of its placement at `--arms soft`.
- [ ] Confirm the head trains inside the forward: `tests/cpu/models/test_mtp_head.py` holds causality and the soft identity per tiny family, and fails a head reading its own target.
- [ ] Serve a `head_only` export on each engine that takes the family: `test_sft_mtp_head_serving.py`. If either loader checks completeness, the export must write what it reads, as GLM-4's trunk copies do.
- [ ] Add the family to the table in [MTP heads](../optimization/mtp-heads.md#families) and its model page.

## New release of a supported family

A checkpoint under a `model_type` a spec already claims often loads with nothing to add. Check before training it:

1. **Read rank 0's `LOAD REPORT`.** transformers prints one where a load is not clean: an `UNEXPECTED`, `MISMATCH` or `CONVERSION` line names a key the class or the mapping does not take ([Correct the conversion mapping](#correct-the-conversion-mapping)).
2. **New config spellings** extend the registries in `halo/models/moe/config_fields.py`, and a new `model_type` extends the spec's `model_types`. A composite wrapper or text tower is a `model_type` too.
3. **A hub modeling file** (`trust_remote_code`, `model_revision` pinned) may differ from transformers' class. Read it for what the EP layer assumes: the expert and router spellings on the spec, `_SHARED_EXPERT_REQUIRED`, `_OPTIONAL_ROUTING_KNOBS` and the router's own `_TOPK_WEIGHT_NORM_EPS`.
4. **Add a recipe** under `examples/`. `tests/cpu/config/test_examples_roster.py` holds it to the cached `config.json`: expert count against `expert_parallel_size`, context window, export contract.
5. **Serving.** Read the [engine table](../infrastructure/rollout-servers.md#which-families-each-engine-serves); the release's own loader, not the family, decides whether an online update lands.

## Add EP support

EP mechanics are in the [Expert Parallelism guide](../parallelism/expert-parallelism.md). A new family is three pieces:

- a frozen `MoEFamilySpec` constant in `halo/models/moe/family_specs.py`, listed in `MOE_FAMILY_SPECS` — the family's facts, which offline tools, config writers, weight sync and the loaders read by `config.model_type` without importing EP;
- one EP layer class in `halo/distributed/expert_parallel/families/<name>.py` declaring `FAMILY = <SPEC>`;
- one import line in `families/roster.py`, which builds `MOE_LAYER_MAP` from the imported classes' `HF_MODULE_NAMES` (a duplicate HF class name raises at import).

`tests/cpu/models/test_moe_family_specs.py` holds every spec to exactly one family class, and the spec index refuses a `model_type` two specs claim.

**1. Write the EP layer.** Subclass `EPSharedExpertsMoELayerBase` when the forward is the routed template — route the flattened tokens, then routed plus shared experts — and supply `route_tokens_to_experts` over the fp32 router logits (LFM-2, Cohere2 MoE) or override `_route_hidden_states` where the router routes itself (Qwen3, Qwen3.5, Bailing, DeepSeek-V4's hash layers). `EPGroupLimitedMoELayerBase` sits on top of it when routing carries the six DeepSeek-V3 group-limited knobs (GLM-4, Laguna, Mistral4, GLM-5 Next, Step-3.7, Nemotron-H). Subclass `EPMoELayerBase` directly only for a forward of another shape (GPT-OSS, Gemma4, Inkling, Zaya). Crib the closest layout:

| Layout | Closest EP layer |
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
| Gateless experts on a latent, projected in and out around the dispatch | `EPNemotronHMoELayer` |
| Identical to a shipped family bar a routing default or a key spelling | subclass that EP layer — `EPLagunaMoELayer(EPGlm4MoELayer)`, `EPQwen3NextMoELayer(EPQwen3_5MoELayer)`, `EPQwen4ExpMoELayer(EPQwen3NextMoELayer)` |

The base (`moe_layer/layer.py`, with its compute, routing, grad-sync, expert-LoRA and balancing-bias mixins) owns dispatch→compute→combine, GC dispatch replay, grad-sync hooks, grouped-GEMM/loop compute, fused-GLU param init — and `__init__` itself. That template resolves the router and the expert container off the declarations below, builds the EP state, adopts the router, then calls one hook per construction step. A family overrides a hook, never the sequence.

`self.top_k` and the shared expert are set for you. The EP layer supplies `forward` (FP32 `topk_weights` — a DeepEP requirement) plus whichever hooks differ:

| Hook | Default | Overridden by |
|---|---|---|
| `_detect_hidden_dim(router, experts)` | the router projection's input width, else `gate_up_proj`'s | Nemotron-H, whose wire carries the experts' width, the latent where a release has one; Gemma4 and Zaya reach the fallback by having no bare router projection |
| `_adopt_routing(block)` | `self.top_k = self._find_top_k(block)` | every family with more knobs: a selection function, a correction-bias buffer. Group-limited families inherit `EPGroupLimitedMoELayerBase`, whose `_adopt_routing` resolves the DeepSeek-V3 knobs off block → gate → config |
| `_adopt_shared_experts(block)` | adopt `FAMILY.shared_expert_attrs` | Cohere2 (combination strategy), Qwen3.5/3.6 (a second gate module) |
| `_adopt_expert_compute(block, experts)` | `_resolve_activation` off the container | families owning their combine: GptOss (`alpha`/`limit`), DeepSeek-V4, GLM-5, Step-3.7 (clamp bounds), Bailing (activation on the per-expert module); Nemotron-H adds its latent projections |
| `_adopt_expert_params(experts, weights_already_sharded)` | `_init_fused_glu_params`, which stores a `SEPARATE_GLU` family's halves apart (Qwen3) | Bailing (stack), GptOss (interleave), Nemotron-H (no gate) |
| `_init_summary_extras(block)` | `()`; the group-limited base adds its resolved knobs, and every line already prints `wire_width` and `top_k` | any family with a field worth printing at construction, extending `super()` |

The forward's seams, each overridden only where the family differs:

| Hook | Default | Overridden by |
|---|---|---|
| `_route_hidden_states(flat, **kwargs)` | `route_tokens_to_experts` over `_router_logits` | a router that selects itself: Qwen3, Qwen3.5/3.6, Bailing, DeepSeek-V4 |
| `_router_logits(flat)` | the gate's logits | a router whose forward also selects: LFM-2; Nemotron-H (fp32) |
| `_gate_weights_at(logits, indices)` | required where routing replay or `_biased_route` runs | every family whose router selects or whose route is biased in the base, with the router's own arithmetic |
| `_routing_scores(logits)` | sigmoid, on the group-limited base | Mistral4 (softmax) |
| `_shared_expert_output(hidden)` | the shared expert | Qwen3.5/3.6 (sigmoid gate) |
| `_dispatch_compute_combine(...)` | dispatch, compute, combine; records the expert load | Nemotron-H (latent projections around it) |
| `_glu_ffn(x, proj)` / `_grouped_expert_ffn(...)` | the GLU FFN on either stored layout | Nemotron-H (no gate) / GptOss (three GEMMs with biases) |
| `_warm_expert_activation(gate_up)` | the combine over a synthetic `[T, 2M]` | GptOss, Nemotron-H |
| `_glu_combine_name()` | read off the `_fused_glu_mul` latch | GptOss, Nemotron-H |

A hook may read only the wrapped module and the `EPConfig`. **No hook may issue a collective**: some load paths patch layers before every rank has reached group setup, so a collective here deadlocks the job rather than failing it (DeepEP builds its buffer lazily, at the first dispatch, for that reason).

No hook may allocate state sized by `world_size` either — invisible at 8 GPUs, an OOM at 512. Both rules are pinned per family in `tests/cpu/parallelism/test_ep_prologue_parity.py`.

**2. Declare the spec.** The `MoEFamilySpec` fields every family-keyed reader consults; a field left out takes the default most families share.

- `model_types` — every `config.model_type` its checkpoints carry, the family's own first, then the composite wrappers, text towers and remote-code spellings. That union resolves a checkpoint back to its family offline (sharded merge, lazy load, quantize, unfuse, config export).
- `expert_weight_roots` — every attribute the family may store as a direct expert `nn.Parameter`, across **all** config branches (fused vs separate/ETP); the default `GLU_PROJECTIONS` covers `gate_up_proj`/`gate_proj`/`up_proj`/`down_proj`, GptOss extends it.

    Highest blast radius: it feeds `expert_named_params()` (the expert grad sync, FP32 upcast, checkpoint skip-set) and the name-only weight-sync classifier. An undeclared root leaves a trainable param with no gradient sync — the construction-time scan (`reject_unsynced_trainable_params`, `halo/trainers/parallelism/mode_setup.py`) raises on it.

- `experts_container_attrs` / `fused_contraction_axis` — the attribute the block holds its experts under (default `experts`; declare every spelling across revisions, as GLM-4's `routed_experts`), and which axis of the fused 3D tensors contracts (the base export's `F.linear` `[E, N, K]`; only GptOss says `1`). The wrong axis block-scales the output dim on a low-precision export and corrupts every dequantized expert.
- `shared_expert_attrs` — the attribute name(s) the block may hold its shared expert under, tried in order; the first the block carries becomes `_shared_expert_attr`, the *export* key. Declare every spelling across revisions (GLM-4 serves `shared_experts` and `shared_expert`).
- `export_key_renames` — `(module spelling, hub spelling)` pairs for a family whose in-library module names differ from its checkpoint's inside an EP layer. The gathered save and the RL weight sync revert transformers' own `WeightRenaming` entries off the load's record, but `merge_ep_shards.py`, which streams key by key with no model behind it, reads only these pairs, as does the EP layers' own gather. Undeclared, a merged save keeps the module spelling, which serving engines skip (Laguna's `shared_expert` ↔ `shared_experts`).
- `balancing_slot` — the checkpoint-exported routing-bias slot, every spelling (live first, then the hub respelling) plus the config flag that gates it (LFM-2 `use_expert_bias`). Strict `bias_update` refuses a family without one.
- `engine_loaded_buffers` — hub tensors the trainer holds as buffers and an engine loads beside the parameters of one module (Ling's `gate.expert_bias`), plus the config field that creates them on the engine. vLLM's reload rewrites a module that received part of its tensors from uninitialized storage ([Partly loaded layers](../infrastructure/rollout-servers.md#partly-loaded-layers)). `tests/cpu/grpo/test_weight_sync_engine_loaded_buffers.py` fails on a syncable family's checkpoint tensor the push leaves out until its table says why the engine keeps it, and `tests/gpu/trainers/grpo/test_weight_push_exact.py` owes the family a row per engine.
- `hub_conversion_keys` — transformers' conversion-mapping keys for a vendor-namespace hub checkpoint. Each must resolve to entries in the pinned transformers, or the lazy load refuses; a text-only class transformers maps no conversions for plans with them.
- `exports_source_config_schema` — exports carry the source repo's `config.json` schema and `auto_map` modules, for a family a pinned rollout server cannot parse through transformers' serialization ([Checkpoints](../reference/checkpoints.md#exported-configjson)). A source spelling a field through a legacy name, or a float JSON cannot spell, needs nothing more; one naming no modules carries only as a class in transformers' causal mapping, and `test_the_no_auto_map_rule_reaches_the_text_class_alone` lists those.
- `text_only_layout` — a composite transformers' CausalLM mapping does not name ([Text-only load of a composite](#text-only-load-of-a-composite)).
- `refuses_trl_image_rows` — why TRL's own vision collator and loss (DPO, KTO) do not train the family on images, refused before the load.
- `remote_composite_refusal` — why the composite of the release's own modeling file does not train, for a family transformers also ships natively, refused at the load unless a text-only route narrows it away.
- `ep_layer_needs_remote_code` — why the family's EP layer needs the release's own modeling file, for a family whose native class builds a block no EP layer claims (Ling 3.0). `reject_ep_without_family` refuses an EP group on the native class before the process groups.
- `supports_lazy_loading` / `supports_gradient_checkpointing` / `supports_routing_replay` / `supports_fp32_non_ep_params` — all default `True`. Set `supports_lazy_loading` to `False` and the EP loader falls back to `from_pretrained`, logging `lazy_family_unverified`; set any of the others to `False` and the owning gate rejects loudly instead of corrupting silently. `weight_sync_refusal` — `None`, or the reason no end-to-end weight sync into the family's served model is validated. Which family switches off which is published, pinned against the specs, in [Per-family EP restrictions](../parallelism/expert-parallelism.md#per-family-ep-restrictions).
- `tool_call_terminator` — the special token that ends a rendered tool call, for a family where it is no eos of its own `generation_config` (gpt-oss's `<|call|>`). The trainer refuses a tool environment that does not stop on it; left unset the gate passes the family unread, and a served turn runs past its call.
- `eager_only_attention` / `dense_packing_leaks` — the attention facts the resolver and the collator gate read: the reason the family runs eager only (DeepSeek-V4), and a forward that passes no `position_ids` to its mask, so packing is refused on a dense backend (GPT-OSS, Bailing/Ling, GLM-5 Next). FA4's head_dim-256 kernel is read off the head widths (`uses_fa4_dedicated_head_dim`), and the multi-head latent attention it does not train off `kv_lora_rank` (`has_latent_attention`), not declared.
- `attention_reads_row_positions` — the family's attention reads a token's row index, not only relative distance, so offline GRPO scores each row alone and refuses a pipeline batch above one row ([Row positions](../training-methods/grpo/offline.md#row-positions)).
- `unmarked_sequence_mixer` — a sequence mixer its forward reaches with no segment marker (Inkling's short convolutions), named in the refusal of multi-document rows, which `allow_cross_document_linear_attention` lifts ([Document isolation](../data/collators.md#document-isolation-under-packing)).
- `token_indexed_table` — a buffer its routers index by token id (DeepSeek-V4's `gate.tid2eid`), which gives a new token no row, so a vocabulary grow is refused.
- GatedDeltaNet layers need no entry: `has_gated_delta_net_layers` (`halo/models/attention/layout.py`) reads `linear_attention` in `layer_types` on a config declaring `linear_num_key_heads` / `linear_num_value_heads`, under any `model_type`, and gives packed rows their segment markers, kernel refusal and multi-document FA2 default. These sites stay keyed on `model_type`, since dense or spec-less families share their facts; a new hybrid, Mistral-like or VL family checks each: `_SEQ_IDX_MODEL_TYPE_PREFIXES` in `halo/models/segment_markers.py` (the `seq_idx` of ShortConv and Nemotron-H's Mamba), the `mistral4`, `zaya`, `nemotron_h` and `qwen4_exp` branches of `apply_family_patches` in `halo/model_loading/model_preparation.py` (Mistral 4's flash `position_ids` plumbing, Zaya's load patches, Nemotron-H's Mamba scan, Qwen3.8-Flash-Next's host n-gram table and document forwards), and `_VISION_REUSE_MODEL_TYPES` in `halo/trainers/distillation/self_distillation.py` (self-distillation's image-feature replay).

**3. Declare the class contract.** These class attributes are what the layer machinery reads — `EPDeepseekV4MoELayer` exercises most of them.

- `HF_MODULE_NAMES` — the HF MoE block class name(s) this EP layer replaces. The sole registration point.
- The expert count is not declared: the base `detect_num_experts()` reads it off the expert container, a per-expert `nn.ModuleList`'s length or else its `num_experts`, so a container carrying neither raises at the first build.

    The config-level spellings are separate and central: `resolve_expert_count` / `resolve_router_topk` / `resolve_expert_ffn_width` / `resolve_dispatch_width` (the hidden state, or the latent of a block projecting into one around its experts) in `halo/models/moe/config_fields.py`. A family whose config invents another name extends the spellings there rather than teaching one consumer about it.

- `MoEFamilySpec.expert_layout` — the family's one `ExpertLayout` (`halo/models/moe/expert_layout.py`), which the EP gather, the RL weight sync and `merge_ep_shards.py` all apply through `to_hub_expert_tensors`:

    | Kind | Hub layout | Families |
    |---|---|---|
    | `FUSED` (default) | fused `experts.gate_up_proj` / `down_proj`, `F.linear` convention; `key_prefix=""` drops `experts.` where the EP layer replaces the container itself | Cohere2 MoE, DeepSeek-V4, GLM-5 Next, Qwen3.5/3.6, Qwen3.8-Flash-Next, Mistral4, Inkling, Step-3.7, Zaya; Gemma4 bare |
    | `FUSED_SPLIT_PER_EXPERT` | the fused bank split into `experts.{i}.<gate\|up\|down>.weight` | GLM-4, Laguna, Qwen3-Next, LFM-2 (`w1`/`w3`/`w2`) |
    | `SEPARATE_GLU` | gate/up/down stored apart, written per expert | Qwen3, Bailing |
    | `SEPARATE_MLP` | up and down of a gateless expert, written per expert under those names | Nemotron-H |
    | `INTERLEAVED_GLU` | GPT-OSS's interleaved `[g0, u0, ...]` pair with biases, matmul convention | GPT-OSS |

    `per_expert_keys` names the `(gate, up, down)` per-expert spelling. A `FUSED` family declares it where transformers also loads that spelling — its hub's for DeepSeek-V4, Cohere2 MoE and GLM-5 Next, the one `unfuse_moe_experts.py` writes for Qwen3.5/3.6 and Qwen3.8-Flash-Next; a family without one is what that tool refuses, `SEPARATE_MLP` included, whose saves are per expert already. A runtime tensor a layout has no branch for raises rather than dropping out of the export.

- **Rollout-engine support is not a family declaration.** Both pinned engines read the layout the family's `ExpertLayout` writes, so nothing here is per engine.

    Which families an engine can take an *online* update for lives on its weight-sync client (`UNSERVABLE_MODEL_TYPES` in `halo/weight_sync/clients/`), one loader fact per `model_type`. Add an entry there when the engine's loader cannot take the chunked update; `reject_unsyncable_model` then refuses the pair at trainer construction quoting it ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)). An engine that cannot enforce a thinking budget for the family takes an entry in the same client's `UNENFORCEABLE_THINKING_BUDGET_MODEL_TYPES` (vLLM's Mistral 4), which the budget refusal quotes.

- `SUPPORTS_BIAS_BALANCING` — `True` when routing *selection* happens in-layer, which every `EPSharedExpertsMoELayerBase` family inherits. Select through the shared helpers, not a hand-added `self._transient_balancing_bias(...)`, which is `None` when balancing is off and in native-adoption mode (the adopted slot already sits in the selection): score-routed layers take top-k over `self._selection_scores(scores)` (the group-limited base, DeepSeek-V4, Inkling, LFM-2); logit-routed ones use `self._biased_topk(logits)` or `_biased_route`, which also returns the family's unbiased `_gate_weights_at` (GPT-OSS, Qwen3, Qwen3.5, Cohere2 MoE). Gate weights come from the **unbiased** scores, and `_dispatch_compute_combine` records the load, so a route never does.

    A layer can refuse per-instance by overriding `enable_bias_balancing` (DeepSeek-V4 hash layers). Leave `False` when the router sits outside the EP layer (Gemma4) or the family's own gate owns a native balancing buffer (Zaya). An explicit `bias_update` on a model where no layer accepts the bias raises.

    Declaring support is also one of the three signals `moe_balancing: auto` reads — [`auto` resolution per family](../training-methods/callbacks.md#auto-resolution-per-family).

- `MoEFamilySpec.balancing_slot` — the family's exported balancing slot: its first spelling is the live-tree path the layer adopts, so the sign-updates land in exported state (`bias_update_transient` is the user's explicit opt-in where there is none), and its `config_flag` the field flipped for a slot the config gates. A config-gated slot also implements the `_materialize_native_balancing_slot` hook (LFM-2).

- `SEVERS_AUX_LOSS` — `True` when the EP layer bypasses the HF router *module* entirely, so `outputs.router_logits` never populates and the HF aux-loss path is dead under EP. `moe_balancing: auto` then resolves the family to `bias_update` (`resolve_balancing_mode` in `halo/models/moe/balancing.py`, applied by `halo/callbacks/balancing.py`).
- `_SHARED_EXPERT_REQUIRED` — `True` by default: a declared shared expert the block does not carry raises as a rename instead of dropping the FFN from every output. `False` only where the block may lack one: a config flag decides (Bailing, Cohere2 MoE) or upstream ships none (LFM-2).

    `replicated_named_params()` is derived from the adopted name, so a family with the ordinary single shared expert writes no override. Replicated submodules are FSDP-ignored and DP-averaged by the router hook, and an undeclared one trips the same construction-time scan. Override `replicated_named_params()` only when the replicated set is a different *shape* — Qwen3.5/3.6, whose gate is a second module, and Nemotron-H, with its latent projections.

- `_TOPK_WEIGHT_NORM_EPS` — the floor the family's router adds to a top-k weight sum before dividing by it (default DeepSeek-V3's `1e-20`). Read it off the router: Qwen3, Qwen3.5/3.6, Cohere2 MoE, Laguna and Step-3.7 say `0.0`, LFM-2 `1e-6`.
- `_ROUTER_ATTR` — the attribute the router hangs off on both the block and the EP layer (default `gate`; GptOss says `router`).

    The expert activation has no such declaration: `_resolve_activation` reads `act_fn` off the expert container, then `hidden_act` off the block, then off its config, and raises when none names one.

    `_ROUTER_ATTR = None` declares the router *external* to the EP layer — Gemma4's sibling module. Nothing is adopted, no routing knob is read, and no grad-sync hook registers: FSDP-ignored experts sync in the trainer's post-backward sweep.

- `_fused_glu_mul` — the single seam for the GLU combine. `_resolve_activation` latches it from the resolved activation.

    A family with a clamped SwiGLU (DeepSeek-V4 and GLM-5 Next clamp before the activation, Step-3.7 after it) rebinds it in `_adopt_expert_compute` to `resolve_clamped_glu_mul` (`halo/kernels/fused_glu.py`): the fused kernel for a SiLU gate, the same clamps eagerly for any other.

    Latch it rather than overriding `_glu_combine`. The construction summary names the latched callable, so an override makes the reported combine and the running one two declarations that can disagree, and the fused `[gate | up]` path runs a latch's packed form without calling `_glu_combine` at all.

- `_OPTIONAL_ROUTING_KNOBS` — for a `EPGroupLimitedMoELayerBase` family only: the routing knobs (any spelling) its block, router and config genuinely do not declare, so the shared `_adopt_routing` may substitute a neutral default.

    Everything else is required — an upstream rename raises instead of silently under-scaling every routed weight. Laguna and Step-3.7 carry no group limiting and no `norm_topk_prob`; no family opts out of `routed_scaling_factor`.

**4. Expert LoRA.** A new stored weight root needs a `LORA_PROJECTION_COVERAGE` entry (`halo/models/moe/expert_lora_spec.py`) mapping it to the logical `gate`/`up`/`down` projection — without it `expert_lora` silently never adapts that weight (`tests/cpu/peft/test_expert_lora_projection_coverage.py` fails on any registered root missing an entry).

**5. Sharded-save merge.** `merge_ep_shards.py` resolves a checkpoint to its spec through `find_moe_family_spec(model_type)` and writes the spec's `ExpertLayout`, the same transform the gathered save applies, so a family needs nothing more; a `model_type` no spec claims **raises** at trainer construction under `save_sharded_ep=True` (`reject_unmergeable_ep_sharded_save`). So does a load whose conversions the merge's key-by-key stream cannot revert, a vendor namespace or SigLIP's `vision_model` level (`_reject_unmergeable_hub_keys`, `halo/checkpoint/parallel/ep.py`); those families use the gathered save. Add the family to `test_full_merge_matches_gathered_save` in `tests/cpu/checkpoint/test_merge_ep_shards.py`.

**6. Buffers.** `finalize_loaded_model()` (`halo/model_loading/finalization.py`) is the post-load repair every load path runs: it walks the two fixer chains of `halo/model_loading/buffer_repair.py` — `_ROTARY_FIXERS` for `inv_freq`, `_NON_PERSISTENT_FIXERS` for the rest — re-ties shared weights, sets the FLA launch options loaded modules need before their first launch (`apply_fla_launch_options`: FLA's delta-rule state kernel on Blackwell; [Qwen3.5 → Linear attention](qwen3_5.md#linear-attention-qwen3_5moegateddeltanet)), and attaches the external tables a family holds outside its module state ([Checkpoints](../reference/checkpoints.md#external-tables)). transformers 5 re-materializes every non-persistent buffer as `torch.empty_like`, so add a fixer to the matching chain for any the family carries; an uncovered one reaches device placement still on meta, where the trainer raises rather than training on uninitialized memory.

**7. Tests.** The DeepSeek-V4 trio is the template: an EP=2-vs-stock row (`tests/gpu/parallelism/ep/test_ep_family_vs_reference.py --family deepseek_v4`, generated from `TINY_MOE_FAMILIES`; family-specific checks go in its `FAMILY_CHECKS`), a trainer test that saves via the gathered EP path and reloads the checkpoint as a plain HF model (`tests/gpu/trainers/sft/test_sft_ep_roundtrip.py --family deepseek_v4`, a thin subclass of `EPSftRoundTrip` in `tests/common/ep_sft_roundtrip.py` — add one beside it and override only the family hooks), both registered in `tests/gpu/manifest.py`, plus a CPU test covering registration, gather/merge layout, and balancing resolution (`tests/cpu/models/test_deepseek_v4_support.py`). The family's tiny model also goes into `TINY_MOE_FAMILIES` (`tests/common/tiny_models.py`), keyed by its `model_type`, and its name into `_TINY_MOE_FAMILY_MARKERS` in the manifest: the merge-on-save and precompute-resume sweeps run every entry, and `tests/cpu/conventions/test_tiny_family_roster.py` fails on a registered EP family missing from either. `tests/cpu/parallelism/test_ep_family_instantiation.py` builds every registered family's EP layer on its tiny model, and `tests/cpu/parallelism/test_ep_gc_replay.py` replays its dispatch under gradient checkpointing for every family whose layer supports it.

    Past those, a family joins a roster per axis it runs: `FAMILIES` in `tests/common/equivalence/cp_equivalence.py` with `ulysses_cp=True` on its `TinyFamily` for a CP wrapper, `TP_MOE_FAMILIES` in `tests/gpu/parallelism/tp/test_tp_family_vs_reference.py` for a TP-planned attention, `_PP_MOE_FAMILIES` in the manifest for a PP backbone, and `TINY_MOE_VLM_FAMILIES` for each multimodal wrapper it ships under. A VL family's rows read its grid columns from `VLM_GRID_COLUMNS` and its masked placeholders from `resolve_image_token_ids` (`halo/data/vlm_render.py`); a family TRL's vision path does not carry declares `refuses_trl_image_rows`, which `trl_renders_rows` (`halo/entrypoints/data_setup.py`) reads for DPO and KTO.

    A tiny row has to reach the mechanism it names, and assert that it did. Green rows that never ran it are the usual trap: a document within a sparse attention's top-k span attends densely, so selection needs a row longer than the blocks it can pick; a head dim no flash kernel accepts never reaches flash; a bias near zero survives a bf16 cast, so a lost fp32 pin passes (`fill_release_scale_bias`, `tests/common/tiny_models.py`, refuses a scale where the cast changes nothing); a row with no zeroed-pixels control can pass with no image. A composite's image row builds its batch with `tiny_processor` and `tiny_image_batch` (`tests/common/tiny_processors.py`), so it needs no released processor in the cache. Before a family has a spec, its tiny model sits in `TINY_UNWRAPPED_COMPOSITE_FAMILIES`.

    A family at 16 query heads per KV head and `head_dim` 128 takes a row in `tests/gpu/kernels/test_fa4_gqa_ratio.py` at its head counts, which holds FA4 to an fp64 reference for the causal, varlen, non-causal and rectangular log-sum-exp calls.

**8. Wire the docs.** A supported family gets `agent-docs/models/<name>.md`, an `examples/sft/<family>/` config, and a row in each family roster. Three are pinned to the registries and fail `tests/cpu/parallelism/test_docs_limitation_tables.py` until the row lands: the [compatibility matrix](README.md#compatibility-matrix), the [EP supported-models table](../parallelism/expert-parallelism.md#supported-models) and the human [Supported Matrix](../../human-docs/supported-matrix.md#model-families). The rest are kept by hand: the per-family list in [Supported Models](README.md#per-family-pages) and the human guide's [Supported Models](../../human-docs/models.md) table.

## Add CP support

Full procedure in [Context Parallelism — Adding a new model](../parallelism/context-parallelism.md#adding-a-new-model). The machinery lives in `halo/distributed/context_parallel/`.

1. **Write a wrapper** in `families.py` and declare `HF_MODULE_NAMES`. Two templates: subclass `UlyssesAttention` for the optimized path (RoPE before all-to-all, native GQA — copy `Qwen3MoeUlyssesAttention`), or `MLAUlyssesAttention` when Q/K and V head dims differ (copy `Glm4MoeLiteUlyssesAttention`).

    There is no accept list to edit. `ULYSSES_ATTENTION_MAP` is read off the `UlyssesAttention` subclass tree, so the wrapper registers by existing and a duplicate HF name raises.

2. **Runtime requirements**: a real Flash Attention impl (`SUPPORTED_ATTN_IMPLEMENTATIONS` in `model_rules.py`) unless the wrapper declares `REQUIRES_FLASH_ATTN_LABEL = False` (Bailing, whose wrapper runs on any label), and both `num_attention_heads` and `num_key_value_heads` divisible by `cp_size`.

    Override `_resolve_scaling` / `OUTPUT_PROJECTION` when the module computes its scale inline or names its output projection something other than `o_proj`.

3. **Test** with `tests/gpu/parallelism/cp/test_cp_correctness.py`, and add the family to the CP roster: `FAMILIES` in `tests/common/equivalence/cp_equivalence.py`, `ulysses_cp=True` on its `TinyFamily`. `tests/cpu/conventions/test_tiny_family_roster.py` fails on a wrapper no roster family builds.

The wrapper splits Q/K/V into the local-rank slice before the all-to-all, applies RoPE on the correct side of it for the layout, runs the output projection locally and passes position IDs through the chunk boundary; the final all-to-all is the base's. Live per-head buffers are refused rather than sliced: GPT-OSS's wrapper raises on sinks that are not reset. CP runs only on full-attention blocks — linear-attention and gated-delta layers are rejected by `_UNSUPPORTED_SEQUENCE_AXIS_LAYERS`.

## Add TP support

- **Model already has `base_model_tp_plan`** — `tp_plan="auto"` does everything (Llama, Mistral, most Qwen variants). Nothing to add.
- **Selective TP for EP+TP** — add the attention class to `TP_SHARDABLE_ATTENTION_CLASSES` in `halo/distributed/tensor_parallel/parallelize_attention.py`, and the family to `TP_MOE_FAMILIES` in `tests/gpu/parallelism/tp/test_tp_family_vs_reference.py`. The selective path only visits attention submodules, so MoE blocks need no opt-out; `embed_tokens` and `lm_head` stay replicated.

    If the attention has a non-standard shape (GLM-4's LoRA-style compression, GPT-OSS's sinks), add a case in `parallelize_attention.py`. A non-shardable attention (DeepSeek-V4's shared-KV MQA) stays off the list, and `apply_tp_to_attention_only` raises when it ends up patching zero modules.

- MLA attention (a `q_b_proj` / `kv_b_proj` pair) is handled by that plan already: only the expansions shard, the KV-head divisibility check is skipped, and `register_mla_rope_grad_reduction` sums the per-rank-partial rope rows of `kv_a_proj_with_mqa` in backward.

    A hybrid stack shards its full-attention layers only, so its per-rank footprint falls by far less than `1/tp_size`.

## Vendoring a model

If a model isn't in transformers yet, or its `trust_remote_code` conflicts with the v5 pin, vendor it — and remove the vendoring at step 5 once upstream ships the family.

1. **Copy `configuration_*.py` and `modeling_*.py`** into `halo/models/<name>/`. Seed
   `HARDCODED_CONFIG_FOR_MODELS["<name>"]` (`transformers.utils.auto_docstring`) before importing the
   modeling module, or `@auto_docstring` raises "Config not found". A hub checkpoint that stores experts
   per expert while the module fuses them gets no entry in transformers' conversion mapping, so declare
   the layout on the spec (`expert_layout` with its `per_expert_keys`), which the gather and the lazy
   loader read.
2. **Register in a module**, not in the package `__init__.py` — every package init under `halo/` but the `halo/__init__.py` bootstrap carries a docstring only (`tests/cpu/conventions/test_package_inits.py`). Put the calls in `halo/models/<name>/registration.py`:

    ```python
    AutoConfig.register("<name>", YourConfig)
    AutoModelForCausalLM.register(YourConfig, YourForCausalLM)
    ```

3. **Trigger registration** by adding a module-level side-effect import of that module to `halo/model_loading/model_preparation.py`, which every loader imports before it resolves an `Auto*` class (that is how the toolkit's seq-cls heads in `halo/models/seq_cls_heads.py` register).

    `apply_remote_code_compat_shims()` restores removed transformers APIs (`is_torch_fx_available`, `ROPE_INIT_FUNCTIONS['default']`), binds names remote files call without importing (`DynamicLayer`), and patches every loaded remote module for dispatchless eager attention and legacy list-form tied-weight keys, which convert to the v5 dict form; `reject_untieable_list_ties` refuses a tied config that carried one, since `{}` ties nothing.

4. **Add a CPU test** — `tests/cpu/models/test_<name>_registration.py` verifying `AutoModelForCausalLM.from_pretrained` resolves to your class.
5. **Remove the vendoring** once upstream lands — the directory, its registration import, and the registration test all go, and the EP layer, Liger spec and any patches re-key to the upstream module. Name the removal trigger (the upstream PR) in the docstring of `halo/models/<name>/__init__.py`.

HF models that only need `trust_remote_code=True` (Bailing/Ling, Laguna) take the shims path with no vendoring.

A node's ranks share the module cache transformers copies remote code into (`HF_MODULES_CACHE`), and transformers rewrites a module there in place. `halo/model_loading/remote_code.py`, which every loader, training entry point and `from_pretrained_verified` caller imports, makes each copy land whole through the checkpoint staging writes (the remote code a save writes beside a checkpoint too), so a rank never imports a module a peer is still writing.

## Declare Liger coverage

Upstream Liger's `MODEL_TYPE_TO_APPLY_LIGER_FN` doesn't cover every supported model, and covers some only in part. A family that neither registry covers warns at model load that the run trains unfused, and refuses an explicit `liger_kernel_config` key. Add one `LigerFamilySpec` to `halo/model_loading/liger/family_specs.py` — there is no per-family applier module:

1. **Name the classes that fill each role** — `rms_norm`, `gated_rms_norm` (a linear-attention block's `norm(x) * w * act(gate)`, served by `fla`), `glu_mlp`, `causal_lm`.

    Then the variant parameters: `rms_norm_offset` / `rms_norm_casting_mode` for a Gemma-style `(1 + w)` norm or an fp32 weight multiply, `rms_norm_kernel="native"` to serve an offset-free norm with torch's fused `F.rms_norm` instead of Liger's, `router_aux_loss_in_head` for a head that adds the router aux loss after the projection, `rope=True` only for a full-width `rotate_half` rotary.

    A `trust_remote_code` family sets `remote_classes` (every class a role names) instead of `modeling_module`; its patch fires when transformers loads the modeling file. A native family whose hub repos also ship their own modeling file through `auto_map` (Laguna) sets both.

    List a multimodal wrapper's `model_type` only when its text tower can be nothing but this family. Otherwise the dispatch resolves it through `text_config`, and the fused loss is forced off there because the wrapper's own head runs.

2. **Declare a role only where the kernel is EXACT.** A clamped GLU, a trained output scale, a weightless or gated norm, a partial or interleaved rotary, a head that truncates the vocab — each would compute a different function silently.

    Leave it off and say why in the spec comment.

3. **Set `flce_default=True`** only when the `[seq, vocab]` logits plane is the family's binding memory limit; the generic default keeps logits for metrics. **Set `fp32_logits=True`** where the head casts its logits to fp32, which `tests/cpu/model_loading/test_liger_family_coverage.py` reads off its forward.
4. **Set `delegates_to_upstream=True`** when upstream already covers the family and the spec only ADDS to it (dense Qwen3.5/3.6), or takes over a role upstream gets wrong (GptOss's norm casting, Gemma 4's EP-surviving dense MLP, the Qwen MoE `swiglu` that would also swap the routed experts — the withheld flag goes in `upstream_off` and the spec must fill that role).

    A MoE family's routed experts never run Liger's fused MoE kernel (`LigerExperts`): where upstream's `swiglu` could swap them and the spec does not withhold it, the dispatch forces the flag off whenever Halo does not wrap the experts ([Routed experts](../optimization/liger-kernels.md#routed-experts)).

    Upstream's applier runs first with every other flag it declares; the spec names only the roles it leaves eager or takes over.

    Three derived guards hold it together. The upstream applier is resolved from liger-kernel's own registry by `model_types`, so a family it stops covering, aliases that stop sharing one applier, or a flag the added role rides that left its signature all raise at import.

    The spec must add a role of its own to a native modeling module. And a named class upstream has already replaced raises at patch time, read off the class's own `__module__`, which is what keeps the two swaps apart.

    `cross_entropy` is never handed to upstream (its branch rebinds `F.cross_entropy` process-wide): the toolkit's scoped patch serves it for every family.

5. **The CPU suite gates it.** `tests/cpu/model_loading/test_liger_family_coverage.py` checks every roster family resolves an applier, that each declared class exists, that each declared norm is what the chosen Liger variant computes, and that each declared MLP and head is the canonical body.

    `test_liger_upstream_delegation.py` pins the delegation seam. Add the GPU numerics case to `tests/gpu/kernels/test_liger_family_kernels.py`.
