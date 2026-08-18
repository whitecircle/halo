# Checkpoints and Training Resume

Saving and loading live in `src/distributed/checkpoint/`. The trainer's `save_model()` hands a
`CheckpointContext` to `save_checkpoint()`, whose `select_checkpoint_saver()` ladder returns the
per-mode saver that gathers or reconstructs the full model before writing; `CheckpointLoader` restores weights and `OptimizerShardStore` the per-rank
optimizer shards + LR scheduler. A `PeftModel` is dispatched to `PeftAdapterSaver` ahead of the ladder.

The **save rank** follows the **output** side of the filesystem declaration
(`DIST_OUTPUT_SHARED_FILESYSTEM`, falling back to the `DIST_SHARED_FILESYSTEM` umbrella, default
shared): on a shared output FS only **global rank 0** writes; on a non-shared one each node's **local
rank 0** writes its own copy.

| Mode | Saver | Which rank saves | Collective |
|------|-------|------------------|------------|
| **PP / PP+EP** ([not yet available](../parallelism/pipeline-parallelism.md)) | `save_pp_checkpoint` — each stage writes its own complete tensors | one writer per stage (`stage_local_rank == 0`; per node on a non-shared FS) | world-wide index exchange |
| **EP** (incl. ETP, EP+CP, EP+TP) | `save_ep_checkpoint` → `save_ep_model()` | Save rank | EP all-gather (+ `full_tensor()` under TP) |
| **EP sharded** | each rank writes its expert shard | All ranks | `exchange_shard_index` (gather to rank 0, broadcast back) |
| **TP** (HF-native or selective) | `save_tp_checkpoint` — reconstruct DTensors, then all-gather the hand-sliced GptOss `sinks` (which the parameter walk skips: it would emit this rank's slice under the same key) | Save rank | `DTensor.full_tensor()` |
| **CP** | `save_cp_checkpoint` — remap Ulysses wrapper keys to HF paths | Save rank | `full_tensor()` |
| **FSDP2** (plain torchrun DP) | `save_fsdp2_checkpoint` — stream the gathered chunks | Save rank | `full_tensor()` |
| **Single-GPU / accelerate** | no saver (the ladder returns `None`) → HF Trainer default | Main process | None |

Selection reads what the model **is**, not the configured sizes: a model carrying EP layers takes
`save_ep_checkpoint` even at `ep_size == 1` (experts replicated, FSDP2-sharded), so an ep1 MoE run
writes through the expert gather rather than a plain gathered write — the same verdict for every
trainer. `EmbeddingTrainer` runs the same ladder with the checkpoint context re-pointed from the
`SentenceTransformer` `nn.Sequential` at the `auto_model` backbone, then writes the ST pipeline config
beside it; its in-place-injected LoRA (not a `PeftModel`, so `PeftAdapterSaver` never sees it) is
folded into the gathered state dict and handed to the same shared writer.

**Gathered saves** (the default everywhere) produce HuggingFace-compatible checkpoints loadable with
`from_pretrained()`. The optional **per-rank sharded EP save** (`save_sharded_ep`) is a
write-bandwidth optimization and must be merged before loading.

Every gather is a collective, so **all** ranks run it, but only the save rank copies the result to
host memory (`materialize_dtensor` on the others, not `resolve_param_tensor`) — a node pays one
writer's copy instead of `gpus_per_node` × model. Under EP the same `retain` flag is threaded into
each family's `gather_expert_state_dict`: the expert all-gathers stay on every rank, while the layout
assembly after them runs only where the result is kept (a family that returns tensors anyway is
rejected, not obeyed). Every gathered save then **streams** through `StageShardWriter`: FSDP2, CP
and TP via `stream_gathered_checkpoint`, one conversion-closed chunk (a decoder layer) at a time; EP
and PP by writing each MoE layer out as it is gathered. The writer never holds more than one chunk
plus one pending shard, so host memory stays well under 1× model
([measured peaks](../parallelism/expert-parallelism.md#checkpointing)) — which is what makes the
layout usable at fine-grained-MoE scale.

The streamed parts are renamed to HF's own `model-{i}-of-{n}` once the count is known, so
`from_pretrained` reads the result exactly as it reads a `save_pretrained` checkpoint. Part
*boundaries* can differ from `save_sharded_state_dict`'s — the streaming writer flushes a block before
an oversized tensor — so the file count and key→file mapping can differ while both load the same.

## What lands on disk — four layouts

Tell them apart by `model.safetensors.index.json`, not the filename: the per-rank layout carries a
`metadata.format` marker (`is_sharded_checkpoint` reads it). The `format: "pt"` inside each
safetensors *file* header is the container format, not this marker.

| On-disk layout | Files | Produced by | Index marker | Loadable? |
|---|---|---|---|---|
| **Single file** | `model.safetensors` | any gathered save under the shard threshold | *(no index)* | Yes |
| **Gathered HF-sharded** | `model-00001-of-000NN.safetensors` + index | any gathered save over the threshold — the default for production-size models | none | Yes |
| **PP per-stage** | `model-pp00000-of-00004-00001.safetensors` — one or more parts per stage + merged index | `pp_size > 1` — [not yet available in this release](../parallelism/pipeline-parallelism.md), so no run produces it today | none | Yes — **no merge step** |
| **EP per-rank sharded** | `model-00000-of-000NN.safetensors` (`.shard_{rank}` keys) + index | `save_sharded_ep: true` | `format: "ep_sharded"` | No — `merge_ep_shards.py` first |

Shard size defaults to 5 GB (`save_max_shard_size` overrides it). It bounds the **gathered** writers
(EP/TP/FSDP2/CP/PP) and the merge tools' output; a per-rank EP shard is one file per rank by design,
so the cap does not apply there — the save logs that rather than appearing to honor it. A directory of
per-rank shards whose index never landed (a run killed between the shard writes and the index write)
is refused by every after-training tool: the shards carry `.shard_N` keys, which a header-only peek
detects without an index.

The **PP layout is not a per-rank format**: each stage writes *complete* tensors under global
parameter names (`stage.global_parameter_name`, with EP experts re-exported through their family
gather) and rank 0 merges the index into HF's native multi-shard layout, which is why it deliberately
carries no marker. Each stage owns a filename prefix derived from its own `pp_rank` and never
coordinates with the other writers, so they stream concurrently into one directory; merging the index
refuses a key claimed by two stages, since a global parameter lives on exactly one.

The exchange gathers the per-writer weight-map fragments to rank 0 and broadcasts back the merged
map, so no other rank ever holds one fragment per rank of it. The collision verdict rides the same
broadcast: a duplicated key raises on every rank, rather than on rank 0 alone while its peers block
in the barrier that follows.

PP does not compose with the per-rank format: `save_sharded_ep` is validated unconditionally at
construction and **raises** under PP — its shards key tensors by unsplit-model names with no stage
layer offset, and a sharded EP save needs one EP group spanning the whole world, which a stage's
group never is.

The per-rank EP layout holds **partial** tensors; the loaders detect the marker and refuse it with an
error naming the merge script, so it is never silently loaded as whole. How weights return on resume
depends on whether the mode transforms the model at construction — see
[Resume by parallelism mode](#resume-by-parallelism-mode).

## What gets saved

| File | `save_only_model=false` | `save_only_model=true` | Purpose |
|------|:-:|:-:|---------|
| `model.safetensors` (or sharded + index) | Yes | Yes | Weights |
| `config.json`, `tokenizer.*` (full processor for VLMs) | Yes | Yes | Architecture config + `processing_class` |
| `generation_config.json` | Yes | Yes | Sampling defaults, stop strings and the custom `eos_token_id` set — written for a generative model only, and paired with `config.json` by `save_model_config`. `PretrainedConfig` does not carry it, so a save that writes only `config.json` serves with defaults. Under PP the stage module mirrors the unsplit model's `generation_config`/`can_generate` so the pair holds there too. |
| `trainer_state.json` | Yes | Yes | Step count, epoch, loss history |
| `optimizer_shard_XXXXX.pt` | Yes | No | Per-rank optimizer shard — every torchrun sharded mode. Under multi-group EP only the replica group's lowest rank carries the replicated (FSDP-ignored) expert state; its peers read it from that shard |
| `optimizer_meta.pt` | Yes | No | Topology fingerprint gating exact resume (+ `pp_stage_partition` under PP) |
| `scheduler.pt` | Yes | Yes | LR scheduler state — re-persisted on every mode so resume continues the schedule |
| `router_balancing_biases.pt` | Yes | Yes | DeepSeek-V3 router balancing biases, restored on resume |
| `rng_state_<rank>.pth` | Yes | No | Per-rank RNG state (`rng_state.pth` single-process) |

`optimizer.pt` is dropped under every
[torchrun sharded mode](#warm-restart-vs-exact-resume-torchrun), which hold per-rank
`optimizer_shard_XXXXX.pt` instead. Plain `accelerate launch` DDP is not intercepted by the mixin, so
the base HF Trainer writes and keeps `optimizer.pt`.

Every gathered save writes parameters **plus persistent buffers** (Gemma4 `layer_scalar`, vision
`std_scale`/`std_bias` — a dropped residual scalar corrupts the layer on reload) and reconstructs the
neutralized GptOss attention `sinks` that FA2 fine-tuning removes from `named_parameters()`.
Non-persistent buffers (rotary caches, attention masks) are omitted and recomputed on load. One
`persistent_buffers()` helper drives the EP, TP, and FSDP2 gathers; CP gets the same set through the
wrapper's own `state_dict()`. Resolving a tensor is a collective (`full_tensor()` on a DTensor), so
the FSDP2, TP and CP gathers share one preamble that runs every leg — parameters *and* buffers — on
every rank and keeps only the writer's host copy: a leg entered by the writer alone hangs the save on
the ranks that never enter it.

The exported `config.json` is serialized with run-scoped router mutations restored
(`config_export_ready`): the balancing strategy's zeroed `router_aux_loss_coef`, forced
`output_router_logits` and toolkit stamp configure the run, not the artifact — exported, they would
destroy the hub's aux coefficient for later stages and make plain-transformers consumers pay the
router-logit plane on every forward.

It is also written in the flat form the pinned rollout server parses. transformers 5.16 serializes a
family's per-layer attention geometry only as `per_layer_config`, which the server's transformers
(the 5.14 line, [Rollout Servers](../infrastructure/rollout-servers.md#config-schema-parity))
refuses at parse; a family declaring `_LEGACY_PER_LAYER_CONFIG_KEYS` on its EP layer class (Gemma 4:
`global_head_dim`, `num_global_key_value_heads` — the full-attention layers' geometry) has every
exported `config.json` rewritten to those keys by `export_legacy_per_layer_config`, on every toolkit
config write (parallel saves, `save_full_checkpoint`, the trainer fallback). A 5.16 reload rebuilds
`per_layer_config` from the flat keys exactly as it does for the hub checkpoint; a config the flat
keys cannot express (a per-layer override outside the declared pairs) is refused rather than
flattened.

For a family whose serving engines have no config class at all, a dialect rewrite is not enough. The
engines read such a family only through its source repo's `config.json` and the remote-code modules
its `auto_map` names, whose vendor spellings transformers absorbs at load (`attribute_map` plus
per-family `__post_init__` kwargs) and never re-emits — so nothing on the save side can reconstruct
them. A family declaring `_EXPORTS_SOURCE_CONFIG_SCHEMA` on its EP layer class (Step-3.7 Flash) has
every exported `config.json` replaced by the **source checkpoint's own**, with this run's changed
values written back under the source's spellings and the `auto_map` modules copied in transitively
(`export_source_config_schema`, same three writers). The source is the checkpoint the model was
loaded from, so a resume or a `patch_vocab.py` output carries the schema on. Correctness is proved
per save, not assumed: the rewritten config is re-parsed and compared against the live one, and a
change the source schema swallows — a field it spells through a legacy key the rewrite cannot update
— raises instead of serving a geometry the trainer never had. A source declaring no `auto_map` has
no schema to hand on; the export warns and keeps transformers' own.

All three rewrites — the `model_type` restore, the legacy per-layer keys, the source schema — are
one step, `finalize_exported_config` (`src/checkpoint/config_export.py`, which owns the whole
exported-config contract), which every writer calls last: the parallel saves through
`save_model_config`, `save_full_checkpoint`, and the single-GPU/DDP/accelerate-FSDP fallback in the
trainer. A path running a subset ships a directory that trains and reloads but that the merge tools
or the pinned server refuse, on a family the run never mentioned.

Gathered saves also reconcile `tie_word_embeddings`: tied embeddings can diverge during distributed
training (independent FSDP2 shards, FLCE training `lm_head` directly), so when the saved `lm_head`
and `embed_tokens` tensors differ, the config is written with `tie_word_embeddings: false` and the
trained `lm_head` is honored on reload. The pair is matched by key suffix, covering wrapped/VLM
layouts. `save_pp_checkpoint` saves the stage config verbatim and skips the reconciliation — PP rejects
tied-embedding models outright when it splits the model.

FSDP2/CP and TP funnel into one writer (`stream_gathered_checkpoint`), so all three get the same
save-dtype and hub-expert-layout normalization and the same auto-sharded safetensors layout. A failing
write raises through `DeferredRankFailure` so every rank sees it at the next collective; the
`.bin`-then-sweep fallback belongs to `write_gathered_checkpoint`, the whole-dict writer the
injected-LoRA embedding merge uses — it needs a dict still whole after the failure, which a streamed
save does not hold. Only the config write is gated on the model carrying one; a model without a
config still gets normalized weights in safetensors.

`save_only_model` is the warm-restart / exact-resume switch — what each keeps, and what exact
resume demands of the topology, is the
[warm vs exact matrix](#warm-restart-vs-exact-resume-torchrun).

## Serving on vLLM / SGLang

A gathered checkpoint is a standard HF checkpoint — stock `from_pretrained` loads it as-is.
(That is serving a **saved artifact**; live RL weight-sync support is narrower —
[Rollout Servers](../infrastructure/rollout-servers.md#the-fused-expert-layout-is-declared-per-family).)

**vLLM** needs two things for some MoE models.

**Un-fuse experts (vLLM only).** transformers keeps MoE experts fused in memory
(`experts.gate_up_proj` / `experts.down_proj`) and reads that layout back on load. Most hub
checkpoints are per-expert (`experts.{i}.{gate,up,down}_proj.weight` /
`experts.{i}.w{1,3,2}.weight`) and `from_pretrained` fuses them on load. The wrapper-less writer
runs the save-side revert (per-expert hub spelling); an EP-gathered save writes whatever the
family's own gather emits — the fused pair for Qwen3.5/3.6, DeepSeek-V4, Cohere2 MoE and GLM-5 Next.
Rewrite such a checkpoint only when the
serving loader is per-expert-only, with `scripts/after_training/unfuse_moe_experts.py`, which emits
the names that family declares and refuses the families whose checkpoints are not per-expert at all
([Scripts](scripts-reference.md#post-training-scripts)). Step-3.7 Flash is one of the refused
(its hub layout is per-layer stacked, not per-expert) and needs no rewrite: its EP-gathered save
already lands in that hub `moe.*` namespace (`_EXPORTS_HUB_NAMESPACE`, [Step-3.7](../models/step3p7.md#checkpoint)),
which its pinned engines read directly.

| Serving loader | Families | Fused `gate_up_proj` / `down_proj` |
|---|---|---|
| vLLM 0.26.0 `RoutedExperts` | Qwen3.5/3.6, Gemma 4 | loaded directly |
| vLLM 0.26.0 `FusedMoE` (`cohere2_moe`, `step3p5`) | Cohere2 MoE, Step-3.7 Flash | loaded directly |
| vLLM 0.26.0 per-expert-only | GLM-4 MoE Lite, Laguna, LFM-2, Bailing/Ling 2.0 | hard-fail or silent drop — un-fuse first |
| vLLM 0.26.0 — no model class | Mistral4, Ling 3.0 (`bailing_hybrid`), Ring (`bailing_moe_linear`) | not servable at all ([Mistral4](../models/mistral4.md#serving), [Bailing](../models/bailing.md)) |
| SGLang 0.5.17 | gpt-oss | loaded directly; every other family needs per-expert |

**MLA backend on Blackwell.** GLM-4 MoE Lite uses MLA; flashinfer's MLA kernel rejects its head
config on SM100+ — serve with vLLM `--attention-backend CUTLASS_MLA` or SGLang
`--attention-backend triton`.

## Saving by parallelism mode

### Expert parallelism (EP, EP+TP, EP+CP) {#expert-parallelism-ep}

`save_ep_model()` (`src/distributed/expert_parallel/saving.py`) owns all three, over three param
classes: EP expert weights, gathered by `gather_ep_layer_weights()` with local → global expert
re-indexing; DTensor params (EP+TP attention, embeddings, `lm_head`, and FSDP2-sharded params such as
the router under ep1 sharding), resolved via `full_tensor()`; and manually TP-sharded plain tensors
(GptOss `sinks`, sliced plain because forward needs local shapes), all-gathered from
`model._tp_sharded_non_dtensor`.

Weights are cast to the save dtype (BF16) except three keep-sets that hold their trained dtype:
normalization params, the live router-balancing tensors, and the family's fp32 pins. Under EP+CP the
iteration runs on the unwrapped model with an explicit CP-prefix strip, so attention weights land
under their hub names; experts are replicated per CP rank (EP ⊥ DP), so any rank holds the complete
set.

Keys are written in the family's **hub** spelling. Where a family's live module names differ —
transformers expresses this as `WeightRenaming` and applies it only inside `from_pretrained` — the EP
layer class declares the pairs in `_EXPORT_KEY_RENAMES` and the gather rewrites them (Laguna is the
one such family). Four families declare transformers' load-side conversion for their hub checkpoints
(`_HUB_CONVERSION_KEYS`), which the lazy loaders replay per key. Three of them are bridged read-side
only, so their gathered exports keep the canonical module spelling: Inkling — which is exactly why
its layer refuses weight sync (`_supports_weight_sync = False`) — DeepSeek-V4, whose sync is refused
because vLLM's V4 loader targets the packed fp8/fp4 release layout, and GLM-5 Next, which no pinned
engine loads. Step-3.7 Flash additionally declares `_EXPORTS_HUB_NAMESPACE`, so its gathered save
runs transformers' own save-side revert per chunk and lands in the hub namespace
([Step-3.7](../models/step3p7.md#checkpoint)). The conversion sources are not all vendor-anchored
(DeepSeek-V4's `\.norm\.` → `.kv_norm.` also matches the canonical final norm), so the lazy loaders
keep a converted key whose targets all miss the model's key space while the key itself resolves —
the same model-key validation transformers' own loader applies — and a canonical checkpoint (an EP
or PP resume of a toolkit save) loads untouched.

vLLM keys on hub names and silently skips unknown ones, so the module spelling would drop those
tensors from serving and from the online-GRPO weight sync. The lazy loader applies the inverse on
read; `merge_ep_shards.py` and the GRPO weight sync apply the same rewrite, keeping
merged-from-sharded and pushed-to-vLLM key-identical to gathered.

**Sharded (`save_sharded_ep: true`)** writes each rank's expert shard keyed by global rank, with
global rank 0 adding non-expert params and the index. Because the shards are global-rank-keyed and
`merge_ep_shards.py` needs them co-located, `validate_ep_sharded_save()` **fails fast at trainer
construction** on any of:

- **Multiple EP groups / DP replicas** (`ep_group_size != world_size`): replicas share an `ep_rank`
  and the same expert slice, so the merge would concatenate duplicated experts.
- **`expert_tp_size > 1`**: TP-sharded expert weights the merge cannot reconstruct.
- **Context parallelism**: the Ulysses key layout is not what the merge expects.
- **Multi-node on a non-shared filesystem**: shards scatter across nodes' local disks with the index
  only on node 0, so no directory holds a complete set.
- **A `model_type` no EP layer class claims.** The merge transform lives on the class
  (`merge_shards_to_hf`, the inverse of its own gather; `__init_subclass__` requires the two be
  overridden together), resolved through the class-declared `HF_MODEL_TYPES`; an unclaimed
  `model_type` has no transform.
- **A family that exports the hub namespace** (`_EXPORTS_HUB_NAMESPACE`, Step-3.7 Flash): its save
  runs transformers' save-side conversion revert per chunk, which the merge's key-by-key stream
  cannot apply.
- **Native expert LoRA**: the grouped `_lora_A`/`_lora_B` adapters would be written as `.shard_N`
  keys the merge's base-root pattern never matches, passing through dead while the merged experts
  stay the frozen base.
- **`merge_expert_lora_on_save: true`**: folding adapters requires the gathered save.
- **A run with no EP layers at all** (dense, or a MoE without EP wrappers): every save would be an
  ordinary gathered checkpoint while the planned merge waits for shards that never appear.

Use gathered for deployment and on a non-shared multi-node filesystem (it writes a complete checkpoint
per node). Use sharded for fast intermediate checkpoints of a large MoE on a shared filesystem or
single node: every rank writes its own shard in parallel, so the pause at each save is bounded by one
rank's shard rather than by the whole artifact. Write bandwidth is the only reason to choose it — the
gathered path streams, so host memory is bounded either way (and a sharded save holds *more* per
node). Merge before loading; an incomplete shard set raises, and `--output_dir` may not be `--input_dir`
— these conversions are not in-place ([input guards](scripts-reference.md#input-guards)):

```bash
python scripts/after_training/merge_ep_shards.py \
    --input_dir /path/to/sharded_checkpoint --output_dir /path/to/merged_checkpoint
```

The merge copies the resume sidecars (`scheduler.pt`, `router_balancing_biases.pt`, `rng_state_*`)
while excluding stale weight artifacts, so the merged directory resumes weights, scheduler and
balancing biases — but **not** the optimizer: the per-rank `optimizer_shard_XXXXX.pt` files are
weight-suffixed and dropped, so a resume from the merged directory warm-restarts. Moot for every
shipped sharded config, which sets `save_only_model: true`. A resume pointed at the unmerged directory
raises the merge-first error directly.

The merge requires a per-rank index carrying `ep_size`, so a gathered, PP or already-merged directory
is declined rather than transformed twice, and an adapter sitting beside the shards is refused up
front (the aux copy carries `adapter_config.json` but no weight file, so the merged directory would
claim an adapter it does not hold). `generation_config.json` rides across verbatim. The tool's flags
are on [Scripts](scripts-reference.md#post-training-scripts).

### Tensor parallelism (TP)

Two TP implementations exist — **HF-native TP** (`tp_plan="auto"`, dense) and **DTensor TP**
(`parallelize_module()`, the MoE attention-only path incl. EP+TP) — and both place their sharded
params as DTensors, so `save_tp_model()` (`src/distributed/tensor_parallel/checkpoint.py`)
reconstructs either with one `full_tensor()` walk. The gather is a collective every TP rank must
enter, and GptOss `sinks` need the separate plain-tensor all-gather. One writer per filesystem scope
(`fs_aware_save_rank`) avoids write races between the multiple TP groups a node holds when
`tp_size < gpus_per_node`.

There is no per-rank TP save: the TP writer always gathers, and streams the result like every other
gathered save.

A tied embedding/head pair is one vocab-sharded DTensor under HF-native TP
([Load paths](../parallelism/tensor-parallelism.md#load-paths)), so the gather walk visits it once
and the checkpoint carries a single `embed_tokens.weight`, no `lm_head.weight`, and the config's
`tie_word_embeddings: true` — the same artifact a single-GPU save produces.

### Context parallelism (CP)

`UlyssesCPModelWrapper.state_dict()` (`src/distributed/context_parallel/wrapper.py`) remaps
Ulysses-internal keys back to HF paths by stripping the `.original_attention.` segment the patcher
introduces. A stale duplicate dense `.mlp.{gate,up,down}_proj` key is dropped **per layer, only on
layers that carry routed experts** — a hybrid model's genuinely-dense layers (GLM-4 MoE Lite,
Mistral4 `first_k_dense_replace`) keep their dense MLP weights. `save_cp_checkpoint` gets the wrapper from
the trainer's `_find_cp_wrapper()`, which finds it in both the plain and PEFT-nested layouts.

## PEFT (LoRA) saving

`PeftAdapterSaver` (`src/distributed/checkpoint/peft.py`) is dispatched before the saver ladder
except in two cases: accelerate-managed FSDP (adapters are flat-param shards there, so the base
Trainer's save owns the layout) and `merge_expert_lora_on_save`, which wants a merged base and so
takes the mode's saver even with adapters present. Only adapter weights are saved.

| Mode | LoRA params | Saving rank | Collective |
|------|-------------|-------------|:----------:|
| **EP** | Attention adapters + native grouped expert adapters | Save rank | Yes (expert gather) |
| **ETP** | Attention adapters only — expert LoRA is rejected at construction under `expert_tp_size > 1` | Save rank | Yes |
| **FSDP2** | DTensors (sharded across DP mesh) | Global rank 0 (shared FS) / local rank 0 | Yes |
| **QLoRA + CP** | Regular tensors (no DTensor adapters) | Save rank | No |

Every save reshards the FSDP2 modules first (`save_model`): mid-training they can still hold the
transient unsharded params a forward registered — objects the optimizer never stepped, so every
writer would read tensors one optimizer step stale. The PEFT path additionally **mis-routes** on
them: its DTensor probe sees plain tensors, takes the rank-0 `save_pretrained` branch, and PEFT's
shared-tensor scan raises on the first parameter that branch cannot address.

LoRA is rejected at construction under TP / EP+TP (adapters are not integrated into the TP DTensor
graph) and under PP. The saver checks for DTensor adapters before the CP branch, so non-quantized
LoRA+CP (whose adapters are FSDP2 DTensors) routes through the FSDP2 path; only QLoRA+CP, with no
DTensor adapters, takes the collective-free CP branch. Under EP the native grouped expert adapters are
gathered across the EP group into the same `adapter_model.safetensors`; `merge_expert_lora_on_save`
routes the save away from this path entirely, to `save_ep_checkpoint`'s merged gathered checkpoint (see
[PEFT](../optimization/peft.md#checkpoint-saving)).

Output: `adapter_config.json`, `adapter_model.safetensors`, `tokenizer.*`. On a safetensors write
failure the saver falls back to `torch.save` (`adapter_model.bin`); resume accepts either — the
candidate present on **every** rank, not each rank's first local hit, since two ranks reading
different files would issue the DTensor loads (sorted by key, a mesh collective per key) in two
different orders. A GptOss
run additionally writes `training_provenance.json` recording whether attention sinks were live or
neutralized — the export tools (`apply_training_sidecars` in `src/checkpoint/tool_io.py`)
re-apply it, because a merge rebuilds the base from the hub, whose sinks are always live, while a
`reset_sinks` run trained its adapter under neutralized ones. It is a separate sidecar, never
`adapter_config.json`, so stock PEFT keeps loading the adapter unchanged.

A directory holding native grouped expert adapters is labeled with a `peft_type` PEFT does not know
— `EXPERT_LORA` for an expert-only save (which carries no PeftModel, so the saver synthesizes the
config), `LORA_WITH_EP_EXPERT_LORA` for a mixed attention+expert one — so an external
`PeftModel.from_pretrained` raises on the label instead of loading the attention half alone. The keys
and the synthesized fields are on [PEFT](../optimization/peft.md#checkpoint-saving).

## Accelerate / FSDP checkpoints

Without EP/CP/TP (standard mode via `accelerate launch`), saving follows HF Trainer defaults:
single-process save without FSDP; with FSDP, `fsdp_state_dict_type: FULL_STATE_DICT` gathers to rank 0
or `SHARDED_STATE_DICT` saves per-rank shards.

> [!WARNING]
> **FSDP v1 corruption bug**
>
> FSDP v1 (both SHARD_GRAD_OP and FULL_SHARD) corrupts internal parameter views after gathering,
> producing silently corrupted checkpoints. Prefer FSDP v2
> (`launcher-configs/accelerate/fsdp2_gradop_config.yaml`) or standard DDP (`launcher-configs/accelerate/multigpu_dp_config.yaml`).
> The `torchrun` path uses FSDP2 exclusively and is unaffected.

Under `torchrun`, HF's built-in FSDP is disabled (`config.fsdp = ""`) and the mixin applies FSDP2
(`fully_shard`) programmatically, excluding EP modules via `ignored_params`.

## Load coverage gate

`from_pretrained` re-initializes any key the checkpoint does not carry and reports it through the
transformers logger only — which `setup_logging` lowers to `error` on every non-logging rank. A
truncated, partially uploaded or wrong-architecture directory would therefore train from random
weights with no exception and no output.

Every toolkit load path — the per-mode loaders, the lazy EP/PP loaders (which bypass `from_pretrained`
entirely), and the reference/teacher-model loads — routes its absent set through
`verify_checkpoint_coverage` (`src/models/loading/checkpoint_coverage.py`) and **raises**. A new
loader gets it by calling `from_pretrained_verified` instead of `from_pretrained`.

Legitimate absences are read off the model class, not a key list: a task head the architecture adds on
top of the backbone (`score`, `classifier`, an untied `lm_head` — anything outside
`base_model_prefix`), which is what makes reward / classification training on a base checkpoint work;
a tensor the checkpoint carried under another name, tested by object identity
(`state_dict(keep_vars=True)`) so it covers any alias, not just a tied `lm_head.weight`; and whatever
the class declares in `_keys_to_ignore_on_load_missing`. A class that *is* the backbone (`AutoModel`)
has no outside, so nothing is excused for it. The gate inspects `missing_keys` only.

The task-head excuse exists for *training* a head. A caller that only consumes one passes
`excuse_task_head=False` — reward-model inference (`scripts/inference/reward_model/`) does, so
pointing it at a plain causal-LM directory raises instead of scoring every row through a random
`score` head. `HALO_ALLOW_MISSING_CHECKPOINT_KEYS=1` downgrades the raise to a warning.

## Resuming training

Set `resume_from_checkpoint: true` in YAML (or on the CLI) and re-run the same command:

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py config.yaml --resume_from_checkpoint=true
```

Resume is resolved once in `init_training_script` (`src/training/script_runner.py`), which finds
the latest checkpoint — broadcasting the path from rank 0 — and asks `resolve_resume_weights_source()`
whether to repoint the weights source. It repoints when `needs_ep_wrappers`, `is_cp_mode` or
`is_tp_mode` holds — with the default `use_grouped_gemm: true` that is **every stock torchrun run**,
dense included — returning the **checkpoint directory** so the trained weights load at model
construction (Path B); only a `use_grouped_gemm: false` run with no CP and no TP keeps
`model_name_or_path` (Path A). The FSDP2 loader detects a model constructed from the checkpoint and
skips the redundant full-state-dict re-read.

`model_config` is not mutated, so a frozen reference/teacher model and the preprocessed-dataset
compatibility check keep using the base path. `trainer.train()` then reads `global_step` / `epoch`
from `trainer_state.json`, computes batches to skip, and restores optimizer and LR scheduler.

WandB does **not** auto-continue the same run — export `WANDB_RUN_ID` (see [Multi-node](#multi-node)).

### Resume by parallelism mode

| Mode | Path | Weight loading | Optimizer | Notes |
|------|------|----------------|-----------|-------|
| **FSDP2** (standard DP) | A at `use_grouped_gemm: false`, else B | Path A: `load_full_state_dict()` → `set_model_state_dict(broadcast_from_rank0)` into DTensor params. Path B: the weights are already in the model from construction and the loader skips the re-read | Per-rank FSDP2 shards | Exact resume, same world size |
| **TP** | B | Weights load at construction and `_load_tp` skips the re-read. Where it does read (a best-model reload, or a model built from elsewhere), each rank streams the checkpoint's full tensors and `distribute_tensor`s them into its own DTensor placements; TP+DP instead raises for a model not constructed from the checkpoint | Per-rank shards | Exact resume, same world size |
| **CP** | B | Skipped — weights via `load_distributed_model()` | Per-rank shards (matching fingerprint) | Checkpoint has HF keys, model has CP wrapper keys |
| **EP / ETP / EP+CP / EP+TP** | B | Skipped in the loader — the trained weights load at construction because the model source points at the checkpoint | Per-rank shards (matching fingerprint) | Checkpoint has unfused HF keys the EP-fused 3D model cannot reload |
| **PP / PP+EP** ([not yet available](../parallelism/pipeline-parallelism.md)) | stage-aware | Each rank reads only its stage's global-named tensors from the merged index and remaps them through `global_parameter_name`; a missing stage-retained tensor raises. On per-node output storage the locally-absent cross-stage tensors are skipped — the stage build drops them anyway | Per-rank shards, gated on the fingerprint **and** `pp_stage_partition` | Any topology drift **raises** (no warm-restart fallback). PP+EP expert weights load at construction |

Path B modes transform the model at init (EP fuses experts into 3D tensors; CP wraps attention), so
the gathered HF-format checkpoint cannot load back through the Trainer. Instead the model source is
repointed at the checkpoint, `load_distributed_model()` loads the trained weights at construction, and
the Trainer then restores `trainer_state.json` and the LR scheduler. TP repoints on `is_tp_mode`
alone rather than riding on `needs_ep_wrappers`: gating it on `use_grouped_gemm` would leave TP+DP —
the one shape whose reload the loader refuses — to surface only after the run had already restarted. A
full-finetune Path-B resume from a checkpoint that is neither directly loadable nor an adapter — an
unmerged per-rank save — **raises** rather than silently continuing from base weights.

**PP+EP** checks that same construction identity explicitly: a stage's fused expert tensors load only
at construction and are absent from its key map, so a stage built from anything but the resume
directory would restore its experts from base weights. The verdict is **world-joined** — each rank
votes, the votes are gathered in one collective over the global group (each rank's reason string
travels with it, so the joined message names the offending path), and the raise fires on all ranks
or none. Both of its inputs are rank-local (a split can leave a stage holding no MoE layer at all, and
the path comparison resolves on that node's filesystem), so raising locally would drop the offended
ranks while their peers walked into the next collective and hung.

Because the Path-B base is rebuilt fresh, two classes of trained state are restored separately by
`CheckpointLoader`:

- **Adapters** (`restore_adapters`, `src/distributed/checkpoint/peft.py`) — native EP expert adapters (keyed
  `<layer>.experts.<attr>.lora_*`) sliced per rank, PEFT attention adapters via
  `set_peft_model_state_dict`, which warns on partial matches and raises if *every* saved key is
  unmatched, so a silent zero-init resume cannot pass quietly. Expert adapters that **no** EP layer can
  receive raise too — resuming with EP off, `use_grouped_gemm: false`, or the expert projections
  dropped from `lora_target_modules` would otherwise discard every saved expert delta while reporting
  a successful restore.
- **Wrapper-added params** (`_restore_extra_trained_params`) — anything the unwrapped model declares
  in `_extra_checkpoint_param_names`, read tensor-by-name
  and broadcast into the live FSDP2-sharded params. No-op when the model declares none.

> [!NOTE]
> **Router balancing bias persists across resume — and exports where a slot exists**
>
> Under `bias_update` the balancing state lives in the family's own checkpoint slot (native buffer,
> adopted `router.bias`/`e_score_correction_bias`/`expert_bias`, or a slot materialized on an LFM-2
> `use_expert_bias: false` checkpoint), so it rides `model.safetensors` at its **trained fp32
> dtype** (`balancing_param_keys` exempts it from the bf16 save cast; the `e_score_correction_bias`
> families additionally pin the slot fp32 at load through upstream's
> `_keep_in_fp32_modules_strict`) and serves exactly as trained.
> Under `bias_update_transient` the bias is a plain per-layer attribute, deliberately off the
> sharded state, and exports **nowhere**
> ([mechanism](../training-methods/callbacks.md#routerbiasbalancingcallback)).

Both modes are all-reduced every step and so replica-identical: the save rank writes them to
`router_balancing_biases.pt` (under PP, remapped through `global_parameter_name` and merged to one file
on that same FS-aware save rank) and every rank copies them back on resume. Two restore verdicts are
loud: a saved bias whose shape does not match the live router **raises** (`copy_` would broadcast it),
and a sidecar matching no live router at all warns per rank and is dropped.

`merge_peft_adapters.py` and `convert_to_bf16.py` re-apply that state through `apply_training_sidecars`
(`src/checkpoint/tool_io.py`), splitting on whether the source ships model weights of its own: an
adapter directory has none, so the sidecar is applied into the assembled model's native slots (the merge
starts from base weights that never saw the updates); a source that does carry weights already holds the
bias, and its balancing tensors are re-read from its own shards at their trained fp32 so the bf16
conversion cannot quantize the sign steps away. A config-gated slot the base was assembled without
(LFM-2 with `use_expert_bias: false`) is re-materialized during the apply and the flag flipped on the
merged config.

### Full-state-dict weight loading

Path-A FSDP2 is the one leg that reconstructs a whole state dict: `load_full_state_dict()` on global
rank 0. Plain `load_state_dict()` cannot work, since FSDP2 and TP wrap params as DTensors. TP reads
through `StreamingCheckpointReader` instead — one tensor at a time, per rank, no rank-0 buffer — and
only when the model was not constructed from the checkpoint.

Where those tensors live is resolved in one place, `resolve_checkpoint_weights()`
(`src/checkpoint/format.py`): index first — read as the map itself, so a key-set or
key-subset read opens no shard — then a single `model.safetensors`, then a legacy `pytorch_model.bin`.
It reports the layout and applies no policy, so each reader keeps its own verdict on an empty
directory, a per-rank sharded index or a torn one. FSDP2 distributes with
`set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True)`; TP `distribute_tensor`s each
tensor into the live param's own placements (`src_data_rank=None`, so the per-rank placement is
collective-free) — per style the exact inverse of the `shard_param` the load ran — and slices the
hand-sliced GptOss `sinks` by `tp_rank`.

Whether the checkpoint is readable is decided once on global rank 0 and broadcast, so a torn or absent
file sends **all** ranks to the base-Trainer fallback together; after a positive verdict a rank-local
read failure raises (a torn checkpoint on a non-shared FS).

### Checkpoint detection

`detect_resume_checkpoint()` (`src/training/environment.py`) handles three cases: `true`
auto-detects the last `checkpoint-N` in `output_dir` via HF's `get_last_checkpoint()`; an explicit
`/path/to/checkpoint-N` uses that path and raises if it does not exist; unset / `false` returns `None`.
When auto-detection finds no checkpoint the run is fresh, so the non-empty-`output_dir` guard re-runs
— set `overwrite_output_dir: true` to bypass it. Detection errors are decided on rank 0 and broadcast
so every rank raises together.

### Configuration

```yaml
resume_from_checkpoint: true        # true=auto-detect, or explicit /path/to/checkpoint-N

save_strategy: steps                # "steps", "epoch", or "no"
save_steps: 500
save_total_limit: 3                 # keep last N, delete older
save_only_model: false              # false: per-rank optimizer shards + scheduler + RNG
                                    #        (exact resume; same world size + layout required)
                                    # true:  weights + trainer_state + scheduler + balancing biases
                                    #        (warm restart, any world size)
save_on_each_node: false            # multi-node; auto-forced true on a non-shared OUTPUT filesystem

save_sharded_ep: false              # EP per-rank sharded save (needs merge_ep_shards.py)
```

All 12 training scripts under `scripts/training/` support resume — each reading the flag from its
own config dataclass.

### Warm restart vs exact resume (torchrun)

Applies to every torchrun sharded mode — FSDP2 DP, EP, EP+ETP, EP+CP, EP+TP, CP, pure TP.

| Aspect | Warm restart (`save_only_model: true`) | Exact resume (`save_only_model: false`) |
|--------|:--:|:--:|
| Model weights | Restored (Path A) / rebuilt via `load_distributed_model()` (Path B) | Same — the weight path is independent of optimizer resume |
| Dataloader position | Skipped to step | Skipped to step |
| Optimizer state | Reinitialized | Restored from per-rank shards (matching fingerprint) |
| LR scheduler | Restored from `scheduler.pt` | Restored from `scheduler.pt` |
| RNG states | New | Restored per-rank |
| Checkpoint size | Smaller | ~2–3× larger (optimizer shards) |
| World size + layout | Any | Must match the saved fingerprint |

Exact resume saves per-rank optimizer shards via `torch.distributed.checkpoint.state_dict` — no gather
to rank 0 — holding FSDP2 DTensor shards for sharded params and plain per-rank tensors for EP experts,
and restores them with `set_optimizer_state_dict` gated on the `optimizer_meta.pt` fingerprint.

**Multi-group EP deduplicates the replicated half.** The EP groups are DP replicas: the FSDP-ignored
params (the whole EP module — experts, router, shared expert) hold identical bytes on every rank of an
`expert_replica_group`, and so do their moments, since the gradients are averaged over that group and
`AdamWBF16` draws its stochastic rounding from a rank-synchronized RNG. Only the group's **lowest rank**
keeps them in its shard; its peers strip them and read them back from that rank's shard on resume. At
Qwen3.5-397B-A17B on 512 GPUs at `ep8` that is 64 copies collapsed to one — a checkpoint's optimizer
state drops from ~103 TB to ~4 TB. The dedup needs a shared output filesystem (a peer must be able to
read the writer's shard); on a per-node filesystem every rank writes its own copy and the save warns
once naming the fix. A rank whose shard is missing that state and cannot find the writer's fails its
read verdict — the whole world warm-restarts (raises under PP) rather than one rank restoring a
partial optimizer — and logs the file it looked for. So does a merge that restores **none** of the
replicated FQNs the saving optimizer tracked: the writer's shard is then keyed differently from this
run's expert parameter names, which the missing-FQN gate cannot see (the dedup keeps `param_groups`
whole, so those FQNs are subtracted out of it). `expert_replica_size` in the fingerprint warm-restarts
a resume under a different replica layout, as every other fingerprint field does; it records how many
shards the dedup collapsed into one, so it is `1` wherever no writer runs (every dense run, and every
`ep1` run at the default `fsdp_shard_ep1_experts`), and a meta that predates the field reads as
compatible with any replica size rather than as `1` — those shards carry every rank's own copy.

Both reads of a resume — this rank's shard and, for a replica follower, the writer's — run under the
node's `max_concurrent_loading` throttle. Unthrottled, a node peaks at `local_world_size ×
(own + writer)` shards in host RAM, and one writer's file is opened by one reader per follower (63 of
them at `ep8` on 512 GPUs). Every rank enters the throttle, whether or not it has a merge to do: it is
a store phase over the node's local ranks.

Before the shard save's `get_optimizer_state_dict` — and before the FSDP2 weight load's
`set_model_state_dict` — every FSDP2 module is resharded (`reshard_fsdp2_modules`,
`src/distributed/fsdp.py`). An eval-only forward has no backward to trigger the reshard
and leaves the transient unsharded params registered (HF evaluates immediately before the
end-of-training save under `eval_strategy: steps`), and the torch APIs map optimizer params to FQNs by
identity against `named_parameters()` — without the reshard every FSDP2 param goes unmapped and the
save produces no optimizer state.

The fingerprint is broader than the sizes: **`world_size`, `pp_size`, `ep_size`, `cp_size`, `tp_size`,
`expert_tp_size`, `ep_scope`, `nvlink_domain_size`, `hsdp`, `use_grouped_gemm`,
`fsdp_shard_ep1_experts`, `expert_replica_size`, and the optimizer class** — a run that changes any one of them (a flipped
`use_grouped_gemm`, a different `ep_scope`) warm-restarts even at the same GPU count.

A fingerprint mismatch falls back to warm restart with a warning naming the differing fields; shards
absent on every rank warm-restart only when nothing proves state was written — if the directory still
holds other ranks' `optimizer_shard_*.pt` or a fingerprint-matched `optimizer_meta.pt` (a non-shared
filesystem whose restart permuted the rank→node placement), the resume raises instead: restore the
original placement, or delete every `optimizer_shard_*.pt` and `optimizer_meta.pt` to accept the warm
restart. Shards present on only a subset under a matching fingerprint are a torn checkpoint and raise.

Shards whose `optimizer_meta.pt` carries **no fingerprint at all** (a pre-fingerprint checkpoint) are
refused outright in every mode, with no warm-restart fallback: nothing records the sharding that
produced those raw local layouts, and the rank-count gate alone admits a permuted restore at the same
world size. Delete every `optimizer_shard_*.pt` and `optimizer_meta.pt` to resume the weights, step and
LR schedule with a fresh optimizer. That verdict is consensused before any rank branches on it: on a
non-shared filesystem `optimizer_meta.pt` is written once per node, so a heterogeneous meta set (one
node holding a fingerprint-less one) would otherwise send some ranks into the raise and the rest into a
collective those ranks never reach.

Under PP every one of these gates **raises** instead — the shards are keyed by stage-local FQNs, so any
drift maps moments onto the wrong layers, and that includes a `pp_stage_partition` differing from the
live split. In every mode, a trainable param the saved shard neither holds state for **nor lists in its
`param_groups`** raises — that is the FQN drift a rename or a different layer split produces. A param
the shard tracked without giving it state (it never received a gradient — GptOss attention sinks under
a sink-dropping kernel) has nothing to restore and passes.

Deleting every `optimizer_shard_*.pt` plus `optimizer_meta.pt` is the explicit warm-restart opt-in.
That opt-in is enforced: resuming a sharded-optimizer checkpoint in a **non-sharded** mode (single GPU,
DDP) raises at load — the base Trainer's loader recognizes only `optimizer.pt`/`.bin`, so it would
otherwise restore no optimizer state while weights, step and LR schedule resume. Every restore verdict
is rank-uniform: `set_optimizer_state_dict` issues DTensor collectives, so all ranks warm-restart
together or none do.

Checkpoint **rotation** (`save_total_limit`) is deferred past the toolkit's sidecars: the base Trainer
rotates as the last step of its own save, which would delete the oldest checkpoint before this save's
optimizer shards and scheduler exist — a preemption in that window under `save_total_limit: 1` would
leave exactly one checkpoint with no optimizer state. The mixin neutralizes the base rotation and
rotates only after every sidecar is on disk, and never after a failed save.

Under `save_only_model: false`, a checkpoint that cannot carry its optimizer state **is** such a failed
save. `OptimizerShardStore.save` raises on every rank when any rank could not produce its optimizer
state — the verdict is a world all-reduce — and the caller's two following steps never run: the base's
rank-0 `optimizer.pt` is left in place and rotation does not fire, so the previous checkpoint stays the
good one. Known trigger: an optimizer whose `state_dict` refuses the live sharding (FlashAdamW on
unevenly-sharded DTensors), which would otherwise write an optimizer-less checkpoint on *every* save
and, under `save_total_limit: 1`, rotate the last good one away at exit code 0. `save_only_model: true`
is the explicit opt-in for weights-only checkpoints.

> [!NOTE]
> **accelerate launch**
>
> Under `accelerate launch` (DDP or accelerate-managed FSDP), `save_only_model: false` is always
> supported; optimizer, scheduler, and RNG states are restored by HF Trainer and accelerate's FSDP
> plugin.

## Multi-node

**Shared output FS (default):** all ranks see the same dir; only rank 0 saves;
`detect_resume_checkpoint()` broadcasts the path from rank 0.

**Non-shared output FS** (`DIST_OUTPUT_SHARED_FILESYSTEM=0`, or the `DIST_SHARED_FILESYSTEM=0`
umbrella): the trainer auto-forces `save_on_each_node=true`, so each node's local rank 0 saves
independently; all nodes must have equivalent checkpoint dirs at the same path. Directory creation is
FS-aware: shared = global rank 0 creates; non-shared = each node's local rank 0 creates, then a
barrier.

**WandB run resumption:** when `WANDB_RUN_ID` is unset, the auto ID is
`md5(f"{output_dir}:{int(time.time())}")[:8]`. The timestamp synchronizes the ID across **ranks of one
launch** (rank 0 broadcasts it) but is **not** stable **across launches** — a resumed launch in the
same `output_dir` gets a new ID and a new run. Export `WANDB_RUN_ID` (and typically
`WANDB_RESUME=allow`) to continue.

Five environment variables govern checkpoint behavior — `DIST_OUTPUT_SHARED_FILESYSTEM` (or its
`DIST_SHARED_FILESYSTEM` umbrella), `DIST_NCCL_TIMEOUT_MINUTES`,
`HALO_ALLOW_MISSING_CHECKPOINT_KEYS`, `WANDB_RUN_ID`, and `WANDB_RESUME`; defaults and full semantics
are in the [Configuration Reference](configuration-reference.md#environment-variables). The one
checkpoint-specific nuance: a gathered save holds non-writers inside its per-layer expert gathers while
the writer streams each flush to disk, so for 100B+ models size `DIST_NCCL_TIMEOUT_MINUTES` to
checkpoint size ÷ filesystem write speed, with margin.

A write that fails on any rank — ENOSPC, EIO, a stale NFS handle — stops the whole job at the next
collective with that rank's real error, rather than leaving its peers in a gather until the watchdog
fires. `DeferredRankFailure` (`src/distributed/runtime.py`) is the seam: it records the local
failure, lets every rank finish the collectives, then raises everywhere.

## Troubleshooting

Symptom→fix for the rest of the toolkit lives in [Troubleshooting](troubleshooting.md); these are
checkpoint-specific.

- **`missing keys: ['lm_head.weight']` on resume:** expected for `tie_word_embeddings=true` models
  (Qwen3); `lm_head.weight` shares storage with the embedding. Harmless.
- **`resume_from_checkpoint=True` but starts at step 0:** check `output_dir` has a `checkpoint-N` dir
  containing `trainer_state.json` with a valid `global_step`.
- **Checkpoint not found on non-shared FS:** set `DIST_OUTPUT_SHARED_FILESYSTEM=0` (or the
  `DIST_SHARED_FILESYSTEM=0` umbrella) so the trainer auto-forces `save_on_each_node=true`; otherwise
  only rank 0 saves.
- **TP resume shape mismatch or hang:** the load path needs a device mesh whose `mesh_dim_names`
  include `"tp"` (a 2-D `(dp, tp)` mesh resolves to its `tp` sub-mesh), and all ranks must reach the
  same checkpoint dir. The weights are gathered, so only the optimizer shards pin `tp_size` to save
  time.
- **OOM saving TP gathered:** each `full_tensor()` reconstructs one whole matrix on the save rank, so
  a single huge embedding can still exceed host RAM even though the write streams. Lower
  `save_max_shard_size` to shrink the pending shard, reduce `tp_size`, or give the node more host RAM.

## Related pages

[Trainer Architecture](trainer-architecture.md) · [Expert Parallelism](../parallelism/expert-parallelism.md) ·
[Tensor Parallelism](../parallelism/tensor-parallelism.md) · [Context Parallelism](../parallelism/context-parallelism.md) ·
[Pipeline Parallelism](../parallelism/pipeline-parallelism.md) · [Multi-Node](../parallelism/multi-node.md)
