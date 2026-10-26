# Checkpoints and Training Resume

Saving and resuming live in `src/checkpoint/parallel/`. The trainer's `save_model()` hands a
`CheckpointContext` (built from explicit trainer fields by `CheckpointingMixin._checkpoint_context`) to
`save_checkpoint()`, which writes the tokenizer, then runs the first saver its `select_checkpoint_saver()`
ladder matches. Every saver takes `(ctx, output_dir)`. On resume `load_resume_weights()` restores the
weights (or declines to the base Trainer's own load), `OptimizerShardStore` the per-rank optimizer shards,
and `sidecars.py` the LR scheduler and the router balancing biases.

The **save rank** follows the **output** side of the filesystem declaration
(`DIST_OUTPUT_SHARED_FILESYSTEM`, falling back to the `DIST_SHARED_FILESYSTEM` umbrella, default shared):
on a shared output FS only **global rank 0** writes; on a non-shared one each node's **local rank 0**
writes its own copy.

| Mode | Saver | Which rank saves | Collective |
|------|-------|------------------|------------|
| **PEFT adapters** | `save_peft_adapters` — adapter weights only ([PEFT saving](#peft-lora-saving)) | Save rank | `full_tensor()` of DTensor adapters |
| **[PP / PP+EP](../parallelism/pipeline-parallelism.md)** | `save_pp_checkpoint` — each stage writes its own complete tensors | one writer per stage (`stage_local_rank == 0`; per node on a non-shared FS) | world-wide index exchange |
| **Expert LoRA** (adapter-only) | `save_expert_lora_adapters` — the gathered grouped adapters | Save rank | EP all-gather |
| **EP** (incl. ETP, EP+CP, EP+TP) | `save_ep_gathered_checkpoint` | Save rank | EP all-gather (+ `full_tensor()` under TP) |
| **EP sharded** | `save_ep_sharded_checkpoint` — each rank writes its expert shard | All ranks | `combine_across_ranks` (gather to rank 0, broadcast back) |
| **TP** (HF-native or selective) | `save_tp_checkpoint` — reconstruct DTensors, then all-gather the hand-sliced GptOss `sinks` (which the parameter walk skips: it would emit this rank's slice under the same key) | Save rank | `DTensor.full_tensor()` |
| **CP** | `save_cp_checkpoint` — remap Ulysses wrapper keys to HF paths | Save rank | `full_tensor()` |
| **FSDP2** (plain torchrun DP) | `save_fsdp2_checkpoint` — stream the gathered chunks | Save rank | `full_tensor()` |
| **Single-GPU / accelerate** (no EP layers) | no saver (the ladder returns `None`) → HF Trainer default | Main process | None |

Selection reads what the model **is**, not the configured sizes: a model carrying EP layers takes the EP
rung even at `ep_size == 1` (experts replicated, FSDP2-sharded), for every trainer.

`EmbeddingTrainer` runs the same ladder with the checkpoint context re-pointed from the
`SentenceTransformer` `nn.Sequential` at the `auto_model` backbone, then writes the ST pipeline config
beside it. Its in-place-injected LoRA is not a `PeftModel`, so the PEFT rung never sees it: that branch
skips the ladder (writing the tokenizer through the same `save_tokenizer`), folds the adapter into each
tensor as it is gathered and streams the result through `stream_gathered_checkpoint`. How it resumes,
LoRA or full fine-tune: [Embedding — Saving](../training-methods/embedding.md#saving).

**Gathered saves** (the default everywhere) produce HuggingFace checkpoints `from_pretrained()` loads.
The optional **per-rank sharded EP save** (`save_sharded_ep`) is a write-bandwidth option that must be
merged before loading.

Every gather is a collective, so **all** ranks run it. Only the save rank copies the result to host memory
(`materialize_dtensor` on the others; under EP the same `retain` flag skips the layout assembly in
`EPGatherMixin.gather_layer_state_dict`), so a node pays one writer's copy instead of `gpus_per_node` ×
model. Every gathered save **streams** through `stream_gathered_checkpoint`: FSDP2, CP and TP one
conversion-closed chunk (a decoder layer) at a time, EP its non-expert tensors and then each MoE layer as
it is gathered. The writer holds at most one chunk plus one pending shard, so host memory stays well
under 1× model ([measured peaks](../parallelism/expert-parallelism.md#checkpointing)).

The tokenizer is written once per save, before the weights (`save_tokenizer`), so a save stopped after
its index names the weight files is already loadable with its tokenizer. A failed tokenizer write raises
on every rank.

The streamed parts are renamed to HF's own `model-{i}-of-{n}` once the count is known, so
`from_pretrained` reads the result exactly as a `save_pretrained` checkpoint. Part *boundaries* can
differ from HF's splitter's — the streaming writer flushes a block before an oversized tensor — so the
file count and key→file mapping can differ while both load the same.

## What lands on disk — four layouts

Tell them apart by `model.safetensors.index.json`, not the filename: the per-rank layout carries a
`metadata.format` marker (`is_ep_sharded_checkpoint` reads it). The `format: "pt"` inside each
safetensors *file* header is the container format, not this marker.

| On-disk layout | Files | Produced by | Index marker | Loadable? |
|---|---|---|---|---|
| **Single file** | `model.safetensors` | any gathered save under the shard threshold | *(no index)* | Yes |
| **Gathered HF-sharded** | `model-00001-of-000NN.safetensors` + index | any gathered save over the threshold — the default for production-size models | none | Yes |
| **PP per-stage** | `model-pp00000-of-00004-00001.safetensors` — one or more parts per stage + merged index | `pp_size > 1` | none | Yes on a shared output FS; per-node directories `merge_pp_stages.py` first |
| **EP per-rank sharded** | `model-00000-of-000NN.safetensors` (`.shard_{rank}` keys) + index | `save_sharded_ep: true` | `format: "ep_sharded"` | No — `merge_ep_shards.py` first |

Shard size defaults to 5 GB (`save_max_shard_size`) and bounds the **gathered** writers. A per-rank EP
shard is one file per rank by design, and the save logs that rather than appearing to honor the cap. The
after-training tools size their own output from their `--max_shard_size` flag
([Scripts](scripts-reference.md#post-training-scripts)).

The per-rank EP layout holds **partial** tensors. Every loader detects the marker and refuses it, naming
the merge script, and every after-training tool refuses a per-rank directory whose index never landed (a
header-only peek finds its `.shard_N` keys) ([input guards](scripts-reference.md#input-guards)).

The **PP layout is not a per-rank format**, which is why it carries no marker. Each stage writes
*complete* tensors under the hub's names — the unsplit model's (`stage.global_parameter_name`, EP experts
re-exported through their family gather), reverted through the load conversions the stage snapshotted at
its build, as the EP-gathered save reverts them ([Serving](#serving-on-vllm--sglang)). Each stage owns a
filename prefix derived from its `pp_rank`, so the writers stream concurrently into one directory. The
index exchange gathers the weight-map fragments to rank 0 and broadcasts back the merged map with the
collision verdict, so a key claimed by two stages raises on every rank. On a per-node output filesystem
each node's directory holds the full index but only its stages' parts: every tool refuses it, naming the
absent parts, until `merge_pp_stages.py` merges the node directories
([Per-node export](../parallelism/pipeline-parallelism.md#per-node-export)).

## What gets saved

| File | `save_only_model=false` | `save_only_model=true` | Purpose |
|------|:-:|:-:|---------|
| `model.safetensors` (or sharded + index) | Yes | Yes | Weights |
| `config.json`, `tokenizer.*` (full processor for VLMs) | Yes | Yes | Architecture config + `processing_class` |
| `generation_config.json` | Yes | Yes | Sampling defaults, stop strings and the custom `eos_token_id` set — written for a generative model only, and paired with `config.json` by `save_model_config`. `PretrainedConfig` does not carry it, so a save that writes only `config.json` serves with defaults. |
| `trainer_state.json` | Yes | Yes | Step count, epoch, loss history — published last ([Interrupted saves](#interrupted-saves)) |
| `optimizer_shard_XXXXX.pt` | Yes | No | Per-rank optimizer shard — every torchrun sharded mode. Under multi-group EP only the replica group's lowest rank carries the replicated (FSDP-ignored) expert state; its peers read it from that shard |
| `optimizer_meta.pt` | Yes | No | Topology fingerprint gating exact resume |
| `scheduler.pt` | Yes | Yes | LR scheduler state — re-persisted on every mode so resume continues the schedule |
| `router_balancing_biases.pt` | Yes | Yes | DeepSeek-V3 router balancing biases, restored on resume |
| `reference_logps.pt` | Yes | Yes | Frozen per-split reference scores: DPO/KTO `precompute_ref_log_probs` columns and offline GRPO raw completion-token log-probs. Resume validates token digests, row counts and reference settings before attaching them ([DPO](../training-methods/preference/dpo.md#resuming-a-precompute-run), [Offline GRPO](../training-methods/grpo/offline.md#reference-model)) |
| `rng_state_<rank>.pth` | Yes | No | Per-rank RNG state (`rng_state.pth` single-process) |
| `prefetch_pending-<rank>-of-<world>.pt` | Yes | Yes | Async GRPO with prefetch only: the rounds a rank submitted but had not trained and the layout they were drawn under, submitted again first on a resume under that layout ([Multiple servers and prefetch](../training-methods/async-grpo/setup.md#multiple-servers-and-prefetch)) |
| `resume_adapter/`, `resume_adapter.json` | Yes | Yes | `merge_expert_lora_on_save` and embedding LoRA only: the unmerged adapter the merged checkpoint resumes from, and the marker that says so ([Merge-on-save checkpoints](#merge-on-save-checkpoints)) |
| `training_provenance.json` | Yes | Yes | GptOss adapter saves only: whether the attention sinks were live or neutralized ([PEFT saving](#peft-lora-saving)) |

`optimizer.pt` is dropped under every torchrun sharded mode, which holds per-rank shards instead. Plain
`accelerate launch` DDP is not intercepted by the mixin, so the base HF Trainer writes and keeps
`optimizer.pt`.

Every gathered save writes parameters **plus persistent buffers** (Gemma4 `layer_scalar`, vision
`std_scale`/`std_bias` — a dropped residual scalar corrupts the layer on reload) and reconstructs the
neutralized GptOss attention `sinks` that FA2 fine-tuning removes from `named_parameters()`.
Non-persistent buffers (rotary caches, attention masks) are recomputed on load. One `persistent_buffers()`
helper drives the EP, TP and FSDP2 gathers (CP gets the set through the wrapper's `state_dict()`); the
buffer leg is collective too, so the FSDP2, TP and CP gathers share one preamble that runs it on every
rank.

A training checkpoint (`checkpoint-N`, `CheckpointContext.training_checkpoint`) writes every tensor at
its live dtype; the `save_model` export casts to bf16 except three keep-sets: normalization params, the
live router-balancing tensors, and the family's fp32 pins. fp32 masters (`fp32_router`, `fp32_experts`,
`fp32_non_ep_params`) and fp32 adapters therefore take 4 bytes per element in a checkpoint —
`fp32_experts` doubles an MoE checkpoint's expert bytes — and every checkpoint start restores them at FP32
under every parallelism mode ([Load precision](../models/README.md#load-precision)). A BF16 export cannot
recover precision it discarded.

`ReferenceLogpsMixin` persists DPO/KTO and offline GRPO scores on the filesystem-aware save rank; the save
completes only after the file and its parent directory are synced. Unchanged checkpoints hardlink the
immutable file (a copy where links are unavailable), and write failures rendezvous across ranks before
rotation can remove the previous complete checkpoint. Offline GRPO's ragged payload and its recovery:
[reference contract](../training-methods/grpo/offline.md#reference-model).

Staging and durable publication live in `src/checkpoint/atomic.py` (`staged_file` for one file). Every
file Halo's writers create gets `0o666` under the process umask; a file copied from a source keeps that
file's mode. safetensors creates its files `0600`, so every safetensors writer goes through
`save_safetensors_file`, and a `save_pretrained` writer is followed by `apply_umask_mode` on its weight
file or on the directory it owns — each sentence-transformers module folder (`2_Dense/`,
`2_Router/query_0_Dense/`) included. `apply_umask_mode` reads the mode off a probe file, so a default
ACL applies too. Export copies omit every unpublished file an interrupted save leaves: a staged write's
stage, the withheld `.trainer_state.json.uncommitted`, and the `.<name>.<uuid hex>.tmp` spelling.

A failed write raises on every rank; no gathered save falls back to `pytorch_model.bin`. Only the
config write is gated on the model carrying one; a model without a config still gets normalized weights
in safetensors.

### Exported `config.json`

One step, `finalize_exported_config` (`src/checkpoint/config_export.py`, which owns the whole contract),
ends every config write — the parallel saves through `save_model_config`, `save_full_checkpoint`, and the
single-GPU/DDP/accelerate-FSDP fallback in the trainer. A path running a subset ships a directory that
trains and reloads but that the merge tools or the pinned server refuse. It rewrites the written
`config.json`, then tags the model card ([Hub model card](#hub-model-card)):

- **The `model_type` restore** (`restore_model_type`): puts the live `model_type` back when a vendor
  config class (Bailing/Ling) leaves it undeclared; without it every model-type-keyed reader (the shard
  merge, the hub renames) sees no family.
- **The source checkpoint's own schema** for a family whose engines have no config class and read it
  only through its `auto_map` modules (`exports_source_config_schema`, Step-3.7 Flash):
  `export_source_config_schema` writes the run's changed values back under the source's spellings and
  copies the `auto_map` modules in transitively. The source is the checkpoint the model was loaded from,
  so a resume or a `patch_vocab.py` output carries the schema on. The rewritten config is re-parsed
  against the live one, so a change the source schema swallows raises; a source with no `auto_map` warns
  and keeps transformers' own.
- **The `auto_map` trim**: an entry naming a module the directory does not carry is dropped (an
  in-library load serializes the hub's `auto_map` without copying the code — Laguna without
  `trust_remote_code`); a cross-repo `repo--module.Class` reference stays.

Everything else is transformers' own serialization. A Gemma 4 export carries its full-attention
geometry as `per_layer_config` instead of the hub's flat `global_head_dim` /
`num_global_key_value_heads`: both pinned engines and transformers 5.16+ read it, and a 5.14-line
reader refuses it.

The parallel saves (`save_model_config`) and the trainer's base-save fallback serialize the config inside
`config_export_ready`, which restores the balancing strategy's zeroed `router_aux_loss_coef`, forced
`output_router_logits` and toolkit stamp before the write: they configure the run, not the artifact, and
exported they would destroy the hub's aux coefficient for later stages. A new writer that serializes a
training run's config needs it too; `finalize_exported_config` does not apply it.

Gathered saves also reconcile `tie_word_embeddings`. Tied embeddings can diverge in distributed training
(independent FSDP2 shards, FLCE training `lm_head` directly), so when the saved `lm_head` and
`embed_tokens` differ (matched by key suffix), the config is written `tie_word_embeddings: false` on every
level that declares it (`untie_word_embeddings`) and the trained `lm_head` is honored on reload.

## Hub model card

Every checkpoint Halo writes, tool conversions included, carries a `README.md` card tagged `halo`
(`HUB_TAGS` in `src/checkpoint/model_card.py`). The tag lands in `finalize_exported_config` (full-model
writes), `copy_checkpoint_aux_files` (exports built from a source directory), `save_peft_adapters` /
`save_expert_lora_adapters`, the checkpointing mixin's `create_model_card` (TRL's per-checkpoint card) and
the SentenceTransformer's `model_card_data`; a new writer ends in one of the first two or calls a tagger
itself. `tag_model_card` changes only the `tags` value of a card already present, re-dumping its metadata
block (YAML comments and flow style are lost), and raises on malformed metadata, failing the save that
called it; the export finalizers and `reset_sinks` call `tag_exported_model_card`, which leaves such a card
untagged with a warning. A fresh card in an adapter directory stock PEFT loads also carries
`library_name: peft` and a Hub `base_model`. An existing card's `base_model` stays as it is: PEFT's own card
(the single-GPU / DDP adapter save) can name a local path, which the Hub refuses — replace it with the Hub
id before uploading.

Under `push_to_hub: true`, the four trainers that extend `transformers.Trainer` directly (SMPO, offline
GRPO, classification, teacher distillation) upload no card to the repo root:
`Trainer._push_from_checkpoint` copies only the model files into `output_dir`. A `hub_strategy` of
`checkpoint` or `all_checkpoints` also uploads the checkpoint directory, whose card lands under that
subfolder only.

## Serving on vLLM / SGLang

A gathered checkpoint is a standard HF checkpoint — stock `from_pretrained` loads it as-is. (Live RL
weight sync is narrower: [Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves).)

Every writer saves the names transformers' own `save_pretrained` writes: the reverse of what the load
converted. transformers reads several hubs through renames its module tree drops — a SigLIP tower's
`vision_model` level (Command A+, LFM2-VL), a vendor namespace (Inkling's `model.llm.*`, GLM-5 Next's
`hc_attn_*`, Step-3.7's `moe.*`, DeepSeek-V4's `attn`/`ffn` names), Mistral 3's legacy llava prefixes.
Each writer reverts the conversions the load recorded (`conversions_to_revert`,
`src/checkpoint/hub_layout.py`); a model with no record (built from its config, or by the lazy loader), or
whose load dequantized fp8/MXFP4 weights, reverts the family's registered mapping. The EP-gathered save
and the RL weight sync leave out only the per-expert merges (`export_conversions`), since the gather owns
the expert layout. vLLM 0.32.0 raises on a SigLIP tower key that lacks its `vision_model` level.

**Un-fuse experts.** transformers keeps MoE experts fused in memory (`experts.gate_up_proj` /
`experts.down_proj`). Most hub checkpoints are per-expert, and `from_pretrained` fuses them on load;
Qwen3.5/3.6 and Zaya ship the fused pair. The wrapper-less writers and the base-Trainer save write the
layout the model was loaded from (a model with no record reverts through the family's mapping —
per-expert for Qwen3.5/3.6); an EP-gathered save writes whatever the family's gather emits — the fused
pair for Qwen3.5/3.6, DeepSeek-V4, Cohere2 MoE and GLM-5 Next. A reload into a built model maps any of
these back ([Resuming training](#resuming-training)). Rewrite such a checkpoint only for a
per-expert-only serving loader, with `scripts/after_training/unfuse_moe_experts.py`; Step-3.7 Flash needs
no rewrite (its EP-gathered save lands in the hub's per-layer-stacked `moe.*` namespace,
[Step-3.7](../models/step3p7.md#checkpoint)).

| Serving loader | Families | Fused `gate_up_proj` / `down_proj` |
|---|---|---|
| vLLM 0.32.0 `RoutedExperts` | Qwen3.5/3.6, Gemma 4, Cohere2 MoE ([Command A+](../models/cohere2-moe.md#loading-and-serving)), Laguna, Bailing/Ling 2.0 | loaded directly |
| vLLM 0.32.0 `step3p5` | Step-3.7 Flash | not needed — the EP-gathered save writes the hub `moe.*` tensors |
| vLLM 0.32.0 per-expert-only | GLM-4 MoE Lite (raises), LFM-2 (drops it silently), GLM-5 Next (raises), DeepSeek-V4 (the release's `w1/w2/w3` names) | un-fuse first; GLM-5 Next and DeepSeek-V4 unverified end to end ([DeepSeek-V4](../models/deepseek-v4.md)) |
| vLLM 0.32.0 `bailing_moe_v3` | Ling 3.0 (`bailing_hybrid`) | per-expert loader; no export served end to end ([Bailing](../models/bailing.md)) |
| vLLM 0.32.0 — no model class | Mistral4, Ring — its `BailingMoeLinearV2ForCausalLM` matches no registered architecture ([Rollout Servers](../infrastructure/rollout-servers.md#which-families-each-engine-serves)) | not servable at all ([Mistral4](../models/mistral4.md#serving), [Bailing](../models/bailing.md)) |
| vLLM 0.32.0 — no native class | Zaya | resolves through vLLM's generic transformers backend; unvalidated ([Zaya](../models/zaya.md)) |
| vLLM 0.32.0 `inkling` | Inkling | the export writes the hub's fused `w13_weight` under `model.llm.*`; unverified end to end |

SGLang 0.5.22 reads each family's hub layout through its own per-family loader, so the same un-fuse rule
applies, Cohere2 MoE, DeepSeek-V4 and GLM-5 Next included (their loaders drop the fused pair without an
error; a DeepSeek-V4 export is unverified there even after the rewrite); it registers no class for
Mistral4 or Ring's architecture, and its Zaya loader reads a per-expert layout the gathered export does
not carry.

**MLA backend on Blackwell.** flashinfer's MLA kernel rejects GLM-4 MoE Lite's head config on SM100+ —
serve with vLLM `--attention-backend CUTLASS_MLA` or SGLang `--attention-backend triton`.

**Text-only exports of a multimodal checkpoint.** A `text_only_model` load (or a merge of its adapter)
reads the checkpoint through its `*ForCausalLM` sibling, which strips `model.language_model.` to `model.`.
Every writer (the gathered and EP saves, the base-Trainer save, the weight sync, the tools'
`save_full_checkpoint`) reverts the load's conversions minus that strip (`save_pretrained_layout`), so the
export stores `model.*` under its `*_text` config; `reattach_vision_tower.py` restores the multimodal
layout ([Qwen3.5/3.6](../models/qwen3_5.md)).

## Saving by parallelism mode

### Expert parallelism (EP, EP+TP, EP+CP)

`save_ep_gathered_checkpoint()` (`src/checkpoint/parallel/ep.py`) owns all three, over three param
classes: EP expert weights, gathered by each layer's `gather_layer_state_dict()` into the full runtime
tensors and written in the family's `ExpertLayout` through `to_hub_expert_tensors`
(`src/models/moe/expert_layout.py`); DTensor params (EP+TP attention, embeddings, `lm_head`, FSDP2-sharded
params such as the router under ep1 sharding), resolved via `full_tensor()`; and the hand-sliced GptOss
`sinks`, all-gathered by `gather_hand_sliced_params`. Under EP+CP the iteration runs on the unwrapped model
with an explicit CP-prefix strip; experts are replicated per CP rank, so any rank holds the complete set.

Keys are written in the **hub** spelling: transformers' save-side revert of the load's conversions, minus
the per-expert merges, runs on every streamed chunk, so a vendor or wrapper namespace comes back as the hub
spells it. No reverse entry fuses tensors across two chunks, which
`tests/cpu/checkpoint/test_ep_hub_namespace_export.py` pins per family. Inside an EP layer a family whose
live names differ from its hub's declares the pairs in its spec's `export_key_renames` (Laguna), which the
gather and `merge_ep_shards.py` apply; the lazy loader reads the hub spelling through transformers' own
Laguna renamings. Every lazy build plans its keys with the family's whole transformers conversion mapping,
as `from_pretrained` does (`src/model_loading/checkpoint_keys.py`), a SigLIP tower's `vision_model` level
included. Four families declare the mapping their hub checkpoints need (`hub_conversion_keys`: Inkling,
DeepSeek-V4, GLM-5 Next, Step-3.7 Flash); each declared key must resolve in the installed transformers, and
a text-only class with no mapping of its own (Inkling's text artifact) plans with those entries. A key stays
unconverted when its rename target misses the live names while the key itself is live, the retry
transformers' own loader makes, so a canonical checkpoint (an EP resume of a toolkit save) loads untouched.

**Sharded (`save_sharded_ep: true`)** writes each rank's expert shard keyed by global rank, with global
rank 0 adding non-expert params and the index. `merge_ep_shards.py` needs the shards co-located and
complete, so `reject_unmergeable_ep_sharded_save()` **fails fast at trainer construction** (and again at
the save) on:

- **multiple EP groups / DP replicas** (`ep_group_size != world_size`): replicas share an `ep_rank` and the
  same expert slice;
- **`expert_tp_size > 1`** or **context parallelism**: layouts the merge cannot reconstruct;
- **multi-node on a non-shared filesystem**: no directory holds a complete set;
- **a `model_type` no family spec claims** (`find_moe_family_spec` finds no `ExpertLayout`), or **a load
  that converted names the merge cannot respell** (Step-3.7 Flash, Inkling, DeepSeek-V4, GLM-5 Next, a
  SigLIP tower, a legacy Mistral 3 checkpoint), decided per model off the load's record;
- **native expert LoRA** (the `_lora_A`/`_lora_B` keys would pass the merge dead) and
  **`merge_expert_lora_on_save: true`** (folding needs the gathered save);
- **a run with no EP layers at all**, whose saves are ordinary gathered checkpoints.

Use gathered for deployment and on a non-shared multi-node filesystem (it writes a complete checkpoint per
node). Use sharded only for fast intermediate checkpoints of a large MoE on a shared filesystem or single
node: every rank writes its shard in parallel, so the pause per save is bounded by one rank's shard. Write
bandwidth is the only reason — the gathered path streams, so host memory is bounded either way. Merge
before loading ([tool and guards](scripts-reference.md#post-training-scripts)):

```bash
python scripts/after_training/merge_ep_shards.py \
    --input_dir /path/to/sharded_checkpoint --output_dir /path/to/merged_checkpoint
```

The merge carries the resume state with the weights: step counters, scheduler, balancing biases, reference
log-probs, pending prefetch rounds, RNG states, and every rank's optimizer shard with the `optimizer_meta.pt`
that vouches for them. The merged directory therefore resumes exactly on the topology that wrote it, and
under any other meets the same [fingerprint gates](#warm-restart-vs-exact-resume-torchrun) as the save. A
meta naming a rank whose shard is absent refuses the merge. A resume pointed at the unmerged directory raises
the merge-first error.

### Tensor parallelism (TP)

HF-native TP (`tp_plan="auto"`, dense) and DTensor TP (`parallelize_module()`, the MoE attention-only path
incl. EP+TP) both place their sharded params as DTensors, so `save_tp_checkpoint()` reconstructs either
with one `full_tensor()` walk, plus the plain-tensor all-gather of GptOss `sinks`. One writer per
filesystem scope (`is_fs_save_rank`, never `tp_rank == 0`) avoids write races between the TP groups a node
holds. There is no per-rank TP save. A tied embedding/head pair is one vocab-sharded DTensor under
HF-native TP ([Load paths](../parallelism/tensor-parallelism.md#load-paths)), so the checkpoint carries a
single `embed_tokens.weight` and `tie_word_embeddings: true`, as a single-GPU save does.

### Context parallelism (CP)

`UlyssesModel.state_dict()` (`src/distributed/context_parallel/ulysses_model.py`) strips the
`.original_attention.` segment the patcher introduces. It drops the stale duplicate dense
`.mlp.{gate,up,down}_proj` keys **per layer, only on layers that carry routed experts**, so a hybrid
model's genuinely-dense layers (GLM-4 MoE Lite, Mistral4 `first_k_dense_replace`) keep their MLP weights.
`save_cp_checkpoint` finds the wrapper through the trainer's `_find_cp_wrapper()` (plain and PEFT-nested
layouts). A tied pair is written once under `embed_tokens.weight`, the key set the FSDP2 and TP savers
write.

## PEFT (LoRA) saving

`save_peft_adapters` (`src/checkpoint/parallel/adapters.py`) is the ladder's first rung except under
accelerate-managed FSDP (the base Trainer's save owns the flat-param layout) and
`merge_expert_lora_on_save` (which takes the mode's saver for a merged base). Only adapter weights are
saved: `adapter_config.json`, `adapter_model.safetensors` (`adapter_model.bin` on a safetensors write
failure), `tokenizer.*`.

| Mode | LoRA params | Saving rank | Collective |
|------|-------------|-------------|:----------:|
| **EP** | Attention adapters + native grouped expert adapters | Save rank | Yes (expert gather) |
| **ETP** | Attention adapters only — expert LoRA is rejected at construction under `expert_tp_size > 1` | Save rank | Yes |
| **FSDP2** | DTensors (sharded across DP mesh) | Global rank 0 (shared FS) / local rank 0 | Yes |
| **QLoRA + CP** | Regular tensors (no DTensor adapters) | Save rank | No |

LoRA is rejected at construction under TP / EP+TP. The saver checks for DTensor adapters before the CP
branch, so non-quantized LoRA+CP routes through the FSDP2 path; only QLoRA+CP takes the collective-free CP
branch. Under EP the native grouped expert adapters are gathered into the same
`adapter_model.safetensors`; a directory holding them carries a `peft_type` stock PEFT does not know, so an
external load raises instead of loading the attention half alone
([PEFT](../optimization/peft.md#checkpoint-saving)).

Every save reshards the FSDP2 modules first (`save_model`): mid-training they can still hold the transient
unsharded params a forward registered, which the optimizer never stepped, so every writer would read
tensors one step stale, and the PEFT path's DTensor probe would mis-route to the rank-0 `save_pretrained`
branch. Resume reads the adapter file present on **every** rank (safetensors or `.bin`), since two ranks
reading different files would order the per-key DTensor loads differently.

A GptOss run also writes `training_provenance.json` (`write_training_provenance`,
`src/checkpoint/sidecars.py`) recording whether attention sinks were live or neutralized. The export tools
re-apply it through `apply_training_sidecars`, because a merge rebuilds the base from the hub, whose sinks
are always live. It is a separate sidecar, so stock PEFT keeps loading the adapter unchanged.

### Merge-on-save checkpoints

A `merge_expert_lora_on_save` checkpoint serves from its merged weights but cannot resume from them: the
bf16 fold loses part of the delta, and the optimizer moments belong to the adapters. So `_save_checkpoint`
adds the unmerged adapter to every training checkpoint in `resume_adapter/` (`save_peft_adapters` for a
mixed run, `save_expert_lora_adapters` for expert-only), then the root marker `resume_adapter.json` once
every save rank's copy is complete. Each save first clears its step directory of an earlier save's markers
([Interrupted saves](#interrupted-saves)). The adapter sits in a subdirectory because a root
`adapter_config.json` would make `from_pretrained` apply it on top of weights that already hold its delta;
vLLM, SGLang and `from_pretrained` read only the root. The final `save_model()` export carries neither.
The attention adapters fold out of place as each tensor is written, with the weight sync's fold
([PEFT](../optimization/peft.md#online-rl--rollout-server-weight-sync)), so the run stays bit-identical.

On resume the marker is the verdict: `resolve_resume_weights_source` keeps `model_name_or_path`, and the
loader restores `resume_adapter/` onto the base-built model at the adapters' live dtype (exact for fp32
adapters). It refuses a model built from the merged weights (the delta would apply twice) and a marked
checkpoint missing its adapter. An unmarked merged checkpoint resumed by an adapter run raises; start a new
run from those weights (`model_name_or_path` pointed at the checkpoint) instead. Covering test:
`tests/common/merged_resume_e2e.py`.

`copy_checkpoint_aux_files` treats the adapter and its marker as resume sidecars: a tool whose output is the
same run's weights (`convert_to_bf16`, `merge_ep_shards`, `unfuse_moe_experts`, `reset_sinks`,
`quantize_to_lowp`, `reattach_vision_tower`) keeps them, and a tool whose output is a new base drops them
with every other resume sidecar (`merge_models`, `patch_vocab.py`, `merge_peft_adapters.py`). An embedding
LoRA checkpoint takes the same marker and directory, written and restored by `EmbeddingTrainer`
([Embedding — Saving](../training-methods/embedding.md#saving)).

## Accelerate / FSDP checkpoints

Without EP/CP/TP (`accelerate launch`), saving follows HF Trainer defaults: a single-process save without
FSDP; with FSDP, `fsdp_state_dict_type: FULL_STATE_DICT` gathers to rank 0 and `SHARDED_STATE_DICT` saves
per-rank shards. Under `torchrun`, HF's built-in FSDP is disabled (`config.fsdp = ""`) and the mixin applies
FSDP2 (`fully_shard`) itself.

> [!WARNING]
> **FSDP v1 corruption bug**
>
> FSDP v1 (both SHARD_GRAD_OP and FULL_SHARD) corrupts internal parameter views after gathering,
> producing silently corrupted checkpoints. Prefer FSDP v2
> (`launcher-configs/accelerate/fsdp2_gradop_config.yaml`) or standard DDP (`launcher-configs/accelerate/multigpu_dp_config.yaml`).
> The `torchrun` path uses FSDP2 exclusively and is unaffected.

## Load coverage gate

`from_pretrained` re-initializes any key the checkpoint does not carry and reports it only through the
transformers logger, which non-logging ranks silence — a truncated or wrong-architecture directory would
train from random weights with no output. Every toolkit load path (the per-mode loaders, the lazy EP
loader, the reference/teacher loads) routes its absent set through `verify_checkpoint_coverage`
(`src/model_loading/checkpoint_coverage.py`) and **raises**; a new loader calls `from_pretrained_verified`.

Legitimate absences are read off the model class: a task head outside `base_model_prefix` (`score`,
`classifier`, an untied `lm_head` — what makes reward / classification training on a base checkpoint
work), a tensor the checkpoint carried under another name (object identity over
`state_dict(keep_vars=True)`), and the class's `_keys_to_ignore_on_load_missing`. A class that *is* the
backbone excuses nothing, and the gate inspects `missing_keys` only. A caller that only consumes a head
passes `excuse_task_head=False` (reward-model inference), so a plain causal-LM directory raises instead of
scoring through a random `score` head.
`HALO_ALLOW_MISSING_CHECKPOINT_KEYS=1` downgrades the raise to a warning.

## Resuming training

Set `resume_from_checkpoint: true` (or a `checkpoint-N` path) and re-run the same command, leaving
`model_name_or_path` at the base:

```bash
torchrun --nproc_per_node=8 scripts/training/sft.py config.yaml --resume_from_checkpoint=true
```

Resume is resolved once in `init_training_script` (`src/entrypoints/script.py`): it picks the checkpoint
([Checkpoint detection](#checkpoint-detection)), then `resolve_resume_weights_source()` decides the weights
source. Modes that transform the model at construction (EP fuses experts into 3-D tensors, CP wraps attention)
cannot load the gathered HF checkpoint back through the Trainer, so it repoints to the **checkpoint
directory** (Path B, the trained weights load at construction) when `needs_ep_wrappers`, `is_cp_mode` or
`is_tp_mode` holds — with the default `use_grouped_gemm: true` that is **every stock torchrun run**, dense
included. Only a `use_grouped_gemm: false` run with no EP, CP or TP keeps `model_name_or_path` (Path A, a
reload into the built model), and so does an adapter checkpoint (adapter-only, or merge-on-save through its
marker), whose adapter the loader restores onto the base. TP repoints on `is_tp_mode` alone, not on
`use_grouped_gemm`, so TP+DP — the one shape whose reload into a built model the loader refuses — never meets
that refusal after the run restarted. A full-finetune Path-B resume from a checkpoint that is neither loadable
nor an adapter (an unmerged per-rank save) raises before construction.

`model_config` is not mutated: the frozen reference/teacher and the preprocessed-dataset check keep the base
path, and a policy built from the checkpoint keeps the base as its config's `_name_or_path`, so a resumed
run's `config.json` and model card name the same `base_model` as an uninterrupted run's. `trainer.train()`
then reads `global_step` / `epoch` from `trainer_state.json`, skips the consumed batches, and restores the
optimizer and LR scheduler. WandB does **not** continue the same run on its own ([Multi-node](#multi-node)).

### Resume by parallelism mode

| Mode | Path | Weight loading | Optimizer | Notes |
|------|------|----------------|-----------|-------|
| **FSDP2** (standard DP) | A at `use_grouped_gemm: false`, else B | Path A: `load_full_state_dict()` on rank 0 → `ModuleLayoutView` → `set_model_state_dict(full_state_dict=True, broadcast_from_rank0=True)`. Path B: loaded at construction, the re-read skipped | Per-rank FSDP2 shards | Exact resume, same world size |
| **Single process / DDP** | A at `use_grouped_gemm: false`, else B | Path A: each rank streams the weights into its own copy through `ModuleLayoutView`. Accelerate-managed FSDP and a sentence-transformers pipeline keep the base Trainer's loader | The base Trainer's `optimizer.pt` | Exact resume |
| **TP** | B | Loaded at construction. Where it does read (a best-model reload, a model built elsewhere), each rank streams the checkpoint (`StreamingCheckpointReader`, no rank-0 buffer) and `distribute_tensor`s each tensor into the live placements (`src_data_rank=None`), GptOss `sinks` sliced by `tp_rank`; TP+DP raises for a model not constructed from the checkpoint | Per-rank shards | Exact resume, same world size |
| **CP** | B | Loaded at construction via `load_distributed_model()` | Per-rank shards (matching fingerprint) | Checkpoint has HF keys, model has CP wrapper keys |
| **EP / ETP / EP+CP / EP+TP** | B | Loaded at construction; the checkpoint holds the hub-spelled full expert tensors, not the rank-local slices the EP model registers | Per-rank shards (matching fingerprint) | A model built from anything but the checkpoint makes the loader refuse the resume |
| **[PP / PP+EP](../parallelism/pipeline-parallelism.md)** | B at the default `use_grouped_gemm`, else stage-aware A | Path B: the stage loader read the trained tensors at construction. Path A (and a best-model load): each rank maps the merged index's hub names onto the unsplit model's (`ModuleLayoutView`, with the conversions the stage snapshotted), then onto its stage's through `global_parameter_name`, and reads only those; a missing stage-retained tensor raises, and so does a stage shard absent from this node's copy. On per-node output storage the locally-absent cross-stage tensors are skipped | Per-rank shards, gated on the fingerprint **and** `pp_stage_partition` | Any topology drift **raises** (no warm-restart fallback). PP+EP expert weights load at construction only, so a stage not built from the resume directory raises, on every rank at once |

Every whole-weight read into an already-built model (Path A, and every best-model load) first maps the
checkpoint onto the live names through `ModuleLayoutView` (`src/checkpoint/hub_layout.py`), replaying the
conversions the save reverted, so a per-expert checkpoint lands in fused experts. It plans the keys with the
lazy loaders' planner (`src/model_loading/checkpoint_keys.py`), so a resume and a lazy build read a
checkpoint the same way. Where the tensors live is
resolved by `resolve_checkpoint_weights()` (`src/checkpoint/reading.py`): the index first (read as the map
itself, so a key-set read opens no shard), then a single `model.safetensors`, then a legacy
`pytorch_model.bin`; each reader keeps its own verdict on an empty, per-rank or torn directory. Whether the
checkpoint is readable is decided on global rank 0 and broadcast, so a torn file sends all ranks to the
base-Trainer fallback together; a rank-local read failure after a positive verdict raises.

Because the Path-B base is rebuilt fresh, `load_resume_weights` restores adapters separately
(`restore_adapters`, `src/checkpoint/parallel/adapters.py`): native EP expert adapters sliced per rank, PEFT
attention adapters via `set_peft_model_state_dict`, from the checkpoint root or `resume_adapter/`. It warns
on partial matches and raises if *every* saved key is unmatched, if no EP layer can receive the expert
adapters, or if the LoRA scaling differs from the saved `adapter_config.json`'s
([PEFT](../optimization/peft.md#checkpoint-saving)).

Router balancing biases: under `bias_update` they live in the family's own checkpoint slot, ride
`model.safetensors` at their trained fp32 (`balancing_param_keys` exempts them from the bf16 cast) and serve
as trained; under `bias_update_transient` they are a plain per-layer attribute that exports nowhere
([mechanism](../training-methods/callbacks.md#balancingbiascallback)). The save rank writes both modes'
biases to `router_balancing_biases.pt` and every rank restores them
([The balancing sidecar](../training-methods/callbacks.md#the-balancing-sidecar)); the export tools re-apply
them through `apply_training_sidecars` — into an adapter merge's native slots, or re-read at fp32 from a
source that carries weights.

### Checkpoint detection

`resume_checkpoint_from_config()` (`src/entrypoints/run_setup.py`): `true` auto-detects the newest
`checkpoint-N` in `output_dir` whose `trainer_state.json` exists on every rank, passing over newer step
directories a save never completed and raising when only incomplete ones exist; an explicit path must exist
with its trainer state on every rank; unset / `false` returns `None`. When auto-detection finds nothing the
run is fresh and the non-empty-`output_dir` guard applies (`overwrite_output_dir: true` bypasses it).

The pick is `resolve_resume_checkpoint()` (`src/checkpoint/parallel/resume_source.py`), which
`trainer.train(resume_from_checkpoint=True)` calls too, so a Python-API resume takes the same checkpoint (the
base Trainer would take the highest-numbered directory, complete or not). Every FS-aware save rank lists its
own `output_dir`; rank 0 merges and broadcasts the lists, and each candidate's completeness is one
all-reduce, so every rank picks, or raises over, the same checkpoint.

A resume then moves every incomplete step directory into `output_dir/_incomplete_checkpoints/` on each save
rank holding a copy, data kept (a taken name gets a `.1`, `.2` suffix). Left in place, a torn directory
numbered above the next saves would take a rotation slot: rotation orders by mtime but falls back to step
order on a mount whose mtimes it distrusts (fuse/S3), where `save_total_limit: 1` would delete the complete
checkpoints instead. Nothing moves when the resume raises. The `_` prefix keeps the directory out of the Hub
push and export copies; it counts as content for a later fresh run's `output_dir` guard.

### Configuration

The resume knobs are `resume_from_checkpoint` (`true` auto-detects, or a `checkpoint-N` path),
`save_strategy` / `save_steps` / `save_total_limit`, `save_only_model` (the warm-restart / exact-resume
switch below), `save_on_each_node` (forced by the output filesystem declaration),
`allow_optimizer_warm_restart` and `save_sharded_ep`
([Configuration Reference](configuration-reference.md)). All 12 training scripts under `scripts/training/`
support resume.

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

Exact resume saves per-rank optimizer shards via `torch.distributed.checkpoint.state_dict` — no gather to
rank 0 — and restores them with `set_optimizer_state_dict`, gated on the `optimizer_meta.pt` fingerprint:
**`world_size`, `pp_size`, `ep_size`, `cp_size`, `tp_size`, `expert_tp_size`, `ep_scope`,
`nvlink_domain_size`, `use_hsdp` (saved as `hsdp`), `use_grouped_gemm`, `fsdp_shard_ep1_experts`,
`expert_replica_size`, and the optimizer class**. Before the shard save, and before the FSDP2 weight load,
every FSDP2 module is resharded (`reshard_fsdp2_modules`): an eval-only forward (HF evaluates right before
the end-of-training save under `eval_strategy: steps`) leaves transient unsharded params that the torch APIs
cannot map to FQNs, and the save would hold no optimizer state.

**Multi-group EP deduplicates the replicated half.** The EP groups are DP replicas, so the FSDP-ignored
params (the whole EP module) and their moments are identical across an `expert_replica_group` (gradients
averaged over it, `AdamWBF16` rounding keyed by step and parameter name). Only the group's **lowest rank**
keeps them in its shard; its peers read them back from it — at Qwen3.5-397B-A17B on 512 GPUs at `ep64`, 8
copies become one (~14 TB → ~1.8 TB of EP optimizer state). The dedup needs a shared output filesystem; on a
per-node one every rank writes its own copy and the save warns once. `expert_replica_size` in the
fingerprint records how many shards collapsed into one (`1` for every dense run and every default `ep1` run).
Both reads of a resume (this rank's shard and, for a follower, the writer's) run under the node's
`max_concurrent_loading` throttle, which every rank enters
([Multi-Node → Model loading](../parallelism/multi-node.md#model-loading)).

Every restore verdict is rank-uniform — all ranks restore, warm-restart or raise together:

- **Fingerprint mismatch** → warm restart, with a warning naming each differing key (`saved='<absent>'` for
  a key the meta lacks). A meta written before the dedup lacks `expert_replica_size`; its shards hold every
  rank's own expert moments, so they restore under any replica size.
- **Shards absent on every rank** → warm restart, unless the directory still proves state was written
  (other ranks' shards, or a fingerprint-matched meta — a non-shared filesystem whose restart permuted the
  rank→node placement), which raises. **Shards on a subset** under a matching fingerprint raise (torn).
- **An interrupted save** raises: shards whose `optimizer_meta.pt` is missing or unreadable (the meta is
  written after every rank's shard), or the base Trainer's `optimizer.pt` with no shards beside it
  (`OptimizerShardStore.save` deletes it only once the shards are on disk).
- **A meta with no fingerprint at all** is refused outright in every mode, with no warm-restart fallback:
  nothing records the sharding that produced those layouts.
- **FQN drift** — a trainable param the shard neither holds state for nor lists in its `param_groups` —
  raises. A param it tracked without state (never received a gradient: GptOss sinks under a sink-dropping
  kernel) passes. A replica follower whose writer's shard restores none of the replicated FQNs raises too.
- **A restore that passes every gate and still fails** on some rank (an unreadable shard, a replica
  writer's shard out of reach, a CUDA OOM inside `set_optimizer_state_dict`, a moment shaped unlike its
  parameter) raises on every rank, naming the failing ranks and the first one's error.

`allow_optimizer_warm_restart: true` turns an interrupted save and a failed restore into a warm restart (rank
0 warns with the same summary); the other gates keep their outcomes. Deleting every `optimizer_shard_*.pt`
plus `optimizer_meta.pt` is the explicit warm-restart opt-in. Resuming a sharded-optimizer checkpoint in a
**non-sharded** mode (single GPU, DDP) raises: the base Trainer's loader knows only `optimizer.pt`/`.bin`.

Checkpoint **rotation** (`save_total_limit`) runs only after every toolkit sidecar is on disk, and never
after a failed save: the base Trainer would otherwise delete the oldest checkpoint before this save's
optimizer shards exist. Under `save_only_model: false` a checkpoint that cannot carry its optimizer state
**is** a failed save: `OptimizerShardStore.save` raises on every rank when any rank could not produce its
state (a world all-reduce), so the base's `optimizer.pt` stays and rotation does not fire. Known trigger: an
optimizer whose `state_dict` refuses the live sharding (FlashAdamW on unevenly-sharded DTensors).
`save_only_model: true` is the explicit opt-in for weights-only checkpoints.

#### Interrupted saves

`trainer_state.json` is what resume detection takes a checkpoint by, so it is published last:
`_save_checkpoint` writes it as `.trainer_state.json.uncommitted` and renames it, synced with its directory,
only once every other file is on disk and before rotation. Each FS-aware save rank does this for its own
copy, and a failure raises on every rank — a save rank with no uncommitted copy to rename included, since
rotation would otherwise remove the last checkpoint a resume takes. The base save's Hub push (per
`hub_strategy`) starts only after that commit; on a per-node output filesystem it uploads global rank 0's
node's copy, holding that node's optimizer shards only. Every strategy but `end` uploads the top level of
`output_dir` at each save (minus `_*` and `checkpoint-*`), `log/run.log` and a GRPO run's `completions/`
parquet (prompts included) among it; `checkpoint` and `all_checkpoints` also upload the checkpoint
directory, training state included; `end` pushes only from a final `save_model` or `push_to_hub` call,
which no training script makes.

A resumed run that saves a step it saved before writes into that step's existing directory, so every save
first removes the directory's completion markers (`trainer_state.json`, its uncommitted spelling,
`optimizer_meta.pt`, `resume_adapter.json`). A save stopped mid-weights, mid-sidecars or mid-shards (a
preemption, a full disk) thus leaves a step directory with no `trainer_state.json`, which
`resume_from_checkpoint: true` passes over, moving it out of rotation's sight with a warning; an explicit
path to it raises.

> [!NOTE]
> **accelerate launch**
>
> Under `accelerate launch` (DDP or accelerate-managed FSDP), `save_only_model: false` is always
> supported; optimizer, scheduler, and RNG states are restored by HF Trainer and accelerate's FSDP
> plugin.

## Multi-node

The save rank and resume detection's listing follow the output filesystem
([above](#checkpoints-and-training-resume)). On a shared filesystem the trainer forces `save_on_each_node`
off, since every node's local rank 0 would otherwise write and rotate the same files. On a non-shared one it
forces it on, all nodes must hold equivalent checkpoint dirs at the same path, and each node's local rank 0
creates the directory before a barrier.

**WandB run resumption:** the auto run ID is stamped per launch; export `WANDB_RUN_ID` (and
`WANDB_RESUME=allow`) to continue ([Environment variables](configuration-reference.md#environment-variables)).
A gathered save holds non-writers inside its per-layer expert gathers while the writer streams each flush to
disk, so for 100B+ models size `DIST_NCCL_TIMEOUT_MINUTES` to checkpoint size ÷ filesystem write speed, with
margin.

A write that fails on any rank — ENOSPC, EIO, a stale NFS handle — stops the whole job at the next collective
with that rank's real error. `DeferredRankFailure` (`src/runtime/consensus.py`) records the local failure,
lets every rank finish the collectives, then raises everywhere.

## Troubleshooting

Checkpoint-specific; the rest of the toolkit's symptoms are in [Troubleshooting](troubleshooting.md).

- **`missing keys: ['lm_head.weight']` on resume:** expected for `tie_word_embeddings=true` models
  (Qwen3); `lm_head.weight` shares storage with the embedding.
- **`resume_from_checkpoint: true` but starts at step 0:** `output_dir` holds no `checkpoint-N`
  directory, so the run warns `no checkpoint found` and starts fresh; check the path.
- **TP resume shape mismatch or hang:** the load needs a device mesh whose `mesh_dim_names` include
  `"tp"` (a 2-D `(dp, tp)` mesh resolves to its `tp` sub-mesh), and all ranks must reach the same checkpoint
  dir. The weights are gathered, so only the optimizer shards need the saving run's `tp_size`.
- **OOM saving TP gathered:** each `full_tensor()` reconstructs one whole matrix on the save rank, so a huge
  embedding can exceed host RAM even though the write streams. Lower `save_max_shard_size`, reduce
  `tp_size`, or give the node more host RAM.
- **A re-saved VLM fails to reload as a VLM:** the save kept only the tokenizer, dropping
  `preprocessor_config.json` / `processor_config.json` / `chat_template`. The tools save the full
  processor through `load_processing_class` (`src/model_loading/tokenizer_setup.py`); an adapter from a
  `text_only_model` run is the exception, merged or converted with the run's tokenizer and no processor
  files, which `reattach_vision_tower.py` restores for serving.
- **An export serves different routing or sinks than training:** the router-balancing biases
  (`router_balancing_biases.pt`, the only copy for families with no native slot) and the GptOss sink policy
  (`training_provenance.json`) are not in the weights. Export through `merge_peft_adapters.py` or
  `convert_to_bf16.py`, which re-apply them via `apply_training_sidecars`, or near-tied top-k picks flip
  against training with no error.

## Related pages

[Trainer Architecture](trainer-architecture.md) · [Expert Parallelism](../parallelism/expert-parallelism.md) ·
[Tensor Parallelism](../parallelism/tensor-parallelism.md) · [Context Parallelism](../parallelism/context-parallelism.md) ·
[Pipeline Parallelism](../parallelism/pipeline-parallelism.md) · [Multi-Node](../parallelism/multi-node.md)
