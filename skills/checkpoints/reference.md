# checkpoints — reference

Per-script detail, the format matrix, resume internals, and the footguns. Grounded in the code
(verify against the cited files before acting — they are the source of truth).

## Checkpoint format matrix

Four on-disk layouts, distinguished by `metadata.format` in `model.safetensors.index.json`, NOT by
filename (`agent-docs/reference/checkpoints.md`):

| Layout | Files | `metadata.format` | Loadable directly? | Merge tool |
|---|---|---|---|---|
| Single file | `model.safetensors` | *(no index)* | yes | none |
| Gathered HF-sharded | `model-0000N-of-…` + index | *absent* | yes | none |
| PP per-stage | `model-pp0000N-of-…` + merged index | *absent* | yes (a per-node save: no) | `merge_pp_stages.py` for per-node directories |
| EP per-rank sharded | `model-…` + `.shard_{rank}` keys + index | `ep_sharded` | no | `merge_ep_shards.py` |

Gathered is the default for every mode; the sharded EP save is an opt-in write-bandwidth option
(there is no per-rank TP save). Nothing loads `ep_sharded` directly: `reject_ep_sharded_checkpoint`
raises world-uniformly at EP load (`src/model_loading/parallel/ep.py`) and
`load_full_state_dict` refuses the marker with the merge instructions
(`src/checkpoint/reading.py`); a shard directory whose index never landed is caught tool-side by the
`.shard_N` header peek (`_reject_indexless_ep_shards`, the same module). `merge_ep_shards`
additionally needs `ep_size` in the index metadata, not just the marker.

## Resume — three paths

- **Path A — reload from checkpoint** (only a `use_grouped_gemm: false` run with no EP/ETP/CP/TP):
  `model_name_or_path` stays the weights source. Under FSDP2 `load_resume_weights` reshards, reads the whole
  dict on rank 0 via `load_full_state_dict()` and hands it to `set_model_state_dict(broadcast_from_rank0)`,
  gated on key coverage; a single process or DDP streams it per rank. Both map the
  checkpoint onto the live names first (`ModuleLayoutView`, the inverse of the save-side revert), so a
  per-expert expert checkpoint reaches the fused live experts. Accelerate-managed FSDP and the
  sentence-transformers pipeline keep the base Trainer's loader.
- **Path B — load at construction** (EP, ETP, EP+TP, EP+CP, CP, TP, and at the default
  `use_grouped_gemm: true` every other run, dense included): `resolve_resume_weights_source`
  (`src/checkpoint/parallel/resume_source.py`) repoints the policy's weights source at the checkpoint dir (an
  adapter-only one keeps the base, and so does a merged one — `merge_expert_lora_on_save` or
  embedding LoRA — whose `resume_adapter.json` marker sends the resume to its `resume_adapter/`), so
  `load_distributed_model()` builds the model from the trained
  weights. `model_config` is not mutated — the DPO/KTO/SDPG reference and the dataset-compat check
  keep the base, so **leave `model_name_or_path` at the base**. An unmerged per-rank save raises
  here, before construction. The loader then skips the re-read: EP/CP always, FSDP2 / TP / single-process when the model was
  constructed from that checkpoint. A model built from anything else makes the loader raise
  (rank-0 verdict, broadcast) under EP/CP when the checkpoint ships base weights (a marked
  merge-on-save checkpoint inverts this: it refuses a model built from itself), and under TP+DP,
  whose strided dp-over-tp placement `distribute_tensor` does not invert; pure TP instead streams the
  checkpoint one tensor at a time per rank and `distribute_tensor`s each into the live placements
  (GptOss sinks sliced by `tp_rank`). `load_best_model_at_end` raises for a full fine-tune under CP,
  on a MoE carrying EP or grouped-GEMM wrappers (`ep_size: 1` included), and under TP+DP; dense
  FSDP2 and pure TP re-read for it.
- **Path C — PP** (`restore_pp_stage_weights`, dispatched first): stage-local restore.

All three restore `trainer_state.json`, the LR scheduler from `scheduler.pt` (written even under
`save_only_model`), LoRA adapters (`restore_adapters`, `src/checkpoint/parallel/adapters.py`),
the router-balancing biases
(`restore_balancing_biases`, `src/checkpoint/parallel/sidecars.py`) and the frozen reference
scores in `reference_logps.pt` (DPO/KTO precompute, offline GRPO KL; `src/trainers/reference_logps.py`,
which validates token digests and settings before attaching them). Optimizer state resumes from the per-rank shards when
`OptimizerStateFingerprint` matches; a mismatch warm-restarts, and
shards whose `optimizer_meta.pt` carries no fingerprint at all raise — delete every
`optimizer_shard_*.pt` + `optimizer_meta.pt` to accept a warm restart. A matched restore that fails
on any rank (an unreadable shard, a CUDA OOM) raises on every rank unless `allow_optimizer_warm_restart: true`.

Source: `src/checkpoint/parallel/resume_weights.py` (`load_resume_weights`),
`src/checkpoint/parallel/pp.py` (`restore_pp_stage_weights`), `src/checkpoint/parallel/optimizer.py`
(`OptimizerShardStore.load` / `.save`), `src/checkpoint/parallel/optimizer_meta.py`,
`src/checkpoint/parallel/sidecars.py` (`restore_lr_scheduler`), `src/checkpoint/reading.py`
(`load_full_state_dict`), `src/checkpoint/hub_layout.py` (`ModuleLayoutView`).
`src/trainers/checkpointing.py::_load_from_checkpoint` runs the base Trainer's load (`_load_weights_through_base`)
where `load_resume_weights` declines; `_weights_load_context` is the context the weight resume reads.

## Per-script reference (`scripts/after_training/`)

Read the script's argparse to confirm flags before running — these are the load-bearing ones.

### `merge_ep_shards.py`

Merge a `save_sharded_ep` checkpoint (per-rank `.shard_N` expert keys) into HF format; converts
EP-internal layouts back to each family's HF layout. Flags: `--input_dir`, `--output_dir`, `--quiet`,
`--max_shard_size` (`5GB`), `--delete_input_shards`. The merge casts nothing: the sharded writer
already applied `save_dtype_caster` (`src/checkpoint/writing.py` — BF16 except the
module-tree keep-sets: norms, balancing tensors, the family's fp32 pins; a training checkpoint keeps
every tensor at its live dtype), so the stored dtype is what the merge writes. The index metadata is HF's own — there is no `merged_from_*` marker.

- **Family support is the spec's, not a table.** `find_moe_family_spec` maps a checkpoint's
  `model_type` to its `MoEFamilySpec` (`src/models/moe/family_specs.py`, which also owns
  `expert_weight_roots`); the merge writes the spec's `expert_layout` through `to_hub_expert_tensors`
  (`src/models/moe/expert_layout.py`), the transform the gathered save applies, then `spec.to_hub_key`.
  Construction and save time gate on
  `reject_unmergeable_ep_sharded_save` (`src/checkpoint/parallel/ep.py`), which among its refusals
  covers two family cases: a `model_type` no family spec claims, and a model whose load converted
  names the merge cannot respell (the gathered save reverts them through transformers' save-side
  conversion, which a key-by-key merge stream cannot apply) — Step-3.7 Flash, Inkling, DeepSeek-V4 and GLM-5 Next hubs, a SigLIP tower
  (Command A+, LFM2-VL), a legacy Mistral 3 checkpoint. Laguna's renames are the merge's own
  (`MoEFamilySpec.to_hub_key`), so it merges, as do the families whose loads convert only the experts.
- GptOss gate/up are de-interleaved for grouped-GEMM training and re-interleaved on merge; the expert
  key set (incl. the 2-D `gate_up_proj_bias`/`down_proj_bias`) is derived from `expert_weight_roots()`
  in `src/models/moe/family_specs.py`, not hand-listed.

### `merge_peft_adapters.py`

The one adapter-merge tool: load base + adapter, `merge_and_unload()`, save standalone HF checkpoint
(base path read from `adapter_config.json`). Flags: `--adapter_dir`, `--output_dir`,
`--task {causal_lm,classification}`, `--dtype` (`bfloat16`), `--device_map` (`auto`/`cpu` for big
models), `--num_labels`, `--max_shard_size` (`5GB`), `--attn_implementation`, `--fp32_norms` (every
normalization leaf written fp32, `convert_to_bf16.py`'s layout), plus the shared
`--trust_remote_code` / `--quiet`. The base loads through the class the adapter's keys address
(`load_base_for_adapter`): the widest one (`resolve_auto_model_class`, the full
`*ForImageTextToText` wrapper for a VLM), or the `*ForCausalLM` sibling when the adapter came from a
`text_only_model` run, whose merge keeps the text-only `model.*` layout. The work happens in the
shared `merge_adapter_into_base` (`src/checkpoint/adapter_merge.py`), which also refuses a
per-rank-sharded adapter or base, an in-place output and a native expert-LoRA adapter, and picks the
processing class through `resolve_peft_processing_class` (`src/model_loading/tokenizer_setup.py`):
for a text-only merge the run's tokenizer from the adapter dir (the base's only when the adapter
saved none) and no processor files; otherwise the adapter dir's when it is a full processor, else the
**base's** — a tokenizer-only save leaves an unloadable VLM. It applies `apply_training_sidecars` to the
merged model (see below) — the merge rebuilds the base from the hub, so without it a `bias_update`
run's routing and a `reset_sinks` run's sinks are lost.

### `convert_to_bf16.py`

Re-save a model or a PEFT adapter in BF16 (merging an adapter is `merge_peft_adapters.py`'s). Flags:
`--input_dir`, `--output_dir`, `--model_type {causal_lm,classifier,base}`, `--peft`,
`--device_map`, `--verify`, `--check_inference`, `--max_shard_size`, `--trust_remote_code` /
`--no-trust_remote_code` (default **True** — a local checkpoint source; Bailing/Ling need remote code.
`patch_vocab.py`, `convert_deepseek_v4_bf16.py`, `reattach_vision_tower.py` and `reset_sinks.py`
read a Hub-capable `--model_id`, so those default **off**).
**Forces every normalization leaf (torch norm bases, or a class named `*Norm*`) back to fp32** (BF16
body + fp32 norms) — as `merge_peft_adapters.py --fp32_norms` does; without it the merge leaves norms
at the `--dtype` it loads the base at, except modules a family pins through transformers'
`_keep_in_fp32_modules_strict` (e.g. DeepSeek-V4's norms and HC modules), which load fp32 under a
bf16 or fp16 `--dtype`. GPT-OSS declares its norms in the non-strict `_keep_in_fp32_modules`, which
transformers keeps fp32 only under `--dtype float16`. `merge_ep_shards.py` keeps norms at their
trained dtype. Applies `apply_training_sidecars` to the loaded model (see below).

### `quantize_to_lowp.py`

Post-training quantize bf16/fp32 → block-scaled **mxfp8 / mxfp4 / nvfp4** (compressed-tensors triples +
a manifest). Pairs with the QAT training path (`lowp_precision: …`). Flags: `--input_dir`, `--output_dir`,
`--format {mxfp8,mxfp4,nvfp4}`, `--contraction_axis` (default -1; **2-D `*.weight` matrices only** —
a fused 3-D expert tensor uses the `fused_contraction_axis` its family spec declares and
ignores the flag, and an unresolvable family raises rather than guessing), `--include`/`--exclude`
(regex), `--verify`, and the four training-scope flags the checkpoint does not carry:
`--lowp_apply_dense_mlp` / `--lowp_apply_moe_experts` (both default on) and
`--lowp_keep_first_blocks` / `--lowp_keep_last_blocks` (both `0`) — same names and defaults as the
`ParallelismConfig` knobs, so a training config transfers verbatim. Quantizes the dense MLP projections
**and** the declared expert weights, fused 3-D banks included (`...experts.gate_up_proj` / `down_proj`, no `.weight`
suffix — the largest tensors in gpt-oss/Qwen3.5/GLM4/LFM2) that match include∧¬exclude. For a sharded input it rebuilds `model.safetensors.index.json` and copies every
non-weight file (config, tokenizer, `chat_template.jinja`, remote-code `.py`). An **export tool, not a
speedup** — bf16 stays optimal at these shapes.

### `reset_sinks.py`

Set every `*.sinks` param to dtype-min (neutralize the attention sink), matching the GptOss FA2-finetune
behavior. Flags: `--model_id` (required; local dir or HF repo id, no `--revision`), `--output_dir`
(required unless `--in_place` or `--dry_run`), `--in_place` (rewrites the `--model_id` directory, no undo — never
valid for a repo id), `--dry_run`, plus the shared `--max_shard_size` / `--trust_remote_code`
(default **off** — a Hub-capable source). Direct safetensors edit when `model.safetensors`
exists, else `from_pretrained` + `save_pretrained` for sharded checkpoints.

### `unfuse_moe_experts.py`

Rewrite a gathered checkpoint's fused expert tensors (`experts.gate_up_proj` `[E, 2I, H]` +
`experts.down_proj`) into the per-expert hub keys `experts.{i}.{gate,up,down}_proj.weight` for
engines that only read those (vLLM's `glm4_moe_lite`). Flags: `--input_dir`, `--output_dir`, plus
`--max_shard_size`. Weights are unchanged — key layout and the gate/up split only. Projection names
come from the family's spec (`MoEFamilySpec.expert_layout.per_expert_keys`) —
`gate_proj`/`up_proj`/`down_proj` for most families, `w1`/`w3`/`w2` for LFM-2 and DeepSeek-V4. A
family whose spec declares no per-expert keys is **refused**: its checkpoints never store one
tensor per expert (fused-native, or already interleaved as engines expect), so every name emitted
would be one nothing reads. A checkpoint with no fused expert keys is copied through.

### `merge_models.py`

Merge several fine-tuned checkpoints **of the same architecture** into one — no mergekit dependency.
Methods (`--method`): `linear` (weighted average, normalized by the weight sum), `slerp` (exactly two
models, `--t`), `task_arithmetic` (`--base_model` + Σ wᵢ·task vector), `ties` (`--density`,
`--lambda`, arXiv:2306.01708). Flags: `--models path[:weight] …` (local dirs only), `--output_dir`,
`--base_model`, `--dtype` (`bfloat16`), `--tokenizer_source` (config/tokenizer/remote-code source;
default base or first model, a Hub id is downloaded weights-excluded), `--max_shard_size`,
`--quiet`, `--allow_missing_tokenizer`, `--trust_remote_code`. Streams one tensor at a time across the inputs, so peak host memory scales with the largest
tensor, not N models (knob ranges and per-method working set: `agent-docs/reference/model-merging.md`).
Deliberately copies **no** resume sidecars (`trainer_state.json`, `rng_state*`, `scheduler.pt`,
`router_balancing_biases.pt`, `reference_logps.pt`, `prefetch_pending-*`, `resume_adapter/` and its
marker) — they describe one run, not the merge. Inputs must share the reference key set at one shape
(checked from the headers before any write), except a Hub base's keys the shipped model never reads:
a stored tied head, and the keys its class drops at load (`_keys_to_ignore_on_load_unexpected`, e.g.
Qwen3.5/3.6's `mtp.*`).

### `reattach_vision_tower.py`

Rebuild the multimodal wrapper layout around a `text_only_model` export: text weights re-prefixed to
`model.language_model.*`, `model.visual.*` / `mtp.*` streamed back from the base, the composite
config regrafted with the trained text config. Flags: `--input_dir` (the text-only export),
`--model_id` + `--revision` (the multimodal base — Hub id or local dir), `--output_dir`,
`--max_shard_size`, `--trust_remote_code` (default **off** — a Hub-capable source). Refuses a
wrapper-layout input and a base with no vision keys. Both pinned engines (vLLM 0.32.0, SGLang
0.5.22) load a text-only export as-is; this restores the multimodal artifact.

## Training sidecars — `apply_training_sidecars`

`router_balancing_biases.pt` (trained router-balancing biases, written by
`save_balancing_biases` in `src/checkpoint/parallel/sidecars.py` every save) and `training_provenance.json` (the GptOss sink
policy) hold training state the weights cannot. `apply_training_sidecars(model, source_dir)`
(`src/checkpoint/sidecars.py`) is the single seam that re-applies both to an assembled model:
it restores the balancing tensors **at their trained precision** (a bf16 round trip quantizes away
the ~1e-3 sign steps), then, only where the source ships no weights of its own (a PEFT adapter
dir), fills the native slots from the sidecar via `apply_balancing_bias_sidecar`
(`src/models/moe/balancing.py`, off the family spec's `balancing_slot`) and **warns**
about routers whose bias was transient — those exports serve without it. `merge_peft_adapters.py`
and `convert_to_bf16.py` call it and print the returned actions; `copy_training_sidecars` moves
`EXPORT_SIDECAR_FILES` into an output dir a full save did not write them to.

## Footguns (source-cited)

- **Every writer reads resharded params** — `save_model`
  (`src/trainers/checkpointing.py`) calls `reshard_fsdp2_modules` on the top-level model
  before it touches a tensor. A mid-training save can land while the FSDP2 modules still hold the
  transient UNSHARDED params a forward registered — objects the optimizer never stepped — so every
  saver below would write the wrong tensors, and the PEFT path additionally **mis-routes**: its
  DTensor probe sees plain tensors, takes the rank-0 `save_pretrained` branch, and PEFT's
  shared-tensor scan raises on the first param that branch cannot address. Any new save path must
  reshard first.
- **GptOss EP-save biases** — the `INTERLEAVED_GLU` expert layout (`src/models/moe/expert_layout.py`)
  carries the 2-D `gate_up_proj_bias` / `down_proj_bias` alongside the 3-D expert weights, so the
  gathered checkpoint reloads cleanly.
- **CP-only state_dict drops dense MLP only on sparse layers** — `UlyssesModel.state_dict`
  (`src/distributed/context_parallel/ulysses_model.py`) filters duplicate dense `.mlp.{gate,up,down}_proj`
  keys per layer, gated on layers that actually carry routed experts. A model-global filter would drop
  the genuinely-dense early layers (GLM-4 MoE Lite, Mistral4 `first_k_dense_replace`) → a corrupt CP-only
  checkpoint.
- **TP attention head divisibility** — `validate_tp_head_divisibility` (`parallelize_attention.py`)
  raises from `config.json` before any weight loads, and again before sharding, when
  `num_attention_heads` (or non-MLA GQA `num_key_value_heads`) is not divisible by `tp_size`;
  `ColwiseParallel` would otherwise split Q/K/V inside a head and silently corrupt attention. MLA
  (GLM4 latent attn) shards by query head, so the KV-head check is skipped there.
- **Zaya has no native serving path on either engine** — vLLM 0.32.0 ships no native Zaya class (an
  export resolves only through vLLM's transformers backend, generation quality unverified here), and
  SGLang 0.5.22's loader reads the pre-transformers-5.14 per-expert layout, not the native fused one
  the trainer holds. RL weight sync is refused at construction for both, off the engine client's
  `UNSERVABLE_MODEL_TYPES` (`src/weight_sync/clients/`) — **not** off the family spec, whose
  `weight_sync_refusal` covers glm5_next, inkling and cohere2_moe instead. Zaya's gathered
  save is the native fused layout (`experts.gate_up_proj` / `down_proj`), which `from_pretrained`
  reads back directly.
- **VLM re-save must persist the processor** — `merge_peft_adapters` / `convert_to_bf16` /
  `scripts/before_training/patch_vocab.py`
  load the VLM class correctly and save the full **processor**, not just the tokenizer; saving only a
  tokenizer drops `preprocessor_config.json` / `processor_config.json` / `chat_template` and the result
  fails to reload as a VLM. The exception is an adapter from a `text_only_model` run: merged or
  converted, it ships the run's tokenizer and no processor files, since the class it goes through has
  no vision path (`reattach_vision_tower.py` restores the processor files for serving). The shared `load_processing_class`
  (`src/model_loading/tokenizer_setup.py`) returns the processor for multimodal models and a
  plain tokenizer otherwise; all three scripts call it, and `resolve_auto_model_class` /
  `auto_load_model` (`src/model_loading/model_preparation.py`) load a multimodal config as the
  full `*ForImageTextToText` wrapper.
- **Save retains state on the save-rank only** — `save_ep_gathered_checkpoint`
  (`src/checkpoint/parallel/ep.py`) has all ranks run each gather collective but only the
  save rank keeps the tensor (per-node CPU RAM peaks well under 1× model, not gpus_per_node×). TP
  mirrors this, and on the same `is_fs_save_rank()` — never `tp_rank == 0`, which the several TP
  groups a node holds would race on identical files. Don't "optimize" by skipping ranks in the
  collective. The save rank is
  `is_fs_save_rank()` (`src/runtime/filesystem.py`): on a **shared FS** only global rank 0
  writes the shards + index (avoids the NFS write race of many ranks touching one path); on a
  **non-shared FS** (`DIST_SHARED_FILESYSTEM=0`) each node's local rank 0 writes its own copy to
  node-local storage.
