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
| PP per-stage *(not producible — PP is not yet available in this release)* | `model-pp0000N-of-…` + merged index | *absent* | yes | none |
| EP per-rank sharded | `model-…` + `.shard_{rank}` keys + index | `ep_sharded` | no | `merge_ep_shards.py` |

Gathered is the default for every mode; the sharded EP save is an opt-in write-bandwidth option
(there is no per-rank TP save). Nothing loads `ep_sharded` directly: `reject_ep_sharded_checkpoint`
raises world-uniformly at EP load (`src/distributed/expert_parallel/loading.py`) and
`load_full_state_dict` refuses the marker with the merge instructions
(`src/checkpoint/format.py`); a shard directory whose index never landed is caught by a `.shard_N`
header peek.

## Resume — three paths

- **Path A — reload from checkpoint** (TP-only, dense FSDP2/DDP): weights come from the checkpoint.
  TP (`_load_tp`) streams the checkpoint one tensor at a time per rank and `distribute_tensor`s each
  into the live DTensor's placements before `copy_` (hand-sharded non-DTensor params — GptOss sinks —
  are sliced by `tp_rank`); FSDP2 (`_load_fsdp2`) reads the whole dict on rank 0 via
  `load_full_state_dict()` and hands it to `set_model_state_dict(broadcast_from_rank0)`. Both gate on
  key coverage across ranks and restore weights + trainer state.
- **Path B — skip reload** (EP, ETP, EP+TP, EP+CP, CP): model rebuilt by `load_distributed_model()` at
  `__init__`, so **weights are not reloaded**. **Set `model_name_or_path` to the gathered checkpoint
  dir** or you resume the original base weights.
- **Path C — PP** (`_load_pp_stage`, dispatched first): stage-local restore. Unreachable while PP is
  unavailable in this release.

All three restore `trainer_state.json`, the LR scheduler from `scheduler.pt` (written even under
`save_only_model`), LoRA adapters (`restore_adapters`) and wrapper-level trained params
(`_restore_extra_trained_params`). Optimizer state resumes from the per-rank shards when
`OptimizerStateFingerprint` matches; a mismatch warm-restarts (under PP it raises instead), and
shards whose `optimizer_meta.pt` carries no fingerprint at all raise — delete every
`optimizer_shard_*.pt` + `optimizer_meta.pt` to accept a warm restart.

Source: `src/distributed/checkpoint/loader.py` (`CheckpointLoader`, `_load_tp`, `_load_fsdp2`,
`_load_pp_stage`), `src/distributed/checkpoint/optimizer.py` (`OptimizerShardStore.load` / `.save` /
`.restore_lr_scheduler`), `src/distributed/checkpoint/fingerprint.py`,
`src/checkpoint/format.py` (`load_full_state_dict`).
`src/trainers/mixins/checkpointing.py::_load_from_checkpoint` is a thin delegate over the loader.

## Per-script reference (`scripts/after_training/`)

Read the script's argparse to confirm flags before running — these are the load-bearing ones.

### `merge_ep_shards.py`
Merge a `save_sharded_ep` checkpoint (per-rank `.shard_N` expert keys) into HF format; converts
EP-internal layouts back to each family's HF layout. Flags: `--input_dir`, `--output_dir`, `--quiet`,
`--max_shard_size` (`5GB`), `--delete_input_shards`. The merge casts nothing: the sharded writer
already applied `save_dtype_caster` (`src/checkpoint/format.py` — BF16 except the
module-tree keep-sets: norms, balancing tensors, the family's fp32 pins), so the stored dtype is the
export dtype. The index metadata is HF's own — there is no `merged_from_*` marker.
- **Family support is class-owned, not a table.** `resolve_ep_merge_layer_class` /
  `supported_ep_merge_model_types` (`expert_weights.py`, which also owns `expert_weight_roots` and
  `to_hub_layer_key`) map a checkpoint's `model_type` to the EP layer class via each class's
  `HF_MODEL_TYPES`; the transform itself is that class's `merge_shards_to_hf` (`expert_gather.py`,
  overridden where the layout differs). Save time gates on `_check_ep_merge_family_supported`
  (`saving.py`), which refuses two cases: a `model_type` no EP layer class claims, and a family
  declaring `_EXPORTS_HUB_NAMESPACE` — Step-3.7 Flash, whose hub spelling comes from transformers'
  save-side conversion revert that a key-by-key merge stream cannot apply. Every other shipped
  family merges, mistral4, gemma4 and Zaya included.
- GptOss gate/up are de-interleaved for grouped-GEMM training and re-interleaved on merge; the expert
  key set (incl. the 2-D `gate_up_proj_bias`/`down_proj_bias`) is derived from `expert_weight_roots()`
  in `src/distributed/expert_parallel/expert_weights.py`, not hand-listed.

### `merge_peft_adapters.py`
Load base + adapter, `merge_and_unload()`, save standalone HF checkpoint (base path read from
`adapter_config.json`). Flags: `--adapter_dir`, `--output_dir`,
`--task {causal_lm,classification}`, `--dtype {bf16,fp16,fp32}`, `--device_map` (`auto`/`cpu` for big
models), `--num_labels`, `--max_shard_size` (`5GB`), `--attn_implementation`, plus the shared
`--trust_remote_code` / `--quiet`. Uses
`resolve_auto_model_class`, so **VLM bases load as the full `*ForImageTextToText` wrapper** (avoids
adapter-key mismatch), and saves via `load_processing_class` so a VLM keeps its image preprocessor +
chat template (adapter dir first, base model as fallback). Applies `apply_training_sidecars` to the
merged model (see below) — the merge rebuilds the base from the hub, so without it a `bias_update`
run's routing and a `reset_sinks` run's sinks are lost.

### `convert_to_bf16.py`
Re-save a model in BF16, optionally merging a PEFT adapter in the same pass. Flags: `--input_dir`,
`--output_dir`, `--model_type {causal_lm,classifier,base}`, `--peft`, `--merge_adapter`,
`--device_map`, `--verify`, `--check_inference`, `--max_shard_size`, `--trust_remote_code` /
`--no-trust_remote_code` (default **True** — a local checkpoint source; Bailing/Ling need remote code.
`scripts/before_training/patch_vocab.py` and `scripts/before_training/convert_deepseek_v4_bf16.py`
read a Hub-capable `--model_id`, so those default **off**).
**Forces `LayerNorm` / any `*norm*` module back to fp32** (BF16 body + fp32 norms), matching both
merge scripts. Applies `apply_training_sidecars` to the loaded model (see below).

### `quantize_to_lowp.py`
Post-training quantize bf16/fp32 → block-scaled **mxfp8 / mxfp4 / nvfp4** (compressed-tensors triples +
a manifest). Pairs with the QAT training path (`lowp_precision: …`). Flags: `--input_dir`, `--output_dir`,
`--format {mxfp8,mxfp4,nvfp4}`, `--contraction_axis` (default -1; **2-D `*.weight` matrices only** —
a fused 3-D expert tensor uses the `HF_FUSED_EXPERT_CONTRACTION_AXIS` its EP layer class declares and
ignores the flag, and an unresolvable family raises rather than guessing), `--include`/`--exclude`
(regex), `--verify`, and the four training-scope flags the checkpoint does not carry:
`--lowp_apply_dense_mlp` / `--lowp_apply_moe_experts` (both default on) and
`--lowp_keep_first_blocks` / `--lowp_keep_last_blocks` (both `0`) — same names and defaults as the
`ParallelismConfig` knobs, so a training config transfers verbatim. Quantizes any `*.weight` matrix
**and** the fused 3-D MoE expert tensors (`...experts.gate_up_proj` / `down_proj`, no `.weight`
suffix — the largest tensors in gpt-oss/Qwen3.5/GLM4/LFM2) that match include∧¬exclude. For a sharded input it rebuilds `model.safetensors.index.json` and copies every
non-weight file (config, tokenizer, `chat_template.jinja`, remote-code `.py`). An **export tool, not a
speedup** — bf16 stays optimal at these shapes.

### `reset_sinks.py`
Set every `*.sinks` param to dtype-min (neutralize the attention sink), matching the GptOss FA2-finetune
behavior. Flags: `--input_dir` (local dir or HF repo id), `--output_dir` (**required** unless
`--in_place`), `--dry_run`, plus the shared `--max_shard_size` / `--trust_remote_code`. Direct
safetensors edit when `model.safetensors`
exists, else `from_pretrained` + `save_pretrained` for sharded checkpoints.

### `unfuse_moe_experts.py`
Rewrite a gathered checkpoint's fused expert tensors (`experts.gate_up_proj` `[E, 2I, H]` +
`experts.down_proj`) into the legacy per-expert keys `experts.{i}.{gate,up,down}_proj.weight` for
engines that only read those (vLLM's `glm4_moe_lite`). Flags: `--input_dir`, `--output_dir`, plus
`--max_shard_size`. Weights are unchanged — key layout and the gate/up split only. Projection names
come from the family's EP layer class (`EPMoELayerBase.hub_per_expert_keys`) —
`gate_proj`/`up_proj`/`down_proj` for most families, `w1`/`w3`/`w2` for LFM-2 and DeepSeek-V4. A
family whose class declares no per-expert keys is **refused**: its checkpoints never store one
tensor per expert (fused-native, or already interleaved as engines expect), so every name emitted
would be one nothing reads. A checkpoint with no fused expert keys is copied through.

### `merge_models.py`
Merge several fine-tuned checkpoints **of the same architecture** into one — no mergekit dependency.
Methods (`--method`): `linear` (weighted average, normalized by the weight sum), `slerp` (exactly two
models, `--t`), `task_arithmetic` (`--base_model` + Σ wᵢ·task vector), `ties` (`--density`,
`--lambda`, arXiv:2306.01708). Flags: `--models path[:weight] …` (local dirs only), `--output_dir`,
`--base_model`, `--dtype` (`bfloat16`), `--tokenizer_source` (config/tokenizer/remote-code source;
default base or first model, a Hub id is downloaded weights-excluded), `--max_shard_size`,
`--quiet`, `--allow_missing_tokenizer`, `--trust_remote_code`. Streams one tensor at a time across the inputs, so peak extra memory is a layer, not N
models. Deliberately copies **no** resume sidecars (`rng_state*`, `scheduler.pt`,
`router_balancing_biases.pt`) — they describe one run, not the merge.

## Training sidecars — `apply_training_sidecars`

`router_balancing_biases.pt` (trained router-balancing biases, written by
`_persist_router_balancing_biases` every save) and `training_provenance.json` (the GptOss sink
policy) hold training state the weights cannot. `apply_training_sidecars(model, source_dir)`
(`src/checkpoint/tool_io.py`) is the single seam that re-applies both to an assembled model:
it restores the balancing tensors **at their trained precision** (a bf16 round trip quantizes away
the ~1e-3 sign steps), then, only where the source ships no weights of its own (a PEFT adapter
dir), fills the native slots from the sidecar via `apply_router_balancing_sidecar`
(`src/models/moe_balancing.py`, off the slot registry the EP layer classes register into) and **warns**
about routers whose bias was transient — those exports serve without it. `merge_peft_adapters.py`
and `convert_to_bf16.py` call it and print the returned actions; `copy_training_sidecars` moves
`TRAINING_STATE_FILES` into an output dir a full save did not write them to.

## Footguns (source-cited)

- **Every writer reads resharded params** — `save_model`
  (`src/trainers/mixins/checkpointing.py`) calls `reshard_fsdp2_modules` on the top-level model
  before it touches a tensor. A mid-training save can land while the FSDP2 modules still hold the
  transient UNSHARDED params a forward registered — objects the optimizer never stepped — so every
  saver below would write the wrong tensors, and the PEFT path additionally **mis-routes**: its
  DTensor probe sees plain tensors, takes the rank-0 `save_pretrained` branch, and PEFT's
  shared-tensor scan raises on the first param that branch cannot address. Any new save path must
  reshard first.
- **GptOss EP-save biases** — `EPGptOssMoELayer.gather_expert_state_dict`
  (`src/distributed/expert_parallel/layers/gpt_oss.py`) carries the 2-D `gate_up_proj_bias` /
  `down_proj_bias` alongside the 3-D expert weights, so the gathered checkpoint reloads cleanly. GptOss
  EP checkpoints produced by a `param.dim()==3`-only filter (no 2-D biases) must be patched from the
  source biases before reload.
- **CP-only state_dict drops dense MLP only on sparse layers** — `UlyssesCPModelWrapper.state_dict`
  (`src/distributed/context_parallel/wrapper.py`) filters duplicate dense `.mlp.{gate,up,down}_proj`
  keys per layer, gated on layers that actually carry routed experts. A model-global filter would drop
  the genuinely-dense early layers (GLM-4 MoE Lite, Mistral4 `first_k_dense_replace`) → a corrupt CP-only
  checkpoint, masked on resume (CP reloads weights from `model_name_or_path`).
- **TP attention head divisibility** — `parallelize_attention.py` raises before sharding when
  `num_attention_heads` (or non-MLA GQA `num_key_value_heads`) is not divisible by `tp_size`;
  `ColwiseParallel` would otherwise split Q/K/V inside a head and silently corrupt attention. MLA
  (GLM4 latent attn) shards by query head, so the KV-head check is skipped there.
- **Zaya has no native vLLM path** — vLLM 0.26.0 ships no native Zaya implementation; an export
  resolves only through vLLM's transformers backend (generation quality unverified here), and RL
  weight sync is refused at construction for that gap. Its gathered save is the native fused layout
  (`experts.gate_up_proj` / `down_proj`), which `from_pretrained` reads back directly.
- **VLM re-save must persist the processor** — `merge_peft_adapters` / `convert_to_bf16` /
  `scripts/before_training/patch_vocab.py`
  load the VLM class correctly and save the full **processor**, not just the tokenizer; saving only a
  tokenizer drops `preprocessor_config.json` / `processor_config.json` / `chat_template` and the result
  fails to reload as a VLM. The shared `load_processing_class`
  (`src/models/loading/tokenizer_setup.py`) returns the processor for multimodal models and a
  plain tokenizer otherwise; all three scripts call it, and `resolve_auto_model_class` /
  `auto_load_model` (`src/models/loading/model_preparation.py`) load a multimodal config as the
  full `*ForImageTextToText` wrapper.
- **Save retains state on the save-rank only** — `save_ep_model`
  (`src/distributed/expert_parallel/saving.py`) has all ranks run each gather collective but only the
  save rank keeps the tensor (per-node CPU RAM peaks well under 1× model, not gpus_per_node×). TP
  mirrors this, and on the same `fs_aware_save_rank()` — never `tp_rank == 0`, which the several TP
  groups a node holds would race on identical files. Don't "optimize" by skipping ranks in the
  collective. The save rank is
  `fs_aware_save_rank()` (`src/distributed/runtime.py`): on a **shared FS** only global rank 0
  writes the shards + index (avoids the NFS write race of many ranks touching one path); on a
  **non-shared FS** (`DIST_SHARED_FILESYSTEM=0`) each node's local rank 0 writes its own copy to
  node-local storage.
