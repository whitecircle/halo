# Checkpoints: Save, Resume, Export, Upload

A run writes standard HuggingFace checkpoints to `output_dir` on the cadence set
by `save_strategy` / `save_steps`. Every mode (FSDP2, EP, TP, CP, PP and their
combinations) writes a checkpoint that `from_pretrained` loads. The exceptions
are the opt-in sharded EP save, a PP save on per-node storage, and LoRA
adapters.

Every save uses the tensor names of the checkpoint the run loaded, vision towers
included, and an EP save writes the family's own expert layout
([details](../agent-docs/reference/checkpoints.md#serving-on-vllm--sglang) ↗).

| Training mode | Output | Load directly? |
| --- | --- | --- |
| FSDP2, CP, TP | gathered HF checkpoint | yes |
| EP (default) | gathered HF checkpoint, experts at global indices | yes |
| EP with `save_sharded_ep: true` | one expert shard per rank | merge first |
| PP | one part set per stage plus the full index | yes; on per-node storage, merge the nodes' directories first |
| LoRA / QLoRA | adapter only (`adapter_model.safetensors`) | serve as an adapter, or merge into the base |

EP expert LoRA cannot be merged afterwards: train it with
`merge_expert_lora_on_save: true` to save the merged model.

## Sharded EP saves

`save_sharded_ep: true` skips the gather, which makes checkpointing a very large
model faster. Run `halo run merge-ep-shards` before the checkpoint can be used;
the merged directory carries every rank's optimizer shard, so it also resumes
exactly on the topology that wrote it.
It needs one EP group spanning all ranks, no CP, ETP or expert LoRA, a shared
filesystem across nodes and a supported family, all checked at startup
([full list](../agent-docs/reference/checkpoints.md#expert-parallelism-ep-eptp-epcp) ↗).

## Resume

Set `resume_from_checkpoint` to a checkpoint directory, or to `true` for the
newest complete one in `output_dir`, and launch the same training script again.
The training script restores the weights, so resume through it, not through a
hand-built model.

| Situation | What happens |
| --- | --- |
| Same topology fingerprint: world size, every parallel size, the layout knobs, the optimizer class | exact resume: weights, optimizer state, LR schedule and step |
| A different fingerprint, or a save with `save_only_model: true` | warm restart: weights and LR schedule, fresh optimizer moments |
| The optimizer restore fails (a truncated shard, a CUDA OOM, the optimizer half of an interrupted save) | every rank stops, naming the cause; `allow_optimizer_warm_restart: true` accepts fresh moments |
| `resume_from_checkpoint: true` and a step directory lacks `trainer_state.json` (written last) | that save never completed: it moves to `output_dir/_incomplete_checkpoints/` and the newest complete one resumes |
| An explicit path to an incomplete directory, or an `output_dir` whose step directories are all incomplete | raises |

A save is safe to interrupt: `save_total_limit` never deletes a checkpoint
before its successor is complete. LoRA runs restore the adapters onto the base,
and MoE router-balancing state is restored automatically. Checkpoints from Halo
1.1.0 resume on this release; merge a sharded EP save first.
`load_best_model_at_end` is refused on a full fine-tune under CP, on an MoE with
EP layers, and under TP with several replicas
([full mechanics](../agent-docs/reference/checkpoints.md#resuming-training) ↗).

## Per-node PP saves

On per-node storage (`DIST_OUTPUT_SHARED_FILESYSTEM=0`) a PP save leaves each
node only its own stages' parts and its own ranks' optimizer state. Each node
resumes from its own directory while the topology and node ranks stay the same.
To load, serve or upload the result, copy every node's directory onto one host
and merge them:

```bash
halo run merge-pp-stages --input_dir /copies/node0/checkpoint-500 \
    /copies/node1/checkpoint-500 --output_dir /models/merged
```

The merge refuses a missing part or optimizer shard, and the merged directory
also resumes on the same topology
([Per-node export](../agent-docs/parallelism/pipeline-parallelism.md#per-node-export) ↗).

## Post-processing tools

All run as `halo run <tool> <flags>`; `halo run <tool> -- --help` lists the
flags.

| Tool | Use it to |
| --- | --- |
| `merge-ep-shards` | make a sharded EP save loadable |
| `merge-pp-stages` | merge the per-node directories of a PP save into one checkpoint |
| `merge-peft-adapters` | merge a LoRA adapter into its base as one checkpoint |
| `merge-models` | combine models in weight space (linear, SLERP, task arithmetic, TIES) |
| `convert-to-bf16` | cast an fp32 or mixed checkpoint to bf16 for serving |
| `quantize-to-lowp` | write block-scaled mxfp8 / mxfp4 / nvfp4 weights; serving engines do not load its manifest as-is |
| `unfuse-moe-experts` | rewrite fused expert weights to the family's per-expert layout, for a serving loader that reads only that layout |
| `reset-sinks` | disable the attention sinks in a GPT-OSS checkpoint |
| `reattach-vision-tower` | restore the vision tower, MTP head and processor files to a `text_only_model` export; SGLang needs it for a Qwen3.8-Flash-Next export |

The tools write to a new directory (only `reset-sinks` takes `--in_place`) and
reject a sharded EP save until `merge-ep-shards` has run. Only `merge-ep-shards`
and `merge-pp-stages` carry the optimizer shards. Flags and input guards:
[Scripts reference](../agent-docs/reference/scripts-reference.md#post-training-scripts) ↗.

## Serving the result

A gathered checkpoint loads with `from_pretrained`. vLLM and SGLang serve it
only where the pinned engine reads that family's layout:

- **No serving path:** Mistral4 and Ring on either engine (neither registers
  their architecture), and Zaya on SGLang, whose loader reads a layout the export
  does not carry.
- **Untested:** Zaya on vLLM's generic transformers backend, Ling 3.0, Inkling,
  and GLM-5 Next and DeepSeek-V4 after `unfuse-moe-experts`. The exports use the
  hub's names, but no engine has served one.
- **Every other family** serves its normal save as is, except a Cohere2 MoE
  (Command A+) EP save on SGLang: rewrite it with `unfuse-moe-experts` first. On
  1.2.0, of the trained full-size exports, only Gemma 4's has been served, on both
  engines.
- **Fused experts on a per-expert-only loader** are refused or dropped without
  an error. Rewrite such a checkpoint with `unfuse-moe-experts` first; the result
  is the hub's own layout
  ([per-engine loaders](../agent-docs/reference/checkpoints.md#serving-on-vllm--sglang) ↗).
- **Sharded EP saves** need `merge-ep-shards` first, and per-node PP saves
  `merge-pp-stages`.
- **LoRA runs** serve as base plus adapter, or merged.

Serving during training, for the RL methods, is
[Rollout Servers](rollout-servers.md).

## Uploading to the HuggingFace Hub

Upload a final model yourself. A gathered checkpoint is a plain HF model
directory, so the standard Hub CLI works:

```bash
hf auth login          # once, or set HF_TOKEN
hf upload my-org/my-model checkpoints/sft-qwen3-4b-ultrachat/checkpoint-1000
```

Upload the checkpoint directory (or a converted copy), not the whole
`output_dir`, and leave out its training state, or it goes public with the
weights:

- `trainer_state.json` and `scheduler.pt`;
- `optimizer*` and `rng_state*`, unless the run set `save_only_model: true`;
- `router_balancing_biases.pt` (bias balancing);
- `reference_logps.pt` (DPO/KTO `precompute_ref_log_probs`, or offline GRPO full
  fine-tuning at `kl_beta > 0`);
- `prefetch_pending-*.pt` (async GRPO with prefetch): drawn but untrained
  prompts, answers included;
- `resume_adapter/` and `resume_adapter.json` (`merge_expert_lora_on_save` or
  embedding LoRA).

For a LoRA run, upload the adapter directory, or merge it with
`merge-peft-adapters` for a standalone model. Load an uploaded adapter with
`PeftModel.from_pretrained`.

The inherited `push_to_hub` / `hub_strategy` upload is not tested end to end. It
pushes `log/run.log` and, for online or async GRPO, the `completions/` prompts
along with the model files, and `hub_strategy: checkpoint` or `all_checkpoints`
also uploads training state
([details](../agent-docs/reference/checkpoints.md#interrupted-saves) ↗).

### Model card

Every checkpoint Halo writes holds a `README.md` model card tagged `halo`, so the
upload lists under that Hub tag. Before you publish, fill in its body and replace
a local-path `base_model` with the Hub id.

Shard layouts, merge flags and the full resume mechanics:
[Checkpoints](../agent-docs/reference/checkpoints.md) ↗ ·
[Model Merging](../agent-docs/reference/model-merging.md) ↗.
