# Checkpoints: Save, Resume, Export, Upload

A run writes standard HuggingFace checkpoints to `output_dir` on the cadence set
by `save_strategy` / `save_steps`. Every mode — dense FSDP2, EP, TP, CP, and
their combinations — produces a gathered checkpoint you can load with
`from_pretrained`. The exceptions are opt-in sharded saves and LoRA adapters.

| Training mode | Output | Load directly? |
| --- | --- | --- |
| Dense FSDP2 / CP | gathered HF checkpoint | yes |
| EP, gathered (default) | gathered HF checkpoint, experts at global indices | yes |
| TP | gathered HF checkpoint | yes |
| EP with `save_sharded_ep: true` | one expert shard per rank | merge first |
| LoRA / QLoRA | adapter only (`adapter_model.safetensors`) | serve as adapter, or merge into base |

The sharded EP save skips the gather, which speeds up checkpointing for very
large models; the trade is a `halo run merge-ep-shards` before the checkpoint
stands alone. Its restrictions (single EP group, a shared output filesystem) are
checked at startup, so an unsupported combination fails fast. TP needs no
sharded mode — full tensors are reconstructed from the DTensors at save time.

## Resume

Point `resume_from_checkpoint` at a checkpoint directory, or set it to `true` to
pick up the latest one in `output_dir`.

A distributed run also saves per-rank optimizer shards (single-GPU and DDP keep
HF's `optimizer.pt`), the schedule, and the step. Resume is **exact**, optimizer
state included, as long as the run's topology fingerprint matches. That
fingerprint covers more than the GPU count: world size, the EP / ETP / CP / TP /
PP sizes, `ep_scope`, `nvlink_domain_size`, `hsdp`, `use_grouped_gemm`,
`fsdp_shard_ep1_experts`, `expert_replica_size`, and the optimizer class. The
last few change slice ownership or expert parameter names while every tensor
shape stays identical, so without them a restore would report success over
permuted state.

If any field differs, or the run saved with `save_only_model: true`, you get a
warm restart instead: weights and learning-rate schedule restored, optimizer
moments fresh. MoE router-balancing state round-trips automatically where it
applies.

EP and CP resume by rebuilding the model *from* the checkpoint rather than
loading weights back into a running model, which the fused-expert and CP-wrapped
layouts cannot accept. One consequence: the training scripts must launch the
resume, since they repoint the model source.

`load_best_model_at_end` is refused at startup for a full fine-tune under CP, on
a MoE model wrapped for expert compute (which includes plain FSDP2 at the
default `use_grouped_gemm: true`, not just EP), and under TP with more than
one data-parallel replica (`tp_size > 1` with `data_parallel_size > 1`). The end-of-run reload would be refused *after* the whole run,
and the export would quietly carry the last weights — so the check fails fast
instead. Export the best checkpoint directly. Dense **pure** TP is allowed: its
shards are plain tensors that reload normally. Adapter-only runs are exempt
unless `merge_expert_lora_on_save` folds them into a full base checkpoint.

## Post-processing tools

All run as `halo run <tool> -- <flags>`; each answers `-- --help`.

| Tool | When |
| --- | --- |
| `merge-ep-shards` | make a sharded EP save loadable |
| `merge-peft-adapters` | fold a LoRA adapter into its base as one checkpoint |
| `merge-models` | weight-space combine (linear, SLERP, task-arithmetic, TIES) |
| `convert-to-bf16` | cast an fp32/mixed checkpoint down for serving |
| `quantize-to-lowp` | write a block-scaled `compressed-tensors` checkpoint (mxfp8 / mxfp4 / nvfp4) |
| `unfuse-moe-experts` | rewrite fused MoE expert weights to the per-expert hub layout, for a family whose hub form is per-expert (GLM-4 Lite, Laguna, LFM-2, Qwen3 MoE, Qwen3.5/3.6, DeepSeek-V4, Bailing/Ling, Command A+, GLM-5 Next); refuses the families that store fused |
| `reset-sinks` | disable the attention-sink mechanism in a GPT-OSS checkpoint |

Give each tool an output path different from its input: they refuse to run in
place, because writing over the source deletes the shards they don't overwrite.
`reset-sinks` alone takes an explicit `--in_place` flag (there is no undo, so it
has to be asked for). They also reject a per-rank EP-sharded checkpoint — merge
it first with `merge-ep-shards`, the one tool that takes that layout.

## Serving with vLLM

Any gathered checkpoint loads into vLLM directly. Merge sharded saves first, and
serve a LoRA run either as base-plus-adapter or merged. If vLLM's loader
complains about fused expert weights, run `unfuse-moe-experts` on the checkpoint.
Step-3.7 Flash needs no rewrite — its gathered save is already in the hub layout
vLLM reads, and the tool refuses it accordingly.

## Uploading to the HuggingFace Hub

Halo wires no Hub upload into the save path. The inherited `push_to_hub` /
`hub_model_id` `TrainingArguments` fields still parse, but nothing guards or
tests them against Halo's overridden gathered/EP save, and the end-of-training
push never fires (no Halo script calls `save_model()`). Upload explicitly
instead — a gathered checkpoint is a plain HF model directory, so the standard
Hub CLI works:

```bash
hf auth login          # once, or set HF_TOKEN
hf upload my-org/my-model checkpoints/sft-qwen3-4b-ultrachat/checkpoint-1000
```

Upload the checkpoint directory itself, not the whole `output_dir` — that also
holds optimizer shards and logs you don't want public. For a LoRA run, upload
the adapter directory, or merge first for a standalone model.

Shard layouts, merge flags, and the full resume mechanics:
[Checkpoints](../agent-docs/reference/checkpoints.md) ↗ ·
[Model Merging](../agent-docs/reference/model-merging.md) ↗.
