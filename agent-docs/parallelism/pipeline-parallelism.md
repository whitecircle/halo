# Pipeline Parallelism

PP splits a model's decoder layers into contiguous **stages**, each owning a rank block. It is the
outermost parallelism dimension and the only one designed to cross NVLink domains: the only traffic
leaving a domain is the pipeline's point-to-point boundary activations. It composes with the expert
axes only — EP, or pure ETP, never both — and FSDP2 data parallelism runs inside each stage. The
bandwidth argument: [GPU Training Theory §9](../reference/gpu-training-theory.md#pipeline-parallelism-the-bubble-and-the-boundary).

## Launching

Set `pipeline_parallel_size` on a script whose trainer declares `_supports_pp`: SFT, SMPO, DPO,
KTO, offline GRPO, Bradley-Terry reward and classification. Any other script refuses it at config
time with `<Trainer> does not support Pipeline Parallelism (PP)`, before a model loads. Every stage
must be a whole number of NVLink domains holding at least two ranks, so a run spans at least two
domains.

The pipeline's point-to-point buffers freeze their shapes on the first step, so every batch must
keep one shape: a padding-free collator is refused, and packing runs at one packed row per
microbatch.

## Components

- **Config surface.** `ParallelismConfig` declares `pp_size`, `pp_schedule` (`1f1b` / `gpipe`),
  `pp_microbatches`, and `pp_split`, mirrored from the CLI as `pipeline_parallel_size`,
  `pipeline_schedule`, `pipeline_microbatches`, `pipeline_split`. At `pp_size == 1` (the default)
  setting any PP-only knob is rejected, so a config cannot carry silently ignored fields.
- **Rank math and topology rules.** The stage-scoped coordinates (`stage_world_size`, `pp_rank`,
  `stage_base_rank`, `stage_local_rank`) are the base every other axis (EP/CP/TP/ETP) computes its
  groups from; at `pp_size == 1` they equal the world values. Validation pins stage boundaries to
  NVLink-domain boundaries and rejects single-rank stages. The
  [capability-matrix allowlist](README.md#supported-combinations) admits PP alone, PP+EP and PP+ETP;
  every other axis combination (PP+TP, PP+CP, PP+EP+ETP, …) is refused with the breaking mechanism
  named in the error. `_validate_pipeline_parallel` separately refuses the PP-incompatible knobs
  (HSDP, expert LoRA, ZeRO-3, non-`bf16` low precision).
- **Trainer gates.** `_supports_pp` is declare-to-enable per trainer (like `_supports_cp`), and
  `src/trainers/mixins/pp_gates.py` is the shared rejection vocabulary (PEFT, live reference
  models, `compute_metrics`, activation offloading, precomputed-reference-column requirements);
  `PipelineTrainerMixin` (`src/trainers/mixins/pipeline.py`) is the trainer-side seam and stays
  inert at `pp_size == 1`.
- **Stage, split, and loss seams.** `src/distributed/pipeline_parallel/` holds the stage module and
  stage↔global naming contract (`stage.py`), the per-family `PPModelSpec` registry with the
  layer-partition math and the model-structure gates — tied embeddings, live MTP tail layers,
  `layer_types`-period split offsets, a head path the last stage cannot reproduce (`split.py`) —
  the mid-chain stream forwards for the hyper-connection families, whose inter-layer activation is
  the `hc_mult`-widened `[batch, seq, hc_mult, hidden]` stream (DeepSeek-V4, GLM-5 Next — `stage_adapters.py`), the
  pure-tensor loss/label helpers and the `PPLossAdapter` contract (`losses.py`), the group
  constructors (`groups.py`), and the stage-aware safetensors lazy loader (`lazy_loader.py`). The
  batch contract (`PP_BATCH_PAD_VALUES` in `runtime.py`) pins the keys and pad values a pipeline
  consumes.
- **Schedule engine.** `PipelineRuntime` (`src/distributed/pipeline_parallel/runtime.py`) drives one
  optimizer step's microbatches through the FSDP2-wrapped stage with torch's `Schedule1F1B` or
  `ScheduleGPipe`. Each microbatch loss is divided by the step's token count, so the microbatch
  losses sum to the unsplit loss; per-example side tensors reach the last stage's loss split with
  their rows; loss-only and forward-only drives skip the schedule's trailing gradient reduction.
  Construction verifies the private torch members the engine relies on.
- **Head transform.** The last stage applies what the family's `*ForCausalLM.forward` does around
  `lm_head` — Cohere's `logit_scale`, Gemma's softcap, Inkling's μP division and vocabulary cut — to
  the logits it returns and inside the fused head loss. It is the transform the chunked GRPO
  log-probs apply, verified against the family's forward (`src/models/head_transform.py`); a family
  whose forward it does not reproduce is refused by the split gate on every rank.
- **Checkpoint seam.** `save_pp_checkpoint` (`src/distributed/checkpoint/save.py`) defines the PP
  checkpoint layout: one complete-tensor safetensors shard per stage under the hub's names plus a
  merged standard HF index, loadable via plain `from_pretrained`. A stage is no `PreTrainedModel`
  and names its tensors stage-locally, so `build_pipeline_stage` snapshots the unsplit model's load
  conversions; the save reverts each decoder layer, EP layer gather and wrapper stash through them,
  writing the namespace the EP-gathered save writes
  ([Checkpoints](../reference/checkpoints.md#serving-on-vllm--sglang)), and the stage reload maps
  those names back through `ModuleLayoutView`. A module the backbone applies to the token embedding
  alone (Inkling's `embed_norm`, the spec's `INPUT_MODULE_ATTRS`) stays on stage 0 with the
  embedding, so no two stages write it. The writer-set and axis guards (`is_pp_shard_writer`,
  `reject_unhandled_pp_axes`) ship with it.
