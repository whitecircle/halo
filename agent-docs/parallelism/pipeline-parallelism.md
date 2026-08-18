# Pipeline Parallelism

**Status: not yet available in this release.** The pipeline-parallel seams ship — the config
surface, the rank math, the trainer gates, and the stage/split/loss/checkpoint contracts — but the
schedule engine that would drive microbatches through the stages does not.
`pipeline_parallel_size > 1` is rejected at config time with a pointer to this page; the full
engine lands in a future release.

PP would split a model's decoder layers into contiguous **stages**, each owning a rank block, as
the outermost parallelism dimension and the only one designed to cross NVLink domains — the only
traffic leaving a domain being the pipeline's point-to-point boundary activations. The bandwidth
argument: [GPU Training Theory §9](../reference/gpu-training-theory.md#pipeline-parallelism-the-bubble-and-the-boundary).

## What ships today

- **Config surface.** `ParallelismConfig` declares `pp_size`, `pp_schedule` (`1f1b` / `gpipe`),
  `pp_microbatches`, and `pp_split`, mirrored from the CLI as `pipeline_parallel_size`,
  `pipeline_schedule`, `pipeline_microbatches`, `pipeline_split`. At `pp_size == 1` (the default
  and the only value a production run accepts) setting any PP-only knob is rejected, so a config
  cannot carry silently ignored fields.
- **Rank math and topology rules.** The stage-scoped coordinates (`stage_world_size`, `pp_rank`,
  `stage_base_rank`, `stage_local_rank`) are the base every other axis (EP/CP/TP/ETP) computes its
  groups from; at `pp_size == 1` they equal the world values. Validation pins stage boundaries to
  NVLink-domain boundaries and rejects single-rank stages. The
  [capability-matrix allowlist](index.md#supported-combinations) admits PP only alone or with the
  expert axes (PP+EP, PP+ETP); every other combination — PP+TP, PP+CP, PP+HSDP, and the rest — is
  refused with the breaking mechanism named in the error.
- **Trainer gates.** `_supports_pp` is declare-to-enable per trainer (like `_supports_cp`), and
  `src/trainers/mixins/pp_gates.py` is the shared rejection vocabulary (PEFT, live reference
  models, `compute_metrics`, activation offloading, precomputed-reference-column requirements);
  `PipelineTrainerMixin` (`src/trainers/mixins/pipeline.py`) is the trainer-side seam and stays
  inert at `pp_size == 1`.
- **Stage, split, and loss seams.** `src/distributed/pipeline_parallel/` holds the stage module and
  stage↔global naming contract (`stage.py`), the per-family `PPModelSpec` registry with the
  layer-partition math and the model-structure gates — tied embeddings, live MTP tail layers,
  `layer_types`-period split offsets (`split.py`) — the mid-chain stream forwards for the
  hyper-connection families, whose inter-layer activation is the `hc_mult`-widened
  `[batch, seq, hc_mult, hidden]` stream (DeepSeek-V4, GLM-5 Next — `stage_adapters.py`), the
  pure-tensor loss/label helpers and the `PPLossAdapter` contract (`losses.py`), the group
  constructors (`groups.py`), and the stage-aware safetensors lazy loader (`lazy_loader.py`). The
  batch contract (`PP_BATCH_PAD_VALUES` in `runtime.py`) pins the keys and pad values a pipeline
  consumes.
- **Checkpoint seam.** `save_pp_checkpoint` (`src/distributed/checkpoint/save.py`) defines the PP
  checkpoint layout: one complete-tensor safetensors shard per stage under the unsplit model's
  global names plus a merged standard HF index, loadable via plain `from_pretrained`; the
  writer-set and axis guards (`is_pp_shard_writer`, `reject_unhandled_pp_axes`) ship with it.

## What does not ship

`PipelineRuntime` (`src/distributed/pipeline_parallel/runtime.py`) — the seam over
`torch.distributed.pipelining` that would own the schedule, the microbatched step, and the
forward-only pass — raises `NotImplementedError` on construction. Because
`parallelism_config_from_args` (`src/training/parallelism_args.py`) rejects
`pipeline_parallel_size > 1` at the single production entry point, before any rank math or model
loading, no production path reaches it. No PP run is launchable.

## Sharding large models today

PP targets the models whose layer stack outgrows one NVLink domain. Until it lands, shard with the
available axes: [Expert Parallelism](expert-parallelism.md) (orthogonal to DP, the workhorse for
MoE), [Tensor Parallelism](tensor-parallelism.md), [Context Parallelism](context-parallelism.md)
for long sequences, and their supported combinations — see the [Parallelism overview](index.md) and
[Multi-Node](multi-node.md) for topology guidance.
