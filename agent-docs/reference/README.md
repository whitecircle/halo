# Reference

## Architecture & design

- [Architecture](architecture.md): the `halo/` component map, the import layer table, and how every trainer adds one mixin to a TRL, Transformers or SentenceTransformers base.
- [Trainer Architecture](trainer-architecture.md): the distributed mixin, trainer × CP/PP support, the per-mode wrap, gradient sync and clipping, rollout-engine weight sync, and saving across all trainers.
- [Why This Framework](why-this-framework.md): how Halo compares to HF TRL, Accelerate, Unsloth, Axolotl, MS-SWIFT, veRL, and Megatron-LM, plus model compatibility and key dependencies.
- [GPU Training Theory](gpu-training-theory.md): the roofline, the limits a step can hit (compute-, memory- and overhead-bound on one GPU; communication-bound and waiting across many), and what every optimization and parallelism lever actually pulls on, built on measurements taken on one B300.

## Configuration & usage

- [Configuration Reference](configuration-reference.md): every config class field by field, the environment-variable catalog, YAML parsing and CLI overrides.
- [Scripts Reference](scripts-reference.md): every script by category (training, inference, post-training, preparation, profiling) with its flags and input guards, and the `halo launch` / `halo run` CLI.

## Operations

- [Checkpoints & Resume](checkpoints.md): the four on-disk layouts (single file, gathered HF-sharded, PP per-stage, EP per-rank), the resume paths, shard merging, and the per-mode saver ladder.
- [Model Merging](model-merging.md): combine same-architecture checkpoints in weight space (linear, SLERP, task arithmetic and TIES) with `scripts/after_training/merge_models.py`.
- [Capabilities & Limitations at Scale](scale-and-limitations.md): what works and what does not for very large models, multi-week runs, huge corpora, and many nodes.

## Debugging & glossary

- [Troubleshooting](troubleshooting.md): symptom-to-fix lookup for OOM, NCCL hangs, DeepEP faults, config rejections, NaN loss, and attention-backend mismatches.
- [Debugging & Profiling](debugging.md): distributed debug helpers, torch.profiler traces with labeled EP phases, TraceLens trace analysis, py-spy attach, memory snapshots, and multi-node hang diagnosis.
- [Glossary](glossary.md): parallelism, kernel, optimizer and method terms.
