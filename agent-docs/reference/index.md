# Reference

## Architecture & design

<!-- markdownlint-disable MD030 -- mkdocs-material grid cards require the 4-space content indent -->

<div class="grid cards" markdown>

-   **[Architecture](architecture.md)**

    The layer map — config, trainers, distributed, data, models, callbacks, diagnostics, kernels, checkpoints, optimizers, environments — and how every trainer adds one mixin to a TRL/Transformers base.

-   **[Trainer Architecture](trainer-architecture.md)**

    The distributed mixin, EP/TP/CP patching, vLLM weight sync, and saving across all trainers.

-   **[Why This Framework](why-this-framework.md)**

    How the toolkit compares to HF TRL, Accelerate, Unsloth, Axolotl, MS-SWIFT, veRL, and Megatron-LM, plus model compatibility and key dependencies.

-   **[GPU Training Theory](gpu-training-theory.md)**

    The roofline, the four bottlenecks a step can hit, and what every optimization and parallelism lever actually pulls on.

</div>

---

## Configuration & usage

<div class="grid cards" markdown>

-   **[Configuration Reference](configuration-reference.md)**

    Every config class with parameter tables, YAML examples, and CLI overrides.

-   **[Scripts Reference](scripts-reference.md)**

    Every script by category — training, inference, post-training, preparation, and profiling — each with a usage example.

</div>

---

## Operations

<div class="grid cards" markdown>

-   **[Checkpoints & Resume](checkpoints.md)**

    The four on-disk layouts (single file, gathered HF-sharded, PP per-stage, EP per-rank), the resume paths, shard merging, and the per-mode saver ladder.

-   **[Model Merging](model-merging.md)**

    Combine same-architecture checkpoints in weight space — linear, SLERP, task-arithmetic, and TIES — via `scripts/after_training/merge_models.py`.

-   **[Capabilities & Limitations at Scale](scale-and-limitations.md)**

    What works and what does not for very large models, multi-week runs, huge corpora, and many nodes.

</div>

---

## Debugging & glossary

<div class="grid cards" markdown>

-   **[Troubleshooting](troubleshooting.md)**

    Symptom-to-fix lookup for OOM, NCCL hangs, DeepEP faults, config rejections, NaN loss, and attention-backend mismatches.

-   **[Debugging & Profiling](debugging.md)**

    Distributed debug helpers, torch.profiler traces with labeled EP phases, TraceLens trace analysis, py-spy attach, memory snapshots, and multi-node hang diagnosis.

-   **[Glossary](glossary.md)**

    Parallelism, kernel, optimizer, and method terms.

</div>
