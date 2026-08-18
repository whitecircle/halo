---
hide:
  - toc
---

# Halo

Halo is an LLM training toolkit built on HuggingFace TRL, Transformers, and Accelerate. It adds Expert, Context, Tensor, and Expert-Tensor parallelism (EP / CP / TP / ETP) for MoE and dense models (pipeline parallelism is [not yet available in this release](parallelism/pipeline-parallelism.md)), plus pre-training and post-training methods: SFT, preference optimization (SMPO, DPO, KTO), the GRPO family (offline, online/RLVR, environmental), distillation (teacher, self, online SDPG), reward modeling, classification, and embedding.

Trainers subclass TRL/Transformers trainers, so any Hub model trains with the same `TrainingArguments`, Accelerate FSDP configs, and PEFT/LoRA — no checkpoint conversion.

Start with the cards below, or jump to [Training Methods](training-methods/index.md), [Parallelism](parallelism/index.md), [Models](models/index.md), or the [Reference](reference/index.md).

<!-- markdownlint-disable MD030 -- mkdocs-material grid cards require the 4-space content indent -->

<div class="grid cards" markdown>

-   :material-download:{ .lg .middle } **Installation**

    ---

    The prebuilt Blackwell/Hopper image is the supported path — everything, `make install` included, runs inside it.

    [:octicons-arrow-right-24: Installation guide](getting-started/installation.md)

-   :material-rocket-launch:{ .lg .middle } **Quickstart**

    ---

    First SFT run through multi-turn RL, with LoRA, the parallelism flags, and dataset sources.

    [:octicons-arrow-right-24: Quickstart](getting-started/quickstart.md)

-   :material-file-cog:{ .lg .middle } **Configuration Guide**

    ---

    The YAML system, toolkit defaults (BF16, Liger), CLI overrides, and accelerate vs torchrun launchers.

    [:octicons-arrow-right-24: Configuration](getting-started/configuration.md)

-   :material-compass:{ .lg .middle } **Choosing a Training Method**

    ---

    A decision flowchart and comparison table — match your data to a method, and see which trainers take CP.

    [:octicons-arrow-right-24: Choosing a method](getting-started/choosing-a-method.md)

</div>

[GPU Training Theory](reference/gpu-training-theory.md) covers the roofline and training-step bottlenecks behind every optimization lever — read it before tuning. For the design rationale and how the toolkit relates to TRL, Accelerate, Unsloth, Axolotl, MS-SWIFT, veRL, and Megatron-LM, see [Why this framework](reference/why-this-framework.md).
