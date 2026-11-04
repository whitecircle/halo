# Halo

Halo is an LLM training toolkit built on HuggingFace TRL, Transformers, and Accelerate. It adds Expert, Context, Tensor, Expert-Tensor, and Pipeline parallelism (EP / CP / TP / ETP / PP) for MoE and dense models, plus pre-training and post-training methods: SFT, preference optimization (SMPO, DPO, KTO), the GRPO family (offline, online/RLVR, async with environments), distillation (teacher, self, online SDPG), reward modeling, classification, and embedding. SFT also trains the multi-token-prediction head a release ships ([MTP heads](optimization/mtp-heads.md), `mtp_loss_weight`).

Trainers subclass TRL/Transformers trainers, so any Hub model trains with the same `TrainingArguments`, Accelerate FSDP configs, and PEFT/LoRA — no checkpoint conversion, except a one-time BF16 dequantize of the FP8 releases (DeepSeek-V4, GLM-5 Next, Mistral 4; [Models](models/README.md)).

Start with the guides below, or jump to [Training Methods](training-methods/README.md), [Parallelism](parallelism/README.md), [Models](models/README.md), [Data](data/README.md), [Optimization](optimization/README.md), [Infrastructure](infrastructure/README.md), [Contributing](contributing/README.md), or the [Reference](reference/README.md).

- **[Installation](../human-docs/installation.md)** — Pull or build the Blackwell/Hopper image (the supported path, everything compiled in) and start a container.
- **[Quickstart](../human-docs/quickstart.md)** — Launch an example config, watch it, use the result.
- **[Choosing a method](../human-docs/choosing-a-method.md)** — Match the data you have to a trainer.
- **[Configuration Reference](reference/configuration-reference.md)** — The YAML parser, CLI overrides, every field and environment variable.

[GPU Training Theory](reference/gpu-training-theory.md) covers the roofline and training-step bottlenecks behind every optimization lever — read it before tuning. For the design rationale and how the toolkit relates to TRL, Accelerate, Unsloth, Axolotl, MS-SWIFT, veRL, and Megatron-LM, see [Why this framework](reference/why-this-framework.md).
