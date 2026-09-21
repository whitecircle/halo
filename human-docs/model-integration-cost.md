# Model Integration Cost

Halo applies distributed behavior as wrappers around the HuggingFace model
rather than as a fork of it, so adding a family costs little code. What is
already supported is [Supported Models](models.md).

For plain FSDP2 with no parallelism flags there is nothing to do at all: any
`AutoModelForCausalLM` trains as is, multimodal checkpoints included (loaded
through the full vision+text wrapper). Parallelism is where code gets added.

```txt
HuggingFace model load
  -> thin EP/TP/CP wrapper (subclass a base layer)
  -> self-registers by naming the HF class it claims
  -> existing FSDP / EP / ETP / checkpoint paths
```

## What a wrapper costs

Each shipped MoE family adds one file under
`src/distributed/expert_parallel/layers/`, which registers itself by naming the
HF MoE class it claims and the config `model_type`. They subclass a shared base,
so a family whose expert-weight layout matches an existing one subclasses that
instead of starting over — Laguna reuses GLM-4's in 39 lines. Most land between
40 and 140 lines. GPT-OSS, with interleaved gate/up weights and attention sinks,
is the outlier at 366.

Context parallelism is the same shape: one `UlyssesAttentionBase` subclass under
`context_parallel/layers/` declaring the attention classes it wraps. Tensor
parallelism usually costs nothing — a model carrying `base_model_tp_plan` shards
through HuggingFace's own plan.

There are no accept lists to edit anywhere in this: the registries are derived
from the subclass tree, so a wrapper that exists is a wrapper that is used.

## Vendoring

A model transformers has not landed yet can be vendored under
`src/models/<name>/`: its `configuration_*.py` and `modeling_*.py`, registered on
import, with a CPU test that proves the registration. Remove the copy once
upstream ships the model. Today nothing is vendored: the only model definitions
under `src/models/` are the sequence-classification heads transformers does not
ship, for Gemma 4 and Qwen3.5 MoE.

The step-by-step procedure, per mode, is
[Adding a Model](../agent-docs/models/adding-a-model.md) ↗.
