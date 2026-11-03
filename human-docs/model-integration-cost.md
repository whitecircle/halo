# Model integration cost

Halo adds distributed behavior as layers and wrappers around the HuggingFace
model, not as a fork of it, so a new family costs little code. For what is
already supported, see [Supported models](models.md).

Plain FSDP2 needs no code. Any `AutoModelForCausalLM` trains as is, multimodal
checkpoints included (loaded through the full vision-and-text class). Code comes
in only for parallelism:

```txt
HuggingFace model load
  -> thin EP layer or CP wrapper (subclass a base class)
  -> registers by naming the HF class it claims
  -> existing FSDP / EP / ETP / checkpoint paths
```

## What an EP layer costs

Each MoE family adds one file under `halo/distributed/expert_parallel/families/`
naming the HF MoE class it claims, one frozen `MoEFamilySpec` holding its config
`model_type` spellings and capabilities (`halo/models/moe/family_specs.py`), and
one import line in the family roster.

- EP layers subclass a shared base. A family whose expert-weight layout matches
  an existing one subclasses that instead: Laguna reuses GLM-4's in 19 lines,
  Qwen3-Next reuses Qwen3.5-MoE's in 25, and Qwen3.8-Flash-Next reuses
  Qwen3-Next's in 11.
- Every other EP layer file is 98 lines or fewer.
- GPT-OSS is the outlier at 171, because of its interleaved, biased gate/up
  weights and its own clamped SwiGLU compute.

Context parallelism works the same way: one `UlyssesAttention` subclass in
`halo/distributed/context_parallel/families.py` that declares the attention classes it
wraps. Both registries are built from the subclass tree, so a class that exists
gets used.

Pipeline parallelism splits a family only once its backbone class is
registered in `halo/distributed/pipeline_parallel/families.py`; any other is
refused by name before a weight loads.

Tensor parallelism on a dense model uses HuggingFace's own `base_model_tp_plan`.
On a MoE model it shards attention only, and the one list to edit is
`TP_SHARDABLE_ATTENTION_CLASSES`.

## Vendoring

A model that transformers doesn't ship yet can be vendored under
`halo/models/<name>/`: its `configuration_*.py` and `modeling_*.py`, registered on
import, with a CPU test that proves the registration. Remove the copy once
upstream ships the model.

No model is vendored. The only model classes Halo adds are the
sequence-classification heads transformers doesn't ship, for Gemma 4 and
Qwen3.5/3.6 MoE (`halo/models/seq_cls_heads.py`).

The step-by-step procedure for each mode is in
[Adding a Model](../agent-docs/models/adding-a-model.md) ↗.
