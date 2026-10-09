# Model integration cost

Halo adds distributed behavior as wrappers around the HuggingFace model, not as a
fork of it, so a new family costs little code. For what is already supported,
see [Supported models](models.md).

Plain FSDP2 needs no code. Any `AutoModelForCausalLM` trains as is, multimodal
checkpoints included (loaded through the full vision-and-text class). Code comes
in only for parallelism:

```txt
HuggingFace model load
  -> thin EP/TP/CP wrapper (subclass a base layer)
  -> self-registers by naming the HF class it claims
  -> existing FSDP / EP / ETP / checkpoint paths
```

## What a wrapper costs

Each MoE family adds one file under `src/distributed/expert_parallel/layers/`. The
file registers itself by naming the HF MoE class it claims and the config
`model_type`.

- Wrappers subclass a shared base. A family whose expert-weight layout matches
  an existing one subclasses that instead: Laguna reuses GLM-4's in 39 lines.
- Most wrappers stay under 140 lines.
- GPT-OSS is the outlier at about 360, because of its interleaved gate/up
  weights and attention sinks.

Context parallelism works the same way: one `UlyssesAttentionBase` subclass under
`src/distributed/context_parallel/layers/` that declares the attention classes it
wraps. Both registries are built from the subclass tree, so a wrapper that exists
gets used.

Tensor parallelism on a dense model uses HuggingFace's own `base_model_tp_plan`.
On a MoE model it shards attention only, and the one list to edit is
`TP_SHARDABLE_ATTENTION_CLASSES`.

## Vendoring

A model that transformers doesn't ship yet can be vendored under
`src/models/<name>/`: its `configuration_*.py` and `modeling_*.py`, registered on
import, with a CPU test that proves the registration. Remove the copy once
upstream ships the model.

No model is vendored. The only model code under `src/models/` is the
sequence-classification heads transformers doesn't ship, for Gemma 4 and
Qwen3.5/3.6 MoE.

The step-by-step procedure for each mode is in
[Adding a Model](../agent-docs/models/adding-a-model.md) ↗.
