# Model Integration Cost

Halo applies distributed behavior as HuggingFace model wrappers, so adding a model takes little code.

For plain FSDP with no parallelism flags there is nothing to do: any `AutoModelForCausalLM` trains as is, multimodal models included (loaded through the full vision+text wrapper). Parallelism support is where code gets added.

The advanced-parallelism path for an MoE model reuses the shared base:

```txt
HuggingFace model load
  -> thin EP/TP/CP wrapper (subclass a base layer)
  -> self-registers in MOE_LAYER_MAP (patching.py) via HF_MODULE_NAMES
  -> existing FSDP / EP / ETP / checkpoint paths
```

## Expert-parallel wrapper size

Each shipped MoE family adds one wrapper file under `src/distributed/expert_parallel/layers/`, which registers itself by naming the HF MoE class in `HF_MODULE_NAMES` and the config `model_type` in `HF_MODEL_TYPES`. The wrappers subclass `EPMoELayerBase`, so a family whose expert-weight layout matches an existing one subclasses it instead of starting over — Laguna reuses the GLM-4 wrapper in about 40 lines. Most families land between 50 and 145 lines; GPT-OSS (interleaved gate/up weights and attention sinks) is the outlier at about 380.

## Adding parallelism to a new model

The full procedure is in the `agent-docs` [Adding a New Model](../agent-docs/models/adding-a-model.md) ↗ guide. In brief:

- **EP** — pick or subclass a wrapper layout under `expert_parallel/layers/`, name the HF MoE class in its `HF_MODULE_NAMES`, and set `_NUM_EXPERTS_ATTR_PATHS` to the family's expert-count attribute. `MOE_LAYER_MAP` is derived from the subclass tree.
- **CP** — write a `UlyssesAttentionBase` subclass under `context_parallel/layers/` and declare its `HF_MODULE_NAMES`. `WRAPPER_CLASS_MAP` and `CP_SUPPORTED_ATTENTION_CLASSES` are derived from it, so there is no accept list to edit.
- **TP** — a model with `base_model_tp_plan` needs nothing (`tp_plan="auto"`); selective EP+TP adds the attention classes to `module_types.py` (MoE registers through `HF_MODULE_NAMES` / `MOE_LAYER_MAP`).

## Vendoring a model

A model not yet in transformers can be vendored under `src/models/<name>/`: copy its `configuration_*.py` and `modeling_*.py`, register the config and model class on import, and add a CPU registration test. Remove the vendored copy once upstream lands the model. Today `src/models/` carries no vendored modeling code — only the task heads transformers does not ship (sequence-classification wrappers for Gemma 4 and Qwen3.5 MoE).
