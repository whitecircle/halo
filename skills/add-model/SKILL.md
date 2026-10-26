---
name: add-model
description: >-
  Add support for a new model family to Halo — the EP (expert-parallel)
  layer, the Liger coverage spec, router-balancing choice, and vendoring (with
  a removal-on-merge policy). USER-INVOKED ONLY. Use when the user asks to "add a
  new model", "support <model> for EP/MoE", "vendor a model not in transformers",
  "wire up Liger for <model>", or extend the model roster (the families listed in
  agent-docs/models/README.md).
  Mirrors agent-docs/models/adding-a-model.md.
disable-model-invocation: true
allowed-tools:
  - Read
  - Edit
  - Write
  - Grep
  - Glob
  - Bash
---

# Add a new model family

Canonical guide: `agent-docs/models/adding-a-model.md` (read it) — it holds the file list, the EP layer
hooks, every class attribute the machinery reads, and the layout→EP-layer table. **`checklist.md`**
carries the templates and the commands to run.

## Checklist (overview)

1. **Dense or MoE?** Dense `AutoModelForCausalLM` / multimodal `AutoModelForImageTextToText`
   needs **no EP work** — plain FSDP works as is. Only MoE families need an EP layer.
2. **MoE → add an EP layer.** Create `halo/distributed/expert_parallel/families/<name>.py`
   subclassing `EPMoELayerBase` (`moe_layer/layer.py`) — or `EPSharedExpertsMoELayerBase` (the routed
   forward template), `EPGroupLimitedMoELayerBase` where the family fits one. The base
   owns `__init__` and calls one hook per construction step (`_detect_hidden_dim`, `_adopt_routing`,
   `_adopt_shared_experts`, `_adopt_expert_compute`, `_adopt_expert_params`); the EP layer supplies
   `forward` and overrides only the hooks that differ. The family's facts go into one frozen
   `MoEFamilySpec` constant in `halo/models/moe/family_specs.py` (listed in `MOE_FAMILY_SPECS`; its
   `model_types` are the checkpoint `model_type`(s) every offline resolver reads), the class points at
   it with `FAMILY = <SPEC>` and declares `HF_MODULE_NAMES` (the HF block class name(s)), and one import
   line in `families/roster.py` registers it (`MOE_LAYER_MAP` is built from the imported classes). The
   spec's `expert_layout` (`halo/models/moe/expert_layout.py`) is the family's one checkpoint layout:
   the EP gather, the RL weight sync and `merge_ep_shards.py` all write it, so no EP layer overrides
   the export.
   Reuse an existing EP layer if the expert-weight layout matches (layout→EP-layer table in the guide).
3. **Liger coverage.** Add one `LigerFamilySpec` to `LIGER_FAMILY_SPECS` in
   `halo/model_loading/liger/family_specs.py` — there is no per-family applier module; `halo/model_loading/liger/applier.py`
   turns each spec into the applier `halo/model_loading/liger/dispatch.py` resolves.
   Name a role only where the kernel reproduces the family's forward exactly. A family upstream
   Liger already covers sets `delegates_to_upstream=True` and names only the roles it adds.
4. **Router balancing.** Leave `moe_balancing` at `auto` and make the family resolvable. What `auto`
   picks and what `bias_update` needs to export:
   [Callbacks → MoE balancing modes](../../agent-docs/training-methods/callbacks.md#moe-balancing-modes);
   the layer's `SUPPORTS_BIAS_BALANCING` and selection helpers and the spec's `balancing_slot`: steps 2–3 of
   [Add EP support](../../agent-docs/models/adding-a-model.md#add-ep-support).
5. **Not in transformers yet → vendor.** Copy `configuration_*.py` + `modeling_*.py` into
   `halo/models/<name>/`, register the `Auto*` classes in a module of that package (never its
   `__init__.py` — package inits carry a docstring only), and trigger that module's import as a
   module-level side effect of `halo/model_loading/model_preparation.py` (the module every loader
   imports before resolving an `Auto*` class). **Record the removal trigger** —
   the directory goes when upstream ships the family. A `trust_remote_code` model needs no
   vendoring: `apply_remote_code_compat_shims()` (`halo/model_loading/remote_code.py`)
   restores the transformers-v5 APIs those files call.
6. **Pipeline parallelism (optional).** PP is an allowlist: the backbone splits only once a
   `validated=True` `PPFamilySpec` in `PP_FAMILY_SPECS` (`halo/distributed/pipeline_parallel/families.py`)
   names it, added after its PP GPU rows pass; until then it is refused by name
   ([Adding a Model](../../agent-docs/models/adding-a-model.md)).
7. **Configs + tests.** Add `examples/sft/<family>/` YAMLs and the required tests: an EP=2-vs-stock
   equivalence row, a save-and-reload trainer test, a CPU support test, and the Liger numerics
   case — step 7 of [Add EP support](../../agent-docs/models/adding-a-model.md#add-ep-support); the
   commands are in `checklist.md`.

All execution (tests, loading) runs **inside the Docker image** — tools are on `PATH` (uv-built, no
prefix); see `CLAUDE.md`.

## Sources of truth

`agent-docs/models/adding-a-model.md` + `checklist.md` are the guide. The code is the **ultimate** authority:
`halo/models/moe/family_specs.py`, `halo/distributed/expert_parallel/` (`moe_layer/`, `parallelize.py`, the existing `families/*.py`),
`halo/model_loading/liger/family_specs.py`, and the HF `modeling_*.py` you are wrapping are what actually
behave — when a doc, this skill, or memory disagrees, or you are unsure, read the real file before
editing. (`CLAUDE.md`: docs-first, the code wins.) Related skill: `checkpoints` (the save/export format
the new family must round-trip through — EP gather, per-expert layout, vLLM keys).
