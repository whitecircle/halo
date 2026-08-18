---
name: add-model
description: >-
  Add support for a new model family to Halo — the EP (expert-parallel)
  wrapper, the Liger coverage spec, router-balancing choice, and vendoring (with
  a removal-on-merge policy). USER-INVOKED ONLY. Use when the user asks to "add a
  new model", "support <model> for EP/MoE", "vendor a model not in transformers",
  "wire up Liger for <model>", or extend the model roster (the families listed in
  agent-docs/models/index.md).
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

Canonical guide: `agent-docs/models/adding-a-model.md` (read it) — it holds the wrapper hooks, every
class attribute the machinery reads, and the layout→wrapper table. **`checklist.md`** carries the
file-touch list, the traps and the commands; read it before editing.

## Checklist (overview)

1. **Dense or MoE?** Dense `AutoModelForCausalLM` / multimodal `AutoModelForImageTextToText`
   needs **no EP work** — plain FSDP just works. Only MoE families need an EP wrapper.
2. **MoE → add an EP wrapper.** Create `src/distributed/expert_parallel/layers/<name>.py`
   subclassing `EPMoELayerBase` (`base_layer.py`) — or `EPSharedExpertsMoELayerBase`,
   `EPSeparateGluMoELayerBase`, `EPGroupLimitedMoELayerBase` where the family fits one. The base
   owns `__init__` and calls one hook per construction step (`_detect_hidden_dim`, `_init_routing`,
   `_init_shared_experts`, `_init_expert_compute`, `_init_expert_params`); the wrapper supplies
   `forward` and overrides only the hooks that differ. Dropping the file into `layers/` is the whole
   registration — `layers/roster.py` imports every module in the package. Set **both** class attrs: `HF_MODULE_NAMES` (the HF block class name(s) — `build_moe_layer_map()` in
   `patching.py` walks the subclass tree, so there is no map to edit) and
   `HF_MODEL_TYPES` (the checkpoint `model_type`(s) — without it sharded EP save is rejected
   and the offline merge resolver finds nothing). `__init_subclass__` enforces a paired-override
   contract: overriding `gather_expert_state_dict` requires also overriding `merge_shards_to_hf`,
   and declaring `_PER_EXPERT_UNFUSED_KEYS` while overriding the gather is a `TypeError` at import.
   Reuse an existing wrapper if the expert-weight layout matches (layout→wrapper table in the guide).
3. **Liger coverage.** Add one `LigerFamilySpec` to `LIGER_FAMILY_SPECS` in
   `src/kernels/liger/families.py` — there is no per-family applier module; the builder
   (`src/kernels/liger/builder.py`) turns each spec into the applier the orchestrator dispatches.
   Name a role only where the kernel reproduces the family's forward exactly. A family upstream
   Liger already covers sets `delegates_to_upstream=True` and names only the roles it adds.
4. **Router balancing.** `moe_balancing` is
   `Literal["auto", "none", "aux_loss", "bias_update", "bias_update_transient"]`, default **`auto`**.
   Leave it at `auto` and make the family resolvable: `aux_loss` when selection happens inside the
   HF gate (nothing to do), or `bias_update` (set `_supports_bias_balancing = True` + inject
   `_balancing_bias` / `_record_expert_load` in the wrapper's routing). A model with native
   balancing biases resolves to `bias_update` through `auto`. `bias_update` requires the bias to
   land in a checkpoint-**exported** slot (`_NATIVE_BALANCING_BIAS_ATTR`); a family whose
   architecture has no such slot must opt in explicitly with `bias_update_transient` — a
   trainer-only side buffer every export serves without. What `auto` resolves to per family is
   published in `agent-docs/training-methods/callbacks.md`.
5. **Not in transformers yet → vendor.** Copy `configuration_*.py` + `modeling_*.py` into
   `src/models/<name>/`, register the `Auto*` classes in a module of that package (never its
   `__init__.py` — package inits carry a docstring only), and trigger that module's import as a
   module-level side effect of `src/models/loading/model_preparation.py` (the module every loader
   imports before resolving an `Auto*` class). **Record the removal trigger** —
   the directory goes when upstream ships the family. A `trust_remote_code` model needs no
   vendoring: `apply_remote_code_compat_shims()` (`src/models/patches/remote_code_compat.py`)
   restores the transformers-v5 APIs those files call.
6. **Configs + tests.** Add `examples/sft/<family>/` YAMLs and the required tests: an EP-vs-FSDP
   equivalence test, a save-and-reload trainer test, a CPU support test, and the Liger numerics
   case. Exact files and commands in `checklist.md`.

All execution (tests, loading) runs **inside the Docker image** — tools are on `PATH` (uv-built, no
prefix); see `CLAUDE.md`.

## Sources of truth
`agent-docs/models/adding-a-model.md` + `checklist.md` are the guide. The code is the **ultimate** authority:
`src/distributed/expert_parallel/` (`base_layer.py`, `patching.py`, the existing `layers/*.py`),
`src/kernels/liger/families.py`, and the HF `modeling_*.py` you are wrapping are what actually
behave — when a doc, this skill, or memory disagrees, or you are unsure, read the real file before
editing. (`CLAUDE.md`: docs-first, the code wins.) Related skill: `checkpoints` (the save/export format
the new family must round-trip through — EP gather, per-expert layout, vLLM keys).
