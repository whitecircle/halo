# Add-a-model — concrete checklist

`agent-docs/models/adding-a-model.md` is the contract: the wrapper hooks, every class attribute the
machinery reads, the layout→wrapper table, and the CP / TP / vendoring / Liger tracks. Read it
first. This page carries what that guide does not — the file-touch list, the traps, and what to
run. Paths are relative to the repo root; anything that executes runs **inside the Docker image**
(`halo:blackwell` / `:hopper`), where tools are on `PATH` (uv-built, no prefix).

---

## Files you touch

| Concern | File(s) |
|---|---|
| EP wrapper | `src/distributed/expert_parallel/layers/<name>.py` — `layers/roster.py` imports the whole package, so the file's existence *is* the registration |
| EP base / hooks | `src/distributed/expert_parallel/base_layer.py` (gather in `expert_gather.py`, balancing in `balancing.py`) |
| Expert-LoRA coverage | `src/distributed/expert_parallel/config.py` (`LORA_PROJECTION_COVERAGE`) |
| Non-persistent buffers | `src/models/patches/buffer_fixes.py` — extend the `fix_non_persistent_buffers` / `fix_rotary_inv_freq` fixer chains; `finalize_loaded_model` is the seam every load path calls, and a buffer no fixer covers stays on meta and is rejected at device placement |
| Liger coverage | `src/kernels/liger/families.py` (`LIGER_FAMILY_SPECS`) |
| CP wrapper | `src/distributed/context_parallel/layers/<name>.py` |
| Selective TP | `src/distributed/tensor_parallel/module_types.py` (`TP_SHARDABLE_ATTENTION_CLASSES`) |
| Vendoring | `src/models/<name>/` + a side-effect import in `src/models/loading/model_preparation.py` |
| Configs | `examples/sft/<family>/` |
| Tests | `tests/gpu/parallelism/ep/`, `tests/gpu/trainers/sft/`, `tests/cpu/models/`, `tests/gpu/manifest.py` |
| Docs | `agent-docs/models/<name>.md`, `agent-docs/models/README.md`, `CLAUDE.md` |

There is no registry file to edit. `MOE_LAYER_MAP` (`patching.py`) is built by walking the
`EPMoELayerBase` subclass tree, and the CP map the same way — a duplicate HF class name raises at
import. Templates worth cribbing: `layers/qwen3.py` (separate-GLU), `layers/gpt_oss.py` (in-layer
routing, interleaved weights), `layers/deepseek_v4.py` (most class attributes in one place).

---

## Router balancing (`moe_balancing`)

The field is `Literal["auto", "none", "aux_loss", "bias_update", "bias_update_transient"]`, default
`auto` (`src/args/common_script_args.py`); `resolve_balancing_mode`
(`src/models/moe_balancing.py`) resolves it against the live model and
`apply_balancing_strategy` (`src/distributed/expert_parallel/balancing_strategy.py`) applies it.
Leave the YAML at `auto` and make the family resolvable:

- **Selection inside the HF gate** and the loaded class's forward declares
  `output_router_logits` → `aux_loss`. Declare nothing.
- **The wrapper selects** → `_supports_bias_balancing = True`, add `self._balancing_bias(scores)`
  to the **selection** scores before top-k (gate weights come from the *unbiased* scores), then
  `self._record_expert_load(indices)`. `_deepseek_biased_route` does the whole pattern for
  logit-routed families; a layer can refuse per-instance by overriding `enable_bias_balancing`.
- **`bias_update` ships only if the bias exports** — declare `_NATIVE_BALANCING_BIAS_ATTR` (plus
  `_NATIVE_BALANCING_CONFIG_FLAG` and the `_materialize_native_balancing_slot` hook for a
  config-gated slot). Without one, `_enforce_bias_export_contract` refuses `bias_update` and the
  user must opt into `bias_update_transient`, which no export or weight-sync carries.

Which mode `auto` picks per family is published in `agent-docs/training-methods/callbacks.md` — read the
verdict there rather than keeping a second copy.

**Traps.**

- Both bias modes downgrade to `none` on the weight-sync RL scripts (online / environmental GRPO):
  the sync ships parameters and the bias is a buffer (`src/callbacks/wiring.py`), so those
  runs train unbalanced. `aux_loss` is inert under a policy-gradient loss.
- A construction hook may issue **no collective** and allocate nothing sized by `world_size` —
  some load paths patch layers before every rank reaches group setup, and 8-GPU-invisible state is
  an OOM at 512. Both are pinned per family by
  `tests/cpu/parallelism/test_ep_prologue_parity.py`.
- A stored expert-weight root with no `LORA_PROJECTION_COVERAGE` entry leaves `expert_lora`
  silently not adapting it (`tests/cpu/peft/test_expert_lora_projection_coverage.py`).

---

## Vendoring — removal is part of the change

Vendoring is for a family transformers has not shipped, and it is temporary: the directory, its
registration import, the registration test, and any compat shim go the moment upstream lands, with
the EP wrapper and Liger spec re-keyed to the upstream module. Put the removal trigger in the
header docstring of `src/models/<name>/__init__.py`, naming the upstream PR. Procedure:
`agent-docs/models/adding-a-model.md` → *Vendoring a model*.

One detail that bites and is not in the guide: seed
`HARDCODED_CONFIG_FOR_MODELS["<name>"]` (`transformers.utils.auto_docstring`) **before** importing
the modeling module, or `@auto_docstring` raises "Config not found". A vendored family whose hub
checkpoint stores experts per-expert while the module fuses them gets no entry in transformers'
own conversion mapping, so the layout must be declared on the EP wrapper
(`_PER_EXPERT_UNFUSED_KEYS` / `_HUB_PER_EXPERT_KEYS`) instead — the gather and the lazy loader read
that, not `from_pretrained`'s converter.

---

## Verify

Lint on the host; everything else inside the image. Never pin `--master_port` — the harness
allocates one per run.

```bash
make lint
python tests/cpu/models/test_<name>_support.py            # registration, gather/merge, balancing
python tests/cpu/kernels/test_liger_family_coverage.py    # every spec class exists and is exact
python tests/cpu/checkpoint/test_merge_ep_shards.py       # only if you overrode the gather
torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_vs_fsdp_<name>.py
```

Required before the family counts as supported: an EP-vs-FSDP equivalence test (template
`tests/gpu/parallelism/ep/test_ep_vs_fsdp_deepseek_v4.py`), a trainer test that saves through the
gathered EP path and reloads the checkpoint as a plain HF model
(`tests/gpu/trainers/sft/test_sft_deepseek_v4_moe.py`), both registered in
`tests/gpu/manifest.py`, a CPU support test (`tests/cpu/models/test_deepseek_v4_support.py`), and —
where you wired `bias_update` — `tests/gpu/parallelism/ep/test_gptoss_bias_balancing.py`. Add the
Liger numerics case to `tests/gpu/kernels/test_liger_family_kernels.py`.
