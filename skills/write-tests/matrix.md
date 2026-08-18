# Coverage checklist

Tick these off when a change touches a trainer, parallelism mode, or model family. A
support-matrix addition is not done until its **rejection counterpart** exists.

**PP is not available in this release** — the schedule engine is not shipped, and
`parallelism_config_from_args` rejects `pipeline_parallel_size > 1` at config time. Never write a
PP **correctness** suite: no PP run is launchable, so it could only pass by not running. The PP
coverage that belongs here is the rejection kind — the release gate, the validator raises, and the
`_supports_pp` gate — all of it CPU-side.

## Trainer × parallelism support matrix

From the trainer table in `CLAUDE.md`. For each cell marked **Yes**, there should be a
correctness test (loss finite + decreasing over ≥2 steps + cross-rank invariants). For each
**No / NOT SUPPORTED**, there should be a rejection test (must raise).

| Trainer | EP | CP | TP | ETP | PP |
|---|:--:|:--:|:--:|:--:|:--:|
| `DistributedSFTTrainer` | Yes | Yes | Yes | Yes | Yes |
| `SmoothMarginPOTrainer` (SMPO) | Yes | Yes | Yes | Yes | Yes |
| `OfflineGRPOTrainer` | Yes | **No** | Yes | Yes | Yes (any `kl_beta`) |
| `DistributedGRPOTrainer` (online) | Yes | **No** | Yes | Yes | **No** |
| `DistributedDPOTrainer` | Yes | **No** | Yes | Yes | Yes (precompute-only) |
| `DistributedKTOTrainer` | Yes | **No** | Yes | Yes | Yes (`apo_zero_unpaired`, precompute-only) |
| `DistributedRewardTrainer` | Yes | **No** | Yes | Yes | Yes |
| `ClassificationTrainer` | Yes | **No** | Yes | Yes | Yes |
| `DistributedDistillationTrainer` | Yes | **No** | Yes | Yes | **No** |
| `DistributedSelfDistillationTrainer` | Yes | **No** | Yes | Yes | **No** |
| `DistributedSDPGTrainer` | Yes | **No** | Yes | Yes | **No** |
| `EmbeddingTrainer` | Yes | **No** | Yes | Yes | **No** |
| `DistributedAsyncEnvironmentalGRPOTrainer` | Yes | **No** | Yes | Yes | **No** |

`_supports_cp` / `_supports_pp` (`src/trainers/mixins/base.py`, both default `False`) are the class
attributes that drive the rejections. CP is incompatible with any trainer using `logits_to_keep`,
global log-prob sums, full-sequence pooling, or dual models; PP additionally rejects PEFT/LoRA, a
live `ref_model`, `activation_offloading` and reentrant GC at trainer construction. The PP column
records each class's declared `_supports_pp`; while PP is unavailable it drives no runnable cell,
so its **Yes** entries carry no correctness-test obligation.

## Parallelism-mode matrix (from `CLAUDE.md`)

The allowlist is `SUPPORTED_AXIS_SETS` (`src/distributed/parallelism_config.py`) — anything not in
it is rejected before any rank math. Supported (need a correctness test): plain DP, EP-only,
TP-only, CP-only, pure ETP (`ep_size=1`), EP+CP (node-local EP), EP+TP (node-local TP),
EP+ETP (experimental, node-local ETP only). The allowlist also admits PP-only, PP+EP and
PP+ETP, but the release gate rejects `pp_size > 1` ahead of it, so those three take a rejection
test and no correctness suite.

Pure-EP topology caveat — `ep_size=2` (2-rank groups) or `ep_size = nvlink_domain_size` (a single
group filling the domain) is supported. Multiple >2-rank dispatch groups in one domain
(`num_nvlink_domains == 1 and ep_size > 2 and nvlink_domain_size > ep_group_size`, e.g. ep4 on an
8-GPU domain) are rejected — mechanism and evidence in the `parallelism` skill (`matrix.md`, row
*Multi-group >2-rank EP on one NVLink domain*). The two enforcement points a test can target:
`ParallelismConfig._validate_single_domain_multigroup_ep` raises at
config time (predicate `is_racy_single_domain_multigroup_ep`), and
`_setup_ep_gradient_checkpointing` (`src/trainers/mixins/ep_introspection.py`) re-checks at
trainer setup for hand-built configs. Write it as a **rejection** test (CPU config test is
enough), not a correctness test.

## REQUIRED rejection tests (combos that MUST raise)

| Combo | Expected |
|---|---|
| EP + TP + ETP (`ep_size>1` AND `tp_size>1` AND `expert_tp_size>1`) | raises at config/init |
| TP + ETP (`tp_size>1` AND `expert_tp_size>1`, `ep_size==1`) | raises — two shardings of the same ranks |
| TP + CP (`tp_size>1` AND `cp_size>1`) | raises — DTensor mesh conflicts with CP groups |
| ETP + CP (`expert_tp_size>1` AND `cp_size>1`) | raises — ETP sub-EP groups break CP seq reconstruction |
| Any `pipeline_parallel_size > 1` | raises at config time — the schedule engine is not shipped in this release (`parallelism_config_from_args`); constructing `PipelineRuntime` raises `NotImplementedError` |
| PP + TP, PP + CP (`pp_size>1` with either) | raises — never exercised multi-node (TP); CP-scaled gradients with no error (CP) |
| PP + EP + TP, PP + EP + ETP (all three `>1`) | raises — outside the allowlist; the expert-TP reduce cannot be deferred past the combine |
| QLoRA + EP | raises |
| LoRA/PEFT + TP (`tp_size>1`), native EP expert LoRA included | raises at trainer construction — `_validate_lora_tp_compatibility` |
| Multi-group >2-rank EP on one NVLink domain (`ep_size>2` with `nvlink_domain_size > ep_group_size`) | fails fast (DeepEP combine race) |
| `_supports_cp=False` trainer with `cp_size>1`, `_supports_pp=False` with `pp_size>1` | rejects at trainer `__init__` |

A rejection test asserts the *reason*, not just that something raised:

```python
import pytest
with pytest.raises(ValueError, match="not supported"):   # CPU: ParallelismConfig(...)
    ...
```

For trainer-init rejections (`_supports_cp=False`), a pure class-attribute check can live in
a CPU test (see `tests/gpu/parallelism/cp/test_cp_rejection.py` for the existing
attribute-inspection pattern, and `tests/cpu/parallelism/test_parallelism_config.py` for
config-level rejections).

## EP-family `ep_vs_fsdp` baselines (per MoE family)

Every supported MoE family needs an `ep_vs_fsdp` correctness test: run the same seeded data
under FSDP (`ep_size=1`) and under EP, and assert step-0 loss matches within
`TOL.parallel_vs_baseline_loss_abs` and the trend within
`TOL.parallel_vs_baseline_train_loss_abs`. Existing exemplars to copy:

- `parallelism/ep/test_ep_vs_reference_qwen3_moe.py`
- `parallelism/ep/test_ep_vs_fsdp_glm4_moe.py`
- `parallelism/ep/test_ep_vs_fsdp_deepseek_v4.py`
- `parallelism/ep/test_ep_vs_no_ep.py` (gptoss)

Every MoE family on the roster (`agent-docs/models/index.md`) needs one. New family → add the
`ep_vs_fsdp` test + register it in the manifest with the family marker.

## Tier / marker checklist for a new GPU test

- [ ] `'gpu'` marker (always).
- [ ] Exactly one tier: `core` (small/fast, ≤2 GPU, tiny model) **or** `full` (large model /
      many-GPU). Add `vllm_server` when the test needs the live vLLM container — those are
      always `full`. `benchmark_*` perf scripts run via shell, not the manifest. The `core` tier
      already sits past its own contract — 4-GPU entries, multi-billion-parameter checkpoints; tier
      composition lives in `agent-docs/contributing/index.md` ("Tests"). Do not add to the drift — when in
      doubt, mark `full`.
- [ ] A `Ngpu` tag matching `nproc` (`1gpu`/`2gpu`/`4gpu`/`8gpu`).
- [ ] Capability tags as applicable: `ep cp tp etp pp hsdp vlm lora moe vllm_server sglang_server`.
- [ ] A model-family tag — the family half of `ALL_MARKERS` in `tests/gpu/manifest.py`.
- [ ] World-size strictness goes in the **script**, not the manifest:
      `gpu_test_main(exact_world_size=N)` when the body only makes sense at exactly `N` ranks.

## Anti-patterns to avoid

- Verdict set `True` off rank 0 or on a `world_size < 2` fallback → masks rank-skew. Compute
  on **all** ranks via all-gather/broadcast.
- `cosine_similarity` returning 1.0 for near-zero tensors → two dead gradients read as
  "identical". Exclude near-zero-norm params instead.
- Tolerances so loose they can't fail (e.g. `0 < loss < 100`, `LOSS_TOLERANCE=0.5` on a 2–10
  loss). Use the named `TOL.*` registry.
- Trivial pass when `len(losses) < 2` (a `MAX_STEPS=1` run skips the decrease check). Require
  `>= 2` real steps.
- Fragile detection by counting substrings in a rendered prompt (e.g. `"user"` occurrences) —
  breaks on a chat-template change. Use the message structure.
- `benchmark_*` returning exit 0 on crash — must `sys.exit(main())` and propagate non-zero.
