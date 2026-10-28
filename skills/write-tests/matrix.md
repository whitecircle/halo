# Coverage checklist

Tick these off when a change touches a trainer, parallelism mode, or model family. A
support-matrix addition is not done until its **rejection counterpart** exists.

## Trainer × parallelism and mode coverage

Every supported cell needs a correctness test (loss finite + decreasing over ≥2 steps + cross-rank
invariants); every refused one a rejection test (must raise). The cells live once:

- trainer × CP/PP: [Trainer Compatibility](../../agent-docs/reference/trainer-architecture.md#trainer-compatibility),
  read off `SUPPORTS_CP` / `PP_REFUSAL` by `reject_undeclared_axes` (no trainer refuses EP, ETP or
  TP). A PP-supporting trainer still needs the construction-refusal tests of
  [Rejected combinations](../../agent-docs/parallelism/pipeline-parallelism.md#rejected-combinations);
  each PP suite lives under `tests/gpu/parallelism/pp/`;
- supported axis sets and topology rules:
  [Supported combinations](../../agent-docs/parallelism/README.md#supported-combinations);
- family × axis: [Compatibility matrix](../../agent-docs/models/README.md#compatibility-matrix).

`tests/cpu/parallelism/test_docs_limitation_tables.py` pins those tables against the code.

## REQUIRED rejection tests (combos that MUST raise)

| Combo | Expected |
|---|---|
| EP + TP + ETP (`ep_size>1` AND `tp_size>1` AND `expert_tp_size>1`) | raises at config/init |
| TP + ETP (`tp_size>1` AND `expert_tp_size>1`, `ep_size==1`) | raises — two shardings of the same ranks |
| TP + CP (`tp_size>1` AND `cp_size>1`) | raises — both partition the same contiguous rank blocks, and Ulysses redistributes heads TP already split |
| ETP + CP (`expert_tp_size>1` AND `cp_size>1`) | raises — expert-TP partners sum their outputs in token space while CP hands each rank a different chunk |
| PP + TP, PP + CP (`pp_size>1` with either) | raises at the allowlist — untested on real multi-node hardware (TP); CP-scaled gradients with no error (CP) |
| PP + EP + TP, PP + EP + CP, PP + EP + ETP (all three `>1`) | raises — outside the allowlist; FSDP mesh conflict (TP), CP-scaled expert gradients (CP), the expert-TP reduce cannot be deferred past the combine (ETP) |
| QLoRA + EP | raises |
| LoRA/PEFT + TP (`tp_size>1`), native EP expert LoRA included | raises at trainer construction — `reject_lora_under_tp` |
| Multi-group >2-rank EP on one NVLink domain (`ep_size>2` with `nvlink_domain_size > ep_group_size`) | raises at config time (`_reject_single_domain_multigroup_ep`) — a CPU config test is enough |
| `SUPPORTS_CP = False` trainer with `cp_size>1`, a `PP_REFUSAL` trainer with `pp_size>1` | rejects in `parallelism_config_from_args`, and at trainer `__init__` for a hand-built config |

A rejection test asserts the *reason*, not just that something raised:

```python
import pytest
with pytest.raises(ValueError, match="is not a supported parallelism combination"):   # CPU: ParallelismConfig(...)
    ...
```

For trainer-init rejections (`SUPPORTS_CP = False`), a pure class-attribute check can live in
a CPU test (see `tests/cpu/parallelism/test_cp_support_declared.py` for the existing
attribute-inspection pattern, and `tests/cpu/parallelism/test_parallelism_config.py` for
config-level rejections).

## EP-family EP-vs-stock rows (per MoE family)

Every MoE family runs EP=2 against the stock model it was copied from, on its tiny model:
`tests/gpu/parallelism/ep/test_ep_family_vs_reference.py --family <name>`, one manifest row per
`TINY_MOE_FAMILIES` entry. It asserts the loss within `TOL.parallel_vs_baseline_loss_abs` and every
MoE-block gradient in direction and scale, the experts through the export's own gather. A new family
gets its row by joining `TINY_MOE_FAMILIES`; checks of its own go in the script's `FAMILY_CHECKS`. The
real checkpoints (gpt-oss, Qwen3-MoE) run EP, EP+TP and ETP against the undistributed forward in
`test_ep_modes_vs_reference.py`, and gpt-oss also against its no-EP load in `test_ep_vs_no_ep.py`.

## Tier / marker checklist for a new GPU test

- [ ] `'gpu'` marker (always).
- [ ] Exactly one tier: `core` (small/fast, ≤2 GPU, tiny model) **or** `full` (large model /
      many-GPU). Add `vllm_server` when the test needs the live vLLM container — those are
      always `full`. `benchmark_*` perf scripts run via shell, not the launcher, but each must be listed in
      `_UNMANIFESTED_BENCHMARKS` (`tests/gpu/manifest.py`) or GPU collection fails. The `core` tier
      already sits past its own contract — 4-GPU entries, multi-billion-parameter checkpoints; tier
      composition lives in `agent-docs/contributing/README.md` ("Tests"). Do not add to the drift — when in
      doubt, mark `full`.
- [ ] A `Ngpu` tag matching `nproc` (`1gpu`/`2gpu`/`4gpu`/`8gpu`).
- [ ] Capability tags as applicable: `ep cp tp etp pp hsdp vlm lora moe vllm_server sglang_server`.
- [ ] A model-family tag — the family half of `ALL_MARKERS` in `tests/gpu/manifest.py`.
- [ ] World-size strictness goes in the **script**, not the manifest:
      `gpu_test_main(exact_world_size=N)` when the body only makes sense at exactly `N` ranks.

## Anti-patterns to avoid

- Verdict set `True` off rank 0 or on a `world_size < 2` fallback → masks rank-skew. Compute
  on **all** ranks via all-gather/broadcast.
- A cosine that scores a zero-norm tensor: 1.0 reads a dead gradient as matching, 0.0 reads a
  vanished negative-control gradient as decorrelated. Use `tests.common.numerics.cos_sim`, which raises
  on a zero-norm or non-finite operand; a pair that is legitimately all-zero (an expert no token
  reached) is handled explicitly at the call site. An EP-vs-reference gradient test gets this check
  and the norm ratio from `score_ep_grad_pairs` ([harness](harness.md#ports-tolerances-reporting)).
- Tolerances so loose they can't fail (e.g. `0 < loss < 100`, `LOSS_TOLERANCE=0.5` on a 2–10
  loss). Use the named `TOL.*` registry.
- Trivial pass when `len(losses) < 2` (a `MAX_STEPS=1` run skips the decrease check). Require
  `>= 2` real steps.
- Fragile detection by counting substrings in a rendered prompt (e.g. `"user"` occurrences) —
  breaks on a chat-template change. Use the message structure.
- `benchmark_*` returning exit 0 on crash — must `sys.exit(main())` and propagate non-zero.
